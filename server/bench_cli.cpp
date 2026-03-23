// Mesh Parameterization CLI Tool
// Runs a single parameterization method on an OBJ or GLB file.
// Process isolation: each invocation is a separate process.
//
// Usage: meshparam_bench <method> <input.obj|.glb> [--output-glb <path>] [--json <path>]
//        meshparam_bench convert <input.obj> [--output-glb <path>]
// Methods: heat, lscm, igl_arap, slim, stein_admm, cm,
//          cgal_conformal, cgal_arap, cgal_authalic, convert

#include "meshparam/gltf_io.h"
#include "meshparam/parameterizer.h"
#include "meshparam/distortion.h"
#include <tiny_gltf.h>

#include <igl/slim.h>
#include <igl/lscm.h>
#include <igl/arap.h>
#include <igl/boundary_loop.h>
#include <igl/harmonic.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/flipped_triangles.h>
#include <igl/readOBJ.h>
#include "meshparam/benchmark_metrics.h"

// Stein ADMM splitting
#include "parametrization/constrained_qp.cpp"
#include "parametrization/uv_to_jacobian.cpp"
#include "parametrization/polar_decomposition.cpp"
#include "parametrization/argmin_P.cpp"
#include "parametrization/argmin_U.cpp"
#include "parametrization/argmin_W.cpp"
#include "parametrization/step_Lambda.cpp"
#include "parametrization/lagrangian.cpp"
#include "parametrization/lagrangian_error.cpp"
#include "parametrization/rescale_penalties.cpp"
#include "parametrization/rescale_b_mumin.cpp"
#include "parametrization/rescale_h.cpp"
#include "parametrization/termination_conditions.cpp"
#include "parametrization/energy.cpp"
#include "parametrization/quartic_polynomial.cpp"
#include "parametrization/spd_quartic_polynomial.cpp"
#include "parametrization/sqrtm.cpp"
#include "parametrization/rotmat_sym_product.cpp"
#include "parametrization/tutte.cpp"
#include "parametrization/map_to.cpp"

#include "cgalparam/gltf_io.h"
#include "cgalparam/cgal_parameterize.h"
#include "cgalparam/seam_cut.h"
#include "cgalparam/distortion.h"
#include "cgalparam/types.h"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <cmath>
#include <unordered_map>
#include <map>
#include <array>
#include <algorithm>

#include "methods.inc"

// ============================================================
// File I/O helpers
// ============================================================
static bool ends_with(const std::string& s, const std::string& suffix) {
    if (suffix.size() > s.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin(),
        [](char a, char b) { return tolower(a) == tolower(b); });
}

static std::vector<uint8_t> load_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    size_t sz = f.tellg(); f.seekg(0);
    std::vector<uint8_t> data(sz);
    f.read(reinterpret_cast<char*>(data.data()), sz);
    return data;
}

static void save_file(const std::string& path, const std::vector<uint8_t>& data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write: " + path);
    f.write(reinterpret_cast<const char*>(data.data()), data.size());
}

static std::vector<uint8_t> obj_to_glb(const std::string& obj_path) {
    Eigen::MatrixXd V, TC, N;
    Eigen::MatrixXi F, FTC, FN;
    if (!igl::readOBJ(obj_path, V, TC, N, F, FTC, FN))
        throw std::runtime_error("Failed to read OBJ: " + obj_path);
    if (V.rows() == 0 || F.rows() == 0)
        throw std::runtime_error("Empty mesh in " + obj_path);

    meshparam::TriMesh mesh;

    // If OBJ has UVs with different indexing (FTC != F), build unique (V,UV) pairs
    if (TC.rows() > 0 && FTC.rows() == F.rows() && FTC.maxCoeff() < TC.rows()) {
        bool same_indexing = (FTC.array() == F.array()).all();
        if (same_indexing) {
            mesh.V = V;
            mesh.F = F;
            mesh.UV = TC;
        } else {
            // Build unique (vertex_idx, uv_idx) pairs to avoid face-soup expansion
            int nf = F.rows();
            std::map<std::pair<int,int>, int> pair_to_new;
            std::vector<Eigen::Vector3d> newV_vec;
            std::vector<Eigen::Vector2d> newUV_vec;
            Eigen::MatrixXi newF(nf, 3);

            for (int fi = 0; fi < nf; fi++) {
                for (int j = 0; j < 3; j++) {
                    auto key = std::make_pair(F(fi, j), FTC(fi, j));
                    auto it = pair_to_new.find(key);
                    if (it != pair_to_new.end()) {
                        newF(fi, j) = it->second;
                    } else {
                        int idx = static_cast<int>(newV_vec.size());
                        newV_vec.push_back(V.row(F(fi, j)));
                        newUV_vec.push_back(TC.row(FTC(fi, j)));
                        pair_to_new[key] = idx;
                        newF(fi, j) = idx;
                    }
                }
            }

            int nn = static_cast<int>(newV_vec.size());
            mesh.V.resize(nn, 3);
            mesh.UV.resize(nn, 2);
            for (int i = 0; i < nn; i++) {
                mesh.V.row(i) = newV_vec[i];
                mesh.UV.row(i) = newUV_vec[i];
            }
            mesh.F = newF;
        }
    } else {
        mesh.V = V;
        mesh.F = F;
    }

    return meshparam::save_gltf_to_memory(mesh);
}

static std::vector<uint8_t> load_input(const std::string& path) {
    if (ends_with(path, ".glb") || ends_with(path, ".gltf")) {
        return load_file(path);
    } else if (ends_with(path, ".obj")) {
        return obj_to_glb(path);
    } else {
        // Try GLB first (check magic bytes), fall back to OBJ
        auto data = load_file(path);
        if (data.size() >= 4 && data[0] == 'g' && data[1] == 'l' && data[2] == 'T' && data[3] == 'F') {
            return data;
        }
        return obj_to_glb(path);
    }
}

// ============================================================
// Main
// ============================================================
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: meshparam_bench <method> <input.obj|.glb> [options]" << std::endl;
        std::cerr << "  --output-glb <path>   Write result GLB to file" << std::endl;
        std::cerr << "  --json <path>         Write metrics JSON to file (default: stdout)" << std::endl;
        std::cerr << "Methods: heat, lscm, igl_arap, slim, stein_admm, cm," << std::endl;
        std::cerr << "         cgal_conformal, cgal_arap, cgal_authalic, convert" << std::endl;
        return 1;
    }

    std::string method = argv[1];
    std::string input_path = argv[2];
    std::string output_glb_path;
    std::string json_path;

    // Parse optional args
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output-glb" && i + 1 < argc) output_glb_path = argv[++i];
        else if (arg == "--json" && i + 1 < argc) json_path = argv[++i];
        else if (json_path.empty() && arg[0] != '-') json_path = arg;
    }

    MethodResult result;
    result.method = method;

    try {
        // Load input (OBJ or GLB)
        auto glb = load_input(input_path);

        // "convert" mode: just output the GLB, no parameterization
        // Don't weld — preserve UVs from OBJ (artist UVs have split vertices for UV seams)
        if (method == "convert") {
            if (!output_glb_path.empty()) {
                save_file(output_glb_path, glb);
            }
            result.success = true;
            auto mesh = meshparam::load_gltf_from_memory(glb);
            result.vertices = mesh.num_vertices();
            result.faces = mesh.num_faces();
            std::string json = result.to_json();
            if (json_path.empty()) std::cout << json << std::endl;
            else { std::ofstream f(json_path); f << json << std::endl; }
            return 0;
        }

        // Weld + heal + extract largest component (multi-body STEP files)
        auto welded = weld_vertices(glb);
        heal_mesh(welded, false);
        welded = extract_largest_component(welded);

        // Auto-detect _SEAM: if present, split mesh into top/bottom halves,
        // parameterize top with the requested method, bottom with LSCM, combine.
        // If no _SEAM, fall back to BFS geodesic seam (default).
        auto orig_mesh = meshparam::load_gltf_from_memory(glb);
        bool has_seam = orig_mesh.has_seam();
        bool did_split = false;

        if (has_seam) {
            auto weld_mesh = meshparam::load_gltf_from_memory(welded);
            int wnv = weld_mesh.num_vertices(), wnf = weld_mesh.num_faces();

            // Map seam from original to welded mesh
            struct V3Hash {
                size_t operator()(const std::array<int64_t,3>& v) const {
                    size_t h = 0;
                    for (auto x : v) h ^= std::hash<int64_t>()(x) + 0x9e3779b9 + (h<<6) + (h>>2);
                    return h;
                }
            };
            std::unordered_map<std::array<int64_t,3>, int, V3Hash> weld_pos;
            for (int i = 0; i < wnv; i++) {
                std::array<int64_t,3> k = {
                    (int64_t)std::round(weld_mesh.V(i,0)*1e4),
                    (int64_t)std::round(weld_mesh.V(i,1)*1e4),
                    (int64_t)std::round(weld_mesh.V(i,2)*1e4)};
                weld_pos[k] = i;
            }
            std::set<int> w_seam;
            double seam_z = 0;
            int seam_count = 0;
            for (int i = 0; i < orig_mesh.num_vertices(); i++) {
                if (orig_mesh.seam(i) > 0.5) {
                    std::array<int64_t,3> k = {
                        (int64_t)std::round(orig_mesh.V(i,0)*1e4),
                        (int64_t)std::round(orig_mesh.V(i,1)*1e4),
                        (int64_t)std::round(orig_mesh.V(i,2)*1e4)};
                    auto it = weld_pos.find(k);
                    if (it != weld_pos.end()) {
                        w_seam.insert(it->second);
                        seam_z += weld_mesh.V(it->second, 2);
                        seam_count++;
                    }
                }
            }
            if (seam_count > 0) seam_z /= seam_count;

            if (w_seam.size() >= 3) {
                // Split faces into top/bottom by centroid Z, seam faces go to both
                auto split_half = [&](bool is_top) -> std::vector<uint8_t> {
                    std::vector<int> face_list;
                    for (int fi = 0; fi < wnf; fi++) {
                        int n_on_seam = 0;
                        for (int j = 0; j < 3; j++)
                            if (w_seam.count(weld_mesh.F(fi, j))) n_on_seam++;
                        if (n_on_seam >= 2) {
                            face_list.push_back(fi); // seam face: include in both
                        } else {
                            double cz = (weld_mesh.V(weld_mesh.F(fi,0), 2) +
                                        weld_mesh.V(weld_mesh.F(fi,1), 2) +
                                        weld_mesh.V(weld_mesh.F(fi,2), 2)) / 3.0;
                            if (is_top ? (cz >= seam_z) : (cz < seam_z))
                                face_list.push_back(fi);
                        }
                    }
                    // Build submesh
                    std::vector<bool> used(wnv, false);
                    for (int fi : face_list)
                        for (int j = 0; j < 3; j++) used[weld_mesh.F(fi, j)] = true;
                    std::vector<int> old2new(wnv, -1);
                    int nn = 0;
                    for (int i = 0; i < wnv; i++) if (used[i]) old2new[i] = nn++;

                    meshparam::TriMesh sub;
                    sub.V.resize(nn, 3);
                    for (int i = 0; i < wnv; i++)
                        if (old2new[i] >= 0) sub.V.row(old2new[i]) = weld_mesh.V.row(i);
                    sub.F.resize(face_list.size(), 3);
                    for (size_t fi = 0; fi < face_list.size(); fi++)
                        for (int j = 0; j < 3; j++)
                            sub.F(fi, j) = old2new[weld_mesh.F(face_list[fi], j)];
                    return meshparam::save_gltf_to_memory(sub);
                };

                auto top_glb = split_half(true);
                auto bot_glb = split_half(false);
                auto top_mesh = meshparam::load_gltf_from_memory(top_glb);
                auto bot_mesh = meshparam::load_gltf_from_memory(bot_glb);

                std::cerr << "[split] Top: " << top_mesh.num_vertices() << "v "
                          << top_mesh.num_faces() << "f, Bottom: "
                          << bot_mesh.num_vertices() << "v "
                          << bot_mesh.num_faces() << "f" << std::endl;

                // Parameterize top with requested method
                MethodResult top_result;
                if (method == "heat") top_result = run_heat(top_glb, false);
                else if (method == "lscm") top_result = run_lscm(top_glb);
                else if (method == "igl_arap") top_result = run_igl_arap(top_glb);
                else if (method == "slim") top_result = run_slim(top_glb);
                else if (method == "stein_admm") top_result = run_stein(top_glb);
                else if (method == "cgal_conformal") top_result = run_cgal(top_glb, cgalparam::ParamMethod::DiscreteConformal, "cgal_conformal", false, {});
                else if (method == "cgal_arap") top_result = run_cgal(top_glb, cgalparam::ParamMethod::ARAP, "cgal_arap", false, {});
                else if (method == "cgal_authalic") top_result = run_cgal(top_glb, cgalparam::ParamMethod::DiscreteAuthalic, "cgal_authalic", false, {});
#ifdef HAS_COMPMAJOR
                else if (method == "cm") top_result = run_cm(top_glb);
#endif
                else top_result.error = "Unknown method: " + method;

                // Parameterize bottom with LSCM (always works, fast)
                auto bot_result = run_lscm(bot_glb);

                if (top_result.success) {
                    // Combine: merge both parameterized halves into one GLB
                    auto top_param = meshparam::load_gltf_from_memory(top_result.glb);
                    auto bot_param = bot_result.success ?
                        meshparam::load_gltf_from_memory(bot_result.glb) : bot_mesh;

                    int tnv = top_param.num_vertices(), bnv = bot_param.num_vertices();
                    int tnf = top_param.num_faces(), bnf = bot_param.num_faces();

                    meshparam::TriMesh combined;
                    combined.V.resize(tnv + bnv, 3);
                    combined.V.topRows(tnv) = top_param.V;
                    combined.V.bottomRows(bnv) = bot_param.V;
                    combined.F.resize(tnf + bnf, 3);
                    combined.F.topRows(tnf) = top_param.F;
                    combined.F.bottomRows(bnf) = bot_param.F.array() + tnv;
                    if (top_param.has_uvs()) {
                        combined.UV.resize(tnv + bnv, 2);
                        combined.UV.topRows(tnv) = top_param.UV;
                        if (bot_param.has_uvs())
                            combined.UV.bottomRows(bnv) = bot_param.UV;
                        else
                            combined.UV.bottomRows(bnv).setZero();
                    }
                    if (top_param.has_normals() || bot_param.has_normals()) {
                        combined.N.resize(tnv + bnv, 3);
                        if (top_param.has_normals()) combined.N.topRows(tnv) = top_param.N;
                        if (bot_param.has_normals()) combined.N.bottomRows(bnv) = bot_param.N;
                    }

                    result = top_result; // metrics from top half (the important one)
                    result.glb = meshparam::save_gltf_to_memory(combined);
                    result.vertices = tnv + bnv;
                    result.faces = tnf + bnf;
                    did_split = true;
                    std::cerr << "[split] Combined: " << result.vertices << "v "
                              << result.faces << "f (top: " << top_result.method << ")" << std::endl;
                }
            }
        }

        if (!did_split) {
        // No seam or split failed — standard single-mesh approach with BFS fallback
        if (method == "heat") {
            result = run_heat(welded, false);
        } else if (method == "lscm") {
            result = run_lscm(welded);
        } else if (method == "igl_arap") {
            result = run_igl_arap(welded);
        } else if (method == "slim") {
            result = run_slim(welded);
        } else if (method == "stein_admm") {
            result = run_stein(welded);
        } else if (method == "cgal_conformal") {
            result = run_cgal(welded, cgalparam::ParamMethod::DiscreteConformal, "cgal_conformal", false, {});
        } else if (method == "cgal_arap") {
            result = run_cgal(welded, cgalparam::ParamMethod::ARAP, "cgal_arap", false, {});
        } else if (method == "cgal_authalic") {
            result = run_cgal(welded, cgalparam::ParamMethod::DiscreteAuthalic, "cgal_authalic", false, {});
#ifdef HAS_COMPMAJOR
        } else if (method == "cm") {
            result = run_cm(welded);
#endif
        } else {
            result.error = "Unknown method: " + method;
        }
        } // end if (!did_split)

        // Write result GLB if requested
        if (result.success && !output_glb_path.empty() && !result.glb.empty()) {
            save_file(output_glb_path, result.glb);
        }

    } catch (const std::exception& e) {
        result.error = e.what();
    }

    // Output JSON
    std::string json = result.to_json();
    if (json_path.empty() || json_path == "-") {
        std::cout << json << std::endl;
    } else {
        std::ofstream f(json_path);
        f << json << std::endl;
    }

    return result.success ? 0 : 1;
}
