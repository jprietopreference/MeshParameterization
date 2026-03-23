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
    bool auto_seam = false;

    // Parse optional args
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output-glb" && i + 1 < argc) output_glb_path = argv[++i];
        else if (arg == "--json" && i + 1 < argc) json_path = argv[++i];
        else if (arg == "--auto-seam") auto_seam = true;
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

        // Auto-seam: use _SEAM attribute from GLB (set by Gmsh pipeline from B-Rep edge loop)
        // to cut the welded mesh along the seam path
        std::vector<uint8_t> welded_cut = welded;
        if (auto_seam) {
            // Read _SEAM from original GLB, map seam vertices to welded mesh
            auto orig_mesh = meshparam::load_gltf_from_memory(glb);
            if (orig_mesh.has_seam()) {
                auto weld_mesh = meshparam::load_gltf_from_memory(welded);
                struct V3Hash {
                    size_t operator()(const std::array<int64_t,3>& v) const {
                        size_t h = 0;
                        for (auto x : v) h ^= std::hash<int64_t>()(x) + 0x9e3779b9 + (h<<6) + (h>>2);
                        return h;
                    }
                };
                std::unordered_map<std::array<int64_t,3>, int, V3Hash> weld_pos;
                for (int i = 0; i < weld_mesh.num_vertices(); i++) {
                    std::array<int64_t,3> k = {
                        (int64_t)std::round(weld_mesh.V(i,0)*1e4),
                        (int64_t)std::round(weld_mesh.V(i,1)*1e4),
                        (int64_t)std::round(weld_mesh.V(i,2)*1e4)};
                    weld_pos[k] = i;
                }
                // Find welded vertices that are seam vertices
                std::vector<int> seam_verts;
                for (int i = 0; i < orig_mesh.num_vertices(); i++) {
                    if (orig_mesh.seam(i) > 0.5) {
                        std::array<int64_t,3> k = {
                            (int64_t)std::round(orig_mesh.V(i,0)*1e4),
                            (int64_t)std::round(orig_mesh.V(i,1)*1e4),
                            (int64_t)std::round(orig_mesh.V(i,2)*1e4)};
                        auto it = weld_pos.find(k);
                        if (it != weld_pos.end()) seam_verts.push_back(it->second);
                    }
                }
                // Deduplicate
                std::sort(seam_verts.begin(), seam_verts.end());
                seam_verts.erase(std::unique(seam_verts.begin(), seam_verts.end()), seam_verts.end());

                if (seam_verts.size() >= 3) {
                    std::cerr << "[auto-seam] " << seam_verts.size()
                              << " seam vertices from _SEAM attribute" << std::endl;
                    welded_cut = cut_mesh_along_loop(welded, seam_verts);
                }
            }
        }

        // If auto-seam produced a seam, also try split mode:
        // split mesh into two halves along the seam, parameterize each independently
        if (auto_seam && welded_cut.data() != welded.data()) {
            // The welded_cut mesh has boundary from the seam cut
            // Check if boundary_loop works now
            auto cut_mesh = meshparam::load_gltf_from_memory(welded_cut);
            Eigen::VectorXi bnd;
            igl::boundary_loop(cut_mesh.F, bnd);
            if (bnd.size() > 0) {
                std::cerr << "[auto-seam] Cut mesh has boundary: " << bnd.size() << " vertices" << std::endl;
            }
        }

        // Dispatch method — use welded_cut (pre-cut if auto-seam) for methods needing boundary
        if (method == "heat") {
            result = run_heat(welded, false);
        } else if (method == "lscm") {
            result = run_lscm(welded_cut);
        } else if (method == "igl_arap") {
            result = run_igl_arap(welded_cut);
        } else if (method == "slim") {
            result = run_slim(welded_cut);
        } else if (method == "stein_admm") {
            result = run_stein(welded_cut);
        } else if (method == "cgal_conformal") {
            result = run_cgal(welded_cut, cgalparam::ParamMethod::DiscreteConformal, "cgal_conformal", false, {});
        } else if (method == "cgal_arap") {
            result = run_cgal(welded_cut, cgalparam::ParamMethod::ARAP, "cgal_arap", false, {});
        } else if (method == "cgal_authalic") {
            result = run_cgal(welded_cut, cgalparam::ParamMethod::DiscreteAuthalic, "cgal_authalic", false, {});
#ifdef HAS_COMPMAJOR
        } else if (method == "cm") {
            result = run_cm(welded_cut);
#endif
        } else {
            result.error = "Unknown method: " + method;
        }

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
