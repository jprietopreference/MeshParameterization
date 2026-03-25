// Mesh Parameterization CLI Tool
// Runs a single parameterization method on an OBJ or GLB file.
// Process isolation: each invocation is a separate process.
//
// Usage: meshparam_bench <method> <input.obj|.glb> [--output-glb <path>] [--json <path>]
//        meshparam_bench convert <input.obj> [--output-glb <path>]
// Methods: heat, lscm, igl_arap, slim, cm,
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

// Stein ADMM removed — replaced by CM

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
// Count glTF primitives in a GLB
static int count_primitives(const std::vector<uint8_t>& glb_data) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;
    loader.LoadBinaryFromMemory(&model, &err, &warn, glb_data.data(), (unsigned)glb_data.size());
    if (model.meshes.empty()) return 0;
    return (int)model.meshes[0].primitives.size();
}

// Extract a single primitive from a multi-primitive GLB as a standalone GLB
static std::vector<uint8_t> extract_primitive(const std::vector<uint8_t>& glb_data, int prim_idx) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;
    loader.LoadBinaryFromMemory(&model, &err, &warn, glb_data.data(), (unsigned)glb_data.size());
    if (model.meshes.empty() || prim_idx >= (int)model.meshes[0].primitives.size())
        return glb_data;

    auto& prim = model.meshes[0].primitives[prim_idx];
    auto& buf = model.buffers[0].data;

    // Read positions
    auto read_vec3 = [&](int acc_idx) -> Eigen::MatrixXd {
        auto& acc = model.accessors[acc_idx];
        auto& bv = model.bufferViews[acc.bufferView];
        const float* data = (const float*)(buf.data() + bv.byteOffset + acc.byteOffset);
        Eigen::MatrixXd M(acc.count, 3);
        for (int i = 0; i < (int)acc.count; i++)
            for (int j = 0; j < 3; j++) M(i, j) = data[i * 3 + j];
        return M;
    };
    auto read_scalar = [&](int acc_idx) -> Eigen::VectorXd {
        auto& acc = model.accessors[acc_idx];
        auto& bv = model.bufferViews[acc.bufferView];
        const float* data = (const float*)(buf.data() + bv.byteOffset + acc.byteOffset);
        Eigen::VectorXd V(acc.count);
        for (int i = 0; i < (int)acc.count; i++) V(i) = data[i];
        return V;
    };

    meshparam::TriMesh mesh;

    // Read indices and find actual vertex range used by this primitive
    {
        auto& idx_acc = model.accessors[prim.indices];
        auto& idx_bv = model.bufferViews[idx_acc.bufferView];
        const uint32_t* idx_data = (const uint32_t*)(buf.data() + idx_bv.byteOffset + idx_acc.byteOffset);
        int nf = idx_acc.count / 3;

        // Find min/max vertex index used
        uint32_t vmin = UINT32_MAX, vmax = 0;
        for (int i = 0; i < (int)idx_acc.count; i++) {
            vmin = std::min(vmin, idx_data[i]);
            vmax = std::max(vmax, idx_data[i]);
        }
        int nv = vmax - vmin + 1;

        // Read full attribute arrays, then slice [vmin, vmax]
        auto pos_it = prim.attributes.find("POSITION");
        if (pos_it != prim.attributes.end()) {
            auto full = read_vec3(pos_it->second);
            mesh.V = full.middleRows(vmin, nv);
        }
        auto nrm_it = prim.attributes.find("NORMAL");
        if (nrm_it != prim.attributes.end()) {
            auto full = read_vec3(nrm_it->second);
            mesh.N = full.middleRows(vmin, nv);
        }
        auto fid_it = prim.attributes.find("_FACE_ID");
        if (fid_it != prim.attributes.end()) {
            auto full = read_scalar(fid_it->second);
            mesh.face_ids = full.segment(vmin, nv);
        }
        auto seam_it = prim.attributes.find("_SEAM");
        if (seam_it != prim.attributes.end()) {
            auto full = read_scalar(seam_it->second);
            mesh.seam = full.segment(vmin, nv);
        }

        // Remap indices
        mesh.F.resize(nf, 3);
        for (int i = 0; i < nf; i++)
            for (int j = 0; j < 3; j++)
                mesh.F(i, j) = idx_data[i * 3 + j] - vmin;
    }

    return meshparam::save_gltf_to_memory(mesh);
}

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

        // Check if GLB has multiple primitives (front/back split from Gmsh pipeline)
        int num_prims = count_primitives(glb);
        bool did_split = false;

        if (num_prims >= 2) {
            // Two primitives: front (prim 0) and back (prim 1)
            // Parameterize front with requested method, back with LSCM, combine
            auto front_glb = extract_primitive(glb, 0);
            auto back_glb = extract_primitive(glb, 1);

            // Weld + heal each half
            auto front_welded = weld_vertices(front_glb);
            heal_mesh(front_welded, false);
            front_welded = extract_largest_component(front_welded);

            auto back_welded = weld_vertices(back_glb);
            heal_mesh(back_welded, false);
            back_welded = extract_largest_component(back_welded);

            auto fm = meshparam::load_gltf_from_memory(front_welded);
            auto bm = meshparam::load_gltf_from_memory(back_welded);
            std::cerr << "[split] Front: " << fm.num_vertices() << "v " << fm.num_faces()
                      << "f, Back: " << bm.num_vertices() << "v " << bm.num_faces() << "f" << std::endl;

            // Parameterize front with requested method
            MethodResult front_result;
            if (method == "heat") front_result = run_heat(front_welded, false);
            else if (method == "lscm") front_result = run_lscm(front_welded);
            else if (method == "igl_arap") front_result = run_igl_arap(front_welded);
            else if (method == "slim") front_result = run_slim(front_welded);
            else if (method == "stein_admm") front_result = run_stein(front_welded);
            else if (method == "cgal_conformal") front_result = run_cgal(front_welded, cgalparam::ParamMethod::DiscreteConformal, "cgal_conformal", false, {});
            else if (method == "cgal_arap") front_result = run_cgal(front_welded, cgalparam::ParamMethod::ARAP, "cgal_arap", false, {});
            else if (method == "cgal_authalic") front_result = run_cgal(front_welded, cgalparam::ParamMethod::DiscreteAuthalic, "cgal_authalic", false, {});
#ifdef HAS_COMPMAJOR
            else if (method == "cm") front_result = run_cm(front_welded);
#endif
            else front_result.error = "Unknown method: " + method;

            // Parameterize back with LSCM
            auto back_result = run_lscm(back_welded);

            if (front_result.success) {
                // Combine both halves
                auto fp = meshparam::load_gltf_from_memory(front_result.glb);
                auto bp = back_result.success ? meshparam::load_gltf_from_memory(back_result.glb) : bm;
                int fnv = fp.num_vertices(), bnv = bp.num_vertices();
                int fnf = fp.num_faces(), bnf = bp.num_faces();

                // Write 2-primitive GLB manually (raw JSON + binary, like occ_gmsh_pipeline.py)
                result = front_result;
                result.vertices = fnv + bnv;
                result.faces = fnf + bnf;
                {
                    auto orig_front = meshparam::load_gltf_from_memory(front_glb);
                    auto orig_back = meshparam::load_gltf_from_memory(back_glb);

                    // Build binary buffer
                    std::vector<uint8_t> bin;
                    auto appendF32 = [&](const float* data, size_t count) {
                        size_t off = bin.size();
                        bin.insert(bin.end(), (uint8_t*)data, (uint8_t*)(data + count));
                        return off;
                    };
                    auto appendMatF32 = [&](const Eigen::MatrixXd& M) {
                        size_t off = bin.size();
                        for (int i = 0; i < M.rows(); i++)
                            for (int j = 0; j < M.cols(); j++) {
                                float v = (float)M(i,j);
                                bin.insert(bin.end(), (uint8_t*)&v, (uint8_t*)&v + 4);
                            }
                        return off;
                    };
                    auto appendVecF32 = [&](const Eigen::VectorXd& V) {
                        size_t off = bin.size();
                        for (int i = 0; i < V.rows(); i++) {
                            float v = (float)V(i);
                            bin.insert(bin.end(), (uint8_t*)&v, (uint8_t*)&v + 4);
                        }
                        return off;
                    };
                    auto appendIdx = [&](const Eigen::MatrixXi& F) {
                        size_t off = bin.size();
                        for (int i = 0; i < F.rows(); i++)
                            for (int j = 0; j < F.cols(); j++) {
                                uint32_t v = (uint32_t)F(i,j);
                                bin.insert(bin.end(), (uint8_t*)&v, (uint8_t*)&v + 4);
                            }
                        return off;
                    };

                    // Front: pos, nrm, uv, fid, seam
                    size_t o_fp = appendMatF32(fp.V);
                    size_t o_fn = appendMatF32(fp.has_normals() ? fp.N : orig_front.N);
                    size_t o_fu = appendMatF32(fp.UV);
                    size_t o_ff = appendVecF32(orig_front.has_face_ids() ? orig_front.face_ids : Eigen::VectorXd::Constant(fnv,-1));
                    size_t o_fs = appendVecF32(orig_front.has_seam() ? orig_front.seam : Eigen::VectorXd::Zero(fnv));
                    // Back: pos, nrm, uv, fid, seam
                    size_t o_bp = appendMatF32(bp.V);
                    size_t o_bn = appendMatF32(bp.has_normals() ? bp.N : orig_back.N);
                    size_t o_bu = appendMatF32(bp.has_uvs() ? bp.UV : Eigen::MatrixXd::Zero(bnv,2));
                    size_t o_bf = appendVecF32(orig_back.has_face_ids() ? orig_back.face_ids : Eigen::VectorXd::Constant(bnv,-1));
                    size_t o_bs = appendVecF32(orig_back.has_seam() ? orig_back.seam : Eigen::VectorXd::Zero(bnv));
                    // Indices
                    size_t o_fi = appendIdx(fp.F);
                    size_t o_bi = appendIdx(bp.F);
                    while (bin.size() % 4) bin.push_back(0);

                    Eigen::Vector3d fmn=fp.V.colwise().minCoeff(), fmx=fp.V.colwise().maxCoeff();
                    Eigen::Vector3d bmn=bp.V.colwise().minCoeff(), bmx=bp.V.colwise().maxCoeff();

                    // Build JSON manually
                    auto jBV = [](size_t off, size_t len, int tgt) {
                        return "{\"buffer\":0,\"byteOffset\":" + std::to_string(off) +
                               ",\"byteLength\":" + std::to_string(len) +
                               ",\"target\":" + std::to_string(tgt) + "}";
                    };
                    auto jAcc = [](int bv, int ct, int n, const std::string& tp,
                                   const std::string& mn="", const std::string& mx="") {
                        std::string s = "{\"bufferView\":" + std::to_string(bv) +
                            ",\"componentType\":" + std::to_string(ct) +
                            ",\"count\":" + std::to_string(n) +
                            ",\"type\":\"" + tp + "\"";
                        if (!mn.empty()) s += ",\"min\":" + mn + ",\"max\":" + mx;
                        return s + "}";
                    };
                    auto v3s = [](const Eigen::Vector3d& v) {
                        return "[" + std::to_string(v(0)) + "," + std::to_string(v(1)) + "," + std::to_string(v(2)) + "]";
                    };

                    // BV: 0-4 front attrs, 5-9 back attrs, 10 front idx, 11 back idx
                    std::string bvs =
                        jBV(o_fp,fnv*12,34962) + "," + jBV(o_fn,fnv*12,34962) + "," +
                        jBV(o_fu,fnv*8,34962) + "," + jBV(o_ff,fnv*4,34962) + "," +
                        jBV(o_fs,fnv*4,34962) + "," +
                        jBV(o_bp,bnv*12,34962) + "," + jBV(o_bn,bnv*12,34962) + "," +
                        jBV(o_bu,bnv*8,34962) + "," + jBV(o_bf,bnv*4,34962) + "," +
                        jBV(o_bs,bnv*4,34962) + "," +
                        jBV(o_fi,fnf*12,34963) + "," + jBV(o_bi,bnf*12,34963);

                    // Acc: 0-4 front, 5-9 back, 10 front idx, 11 back idx
                    std::string accs =
                        jAcc(0,5126,fnv,"VEC3",v3s(fmn),v3s(fmx)) + "," +
                        jAcc(1,5126,fnv,"VEC3") + "," +
                        jAcc(2,5126,fnv,"VEC2") + "," +
                        jAcc(3,5126,fnv,"SCALAR") + "," +
                        jAcc(4,5126,fnv,"SCALAR") + "," +
                        jAcc(5,5126,bnv,"VEC3",v3s(bmn),v3s(bmx)) + "," +
                        jAcc(6,5126,bnv,"VEC3") + "," +
                        jAcc(7,5126,bnv,"VEC2") + "," +
                        jAcc(8,5126,bnv,"SCALAR") + "," +
                        jAcc(9,5126,bnv,"SCALAR") + "," +
                        jAcc(10,5125,fnf*3,"SCALAR") + "," +
                        jAcc(11,5125,bnf*3,"SCALAR");

                    std::string json_str =
                        "{\"asset\":{\"version\":\"2.0\",\"generator\":\"meshparam_bench\"},"
                        "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],\"nodes\":[{\"mesh\":0}],"
                        "\"meshes\":[{\"primitives\":["
                        "{\"attributes\":{\"POSITION\":0,\"NORMAL\":1,\"TEXCOORD_0\":2,\"_FACE_ID\":3,\"_SEAM\":4},\"indices\":10,\"mode\":4},"
                        "{\"attributes\":{\"POSITION\":5,\"NORMAL\":6,\"TEXCOORD_0\":7,\"_FACE_ID\":8,\"_SEAM\":9},\"indices\":11,\"mode\":4}"
                        "]}],"
                        "\"accessors\":[" + accs + "],"
                        "\"bufferViews\":[" + bvs + "],"
                        "\"buffers\":[{\"byteLength\":" + std::to_string(bin.size()) + "}]}";

                    // Pad JSON to 4-byte alignment
                    while (json_str.size() % 4) json_str += ' ';

                    // Build GLB
                    uint32_t totalSize = 12 + 8 + (uint32_t)json_str.size() + 8 + (uint32_t)bin.size();
                    result.glb.clear();
                    result.glb.reserve(totalSize);
                    auto w32 = [&](uint32_t v) { result.glb.insert(result.glb.end(), (uint8_t*)&v, (uint8_t*)&v+4); };
                    // Header
                    result.glb.insert(result.glb.end(), {'g','l','T','F'});
                    w32(2); w32(totalSize);
                    // JSON chunk
                    w32((uint32_t)json_str.size()); w32(0x4E4F534A);
                    result.glb.insert(result.glb.end(), json_str.begin(), json_str.end());
                    // BIN chunk
                    w32((uint32_t)bin.size()); w32(0x004E4942);
                    result.glb.insert(result.glb.end(), bin.begin(), bin.end());
                }
                did_split = true;
                std::cerr << "[split] Combined: " << result.vertices << "v " << result.faces << "f" << std::endl;
            }
        }

        if (!did_split) {
        // Single primitive — standard path (OBJ files, benchmark meshes)
        auto welded = weld_vertices(glb);
        heal_mesh(welded, false);
        welded = extract_largest_component(welded);

        // Dispatch method (single primitive, BFS seam fallback)
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
