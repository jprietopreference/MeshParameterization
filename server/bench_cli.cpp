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

    // If OBJ has UVs with different indexing (FTC != F), expand to per-vertex UVs
    if (TC.rows() > 0 && FTC.rows() == F.rows() && FTC.maxCoeff() < TC.rows()) {
        // Check if UV indices match vertex indices
        bool same_indexing = (FTC.array() == F.array()).all();
        if (same_indexing) {
            mesh.V = V;
            mesh.F = F;
            mesh.UV = TC;
        } else {
            // Expand: create new vertex per unique (position, UV) pair
            int nf = F.rows();
            // Build expanded mesh
            Eigen::MatrixXd newV(nf * 3, 3);
            Eigen::MatrixXd newUV(nf * 3, 2);
            Eigen::MatrixXi newF(nf, 3);
            for (int fi = 0; fi < nf; fi++) {
                for (int j = 0; j < 3; j++) {
                    int vi = F(fi, j);
                    int ti = FTC(fi, j);
                    newV.row(fi * 3 + j) = V.row(vi);
                    newUV.row(fi * 3 + j) = TC.row(ti);
                    newF(fi, j) = fi * 3 + j;
                }
            }
            mesh.V = newV;
            mesh.F = newF;
            mesh.UV = newUV;
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
        else if (json_path.empty() && arg[0] != '-') json_path = arg; // legacy positional
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

        // Weld + heal
        auto welded = weld_vertices(glb);
        heal_mesh(welded, false);

        // Dispatch method
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
