// Mesh Parameterization Benchmark CLI
// Runs a single parameterization method on an OBJ file.
// Process isolation: each invocation is a separate process.
//
// Usage: meshparam_bench <method> <input.obj> [output.json]
// Methods: heat, lscm, igl_arap, slim, stein_admm,
//          cgal_conformal, cgal_arap, cgal_authalic

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

// ============================================================
// Include shared method code from main.cpp via the same pattern
// (MethodResult, weld, heal, cut, run_* functions)
// ============================================================

// --- Shared: pulled from main.cpp lines 66-890 ---
// Rather than duplicating, we include the same logic.
// The key structures and functions are defined inline below.

#include "methods.inc"

// ============================================================
// OBJ → in-memory GLB conversion
// ============================================================
std::vector<uint8_t> obj_to_glb(const std::string& obj_path) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    if (!igl::readOBJ(obj_path, V, F)) {
        throw std::runtime_error("Failed to read OBJ: " + obj_path);
    }
    if (V.rows() == 0 || F.rows() == 0) {
        throw std::runtime_error("Empty mesh in " + obj_path);
    }

    meshparam::TriMesh mesh;
    mesh.V = V;
    mesh.F = F;
    return meshparam::save_gltf_to_memory(mesh);
}

// ============================================================
// Main
// ============================================================
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: meshparam_bench <method> <input.obj> [output.json]" << std::endl;
        std::cerr << "Methods: heat, lscm, igl_arap, slim, stein_admm," << std::endl;
        std::cerr << "         cgal_conformal, cgal_arap, cgal_authalic" << std::endl;
        return 1;
    }

    std::string method = argv[1];
    std::string input_path = argv[2];
    std::string output_path = (argc > 3) ? argv[3] : "";

    MethodResult result;
    result.method = method;

    try {
        // Load OBJ → GLB
        auto glb = obj_to_glb(input_path);

        // Weld split vertices
        auto welded = weld_vertices(glb);

        // Heal degenerate triangles
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
    } catch (const std::exception& e) {
        result.error = e.what();
    }

    // Output JSON
    std::string json = result.to_json();

    if (output_path.empty() || output_path == "-") {
        std::cout << json << std::endl;
    } else {
        std::ofstream f(output_path);
        f << json << std::endl;
    }

    return result.success ? 0 : 1;
}
