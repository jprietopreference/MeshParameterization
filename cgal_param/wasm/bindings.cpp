#include <emscripten/bind.h>
#include <emscripten/val.h>
#include "cgalparam/gltf_io.h"
#include "cgalparam/cgal_parameterize.h"
#include "cgalparam/distortion.h"
#include "cgalparam/seam_cut.h"
#include "cgalparam/types.h"
#include <string>
#include <sstream>

using namespace emscripten;

static std::string g_metrics_json;

val parameterize_gltf(val input_array, std::string method_str) {
    auto length = input_array["length"].as<unsigned>();
    std::vector<uint8_t> data(length);
    for (unsigned i = 0; i < length; ++i) data[i] = input_array[i].as<uint8_t>();

    // Load mesh
    cgalparam::TriMesh tri = cgalparam::load_gltf_from_memory(data);
    cgalparam::SurfaceMesh sm = cgalparam::to_cgal_mesh(tri);

    // Cut seam if closed
    if (CGAL::is_closed(sm)) {
        auto cut_result = cgalparam::cut_to_disk(sm);
        sm = cut_result.cut_mesh;
    }

    // Parse method
    cgalparam::ParamMethod method = cgalparam::ParamMethod::DiscreteConformal;
    if (method_str == "arap") method = cgalparam::ParamMethod::ARAP;
    else if (method_str == "authalic") method = cgalparam::ParamMethod::DiscreteAuthalic;
    else if (method_str == "mvc") method = cgalparam::ParamMethod::MeanValue;
    else if (method_str == "lscm") method = cgalparam::ParamMethod::LSCM;

    // Parameterize
    Eigen::MatrixXd UV = cgalparam::parameterize(sm, method);

    // Convert back to TriMesh
    cgalparam::TriMesh result = cgalparam::from_cgal_mesh(sm);
    result.UV = UV;

    // Compute metrics
    auto metrics = cgalparam::compute_distortion(result.V, result.F, result.UV);
    std::ostringstream oss;
    oss << "{\"angle_mean\":" << metrics.mean_angle_distortion
        << ",\"angle_max\":" << metrics.max_angle_distortion
        << ",\"area_mean\":" << metrics.mean_area_distortion
        << ",\"stretch_mean\":" << metrics.mean_stretch
        << ",\"stretch_max\":" << metrics.max_stretch
        << ",\"vertices\":" << result.num_vertices()
        << ",\"faces\":" << result.num_faces()
        << "}";
    g_metrics_json = oss.str();

    // Save to memory
    auto output = cgalparam::save_gltf_to_memory(result);

    val out = val::global("Uint8Array").new_(output.size());
    for (size_t i = 0; i < output.size(); ++i) out.set(i, output[i]);
    return out;
}

std::string get_metrics() {
    return g_metrics_json;
}

EMSCRIPTEN_BINDINGS(cgalparam) {
    function("parameterizeGltf", &parameterize_gltf);
    function("getMetrics", &get_metrics);
}
