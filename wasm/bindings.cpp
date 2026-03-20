#include <emscripten/bind.h>
#include <emscripten/val.h>
#include "meshparam/gltf_io.h"
#include "meshparam/parameterizer.h"
#include "meshparam/distortion.h"
#include <vector>
#include <cstdint>
#include <string>
#include <sstream>

using namespace emscripten;

static std::string g_metrics_json;

/// Parameterize a glTF/glb buffer and return the result as a glb buffer.
val parameterize_gltf(val input_array, bool use_poisson_fill) {
    auto length = input_array["length"].as<unsigned>();
    std::vector<uint8_t> input_data(length);
    val memory = val::module_property("HEAPU8");
    // Fast copy from JS
    for (unsigned i = 0; i < length; ++i) {
        input_data[i] = input_array[i].as<uint8_t>();
    }

    auto mesh = meshparam::load_gltf_from_memory(input_data);

    meshparam::ParamConfig config;
    config.use_poisson_fill = use_poisson_fill;
    config.auto_detect_fill = true;

    auto result = meshparam::parameterize(mesh, config);

    // Compute distortion metrics
    std::ostringstream oss;
    oss << "{\"vertices\":" << result.num_vertices()
        << ",\"faces\":" << result.num_faces();
    if (result.has_uvs()) {
        auto metrics = meshparam::compute_distortion(result.V, result.F, result.UV);
        oss << ",\"angle_mean\":" << metrics.mean_angle_distortion
            << ",\"angle_max\":" << metrics.max_angle_distortion
            << ",\"area_mean\":" << metrics.mean_area_distortion
            << ",\"stretch_mean\":" << metrics.mean_stretch
            << ",\"stretch_max\":" << metrics.max_stretch
            << ",\"iso_rms\":" << metrics.isometric_rms;
    }
    oss << "}";
    g_metrics_json = oss.str();

    auto output_data = meshparam::save_gltf_to_memory(result);

    val output_array = val::global("Uint8Array").new_(output_data.size());
    for (size_t i = 0; i < output_data.size(); ++i) {
        output_array.set(i, output_data[i]);
    }
    return output_array;
}

/// Get mesh info from a glTF/glb buffer
val get_mesh_info(val input_array) {
    auto length = input_array["length"].as<unsigned>();
    std::vector<uint8_t> input_data(length);
    for (unsigned i = 0; i < length; ++i) {
        input_data[i] = input_array[i].as<uint8_t>();
    }
    auto mesh = meshparam::load_gltf_from_memory(input_data);

    val info = val::object();
    info.set("vertices", mesh.num_vertices());
    info.set("faces", mesh.num_faces());
    info.set("hasUVs", mesh.has_uvs());
    info.set("hasBoundary", mesh.has_boundary());
    return info;
}

std::string get_metrics() {
    return g_metrics_json;
}

EMSCRIPTEN_BINDINGS(meshparam) {
    function("parameterizeGltf", &parameterize_gltf);
    function("getMeshInfo", &get_mesh_info);
    function("getMetrics", &get_metrics);
}
