#include <emscripten/bind.h>
#include <emscripten/val.h>
#include "meshparam/gltf_io.h"
#include "meshparam/parameterizer.h"
#include <vector>
#include <cstdint>

using namespace emscripten;

/// Parameterize a glTF/glb buffer and return the result as a glb buffer.
val parameterize_gltf(val input_array, bool use_poisson_fill) {
    // Convert JS Uint8Array to std::vector<uint8_t>
    auto length = input_array["length"].as<unsigned>();
    std::vector<uint8_t> input_data(length);
    for (unsigned i = 0; i < length; ++i) {
        input_data[i] = input_array[i].as<uint8_t>();
    }

    // Load mesh
    auto mesh = meshparam::load_gltf_from_memory(input_data);

    // Configure and run parameterization
    meshparam::ParamConfig config;
    config.use_poisson_fill = use_poisson_fill;
    config.auto_detect_fill = true;

    auto result = meshparam::parameterize(mesh, config);

    // Save to glb buffer
    auto output_data = meshparam::save_gltf_to_memory(result);

    // Convert to JS Uint8Array
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
    info.set("boundaryLoops", static_cast<int>(mesh.boundary_loops().size()));
    return info;
}

EMSCRIPTEN_BINDINGS(meshparam) {
    function("parameterizeGltf", &parameterize_gltf);
    function("getMeshInfo", &get_mesh_info);
}
