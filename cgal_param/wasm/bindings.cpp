#include <emscripten/bind.h>
#include <emscripten/val.h>
#include "cgalparam/gltf_io.h"
#include "cgalparam/cgal_parameterize.h"

using namespace emscripten;

val parameterize_gltf(val input_array, std::string method_str) {
    auto length = input_array["length"].as<unsigned>();
    std::vector<uint8_t> data(length);
    for (unsigned i = 0; i < length; ++i) data[i] = input_array[i].as<uint8_t>();

    // TODO: implement from-memory loading
    return val::null();
}

EMSCRIPTEN_BINDINGS(cgalparam) {
    function("parameterizeGltf", &parameterize_gltf);
}
