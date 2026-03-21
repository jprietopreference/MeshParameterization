#include "meshparam/gltf_io.h"
#include "meshparam/parameterizer.h"
#include <iostream>
#include <string>

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <input.glb|gltf> <output.glb> [--no-fill]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Parameterizes a mesh using heat-based geodesics and MDS." << std::endl;
    std::cerr << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --no-fill        Disable Poisson surface fill" << std::endl;
    std::cerr << "  --view-weighted  Weight UV quality toward +Z facing faces" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];

    meshparam::ParamConfig config;
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--no-fill") {
            config.use_poisson_fill = false;
            config.auto_detect_fill = false;
        }
        if (arg == "--view-weighted") {
            config.view_weighted = true;
        }
    }

    try {
        std::cout << "Loading mesh from: " << input_path << std::endl;
        auto mesh = meshparam::load_gltf(input_path);
        std::cout << "Loaded: " << mesh.num_vertices() << " vertices, "
                  << mesh.num_faces() << " faces" << std::endl;

        if (mesh.has_uvs()) {
            std::cout << "Warning: Mesh already has UV coordinates. They will be overwritten." << std::endl;
        }

        auto result = meshparam::parameterize(mesh, config);

        std::cout << "Saving parameterized mesh to: " << output_path << std::endl;
        meshparam::save_gltf(output_path, result);
        std::cout << "Done." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
