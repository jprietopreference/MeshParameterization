#include "cgalparam/gltf_io.h"
#include "cgalparam/cgal_parameterize.h"
#include "cgalparam/distortion.h"
#include <iostream>
#include <string>
#include <map>

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <input.glb> <output.glb> [--method METHOD]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Methods:" << std::endl;
    std::cerr << "  lscm      Least Squares Conformal Maps (default)" << std::endl;
    std::cerr << "  arap      As Rigid As Possible" << std::endl;
    std::cerr << "  conformal Discrete Conformal Map (fixed border)" << std::endl;
    std::cerr << "  authalic  Discrete Authalic (fixed border)" << std::endl;
    std::cerr << "  meanvalue Mean Value Coordinates (fixed border)" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) { print_usage(argv[0]); return 1; }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    cgalparam::ParamMethod method = cgalparam::ParamMethod::LSCM;

    static const std::map<std::string, cgalparam::ParamMethod> methods = {
        {"lscm", cgalparam::ParamMethod::LSCM},
        {"arap", cgalparam::ParamMethod::ARAP},
        {"conformal", cgalparam::ParamMethod::DiscreteConformal},
        {"authalic", cgalparam::ParamMethod::DiscreteAuthalic},
        {"meanvalue", cgalparam::ParamMethod::MeanValue},
    };

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--method" && i + 1 < argc) {
            auto it = methods.find(argv[++i]);
            if (it == methods.end()) {
                std::cerr << "Unknown method: " << argv[i] << std::endl;
                print_usage(argv[0]);
                return 1;
            }
            method = it->second;
        }
    }

    try {
        std::cout << "Loading: " << input_path << std::endl;
        auto mesh = cgalparam::load_gltf(input_path);
        std::cout << "Loaded: " << mesh.num_vertices() << " vertices, "
                  << mesh.num_faces() << " faces" << std::endl;

        auto sm = cgalparam::to_cgal_mesh(mesh);
        auto UV = cgalparam::parameterize(sm, method);

        // Rebuild mesh from (possibly cut) CGAL mesh to match UV vertex count
        mesh = cgalparam::from_cgal_mesh(sm);
        mesh.UV = UV;

        auto metrics = cgalparam::compute_distortion(mesh.V, mesh.F, UV);
        metrics.print_summary();

        std::cout << "Saving: " << output_path << std::endl;
        cgalparam::save_gltf(output_path, mesh);
        std::cout << "Done." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
