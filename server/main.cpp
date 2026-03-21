// Mesh Parameterization HTTP Server
// Serves parameterization API + static web frontend.

#include "httplib.h"

// Heat-geodesic parameterizer
#include "meshparam/gltf_io.h"
#include "meshparam/parameterizer.h"
#include "meshparam/distortion.h"

// CGAL parameterizer
#include "cgalparam/gltf_io.h"
#include "cgalparam/cgal_parameterize.h"
#include "cgalparam/seam_cut.h"
#include "cgalparam/distortion.h"
#include "cgalparam/types.h"

#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <cstdlib>

namespace {

std::string metrics_json(const meshparam::DistortionMetrics& m, int nv, int nf, double elapsed_ms) {
    std::ostringstream oss;
    oss << "{\"vertices\":" << nv
        << ",\"faces\":" << nf
        << ",\"elapsed_ms\":" << elapsed_ms
        << ",\"angle_mean\":" << m.mean_angle_distortion
        << ",\"angle_max\":" << m.max_angle_distortion
        << ",\"area_mean\":" << m.mean_area_distortion
        << ",\"stretch_mean\":" << m.mean_stretch
        << ",\"stretch_max\":" << m.max_stretch
        << ",\"iso_rms\":" << m.isometric_rms
        << "}";
    return oss.str();
}

std::string metrics_json(const cgalparam::DistortionMetrics& m, int nv, int nf, double elapsed_ms) {
    std::ostringstream oss;
    oss << "{\"vertices\":" << nv
        << ",\"faces\":" << nf
        << ",\"elapsed_ms\":" << elapsed_ms
        << ",\"angle_mean\":" << m.mean_angle_distortion
        << ",\"angle_max\":" << m.max_angle_distortion
        << ",\"area_mean\":" << m.mean_area_distortion
        << ",\"stretch_mean\":" << m.mean_stretch
        << ",\"stretch_max\":" << m.max_stretch
        << "}";
    return oss.str();
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    int port = 8080;
    std::string web_root = "../web";
    std::string occ_cli = "";
    int threads = 8;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
        if (arg == "--web-root" && i + 1 < argc) web_root = argv[++i];
        if (arg == "--occ-cli" && i + 1 < argc) occ_cli = argv[++i];
        if (arg == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
    }

    httplib::Server svr;

    // Allow large request bodies (up to 100 MB for large meshes/STEP files)
    svr.set_payload_max_length(100 * 1024 * 1024);

    // Thread pool for concurrent requests
    svr.new_task_queue = [threads] { return new httplib::ThreadPool(threads); };

    // CORS headers on all responses
    svr.set_post_routing_handler([](const auto& req, auto& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        res.set_header("Access-Control-Expose-Headers", "X-Metrics");
    });

    // --- Health check ---
    svr.Get("/api/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    // --- Heat parameterization ---
    svr.Post("/api/parameterize/heat", [](const httplib::Request& req, httplib::Response& res) {
        auto t0 = std::chrono::high_resolution_clock::now();

        try {
            std::vector<uint8_t> input(req.body.begin(), req.body.end());
            auto mesh = meshparam::load_gltf_from_memory(input);

            meshparam::ParamConfig config;
            config.auto_detect_fill = true;
            config.use_poisson_fill = true;

            if (req.has_param("viewWeighted") && req.get_param_value("viewWeighted") == "true") {
                config.view_weighted = true;
            }

            auto result = meshparam::parameterize(mesh, config);
            auto output = meshparam::save_gltf_to_memory(result);

            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            auto metrics = meshparam::compute_distortion(result.V, result.F, result.UV);
            std::string mj = metrics_json(metrics, result.num_vertices(), result.num_faces(), ms);

            res.set_header("X-Metrics", mj);
            // CORS: handled globally
            res.set_header("Access-Control-Expose-Headers", "X-Metrics");
            res.set_content(std::string(output.begin(), output.end()), "model/gltf-binary");

        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(std::string("{\"error\":\"") + e.what() + "\"}", "application/json");
        }
    });

    // --- CGAL parameterization ---
    svr.Post("/api/parameterize/cgal", [](const httplib::Request& req, httplib::Response& res) {
        auto t0 = std::chrono::high_resolution_clock::now();

        try {
            std::vector<uint8_t> input(req.body.begin(), req.body.end());
            auto tri = cgalparam::load_gltf_from_memory(input);
            auto sm = cgalparam::to_cgal_mesh(tri);

            // Cut seam if closed
            if (CGAL::is_closed(sm)) {
                auto cut = cgalparam::cut_to_disk(sm);
                sm = cut.cut_mesh;
            }

            // Parse method
            std::string method_str = req.has_param("method") ? req.get_param_value("method") : "conformal";
            cgalparam::ParamMethod method = cgalparam::ParamMethod::DiscreteConformal;
            if (method_str == "arap") method = cgalparam::ParamMethod::ARAP;
            else if (method_str == "authalic") method = cgalparam::ParamMethod::DiscreteAuthalic;
            else if (method_str == "mvc") method = cgalparam::ParamMethod::MeanValue;

            Eigen::MatrixXd UV = cgalparam::parameterize(sm, method);
            auto result = cgalparam::from_cgal_mesh(sm);
            result.UV = UV;

            auto output = cgalparam::save_gltf_to_memory(result);

            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            auto metrics = cgalparam::compute_distortion(result.V, result.F, result.UV);
            std::string mj = metrics_json(metrics, result.num_vertices(), result.num_faces(), ms);

            res.set_header("X-Metrics", mj);
            // CORS: handled globally
            res.set_header("Access-Control-Expose-Headers", "X-Metrics");
            res.set_content(std::string(output.begin(), output.end()), "model/gltf-binary");

        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(std::string("{\"error\":\"") + e.what() + "\"}", "application/json");
        }
    });

    // --- STEP tessellation (delegates to occ_tessellate_cli) ---
    svr.Post("/api/tessellate/step", [&occ_cli](const httplib::Request& req, httplib::Response& res) {
        if (occ_cli.empty()) {
            res.status = 501;
            res.set_content("{\"error\":\"OCC CLI not configured. Use --occ-cli flag.\"}", "application/json");
            return;
        }

        try {
            // Write STEP to temp file
            std::string tmp_step = "temp_input.step";
            std::string tmp_glb = "temp_output.glb";
            {
                std::ofstream f(tmp_step, std::ios::binary);
                f.write(req.body.data(), req.body.size());
            }

            // Call OCC CLI
            std::string cmd = occ_cli + " " + tmp_step + " " + tmp_glb + " --deflection 1.0";
            int ret = std::system(cmd.c_str());
            std::remove(tmp_step.c_str());

            if (ret != 0) {
                std::remove(tmp_glb.c_str());
                res.status = 500;
                res.set_content("{\"error\":\"OCC tessellation failed\"}", "application/json");
                return;
            }

            // Read result
            std::ifstream f(tmp_glb, std::ios::binary | std::ios::ate);
            size_t sz = f.tellg();
            f.seekg(0);
            std::string data(sz, '\0');
            f.read(data.data(), sz);
            f.close();
            std::remove(tmp_glb.c_str());

            // CORS: handled globally
            res.set_content(data, "model/gltf-binary");

        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(std::string("{\"error\":\"") + e.what() + "\"}", "application/json");
        }
    });

    // --- CORS preflight (global handler covers headers, just return 204) ---
    svr.Options("/api/(.*)", [](const httplib::Request&, httplib::Response& res) {
        res.status = 204;
    });

    // --- Static file serving (web frontend) ---
    svr.set_mount_point("/", web_root);

    std::cout << "Mesh Parameterization Server" << std::endl;
    std::cout << "  Port:     " << port << std::endl;
    std::cout << "  Threads:  " << threads << std::endl;
    std::cout << "  Web root: " << web_root << std::endl;
    std::cout << "  OCC CLI:  " << (occ_cli.empty() ? "(not configured)" : occ_cli) << std::endl;
    std::cout << "  Endpoints:" << std::endl;
    std::cout << "    POST /api/parameterize/heat" << std::endl;
    std::cout << "    POST /api/parameterize/cgal?method=conformal|arap|authalic|mvc" << std::endl;
    std::cout << "    POST /api/tessellate/step" << std::endl;
    std::cout << "    GET  /api/health" << std::endl;
    std::cout << std::endl;
    std::cout << "Listening on http://localhost:" << port << std::endl;

    svr.listen("0.0.0.0", port);
    return 0;
}
