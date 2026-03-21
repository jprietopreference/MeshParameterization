// Mesh Parameterization Broker Server
// Runs multiple parameterization methods in parallel, picks the best result.

#include "httplib.h"

#include "meshparam/gltf_io.h"
#include "meshparam/parameterizer.h"
#include "meshparam/distortion.h"

#include "cgalparam/gltf_io.h"
#include "cgalparam/cgal_parameterize.h"
#include "cgalparam/seam_cut.h"
#include "cgalparam/distortion.h"
#include "cgalparam/types.h"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <atomic>

// ============================================================
// Method result
// ============================================================
struct MethodResult {
    std::string method;
    bool success = false;
    std::string error;
    double elapsed_ms = 0;
    double angle_mean = 999;
    double angle_max = 999;
    double area_mean = 0;
    double area_std = 999;
    double stretch_mean = 999;
    double stretch_max = 999;
    double iso_rms = -1;
    int vertices = 0;
    int faces = 0;
    std::vector<uint8_t> glb;

    // Composite quality score (lower = better)
    double score() const {
        if (!success) return 1e18;
        // Weighted: angle distortion (40%), stretch (40%), area uniformity (20%)
        return angle_mean * 0.4 + std::log1p(stretch_mean) * 10.0 * 0.4 + area_std * 0.2;
    }

    std::string to_json() const {
        std::ostringstream o;
        o << "{\"method\":\"" << method << "\""
          << ",\"success\":" << (success ? "true" : "false");
        if (!success) {
            o << ",\"error\":\"" << error << "\"";
        } else {
            o << ",\"elapsed_ms\":" << elapsed_ms
              << ",\"vertices\":" << vertices
              << ",\"faces\":" << faces
              << ",\"angle_mean\":" << angle_mean
              << ",\"angle_max\":" << angle_max
              << ",\"area_mean\":" << area_mean
              << ",\"area_std\":" << area_std
              << ",\"stretch_mean\":" << stretch_mean
              << ",\"stretch_max\":" << stretch_max
              << ",\"iso_rms\":" << iso_rms
              << ",\"score\":" << score();
        }
        o << "}";
        return o.str();
    }
};

// ============================================================
// Run individual methods
// ============================================================
MethodResult run_heat(const std::vector<uint8_t>& input_glb, bool view_weighted) {
    MethodResult r;
    r.method = "heat";
    auto t0 = std::chrono::high_resolution_clock::now();
    try {
        auto mesh = meshparam::load_gltf_from_memory(input_glb);
        meshparam::ParamConfig config;
        config.auto_detect_fill = true;
        config.use_poisson_fill = true;
        config.view_weighted = view_weighted;

        auto result = meshparam::parameterize(mesh, config);
        auto metrics = meshparam::compute_distortion(result.V, result.F, result.UV);

        r.glb = meshparam::save_gltf_to_memory(result);
        r.success = true;
        r.vertices = result.num_vertices();
        r.faces = result.num_faces();
        r.angle_mean = metrics.mean_angle_distortion;
        r.angle_max = metrics.max_angle_distortion;
        r.area_mean = metrics.mean_area_distortion;
        r.area_std = metrics.std_area_distortion;
        r.stretch_mean = metrics.mean_stretch;
        r.stretch_max = metrics.max_stretch;
        r.iso_rms = metrics.isometric_rms;
    } catch (const std::exception& e) {
        r.error = e.what();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    r.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return r;
}

MethodResult run_cgal(const std::vector<uint8_t>& input_glb, cgalparam::ParamMethod method, const std::string& method_name) {
    MethodResult r;
    r.method = method_name;
    auto t0 = std::chrono::high_resolution_clock::now();
    try {
        auto tri = cgalparam::load_gltf_from_memory(input_glb);
        auto sm = cgalparam::to_cgal_mesh(tri);

        if (CGAL::is_closed(sm)) {
            auto cut = cgalparam::cut_to_disk(sm);
            sm = cut.cut_mesh;
        }

        Eigen::MatrixXd UV = cgalparam::parameterize(sm, method);
        auto result = cgalparam::from_cgal_mesh(sm);
        result.UV = UV;

        auto metrics = cgalparam::compute_distortion(result.V, result.F, result.UV);

        r.glb = cgalparam::save_gltf_to_memory(result);
        r.success = true;
        r.vertices = result.num_vertices();
        r.faces = result.num_faces();
        r.angle_mean = metrics.mean_angle_distortion;
        r.angle_max = metrics.max_angle_distortion;
        r.area_mean = metrics.mean_area_distortion;
        r.area_std = metrics.std_area_distortion;
        r.stretch_mean = metrics.mean_stretch;
        r.stretch_max = metrics.max_stretch;
    } catch (const std::exception& e) {
        r.error = e.what();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    r.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return r;
}

// ============================================================
// STEP tessellation via external CLI
// ============================================================
std::vector<uint8_t> tessellate_step(const std::string& occ_cli, const std::string& step_data, double deflection) {
    // Use thread-safe temp file names
    // Use absolute temp paths to avoid working directory issues
    auto tid = std::this_thread::get_id();
    std::ostringstream ss;
    // Use system temp directory
    const char* tmp_dir = std::getenv("TEMP");
    if (!tmp_dir) tmp_dir = std::getenv("TMP");
    if (!tmp_dir) tmp_dir = ".";
    ss << tmp_dir << "/meshparam_" << tid;
    std::string tmp_step = ss.str() + ".step";
    std::string tmp_glb = ss.str() + ".glb";

    { std::ofstream f(tmp_step, std::ios::binary); f.write(step_data.data(), step_data.size()); }

    std::string cmd = occ_cli + " \"" + tmp_step + "\" \"" + tmp_glb + "\" --deflection " + std::to_string(deflection);
    int ret = std::system(cmd.c_str());
    std::remove(tmp_step.c_str());

    if (ret != 0) {
        std::remove(tmp_glb.c_str());
        throw std::runtime_error("OCC tessellation failed");
    }

    std::ifstream f(tmp_glb, std::ios::binary | std::ios::ate);
    size_t sz = f.tellg(); f.seekg(0);
    std::vector<uint8_t> data(sz);
    f.read(reinterpret_cast<char*>(data.data()), sz);
    f.close();
    std::remove(tmp_glb.c_str());
    return data;
}

// ============================================================
// Main
// ============================================================
int main(int argc, char* argv[]) {
    int port = 8080;
    std::string web_root = "../web";
    std::string occ_cli = "";
    std::string gmsh_cli = "";
    int threads = 8;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
        if (arg == "--web-root" && i + 1 < argc) web_root = argv[++i];
        if (arg == "--occ-cli" && i + 1 < argc) occ_cli = argv[++i];
        if (arg == "--gmsh-cli" && i + 1 < argc) gmsh_cli = argv[++i];
        if (arg == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
    }

    httplib::Server svr;
    svr.set_payload_max_length(100 * 1024 * 1024);
    svr.new_task_queue = [threads] { return new httplib::ThreadPool(threads); };

    // CORS on all responses
    svr.set_post_routing_handler([](const auto&, auto& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        res.set_header("Access-Control-Expose-Headers", "X-Metrics, X-Method, X-All-Methods");
    });

    svr.Options("/api/(.*)", [](const httplib::Request&, httplib::Response& res) {
        res.status = 204;
    });

    // --- Health ---
    svr.Get("/api/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    // --- Broker: run all methods, pick best ---
    svr.Post("/api/parameterize", [](const httplib::Request& req, httplib::Response& res) {
        std::string forced_method = req.has_param("method") ? req.get_param_value("method") : "auto";
        bool view_weighted = req.has_param("viewWeighted") && req.get_param_value("viewWeighted") == "true";

        std::vector<uint8_t> input(req.body.begin(), req.body.end());

        // Determine which methods to run
        struct MethodDef { std::string name; };
        std::vector<MethodDef> methods_to_run;

        if (forced_method == "auto") {
            methods_to_run = {{"heat"}, {"cgal_conformal"}, {"cgal_arap"}, {"cgal_authalic"}};
        } else {
            methods_to_run = {{forced_method}};
        }

        // Run all methods in parallel
        std::vector<MethodResult> results(methods_to_run.size());
        std::vector<std::thread> workers;

        for (size_t i = 0; i < methods_to_run.size(); ++i) {
            workers.emplace_back([&, i]() {
                const auto& mdef = methods_to_run[i];
                if (mdef.name == "heat") {
                    results[i] = run_heat(input, view_weighted);
                } else if (mdef.name == "cgal_conformal") {
                    results[i] = run_cgal(input, cgalparam::ParamMethod::DiscreteConformal, "cgal_conformal");
                } else if (mdef.name == "cgal_arap") {
                    results[i] = run_cgal(input, cgalparam::ParamMethod::ARAP, "cgal_arap");
                } else if (mdef.name == "cgal_authalic") {
                    results[i] = run_cgal(input, cgalparam::ParamMethod::DiscreteAuthalic, "cgal_authalic");
                } else if (mdef.name == "cgal_mvc") {
                    results[i] = run_cgal(input, cgalparam::ParamMethod::MeanValue, "cgal_mvc");
                } else {
                    results[i].method = mdef.name;
                    results[i].error = "Unknown method";
                }
            });
        }
        for (auto& w : workers) w.join();

        // Pick best by score
        int best_idx = -1;
        double best_score = 1e18;
        for (size_t i = 0; i < results.size(); ++i) {
            double s = results[i].score();
            std::cout << "  [broker] " << results[i].method
                      << ": " << (results[i].success ? "OK" : "FAIL")
                      << " score=" << s
                      << " angle=" << results[i].angle_mean
                      << " stretch=" << results[i].stretch_mean
                      << " (" << results[i].elapsed_ms << " ms)" << std::endl;
            if (s < best_score) { best_score = s; best_idx = static_cast<int>(i); }
        }

        if (best_idx < 0 || !results[best_idx].success) {
            // All methods failed
            std::ostringstream err;
            err << "{\"error\":\"All methods failed\",\"methods\":[";
            for (size_t i = 0; i < results.size(); ++i) {
                if (i > 0) err << ",";
                err << results[i].to_json();
            }
            err << "]}";
            res.status = 500;
            res.set_content(err.str(), "application/json");
            return;
        }

        auto& best = results[best_idx];
        std::cout << "  [broker] Winner: " << best.method
                  << " (score=" << best.score() << ")" << std::endl;

        // Build all-methods JSON
        std::ostringstream all;
        all << "[";
        for (size_t i = 0; i < results.size(); ++i) {
            if (i > 0) all << ",";
            all << results[i].to_json();
        }
        all << "]";

        res.set_header("X-Method", best.method);
        res.set_header("X-Metrics", best.to_json());
        res.set_header("X-All-Methods", all.str());
        res.set_content(std::string(best.glb.begin(), best.glb.end()), "model/gltf-binary");
    });

    // --- Individual method endpoints (still available) ---
    svr.Post("/api/parameterize/heat", [](const httplib::Request& req, httplib::Response& res) {
        bool vw = req.has_param("viewWeighted") && req.get_param_value("viewWeighted") == "true";
        std::vector<uint8_t> input(req.body.begin(), req.body.end());
        auto r = run_heat(input, vw);
        if (!r.success) { res.status = 500; res.set_content("{\"error\":\"" + r.error + "\"}", "application/json"); return; }
        res.set_header("X-Metrics", r.to_json());
        res.set_content(std::string(r.glb.begin(), r.glb.end()), "model/gltf-binary");
    });

    svr.Post("/api/parameterize/cgal", [](const httplib::Request& req, httplib::Response& res) {
        std::string ms = req.has_param("method") ? req.get_param_value("method") : "conformal";
        cgalparam::ParamMethod m = cgalparam::ParamMethod::DiscreteConformal;
        if (ms == "arap") m = cgalparam::ParamMethod::ARAP;
        else if (ms == "authalic") m = cgalparam::ParamMethod::DiscreteAuthalic;
        else if (ms == "mvc") m = cgalparam::ParamMethod::MeanValue;
        std::vector<uint8_t> input(req.body.begin(), req.body.end());
        auto r = run_cgal(input, m, "cgal_" + ms);
        if (!r.success) { res.status = 500; res.set_content("{\"error\":\"" + r.error + "\"}", "application/json"); return; }
        res.set_header("X-Metrics", r.to_json());
        res.set_content(std::string(r.glb.begin(), r.glb.end()), "model/gltf-binary");
    });

    // --- STEP tessellation ---
    svr.Post("/api/tessellate/step", [&occ_cli](const httplib::Request& req, httplib::Response& res) {
        if (occ_cli.empty()) {
            res.status = 501; res.set_content("{\"error\":\"OCC CLI not configured\"}", "application/json"); return;
        }
        try {
            double defl = req.has_param("deflection") ? std::stod(req.get_param_value("deflection")) : 1.0;
            auto glb = tessellate_step(occ_cli, req.body, defl);
            res.set_content(std::string(glb.begin(), glb.end()), "model/gltf-binary");
        } catch (const std::exception& e) {
            res.status = 500; res.set_content("{\"error\":\"" + std::string(e.what()) + "\"}", "application/json");
        }
    });

    // --- Static files ---
    svr.set_mount_point("/", web_root);

    std::cout << "Mesh Parameterization Broker" << std::endl;
    std::cout << "  Port:     " << port << std::endl;
    std::cout << "  Threads:  " << threads << std::endl;
    std::cout << "  Web root: " << web_root << std::endl;
    std::cout << "  OCC CLI:  " << (occ_cli.empty() ? "(none)" : occ_cli) << std::endl;
    std::cout << "  Endpoints:" << std::endl;
    std::cout << "    POST /api/parameterize          (broker: auto-picks best)" << std::endl;
    std::cout << "    POST /api/parameterize?method=X  (force: heat|cgal_conformal|cgal_arap|cgal_authalic)" << std::endl;
    std::cout << "    POST /api/parameterize/heat     (direct)" << std::endl;
    std::cout << "    POST /api/parameterize/cgal     (direct)" << std::endl;
    std::cout << "    POST /api/tessellate/step" << std::endl;
    std::cout << "    GET  /api/health" << std::endl;
    std::cout << "\nListening on http://localhost:" << port << std::endl;

    svr.listen("0.0.0.0", port);
    return 0;
}
