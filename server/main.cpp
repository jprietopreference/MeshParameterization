// Mesh Parameterization Broker Server
// Runs multiple parameterization methods in parallel, picks the best result.

#include "httplib.h"

#include "meshparam/gltf_io.h"
#include "meshparam/parameterizer.h"
#include "meshparam/distortion.h"
#include <tiny_gltf.h>

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
#include <unordered_map>
#include <array>

// ============================================================
// Vertex welding: merge split vertices (same position) for parameterization
// ============================================================
std::vector<uint8_t> weld_vertices(const std::vector<uint8_t>& glb_data) {
    auto mesh = meshparam::load_gltf_from_memory(glb_data);
    int n = mesh.num_vertices();
    int m = mesh.num_faces();

    // Build position → first vertex index map (quantize to 1e-6 for welding)
    struct Vec3Hash {
        size_t operator()(const std::array<int64_t,3>& v) const {
            size_t h = 0;
            for (auto x : v) h ^= std::hash<int64_t>()(x) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };
    std::unordered_map<std::array<int64_t,3>, int, Vec3Hash> pos_map;
    std::vector<int> remap(n);
    int new_n = 0;

    for (int i = 0; i < n; ++i) {
        std::array<int64_t,3> key = {
            static_cast<int64_t>(std::round(mesh.V(i,0) * 1e6)),
            static_cast<int64_t>(std::round(mesh.V(i,1) * 1e6)),
            static_cast<int64_t>(std::round(mesh.V(i,2) * 1e6))
        };
        auto it = pos_map.find(key);
        if (it != pos_map.end()) {
            remap[i] = it->second;
        } else {
            remap[i] = new_n;
            pos_map[key] = new_n;
            new_n++;
        }
    }

    if (new_n == n) return glb_data; // already welded

    std::cout << "[weld] Welded " << n << " → " << new_n << " vertices" << std::endl;

    // Build welded mesh (positions only, no normals — those stay on the original)
    meshparam::TriMesh welded;
    welded.V.resize(new_n, 3);
    for (int i = 0; i < n; ++i) {
        welded.V.row(remap[i]) = mesh.V.row(i);
    }
    welded.F.resize(m, 3);
    for (int i = 0; i < m; ++i) {
        welded.F(i, 0) = remap[mesh.F(i, 0)];
        welded.F(i, 1) = remap[mesh.F(i, 1)];
        welded.F(i, 2) = remap[mesh.F(i, 2)];
    }

    return meshparam::save_gltf_to_memory(welded);
}

// ============================================================
// Mesh healing: remove degenerate triangles before parameterization
// ============================================================
void heal_mesh(std::vector<uint8_t>& glb_data, bool force = false) {
    auto mesh = meshparam::load_gltf_from_memory(glb_data);
    int n = mesh.num_vertices();
    int m = mesh.num_faces();

    // Two thresholds:
    //   strict (1e-10): truly degenerate — always removed
    //   loose (1e-6):   near-degenerate — removed only when force=true
    // This way auto-heal catches zero-area triangles silently,
    // and "Force mesh healing" also catches borderline cases.
    double min_area = force ? 1e-6 : 1e-10;

    std::vector<bool> keep(m, true);
    int removed = 0;

    for (int i = 0; i < m; ++i) {
        int i0 = mesh.F(i, 0), i1 = mesh.F(i, 1), i2 = mesh.F(i, 2);
        // Only remove triangles with duplicate vertex indices.
        // Zero-area collinear triangles are kept — they're topologically needed
        // to maintain the surface closure (removing them creates holes → genus change).
        if (i0 == i1 || i1 == i2 || i0 == i2) { keep[i] = false; removed++; continue; }
        if (force) {
            Eigen::Vector3d e1 = mesh.V.row(i1) - mesh.V.row(i0);
            Eigen::Vector3d e2 = mesh.V.row(i2) - mesh.V.row(i0);
            double area = e1.cross(e2).norm() * 0.5;
            if (area < min_area) { keep[i] = false; removed++; }
        }
    }

    if (removed == 0) return;

    std::cout << "[heal] Removed " << removed << " degenerate triangles (" << m << " → " << (m - removed) << ")"
              << (force ? " [forced]" : " [auto]") << std::endl;

    // Rebuild face matrix
    Eigen::MatrixXi newF(m - removed, 3);
    int j = 0;
    for (int i = 0; i < m; ++i) {
        if (keep[i]) { newF.row(j++) = mesh.F.row(i); }
    }
    mesh.F = newF;

    // Remove unreferenced vertices — preserve normals
    std::vector<bool> used(n, false);
    for (int i = 0; i < mesh.num_faces(); ++i) {
        used[mesh.F(i, 0)] = true;
        used[mesh.F(i, 1)] = true;
        used[mesh.F(i, 2)] = true;
    }
    std::vector<int> remap(n, -1);
    int new_n = 0;
    for (int i = 0; i < n; ++i) {
        if (used[i]) remap[i] = new_n++;
    }

    Eigen::MatrixXd newV(new_n, 3);
    Eigen::MatrixXd newN;
    bool has_normals = mesh.has_normals();
    if (has_normals) newN.resize(new_n, 3);

    for (int i = 0; i < n; ++i) {
        if (remap[i] >= 0) {
            newV.row(remap[i]) = mesh.V.row(i);
            if (has_normals) newN.row(remap[i]) = mesh.N.row(i);
        }
    }
    for (int i = 0; i < mesh.num_faces(); ++i) {
        mesh.F(i, 0) = remap[mesh.F(i, 0)];
        mesh.F(i, 1) = remap[mesh.F(i, 1)];
        mesh.F(i, 2) = remap[mesh.F(i, 2)];
    }
    mesh.V = newV;
    if (has_normals) mesh.N = newN;

    glb_data = meshparam::save_gltf_to_memory(mesh);
}

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

        // Check for NaN/Inf UVs
        for (int i = 0; i < result.UV.rows(); ++i) {
            if (!std::isfinite(result.UV(i, 0)) || !std::isfinite(result.UV(i, 1))) {
                r.error = "UV contains NaN";
                auto t1 = std::chrono::high_resolution_clock::now();
                r.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                return r;
            }
        }

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

MethodResult run_cgal(const std::vector<uint8_t>& input_glb, cgalparam::ParamMethod method, const std::string& method_name,
                      bool use_silhouette_seam = false,
                      const std::vector<uint8_t>& orig_glb_for_seam = {}) {
    MethodResult r;
    r.method = method_name;
    auto t0 = std::chrono::high_resolution_clock::now();
    try {
        auto tri = cgalparam::load_gltf_from_memory(input_glb);
        auto sm = cgalparam::to_cgal_mesh(tri);

        if (CGAL::is_closed(sm)) {
            if (use_silhouette_seam && !orig_glb_for_seam.empty()) {
                auto orig = meshparam::load_gltf_from_memory(orig_glb_for_seam);
                // Try to read _FACE_ID from original GLB
                Eigen::VectorXd face_ids;
                // Parse face IDs from GLB manually (not in TriMesh)
                {
                    tinygltf::Model model;
                    tinygltf::TinyGLTF loader;
                    std::string err, warn;
                    loader.LoadBinaryFromMemory(&model, &err, &warn,
                        orig_glb_for_seam.data(), static_cast<unsigned>(orig_glb_for_seam.size()));
                    if (!model.meshes.empty()) {
                        auto& prim = model.meshes[0].primitives[0];
                        auto it = prim.attributes.find("_FACE_ID");
                        if (it != prim.attributes.end()) {
                            auto& acc = model.accessors[it->second];
                            auto& bv = model.bufferViews[acc.bufferView];
                            auto& buf = model.buffers[bv.buffer];
                            const float* data = reinterpret_cast<const float*>(
                                buf.data.data() + bv.byteOffset + acc.byteOffset);
                            face_ids.resize(acc.count);
                            for (int i = 0; i < (int)acc.count; ++i) face_ids(i) = data[i];
                        }
                    }
                }
                if (face_ids.size() > 0) {
                    auto cut = cgalparam::cut_brep_silhouette(sm, orig.V, orig.N, orig.F, face_ids);
                    sm = cut.cut_mesh;
                } else {
                    std::cout << "[broker] No _FACE_ID in input, using BFS seam" << std::endl;
                    auto cut = cgalparam::cut_to_disk(sm);
                    sm = cut.cut_mesh;
                }
            } else {
                auto cut = cgalparam::cut_to_disk(sm);
                sm = cut.cut_mesh;
            }
        }

        Eigen::MatrixXd UV = cgalparam::parameterize(sm, method);
        auto result = cgalparam::from_cgal_mesh(sm);
        result.UV = UV;

        // Check for NaN/Inf UVs (CGAL solvers may silently diverge)
        for (int i = 0; i < result.UV.rows(); ++i) {
            if (!std::isfinite(result.UV(i, 0)) || !std::isfinite(result.UV(i, 1))) {
                r.error = "UV contains NaN (solver diverged)";
                auto t1 = std::chrono::high_resolution_clock::now();
                r.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                return r;
            }
        }

        auto metrics = cgalparam::compute_distortion(result.V, result.F, result.UV);

        // Convert to meshparam::TriMesh to get normals in the output
        meshparam::TriMesh out;
        out.V = result.V; out.F = result.F; out.UV = result.UV;
        out.compute_normals();
        r.glb = meshparam::save_gltf_to_memory(out);
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

        // Always auto-detect and heal degenerate triangles.
        // The "heal" flag forces healing even for near-degenerate cases (looser threshold).
        // Save original input (with OCC split-vertex normals) for seam analysis
        std::vector<uint8_t> input_original = input;

        bool force_heal = req.has_param("heal") && req.get_param_value("heal") == "true";
        heal_mesh(input, force_heal);

        // Weld split vertices for parameterization (OCC splits at face boundaries
        // for correct normals, but parameterization needs shared connectivity).
        // We keep the healed split mesh to re-apply normals to the output.
        std::vector<uint8_t> input_for_param = weld_vertices(input);

        // Heal again after welding — welding can create new degenerate triangles
        // when two split vertices at the same position get merged
        heal_mesh(input_for_param, force_heal);

        // Determine which methods to run
        struct MethodDef { std::string name; };
        std::vector<MethodDef> methods_to_run;

        if (forced_method == "auto") {
            methods_to_run = {{"heat"}, {"cgal_conformal"}, {"cgal_arap"}, {"cgal_authalic"}};
        } else {
            methods_to_run = {{forced_method}};
        }

        // Run all methods in parallel on the WELDED mesh
        std::vector<MethodResult> results(methods_to_run.size());
        std::vector<std::thread> workers;

        for (size_t i = 0; i < methods_to_run.size(); ++i) {
            workers.emplace_back([&, i]() {
                const auto& mdef = methods_to_run[i];
                if (mdef.name == "heat") {
                    results[i] = run_heat(input_for_param, view_weighted);
                } else if (mdef.name == "cgal_conformal") {
                    results[i] = run_cgal(input_for_param, cgalparam::ParamMethod::DiscreteConformal, "cgal_conformal", view_weighted, input_original);
                } else if (mdef.name == "cgal_arap") {
                    results[i] = run_cgal(input_for_param, cgalparam::ParamMethod::ARAP, "cgal_arap", view_weighted, input_original);
                } else if (mdef.name == "cgal_authalic") {
                    results[i] = run_cgal(input_for_param, cgalparam::ParamMethod::DiscreteAuthalic, "cgal_authalic", view_weighted, input_original);
                } else if (mdef.name == "cgal_mvc") {
                    results[i] = run_cgal(input_for_param, cgalparam::ParamMethod::MeanValue, "cgal_mvc", view_weighted, input_original);
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

        // Re-apply original normals from the input mesh to the parameterized output.
        // The parameterization ran on welded vertices; now map UVs back to the
        // original split-vertex mesh so normals from OCC surfaces are preserved.
        {
            auto orig = meshparam::load_gltf_from_memory(input); // original with normals
            auto param = meshparam::load_gltf_from_memory(best.glb); // welded with UVs

            if (orig.has_normals() && param.has_uvs() && orig.num_vertices() != param.num_vertices()) {
                // Build position → UV map from welded parameterized mesh
                struct Vec3Hash {
                    size_t operator()(const std::array<int64_t,3>& v) const {
                        size_t h = 0;
                        for (auto x : v) h ^= std::hash<int64_t>()(x) + 0x9e3779b9 + (h << 6) + (h >> 2);
                        return h;
                    }
                };
                std::unordered_map<std::array<int64_t,3>, int, Vec3Hash> pos_to_welded;
                for (int i = 0; i < param.num_vertices(); ++i) {
                    std::array<int64_t,3> key = {
                        static_cast<int64_t>(std::round(param.V(i,0) * 1e6)),
                        static_cast<int64_t>(std::round(param.V(i,1) * 1e6)),
                        static_cast<int64_t>(std::round(param.V(i,2) * 1e6))
                    };
                    pos_to_welded[key] = i;
                }

                // Map UVs from welded → original split vertices
                orig.UV.resize(orig.num_vertices(), 2);
                int mapped = 0;
                for (int i = 0; i < orig.num_vertices(); ++i) {
                    std::array<int64_t,3> key = {
                        static_cast<int64_t>(std::round(orig.V(i,0) * 1e6)),
                        static_cast<int64_t>(std::round(orig.V(i,1) * 1e6)),
                        static_cast<int64_t>(std::round(orig.V(i,2) * 1e6))
                    };
                    auto it = pos_to_welded.find(key);
                    if (it != pos_to_welded.end()) {
                        orig.UV.row(i) = param.UV.row(it->second);
                        mapped++;
                    }
                }
                std::cout << "  [broker] Mapped UVs to " << mapped << "/" << orig.num_vertices()
                          << " split vertices (normals preserved)" << std::endl;
                best.glb = meshparam::save_gltf_to_memory(orig);
                best.vertices = orig.num_vertices();
                best.faces = orig.num_faces();
            }
        }

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
