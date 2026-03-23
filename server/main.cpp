// Mesh Parameterization Broker Server
// Runs multiple parameterization methods in parallel, picks the best result.

#include "httplib.h"

#include "meshparam/gltf_io.h"
#include "meshparam/parameterizer.h"
#include "meshparam/distortion.h"
#include <tiny_gltf.h>

// libigl parameterization methods
#include <igl/slim.h>
#include <igl/lscm.h>
#include <igl/arap.h>
#include <igl/boundary_loop.h>
#include <igl/harmonic.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/flipped_triangles.h>
#include "meshparam/benchmark_metrics.h"

// Stein ADMM splitting (flip-free parametrization)
// Include .cpp files for template instantiation (header-only style)
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
#include <random>

#ifdef _WIN32
#include <windows.h>
#endif

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
struct HealStats {
    int removed = 0;
    int perturbed = 0;
    bool forced = false;
    std::string to_json() const {
        std::ostringstream o;
        o << "{\"removed\":" << removed << ",\"perturbed\":" << perturbed
          << ",\"forced\":" << (forced ? "true" : "false") << "}";
        return o.str();
    }
};

HealStats heal_mesh(std::vector<uint8_t>& glb_data, bool force = false) {
    HealStats stats;
    stats.forced = force;
    auto mesh = meshparam::load_gltf_from_memory(glb_data);
    int n = mesh.num_vertices();
    int m = mesh.num_faces();

    // Area threshold for collinear detection and perturbation.
    // Triangles below this threshold get their middle vertex perturbed.
    // Must be generous enough to catch near-collinear triangles that
    // produce NaN cotangent weights.
    double min_area = force ? 1e-4 : 1e-8;

    std::vector<bool> keep(m, true);
    int removed = 0;

    int perturbed = 0;
    for (int i = 0; i < m; ++i) {
        int i0 = mesh.F(i, 0), i1 = mesh.F(i, 1), i2 = mesh.F(i, 2);
        // Remove triangles with duplicate vertex indices
        if (i0 == i1 || i1 == i2 || i0 == i2) { keep[i] = false; removed++; continue; }
        // Check for collinear/zero-area triangles
        Eigen::Vector3d e1 = mesh.V.row(i1) - mesh.V.row(i0);
        Eigen::Vector3d e2 = mesh.V.row(i2) - mesh.V.row(i0);
        Eigen::Vector3d cross = e1.cross(e2);
        double area = cross.norm() * 0.5;
        if (area < min_area) {
            // Don't remove — perturb the middle vertex to create non-zero area.
            // This preserves topology while fixing the degenerate cotangent weights.
            Eigen::Vector3d edge = (mesh.V.row(i2) - mesh.V.row(i0)).normalized();
            // Find a perpendicular direction
            Eigen::Vector3d perp;
            if (std::abs(edge.x()) < 0.9) perp = edge.cross(Eigen::Vector3d(1,0,0));
            else perp = edge.cross(Eigen::Vector3d(0,1,0));
            perp.normalize();
            // Perturb middle vertex (i1) by a tiny epsilon perpendicular to the edge
            double epsilon = 1e-6;
            mesh.V.row(i1) = mesh.V.row(i1) + epsilon * perp.transpose();
            perturbed++;
        }
    }

    stats.removed = removed;
    stats.perturbed = perturbed;
    if (removed == 0 && perturbed == 0) return stats;

    if (perturbed > 0)
        std::cout << "[heal] Perturbed " << perturbed << " collinear triangles" << std::endl;
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
    return stats;
}

// ============================================================
// CGAL isotropic remeshing via external CLI
// ============================================================
std::vector<uint8_t> remesh_isotropic(const std::string& remesh_cli, const std::vector<uint8_t>& glb_data) {
    auto tid = std::this_thread::get_id();
    std::ostringstream ss;
    const char* tmp_dir = std::getenv("TEMP");
    if (!tmp_dir) tmp_dir = std::getenv("TMP");
    if (!tmp_dir) tmp_dir = ".";
    ss << tmp_dir << "/meshparam_remesh_" << tid;
    std::string tmp_in = ss.str() + "_in.glb";
    std::string tmp_out = ss.str() + "_out.glb";

    { std::ofstream f(tmp_in, std::ios::binary);
      f.write(reinterpret_cast<const char*>(glb_data.data()), glb_data.size()); }

    std::string cmd = remesh_cli + " \"" + tmp_in + "\" \"" + tmp_out + "\"";
    int ret = std::system(cmd.c_str());
    std::remove(tmp_in.c_str());

    if (ret != 0) {
        std::remove(tmp_out.c_str());
        throw std::runtime_error("CGAL remeshing failed");
    }

    std::ifstream f(tmp_out, std::ios::binary | std::ios::ate);
    size_t sz = f.tellg(); f.seekg(0);
    std::vector<uint8_t> result(sz);
    f.read(reinterpret_cast<char*>(result.data()), sz);
    f.close();
    std::remove(tmp_out.c_str());
    return result;
}

// ============================================================
// ACIS tessellation via external CLI
// ============================================================
std::vector<uint8_t> tessellate_step_acis(const std::string& acis_cli, const std::string& step_data, double deflection) {
    auto tid = std::this_thread::get_id();
    std::ostringstream ss;
    const char* tmp_dir = std::getenv("TEMP");
    if (!tmp_dir) tmp_dir = std::getenv("TMP");
    if (!tmp_dir) tmp_dir = ".";
    ss << tmp_dir << "/meshparam_acis_" << tid;
    std::string tmp_in = ss.str() + ".step";
    std::string tmp_out = ss.str() + ".glb";

    { std::ofstream f(tmp_in, std::ios::binary);
      f.write(step_data.data(), step_data.size()); }

    std::string cmd = acis_cli + " \"" + tmp_in + "\" \"" + tmp_out + "\" --deflection " + std::to_string(deflection);
    int ret = std::system(cmd.c_str());
    std::remove(tmp_in.c_str());

    if (ret != 0) {
        std::remove(tmp_out.c_str());
        throw std::runtime_error("ACIS tessellation failed");
    }

    std::ifstream f(tmp_out, std::ios::binary | std::ios::ate);
    size_t sz = f.tellg(); f.seekg(0);
    std::vector<uint8_t> result(sz);
    f.read(reinterpret_cast<char*>(result.data()), sz);
    f.close();
    std::remove(tmp_out.c_str());
    return result;
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
    // Benchmark metrics (Stein et al. 2022)
    int flipped_tris = -1;
    double sym_dirichlet = -1;
    double l2_area = -1;
    double linf_area = -1;
    int vertices = 0;
    int faces = 0;
    std::vector<uint8_t> glb;

    // Composite quality score (lower = better)
    // Updated to use Symmetric Dirichlet + flipped triangles penalty
    double score() const {
        if (!success) return 1e18;
        // Primary: Symmetric Dirichlet energy (lower = better, 4.0 = perfect isometry)
        // Penalty: flipped triangles (1e6 per flip)
        double sd = (sym_dirichlet > 0) ? sym_dirichlet : (angle_mean * 0.4 + std::log1p(stretch_mean) * 10.0 * 0.4 + area_std * 0.2);
        double flip_penalty = (flipped_tris > 0) ? flipped_tris * 1e6 : 0;
        return sd + flip_penalty;
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
              << ",\"flipped_tris\":" << flipped_tris
              << ",\"sym_dirichlet\":" << sym_dirichlet
              << ",\"l2_area\":" << l2_area
              << ",\"linf_area\":" << linf_area
              << ",\"score\":" << score();
        }
        o << "}";
        return o.str();
    }
};

// ============================================================
// Run individual methods
// ============================================================
void fill_benchmark_metrics(MethodResult& r, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXd& UV) {
    auto bm = meshparam::compute_benchmark_metrics(V, F, UV);
    r.flipped_tris = bm.flipped_triangles;
    r.sym_dirichlet = bm.symmetric_dirichlet;
    r.l2_area = bm.l2_area_distortion;
    r.linf_area = bm.linf_area_distortion;
}

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
        fill_benchmark_metrics(r, result.V, result.F, result.UV);
    } catch (const std::exception& e) {
        r.error = e.what();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    r.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return r;
}

// Cut a closed mesh to disk topology using CGAL's BFS seam,
// returning GLB with boundary. Used by SLIM/igl_arap which need boundary.
std::vector<uint8_t> cut_closed_mesh(const std::vector<uint8_t>& input_glb) {
    auto tri = cgalparam::load_gltf_from_memory(input_glb);
    auto sm = cgalparam::to_cgal_mesh(tri);
    if (!CGAL::is_closed(sm)) return input_glb; // already has boundary
    auto cut = cgalparam::cut_to_disk(sm);
    auto cut_tri = cgalparam::from_cgal_mesh(cut.cut_mesh);
    return cgalparam::save_gltf_to_memory(cut_tri);
}

MethodResult run_lscm(const std::vector<uint8_t>& input_glb) {
    MethodResult r;
    r.method = "lscm";
    auto t0 = std::chrono::high_resolution_clock::now();
    try {
        auto mesh = meshparam::load_gltf_from_memory(input_glb);

        // LSCM needs at least 2 pinned vertices
        Eigen::VectorXi bnd;
        igl::boundary_loop(mesh.F, bnd);

        Eigen::VectorXi b(2);
        Eigen::MatrixXd bc(2, 2);
        if (bnd.size() >= 2) {
            b(0) = bnd(0);
            b(1) = bnd(bnd.size() / 2);
        } else {
            b(0) = 0; b(1) = 1;
        }
        bc << 0, 0, 1, 0; // pin 2 vertices

        Eigen::MatrixXd UV;
        igl::lscm(mesh.V, mesh.F, b, bc, UV);

        // Check for NaN
        for (int i = 0; i < UV.rows(); ++i) {
            if (!std::isfinite(UV(i,0)) || !std::isfinite(UV(i,1))) {
                r.error = "LSCM produced NaN UVs";
                auto t1 = std::chrono::high_resolution_clock::now();
                r.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                return r;
            }
        }

        // Normalize UV
        Eigen::Vector2d mn = UV.colwise().minCoeff();
        Eigen::Vector2d mx = UV.colwise().maxCoeff();
        for (int i = 0; i < 2; ++i) {
            double range = mx(i) - mn(i);
            if (range > 1e-12) UV.col(i) = (UV.col(i).array() - mn(i)) / range;
        }

        mesh.UV = UV;
        mesh.compute_normals();

        auto metrics = meshparam::compute_distortion(mesh.V, mesh.F, mesh.UV);
        r.glb = meshparam::save_gltf_to_memory(mesh);
        r.success = true;
        r.vertices = mesh.num_vertices();
        r.faces = mesh.num_faces();
        r.angle_mean = metrics.mean_angle_distortion;
        r.angle_max = metrics.max_angle_distortion;
        r.area_mean = metrics.mean_area_distortion;
        r.area_std = metrics.std_area_distortion;
        r.stretch_mean = metrics.mean_stretch;
        r.stretch_max = metrics.max_stretch;
        fill_benchmark_metrics(r, mesh.V, mesh.F, mesh.UV);
    } catch (const std::exception& e) {
        r.error = e.what();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    r.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return r;
}

MethodResult run_igl_arap(const std::vector<uint8_t>& input_glb) {
    MethodResult r;
    r.method = "igl_arap";
    auto t0 = std::chrono::high_resolution_clock::now();
    try {
        // Cut closed meshes to create boundary
        auto glb_with_bnd = cut_closed_mesh(input_glb);
        auto mesh = meshparam::load_gltf_from_memory(glb_with_bnd);
        Eigen::VectorXi bnd;
        igl::boundary_loop(mesh.F, bnd);
        if (bnd.size() == 0) { r.error = "needs boundary (cut failed)"; goto end_arap; }
        {
            // Init from LSCM
            Eigen::VectorXi b2(2); Eigen::MatrixXd bc2(2,2);
            b2(0)=bnd(0); b2(1)=bnd(bnd.size()/2); bc2<<0,0,1,0;
            Eigen::MatrixXd UV;
            igl::lscm(mesh.V, mesh.F, b2, bc2, UV);
            for(int i=0;i<UV.rows();++i) if(!std::isfinite(UV(i,0))||!std::isfinite(UV(i,1))){r.error="LSCM init NaN";goto end_arap;}

            Eigen::MatrixXd bc(bnd.size(),2);
            for(int i=0;i<bnd.size();++i) bc.row(i)=UV.row(bnd(i));

            igl::ARAPData ad; ad.max_iter=100;
            arap_precomputation(mesh.V,mesh.F,2,bnd,ad);
            arap_solve(bc,ad,UV);

            for(int i=0;i<UV.rows();++i) if(!std::isfinite(UV(i,0))||!std::isfinite(UV(i,1))){r.error="ARAP NaN";goto end_arap;}

            Eigen::Vector2d mn=UV.colwise().minCoeff(), mx=UV.colwise().maxCoeff();
            for(int i=0;i<2;++i){double rng=mx(i)-mn(i);if(rng>1e-12)UV.col(i)=(UV.col(i).array()-mn(i))/rng;}

            mesh.UV=UV; mesh.compute_normals();
            auto metrics=meshparam::compute_distortion(mesh.V,mesh.F,mesh.UV);
            r.glb=meshparam::save_gltf_to_memory(mesh);
            r.success=true; r.vertices=mesh.num_vertices(); r.faces=mesh.num_faces();
            r.angle_mean=metrics.mean_angle_distortion; r.angle_max=metrics.max_angle_distortion;
            r.area_mean=metrics.mean_area_distortion; r.area_std=metrics.std_area_distortion;
            r.stretch_mean=metrics.mean_stretch; r.stretch_max=metrics.max_stretch;
            fill_benchmark_metrics(r,mesh.V,mesh.F,mesh.UV);
        }
    } catch(const std::exception& e) { r.error=e.what(); }
    end_arap:
    { auto t1=std::chrono::high_resolution_clock::now(); r.elapsed_ms=std::chrono::duration<double,std::milli>(t1-t0).count(); }
    return r;
}

MethodResult run_stein(const std::vector<uint8_t>& input_glb) {
    MethodResult r;
    r.method = "stein_admm";
    auto t0 = std::chrono::high_resolution_clock::now();
    try {
        auto mesh = meshparam::load_gltf_from_memory(input_glb);
        Eigen::VectorXi bnd;
        igl::boundary_loop(mesh.F, bnd);
        if (bnd.size() == 0) { r.error = "needs boundary"; goto end_stein; }
        {
            // Init from Tutte (bijective) or LSCM fallback
            Eigen::MatrixXd W;
            try {
                parametrization::tutte<false>(mesh.V, mesh.F, W);
            } catch (...) {
                W.resize(0, 0);
            }

            if (W.rows() == 0 || W.rows() != mesh.V.rows()) {
                // Fallback to LSCM
                Eigen::VectorXi b2(2); Eigen::MatrixXd bc2(2,2);
                b2(0)=bnd(0); b2(1)=bnd(bnd.size()/2); bc2<<0,0,1,0;
                igl::lscm(mesh.V, mesh.F, b2, bc2, W);
                if (W.rows() == 0) { r.error = "init failed"; goto end_stein; }
            }

            // Check initial UV for flipped triangles
            Eigen::VectorXi flipped = igl::flipped_triangles(W, mesh.F);
            if (flipped.size() > 0) {
                // Try Tutte specifically (guaranteed flip-free)
                try {
                    parametrization::tutte<false>(mesh.V, mesh.F, W);
                    flipped = igl::flipped_triangles(W, mesh.F);
                } catch (...) {}
                if (flipped.size() > 0) {
                    r.error = "Cannot produce flip-free init";
                    goto end_stein;
                }
            }

            // Check for degenerate triangles in UV space
            {
                Eigen::MatrixXd UV3(W.rows(), 3);
                UV3.col(0) = W.col(0); UV3.col(1) = W.col(1); UV3.col(2).setZero();
                Eigen::VectorXd areas;
                igl::doublearea(UV3, mesh.F, areas);
                if (areas.minCoeff() <= 0) {
                    r.error = "Degenerate UV triangles in init";
                    goto end_stein;
                }
            }

            // Run Stein ADMM optimization (Symmetric Dirichlet energy)
            {
                parametrization::OptimizationOptions<double> opts;
                opts.maxIter = 200;
                bool success = parametrization::map_to<false, parametrization::EnergyType::SymmetricDirichlet>(
                    mesh.V, mesh.F, W, opts);

                if (!success) { r.error = "Stein ADMM optimization failed"; goto end_stein; }
            }

            for(int i=0;i<W.rows();++i) if(!std::isfinite(W(i,0))||!std::isfinite(W(i,1))){r.error="NaN UVs";goto end_stein;}

            // Normalize
            Eigen::Vector2d mn=W.colwise().minCoeff(), mx=W.colwise().maxCoeff();
            for(int i=0;i<2;++i){double rng=mx(i)-mn(i);if(rng>1e-12)W.col(i)=(W.col(i).array()-mn(i))/rng;}

            mesh.UV=W; mesh.compute_normals();
            auto metrics=meshparam::compute_distortion(mesh.V,mesh.F,mesh.UV);
            r.glb=meshparam::save_gltf_to_memory(mesh);
            r.success=true; r.vertices=mesh.num_vertices(); r.faces=mesh.num_faces();
            r.angle_mean=metrics.mean_angle_distortion; r.angle_max=metrics.max_angle_distortion;
            r.area_mean=metrics.mean_area_distortion; r.area_std=metrics.std_area_distortion;
            r.stretch_mean=metrics.mean_stretch; r.stretch_max=metrics.max_stretch;
            fill_benchmark_metrics(r,mesh.V,mesh.F,mesh.UV);
        }
    } catch(const std::exception& e) { r.error=e.what(); }
    end_stein:
    { auto t1=std::chrono::high_resolution_clock::now(); r.elapsed_ms=std::chrono::duration<double,std::milli>(t1-t0).count(); }
    return r;
}

MethodResult run_slim(const std::vector<uint8_t>& input_glb, int iterations = 50) {
    MethodResult r;
    r.method = "slim";
    auto t0 = std::chrono::high_resolution_clock::now();
    try {
        // Cut closed meshes to create boundary
        auto glb_with_bnd = cut_closed_mesh(input_glb);
        auto mesh = meshparam::load_gltf_from_memory(glb_with_bnd);
        int nv = mesh.num_vertices(), nf = mesh.num_faces();

        Eigen::VectorXi bnd;
        igl::boundary_loop(mesh.F, bnd);

        if (bnd.size() == 0) {
            r.error = "SLIM: mesh still closed after cut attempt";
            auto t1 = std::chrono::high_resolution_clock::now();
            r.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            return r;
        }

        // Initial UV: try LSCM first (better starting point), fall back to Tutte
        Eigen::MatrixXd V_init;
        {
            Eigen::VectorXi b_lscm(2);
            Eigen::MatrixXd bc_lscm(2, 2);
            b_lscm(0) = bnd(0);
            b_lscm(1) = bnd(bnd.size() / 2);
            bc_lscm << 0, 0, 1, 0;
            igl::lscm(mesh.V, mesh.F, b_lscm, bc_lscm, V_init);

            bool has_nan = false;
            for (int i = 0; i < V_init.rows() && !has_nan; ++i)
                if (!std::isfinite(V_init(i,0)) || !std::isfinite(V_init(i,1))) has_nan = true;

            if (has_nan || V_init.rows() == 0) {
                // Fall back to Tutte
                Eigen::MatrixXd bnd_uv;
                igl::map_vertices_to_circle(mesh.V, bnd, bnd_uv);
                igl::harmonic(mesh.V, mesh.F, bnd, bnd_uv, 1, V_init);
            }
        }

        // Final NaN check
        for (int i = 0; i < V_init.rows(); ++i) {
            if (!std::isfinite(V_init(i, 0)) || !std::isfinite(V_init(i, 1))) {
                r.error = "SLIM: initial UV contains NaN";
                auto t1 = std::chrono::high_resolution_clock::now();
                r.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                return r;
            }
        }

        // SLIM optimization
        igl::SLIMData slim_data;
        Eigen::VectorXi b; // no fixed vertices (soft constraints only)
        Eigen::MatrixXd bc;
        slim_precompute(mesh.V, mesh.F, V_init, slim_data,
                        igl::SYMMETRIC_DIRICHLET, b, bc, 0);

        Eigen::MatrixXd UV = slim_solve(slim_data, iterations);

        // Check for NaN
        for (int i = 0; i < UV.rows(); ++i) {
            if (!std::isfinite(UV(i, 0)) || !std::isfinite(UV(i, 1))) {
                r.error = "SLIM produced NaN UVs";
                auto t1 = std::chrono::high_resolution_clock::now();
                r.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                return r;
            }
        }

        // Normalize UV to [0,1]
        Eigen::Vector2d uv_min = UV.colwise().minCoeff();
        Eigen::Vector2d uv_max = UV.colwise().maxCoeff();
        Eigen::Vector2d uv_range = uv_max - uv_min;
        for (int i = 0; i < 2; ++i) {
            if (uv_range(i) > 1e-12)
                UV.col(i) = (UV.col(i).array() - uv_min(i)) / uv_range(i);
        }

        mesh.UV = UV;
        mesh.compute_normals();

        auto metrics = meshparam::compute_distortion(mesh.V, mesh.F, mesh.UV);

        // Tag seam vertices from UV discontinuity
        {
            mesh.seam = Eigen::VectorXd::Zero(nv);
            std::unordered_map<int64_t, std::vector<int>> pos_groups;
            for (int i = 0; i < nv; ++i) {
                int64_t key = (int64_t(std::round(mesh.V(i,0)*1e4)) * 100000007LL +
                               int64_t(std::round(mesh.V(i,1)*1e4))) * 100000007LL +
                               int64_t(std::round(mesh.V(i,2)*1e4));
                pos_groups[key].push_back(i);
            }
            for (auto& [key, group] : pos_groups) {
                if (group.size() < 2) continue;
                for (size_t a = 0; a < group.size(); ++a) {
                    for (size_t b = a+1; b < group.size(); ++b) {
                        double du = std::abs(mesh.UV(group[a],0) - mesh.UV(group[b],0));
                        double dv = std::abs(mesh.UV(group[a],1) - mesh.UV(group[b],1));
                        if (du > 0.001 || dv > 0.001) {
                            for (int idx : group) mesh.seam(idx) = 1.0;
                            goto next_slim_group;
                        }
                    }
                }
                next_slim_group:;
            }
        }

        r.glb = meshparam::save_gltf_to_memory(mesh);
        r.success = true;
        r.vertices = nv;
        r.faces = nf;
        r.angle_mean = metrics.mean_angle_distortion;
        r.angle_max = metrics.max_angle_distortion;
        r.area_mean = metrics.mean_area_distortion;
        r.area_std = metrics.std_area_distortion;
        r.stretch_mean = metrics.mean_stretch;
        r.stretch_max = metrics.max_stretch;
        r.iso_rms = metrics.isometric_rms;
        fill_benchmark_metrics(r, mesh.V, mesh.F, mesh.UV);
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

        if (use_silhouette_seam && !orig_glb_for_seam.empty()) {
            // Always apply silhouette seam when view-weighted, even if mesh has
            // small boundary holes from healing. The silhouette seam ensures the
            // front-facing side gets a continuous UV patch.
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
                    try {
                        auto cut = cgalparam::cut_brep_silhouette(sm, orig.V, orig.N, orig.F, face_ids);
                        sm = cut.cut_mesh;
                    } catch (const std::exception& e) {
                        std::cerr << "[broker] Silhouette cut exception: " << e.what() << ", fallback to BFS" << std::endl;
                        if (CGAL::is_closed(sm)) {
                            auto cut = cgalparam::cut_to_disk(sm);
                            sm = cut.cut_mesh;
                        }
                    }
                } else {
                    std::cerr << "[broker] No _FACE_ID in input, using BFS seam" << std::endl;
                    if (CGAL::is_closed(sm)) {
                        auto cut = cgalparam::cut_to_disk(sm);
                        sm = cut.cut_mesh;
                    }
                }
        } else if (CGAL::is_closed(sm)) {
            auto cut = cgalparam::cut_to_disk(sm);
            sm = cut.cut_mesh;
        }

        // Validate mesh before parameterization
        if (!sm.is_valid(false)) {
            r.error = "Invalid mesh after seam cut";
            auto t1 = std::chrono::high_resolution_clock::now();
            r.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            return r;
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

        // Convert to meshparam::TriMesh to get normals + seam tags
        meshparam::TriMesh out;
        out.V = result.V; out.F = result.F; out.UV = result.UV;
        out.compute_normals();

        // Tag seam vertices by detecting UV discontinuity at shared positions.
        // Two vertices at the same 3D position but different UVs = seam.
        {
            int nv = out.num_vertices();
            out.seam = Eigen::VectorXd::Zero(nv);
            std::unordered_map<int64_t, std::vector<int>> pos_groups;
            for (int i = 0; i < nv; ++i) {
                int64_t key = (int64_t(std::round(out.V(i,0)*1e4)) * 100000007LL +
                               int64_t(std::round(out.V(i,1)*1e4))) * 100000007LL +
                               int64_t(std::round(out.V(i,2)*1e4));
                pos_groups[key].push_back(i);
            }
            for (auto& [key, group] : pos_groups) {
                if (group.size() < 2) continue;
                // Check if any pair has different UVs
                for (size_t a = 0; a < group.size(); ++a) {
                    for (size_t b = a+1; b < group.size(); ++b) {
                        double du = std::abs(out.UV(group[a],0) - out.UV(group[b],0));
                        double dv = std::abs(out.UV(group[a],1) - out.UV(group[b],1));
                        if (du > 0.001 || dv > 0.001) {
                            for (int idx : group) out.seam(idx) = 1.0;
                            goto next_group;
                        }
                    }
                }
                next_group:;
            }
        }
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
        fill_benchmark_metrics(r, out.V, out.F, out.UV);
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
// Gmsh tessellation (OCC import + Gmsh isotropic mesh + OCC normals)
// ============================================================
std::vector<uint8_t> tessellate_step_gmsh(const std::string& gmsh_cli, const std::string& step_data, double deflection) {
    auto tid = std::this_thread::get_id();
    std::ostringstream ss;
    const char* tmp_dir = std::getenv("TEMP");
    if (!tmp_dir) tmp_dir = std::getenv("TMP");
    if (!tmp_dir) tmp_dir = ".";
    ss << tmp_dir << "/meshparam_gmsh_" << tid;
    std::string tmp_step = ss.str() + ".step";
    std::string tmp_glb = ss.str() + ".glb";

    { std::ofstream f(tmp_step, std::ios::binary); f.write(step_data.data(), step_data.size()); }

    // gmsh_cli is "python scripts/occ_gmsh_pipeline.py"
    std::string cmd = gmsh_cli + " \"" + tmp_step + "\" \"" + tmp_glb + "\"";
    int ret = std::system(cmd.c_str());
    std::remove(tmp_step.c_str());

    if (ret != 0) {
        std::remove(tmp_glb.c_str());
        throw std::runtime_error("Gmsh tessellation failed");
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
// ============================================================
// Draco compression (optional, compile with -DHAS_DRACO=1)
// ============================================================
#ifdef HAS_DRACO
#include <draco/compression/encode.h>
#include <draco/mesh/mesh.h>
#include <draco/core/encoder_buffer.h>

std::vector<uint8_t> draco_compress_glb(const std::vector<uint8_t>& glb_data) {
    auto mesh = meshparam::load_gltf_from_memory(glb_data);
    int nv = mesh.num_vertices(), nf = mesh.num_faces();
    if (nv == 0) return glb_data;
    std::cerr << "[draco] Input: " << nv << "v " << nf << "f"
              << " seam=" << mesh.has_seam()
              << " fid=" << mesh.has_face_ids()
              << " uv=" << mesh.has_uvs()
              << " nrm=" << mesh.has_normals() << std::endl;

    // Build Draco mesh
    draco::Mesh draco_mesh;
    draco_mesh.SetNumFaces(nf);

    // Add position attribute
    draco::GeometryAttribute pos_att;
    pos_att.Init(draco::GeometryAttribute::POSITION, nullptr, 3, draco::DT_FLOAT32, false, sizeof(float) * 3, 0);
    int pos_att_id = draco_mesh.AddAttribute(pos_att, true, nv);
    for (int i = 0; i < nv; ++i) {
        float v[3] = {(float)mesh.V(i,0), (float)mesh.V(i,1), (float)mesh.V(i,2)};
        draco_mesh.attribute(pos_att_id)->SetAttributeValue(draco::AttributeValueIndex(i), v);
    }

    // Add normal attribute if available
    int nrm_att_id = -1;
    if (mesh.has_normals()) {
        draco::GeometryAttribute nrm_att;
        nrm_att.Init(draco::GeometryAttribute::NORMAL, nullptr, 3, draco::DT_FLOAT32, false, sizeof(float) * 3, 0);
        nrm_att_id = draco_mesh.AddAttribute(nrm_att, true, nv);
        for (int i = 0; i < nv; ++i) {
            float n[3] = {(float)mesh.N(i,0), (float)mesh.N(i,1), (float)mesh.N(i,2)};
            draco_mesh.attribute(nrm_att_id)->SetAttributeValue(draco::AttributeValueIndex(i), n);
        }
    }

    // Add UV attribute if available
    int uv_att_id = -1;
    if (mesh.has_uvs()) {
        draco::GeometryAttribute uv_att;
        uv_att.Init(draco::GeometryAttribute::TEX_COORD, nullptr, 2, draco::DT_FLOAT32, false, sizeof(float) * 2, 0);
        uv_att_id = draco_mesh.AddAttribute(uv_att, true, nv);
        for (int i = 0; i < nv; ++i) {
            float uv[2] = {(float)mesh.UV(i,0), (float)mesh.UV(i,1)};
            draco_mesh.attribute(uv_att_id)->SetAttributeValue(draco::AttributeValueIndex(i), uv);
        }
    }

    // Custom attributes (_SEAM, _FACE_ID) are NOT Draco-encoded.
    // BabylonJS Draco decoder doesn't expose GENERIC attributes.
    // They'll be added as separate uncompressed bufferViews in the GLB.

    // Set faces
    for (int i = 0; i < nf; ++i) {
        draco::Mesh::Face face;
        face[0] = draco::PointIndex(mesh.F(i,0));
        face[1] = draco::PointIndex(mesh.F(i,1));
        face[2] = draco::PointIndex(mesh.F(i,2));
        draco_mesh.SetFace(draco::FaceIndex(i), face);
    }

    // Encode
    draco::Encoder encoder;
    encoder.SetSpeedOptions(5, 5); // balanced speed/compression
    encoder.SetAttributeQuantization(draco::GeometryAttribute::POSITION, 14);
    encoder.SetAttributeQuantization(draco::GeometryAttribute::NORMAL, 10);
    encoder.SetAttributeQuantization(draco::GeometryAttribute::TEX_COORD, 12);

    draco::EncoderBuffer buffer;
    auto status = encoder.EncodeMeshToBuffer(draco_mesh, &buffer);
    if (!status.ok()) {
        std::cerr << "[draco] Encoding failed: " << status.error_msg_string() << std::endl;
        return glb_data; // fallback to uncompressed
    }

    size_t orig_size = glb_data.size();
    size_t draco_size = buffer.size();
    std::cout << "[draco] Compressed: " << orig_size << " -> " << draco_size
              << " bytes (" << (100 - draco_size * 100 / orig_size) << "% reduction)" << std::endl;

    // Build GLB with KHR_draco_mesh_compression
    // The Draco buffer replaces the original mesh data
    std::string json_str;
    {
        std::ostringstream js;
        // Build custom attribute buffers (uncompressed, after Draco buffer)
        std::vector<uint8_t> custom_buf;
        size_t seam_bv_offset = 0, seam_bv_size = 0;
        size_t fid_bv_offset = 0, fid_bv_size = 0;
        bool has_seam = mesh.has_seam();
        bool has_fid = mesh.has_face_ids();

        if (has_seam) {
            seam_bv_offset = custom_buf.size();
            seam_bv_size = nv * sizeof(float);
            custom_buf.resize(seam_bv_offset + seam_bv_size);
            for (int i = 0; i < nv; ++i) {
                float s = (float)mesh.seam(i);
                memcpy(custom_buf.data() + seam_bv_offset + i * 4, &s, 4);
            }
        }
        if (has_fid) {
            fid_bv_offset = custom_buf.size();
            fid_bv_size = nv * sizeof(float);
            custom_buf.resize(fid_bv_offset + fid_bv_size);
            for (int i = 0; i < nv; ++i) {
                float f = (float)mesh.face_ids(i);
                memcpy(custom_buf.data() + fid_bv_offset + i * 4, &f, 4);
            }
        }
        while (custom_buf.size() % 4) custom_buf.push_back(0);

        // Draco attrs (in Draco buffer)
        struct DracoAttr { std::string name; int draco_id; std::string type; };
        std::vector<DracoAttr> draco_attrs;
        draco_attrs.push_back({"POSITION", pos_att_id, "VEC3"});
        if (nrm_att_id >= 0) draco_attrs.push_back({"NORMAL", nrm_att_id, "VEC3"});
        if (uv_att_id >= 0) draco_attrs.push_back({"TEXCOORD_0", uv_att_id, "VEC2"});

        int next_acc = (int)draco_attrs.size(); // accessor index for custom attrs

        js << "{\"asset\":{\"version\":\"2.0\",\"generator\":\"meshparam+draco\"},"
           << "\"extensionsUsed\":[\"KHR_draco_mesh_compression\"],"
           << "\"extensionsRequired\":[\"KHR_draco_mesh_compression\"],"
           << "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],\"nodes\":[{\"mesh\":0}],"
           << "\"meshes\":[{\"primitives\":[{"
           << "\"attributes\":{";
        for (size_t i = 0; i < draco_attrs.size(); ++i) {
            if (i > 0) js << ",";
            js << "\"" << draco_attrs[i].name << "\":" << i;
        }
        if (has_seam) js << ",\"_SEAM\":" << next_acc++;
        if (has_fid) js << ",\"_FACE_ID\":" << next_acc++;
        js << "},\"mode\":4,"
           << "\"extensions\":{\"KHR_draco_mesh_compression\":{"
           << "\"bufferView\":0,\"attributes\":{";
        for (size_t i = 0; i < draco_attrs.size(); ++i) {
            if (i > 0) js << ",";
            js << "\"" << draco_attrs[i].name << "\":" << draco_attrs[i].draco_id;
        }
        js << "}}}}]}],"
           << "\"accessors\":[";
        // Draco accessors
        for (size_t i = 0; i < draco_attrs.size(); ++i) {
            if (i > 0) js << ",";
            js << "{\"componentType\":5126,\"count\":" << nv << ",\"type\":\"" << draco_attrs[i].type << "\"";
            if (i == 0) { // POSITION min/max
                js << ",\"min\":[" << mesh.V.col(0).minCoeff() << "," << mesh.V.col(1).minCoeff() << "," << mesh.V.col(2).minCoeff() << "]"
                   << ",\"max\":[" << mesh.V.col(0).maxCoeff() << "," << mesh.V.col(1).maxCoeff() << "," << mesh.V.col(2).maxCoeff() << "]";
            }
            js << "}";
        }
        // Custom accessors (uncompressed)
        int custom_bv_start = 1; // BV 0 = Draco, BV 1+ = custom
        if (has_seam) {
            js << ",{\"bufferView\":" << custom_bv_start++ << ",\"componentType\":5126,\"count\":" << nv << ",\"type\":\"SCALAR\"}";
        }
        if (has_fid) {
            js << ",{\"bufferView\":" << custom_bv_start++ << ",\"componentType\":5126,\"count\":" << nv << ",\"type\":\"SCALAR\"}";
        }
        js << "],\"bufferViews\":[{\"buffer\":0,\"byteOffset\":0,\"byteLength\":" << draco_size << "}";
        // Custom bufferViews (in the same buffer, after Draco data)
        size_t custom_base = draco_size;
        while (custom_base % 4) custom_base++;
        if (has_seam) {
            js << ",{\"buffer\":0,\"byteOffset\":" << (custom_base + seam_bv_offset)
               << ",\"byteLength\":" << seam_bv_size << ",\"target\":34962}";
        }
        if (has_fid) {
            js << ",{\"buffer\":0,\"byteOffset\":" << (custom_base + fid_bv_offset)
               << ",\"byteLength\":" << fid_bv_size << ",\"target\":34962}";
        }
        size_t total_buf = custom_base + custom_buf.size();
        js << "],\"buffers\":[{\"byteLength\":" << total_buf << "}]}";
        json_str = js.str();
    }
    while (json_str.size() % 4) json_str += ' ';

    size_t buf_padded = draco_size;
    while (buf_padded % 4) buf_padded++;

    uint32_t total = 12 + 8 + json_str.size() + 8 + buf_padded;
    std::vector<uint8_t> result(total, 0);
    uint8_t* w = result.data();

    memcpy(w, "glTF", 4); w += 4;
    uint32_t ver = 2; memcpy(w, &ver, 4); w += 4;
    memcpy(w, &total, 4); w += 4;

    uint32_t jl = json_str.size(); memcpy(w, &jl, 4); w += 4;
    uint32_t jt = 0x4E4F534A; memcpy(w, &jt, 4); w += 4;
    memcpy(w, json_str.data(), jl); w += jl;

    uint32_t bl = buf_padded; memcpy(w, &bl, 4); w += 4;
    uint32_t bt = 0x004E4942; memcpy(w, &bt, 4); w += 4;
    memcpy(w, buffer.data(), draco_size);

    return result;
}
#endif

// ============================================================
// Session store: keeps all method results for client picking
// ============================================================
struct Session {
    std::vector<MethodResult> results;
    std::chrono::steady_clock::time_point created;
};

std::mutex g_sessions_mutex;
std::unordered_map<std::string, Session> g_sessions;

std::string generate_session_id() {
    static std::mt19937 rng(std::random_device{}());
    static const char chars[] = "abcdefghijklmnopqrstuvwxyz0123456789";
    std::string id(12, ' ');
    for (auto& c : id) c = chars[rng() % (sizeof(chars) - 1)];
    return id;
}

void cleanup_sessions() {
    auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(g_sessions_mutex);
    for (auto it = g_sessions.begin(); it != g_sessions.end();) {
        auto age = std::chrono::duration_cast<std::chrono::minutes>(now - it->second.created).count();
        if (age > 10) it = g_sessions.erase(it); else ++it;
    }
}

int main(int argc, char* argv[]) {
    int port = 8080;
    std::string web_root = "../web";
    std::string occ_cli = "";
    std::string gmsh_cli = "";
    int threads = 8;

    std::string remesh_cli = "";
    std::string acis_cli = "";
    std::string benchmark_dir = "../benchmark/Obj_Files";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
        if (arg == "--web-root" && i + 1 < argc) web_root = argv[++i];
        if (arg == "--occ-cli" && i + 1 < argc) occ_cli = argv[++i];
        if (arg == "--acis-cli" && i + 1 < argc) acis_cli = argv[++i];
        if (arg == "--gmsh-cli" && i + 1 < argc) gmsh_cli = argv[++i];
        if (arg == "--benchmark-dir" && i + 1 < argc) benchmark_dir = argv[++i];
        if (arg == "--remesh-cli" && i + 1 < argc) remesh_cli = argv[++i];
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
        res.set_header("Access-Control-Expose-Headers", "X-Metrics, X-Method, X-All-Methods, X-Heal-Info, X-Session");
    });

    svr.Options("/api/(.*)", [](const httplib::Request&, httplib::Response& res) {
        res.status = 204;
    });

    // --- Health ---
    svr.Get("/api/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    // --- Broker: run all methods via subprocesses, pick best ---
    svr.Post("/api/parameterize", [&](const httplib::Request& req, httplib::Response& res) {
        std::string forced_method = req.has_param("method") ? req.get_param_value("method") : "auto";
        int timeout_sec = req.has_param("timeout") ? std::stoi(req.get_param_value("timeout")) : 60;

        std::vector<uint8_t> input(req.body.begin(), req.body.end());
        std::vector<uint8_t> input_original = input;

        // Find bench CLI executable (next to this server binary)
        std::string bench_exe;
        {
            // Derive path from argv[0] or use known location
            const char* tmp_dir = std::getenv("TEMP");
            if (!tmp_dir) tmp_dir = std::getenv("TMP");
            if (!tmp_dir) tmp_dir = ".";

            // Look for meshparam_bench in same directory as server
            char self_path[4096] = {};
#ifdef _WIN32
            GetModuleFileNameA(nullptr, self_path, sizeof(self_path));
#endif
            std::string self_dir = std::string(self_path);
            auto pos = self_dir.find_last_of("\\/");
            if (pos != std::string::npos) self_dir = self_dir.substr(0, pos + 1);
            else self_dir = "";
            bench_exe = self_dir + "meshparam_bench.exe";
            if (!std::ifstream(bench_exe).good()) {
                bench_exe = "meshparam_bench.exe"; // fallback to PATH
            }
        }

        // Write input GLB to temp file
        auto tid = std::this_thread::get_id();
        std::ostringstream ss;
        const char* tmp_dir = std::getenv("TEMP");
        if (!tmp_dir) tmp_dir = std::getenv("TMP");
        if (!tmp_dir) tmp_dir = ".";
        ss << tmp_dir << "/meshparam_broker_" << tid;
        std::string tmp_base = ss.str();
        std::string tmp_input = tmp_base + "_input.glb";
        { std::ofstream f(tmp_input, std::ios::binary);
          f.write(reinterpret_cast<const char*>(input.data()), input.size()); }

        // Determine methods to run
        std::vector<std::string> methods;
        if (forced_method == "auto") {
            methods = {"heat", "lscm", "igl_arap", "slim",
                       "cgal_conformal", "cgal_arap", "cgal_authalic"
#ifdef HAS_COMPMAJOR
                       , "cm"
#endif
                       };
        } else {
            methods = {forced_method};
        }

        // Launch all methods as subprocesses in parallel
        struct SubProc {
            std::string method;
            std::string json_path;
            std::string glb_path;
#ifdef _WIN32
            HANDLE hProcess = nullptr;
            HANDLE hThread = nullptr;
#else
            pid_t pid = 0;
#endif
        };

        std::vector<SubProc> procs(methods.size());
        for (size_t i = 0; i < methods.size(); i++) {
            procs[i].method = methods[i];
            procs[i].json_path = tmp_base + "_" + methods[i] + ".json";
            procs[i].glb_path = tmp_base + "_" + methods[i] + ".glb";

            std::string cmd = "\"" + bench_exe + "\" " + methods[i]
                + " \"" + tmp_input + "\""
                + " --json \"" + procs[i].json_path + "\""
                + " --output-glb \"" + procs[i].glb_path + "\""
;

#ifdef _WIN32
            STARTUPINFOA si = {}; si.cb = sizeof(si);
            PROCESS_INFORMATION pi = {};
            si.dwFlags = STARTF_USESHOWWINDOW;
            si.wShowWindow = SW_HIDE;
            if (CreateProcessA(nullptr, const_cast<char*>(cmd.c_str()), nullptr, nullptr, FALSE,
                              CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
                procs[i].hProcess = pi.hProcess;
                procs[i].hThread = pi.hThread;
            } else {
                std::cerr << "[broker] Failed to spawn " << methods[i] << std::endl;
            }
#else
            // POSIX: fork+exec (simplified)
            std::string full_cmd = cmd + " &";
            std::system(full_cmd.c_str());
#endif
        }

        // Wait for all with timeout
        auto broker_start = std::chrono::steady_clock::now();
#ifdef _WIN32
        std::vector<HANDLE> handles;
        for (auto& p : procs) {
            if (p.hProcess) handles.push_back(p.hProcess);
        }
        if (!handles.empty()) {
            DWORD wait_ms = static_cast<DWORD>(timeout_sec * 1000);
            WaitForMultipleObjects(static_cast<DWORD>(handles.size()), handles.data(), TRUE, wait_ms);
        }
        // Kill any still running
        for (auto& p : procs) {
            if (p.hProcess) {
                DWORD exit_code = 0;
                GetExitCodeProcess(p.hProcess, &exit_code);
                if (exit_code == STILL_ACTIVE) {
                    std::cerr << "[broker] Killing timed-out process: " << p.method << std::endl;
                    TerminateProcess(p.hProcess, 1);
                    WaitForSingleObject(p.hProcess, 5000);
                }
                CloseHandle(p.hProcess);
                CloseHandle(p.hThread);
            }
        }
#else
        // POSIX: wait with timeout (simplified — use alarm/signal)
        sleep(timeout_sec);
#endif

        // Collect results
        std::vector<MethodResult> results;
        for (auto& p : procs) {
            MethodResult r;
            r.method = p.method;
            // Read JSON result
            std::ifstream jf(p.json_path);
            if (jf.good()) {
                std::string json_str((std::istreambuf_iterator<char>(jf)), std::istreambuf_iterator<char>());
                jf.close();
                // Parse JSON manually (lightweight — no JSON library dependency)
                auto find_val = [&](const std::string& key) -> std::string {
                    auto pos = json_str.find("\"" + key + "\":");
                    if (pos == std::string::npos) return "";
                    pos += key.size() + 3;
                    if (json_str[pos] == '"') {
                        auto end = json_str.find('"', pos + 1);
                        return json_str.substr(pos + 1, end - pos - 1);
                    }
                    auto end = json_str.find_first_of(",}", pos);
                    return json_str.substr(pos, end - pos);
                };
                r.success = find_val("success") == "true";
                if (r.success) {
                    r.elapsed_ms = std::stod(find_val("elapsed_ms"));
                    r.vertices = std::stoi(find_val("vertices"));
                    r.faces = std::stoi(find_val("faces"));
                    r.angle_mean = std::stod(find_val("angle_mean"));
                    r.angle_max = std::stod(find_val("angle_max"));
                    r.area_mean = std::stod(find_val("area_mean"));
                    r.area_std = std::stod(find_val("area_std"));
                    r.stretch_mean = std::stod(find_val("stretch_mean"));
                    r.stretch_max = std::stod(find_val("stretch_max"));
                    r.flipped_tris = std::stoi(find_val("flipped_tris"));
                    r.sym_dirichlet = std::stod(find_val("sym_dirichlet"));
                    r.l2_area = std::stod(find_val("l2_area"));
                    r.linf_area = std::stod(find_val("linf_area"));
                    // Load result GLB
                    std::ifstream gf(p.glb_path, std::ios::binary | std::ios::ate);
                    if (gf.good()) {
                        size_t sz = gf.tellg(); gf.seekg(0);
                        r.glb.resize(sz);
                        gf.read(reinterpret_cast<char*>(r.glb.data()), sz);
                    }
                } else {
                    r.error = find_val("error");
                    if (r.error.empty()) r.error = "process failed";
                }
            } else {
                r.error = "no output (crash or timeout)";
            }
            results.push_back(std::move(r));
            // Cleanup temp files
            std::remove(p.json_path.c_str());
            std::remove(p.glb_path.c_str());
        }
        std::remove(tmp_input.c_str());

        auto broker_end = std::chrono::steady_clock::now();
        double broker_ms = std::chrono::duration<double, std::milli>(broker_end - broker_start).count();

        // Pick best by score
        int best_idx = -1;
        double best_score = 1e18;
        for (size_t i = 0; i < results.size(); ++i) {
            double s = results[i].score();
            std::cout << "  [broker] " << results[i].method
                      << ": " << (results[i].success ? "OK" : "FAIL")
                      << " score=" << s
                      << " (" << results[i].elapsed_ms << " ms)" << std::endl;
            if (s < best_score) { best_score = s; best_idx = static_cast<int>(i); }
        }
        std::cout << "  [broker] Total: " << broker_ms << " ms" << std::endl;

        if (best_idx < 0 || !results[best_idx].success) {
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

        // Re-apply original normals: map UVs from welded result back to split-vertex input.
        // Skip if the result came from split-and-combine (param has more faces than original
        // due to seam faces duplicated in both halves).
        {
            auto orig = meshparam::load_gltf_from_memory(input_original);
            auto param = meshparam::load_gltf_from_memory(best.glb);
            bool is_split_result = param.num_faces() > orig.num_faces();

            if (!is_split_result && orig.has_normals() && param.has_uvs() && orig.num_vertices() != param.num_vertices()) {
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
                orig.UV.resize(orig.num_vertices(), 2);
                if (param.has_seam()) orig.seam.resize(orig.num_vertices());
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
                        if (param.has_seam()) orig.seam(i) = param.seam(it->second);
                        mapped++;
                    }
                }
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

        // Store session for client method switching
        cleanup_sessions();
        std::string session_id = generate_session_id();
        {
            std::lock_guard<std::mutex> lock(g_sessions_mutex);
            g_sessions[session_id] = {results, std::chrono::steady_clock::now()};
        }

        res.set_header("X-Method", best.method);
        res.set_header("X-Metrics", best.to_json());
        res.set_header("X-All-Methods", all.str());
        res.set_header("X-Session", session_id);
        res.set_content(std::string(best.glb.begin(), best.glb.end()), "model/gltf-binary");
    });

    // --- Fetch specific method result from session ---
    svr.Get("/api/result/:session/:method", [](const httplib::Request& req, httplib::Response& res) {
        std::string session_id = req.path_params.at("session");
        std::string method = req.path_params.at("method");

        std::lock_guard<std::mutex> lock(g_sessions_mutex);
        auto it = g_sessions.find(session_id);
        if (it == g_sessions.end()) {
            res.status = 404;
            res.set_content("{\"error\":\"Session expired\"}", "application/json");
            return;
        }

        for (auto& r : it->second.results) {
            if (r.method == method && r.success) {
                res.set_header("X-Metrics", r.to_json());
                res.set_content(std::string(r.glb.begin(), r.glb.end()), "model/gltf-binary");
                return;
            }
        }

        res.status = 404;
        res.set_content("{\"error\":\"Method not found or failed\"}", "application/json");
    });

    // --- Individual method endpoints (still available) ---
    svr.Post("/api/parameterize/heat", [](const httplib::Request& req, httplib::Response& res) {
        std::vector<uint8_t> input(req.body.begin(), req.body.end());
        auto r = run_heat(input, false);
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

    // --- OBJ → GLB conversion via bench CLI ---
    svr.Post("/api/convert/obj", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            const char* tmp_dir = std::getenv("TEMP");
            if (!tmp_dir) tmp_dir = std::getenv("TMP");
            if (!tmp_dir) tmp_dir = ".";
            auto tid = std::this_thread::get_id();
            std::ostringstream ss; ss << tmp_dir << "/meshparam_obj_" << tid;
            std::string tmp_obj = ss.str() + ".obj";
            std::string tmp_glb = ss.str() + ".glb";

            { std::ofstream f(tmp_obj, std::ios::binary); f.write(req.body.data(), req.body.size()); }

            // Use bench CLI in "convert" mode
            char self_path[4096] = {};
#ifdef _WIN32
            GetModuleFileNameA(nullptr, self_path, sizeof(self_path));
#endif
            std::string self_dir = std::string(self_path);
            auto pos = self_dir.find_last_of("\\/");
            if (pos != std::string::npos) self_dir = self_dir.substr(0, pos + 1);
            std::string bench_exe = self_dir + "meshparam_bench.exe";

            // Use CreateProcess instead of system() to avoid quote issues on Windows
#ifdef _WIN32
            std::string cmd = "\"" + bench_exe + "\" convert \"" + tmp_obj + "\" --output-glb \"" + tmp_glb + "\"";
            STARTUPINFOA si = {}; si.cb = sizeof(si);
            si.dwFlags = STARTF_USESHOWWINDOW; si.wShowWindow = SW_HIDE;
            PROCESS_INFORMATION pi = {};
            int ret = -1;
            if (CreateProcessA(bench_exe.c_str(), const_cast<char*>(cmd.c_str()),
                              nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
                WaitForSingleObject(pi.hProcess, 30000);
                DWORD exit_code = 1;
                GetExitCodeProcess(pi.hProcess, &exit_code);
                ret = (exit_code == 0) ? 0 : 1;
                CloseHandle(pi.hProcess);
                CloseHandle(pi.hThread);
            }
#else
            std::string cmd = bench_exe + " convert " + tmp_obj + " --output-glb " + tmp_glb;
            int ret = std::system(cmd.c_str());
#endif
            std::remove(tmp_obj.c_str());

            if (ret != 0) {
                std::remove(tmp_glb.c_str());
                throw std::runtime_error("OBJ conversion failed");
            }

            std::ifstream f(tmp_glb, std::ios::binary | std::ios::ate);
            size_t sz = f.tellg(); f.seekg(0);
            std::vector<uint8_t> data(sz);
            f.read(reinterpret_cast<char*>(data.data()), sz);
            f.close();
            std::remove(tmp_glb.c_str());

            res.set_content(std::string(data.begin(), data.end()), "model/gltf-binary");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content("{\"error\":\"" + std::string(e.what()) + "\"}", "application/json");
        }
    });

    // --- Artist UV lookup: find matching artist OBJ and convert to GLB ---
    svr.Get("/api/artist-uvs/:filename", [&](const httplib::Request& req, httplib::Response& res) {
        std::string filename = req.path_params.at("filename");
        // Search in Artist_UVs/Cut/ and Artist_UVs/Uncut/
        std::vector<std::string> search_dirs = {
            benchmark_dir + "/Artist_UVs/Cut/",
            benchmark_dir + "/Artist_UVs/Uncut/",
        };
        std::string artist_path;
        for (auto& dir : search_dirs) {
            std::string path = dir + filename;
            if (std::ifstream(path).good()) { artist_path = path; break; }
        }
        if (artist_path.empty()) {
            res.status = 404;
            res.set_content("{\"error\":\"No artist UVs for " + filename + "\"}", "application/json");
            return;
        }
        // Convert artist OBJ to GLB (preserving UVs) via bench CLI
        try {
            const char* tmp_dir = std::getenv("TEMP");
            if (!tmp_dir) tmp_dir = std::getenv("TMP");
            if (!tmp_dir) tmp_dir = ".";
            std::string tmp_glb = std::string(tmp_dir) + "/meshparam_artist.glb";

            char self_path[4096] = {};
#ifdef _WIN32
            GetModuleFileNameA(nullptr, self_path, sizeof(self_path));
#endif
            std::string self_dir = std::string(self_path);
            auto pos = self_dir.find_last_of("\\/");
            if (pos != std::string::npos) self_dir = self_dir.substr(0, pos + 1);
            std::string bench_exe = self_dir + "meshparam_bench.exe";

            std::string cmd = "\"" + bench_exe + "\" convert \"" + artist_path + "\" --output-glb \"" + tmp_glb + "\"";
#ifdef _WIN32
            STARTUPINFOA si = {}; si.cb = sizeof(si);
            si.dwFlags = STARTF_USESHOWWINDOW; si.wShowWindow = SW_HIDE;
            PROCESS_INFORMATION pi = {};
            int ret = -1;
            if (CreateProcessA(bench_exe.c_str(), const_cast<char*>(cmd.c_str()),
                              nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
                WaitForSingleObject(pi.hProcess, 30000);
                DWORD exit_code = 1;
                GetExitCodeProcess(pi.hProcess, &exit_code);
                ret = (exit_code == 0) ? 0 : 1;
                CloseHandle(pi.hProcess);
                CloseHandle(pi.hThread);
            }
#else
            int ret = std::system(cmd.c_str());
#endif
            if (ret != 0) {
                std::remove(tmp_glb.c_str());
                throw std::runtime_error("Artist OBJ conversion failed");
            }
            std::ifstream f(tmp_glb, std::ios::binary | std::ios::ate);
            size_t sz = f.tellg(); f.seekg(0);
            std::vector<uint8_t> data(sz);
            f.read(reinterpret_cast<char*>(data.data()), sz);
            f.close();
            std::remove(tmp_glb.c_str());

            res.set_header("X-Artist-File", artist_path);
            res.set_content(std::string(data.begin(), data.end()), "model/gltf-binary");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content("{\"error\":\"" + std::string(e.what()) + "\"}", "application/json");
        }
    });

    // --- STEP tessellation (Gmsh default, OCC fallback) ---
    svr.Post("/api/tessellate/step", [&occ_cli, &gmsh_cli](const httplib::Request& req, httplib::Response& res) {
        std::string tess = req.has_param("tessellator") ? req.get_param_value("tessellator") : "gmsh";
        double defl = req.has_param("deflection") ? std::stod(req.get_param_value("deflection")) : 1.0;
        try {
            std::vector<uint8_t> glb;
            if (tess == "gmsh" && !gmsh_cli.empty()) {
                glb = tessellate_step_gmsh(gmsh_cli, req.body, defl);
            } else if (!occ_cli.empty()) {
                glb = tessellate_step(occ_cli, req.body, defl);
            } else {
                res.status = 501; res.set_content("{\"error\":\"No tessellator configured\"}", "application/json"); return;
            }
            res.set_header("X-Tessellator", tess);
            res.set_content(std::string(glb.begin(), glb.end()), "model/gltf-binary");
        } catch (const std::exception& e) {
            res.status = 500; res.set_content("{\"error\":\"" + std::string(e.what()) + "\"}", "application/json");
        }
    });

    // (Removed: /api/parameterize/step — Gmsh now tessellates STEP to GLB, then normal /api/parameterize is used)


    // --- Static files ---
    svr.set_mount_point("/", web_root);

    std::cout << "Mesh Parameterization Broker" << std::endl;
    std::cout << "  Port:     " << port << std::endl;
    std::cout << "  Threads:  " << threads << std::endl;
    std::cout << "  Web root: " << web_root << std::endl;
    std::cout << "  Gmsh CLI: " << (gmsh_cli.empty() ? "(none)" : gmsh_cli) << std::endl;
    std::cout << "  OCC CLI:  " << (occ_cli.empty() ? "(none - fallback)" : occ_cli) << std::endl;
    std::cout << "  Endpoints:" << std::endl;
    std::cout << "    POST /api/parameterize          (broker: auto-picks best)" << std::endl;
    std::cout << "    POST /api/parameterize?method=X  (force: heat|cgal_conformal|cgal_arap|cgal_authalic)" << std::endl;
    std::cout << "    POST /api/parameterize/heat     (direct)" << std::endl;
    std::cout << "    POST /api/parameterize/cgal     (direct)" << std::endl;
    std::cout << "    POST /api/tessellate/step       (Gmsh default, OCC fallback)" << std::endl;
    std::cout << "    GET  /api/health" << std::endl;
    std::cout << "\nListening on http://localhost:" << port << std::endl;

    svr.listen("0.0.0.0", port);
    return 0;
}
