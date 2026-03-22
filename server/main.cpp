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
#include <random>

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

    // Add seam attribute if available
    int seam_att_id = -1;
    if (mesh.has_seam()) {
        draco::GeometryAttribute seam_att;
        seam_att.Init(draco::GeometryAttribute::GENERIC, nullptr, 1, draco::DT_FLOAT32, false, sizeof(float), 0);
        seam_att_id = draco_mesh.AddAttribute(seam_att, true, nv);
        for (int i = 0; i < nv; ++i) {
            float s = (float)mesh.seam(i);
            draco_mesh.attribute(seam_att_id)->SetAttributeValue(draco::AttributeValueIndex(i), &s);
        }
    }

    // Add face ID attribute if available
    int fid_att_id = -1;
    if (mesh.has_face_ids()) {
        draco::GeometryAttribute fid_att;
        fid_att.Init(draco::GeometryAttribute::GENERIC, nullptr, 1, draco::DT_FLOAT32, false, sizeof(float), 0);
        fid_att_id = draco_mesh.AddAttribute(fid_att, true, nv);
        for (int i = 0; i < nv; ++i) {
            float f = (float)mesh.face_ids(i);
            draco_mesh.attribute(fid_att_id)->SetAttributeValue(draco::AttributeValueIndex(i), &f);
        }
    }

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
    // GENERIC attributes (seam, face_id) — low precision is fine
    if (seam_att_id >= 0)
        encoder.SetAttributeQuantization(seam_att_id, 1); // just 0 or 1
    if (fid_att_id >= 0)
        encoder.SetAttributeQuantization(fid_att_id, 8); // up to 256 faces

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
        // Build accessor list and attribute map dynamically
        int acc_idx = 0;
        struct AttrDef { std::string name; int draco_id; std::string type; bool has_minmax; };
        std::vector<AttrDef> attrs;
        attrs.push_back({"POSITION", pos_att_id, "VEC3", true});
        if (nrm_att_id >= 0) attrs.push_back({"NORMAL", nrm_att_id, "VEC3", false});
        if (uv_att_id >= 0) attrs.push_back({"TEXCOORD_0", uv_att_id, "VEC2", false});
        if (seam_att_id >= 0) attrs.push_back({"_SEAM", seam_att_id, "SCALAR", false});
        if (fid_att_id >= 0) attrs.push_back({"_FACE_ID", fid_att_id, "SCALAR", false});

        js << "{\"asset\":{\"version\":\"2.0\",\"generator\":\"meshparam+draco\"},"
           << "\"extensionsUsed\":[\"KHR_draco_mesh_compression\"],"
           << "\"extensionsRequired\":[\"KHR_draco_mesh_compression\"],"
           << "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],\"nodes\":[{\"mesh\":0}],"
           << "\"meshes\":[{\"primitives\":[{"
           << "\"attributes\":{";
        for (size_t i = 0; i < attrs.size(); ++i) {
            if (i > 0) js << ",";
            js << "\"" << attrs[i].name << "\":" << i;
        }
        js << "},\"mode\":4,"
           << "\"extensions\":{\"KHR_draco_mesh_compression\":{"
           << "\"bufferView\":0,\"attributes\":{";
        for (size_t i = 0; i < attrs.size(); ++i) {
            if (i > 0) js << ",";
            js << "\"" << attrs[i].name << "\":" << attrs[i].draco_id;
        }
        js << "}}}}]}],"
           << "\"accessors\":[";
        for (size_t i = 0; i < attrs.size(); ++i) {
            if (i > 0) js << ",";
            js << "{\"componentType\":5126,\"count\":" << nv << ",\"type\":\"" << attrs[i].type << "\"";
            if (attrs[i].has_minmax) {
                js << ",\"min\":[" << mesh.V.col(0).minCoeff() << "," << mesh.V.col(1).minCoeff() << "," << mesh.V.col(2).minCoeff() << "]"
                   << ",\"max\":[" << mesh.V.col(0).maxCoeff() << "," << mesh.V.col(1).maxCoeff() << "," << mesh.V.col(2).maxCoeff() << "]";
            }
            js << "}";
        }
        js << "],\"bufferViews\":[{\"buffer\":0,\"byteOffset\":0,\"byteLength\":" << draco_size << "}],"
           << "\"buffers\":[{\"byteLength\":" << draco_size << "}]}";
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
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
        if (arg == "--web-root" && i + 1 < argc) web_root = argv[++i];
        if (arg == "--occ-cli" && i + 1 < argc) occ_cli = argv[++i];
        if (arg == "--acis-cli" && i + 1 < argc) acis_cli = argv[++i];
        if (arg == "--gmsh-cli" && i + 1 < argc) gmsh_cli = argv[++i];
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

    // --- Broker: run all methods, pick best ---
    svr.Post("/api/parameterize", [&remesh_cli](const httplib::Request& req, httplib::Response& res) {
        std::string forced_method = req.has_param("method") ? req.get_param_value("method") : "auto";
        bool view_weighted = req.has_param("viewWeighted") && req.get_param_value("viewWeighted") == "true";

        std::vector<uint8_t> input(req.body.begin(), req.body.end());

        // Always auto-detect and heal degenerate triangles.
        // The "heal" flag forces healing even for near-degenerate cases (looser threshold).
        // Save original input (with OCC split-vertex normals) for seam analysis
        std::vector<uint8_t> input_original = input;

        bool force_heal = req.has_param("heal") && req.get_param_value("heal") == "true";
        auto heal1 = heal_mesh(input, force_heal);

        // Weld split vertices for parameterization
        std::vector<uint8_t> input_for_param = weld_vertices(input);

        // Heal again after welding
        auto heal2 = heal_mesh(input_for_param, force_heal);

        // Combine heal stats
        HealStats heal_total;
        heal_total.removed = heal1.removed + heal2.removed;
        heal_total.perturbed = heal1.perturbed + heal2.perturbed;
        heal_total.forced = force_heal;

        // Optionally create a remeshed version for a parallel path
        std::vector<uint8_t> input_remeshed;
        bool has_remesh = !remesh_cli.empty();
        if (has_remesh && forced_method == "auto") {
            try {
                auto t_r0 = std::chrono::high_resolution_clock::now();
                input_remeshed = remesh_isotropic(remesh_cli, input_for_param);
                auto t_r1 = std::chrono::high_resolution_clock::now();
                double rms = std::chrono::duration<double, std::milli>(t_r1 - t_r0).count();
                std::cout << "[broker] Remeshed in " << rms << " ms" << std::endl;
            } catch (const std::exception& e) {
                std::cout << "[broker] Remesh failed: " << e.what() << std::endl;
                has_remesh = false;
            }
        }

        // Determine which methods to run
        struct MethodDef { std::string name; bool on_remeshed; };
        std::vector<MethodDef> methods_to_run;

        if (forced_method == "auto") {
            // Path A: original healed+welded mesh
            methods_to_run.push_back({"heat", false});
            methods_to_run.push_back({"cgal_conformal", false});
            methods_to_run.push_back({"cgal_arap", false});
            methods_to_run.push_back({"cgal_authalic", false});
            // Path B: remeshed (if available)
            if (has_remesh) {
                methods_to_run.push_back({"heat", true});
                methods_to_run.push_back({"cgal_conformal", true});
                methods_to_run.push_back({"cgal_arap", true});
                methods_to_run.push_back({"cgal_authalic", true});
            }
        } else {
            methods_to_run.push_back({forced_method, false});
        }

        // Run all methods in parallel
        std::vector<MethodResult> results(methods_to_run.size());
        std::vector<std::thread> workers;

        for (size_t i = 0; i < methods_to_run.size(); ++i) {
            workers.emplace_back([&, i]() {
                const auto& mdef = methods_to_run[i];
                const auto& mesh_input = mdef.on_remeshed ? input_remeshed : input_for_param;
                std::string suffix = mdef.on_remeshed ? "_remeshed" : "";
                if (mdef.name == "heat") {
                    results[i] = run_heat(mesh_input, view_weighted);
                    results[i].method += suffix;
                } else if (mdef.name == "cgal_conformal") {
                    results[i] = run_cgal(mesh_input, cgalparam::ParamMethod::DiscreteConformal, "cgal_conformal" + suffix, view_weighted, mdef.on_remeshed ? std::vector<uint8_t>{} : input_original);
                } else if (mdef.name == "cgal_arap") {
                    results[i] = run_cgal(mesh_input, cgalparam::ParamMethod::ARAP, "cgal_arap" + suffix, view_weighted, mdef.on_remeshed ? std::vector<uint8_t>{} : input_original);
                } else if (mdef.name == "cgal_authalic") {
                    results[i] = run_cgal(mesh_input, cgalparam::ParamMethod::DiscreteAuthalic, "cgal_authalic" + suffix, view_weighted, mdef.on_remeshed ? std::vector<uint8_t>{} : input_original);
                } else if (mdef.name == "cgal_mvc") {
                    results[i] = run_cgal(mesh_input, cgalparam::ParamMethod::MeanValue, "cgal_mvc" + suffix, view_weighted, mdef.on_remeshed ? std::vector<uint8_t>{} : input_original);
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

                // Map UVs + seam from welded → original split vertices
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
                // face_ids already on orig from the OCC/ACIS tessellator
                std::cout << "  [broker] Mapped UVs to " << mapped << "/" << orig.num_vertices()
                          << " split vertices (normals + face_ids + seam preserved)" << std::endl;
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

        // Draco compress the best result (which has seam + face_ids after UV remap)
#ifdef HAS_DRACO
        if (!best.glb.empty()) {
            best.glb = draco_compress_glb(best.glb);
            results[best_idx].glb = best.glb;
        }
        // Also compress other successful results for session access
        for (size_t i = 0; i < results.size(); ++i) {
            if ((int)i != best_idx && results[i].success && !results[i].glb.empty()) {
                results[i].glb = draco_compress_glb(results[i].glb);
            }
        }
#endif

        // Store all results in a session for client picking
        cleanup_sessions();
        std::string session_id = generate_session_id();
        {
            std::lock_guard<std::mutex> lock(g_sessions_mutex);
            g_sessions[session_id] = {results, std::chrono::steady_clock::now()};
        }

        res.set_header("X-Method", best.method);
        res.set_header("X-Metrics", best.to_json());
        res.set_header("X-All-Methods", all.str());
        res.set_header("X-Heal-Info", heal_total.to_json());
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
    svr.Post("/api/tessellate/step", [&occ_cli, &acis_cli](const httplib::Request& req, httplib::Response& res) {
        std::string tess = req.has_param("tessellator") ? req.get_param_value("tessellator") : "occ";
        double defl = req.has_param("deflection") ? std::stod(req.get_param_value("deflection")) : 1.0;
        try {
            std::vector<uint8_t> glb;
            if (tess == "acis" && !acis_cli.empty()) {
                glb = tessellate_step_acis(acis_cli, req.body, defl);
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

    // --- STEP → dual tessellation → parameterize (all-in-one) ---
    svr.Post("/api/parameterize/step", [&occ_cli, &acis_cli, &remesh_cli](const httplib::Request& req, httplib::Response& res) {
        bool view_weighted = req.has_param("viewWeighted") && req.get_param_value("viewWeighted") == "true";
        bool do_remesh = req.has_param("remesh") && req.get_param_value("remesh") == "true";
        bool force_heal = req.has_param("heal") && req.get_param_value("heal") == "true";
        double defl = req.has_param("deflection") ? std::stod(req.get_param_value("deflection")) : 1.0;

        // Tessellate with available engines
        struct TessResult { std::string name; std::vector<uint8_t> glb; bool ok; };
        std::vector<TessResult> tessellations;

        if (!occ_cli.empty()) {
            try {
                auto t0 = std::chrono::high_resolution_clock::now();
                auto glb = tessellate_step(occ_cli, req.body, defl);
                auto ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
                std::cout << "[step] OCC tessellation: " << glb.size() << " bytes (" << ms << " ms)" << std::endl;
                tessellations.push_back({"occ", std::move(glb), true});
            } catch (const std::exception& e) {
                std::cout << "[step] OCC failed: " << e.what() << std::endl;
            }
        }

        if (!acis_cli.empty()) {
            try {
                auto t0 = std::chrono::high_resolution_clock::now();
                auto glb = tessellate_step_acis(acis_cli, req.body, defl);
                auto ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
                std::cout << "[step] ACIS tessellation: " << glb.size() << " bytes (" << ms << " ms)" << std::endl;
                tessellations.push_back({"acis", std::move(glb), true});
            } catch (const std::exception& e) {
                std::cout << "[step] ACIS failed: " << e.what() << std::endl;
            }
        }

        if (tessellations.empty()) {
            res.status = 500;
            res.set_content("{\"error\":\"All tessellations failed\"}", "application/json");
            return;
        }

        // For each tessellation, run all parameterization methods
        std::vector<MethodResult> all_results;
        std::vector<std::thread> workers;
        std::mutex results_mutex;

        std::vector<std::string> methods = {"heat", "cgal_conformal", "cgal_arap", "cgal_authalic"};

        for (auto& tess : tessellations) {
            // Heal + weld
            auto input = tess.glb;
            auto input_original = input;
            heal_mesh(input, force_heal);
            auto input_welded = weld_vertices(input);
            heal_mesh(input_welded, force_heal);

            for (auto& method_name : methods) {
                workers.emplace_back([&, method_name, input_welded, input_original, tess_name = tess.name]() {
                    MethodResult r;
                    if (method_name == "heat") {
                        r = run_heat(input_welded, view_weighted);
                    } else if (method_name == "cgal_conformal") {
                        r = run_cgal(input_welded, cgalparam::ParamMethod::DiscreteConformal, "cgal_conformal", view_weighted, input_original);
                    } else if (method_name == "cgal_arap") {
                        r = run_cgal(input_welded, cgalparam::ParamMethod::ARAP, "cgal_arap", view_weighted, input_original);
                    } else if (method_name == "cgal_authalic") {
                        r = run_cgal(input_welded, cgalparam::ParamMethod::DiscreteAuthalic, "cgal_authalic", view_weighted, input_original);
                    }
                    r.method = method_name + "_" + tess_name;

                    std::lock_guard<std::mutex> lock(results_mutex);
                    all_results.push_back(std::move(r));
                });
            }

            // Optionally add remeshed variants
            if (do_remesh && !remesh_cli.empty()) {
                try {
                    auto remeshed = remesh_isotropic(remesh_cli, input_welded);
                    for (auto& method_name : methods) {
                        workers.emplace_back([&, method_name, remeshed, tess_name = tess.name]() {
                            MethodResult r;
                            if (method_name == "heat") {
                                r = run_heat(remeshed, view_weighted);
                            } else if (method_name == "cgal_conformal") {
                                r = run_cgal(remeshed, cgalparam::ParamMethod::DiscreteConformal, "cgal_conformal", false, {});
                            } else if (method_name == "cgal_arap") {
                                r = run_cgal(remeshed, cgalparam::ParamMethod::ARAP, "cgal_arap", false, {});
                            } else if (method_name == "cgal_authalic") {
                                r = run_cgal(remeshed, cgalparam::ParamMethod::DiscreteAuthalic, "cgal_authalic", false, {});
                            }
                            r.method = method_name + "_" + tess_name + "_remeshed";

                            std::lock_guard<std::mutex> lock(results_mutex);
                            all_results.push_back(std::move(r));
                        });
                    }
                } catch (...) {}
            }
        }

        for (auto& w : workers) w.join();

        // Pick best
        int best_idx = -1;
        double best_score = 1e18;
        for (size_t i = 0; i < all_results.size(); ++i) {
            auto s = all_results[i].score();
            std::cout << "  [step-broker] " << all_results[i].method
                      << ": " << (all_results[i].success ? "OK" : "FAIL")
                      << " score=" << s << std::endl;
            if (s < best_score) { best_score = s; best_idx = static_cast<int>(i); }
        }

        if (best_idx < 0) {
            res.status = 500;
            res.set_content("{\"error\":\"All methods failed\"}", "application/json");
            return;
        }

        auto& best = all_results[best_idx];
        std::cout << "  [step-broker] Winner: " << best.method << " (score=" << best.score() << ")" << std::endl;

#ifdef HAS_DRACO
        for (auto& r : all_results) {
            if (r.success && !r.glb.empty()) r.glb = draco_compress_glb(r.glb);
        }
#endif

        // Store session
        cleanup_sessions();
        std::string session_id = generate_session_id();
        { std::lock_guard<std::mutex> lock(g_sessions_mutex);
          g_sessions[session_id] = {all_results, std::chrono::steady_clock::now()}; }

        // Build all-methods JSON
        std::ostringstream all_json;
        all_json << "[";
        for (size_t i = 0; i < all_results.size(); ++i) {
            if (i > 0) all_json << ",";
            all_json << all_results[i].to_json();
        }
        all_json << "]";

        res.set_header("X-Method", best.method);
        res.set_header("X-Metrics", best.to_json());
        res.set_header("X-All-Methods", all_json.str());
        res.set_header("X-Session", session_id);
        res.set_content(std::string(best.glb.begin(), best.glb.end()), "model/gltf-binary");
    });

    // --- Static files ---
    svr.set_mount_point("/", web_root);

    std::cout << "Mesh Parameterization Broker" << std::endl;
    std::cout << "  Port:     " << port << std::endl;
    std::cout << "  Threads:  " << threads << std::endl;
    std::cout << "  Web root: " << web_root << std::endl;
    std::cout << "  OCC CLI:  " << (occ_cli.empty() ? "(none)" : occ_cli) << std::endl;
    std::cout << "  ACIS CLI: " << (acis_cli.empty() ? "(none)" : acis_cli) << std::endl;
    std::cout << "  Remesh:   " << (remesh_cli.empty() ? "(none)" : remesh_cli) << std::endl;
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
