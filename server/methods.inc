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

    std::cerr << "[weld] Welded " << n << " → " << new_n << " vertices" << std::endl;

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
        std::cerr << "[heal] Perturbed " << perturbed << " collinear triangles" << std::endl;
    std::cerr << "[heal] Removed " << removed << " degenerate triangles (" << m << " → " << (m - removed) << ")"
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
// Extract the largest connected component from a mesh
std::vector<uint8_t> extract_largest_component(const std::vector<uint8_t>& input_glb) {
    auto mesh = meshparam::load_gltf_from_memory(input_glb);
    int nv = mesh.num_vertices(), nf = mesh.num_faces();
    if (nf == 0) return input_glb;

    // BFS to find connected components via face adjacency
    std::vector<std::vector<int>> vert_faces(nv);
    for (int f = 0; f < nf; f++) {
        for (int j = 0; j < 3; j++) vert_faces[mesh.F(f, j)].push_back(f);
    }

    std::vector<int> face_comp(nf, -1);
    int num_comp = 0;
    for (int f = 0; f < nf; f++) {
        if (face_comp[f] >= 0) continue;
        int comp = num_comp++;
        std::vector<int> stack = {f};
        while (!stack.empty()) {
            int cf = stack.back(); stack.pop_back();
            if (face_comp[cf] >= 0) continue;
            face_comp[cf] = comp;
            for (int j = 0; j < 3; j++) {
                int v = mesh.F(cf, j);
                for (int nf2 : vert_faces[v]) {
                    if (face_comp[nf2] < 0) stack.push_back(nf2);
                }
            }
        }
    }

    if (num_comp <= 1) return input_glb; // single component

    // Find largest component
    std::vector<int> comp_size(num_comp, 0);
    for (int f = 0; f < nf; f++) comp_size[face_comp[f]]++;
    int largest = std::max_element(comp_size.begin(), comp_size.end()) - comp_size.begin();

    std::cerr << "[component] " << num_comp << " components, extracting largest ("
              << comp_size[largest] << "/" << nf << " faces)" << std::endl;

    // Extract faces and remap vertices
    std::vector<bool> used(nv, false);
    std::vector<Eigen::Vector3i> new_faces;
    for (int f = 0; f < nf; f++) {
        if (face_comp[f] != largest) continue;
        new_faces.push_back(mesh.F.row(f));
        for (int j = 0; j < 3; j++) used[mesh.F(f, j)] = true;
    }
    std::vector<int> remap(nv, -1);
    int new_nv = 0;
    for (int i = 0; i < nv; i++) {
        if (used[i]) remap[i] = new_nv++;
    }

    meshparam::TriMesh out;
    out.V.resize(new_nv, 3);
    if (mesh.has_normals()) out.N.resize(new_nv, 3);
    if (mesh.has_face_ids()) out.face_ids.resize(new_nv);
    for (int i = 0; i < nv; i++) {
        if (remap[i] < 0) continue;
        out.V.row(remap[i]) = mesh.V.row(i);
        if (mesh.has_normals()) out.N.row(remap[i]) = mesh.N.row(i);
        if (mesh.has_face_ids()) out.face_ids(remap[i]) = mesh.face_ids(i);
    }
    out.F.resize(new_faces.size(), 3);
    for (size_t f = 0; f < new_faces.size(); f++) {
        for (int j = 0; j < 3; j++) out.F(f, j) = remap[new_faces[f](j)];
    }

    return meshparam::save_gltf_to_memory(out);
}

// ============================================================
// Find the longest Z-perpendicular B-Rep edge loop for auto-seam
// Returns vertex indices of the loop, or empty if none found.
// ============================================================
std::vector<int> find_longest_z_loop(const std::vector<uint8_t>& input_glb, double z_tol = 0.5) {
    auto mesh = meshparam::load_gltf_from_memory(input_glb);
    if (!mesh.has_face_ids()) return {};
    int nv = mesh.num_vertices(), nf = mesh.num_faces();

    // Build B-Rep boundary edges (adjacent tris with different face_id)
    struct PosKey {
        int64_t x, y, z;
        bool operator<(const PosKey& o) const {
            return std::tie(x,y,z) < std::tie(o.x,o.y,o.z);
        }
        bool operator==(const PosKey& o) const {
            return x==o.x && y==o.y && z==o.z;
        }
    };
    auto make_key = [&](int i) -> PosKey {
        return {(int64_t)std::round(mesh.V(i,0)*1e4),
                (int64_t)std::round(mesh.V(i,1)*1e4),
                (int64_t)std::round(mesh.V(i,2)*1e4)};
    };
    struct PosKeyHash {
        size_t operator()(const PosKey& k) const {
            size_t h = 0;
            h ^= std::hash<int64_t>()(k.x) + 0x9e3779b9 + (h<<6) + (h>>2);
            h ^= std::hash<int64_t>()(k.y) + 0x9e3779b9 + (h<<6) + (h>>2);
            h ^= std::hash<int64_t>()(k.z) + 0x9e3779b9 + (h<<6) + (h>>2);
            return h;
        }
    };

    // Edge key: sorted pair of position keys
    struct EdgeKey {
        PosKey a, b;
        bool operator==(const EdgeKey& o) const { return a==o.a && b==o.b; }
    };
    struct EdgeKeyHash {
        PosKeyHash h;
        size_t operator()(const EdgeKey& e) const { return h(e.a) ^ (h(e.b) * 31); }
    };
    auto make_edge = [](PosKey a, PosKey b) -> EdgeKey {
        return (a < b) ? EdgeKey{a, b} : EdgeKey{b, a};
    };

    struct EdgeInfo { double fid; int va, vb; };
    std::unordered_map<EdgeKey, std::vector<EdgeInfo>, EdgeKeyHash> edge_map;
    for (int f = 0; f < nf; f++) {
        double fid = mesh.face_ids(mesh.F(f, 0));
        for (int e = 0; e < 3; e++) {
            int a = mesh.F(f, e), b = mesh.F(f, (e+1)%3);
            auto ek = make_edge(make_key(a), make_key(b));
            edge_map[ek].push_back({fid, a, b});
        }
    }

    // Group boundary edges by face-pair (sorted pair of face IDs)
    // Each face-pair boundary forms one or more edge chains
    struct BndEdge { int a, b; };
    std::map<std::pair<int,int>, std::vector<BndEdge>> face_pair_edges;
    for (auto& [ek, entries] : edge_map) {
        std::set<double> fids;
        for (auto& e : entries) fids.insert(e.fid);
        if (fids.size() <= 1) continue;
        // Use sorted pair of the two most common face IDs
        std::vector<int> fid_list(fids.begin(), fids.end());
        for (size_t i = 0; i < fid_list.size(); i++) {
            for (size_t j = i+1; j < fid_list.size(); j++) {
                auto key = std::make_pair(std::min(fid_list[i], fid_list[j]),
                                          std::max(fid_list[i], fid_list[j]));
                face_pair_edges[key].push_back({entries[0].va, entries[0].vb});
            }
        }
    }

    // For each face-pair, trace loops from its edges
    std::vector<std::vector<int>> loops;
    for (auto& [fpair, edges] : face_pair_edges) {
        // Build local adjacency
        std::unordered_map<PosKey, std::vector<std::pair<PosKey, int>>, PosKeyHash> adj;
        for (auto& e : edges) {
            auto ka = make_key(e.a), kb = make_key(e.b);
            adj[ka].push_back({kb, e.b});
            adj[kb].push_back({ka, e.a});
        }
        // Trace loops in this local graph
        std::unordered_set<PosKey, PosKeyHash> visited;
        for (auto& [start_key, _] : adj) {
            if (visited.count(start_key)) continue;
            std::vector<int> loop;
            PosKey cur = start_key, prev = {INT64_MIN, INT64_MIN, INT64_MIN};
            bool closed = false;
            for (int step = 0; step < 200000; step++) {
                if (visited.count(cur) && loop.size() > 2) {
                    if (cur == start_key) closed = true;
                    break;
                }
                visited.insert(cur);
                auto& nbrs = adj[cur];
                PosKey next_key = {INT64_MIN, INT64_MIN, INT64_MIN};
                for (auto& [nk, ni] : nbrs) {
                    if (nk == prev) continue;
                    if (nk == start_key && loop.size() > 2) {
                        loop.push_back(ni);
                        next_key = start_key;
                        closed = true;
                        break;
                    }
                    if (!visited.count(nk)) {
                        loop.push_back(ni);
                        next_key = nk;
                        break;
                    }
                }
                if (next_key.x == INT64_MIN) break;
                if (closed) break;
                prev = cur;
                cur = next_key;
            }
            if (closed && loop.size() >= 3) loops.push_back(std::move(loop));
        }
    }

    // Filter Z-perpendicular loops and find longest by perimeter
    double best_perim = 0;
    std::vector<int> best_loop;
    for (auto& loop : loops) {
        double z_min = 1e18, z_max = -1e18;
        for (int i : loop) {
            double z = mesh.V(i, 2);
            z_min = std::min(z_min, z);
            z_max = std::max(z_max, z);
        }
        if (z_max - z_min > z_tol) continue;

        // Compute perimeter in mm
        double perim = 0;
        for (size_t i = 0; i < loop.size(); i++) {
            int a = loop[i], b = loop[(i+1) % loop.size()];
            perim += (mesh.V.row(a) - mesh.V.row(b)).norm();
        }
        if (perim > best_perim) {
            best_perim = perim;
            best_loop = loop;
        }
    }

    int total_bnd = 0;
    for (auto& [k, v] : face_pair_edges) total_bnd += v.size();
    std::cerr << "[auto-seam] " << nv << "v " << nf << "f, "
              << total_bnd << " boundary edges, "
              << face_pair_edges.size() << " face pairs, "
              << loops.size() << " closed loops" << std::endl;
    if (!best_loop.empty()) {
        std::cerr << "[auto-seam] Longest Z-perp loop: "
                  << best_loop.size() << " vertices, " << best_perim << " mm perimeter" << std::endl;
    } else {
        // Count how many loops were Z-perpendicular vs not
        int z_count = 0;
        for (auto& loop : loops) {
            double zmin = 1e18, zmax = -1e18;
            for (int i : loop) { zmin = std::min(zmin, mesh.V(i,2)); zmax = std::max(zmax, mesh.V(i,2)); }
            if (zmax - zmin <= z_tol) z_count++;
        }
        std::cerr << "[auto-seam] No suitable Z-perp loops (" << z_count << " Z-perp of " << loops.size() << " total)" << std::endl;
    }
    return best_loop;
}

// Normalize UVs to [0,1] with uniform scaling (preserves aspect ratio)
void normalize_uv_uniform(Eigen::MatrixXd& UV) {
    Eigen::Vector2d mn = UV.colwise().minCoeff();
    Eigen::Vector2d mx = UV.colwise().maxCoeff();
    double range = std::max(mx(0) - mn(0), mx(1) - mn(1));
    if (range > 1e-12) {
        UV.col(0) = (UV.col(0).array() - mn(0)) / range;
        UV.col(1) = (UV.col(1).array() - mn(1)) / range;
    }
}

// --- MethodResult and run_* functions ---
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

// Cut a closed mesh along a set of seam vertices (from auto-seam detection).
// Orders the vertices into a connected path by following mesh edges.
std::vector<uint8_t> cut_mesh_along_loop(const std::vector<uint8_t>& input_glb,
                                          const std::vector<int>& seam_verts) {
    if (seam_verts.size() < 3) return cut_closed_mesh(input_glb);

    auto tri = cgalparam::load_gltf_from_memory(input_glb);
    auto sm = cgalparam::to_cgal_mesh(tri);
    int cgal_nv = static_cast<int>(sm.number_of_vertices());

    // Build set of seam vertex indices for quick lookup
    std::set<int> seam_set(seam_verts.begin(), seam_verts.end());

    // Build adjacency among seam vertices via mesh edges
    std::map<int, std::vector<int>> adj;
    for (auto e : sm.edges()) {
        auto h = sm.halfedge(e);
        int a = static_cast<int>(sm.source(h));
        int b = static_cast<int>(sm.target(h));
        if (seam_set.count(a) && seam_set.count(b)) {
            adj[a].push_back(b);
            adj[b].push_back(a);
        }
    }

    // Trace ordered path from the seam adjacency graph
    // Start from any seam vertex, follow edges to form a loop
    if (adj.empty()) {
        std::cerr << "[auto-seam] No adjacent seam edges found, fallback to BFS" << std::endl;
        if (CGAL::is_closed(sm)) {
            auto cut = cgalparam::cut_to_disk(sm);
            return cgalparam::save_gltf_to_memory(cgalparam::from_cgal_mesh(cut.cut_mesh));
        }
        return input_glb;
    }

    std::vector<int> ordered;
    std::set<int> visited;
    int start = adj.begin()->first;
    int cur = start, prev = -1;
    for (int step = 0; step < (int)seam_verts.size() + 10; step++) {
        ordered.push_back(cur);
        visited.insert(cur);
        int next = -1;
        for (int nb : adj[cur]) {
            if (nb == prev) continue;
            if (nb == start && ordered.size() > 2) {
                // Closed the loop
                next = -2;
                break;
            }
            if (!visited.count(nb)) {
                next = nb;
                break;
            }
        }
        if (next == -2) break; // closed loop
        if (next < 0) break;   // dead end
        prev = cur;
        cur = next;
    }

    std::cerr << "[auto-seam] Ordered path: " << ordered.size() << " vertices from "
              << seam_verts.size() << " seam vertices" << std::endl;

    // Convert to CGAL vertex descriptors
    std::vector<cgalparam::SurfaceMesh::Vertex_index> path;
    for (int vi : ordered) {
        if (vi >= 0 && vi < cgal_nv)
            path.push_back(cgalparam::SurfaceMesh::Vertex_index(vi));
    }

    if (path.size() < 3) {
        std::cerr << "[auto-seam] Path too short, fallback to BFS" << std::endl;
        if (CGAL::is_closed(sm)) {
            auto cut = cgalparam::cut_to_disk(sm);
            return cgalparam::save_gltf_to_memory(cgalparam::from_cgal_mesh(cut.cut_mesh));
        }
        return input_glb;
    }

    try {
        auto cut = cgalparam::cut_along_path(sm, path);
        auto cut_tri = cgalparam::from_cgal_mesh(cut.cut_mesh);
        std::cerr << "[auto-seam] Cut: " << tri.V.rows() << " -> "
                  << cut_tri.V.rows() << " verts" << std::endl;
        return cgalparam::save_gltf_to_memory(cut_tri);
    } catch (const std::exception& e) {
        std::cerr << "[auto-seam] Cut failed: " << e.what() << ", fallback to BFS" << std::endl;
        if (CGAL::is_closed(sm)) {
            auto cut = cgalparam::cut_to_disk(sm);
            return cgalparam::save_gltf_to_memory(cgalparam::from_cgal_mesh(cut.cut_mesh));
        }
        return input_glb;
    }
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

        normalize_uv_uniform(UV);

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

            normalize_uv_uniform(UV);

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

// Stein ADMM removed — crashed on ~50% of meshes, replaced by CM
MethodResult run_stein(const std::vector<uint8_t>& input_glb) {
    MethodResult r; r.method = "stein_admm"; r.error = "Stein ADMM removed"; return r;
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

        normalize_uv_uniform(UV);

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
// Composite Majorization (Shtengel et al. 2017)
// ============================================================
#ifdef HAS_COMPMAJOR
#include "Newton.h"

MethodResult run_cm(const std::vector<uint8_t>& input_glb, int iterations = 100) {
    MethodResult r;
    r.method = "cm";
    auto t0 = std::chrono::high_resolution_clock::now();
    try {
        auto glb_with_bnd = cut_closed_mesh(input_glb);
        // Heal after cut — seam cutting can create degenerate triangles
        heal_mesh(glb_with_bnd, true);
        auto mesh = meshparam::load_gltf_from_memory(glb_with_bnd);
        int nv = mesh.num_vertices(), nf = mesh.num_faces();

        Eigen::VectorXi bnd;
        igl::boundary_loop(mesh.F, bnd);
        if (bnd.size() == 0) {
            r.error = "CM requires mesh with boundary";
            auto t1 = std::chrono::high_resolution_clock::now();
            r.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            return r;
        }

        // Normalize mesh area (CompMajor convention)
        Eigen::VectorXd areas;
        igl::doublearea(mesh.V, mesh.F, areas);
        areas /= 2.0;
        double mesh_area = areas.sum();
        Eigen::MatrixXd V_norm = mesh.V / sqrt(mesh_area);

        MatX3 V_cm = V_norm;
        MatX3i F_cm = mesh.F;

        Newton solver;

        // Try LSCM init first (better starting point), fall back to Tutte
        bool init_ok = false;
        {
            Eigen::VectorXi b2(2); Eigen::MatrixXd bc2(2,2);
            b2(0) = bnd(0); b2(1) = bnd(bnd.size()/2);
            bc2 << 0, 0, 1, 0;
            Eigen::MatrixXd UV_lscm;
            igl::lscm(V_cm, F_cm, b2, bc2, UV_lscm);
            bool lscm_ok = UV_lscm.rows() == V_cm.rows();
            if (lscm_ok) {
                for (int i = 0; i < UV_lscm.rows() && lscm_ok; i++)
                    if (!std::isfinite(UV_lscm(i,0)) || !std::isfinite(UV_lscm(i,1))) lscm_ok = false;
            }
            // Prefer LSCM without flips, but accept with flips as last resort
            if (lscm_ok) {
                bool no_flips = igl::flipped_triangles(UV_lscm, F_cm).size() == 0;
                if (no_flips) {
                    try {
                        solver.init_with_uv(V_cm, F_cm, MatX2(UV_lscm));
                        init_ok = true;
                    } catch (...) {}
                }
                if (!init_ok) {
                    // Try Tutte (default)
                    try {
                        solver.init(V_cm, F_cm);
                        init_ok = true;
                    } catch (...) {}
                }
                if (!init_ok && !no_flips) {
                    // Last resort: LSCM with flips — line search may fix them
                    try {
                        solver.init_with_uv(V_cm, F_cm, MatX2(UV_lscm));
                        init_ok = true;
                    } catch (...) {}
                }
            } else {
                // No LSCM, just Tutte
                try {
                    solver.init(V_cm, F_cm);
                    init_ok = true;
                } catch (...) {}
            }
        }
        if (!init_ok) {
            r.error = "CM: all initialization strategies failed";
        }

        if (init_ok) {
            for (int i = 0; i < iterations; i++) {
                int res = solver.step();
                if (res != 0) {
                    r.error = "CM: Newton step failed at iteration " + std::to_string(i);
                    break;
                }
                solver.linesearch();
                if (solver.g.norm() < 1e-8) break;
            }
        }

        if (r.error.empty()) {
            Eigen::MatrixXd uv = Eigen::Map<MatX2>(solver.m_x.data(), solver.m_x.rows() / 2, 2);
            normalize_uv_uniform(uv);

            mesh.UV = uv;
            mesh.compute_normals();

            auto metrics = meshparam::compute_distortion(mesh.V, mesh.F, mesh.UV);
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
            fill_benchmark_metrics(r, mesh.V, mesh.F, mesh.UV);
        }
    } catch (const std::exception& e) {
        r.error = e.what();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    r.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return r;
}
#endif // HAS_COMPMAJOR
