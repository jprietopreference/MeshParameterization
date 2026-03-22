#include "cgalparam/seam_cut.h"
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/boost/graph/Euler_operations.h>

#include <queue>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <limits>

namespace cgalparam {

namespace {

using VD = vertex_descriptor;
using HD = halfedge_descriptor;
using ED = SurfaceMesh::Edge_index;

/// BFS to find the farthest vertex from source (hop distance).
VD bfs_farthest(const SurfaceMesh& mesh, VD source) {
    std::map<VD, int> dist;
    std::queue<VD> q;
    dist[source] = 0;
    q.push(source);
    VD farthest = source;
    int max_dist = 0;

    while (!q.empty()) {
        VD v = q.front(); q.pop();
        for (auto h : mesh.halfedges_around_target(mesh.halfedge(v))) {
            VD u = mesh.source(h);
            if (dist.find(u) == dist.end()) {
                dist[u] = dist[v] + 1;
                if (dist[u] > max_dist) {
                    max_dist = dist[u];
                    farthest = u;
                }
                q.push(u);
            }
        }
    }
    return farthest;
}

/// Find two "pole" vertices that are far apart (approximate diameter).
/// Uses double-BFS: BFS from arbitrary vertex, then BFS from farthest.
std::pair<VD, VD> find_poles(const SurfaceMesh& mesh) {
    VD start = *mesh.vertices().begin();
    VD pole1 = bfs_farthest(mesh, start);
    VD pole2 = bfs_farthest(mesh, pole1);
    return {pole1, pole2};
}

/// Dijkstra shortest path on mesh edges (weighted by edge length).
/// Returns the path as a list of vertices from source to target.
std::vector<VD> dijkstra_path(const SurfaceMesh& mesh, VD source, VD target) {
    using PQEntry = std::pair<double, VD>;  // (distance, vertex)
    std::priority_queue<PQEntry, std::vector<PQEntry>, std::greater<PQEntry>> pq;
    std::map<VD, double> dist;
    std::map<VD, VD> prev;

    dist[source] = 0;
    pq.push({0, source});

    while (!pq.empty()) {
        auto [d, v] = pq.top(); pq.pop();

        if (v == target) break;
        if (d > dist[v]) continue;  // stale entry

        for (auto h : mesh.halfedges_around_target(mesh.halfedge(v))) {
            VD u = mesh.source(h);
            auto p1 = mesh.point(v);
            auto p2 = mesh.point(u);
            double edge_len = std::sqrt(
                (p1.x()-p2.x())*(p1.x()-p2.x()) +
                (p1.y()-p2.y())*(p1.y()-p2.y()) +
                (p1.z()-p2.z())*(p1.z()-p2.z()));
            double new_dist = dist[v] + edge_len;

            if (dist.find(u) == dist.end() || new_dist < dist[u]) {
                dist[u] = new_dist;
                prev[u] = v;
                pq.push({new_dist, u});
            }
        }
    }

    // Reconstruct path
    std::vector<VD> path;
    if (prev.find(target) == prev.end() && target != source) {
        return path;  // no path found
    }
    VD v = target;
    while (v != source) {
        path.push_back(v);
        v = prev[v];
    }
    path.push_back(source);
    std::reverse(path.begin(), path.end());
    return path;
}

/// Find the edge between two adjacent vertices (if exists).
ED find_edge(const SurfaceMesh& mesh, VD v1, VD v2) {
    for (auto h : mesh.halfedges_around_target(mesh.halfedge(v1))) {
        if (mesh.source(h) == v2) {
            return mesh.edge(h);
        }
    }
    return SurfaceMesh::null_edge();
}

/// Cut a mesh along a given vertex path. Duplicates interior path vertices
/// to create a boundary. Returns the cut mesh.
SeamCutResult cut_along_path(const SurfaceMesh& input_mesh, const std::vector<VD>& path) {
    SeamCutResult result;
    int n_orig = static_cast<int>(input_mesh.number_of_vertices());
    result.original_vertex_count = n_orig;

    if (path.size() < 2) {
        // Can't cut with less than 2 vertices — remove a face as fallback
        result.cut_mesh = input_mesh;
        auto f_it = result.cut_mesh.faces_begin();
        if (f_it != result.cut_mesh.faces_end()) {
            CGAL::Euler::remove_face(result.cut_mesh.halfedge(*f_it), result.cut_mesh);
            result.cut_mesh.collect_garbage();
        }
        result.vertex_map.resize(n_orig);
        for (int i = 0; i < n_orig; ++i) result.vertex_map[i] = i;
        return result;
    }

    // Collect seam edges
    std::set<ED> seam_edges;
    for (size_t i = 0; i + 1 < path.size(); ++i) {
        ED e = find_edge(input_mesh, path[i], path[i+1]);
        if (e != SurfaceMesh::null_edge()) seam_edges.insert(e);
    }

    // BFS to assign face sides (0 or 1) across seam edges
    const SurfaceMesh& sm = input_mesh;
    std::map<face_descriptor, int> face_side;
    std::set<face_descriptor> visited_faces;
    std::queue<face_descriptor> bfs_q;

    ED first_seam = *seam_edges.begin();
    HD h0 = sm.halfedge(first_seam);
    face_descriptor seed = sm.face(h0);
    if (seed == SurfaceMesh::null_face()) seed = sm.face(sm.opposite(h0));

    face_side[seed] = 0;
    bfs_q.push(seed);
    visited_faces.insert(seed);

    while (!bfs_q.empty()) {
        face_descriptor f = bfs_q.front(); bfs_q.pop();
        int side = face_side[f];
        for (auto h : sm.halfedges_around_face(sm.halfedge(f))) {
            ED e = sm.edge(h);
            face_descriptor nb = sm.face(sm.opposite(h));
            if (nb == SurfaceMesh::null_face() || visited_faces.count(nb)) continue;
            face_side[nb] = seam_edges.count(e) ? (1 - side) : side;
            visited_faces.insert(nb);
            bfs_q.push(nb);
        }
    }

    // Rebuild mesh with duplicated interior seam vertices
    std::set<VD> interior_seam(path.begin() + 1, path.end() - 1);

    SurfaceMesh new_mesh;
    std::vector<VD> orig_to_new(n_orig);
    for (auto v : sm.vertices()) {
        orig_to_new[static_cast<int>(v)] = new_mesh.add_vertex(sm.point(v));
    }

    result.vertex_map.resize(n_orig);
    for (int i = 0; i < n_orig; ++i) result.vertex_map[i] = i;

    std::map<int, VD> duplicates;
    for (VD sv : interior_seam) {
        int si = static_cast<int>(sv);
        VD dup = new_mesh.add_vertex(sm.point(sv));
        duplicates[si] = dup;
        int di = static_cast<int>(dup);
        if (di >= (int)result.vertex_map.size()) result.vertex_map.resize(di + 1);
        result.vertex_map[di] = si;
    }

    for (auto f : sm.faces()) {
        int side = face_side.count(f) ? face_side[f] : 0;
        VD fv[3]; int k = 0;
        for (auto v : sm.vertices_around_face(sm.halfedge(f))) {
            int vi = static_cast<int>(v);
            fv[k++] = (side == 1 && interior_seam.count(v)) ? duplicates[vi] : orig_to_new[vi];
        }
        if (k == 3) new_mesh.add_face(fv[0], fv[1], fv[2]);
    }

    new_mesh.collect_garbage();
    result.cut_mesh = new_mesh;
    result.vertex_map.resize(result.cut_mesh.number_of_vertices());

    // Verify
    if (!result.cut_mesh.is_valid(false)) {
        std::cerr << "[seam] WARNING: cut mesh invalid, falling back" << std::endl;
        result.cut_mesh = input_mesh;
        auto f_it = result.cut_mesh.faces_begin();
        CGAL::Euler::remove_face(result.cut_mesh.halfedge(*f_it), result.cut_mesh);
        result.cut_mesh.collect_garbage();
        result.vertex_map.resize(n_orig);
        for (int i = 0; i < n_orig; ++i) result.vertex_map[i] = i;
        return result;
    }

    HD new_border = CGAL::Polygon_mesh_processing::longest_border(result.cut_mesh).first;
    if (new_border == SurfaceMesh::null_halfedge()) {
        std::cerr << "[seam] WARNING: no boundary after cut, falling back" << std::endl;
        result.cut_mesh = input_mesh;
        auto f_it = result.cut_mesh.faces_begin();
        CGAL::Euler::remove_face(result.cut_mesh.halfedge(*f_it), result.cut_mesh);
        result.cut_mesh.collect_garbage();
        result.vertex_map.resize(n_orig);
        for (int i = 0; i < n_orig; ++i) result.vertex_map[i] = i;
    } else {
        int new_V = static_cast<int>(result.cut_mesh.number_of_vertices());
        std::cout << "[seam] Cut: " << n_orig << " -> " << new_V << " verts "
                  << "(+" << (new_V - n_orig) << " seam)" << std::endl;
    }

    return result;
}

} // anonymous namespace

SeamCutResult cut_to_disk(const SurfaceMesh& input_mesh) {
    SeamCutResult result;

    // Check if mesh already has a boundary
    HD border = CGAL::Polygon_mesh_processing::longest_border(input_mesh).first;
    if (border != SurfaceMesh::null_halfedge()) {
        // Already has boundary — copy as-is
        result.cut_mesh = input_mesh;
        result.original_vertex_count = static_cast<int>(input_mesh.number_of_vertices());
        result.vertex_map.resize(result.original_vertex_count);
        for (int i = 0; i < result.original_vertex_count; ++i)
            result.vertex_map[i] = i;
        return result;
    }

    int n_orig = static_cast<int>(input_mesh.number_of_vertices());
    result.original_vertex_count = n_orig;

    // Compute genus: V - E + F = 2 - 2g
    int V = static_cast<int>(input_mesh.number_of_vertices());
    int E = static_cast<int>(input_mesh.number_of_edges());
    int F = static_cast<int>(input_mesh.number_of_faces());
    int genus = (2 - V + E - F) / 2;
    std::cout << "[seam] Mesh: V=" << V << " E=" << E << " F=" << F
              << " genus=" << genus << std::endl;

    // Step 1: Find two poles (far apart vertices)
    auto [pole1, pole2] = find_poles(input_mesh);
    std::cout << "[seam] Poles: " << static_cast<int>(pole1) << " ↔ "
              << static_cast<int>(pole2) << std::endl;

    // Step 2: Find shortest geodesic path between poles
    auto path = dijkstra_path(input_mesh, pole1, pole2);
    if (path.size() < 2) {
        std::cerr << "[seam] WARNING: no path found, falling back to face removal" << std::endl;
        result.cut_mesh = input_mesh;
        auto f_it = result.cut_mesh.faces_begin();
        if (f_it != result.cut_mesh.faces_end()) {
            CGAL::Euler::remove_face(result.cut_mesh.halfedge(*f_it), result.cut_mesh);
            result.cut_mesh.collect_garbage();
        }
        result.vertex_map.resize(n_orig);
        for (int i = 0; i < n_orig; ++i) result.vertex_map[i] = i;
        return result;
    }

    std::cout << "[seam] Seam path: " << path.size() << " vertices" << std::endl;
    return cut_along_path(input_mesh, path);
}

SeamCutResult cut_brep_silhouette(const SurfaceMesh& input_mesh,
                                   const Eigen::MatrixXd& orig_V,
                                   const Eigen::MatrixXd& orig_N,
                                   const Eigen::MatrixXi& orig_F,
                                   const Eigen::VectorXd& orig_face_ids,
                                   double z_threshold) {
    // Always apply silhouette seam, even if mesh already has a small boundary
    // from healed degenerate triangles. The silhouette cut ensures front-facing
    // faces get a continuous UV patch.

    // Use OCC face IDs to classify each B-Rep face as Z+ (front) or not.
    // Then find welded mesh edges where adjacent triangles belong to
    // different OCC faces with different Z-classification.

    int orig_nv = static_cast<int>(orig_V.rows());
    int orig_nf = static_cast<int>(orig_F.rows());

    // Compute per-OCC-face average normal Z
    std::map<int, std::pair<double, int>> occ_face_z_sum; // face_id -> (sum_z, count)
    for (int i = 0; i < orig_nv; ++i) {
        int fid = static_cast<int>(orig_face_ids(i));
        occ_face_z_sum[fid].first += orig_N(i, 2);
        occ_face_z_sum[fid].second += 1;
    }
    std::map<int, bool> occ_face_is_front; // true if Z+ facing
    for (auto& [fid, p] : occ_face_z_sum) {
        double avg_z = p.first / p.second;
        occ_face_is_front[fid] = (avg_z > z_threshold);
    }

    int n_front = 0, n_back = 0;
    for (auto& [fid, is_front] : occ_face_is_front) {
        if (is_front) n_front++; else n_back++;
    }
    std::cout << "[seam_brep] OCC faces: " << occ_face_is_front.size()
              << " (Z+ front: " << n_front << ", other: " << n_back << ")" << std::endl;

    // Map original triangle -> OCC face ID (from first vertex of each triangle)
    std::vector<int> orig_tri_face_id(orig_nf);
    for (int fi = 0; fi < orig_nf; ++fi) {
        orig_tri_face_id[fi] = static_cast<int>(orig_face_ids(orig_F(fi, 0)));
    }

    // Build position -> OCC face IDs set (to map welded vertices to their OCC faces)
    struct PosKey {
        int64_t x, y, z;
        bool operator<(const PosKey& o) const {
            if (x != o.x) return x < o.x; if (y != o.y) return y < o.y; return z < o.z;
        }
    };
    auto make_key = [](double px, double py, double pz) -> PosKey {
        return {(int64_t)std::round(px*1e6), (int64_t)std::round(py*1e6), (int64_t)std::round(pz*1e6)};
    };

    // For each position, collect the OCC face IDs and their front/back status
    std::map<PosKey, std::set<int>> pos_to_face_ids;
    for (int i = 0; i < orig_nv; ++i) {
        auto key = make_key(orig_V(i,0), orig_V(i,1), orig_V(i,2));
        pos_to_face_ids[key].insert(static_cast<int>(orig_face_ids(i)));
    }

    // A position is a B-Rep silhouette vertex if it touches both Z+ and non-Z+ OCC faces
    std::set<PosKey> silhouette_positions;
    for (auto& [key, fids] : pos_to_face_ids) {
        if (fids.size() < 2) continue;
        bool has_front = false, has_back = false;
        for (int fid : fids) {
            if (occ_face_is_front[fid]) has_front = true;
            else has_back = true;
        }
        if (has_front && has_back) silhouette_positions.insert(key);
    }

    std::cout << "[seam_brep] Silhouette positions (Z+ meets non-Z+): "
              << silhouette_positions.size() << std::endl;

    if (silhouette_positions.size() < 3) {
        std::cout << "[seam_brep] Not enough silhouette positions, falling back to BFS" << std::endl;
        return cut_to_disk(input_mesh);
    }

    // Map welded mesh vertices to silhouette positions
    std::set<VD> seam_vertices;
    for (auto v : input_mesh.vertices()) {
        auto pt = input_mesh.point(v);
        auto key = make_key(pt.x(), pt.y(), pt.z());
        if (silhouette_positions.count(key)) seam_vertices.insert(v);
    }

    // Find welded mesh edges where BOTH endpoints are silhouette vertices
    std::set<ED> seam_edges;
    for (auto e : input_mesh.edges()) {
        VD v0 = input_mesh.vertex(e, 0);
        VD v1 = input_mesh.vertex(e, 1);
        if (seam_vertices.count(v0) && seam_vertices.count(v1)) {
            seam_edges.insert(e);
        }
    }

    std::cout << "[seam_brep] Seam edges: " << seam_edges.size() << std::endl;

    if (seam_edges.empty()) {
        std::cout << "[seam_brep] No seam edges, falling back to BFS" << std::endl;
        return cut_to_disk(input_mesh);
    }

    // Find the longest connected path among seam edges
    std::map<VD, std::vector<VD>> seam_adj;
    for (auto e : seam_edges) {
        VD v0 = input_mesh.vertex(e, 0);
        VD v1 = input_mesh.vertex(e, 1);
        seam_adj[v0].push_back(v1);
        seam_adj[v1].push_back(v0);
    }

    std::set<VD> visited;
    std::vector<VD> best_path;
    for (auto& [start, _] : seam_adj) {
        if (visited.count(start)) continue;
        std::vector<VD> path;
        VD cur = start;
        VD prev_v = SurfaceMesh::null_vertex();
        while (true) {
            visited.insert(cur);
            path.push_back(cur);
            VD next = SurfaceMesh::null_vertex();
            for (VD nb : seam_adj[cur]) {
                if (nb != prev_v && !visited.count(nb)) { next = nb; break; }
            }
            if (next == SurfaceMesh::null_vertex()) break;
            prev_v = cur;
            cur = next;
        }
        if (path.size() > best_path.size()) best_path = path;
    }

    std::cout << "[seam_brep] Best seam path: " << best_path.size() << " vertices" << std::endl;

    if (best_path.size() < 3) {
        std::cout << "[seam_brep] Path too short, falling back to BFS" << std::endl;
        return cut_to_disk(input_mesh);
    }

    // The silhouette path is built from edges where both endpoints are silhouette
    // vertices. Verify consecutive vertices are connected by mesh edges.
    // If not, bridge gaps with Dijkstra segments.
    std::vector<VD> valid_path;
    valid_path.push_back(best_path[0]);
    for (size_t i = 1; i < best_path.size(); ++i) {
        ED e = find_edge(input_mesh, best_path[i-1], best_path[i]);
        if (e != SurfaceMesh::null_edge()) {
            valid_path.push_back(best_path[i]);
        } else {
            // Bridge the gap with Dijkstra
            auto bridge = dijkstra_path(input_mesh, best_path[i-1], best_path[i]);
            for (size_t j = 1; j < bridge.size(); ++j)
                valid_path.push_back(bridge[j]);
        }
    }

    std::cout << "[seam_brep] Cutting along silhouette path: " << valid_path.size()
              << " vertices (from " << best_path.size() << " silhouette vertices)" << std::endl;

    if (valid_path.size() < 2) {
        std::cout << "[seam_brep] Invalid path, falling back to BFS" << std::endl;
        return cut_to_disk(input_mesh);
    }

    return cut_along_path(input_mesh, valid_path);
}

} // namespace cgalparam
