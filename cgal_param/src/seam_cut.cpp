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

    // Step 3: Collect seam edges
    std::set<ED> seam_edges;
    for (size_t i = 0; i + 1 < path.size(); ++i) {
        ED e = find_edge(input_mesh, path[i], path[i+1]);
        if (e != SurfaceMesh::null_edge()) {
            seam_edges.insert(e);
        }
    }

    // Step 4: Build cut mesh by duplicating vertices along the seam.
    // For each seam vertex (except endpoints), create a duplicate.
    // Faces on one side of the seam use the original vertex,
    // faces on the other side use the duplicate.

    // Start with a copy
    result.cut_mesh = input_mesh;
    SurfaceMesh& sm = result.cut_mesh;

    // Initialize vertex map: identity
    result.vertex_map.resize(n_orig + path.size());  // room for duplicates
    for (int i = 0; i < n_orig; ++i) result.vertex_map[i] = i;

    // For each interior seam vertex, determine which faces are on each side.
    // We walk around the vertex and split at seam edges.
    // Vertices at the path interior (not endpoints) need duplication.

    // Identify which halfedges of seam edges to use for the "left" side
    std::set<VD> seam_vertices(path.begin(), path.end());

    // Use CGAL's Euler split operations along the seam path.
    // The approach: for each edge in the seam path, we split the edge
    // by inserting a new vertex, effectively creating a slit.
    // But this changes the mesh topology significantly.

    // Alternative simpler approach: rebuild the mesh from scratch,
    // duplicating seam vertices for faces on one side.

    // Determine face sides using BFS from seam edges.
    // Each seam edge has two adjacent faces. We assign "side 0" and "side 1"
    // by flood-filling from one side, stopping at seam edges.

    // Build face adjacency (excluding seam edges)
    std::map<face_descriptor, int> face_side;
    std::set<face_descriptor> visited;
    std::queue<face_descriptor> bfs_q;

    // Start from one face adjacent to the first seam edge
    ED first_seam = *seam_edges.begin();
    HD h0 = sm.halfedge(first_seam);
    face_descriptor seed_face = sm.face(h0);
    if (seed_face == SurfaceMesh::null_face()) {
        seed_face = sm.face(sm.opposite(h0));
    }

    face_side[seed_face] = 0;
    bfs_q.push(seed_face);
    visited.insert(seed_face);

    while (!bfs_q.empty()) {
        face_descriptor f = bfs_q.front(); bfs_q.pop();
        int side = face_side[f];

        for (auto h : sm.halfedges_around_face(sm.halfedge(f))) {
            ED e = sm.edge(h);
            face_descriptor neighbor = sm.face(sm.opposite(h));
            if (neighbor == SurfaceMesh::null_face()) continue;
            if (visited.count(neighbor)) continue;

            if (seam_edges.count(e)) {
                // Crossing a seam edge → flip side
                face_side[neighbor] = 1 - side;
            } else {
                face_side[neighbor] = side;
            }
            visited.insert(neighbor);
            bfs_q.push(neighbor);
        }
    }

    // Now rebuild the mesh. For seam vertices, create duplicates for side 1.
    // Endpoints of the path are NOT duplicated (they become the boundary endpoints).

    std::set<VD> interior_seam_verts(path.begin() + 1, path.end() - 1);

    // Map: (original_vertex, side) → new vertex index
    std::map<std::pair<int, int>, VD> vert_remap;

    SurfaceMesh new_mesh;
    // First, add all original vertices
    std::vector<VD> orig_to_new(n_orig);
    for (auto v : sm.vertices()) {
        int vi = static_cast<int>(v);
        orig_to_new[vi] = new_mesh.add_vertex(sm.point(v));
    }

    // Add duplicates for interior seam vertices (side 1)
    std::map<int, VD> seam_duplicates;  // original index → duplicate VD
    for (VD sv : interior_seam_verts) {
        int si = static_cast<int>(sv);
        VD dup = new_mesh.add_vertex(sm.point(sv));
        seam_duplicates[si] = dup;
        // Update vertex map
        int dup_idx = static_cast<int>(dup);
        if (dup_idx >= static_cast<int>(result.vertex_map.size())) {
            result.vertex_map.resize(dup_idx + 1);
        }
        result.vertex_map[dup_idx] = si;  // duplicate maps to original
    }

    // Add faces, remapping seam vertices for side 1
    for (auto f : sm.faces()) {
        int side = face_side.count(f) ? face_side[f] : 0;

        VD fv[3];
        int k = 0;
        for (auto v : sm.vertices_around_face(sm.halfedge(f))) {
            int vi = static_cast<int>(v);
            if (side == 1 && interior_seam_verts.count(v)) {
                fv[k] = seam_duplicates[vi];
            } else {
                fv[k] = orig_to_new[vi];
            }
            k++;
        }
        if (k == 3) {
            new_mesh.add_face(fv[0], fv[1], fv[2]);
        }
    }

    new_mesh.collect_garbage();
    result.cut_mesh = new_mesh;

    // Resize vertex map to actual size
    result.vertex_map.resize(result.cut_mesh.number_of_vertices());

    // Verify mesh validity
    if (!result.cut_mesh.is_valid(false)) {
        std::cerr << "[seam] WARNING: rebuilt mesh is invalid, falling back to face removal" << std::endl;
        result.cut_mesh = input_mesh;
        auto f_it = result.cut_mesh.faces_begin();
        CGAL::Euler::remove_face(result.cut_mesh.halfedge(*f_it), result.cut_mesh);
        result.cut_mesh.collect_garbage();
        result.vertex_map.resize(n_orig);
        for (int i = 0; i < n_orig; ++i) result.vertex_map[i] = i;
        return result;
    }

    // Verify we now have a boundary
    HD new_border = CGAL::Polygon_mesh_processing::longest_border(result.cut_mesh).first;
    if (new_border == SurfaceMesh::null_halfedge()) {
        std::cerr << "[seam] WARNING: cut did not create boundary, falling back to face removal"
                  << std::endl;
        result.cut_mesh = input_mesh;
        auto f_it = result.cut_mesh.faces_begin();
        CGAL::Euler::remove_face(result.cut_mesh.halfedge(*f_it), result.cut_mesh);
        result.cut_mesh.collect_garbage();
        result.vertex_map.resize(n_orig);
        for (int i = 0; i < n_orig; ++i) result.vertex_map[i] = i;
    } else {
        int new_V = static_cast<int>(result.cut_mesh.number_of_vertices());
        int new_F = static_cast<int>(result.cut_mesh.number_of_faces());
        std::cout << "[seam] Cut mesh: " << new_V << " verts, " << new_F << " faces"
                  << " (added " << (new_V - n_orig) << " seam vertices)" << std::endl;
    }

    return result;
}

} // namespace cgalparam
