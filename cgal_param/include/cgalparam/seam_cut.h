#pragma once

#include "types.h"
#include <vector>

namespace cgalparam {

/// Cut a closed mesh along a geodesic seam to make it a topological disk.
///
/// Algorithm:
///   1. Find two "pole" vertices far apart on the surface (BFS diameter)
///   2. Compute shortest geodesic path between them (Dijkstra on edges)
///   3. For genus > 0, find additional cut loops to reduce to genus 0
///   4. Duplicate vertices along the seam path to create a real boundary
///
/// The result is a mesh with boundary suitable for parameterization.
/// Returns the modified mesh (vertices duplicated along the cut).

struct SeamCutResult {
    SurfaceMesh cut_mesh;
    /// Mapping from cut mesh vertex to original vertex index
    std::vector<int> vertex_map;
    /// Number of vertices in the original mesh
    int original_vertex_count;
};

/// Cut a closed mesh to create a topological disk.
/// If the mesh already has a boundary, returns it unchanged.
SeamCutResult cut_to_disk(const SurfaceMesh& mesh);

} // namespace cgalparam
