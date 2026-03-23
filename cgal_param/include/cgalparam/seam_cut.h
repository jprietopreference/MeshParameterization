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

/// Cut along a specific vertex path.
SeamCutResult cut_along_path(const SurfaceMesh& input_mesh,
                             const std::vector<SurfaceMesh::Vertex_index>& path);

/// Cut along B-Rep face boundaries where one OCC face is Z+ (front) and
/// the adjacent OCC face is not Z+ (perpendicular or Z-). Uses per-vertex
/// OCC face IDs from the tessellator (_FACE_ID attribute).
///
/// @param mesh  Closed surface mesh (welded for parameterization)
/// @param orig_V  Original split-vertex positions (n_orig x 3)
/// @param orig_N  Original split-vertex normals (n_orig x 3)
/// @param orig_F  Original split-vertex faces (m x 3)
/// @param orig_face_ids  Per-vertex OCC face ID (n_orig x 1)
/// @param z_threshold  Z-component threshold below which an OCC face is
///        considered "back-facing". Default -0.1: only strictly backward faces
///        are on the "back" side of the seam. Front + perpendicular stay together.
/// @returns Cut mesh, or fallback to BFS if no suitable boundary found
SeamCutResult cut_brep_silhouette(const SurfaceMesh& mesh,
                                   const Eigen::MatrixXd& orig_V,
                                   const Eigen::MatrixXd& orig_N,
                                   const Eigen::MatrixXi& orig_F,
                                   const Eigen::VectorXd& orig_face_ids,
                                   double z_threshold = -0.1);

} // namespace cgalparam
