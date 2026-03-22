#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>
#include <string>
#include <cstdint>

namespace meshparam {

/// Triangle mesh representation compatible with the paper's notation.
/// M = (X, T) where X = vertices, T = triangles.
struct TriMesh {
    /// n x 3 vertex positions (X in the paper)
    Eigen::MatrixXd V;
    /// m x 3 triangle indices (T in the paper)
    Eigen::MatrixXi F;

    /// Optional: existing UV coordinates (n x 2)
    Eigen::MatrixXd UV;
    /// Optional: per-vertex normals (n x 3)
    Eigen::MatrixXd N;
    /// Optional: per-vertex seam flag (n x 1, 0.0 or 1.0)
    Eigen::VectorXd seam;
    /// Optional: per-vertex B-Rep face ID (n x 1)
    Eigen::VectorXd face_ids;

    int num_vertices() const { return static_cast<int>(V.rows()); }
    int num_faces() const { return static_cast<int>(F.rows()); }
    bool has_uvs() const { return UV.rows() == V.rows(); }
    bool has_normals() const { return N.rows() == V.rows(); }
    bool has_seam() const { return seam.rows() == V.rows(); }
    bool has_face_ids() const { return face_ids.rows() == V.rows(); }

    /// Compute per-vertex normals (area-weighted average of face normals)
    void compute_normals();

    /// Detect if mesh has boundary (holes or concavities)
    bool has_boundary() const;

    /// Compute the longest edge length (used for Δt)
    double max_edge_length() const;

    /// Get boundary loop vertex indices
    std::vector<std::vector<int>> boundary_loops() const;
};

} // namespace meshparam
