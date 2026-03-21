#pragma once

#include <Eigen/Core>

namespace meshparam {

/// Classical Multidimensional Scaling (Section 2.4 of paper).
///
/// Given n×n geodesic distance matrix G, computes 2D coordinates Φ:
///   1. G² = element-wise squared distances
///   2. C = -½ (In - 1/n Jn) G² (In - 1/n Jn)   (Eq. 12)
///   3. Eigendecompose C → λ1, V1, λ2, V2
///   4. Φ = [√λ1·V1, √λ2·V2]                     (Eq. 14)
///
/// Returns n×2 matrix of UV coordinates.
Eigen::MatrixXd classical_mds(const Eigen::MatrixXd& G);

/// Weighted Classical MDS.
///
/// Same as classical_mds but applies per-vertex weights to the double-centering
/// matrix, so vertices with higher weight have more influence on the embedding
/// (their pairwise distances are preserved more faithfully).
///
/// @param G  n×n geodesic distance matrix
/// @param W  n×1 per-vertex weight vector (0 = ignore, 1 = normal importance)
/// @returns n×2 UV coordinates
Eigen::MatrixXd weighted_mds(const Eigen::MatrixXd& G, const Eigen::VectorXd& W);

/// Compute per-vertex view importance weights from mesh geometry.
///
/// For each vertex, averages the dot product of its adjacent face normals
/// with the view direction. Result is in [0, 1]: 1 = fully facing the viewer,
/// 0 = facing away. A minimum floor is applied so back-facing vertices aren't
/// completely ignored.
///
/// @param V  n×3 vertices
/// @param F  m×3 faces
/// @param view_dir  unit view direction vector
/// @param floor  minimum weight for back-facing vertices (default 0.2)
/// @returns n×1 weight vector
Eigen::VectorXd compute_view_weights(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::Vector3d& view_dir,
    double floor = 0.2);

} // namespace meshparam
