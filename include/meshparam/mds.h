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

} // namespace meshparam
