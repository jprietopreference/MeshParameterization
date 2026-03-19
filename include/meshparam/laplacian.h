#pragma once

#include <Eigen/Sparse>
#include <Eigen/Core>

namespace meshparam {

/// Build the cotangent Laplace-Beltrami matrix L (Eq. 5 in paper).
///   L_ij = (cot α_ij + cot β_ij) / 2   for edges
///   L_ii = -Σ_k L_ik
/// Returns n×n sparse symmetric matrix.
Eigen::SparseMatrix<double> cotangent_laplacian(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F);

/// Build the mass matrix B (Eq. 6 in paper).
///   B_ij = (|t1| + |t2|) / 12   for edges with adjacent triangles t1, t2
///   B_ii = Σ_k B_ik
/// Returns n×n sparse symmetric matrix.
Eigen::SparseMatrix<double> mass_matrix(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F);

} // namespace meshparam
