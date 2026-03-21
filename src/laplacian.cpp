#include "meshparam/laplacian.h"
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <cmath>
#include <iostream>

namespace meshparam {

Eigen::SparseMatrix<double> cotangent_laplacian(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F)
{
    Eigen::SparseMatrix<double> L;
    igl::cotmatrix(V, F, L);
    L = -L;  // Now L is positive semi-definite, matching Eq. 5

    // Sanitize: degenerate triangles (zero area, near-zero angles) produce
    // NaN/Inf cotangent weights. Replace with 0 and fix the diagonal.
    int sanitized = 0;
    for (int k = 0; k < L.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
            if (std::isnan(it.valueRef()) || std::isinf(it.valueRef())) {
                it.valueRef() = 0.0;
                sanitized++;
            }
        }
    }

    if (sanitized > 0) {
        std::cout << "[laplacian] Sanitized " << sanitized
                  << " NaN/Inf entries (degenerate triangles)" << std::endl;
        // Rebuild diagonal: L_ii = -sum of off-diagonal entries in row i
        int n = static_cast<int>(L.rows());
        for (int i = 0; i < n; ++i) {
            double off_diag_sum = 0;
            for (Eigen::SparseMatrix<double>::InnerIterator it(L, i); it; ++it) {
                if (it.row() != i) off_diag_sum += it.value();
            }
            L.coeffRef(i, i) = -off_diag_sum;
        }
    }

    return L;
}

Eigen::SparseMatrix<double> mass_matrix(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F)
{
    Eigen::SparseMatrix<double> B;
    // Use barycentric mass matrix — more robust than Voronoi
    // for meshes with obtuse triangles or near-degenerate elements.
    igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_BARYCENTRIC, B);

    // Ensure no zero/NaN/Inf masses (degenerate triangles)
    int fixed = 0;
    double min_mass = 1e30;
    for (int k = 0; k < B.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(B, k); it; ++it) {
            if (it.row() == it.col() && std::isfinite(it.value()) && it.value() > 0)
                min_mass = std::min(min_mass, it.value());
        }
    }
    if (min_mass > 1e29) min_mass = 1e-10; // fallback
    for (int k = 0; k < B.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(B, k); it; ++it) {
            if (it.row() == it.col() && (!std::isfinite(it.value()) || it.value() <= 0)) {
                it.valueRef() = min_mass;
                fixed++;
            }
        }
    }
    if (fixed > 0)
        std::cout << "[mass] Fixed " << fixed << " zero/NaN mass entries" << std::endl;

    return B;
}

} // namespace meshparam
