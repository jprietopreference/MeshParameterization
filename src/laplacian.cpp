#include "meshparam/laplacian.h"
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>

namespace meshparam {

Eigen::SparseMatrix<double> cotangent_laplacian(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F)
{
    Eigen::SparseMatrix<double> L;
    // libigl's cotmatrix computes the standard cotangent Laplacian.
    // Convention: L is negative semi-definite (L_ii = -Σ_j L_ij).
    // The paper uses the positive semi-definite convention, so we negate.
    igl::cotmatrix(V, F, L);
    L = -L;  // Now L is positive semi-definite, matching Eq. 5
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
    return B;
}

} // namespace meshparam
