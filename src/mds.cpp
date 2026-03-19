#include "meshparam/mds.h"
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

namespace meshparam {

Eigen::MatrixXd classical_mds(const Eigen::MatrixXd& G) {
    int n = static_cast<int>(G.rows());

    if (n < 3) {
        throw std::runtime_error("MDS requires at least 3 vertices");
    }

    // Step 8: G² = element-wise squared distances
    Eigen::MatrixXd G2 = G.array().square().matrix();

    // Step 8: C = -½ (In - 1/n Jn) G² (In - 1/n Jn)   (Eq. 12)
    // This is the double-centering of G²
    Eigen::VectorXd row_means = G2.rowwise().mean();
    double grand_mean = G2.mean();

    Eigen::MatrixXd C(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C(i, j) = -0.5 * (G2(i, j) - row_means(i) - row_means(j) + grand_mean);
        }
    }

    // Step 9: Eigendecompose C → top 2 eigenvalues and eigenvectors
    // Use Spectra for efficient computation of just the top eigenvalues
    Spectra::DenseSymMatProd<double> op(C);

    // Request top 2 eigenvalues
    int nev = 2;
    int ncv = std::min(n, std::max(2 * nev + 1, 20));
    Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigs(op, nev, ncv);

    eigs.init();
    int nconv = eigs.compute(Spectra::SortRule::LargestAlge);

    if (eigs.info() != Spectra::CompInfo::Successful) {
        throw std::runtime_error("MDS eigendecomposition failed");
    }

    Eigen::VectorXd eigenvalues = eigs.eigenvalues();
    Eigen::MatrixXd eigenvectors = eigs.eigenvectors();

    // Step 10: Φ = [√λ1·V1, √λ2·V2]   (Eq. 14)
    Eigen::MatrixXd UV(n, 2);

    double lambda1 = std::max(eigenvalues(0), 0.0);
    double lambda2 = std::max(eigenvalues(1), 0.0);

    UV.col(0) = std::sqrt(lambda1) * eigenvectors.col(0);
    UV.col(1) = std::sqrt(lambda2) * eigenvectors.col(1);

    return UV;
}

} // namespace meshparam
