#include "meshparam/mds.h"
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

// Use Spectra for native (faster for large n), Eigen for WASM (more stable)
#ifndef __EMSCRIPTEN__
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>
#endif

namespace meshparam {

namespace {

// Extract top-2 eigenpairs from symmetric matrix C.
// Returns eigenvalues (2x1) and eigenvectors (nx2), sorted descending.
void top2_eigen(const Eigen::MatrixXd& C, Eigen::VectorXd& vals, Eigen::MatrixXd& vecs) {
    int n = static_cast<int>(C.rows());

#ifdef __EMSCRIPTEN__
    // Use Eigen's full eigendecomposition — stable in WASM, no stack issues
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(C);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("MDS eigendecomposition failed");
    // Eigen returns ascending order; top 2 are last
    vals.resize(2);
    vecs.resize(n, 2);
    vals(0) = solver.eigenvalues()(n - 1);
    vals(1) = solver.eigenvalues()(n - 2);
    vecs.col(0) = solver.eigenvectors().col(n - 1);
    vecs.col(1) = solver.eigenvectors().col(n - 2);
#else
    // Use Spectra for efficient top-k extraction on native
    Spectra::DenseSymMatProd<double> op(C);
    int nev = 2;
    int ncv = std::min(n, std::max(2 * nev + 1, 20));
    Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigs(op, nev, ncv);
    eigs.init();
    eigs.compute(Spectra::SortRule::LargestAlge);
    if (eigs.info() != Spectra::CompInfo::Successful)
        throw std::runtime_error("MDS eigendecomposition failed");
    vals = eigs.eigenvalues();
    vecs = eigs.eigenvectors();
#endif
}

} // anonymous namespace

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

    // Step 9-10: Eigendecompose C → top 2, then Φ = [√λ1·V1, √λ2·V2]
    Eigen::VectorXd eigenvalues;
    Eigen::MatrixXd eigenvectors;
    top2_eigen(C, eigenvalues, eigenvectors);

    Eigen::MatrixXd UV(n, 2);
    UV.col(0) = std::sqrt(std::max(eigenvalues(0), 0.0)) * eigenvectors.col(0);
    UV.col(1) = std::sqrt(std::max(eigenvalues(1), 0.0)) * eigenvectors.col(1);

    return UV;
}

Eigen::MatrixXd weighted_mds(const Eigen::MatrixXd& G, const Eigen::VectorXd& W) {
    int n = static_cast<int>(G.rows());

    if (n < 3) {
        throw std::runtime_error("MDS requires at least 3 vertices");
    }

    // G² = element-wise squared distances
    Eigen::MatrixXd G2 = G.array().square().matrix();

    // Apply weights: scale each row and column of G² by sqrt(wi * wj)
    // This makes high-weight vertex pairs dominate the centering matrix,
    // so MDS preserves their pairwise distances more faithfully.
    Eigen::VectorXd Wsqrt = W.array().sqrt();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            G2(i, j) *= Wsqrt(i) * Wsqrt(j);
        }
    }

    // Double-centering: C = -½ (In - 1/n Jn) G² (In - 1/n Jn)
    Eigen::VectorXd row_means = G2.rowwise().mean();
    double grand_mean = G2.mean();

    Eigen::MatrixXd C(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C(i, j) = -0.5 * (G2(i, j) - row_means(i) - row_means(j) + grand_mean);
        }
    }

    // Eigendecompose → top 2, then Φ = [√λ1·V1, √λ2·V2]
    Eigen::VectorXd eigenvalues;
    Eigen::MatrixXd eigenvectors;
    top2_eigen(C, eigenvalues, eigenvectors);

    Eigen::MatrixXd UV(n, 2);
    UV.col(0) = std::sqrt(std::max(eigenvalues(0), 0.0)) * eigenvectors.col(0);
    UV.col(1) = std::sqrt(std::max(eigenvalues(1), 0.0)) * eigenvectors.col(1);

    return UV;
}

Eigen::VectorXd compute_view_weights(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::Vector3d& view_dir,
    double floor) {

    int n = static_cast<int>(V.rows());
    int m = static_cast<int>(F.rows());

    Eigen::Vector3d vd = view_dir.normalized();

    // Accumulate area-weighted face normal dot products per vertex
    Eigen::VectorXd weight_sum = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd area_sum = Eigen::VectorXd::Zero(n);

    for (int f = 0; f < m; ++f) {
        int i0 = F(f, 0), i1 = F(f, 1), i2 = F(f, 2);
        Eigen::Vector3d e1 = V.row(i1) - V.row(i0);
        Eigen::Vector3d e2 = V.row(i2) - V.row(i0);
        Eigen::Vector3d fn = e1.cross(e2);
        double area = fn.norm() * 0.5;
        if (area < 1e-15) continue;
        fn.normalize();

        // dot: 1 = facing viewer, -1 = facing away
        double dot = fn.dot(vd);
        // Map to [0, 1]: facing viewer = 1, perpendicular = 0.5, away = 0
        double importance = std::max(0.0, (dot + 1.0) * 0.5);

        // Distribute to vertices weighted by face area
        weight_sum(i0) += importance * area;
        weight_sum(i1) += importance * area;
        weight_sum(i2) += importance * area;
        area_sum(i0) += area;
        area_sum(i1) += area;
        area_sum(i2) += area;
    }

    // Normalize: per-vertex importance = weighted average of adjacent face importances
    Eigen::VectorXd W(n);
    for (int i = 0; i < n; ++i) {
        double raw = (area_sum(i) > 1e-15) ? weight_sum(i) / area_sum(i) : 0.5;
        // Apply floor so back-facing vertices still have some weight
        W(i) = floor + (1.0 - floor) * raw;
    }

    return W;
}

} // namespace meshparam
