// Debug: analyze geodesic matrix and eigendecomposition for failing meshes
#include "meshparam/gltf_io.h"
#include "meshparam/laplacian.h"
#include "meshparam/heat_geodesics.h"
#include "meshparam/mds.h"
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: debug_eigen <input.glb>" << std::endl;
        return 1;
    }

    auto mesh = meshparam::load_gltf(argv[1]);
    int n = mesh.num_vertices();
    std::cout << "Mesh: " << n << " vertices, " << mesh.num_faces() << " faces" << std::endl;

    // Build Laplacian
    auto L = meshparam::cotangent_laplacian(mesh.V, mesh.F);
    auto B = meshparam::mass_matrix(mesh.V, mesh.F);

    // Check for NaN/Inf in Laplacian
    int nan_L = 0, inf_L = 0;
    for (int k = 0; k < L.outerSize(); ++k)
        for (Eigen::SparseMatrix<double>::InnerIterator it(L, k); it; ++it) {
            if (std::isnan(it.value())) nan_L++;
            if (std::isinf(it.value())) inf_L++;
        }
    std::cout << "Laplacian: " << L.nonZeros() << " nonzeros, NaN=" << nan_L << " Inf=" << inf_L << std::endl;

    // Build heat solver and compute geodesic matrix
    auto solver = meshparam::build_heat_solver(mesh.V, mesh.F, L, B);
    std::cout << "Computing geodesic matrix..." << std::endl;
    auto G = meshparam::compute_geodesic_matrix(solver);

    // Analyze geodesic matrix
    int nan_G = 0, inf_G = 0, neg_G = 0, zero_diag = 0;
    double min_G = 1e30, max_G = -1e30;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double v = G(i, j);
            if (std::isnan(v)) nan_G++;
            if (std::isinf(v)) inf_G++;
            if (v < 0 && i != j) neg_G++;
            if (i != j) {
                min_G = std::min(min_G, v);
                max_G = std::max(max_G, v);
            }
        }
        if (std::abs(G(i, i)) > 1e-10) zero_diag++;
    }
    std::cout << "Geodesic matrix: " << n << "x" << n << std::endl;
    std::cout << "  NaN=" << nan_G << " Inf=" << inf_G << " Negative=" << neg_G << std::endl;
    std::cout << "  Non-zero diagonal=" << zero_diag << std::endl;
    std::cout << "  Range (off-diag): [" << min_G << ", " << max_G << "]" << std::endl;

    // Symmetry check
    double max_asym = 0;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            max_asym = std::max(max_asym, std::abs(G(i,j) - G(j,i)));
    std::cout << "  Max asymmetry: " << max_asym << std::endl;

    // Double-center and check eigenvalues
    Eigen::MatrixXd G2 = G.array().square().matrix();
    Eigen::VectorXd row_means = G2.rowwise().mean();
    double grand_mean = G2.mean();
    Eigen::MatrixXd C(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C(i, j) = -0.5 * (G2(i, j) - row_means(i) - row_means(j) + grand_mean);

    // Check C for NaN/Inf
    int nan_C = 0, inf_C = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            if (std::isnan(C(i,j))) nan_C++;
            if (std::isinf(C(i,j))) inf_C++;
        }
    std::cout << "Centering matrix C: NaN=" << nan_C << " Inf=" << inf_C << std::endl;

    if (nan_C > 0 || inf_C > 0) {
        std::cout << "ABORT: C matrix has NaN/Inf, eigen will fail" << std::endl;
        return 1;
    }

    // Try Eigen full decomposition
    std::cout << "Running SelfAdjointEigenSolver..." << std::endl;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigsolver(C);
    if (eigsolver.info() != Eigen::Success) {
        std::cout << "FAILED: Eigen decomposition failed" << std::endl;
        return 1;
    }

    auto evals = eigsolver.eigenvalues();
    std::cout << "Top 5 eigenvalues: ";
    for (int i = n - 1; i >= std::max(0, n - 5); --i)
        std::cout << evals(i) << " ";
    std::cout << std::endl;

    std::cout << "Bottom 5 eigenvalues: ";
    for (int i = 0; i < std::min(5, n); ++i)
        std::cout << evals(i) << " ";
    std::cout << std::endl;

    double lambda1 = evals(n - 1);
    double lambda2 = evals(n - 2);
    std::cout << "UV eigenvalues: " << lambda1 << ", " << lambda2 << std::endl;
    std::cout << "Condition: " << lambda1 / std::max(std::abs(evals(0)), 1e-15) << std::endl;

    // Now try Spectra
    std::cout << "\nTrying Spectra..." << std::endl;
    try {
        auto UV = meshparam::classical_mds(G);
        std::cout << "Spectra SUCCESS, UV range: ["
                  << UV.col(0).minCoeff() << "," << UV.col(0).maxCoeff() << "] x ["
                  << UV.col(1).minCoeff() << "," << UV.col(1).maxCoeff() << "]" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Spectra FAILED: " << e.what() << std::endl;
    }

    return 0;
}
