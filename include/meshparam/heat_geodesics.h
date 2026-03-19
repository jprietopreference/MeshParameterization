#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <memory>

namespace meshparam {

/// Prefactored system for heat-based geodesic computation.
/// Stores the Cholesky factorizations of (Δt·L + B) and L
/// so they can be reused for every source vertex.
struct HeatGeodesicSolver {
    using CholeskySolver = Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>;

    /// Prefactored system matrix (Δt·L + B) for heat step
    std::unique_ptr<CholeskySolver> heat_solver;
    /// Prefactored Laplacian L for Poisson (geodesic) step
    std::unique_ptr<CholeskySolver> poisson_solver;

    /// Stored matrices
    Eigen::SparseMatrix<double> L;  // Laplace-Beltrami
    Eigen::SparseMatrix<double> B;  // Mass matrix

    /// Mesh data needed for gradient/divergence
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    double dt;  // time step
    int n;      // number of vertices
};

/// Build and prefactor the heat geodesic solver (Steps 3-4 of paper).
/// dt = max edge length, T = dt (Eq. 7).
HeatGeodesicSolver build_heat_solver(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::SparseMatrix<double>& L,
    const Eigen::SparseMatrix<double>& B);

/// Compute geodesic distances from a single source vertex (Steps 5-7).
/// Returns n×1 vector of distances from source to all vertices.
Eigen::VectorXd geodesic_from_vertex(
    const HeatGeodesicSolver& solver,
    int source_vertex);

/// Compute the full n×n geodesic distance matrix G.
/// G_ij = g(x_i, x_j).
Eigen::MatrixXd compute_geodesic_matrix(
    const HeatGeodesicSolver& solver);

} // namespace meshparam
