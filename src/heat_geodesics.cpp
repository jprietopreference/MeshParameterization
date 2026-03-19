#include "meshparam/heat_geodesics.h"
#include <igl/grad.h>
#include <igl/doublearea.h>
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace meshparam {

HeatGeodesicSolver build_heat_solver(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::SparseMatrix<double>& L,
    const Eigen::SparseMatrix<double>& B)
{
    HeatGeodesicSolver solver;
    solver.V = V;
    solver.F = F;
    solver.L = L;
    solver.B = B;
    solver.n = static_cast<int>(V.rows());

    // Step 3: Δt = mean_edge^2 (Crane et al. recommend h^2 where h is mean edge)
    double sum_edge = 0.0;
    int edge_count = 0;
    for (int f = 0; f < F.rows(); ++f) {
        for (int e = 0; e < 3; ++e) {
            int i = F(f, e);
            int j = F(f, (e + 1) % 3);
            sum_edge += (V.row(i) - V.row(j)).norm();
            edge_count++;
        }
    }
    double mean_edge = sum_edge / edge_count;
    solver.dt = mean_edge * mean_edge;  // h^2 as recommended by Crane et al.

    // Step 4: Prefactor (Δt·L + B) for heat step
    double eps = 1e-8;
    if (L.rows() != B.rows() || L.cols() != B.cols()) {
        throw std::runtime_error(
            "Laplacian and mass matrix dimension mismatch: L is " +
            std::to_string(L.rows()) + "x" + std::to_string(L.cols()) +
            ", B is " + std::to_string(B.rows()) + "x" + std::to_string(B.cols()) +
            ", n_vertices=" + std::to_string(V.rows()));
    }
    Eigen::SparseMatrix<double> A = solver.dt * L + B;
    for (int i = 0; i < solver.n; ++i) {
        A.coeffRef(i, i) += eps;
    }
    solver.heat_solver = std::make_unique<HeatGeodesicSolver::CholeskySolver>();
    solver.heat_solver->compute(A);
    if (solver.heat_solver->info() != Eigen::Success) {
        throw std::runtime_error("Heat solver factorization failed");
    }

    // Prefactor L for Poisson (geodesic) step
    Eigen::SparseMatrix<double> L_reg = L;
    for (int i = 0; i < solver.n; ++i) {
        L_reg.coeffRef(i, i) += eps;
    }
    solver.poisson_solver = std::make_unique<HeatGeodesicSolver::CholeskySolver>();
    solver.poisson_solver->compute(L_reg);
    if (solver.poisson_solver->info() != Eigen::Success) {
        throw std::runtime_error("Poisson solver factorization failed");
    }

    return solver;
}

Eigen::VectorXd geodesic_from_vertex(
    const HeatGeodesicSolver& solver,
    int source_vertex)
{
    int n = solver.n;
    const auto& V = solver.V;
    const auto& F = solver.F;
    int m = static_cast<int>(F.rows());

    // ---- Step 1: Solve heat equation ----
    // (Δt·L + B) u = B·δ_source
    Eigen::VectorXd delta = Eigen::VectorXd::Zero(n);
    delta(source_vertex) = 1.0;
    Eigen::VectorXd rhs = solver.B * delta;

    Eigen::VectorXd u = solver.heat_solver->solve(rhs);

    // ---- Step 2: Compute per-face gradient, then normalize to get X = -∇u/|∇u| ----
    // Per-face gradient using libigl (returns 3m x n sparse matrix)
    Eigen::SparseMatrix<double> Grad;
    igl::grad(V, F, Grad);

    Eigen::VectorXd grad_u = Grad * u;  // 3m x 1: [dx; dy; dz] stacked

    // Normalize per-face: X_f = -grad_u_f / |grad_u_f|
    Eigen::MatrixXd X(m, 3);  // per-face normalized vector field
    for (int f = 0; f < m; ++f) {
        Eigen::Vector3d g(grad_u(f), grad_u(f + m), grad_u(f + 2*m));
        double norm = g.norm();
        if (norm > 1e-12) {
            X.row(f) = (-g / norm).transpose();
        } else {
            X.row(f).setZero();
        }
    }

    // ---- Step 3: Compute integrated divergence of X ----
    // Crane et al. 2013, Eq. 8:
    // (div X)_i = (1/2) Σ_{(i,j,k) triangle containing i}
    //             [ cot(θ_j) * (p_k - p_i) · X_f + cot(θ_k) * (p_j - p_i) · X_f ]
    // where θ_j is the angle at vertex j in triangle (i,j,k)

    Eigen::VectorXd div = Eigen::VectorXd::Zero(n);

    for (int fi = 0; fi < m; ++fi) {
        // Triangle vertices
        int vi[3] = {F(fi, 0), F(fi, 1), F(fi, 2)};
        Eigen::Vector3d p[3] = {V.row(vi[0]), V.row(vi[1]), V.row(vi[2])};
        Eigen::Vector3d Xf = X.row(fi);

        // Edges opposite to each vertex:
        // e[0] = p[2] - p[1]  (opposite vi[0])
        // e[1] = p[0] - p[2]  (opposite vi[1])
        // e[2] = p[1] - p[0]  (opposite vi[2])
        Eigen::Vector3d e[3] = {p[2] - p[1], p[0] - p[2], p[1] - p[0]};

        // Cotangents of angles at each vertex
        // cot(angle at vi[k]) = (e_from_k_to_j . e_from_k_to_l) / |e_from_k_to_j × e_from_k_to_l|
        // Equivalently: for vertex vi[0], the angle is between edges (p1-p0) and (p2-p0)
        double area2 = (p[1] - p[0]).cross(p[2] - p[0]).norm();
        if (area2 < 1e-16) continue;

        double cot[3];
        cot[0] = (p[1] - p[0]).dot(p[2] - p[0]) / area2;
        cot[1] = (p[0] - p[1]).dot(p[2] - p[1]) / area2;
        cot[2] = (p[0] - p[2]).dot(p[1] - p[2]) / area2;

        // For each vertex vi[k] in this triangle:
        // The two other vertices are vi[(k+1)%3] = j and vi[(k+2)%3] = l
        // Contribution: 0.5 * [ cot(θ_j) * (p_l - p_k) · X + cot(θ_l) * (p_j - p_k) · X ]
        for (int k = 0; k < 3; ++k) {
            int j = (k + 1) % 3;
            int l = (k + 2) % 3;
            double contrib = 0.5 * (
                cot[j] * (p[l] - p[k]).dot(Xf) +
                cot[l] * (p[j] - p[k]).dot(Xf)
            );
            div(vi[k]) += contrib;
        }
    }

    // ---- Step 4: Solve Poisson equation L·φ = div(X) ----
    Eigen::VectorXd phi = solver.poisson_solver->solve(div);

    // Shift so source vertex has distance 0
    // The Poisson solution is unique up to a constant.
    // Subtract the value at source so distance from source to itself = 0.
    phi.array() -= phi(source_vertex);

    // The distances should be non-negative. If most are negative,
    // the sign convention is wrong — flip.
    int n_neg = (phi.array() < 0).count();
    if (n_neg > n / 2) {
        phi = -phi;
        phi.array() -= phi(source_vertex);
    }

    // Clamp small negative values from numerical noise
    phi = phi.array().max(0.0);

    return phi;
}

Eigen::MatrixXd compute_geodesic_matrix(const HeatGeodesicSolver& solver) {
    int n = solver.n;
    Eigen::MatrixXd G(n, n);

    for (int i = 0; i < n; ++i) {
        G.row(i) = geodesic_from_vertex(solver, i).transpose();
    }

    // Symmetrize
    G = (G + G.transpose()) / 2.0;
    return G;
}

} // namespace meshparam
