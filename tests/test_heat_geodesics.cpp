#include <gtest/gtest.h>
#include "meshparam/laplacian.h"
#include "meshparam/heat_geodesics.h"
#include <Eigen/Core>
#include <cmath>

namespace {

// Flat rectangular mesh (4x2 grid = 6 vertices, 4 triangles)
// Good for testing geodesics since we know exact distances.
//
//  v3---v4---v5
//  |  / |  / |
//  v0---v1---v2
//
void make_flat_grid(Eigen::MatrixXd& V, Eigen::MatrixXi& F) {
    V.resize(6, 3);
    V << 0, 0, 0,
         1, 0, 0,
         2, 0, 0,
         0, 1, 0,
         1, 1, 0,
         2, 1, 0;

    F.resize(4, 3);
    F << 0, 1, 3,
         1, 4, 3,
         1, 2, 4,
         2, 5, 4;
}

} // anonymous namespace

TEST(HeatGeodesics, SolverBuilds) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    make_flat_grid(V, F);

    auto L = meshparam::cotangent_laplacian(V, F);
    auto B = meshparam::mass_matrix(V, F);

    EXPECT_NO_THROW({
        auto solver = meshparam::build_heat_solver(V, F, L, B);
        EXPECT_EQ(solver.n, 6);
        EXPECT_GT(solver.dt, 0.0);
    });
}

TEST(HeatGeodesics, DistanceSelfIsZero) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    make_flat_grid(V, F);

    auto L = meshparam::cotangent_laplacian(V, F);
    auto B = meshparam::mass_matrix(V, F);
    auto solver = meshparam::build_heat_solver(V, F, L, B);

    auto dists = meshparam::geodesic_from_vertex(solver, 0);

    // Distance from vertex to itself should be 0
    EXPECT_NEAR(dists(0), 0.0, 1e-10);
}

TEST(HeatGeodesics, DistancesNonNegative) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    make_flat_grid(V, F);

    auto L = meshparam::cotangent_laplacian(V, F);
    auto B = meshparam::mass_matrix(V, F);
    auto solver = meshparam::build_heat_solver(V, F, L, B);

    auto dists = meshparam::geodesic_from_vertex(solver, 0);

    for (int i = 0; i < dists.size(); ++i) {
        EXPECT_GE(dists(i), 0.0);
    }
}

TEST(HeatGeodesics, DistanceMonotonicity) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    make_flat_grid(V, F);

    auto L = meshparam::cotangent_laplacian(V, F);
    auto B = meshparam::mass_matrix(V, F);
    auto solver = meshparam::build_heat_solver(V, F, L, B);

    auto dists = meshparam::geodesic_from_vertex(solver, 0);

    // On a flat grid from corner (0,0):
    // v0 at (0,0), v1 at (1,0), v2 at (2,0)
    // Distance should increase: d(v0) < d(v1) < d(v2)
    EXPECT_LT(dists(0), dists(1));
    EXPECT_LT(dists(1), dists(2));
}

TEST(HeatGeodesics, MatrixSymmetric) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    make_flat_grid(V, F);

    auto L = meshparam::cotangent_laplacian(V, F);
    auto B = meshparam::mass_matrix(V, F);
    auto solver = meshparam::build_heat_solver(V, F, L, B);

    auto G = meshparam::compute_geodesic_matrix(solver);

    // Should be symmetric after symmetrization in compute_geodesic_matrix
    // The heat method introduces small asymmetries that are averaged out
    for (int i = 0; i < G.rows(); ++i) {
        for (int j = 0; j < G.cols(); ++j) {
            EXPECT_NEAR(G(i, j), G(j, i), 0.5)
                << "Asymmetry too large at (" << i << "," << j << ")";
        }
    }
}
