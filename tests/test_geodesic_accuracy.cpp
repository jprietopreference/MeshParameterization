#include <gtest/gtest.h>
#include "meshparam/laplacian.h"
#include "meshparam/heat_geodesics.h"
#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace {

/// Create a flat 10x10 grid (121 vertices, 200 triangles)
/// Vertex (i,j) is at position (i, j, 0) where i,j in [0..10]
/// Exact geodesic distance between (i1,j1) and (i2,j2) = sqrt((i1-i2)^2 + (j1-j2)^2)
void make_flat_grid_10x10(Eigen::MatrixXd& V, Eigen::MatrixXi& F) {
    int nx = 11, ny = 11;
    V.resize(nx * ny, 3);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            V.row(j * nx + i) << static_cast<double>(i), static_cast<double>(j), 0.0;
        }
    }

    F.resize(2 * (nx - 1) * (ny - 1), 3);
    int f = 0;
    for (int j = 0; j < ny - 1; ++j) {
        for (int i = 0; i < nx - 1; ++i) {
            int v00 = j * nx + i;
            int v10 = v00 + 1;
            int v01 = v00 + nx;
            int v11 = v01 + 1;
            F.row(f++) << v00, v10, v01;
            F.row(f++) << v10, v11, v01;
        }
    }
}

/// Get vertex index for grid position (i,j) on 11-wide grid
int grid_idx(int i, int j) { return j * 11 + i; }

} // anonymous namespace

TEST(GeodesicAccuracy, FlatGridCornerToCorner) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    make_flat_grid_10x10(V, F);

    auto L = meshparam::cotangent_laplacian(V, F);
    auto B = meshparam::mass_matrix(V, F);
    auto solver = meshparam::build_heat_solver(V, F, L, B);

    // Compute geodesics from corner (0,0)
    int src = grid_idx(0, 0);
    auto dists = meshparam::geodesic_from_vertex(solver, src);

    // Check against exact Euclidean distances on flat surface
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n  Heat geodesics vs exact (flat 10x10 grid, source at 0,0):\n";
    std::cout << "  Target      Exact     Heat     Error%\n";
    std::cout << "  ------      -----     ----     ------\n";

    struct TestPt { int i, j; const char* name; };
    TestPt pts[] = {
        {1, 0, "(1,0) "},  {5, 0, "(5,0) "},  {10, 0, "(10,0)"},
        {0, 1, "(0,1) "},  {0, 5, "(0,5) "},  {0, 10, "(0,10)"},
        {1, 1, "(1,1) "},  {5, 5, "(5,5) "},  {10, 10, "(10,10)"},
        {3, 4, "(3,4) "},  {7, 2, "(7,2) "},
    };

    double max_rel_err = 0;
    for (auto& pt : pts) {
        int idx = grid_idx(pt.i, pt.j);
        double exact = std::sqrt(pt.i * pt.i + pt.j * pt.j);
        double heat = dists(idx);
        double rel_err = (exact > 1e-10) ? std::abs(heat - exact) / exact * 100.0 : 0.0;
        max_rel_err = std::max(max_rel_err, rel_err);

        std::cout << "  " << pt.name
                  << "   " << std::setw(8) << exact
                  << "   " << std::setw(8) << heat
                  << "   " << std::setw(6) << rel_err << "%\n";
    }
    std::cout << "  Max relative error: " << max_rel_err << "%\n";

    // On a flat grid, heat geodesics should be within ~20% of exact
    // (the method is approximate; large errors indicate a bug)
    EXPECT_LT(max_rel_err, 30.0) << "Heat geodesics deviate too much from exact on flat grid";
}

TEST(GeodesicAccuracy, FlatGridSymmetry) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    make_flat_grid_10x10(V, F);

    auto L = meshparam::cotangent_laplacian(V, F);
    auto B = meshparam::mass_matrix(V, F);
    auto solver = meshparam::build_heat_solver(V, F, L, B);

    // Distance from center to (5,0) should equal distance to (0,5)
    int center = grid_idx(5, 5);
    auto dists = meshparam::geodesic_from_vertex(solver, center);

    double d_right = dists(grid_idx(10, 5));
    double d_up    = dists(grid_idx(5, 10));
    double d_left  = dists(grid_idx(0, 5));
    double d_down  = dists(grid_idx(5, 0));

    std::cout << "\n  Symmetry test from center (5,5):\n";
    std::cout << "  Right(10,5)=" << d_right
              << "  Up(5,10)=" << d_up
              << "  Left(0,5)=" << d_left
              << "  Down(5,0)=" << d_down << "\n";
    std::cout << "  All should be ~5.0\n";

    // All four should be close to 5.0 and close to each other
    EXPECT_NEAR(d_right, 5.0, 2.0);
    EXPECT_NEAR(d_up, 5.0, 2.0);
    EXPECT_NEAR(d_left, 5.0, 2.0);
    EXPECT_NEAR(d_down, 5.0, 2.0);

    // Symmetry: opposite directions should be similar
    EXPECT_NEAR(d_right, d_left, 1.0);
    EXPECT_NEAR(d_up, d_down, 1.0);
}

TEST(GeodesicAccuracy, FullMatrixVsExact) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    make_flat_grid_10x10(V, F);

    auto L = meshparam::cotangent_laplacian(V, F);
    auto B = meshparam::mass_matrix(V, F);
    auto solver = meshparam::build_heat_solver(V, F, L, B);

    auto G = meshparam::compute_geodesic_matrix(solver);

    // Compute RMS relative error against exact Euclidean distances
    int n = static_cast<int>(V.rows());
    double sum_sq = 0;
    int count = 0;
    double max_rel = 0;

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double exact = (V.row(i) - V.row(j)).norm();
            if (exact < 1e-10) continue;
            double rel = std::abs(G(i, j) - exact) / exact;
            sum_sq += rel * rel;
            max_rel = std::max(max_rel, rel);
            count++;
        }
    }

    double rms = std::sqrt(sum_sq / count);
    std::cout << "\n  Full geodesic matrix vs exact (flat grid):\n";
    std::cout << "  RMS relative error: " << rms * 100.0 << "%\n";
    std::cout << "  Max relative error: " << max_rel * 100.0 << "%\n";

    // On a flat surface, we expect <20% RMS error
    EXPECT_LT(rms, 0.3) << "Geodesic matrix has >30% RMS error on flat surface";
}
