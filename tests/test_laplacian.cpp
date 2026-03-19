#include <gtest/gtest.h>
#include "meshparam/laplacian.h"
#include <Eigen/Core>

namespace {

// Simple test mesh: a single quad split into 2 triangles
// Vertices form a unit square in the XY plane:
//   v2---v3
//   |  / |
//   v0---v1
void make_unit_square(Eigen::MatrixXd& V, Eigen::MatrixXi& F) {
    V.resize(4, 3);
    V << 0, 0, 0,
         1, 0, 0,
         0, 1, 0,
         1, 1, 0;

    F.resize(2, 3);
    F << 0, 1, 2,
         1, 3, 2;
}

// Equilateral triangle
void make_equilateral(Eigen::MatrixXd& V, Eigen::MatrixXi& F) {
    V.resize(3, 3);
    V << 0, 0, 0,
         1, 0, 0,
         0.5, std::sqrt(3.0)/2.0, 0;

    F.resize(1, 3);
    F << 0, 1, 2;
}

} // anonymous namespace

TEST(Laplacian, CotangentMatrixSize) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    make_unit_square(V, F);

    auto L = meshparam::cotangent_laplacian(V, F);
    EXPECT_EQ(L.rows(), 4);
    EXPECT_EQ(L.cols(), 4);
}

TEST(Laplacian, CotangentMatrixSymmetric) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    make_unit_square(V, F);

    auto L = meshparam::cotangent_laplacian(V, F);

    // Check symmetry
    for (int i = 0; i < L.rows(); ++i) {
        for (int j = 0; j < L.cols(); ++j) {
            EXPECT_NEAR(L.coeff(i, j), L.coeff(j, i), 1e-12);
        }
    }
}

TEST(Laplacian, RowSumsZero) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    make_unit_square(V, F);

    auto L = meshparam::cotangent_laplacian(V, F);

    // Row sums should be approximately zero (L * ones = 0)
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(L.rows());
    Eigen::VectorXd result = L * ones;
    for (int i = 0; i < result.size(); ++i) {
        EXPECT_NEAR(result(i), 0.0, 1e-10);
    }
}

TEST(Laplacian, MassMatrixSize) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    make_unit_square(V, F);

    auto B = meshparam::mass_matrix(V, F);
    EXPECT_EQ(B.rows(), 4);
    EXPECT_EQ(B.cols(), 4);
}

TEST(Laplacian, MassMatrixPositive) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    make_unit_square(V, F);

    auto B = meshparam::mass_matrix(V, F);

    // Diagonal entries should be positive
    for (int i = 0; i < B.rows(); ++i) {
        EXPECT_GT(B.coeff(i, i), 0.0);
    }
}

TEST(Laplacian, MassMatrixAreaConsistency) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    make_equilateral(V, F);

    auto B = meshparam::mass_matrix(V, F);

    // Sum of diagonal = total area for Voronoi mass matrix
    double total_mass = 0;
    for (int i = 0; i < B.rows(); ++i) {
        total_mass += B.coeff(i, i);
    }
    double expected_area = std::sqrt(3.0) / 4.0;  // area of unit equilateral triangle
    EXPECT_NEAR(total_mass, expected_area, 1e-10);
}
