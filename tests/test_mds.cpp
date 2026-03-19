#include <gtest/gtest.h>
#include "meshparam/mds.h"
#include <Eigen/Core>
#include <cmath>

TEST(MDS, RecoverSquareDistances) {
    // 4 points forming a unit square in 2D
    // Known Euclidean distances:
    //   d(0,1)=1, d(0,2)=1, d(0,3)=√2
    //   d(1,2)=√2, d(1,3)=1, d(2,3)=1
    Eigen::MatrixXd G(4, 4);
    double s2 = std::sqrt(2.0);
    G << 0,  1,  1,  s2,
         1,  0,  s2, 1,
         1,  s2, 0,  1,
         s2, 1,  1,  0;

    Eigen::MatrixXd UV = meshparam::classical_mds(G);

    EXPECT_EQ(UV.rows(), 4);
    EXPECT_EQ(UV.cols(), 2);

    // Check that pairwise distances in UV match the input
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            double d_uv = (UV.row(i) - UV.row(j)).norm();
            EXPECT_NEAR(d_uv, G(i, j), 0.1)
                << "Distance mismatch between vertices " << i << " and " << j;
        }
    }
}

TEST(MDS, OutputDimensions) {
    // 5 random distances (approximately Euclidean)
    Eigen::MatrixXd G(5, 5);
    G << 0, 1, 2, 3, 2,
         1, 0, 1.5, 2.5, 1.8,
         2, 1.5, 0, 1.2, 2.1,
         3, 2.5, 1.2, 0, 1.5,
         2, 1.8, 2.1, 1.5, 0;

    Eigen::MatrixXd UV = meshparam::classical_mds(G);

    EXPECT_EQ(UV.rows(), 5);
    EXPECT_EQ(UV.cols(), 2);
}

TEST(MDS, MeanCentered) {
    Eigen::MatrixXd G(4, 4);
    double s2 = std::sqrt(2.0);
    G << 0,  1,  1,  s2,
         1,  0,  s2, 1,
         1,  s2, 0,  1,
         s2, 1,  1,  0;

    Eigen::MatrixXd UV = meshparam::classical_mds(G);

    // Result should be approximately mean-centered
    Eigen::Vector2d mean = UV.colwise().mean();
    EXPECT_NEAR(mean(0), 0.0, 1e-8);
    EXPECT_NEAR(mean(1), 0.0, 1e-8);
}
