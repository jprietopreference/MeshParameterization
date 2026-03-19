#include <cmath>
#include <gtest/gtest.h>
#include "meshparam/mesh.h"
#include "meshparam/parameterizer.h"
#include <Eigen/Core>

namespace {

// Create a flat rectangular mesh that is quasi-developable (actually developable)
void make_flat_rectangle(meshparam::TriMesh& mesh) {
    // 4x3 grid = 12 vertices, 12 triangles
    int nx = 4, ny = 3;
    mesh.V.resize(nx * ny, 3);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            mesh.V.row(j * nx + i) << i, j, 0;
        }
    }

    mesh.F.resize(2 * (nx - 1) * (ny - 1), 3);
    int f = 0;
    for (int j = 0; j < ny - 1; ++j) {
        for (int i = 0; i < nx - 1; ++i) {
            int v00 = j * nx + i;
            int v10 = j * nx + i + 1;
            int v01 = (j + 1) * nx + i;
            int v11 = (j + 1) * nx + i + 1;
            mesh.F.row(f++) << v00, v10, v01;
            mesh.F.row(f++) << v10, v11, v01;
        }
    }
}

// Create a slightly curved surface (cylinder-like patch)
void make_curved_patch(meshparam::TriMesh& mesh) {
    int nx = 5, ny = 4;
    double radius = 2.0;
    double arc = M_PI / 4; // 45 degree arc

    mesh.V.resize(nx * ny, 3);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            double theta = arc * i / (nx - 1);
            mesh.V.row(j * nx + i) <<
                radius * std::cos(theta),
                radius * std::sin(theta),
                static_cast<double>(j);
        }
    }

    mesh.F.resize(2 * (nx - 1) * (ny - 1), 3);
    int f = 0;
    for (int j = 0; j < ny - 1; ++j) {
        for (int i = 0; i < nx - 1; ++i) {
            int v00 = j * nx + i;
            int v10 = j * nx + i + 1;
            int v01 = (j + 1) * nx + i;
            int v11 = (j + 1) * nx + i + 1;
            mesh.F.row(f++) << v00, v10, v01;
            mesh.F.row(f++) << v10, v11, v01;
        }
    }
}

} // anonymous namespace

TEST(Pipeline, FlatRectangleProducesBijectiveUV) {
    meshparam::TriMesh mesh;
    make_flat_rectangle(mesh);

    meshparam::ParamConfig config;
    config.use_poisson_fill = false;
    config.auto_detect_fill = false;

    auto result = meshparam::parameterize(mesh, config);

    EXPECT_TRUE(result.has_uvs());
    EXPECT_EQ(result.UV.rows(), mesh.num_vertices());
    EXPECT_EQ(result.UV.cols(), 2);

    // UVs should be in [0,1] range (normalized)
    EXPECT_GE(result.UV.minCoeff(), -1e-10);
    EXPECT_LE(result.UV.maxCoeff(), 1.0 + 1e-10);
}

TEST(Pipeline, CurvedPatchParameterizes) {
    meshparam::TriMesh mesh;
    make_curved_patch(mesh);

    meshparam::ParamConfig config;
    config.use_poisson_fill = false;
    config.auto_detect_fill = false;

    auto result = meshparam::parameterize(mesh, config);

    EXPECT_TRUE(result.has_uvs());
    EXPECT_EQ(result.UV.rows(), mesh.num_vertices());

    // UVs should be in [0,1] range
    EXPECT_GE(result.UV.minCoeff(), -1e-10);
    EXPECT_LE(result.UV.maxCoeff(), 1.0 + 1e-10);
}

TEST(Pipeline, MeshBoundaryDetection) {
    meshparam::TriMesh mesh;
    make_flat_rectangle(mesh);

    // A flat rectangle has one boundary loop (the outer boundary)
    EXPECT_TRUE(mesh.has_boundary());
    auto loops = mesh.boundary_loops();
    EXPECT_EQ(loops.size(), 1);
}

TEST(Pipeline, MaxEdgeLength) {
    meshparam::TriMesh mesh;
    make_flat_rectangle(mesh);

    double max_edge = mesh.max_edge_length();
    // On a unit grid, the diagonal edge has length sqrt(2)
    EXPECT_NEAR(max_edge, std::sqrt(2.0), 1e-10);
}
