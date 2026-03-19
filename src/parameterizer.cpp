#include "meshparam/parameterizer.h"
#include "meshparam/laplacian.h"
#include "meshparam/heat_geodesics.h"
#include "meshparam/mds.h"
#include "meshparam/poisson_fill.h"
#include "meshparam/distortion.h"
#include <iostream>

namespace meshparam {

TriMesh parameterize(const TriMesh& mesh, const ParamConfig& config) {
    TriMesh result = mesh;

    const Eigen::MatrixXd* V_ptr = &mesh.V;
    const Eigen::MatrixXi* F_ptr = &mesh.F;

    // Step 1: Check for holes/concavities, optionally apply Poisson fill
    PoissonFillResult fill_result;
    bool use_fill = config.use_poisson_fill;

    if (config.auto_detect_fill) {
        auto loops = mesh.boundary_loops();
        use_fill = loops.size() > 1;
    }

    if (use_fill) {
        std::cout << "[meshparam] Applying Poisson surface fill..." << std::endl;
        fill_result = poisson_fill(mesh);
        V_ptr = &fill_result.extended_mesh.V;
        F_ptr = &fill_result.extended_mesh.F;
        std::cout << "[meshparam] Extended mesh: "
                  << V_ptr->rows() << " vertices, "
                  << F_ptr->rows() << " faces" << std::endl;
    }

    int n = static_cast<int>(V_ptr->rows());
    std::cout << "[meshparam] Computing parameterization for " << n << " vertices..." << std::endl;

    // Steps 3-4: Build and prefactor Laplacian and mass matrices
    std::cout << "[meshparam] Building Laplace-Beltrami and mass matrices..." << std::endl;
    auto L = cotangent_laplacian(*V_ptr, *F_ptr);
    auto B = mass_matrix(*V_ptr, *F_ptr);

    std::cout << "[meshparam] Prefactoring solvers..." << std::endl;
    auto solver = build_heat_solver(*V_ptr, *F_ptr, L, B);

    // Steps 5-7: Compute geodesic distance matrix
    std::cout << "[meshparam] Computing geodesic distances (" << n << "x" << n << " matrix)..." << std::endl;
    auto G = compute_geodesic_matrix(solver);

    // Steps 8-10: MDS → UV coordinates
    std::cout << "[meshparam] Running MDS parameterization..." << std::endl;
    Eigen::MatrixXd UV = classical_mds(G);

    // If we used Poisson fill, trim back to original vertices
    Eigen::MatrixXd G_orig = G;
    if (use_fill) {
        std::cout << "[meshparam] Trimming to original boundary..." << std::endl;
        UV = trim_parameterization(UV, fill_result, mesh.num_vertices());
        // Extract original vertices' geodesic submatrix for distortion
        int n_orig = mesh.num_vertices();
        G_orig.resize(n_orig, n_orig);
        for (int i = 0; i < n_orig; ++i) {
            for (int j = 0; j < n_orig; ++j) {
                int mi = fill_result.original_vertex_map[i];
                int mj = fill_result.original_vertex_map[j];
                G_orig(i, j) = G(mi, mj);
            }
        }
    }

    // Compute isometric distortion BEFORE normalizing UVs
    // (MDS output is in the same scale as geodesic distances)
    double iso_rms = compute_isometric_distortion(G_orig, UV);

    // Normalize UV to [0, 1] range
    Eigen::Vector2d uv_min = UV.colwise().minCoeff();
    Eigen::Vector2d uv_max = UV.colwise().maxCoeff();
    Eigen::Vector2d uv_range = uv_max - uv_min;

    for (int i = 0; i < 2; ++i) {
        if (uv_range(i) > 1e-12) {
            UV.col(i) = (UV.col(i).array() - uv_min(i)) / uv_range(i);
        }
    }

    result.UV = UV;

    // Compute distortion metrics
    auto metrics = compute_distortion(mesh.V, mesh.F, UV);
    metrics.isometric_rms = iso_rms;
    metrics.print_summary();

    std::cout << "[meshparam] Done." << std::endl;

    return result;
}

} // namespace meshparam
