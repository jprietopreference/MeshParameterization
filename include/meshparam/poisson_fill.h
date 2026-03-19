#pragma once

#include "mesh.h"

namespace meshparam {

/// Poisson Surface Fill (Section 2.5 of paper).
///
/// For meshes with holes or boundary concavities, computes an
/// extended surface M* that fills the gaps so geodesic paths
/// are not distorted by mesh interruptions.
///
/// After parameterization of M*, the result is trimmed back
/// to the original mesh M boundary.
///
/// Uses Poisson surface reconstruction: solves Δχ = ∇·N
/// where N is the normal field extended from M.

struct PoissonFillResult {
    /// Extended mesh M* (contains M as subset)
    TriMesh extended_mesh;

    /// Mapping from M vertices to M* vertices.
    /// original_vertex_map[i] = index in M* of vertex i from M.
    std::vector<int> original_vertex_map;
};

/// Compute the Poisson surface fill M* for a mesh with holes/concavities.
PoissonFillResult poisson_fill(const TriMesh& mesh);

/// Trim a parameterization of M* back to the original mesh M.
/// Uses original_vertex_map to extract only the original vertices' UVs.
Eigen::MatrixXd trim_parameterization(
    const Eigen::MatrixXd& extended_uv,
    const PoissonFillResult& fill_result,
    int original_vertex_count);

} // namespace meshparam
