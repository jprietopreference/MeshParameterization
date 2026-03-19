#pragma once

#include "mesh.h"

namespace meshparam {

/// Configuration for the parameterization algorithm.
struct ParamConfig {
    /// Whether to use Poisson fill for meshes with holes/concavities.
    /// If true and mesh has boundary issues, Poisson fill is applied.
    /// If false, parameterize directly (may distort near holes).
    bool use_poisson_fill = true;

    /// Whether to auto-detect if Poisson fill is needed.
    /// If true, checks mesh boundary; if false, uses use_poisson_fill flag.
    bool auto_detect_fill = true;
};

/// Main entry point: parameterize a mesh following the full paper pipeline.
///
/// Pipeline (Figure 1 of paper):
///   1. Check for holes/concavities → optionally Poisson fill
///   2. Compute Δt, T simulation parameters
///   3. Build & prefactor Laplace-Beltrami L and mass B matrices
///   4. For each vertex: heat kernel → normalized flux → geodesic distance
///   5. Build geodesic matrix G
///   6. MDS → UV coordinates Φ = [√λ1·V1, √λ2·V2]
///   7. If filled, trim back to original boundary
///
/// Returns a copy of the mesh with UV coordinates set.
TriMesh parameterize(const TriMesh& mesh, const ParamConfig& config = {});

} // namespace meshparam
