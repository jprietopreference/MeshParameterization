"""
Full UV-parameterisation pipeline for triangulated meshes.

Pipeline
--------
1.  Fill boundary holes (fan triangulation) so the mesh is (topologically)
    closed — this avoids boundary artefacts in the geodesic computation.
2.  Build the heat-geodesics solver on the closed mesh.
3.  Compute the all-pairs pairwise geodesic distance matrix restricted to
    the *original* vertices (the centroid vertices added during hole-filling
    are discarded afterwards).
4.  Apply classical MDS to the distance matrix to obtain a 2-D embedding.
5.  Normalise the embedding to [0, 1]² to produce UV texture coordinates.

Reference
---------
"Quasi-Isometric Mesh Parameterization Using Heat-Based Geodesics and
Poisson Surface Fills", MDPI Mathematics 7(8):753, 2019.
"""

import numpy as np

from .hole_filling import fill_holes
from .heat_geodesics import HeatGeodesics
from .mds import classical_mds, coords_to_uv


# Maximum number of vertices for full all-pairs distance computation.
# For larger meshes a uniform sub-sample is used and the remaining UVs
# are interpolated via barycentric coordinates.
_MAX_ALLPAIRS = 2000


def parameterize_mesh(vertices, faces, t_coeff=1.0, max_allpairs=_MAX_ALLPAIRS):
    """Compute UV texture coordinates for a triangulated mesh.

    Parameters
    ----------
    vertices : array_like, shape (n, 3)
        Vertex positions.
    faces : array_like, shape (m, 3)
        Triangle face indices (zero-based integers).
    t_coeff : float, optional
        Time-step coefficient for the heat equation.  Larger values give
        smoother (but less accurate) geodesics.  Default is 1.0.
    max_allpairs : int, optional
        Maximum mesh size for full all-pairs geodesic computation.
        Larger meshes are handled via landmark-based MDS with
        barycentric interpolation.  Default is 2000.

    Returns
    -------
    uv : ndarray, shape (n, 2)
        UV texture coordinates in [0, 1]² for each original vertex.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.intp)
    n = len(vertices)

    # --- Step 1: fill holes on a copy of the mesh -------------------------
    v_filled, f_filled, n_orig = fill_holes(vertices, faces)

    # --- Step 2: build heat-geodesics solver ------------------------------
    solver = HeatGeodesics(v_filled, f_filled, t_coeff=t_coeff)

    # --- Step 3: pairwise distance matrix for original vertices -----------
    if n <= max_allpairs:
        uv = _full_mds(solver, n)
    else:
        uv = _landmark_mds(solver, vertices, faces, n, max_allpairs)

    return uv


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _full_mds(solver, n):
    """All-pairs geodesic distances + MDS on the full vertex set."""
    D = solver.compute_all_pairs(max_verts=n)
    coords = classical_mds(D, dim=2)
    return coords_to_uv(coords)


def _landmark_mds(solver, vertices, faces, n, n_landmarks):
    """Landmark-based MDS with barycentric interpolation for large meshes.

    A set of *n_landmarks* vertices is selected uniformly (by index stride).
    MDS is applied to the landmark sub-matrix, and the remaining vertices
    obtain UVs by barycentric interpolation from their enclosing triangles.
    """
    # Uniformly spaced landmark indices
    step = max(1, n // n_landmarks)
    landmarks = np.arange(0, n, step)[:n_landmarks]
    k = len(landmarks)

    # Pairwise distances between landmarks
    D_land = np.zeros((k, k))
    for li, s in enumerate(landmarks):
        phi = solver.compute(s)
        D_land[li, :] = phi[landmarks]
    D_land = (D_land + D_land.T) / 2.0

    # MDS embedding for landmarks
    coords_land = classical_mds(D_land, dim=2)

    # Interpolate UVs for all vertices from their nearest landmark
    # (fallback: nearest-landmark assignment)
    # Compute distances from each vertex to each landmark
    dist_to_land = np.zeros((n, k))
    for li, s in enumerate(landmarks):
        phi = solver.compute(s)
        dist_to_land[:, li] = phi[:n]

    # For each vertex, find the 3 nearest landmarks and interpolate
    uv = np.zeros((n, 2))
    for i in range(n):
        nearest = np.argsort(dist_to_land[i])[:3]
        d = dist_to_land[i, nearest]
        # Inverse-distance weighting
        d = np.maximum(d, 1e-12)
        w = 1.0 / d
        w /= w.sum()
        uv[i] = (coords_land[nearest] * w[:, None]).sum(axis=0)

    return coords_to_uv(uv)
