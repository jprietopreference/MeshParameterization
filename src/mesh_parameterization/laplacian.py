"""
Cotangent Laplacian and lumped mass matrix for triangulated surface meshes.

The cotangent Laplacian L is **negative semi-definite** with:
  L[i, j] = (cot α_ij + cot β_ij) / 2   for adjacent vertices i, j
  L[i, i] = −Σ_{j≠i} L[i, j]

where α_ij and β_ij are the two interior angles opposite to edge (i, j) in
the triangles that share that edge.

The lumped mass matrix M is diagonal:
  M[i, i] = (1/3) * Σ area(f)  over all triangles f incident to vertex i.

Reference
---------
Desbrun et al., "Implicit Fairing of Irregular Meshes using Diffusion
and Curvature Flow", SIGGRAPH 1999.
"""

import numpy as np
import scipy.sparse as sp


def cotangent_laplacian(vertices, faces):
    """Compute the cotangent Laplacian and lumped mass matrix.

    Parameters
    ----------
    vertices : array_like, shape (n, 3)
        Vertex positions in 3-D Euclidean space.
    faces : array_like, shape (m, 3)
        Triangle face indices (zero-based integers).

    Returns
    -------
    L : scipy.sparse.csr_matrix, shape (n, n)
        Cotangent Laplacian.  Negative semi-definite with positive
        off-diagonal entries for mesh-adjacent vertex pairs.
    M : scipy.sparse.dia_matrix, shape (n, n)
        Lumped (diagonal) mass matrix.  Positive definite.
    cot_weights : ndarray, shape (m, 3)
        Per-face cotangent weights ``[cot_0, cot_1, cot_2]`` where
        ``cot_k`` is the cotangent of the interior angle at vertex
        ``faces[:, k]``.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.intp)

    n = len(vertices)

    # Per-face vertex positions
    v0 = vertices[faces[:, 0]]  # (m, 3)
    v1 = vertices[faces[:, 1]]  # (m, 3)
    v2 = vertices[faces[:, 2]]  # (m, 3)

    # Edge vectors pointing away from each vertex
    e01 = v1 - v0   # v0 → v1
    e12 = v2 - v1   # v1 → v2
    e20 = v0 - v2   # v2 → v0

    # Face normal vector and twice-area (= |cross(e01, v2-v0)|)
    cross = np.cross(e01, -e20)          # cross(v1−v0, v2−v0)
    area2 = np.linalg.norm(cross, axis=1)  # (m,)  2·A_f

    denom = np.maximum(area2, 1e-12)     # avoid division by zero

    # Cotangent of the interior angle at each vertex:
    #   cot at v0: between edges e01 and (−e20) = e02
    #   cot at v1: between edges (−e01) = e10 and e12
    #   cot at v2: between edges (−e12) = e21 and e20
    cot0 = (e01 * (-e20)).sum(axis=1) / denom   # angle at vertex 0
    cot1 = ((-e01) * e12).sum(axis=1) / denom   # angle at vertex 1
    cot2 = ((-e12) * e20).sum(axis=1) / denom   # angle at vertex 2

    # Clamp to prevent extreme values from near-degenerate triangles
    cot0 = np.clip(cot0, -1e4, 1e4)
    cot1 = np.clip(cot1, -1e4, 1e4)
    cot2 = np.clip(cot2, -1e4, 1e4)

    cot_weights = np.stack([cot0, cot1, cot2], axis=1)  # (m, 3)

    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]

    # Each cotangent weight contributes to the edge *opposite* its vertex:
    #   cot0 (angle at v0) → weight for edge (v1, v2)
    #   cot1 (angle at v1) → weight for edge (v0, v2)
    #   cot2 (angle at v2) → weight for edge (v0, v1)
    #
    # Off-diagonal: L[a, b] += cot / 2  (symmetric)
    rows = np.concatenate([i1, i2, i0, i2, i0, i1])
    cols = np.concatenate([i2, i1, i2, i0, i1, i0])
    vals = np.concatenate([cot0, cot0, cot1, cot1, cot2, cot2]) / 2.0

    L_off = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

    # Diagonal: L[i, i] = −Σ_j L[i, j]
    row_sums = np.array(L_off.sum(axis=1)).ravel()
    L_diag = sp.diags(-row_sums)
    L = L_off + L_diag

    # Lumped mass matrix: M[i, i] = Σ_f area(f) / 3
    area = area2 / 2.0
    vert_areas = np.zeros(n)
    np.add.at(vert_areas, faces[:, 0], area / 3.0)
    np.add.at(vert_areas, faces[:, 1], area / 3.0)
    np.add.at(vert_areas, faces[:, 2], area / 3.0)
    vert_areas = np.maximum(vert_areas, 1e-12)
    M = sp.diags(vert_areas)

    return L, M, cot_weights
