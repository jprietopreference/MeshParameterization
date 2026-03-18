"""
Heat-based geodesic distance computation on triangulated meshes.

Algorithm
---------
Crane K., Weischedel C., Wardetzky M.
"Geodesics in Heat: A New Approach to Computing Distance Based on Heat Flow"
ACM Transactions on Graphics, 32(5), 2013.

Steps for a source vertex s
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Solve the heat equation for one implicit time step t:
       (M − t · L) u = e_s
   where L is the cotangent Laplacian, M is the lumped mass matrix, and
   e_s is the unit vector (1 at s, 0 elsewhere).

2. Compute the face-wise gradient of u:
       ∇u_f = Σ_k u_k · (n̂_f × e_k^opp) / (2 A_f)

3. Normalise to unit length and negate (distances increase away from source):
       X_f = −∇u_f / ‖∇u_f‖

4. Compute the integrated divergence at every vertex:
       div(X)[i] = ½ Σ_{f ∋ i} (cot_k · X_f · e_{ij} + cot_j · X_f · e_{ik})

5. Solve the Poisson equation (with one pinned row to fix the gauge):
       L φ = div(X),   φ[s] = 0

6. Shift φ so its minimum is zero.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .laplacian import cotangent_laplacian


class HeatGeodesics:
    """Pre-factored solver for heat-based geodesic distances.

    Pre-factoring the two sparse linear systems (heat equation and Poisson
    equation) lets you query distances from many source vertices efficiently
    on the same mesh.

    Parameters
    ----------
    vertices : array_like, shape (n, 3)
        Vertex positions.
    faces : array_like, shape (m, 3)
        Triangle face indices (zero-based).
    t_coeff : float, optional
        Time-step coefficient.  The actual time step is
        ``t = t_coeff * h²`` where *h* is the mean edge length.
        Default is 1.0.
    """

    def __init__(self, vertices, faces, t_coeff=1.0):
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces = np.asarray(faces, dtype=np.intp)
        n = len(self.vertices)

        L, M, cot_w = cotangent_laplacian(self.vertices, self.faces)
        self._L = L
        self._M = M
        self._cot_w = cot_w          # (m, 3)
        self._n = n

        # Time step t = t_coeff * h²
        h = _mean_edge_length(self.vertices, self.faces)
        self._t = t_coeff * h * h

        # Pre-factor heat system A = M − t·L  (positive definite)
        A = M - self._t * L
        self._heat_solver = spla.factorized(A.tocsc())

        # Poisson system matrix (L with row 0 replaced by identity row)
        self._L_pin = _pin_vertex(L, 0)
        self._poisson_solver = spla.factorized(self._L_pin.tocsc())

    # ------------------------------------------------------------------
    def compute(self, source):
        """Compute geodesic distances from *source* to all vertices.

        Parameters
        ----------
        source : int
            Source vertex index.

        Returns
        -------
        phi : ndarray, shape (n,)
            Geodesic distance from *source* to each vertex.
            ``phi[source] == 0``.
        """
        source = int(source)

        # --- Step 1: solve heat equation ---
        rhs_heat = np.zeros(self._n)
        rhs_heat[source] = 1.0
        u = self._heat_solver(rhs_heat)

        # --- Step 2 & 3: normalised gradient field X per face ---
        X = _face_gradient(self.vertices, self.faces, u)
        grad_norm = np.linalg.norm(X, axis=1, keepdims=True)
        mask = grad_norm[:, 0] > 1e-12
        X[mask] = -X[mask] / grad_norm[mask]
        X[~mask] = 0.0

        # --- Step 4: divergence ---
        div_X = _vertex_divergence(self.vertices, self.faces, X, self._cot_w)

        # --- Step 5: solve Poisson equation L φ = div_X, pinned at vertex 0 ---
        rhs_poisson = div_X.copy()
        rhs_poisson[0] = 0.0          # enforced by _pin_vertex
        phi = self._poisson_solver(rhs_poisson)

        # --- Step 6: shift so that phi[source] = 0 ---
        phi -= phi[source]
        phi = np.abs(phi)             # distances are non-negative
        return phi

    def compute_all_pairs(self, max_verts=None):
        """Compute the full pairwise geodesic distance matrix.

        Parameters
        ----------
        max_verts : int or None
            If given and the mesh has more vertices, only the first
            *max_verts* vertices are used as sources (and targets).
            The returned matrix is (max_verts × max_verts).
            Pass ``None`` (default) to use all vertices.

        Returns
        -------
        D : ndarray, shape (k, k)
            Symmetric distance matrix where k = min(n, max_verts).
        """
        k = self._n if max_verts is None else min(self._n, max_verts)
        D = np.zeros((k, k))
        for s in range(k):
            phi = self.compute(s)
            D[s, :k] = phi[:k]
        # Symmetrise (small numerical asymmetries may arise)
        D = (D + D.T) / 2.0
        return D


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _mean_edge_length(vertices, faces):
    """Return the mean edge length of the mesh."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    l01 = np.linalg.norm(v1 - v0, axis=1)
    l12 = np.linalg.norm(v2 - v1, axis=1)
    l20 = np.linalg.norm(v0 - v2, axis=1)
    return (l01.mean() + l12.mean() + l20.mean()) / 3.0


def _pin_vertex(L, pin=0):
    """Return a modified L with row *pin* replaced by the identity row.

    This fixes the gauge freedom of the Poisson equation on a closed mesh
    (where L has a one-dimensional null space spanned by the constant vector).
    """
    L_mod = L.tolil().astype(float)
    L_mod[pin, :] = 0.0
    L_mod[pin, pin] = 1.0
    return L_mod.tocsr()


def _face_gradient(vertices, faces, u):
    """Compute the piecewise-constant gradient of scalar field *u* per face.

    For a triangle with vertices p_i, p_j, p_k and values u_i, u_j, u_k:

        ∇u_f = (u_i · (n̂ × e_i^opp) + u_j · (n̂ × e_j^opp) + u_k · (n̂ × e_k^opp)) / (2 A_f)

    where e_k^opp is the edge *opposite* to vertex k (i.e. e_jk for vertex i),
    n̂ is the unit face normal, and A_f is the face area.

    Parameters
    ----------
    vertices : (n, 3) float array
    faces    : (m, 3) int array
    u        : (n,) float array

    Returns
    -------
    grad : (m, 3) float array
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    u0 = u[faces[:, 0]][:, None]   # (m, 1)
    u1 = u[faces[:, 1]][:, None]
    u2 = u[faces[:, 2]][:, None]

    # Face normal (unnormalised, magnitude = 2·area)
    cross = np.cross(v1 - v0, v2 - v0)   # (m, 3)
    area2 = np.linalg.norm(cross, axis=1, keepdims=True)  # (m, 1)
    n_hat = cross / np.maximum(area2, 1e-12)   # unit normal (m, 3)

    # Edges opposite to each vertex
    e_opp0 = v2 - v1   # opposite v0
    e_opp1 = v0 - v2   # opposite v1
    e_opp2 = v1 - v0   # opposite v2

    grad = (u0 * np.cross(n_hat, e_opp0) +
            u1 * np.cross(n_hat, e_opp1) +
            u2 * np.cross(n_hat, e_opp2)) / np.maximum(area2, 1e-12)
    return grad   # (m, 3)


def _vertex_divergence(vertices, faces, X, cot_weights):
    """Integrated divergence of piecewise-constant vector field *X* at each vertex.

    For a triangle f = [i, j, k], the contribution to vertex i is:
        (cot_k · (X_f · e_{ij}) + cot_j · (X_f · e_{ik})) / 2

    where e_{ij} = v_j − v_i, e_{ik} = v_k − v_i, cot_j is the cotangent
    of the angle at j, and cot_k is the cotangent at k.

    Parameters
    ----------
    vertices    : (n, 3) float array
    faces       : (m, 3) int array
    X           : (m, 3) float array  – normalised face vector field
    cot_weights : (m, 3) float array  – [cot_0, cot_1, cot_2] per face

    Returns
    -------
    div_X : (n,) float array
    """
    n = len(vertices)
    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
    cot0, cot1, cot2 = cot_weights[:, 0], cot_weights[:, 1], cot_weights[:, 2]

    v0 = vertices[i0]
    v1 = vertices[i1]
    v2 = vertices[i2]

    e01 = v1 - v0    # (m, 3) edge i0 → i1
    e02 = v2 - v0    # (m, 3) edge i0 → i2
    e10 = -e01       # edge i1 → i0
    e12 = v2 - v1    # (m, 3) edge i1 → i2
    e20 = -e02       # edge i2 → i0
    e21 = -e12       # edge i2 → i1

    # dot(X, edge) per face
    Xe01 = (X * e01).sum(axis=1)
    Xe02 = (X * e02).sum(axis=1)
    Xe10 = (X * e10).sum(axis=1)
    Xe12 = (X * e12).sum(axis=1)
    Xe20 = (X * e20).sum(axis=1)
    Xe21 = (X * e21).sum(axis=1)

    # Contributions per vertex (angle opposite to each edge):
    #   vertex i0: edges (i0,i1) opposite i2 → cot2; (i0,i2) opposite i1 → cot1
    #   vertex i1: edges (i1,i0) opposite i2 → cot2; (i1,i2) opposite i0 → cot0
    #   vertex i2: edges (i2,i0) opposite i1 → cot1; (i2,i1) opposite i0 → cot0
    contrib0 = (cot2 * Xe01 + cot1 * Xe02) / 2.0
    contrib1 = (cot2 * Xe10 + cot0 * Xe12) / 2.0
    contrib2 = (cot1 * Xe20 + cot0 * Xe21) / 2.0

    div_X = np.zeros(n)
    np.add.at(div_X, i0, contrib0)
    np.add.at(div_X, i1, contrib1)
    np.add.at(div_X, i2, contrib2)
    return div_X
