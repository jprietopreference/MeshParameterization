"""
Classical Multidimensional Scaling (cMDS).

Given a symmetric pairwise distance matrix D, cMDS finds a low-dimensional
Euclidean embedding that best preserves those distances.

Algorithm
---------
1.  Square all distances: D² = D ∘ D
2.  Double-centre:         B = −½ H D² H   where H = I − (1/n) 11ᵀ
3.  Eigen-decompose:       B ≈ U Λ Uᵀ  (keep positive eigenvalues)
4.  Embedding:             X = U_d √Λ_d   (columns → d-D coordinates)

For mesh UV parameterisation *d = 2*, then UV ∈ [0, 1]² is obtained by
min-max normalisation of the two coordinate axes.

Reference
---------
Torgerson W.S., "Multidimensional Scaling: I. Theory and Method",
Psychometrika 17(4):401-419, 1952.
"""

import numpy as np


def classical_mds(D, dim=2):
    """Compute a *dim*-dimensional embedding via classical MDS.

    Parameters
    ----------
    D : array_like, shape (n, n)
        Symmetric non-negative pairwise distance matrix.
    dim : int, optional
        Number of output dimensions.  Default is 2 (for UV maps).

    Returns
    -------
    coords : ndarray, shape (n, dim)
        Embedding coordinates.  The *i*-th row is the position of the
        *i*-th point in the embedding space.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if D.shape != (n, n):
        raise ValueError(f"D must be square; got shape {D.shape}")

    D2 = D ** 2

    # Double-centering: B = -0.5 * H @ D2 @ H,  H = I - (1/n) 11ᵀ
    row_mean = D2.mean(axis=1, keepdims=True)
    col_mean = D2.mean(axis=0, keepdims=True)
    total_mean = D2.mean()
    B = -0.5 * (D2 - row_mean - col_mean + total_mean)

    # Symmetrise to remove floating-point asymmetry
    B = (B + B.T) / 2.0

    # Eigendecomposition (numpy returns eigenvalues in ascending order)
    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # Keep the *dim* largest positive eigenvalues
    # There are at most (n-1) non-trivial eigenvalues for n points
    available = min(dim, n - 1)
    if available <= 0:
        return np.zeros((n, dim))

    idx = np.argsort(eigenvalues)[::-1][:available]
    lam = np.maximum(eigenvalues[idx], 0.0)   # clip negatives to zero
    U = eigenvectors[:, idx]

    coords = U * np.sqrt(lam)[None, :]        # (n, available)

    # Pad with zeros if fewer dimensions are available than requested
    if available < dim:
        coords = np.concatenate(
            [coords, np.zeros((n, dim - available))], axis=1
        )
    return coords


def coords_to_uv(coords):
    """Normalise 2-D embedding coordinates to the unit square [0, 1]².

    Parameters
    ----------
    coords : array_like, shape (n, 2)
        Raw MDS coordinates.

    Returns
    -------
    uv : ndarray, shape (n, 2)
        Coordinates scaled so each axis spans [0, 1].
    """
    coords = np.asarray(coords, dtype=float)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    rng = hi - lo
    rng = np.where(rng < 1e-12, 1.0, rng)   # avoid division by zero
    uv = (coords - lo) / rng
    return uv
