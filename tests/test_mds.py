"""Tests for classical MDS and UV normalisation."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mesh_parameterization.mds import classical_mds, coords_to_uv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def euclidean_distance_matrix(pts):
    """Compute the exact Euclidean pairwise distance matrix."""
    diff = pts[:, None, :] - pts[None, :, :]   # (n, n, d)
    return np.sqrt((diff ** 2).sum(axis=-1))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestClassicalMDS:

    def test_output_shape(self):
        D = np.array([[0., 1., 2.],
                      [1., 0., 1.],
                      [2., 1., 0.]])
        coords = classical_mds(D, dim=2)
        assert coords.shape == (3, 2)

    def test_recovers_2d_points(self):
        """MDS must recover a 2-D point configuration from its distance matrix."""
        rng = np.random.default_rng(0)
        pts = rng.standard_normal((12, 2))
        D = euclidean_distance_matrix(pts)
        coords = classical_mds(D, dim=2)
        # Recovered pairwise distances should match original
        D_rec = euclidean_distance_matrix(coords)
        assert np.allclose(D_rec, D, atol=1e-8)

    def test_recovers_3d_points_in_3d(self):
        rng = np.random.default_rng(1)
        pts = rng.standard_normal((8, 3))
        D = euclidean_distance_matrix(pts)
        coords = classical_mds(D, dim=3)
        D_rec = euclidean_distance_matrix(coords)
        assert np.allclose(D_rec, D, atol=1e-8)

    def test_single_point(self):
        D = np.array([[0.0]])
        coords = classical_mds(D, dim=2)
        assert coords.shape == (1, 2)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            classical_mds(np.ones((3, 4)), dim=2)

    def test_handles_zero_distances(self):
        """All-zero distance matrix → output should be all zeros."""
        D = np.zeros((5, 5))
        coords = classical_mds(D, dim=2)
        assert np.allclose(coords, 0.0, atol=1e-10)

    def test_dim1_reduction(self):
        """1-D points embedded on a line."""
        pts = np.linspace(0, 1, 6).reshape(-1, 1)
        D = euclidean_distance_matrix(pts)
        coords = classical_mds(D, dim=1)
        assert coords.shape == (6, 1)
        D_rec = euclidean_distance_matrix(coords)
        assert np.allclose(D_rec, D, atol=1e-8)


class TestCoordsToUV:

    def test_output_in_unit_square(self):
        rng = np.random.default_rng(2)
        coords = rng.standard_normal((20, 2))
        uv = coords_to_uv(coords)
        assert uv.min() >= -1e-12
        assert uv.max() <= 1.0 + 1e-12
        assert np.isclose(uv.min(axis=0).min(), 0.0, atol=1e-10)
        assert np.isclose(uv.max(axis=0).max(), 1.0, atol=1e-10)

    def test_constant_input_no_crash(self):
        """Constant coordinate axis → all UVs on the boundary."""
        coords = np.zeros((5, 2))
        coords[:, 0] = np.arange(5)  # only x varies
        uv = coords_to_uv(coords)
        assert uv.shape == (5, 2)

    def test_preserves_relative_order(self):
        """UV u-axis order must match original x-axis order."""
        coords = np.column_stack([np.linspace(0, 1, 10),
                                  np.zeros(10)])
        uv = coords_to_uv(coords)
        assert np.all(np.diff(uv[:, 0]) >= 0)
