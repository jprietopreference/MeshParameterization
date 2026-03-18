"""Tests for the cotangent Laplacian and lumped mass matrix."""

import numpy as np
import scipy.sparse as sp
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mesh_parameterization.laplacian import cotangent_laplacian


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def single_triangle():
    """One equilateral triangle."""
    s = np.sqrt(3) / 2
    verts = np.array([[0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.5, s,  0.0]])
    faces = np.array([[0, 1, 2]])
    return verts, faces


def unit_square_mesh():
    """Unit square split into two triangles.

    Vertices:  0=(0,0,0)  1=(1,0,0)  2=(1,1,0)  3=(0,1,0)
    Faces:     [0,1,2]  [0,2,3]
    """
    verts = np.array([[0.0, 0.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [1.0, 1.0, 0.0],
                      [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2],
                      [0, 2, 3]])
    return verts, faces


def regular_icosphere(subdivisions=1):
    """Small closed triangle mesh (icosphere)."""
    import trimesh
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions)
    return np.array(mesh.vertices), np.array(mesh.faces)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCotangentLaplacian:

    def test_output_shapes(self):
        v, f = unit_square_mesh()
        L, M, cw = cotangent_laplacian(v, f)
        n = len(v)
        assert L.shape == (n, n)
        assert M.shape == (n, n)
        assert cw.shape == (len(f), 3)

    def test_L_is_sparse(self):
        v, f = unit_square_mesh()
        L, M, _ = cotangent_laplacian(v, f)
        assert sp.issparse(L)
        assert sp.issparse(M)

    def test_L_symmetric(self):
        v, f = unit_square_mesh()
        L, _, _ = cotangent_laplacian(v, f)
        diff = (L - L.T).toarray()
        assert np.allclose(diff, 0, atol=1e-10)

    def test_L_row_sums_zero(self):
        """Laplacian rows must sum to zero (constant functions are null)."""
        v, f = regular_icosphere()
        L, _, _ = cotangent_laplacian(v, f)
        row_sums = np.array(L.sum(axis=1)).ravel()
        assert np.allclose(row_sums, 0, atol=1e-8)

    def test_L_negative_semi_definite(self):
        """All eigenvalues of L must be ≤ 0."""
        v, f = regular_icosphere()
        L, _, _ = cotangent_laplacian(v, f)
        # Check via diagonal: should be non-positive
        diag = L.diagonal()
        assert np.all(diag <= 1e-10)

    def test_M_diagonal_positive(self):
        v, f = unit_square_mesh()
        _, M, _ = cotangent_laplacian(v, f)
        diag = M.diagonal()
        assert np.all(diag > 0)

    def test_M_sums_to_total_area(self):
        """Sum of lumped masses must equal total mesh area."""
        v, f = unit_square_mesh()
        _, M, _ = cotangent_laplacian(v, f)
        total_area = 1.0   # unit square
        assert np.isclose(M.diagonal().sum(), total_area, rtol=1e-6)

    def test_equilateral_triangle_cotangents(self):
        """All angles 60° → cot = 1/√3 ≈ 0.5774."""
        v, f = single_triangle()
        _, _, cw = cotangent_laplacian(v, f)
        expected = 1.0 / np.sqrt(3)
        assert np.allclose(cw[0], expected, atol=1e-6)

    def test_right_triangle_cotangents(self):
        """Right-angle triangle at v0: cot(90°)=0, cot(45°)=1."""
        verts = np.array([[0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]])
        faces = np.array([[0, 1, 2]])
        _, _, cw = cotangent_laplacian(verts, faces)
        assert np.isclose(cw[0, 0], 0.0, atol=1e-6)   # right angle at v0
        assert np.isclose(cw[0, 1], 1.0, atol=1e-6)   # 45° at v1
        assert np.isclose(cw[0, 2], 1.0, atol=1e-6)   # 45° at v2

    def test_degenerate_triangle_no_crash(self):
        """Flat (zero-area) triangle should not raise an exception."""
        verts = np.array([[0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [2.0, 0.0, 0.0]])
        faces = np.array([[0, 1, 2]])
        L, M, cw = cotangent_laplacian(verts, faces)
        assert L.shape == (3, 3)
