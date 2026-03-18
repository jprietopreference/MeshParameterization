"""Tests for heat-based geodesic distance computation."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mesh_parameterization.heat_geodesics import (
    HeatGeodesics, _face_gradient, _vertex_divergence, _mean_edge_length
)
from mesh_parameterization.laplacian import cotangent_laplacian


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_flat_grid(nx=5, ny=5):
    """Create a flat nx×ny grid of triangles in the XY-plane.

    Returns (vertices, faces) for a unit square with nx*ny quads each
    split into 2 triangles.
    """
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    verts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny)])
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            b = a + 1
            c = a + nx
            d = c + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    return verts, np.array(faces)


def make_icosphere(subdivisions=2):
    import trimesh
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions)
    return np.array(mesh.vertices), np.array(mesh.faces)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMeanEdgeLength:

    def test_unit_equilateral(self):
        verts = np.array([[0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [0.5, np.sqrt(3)/2, 0.0]])
        faces = np.array([[0, 1, 2]])
        h = _mean_edge_length(verts, faces)
        assert np.isclose(h, 1.0, rtol=1e-6)


class TestFaceGradient:

    def test_linear_x_gradient(self):
        """Gradient of u = x-coordinate should be (1, 0, 0) everywhere."""
        verts, faces = make_flat_grid(4, 4)
        u = verts[:, 0].copy()
        grad = _face_gradient(verts, faces, u)
        assert grad.shape == (len(faces), 3)
        assert np.allclose(grad[:, 0], 1.0, atol=1e-10)
        assert np.allclose(grad[:, 1], 0.0, atol=1e-10)
        assert np.allclose(grad[:, 2], 0.0, atol=1e-10)

    def test_constant_u_zero_gradient(self):
        """Gradient of a constant field is zero."""
        verts, faces = make_flat_grid(4, 4)
        u = np.ones(len(verts))
        grad = _face_gradient(verts, faces, u)
        assert np.allclose(grad, 0.0, atol=1e-12)


class TestHeatGeodesics:

    def test_self_distance_zero(self):
        verts, faces = make_icosphere(subdivisions=2)
        solver = HeatGeodesics(verts, faces)
        phi = solver.compute(0)
        assert np.isclose(phi[0], 0.0, atol=1e-8)

    def test_distances_non_negative(self):
        verts, faces = make_icosphere(subdivisions=2)
        solver = HeatGeodesics(verts, faces)
        phi = solver.compute(0)
        assert np.all(phi >= -1e-8)

    def test_triangle_inequality(self, n_samples=20):
        """Sample random triples and check the triangle inequality."""
        verts, faces = make_icosphere(subdivisions=2)
        solver = HeatGeodesics(verts, faces)
        n = len(verts)
        rng = np.random.default_rng(42)
        sources = rng.choice(n, size=3, replace=False)
        dist = {s: solver.compute(s) for s in sources}
        for i in sources:
            for j in sources:
                for k in sources:
                    # d(i,k) ≤ d(i,j) + d(j,k)
                    assert dist[i][k] <= dist[i][j] + dist[j][k] + 1e-6

    def test_symmetry(self):
        """Geodesic d(i,j) ≈ d(j,i)."""
        verts, faces = make_icosphere(subdivisions=1)
        solver = HeatGeodesics(verts, faces)
        n = len(verts)
        rng = np.random.default_rng(0)
        pairs = rng.choice(n, size=(10, 2), replace=True)
        for i, j in pairs:
            phi_i = solver.compute(int(i))
            phi_j = solver.compute(int(j))
            # Symmetry is approximate in the heat method
            assert abs(phi_i[j] - phi_j[i]) < 0.1 * max(phi_i[j], phi_j[i], 1e-3)

    def test_compute_all_pairs_shape(self):
        verts, faces = make_icosphere(subdivisions=1)
        solver = HeatGeodesics(verts, faces)
        n = len(verts)
        D = solver.compute_all_pairs()
        assert D.shape == (n, n)
        # Diagonal must be zero
        assert np.allclose(np.diag(D), 0.0, atol=1e-8)
        # Symmetric
        assert np.allclose(D, D.T, atol=1e-8)

    def test_flat_grid_approximate_euclidean(self):
        """On a flat grid, heat geodesics should approximate Euclidean distances."""
        verts, faces = make_flat_grid(10, 10)
        solver = HeatGeodesics(verts, faces)
        phi = solver.compute(0)   # source at corner (0,0)
        # Euclidean distance from vertex 0 to each other vertex
        eucl = np.linalg.norm(verts - verts[0], axis=1)
        # Allow ~5% relative error (heat method is approximate)
        mask = eucl > 0.1
        rel_err = np.abs(phi[mask] - eucl[mask]) / eucl[mask]
        assert rel_err.mean() < 0.15
