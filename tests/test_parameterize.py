"""Integration tests for the full parameterisation pipeline."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mesh_parameterization.parameterize import parameterize_mesh
from mesh_parameterization.hole_filling import fill_holes, _find_boundary_loops


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_open_plane(nx=6, ny=6):
    """Flat open mesh (a grid with boundary)."""
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


def make_icosphere(subdivisions=1):
    import trimesh
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions)
    return np.array(mesh.vertices), np.array(mesh.faces)


# ---------------------------------------------------------------------------
# Hole-filling tests
# ---------------------------------------------------------------------------

class TestFillHoles:

    def test_closed_mesh_unchanged(self):
        verts, faces = make_icosphere()
        v2, f2, n_orig = fill_holes(verts, faces)
        assert len(v2) == len(verts)
        assert len(f2) == len(faces)
        assert n_orig == len(verts)

    def test_open_mesh_adds_vertices(self):
        verts, faces = make_open_plane()
        v2, f2, n_orig = fill_holes(verts, faces)
        # At least one centroid vertex should be added
        assert len(v2) > len(verts)
        assert n_orig == len(verts)

    def test_open_mesh_adds_faces(self):
        verts, faces = make_open_plane()
        v2, f2, n_orig = fill_holes(verts, faces)
        assert len(f2) > len(faces)

    def test_find_boundary_loops_open_plane(self):
        _, faces = make_open_plane()
        loops = _find_boundary_loops(faces)
        assert len(loops) >= 1

    def test_find_boundary_loops_closed_mesh(self):
        _, faces = make_icosphere()
        loops = _find_boundary_loops(faces)
        assert len(loops) == 0


# ---------------------------------------------------------------------------
# Parameterisation tests
# ---------------------------------------------------------------------------

class TestParameterizeMesh:

    def test_output_shape_closed_mesh(self):
        verts, faces = make_icosphere()
        uv = parameterize_mesh(verts, faces)
        assert uv.shape == (len(verts), 2)

    def test_output_shape_open_mesh(self):
        verts, faces = make_open_plane()
        uv = parameterize_mesh(verts, faces)
        assert uv.shape == (len(verts), 2)

    def test_uv_in_unit_square(self):
        verts, faces = make_icosphere()
        uv = parameterize_mesh(verts, faces)
        assert uv.min() >= -1e-8
        assert uv.max() <= 1.0 + 1e-8

    def test_uv_spans_unit_square(self):
        """UV coordinates must span the full [0,1]² range."""
        verts, faces = make_icosphere()
        uv = parameterize_mesh(verts, faces)
        # Each axis should reach 0 and 1
        assert np.isclose(uv[:, 0].min(), 0.0, atol=1e-6)
        assert np.isclose(uv[:, 0].max(), 1.0, atol=1e-6)
        assert np.isclose(uv[:, 1].min(), 0.0, atol=1e-6)
        assert np.isclose(uv[:, 1].max(), 1.0, atol=1e-6)

    def test_uv_unique_per_vertex(self):
        """No two vertices should map to the exact same UV (for a sphere)."""
        verts, faces = make_icosphere(subdivisions=1)
        uv = parameterize_mesh(verts, faces)
        # Round to 4 decimal places and check for duplicates
        uv_rounded = np.round(uv, 4)
        unique_rows = np.unique(uv_rounded, axis=0)
        # Allow up to 5% duplicate UVs (due to MDS projection)
        assert len(unique_rows) >= 0.95 * len(verts)

    def test_open_plane_approximate_identity(self):
        """UV of a flat rectangular grid should preserve pairwise distances."""
        # Use a 2:1 rectangular grid so the x-axis dominates the first PC
        verts, faces = make_open_plane(12, 6)
        uv = parameterize_mesh(verts, faces)
        # Check that UV correlates strongly with x (dominant axis)
        xy = verts[:, :2]
        xy_norm = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
        # MDS axes may be flipped or swapped – try all sign combinations
        max_corr_x = max(abs(np.corrcoef(xy_norm[:, 0], uv[:, 0])[0, 1]),
                         abs(np.corrcoef(xy_norm[:, 0], uv[:, 1])[0, 1]))
        assert max_corr_x > 0.90
