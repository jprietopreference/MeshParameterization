"""
mesh_parameterization
=====================
Quasi-isometric UV parameterization for triangulated meshes using
heat-based geodesics and classical multidimensional scaling (MDS).

Based on:
  "Quasi-Isometric Mesh Parameterization Using Heat-Based Geodesics
   and Poisson Surface Fills"
  MDPI Mathematics 7(8):753, 2019.
  https://www.mdpi.com/2227-7390/7/8/753
"""

from .laplacian import cotangent_laplacian
from .heat_geodesics import HeatGeodesics
from .hole_filling import fill_holes
from .mds import classical_mds
from .parameterize import parameterize_mesh

__all__ = [
    "cotangent_laplacian",
    "HeatGeodesics",
    "fill_holes",
    "classical_mds",
    "parameterize_mesh",
]
