#!/usr/bin/env python3
"""
main.py – CLI for quasi-isometric UV parameterisation of GLTF meshes.

Usage
-----
    python main.py <input.gltf|input.glb> <output.gltf|output.glb> [OPTIONS]

Options
-------
    --t-coeff FLOAT   Heat-equation time-step coefficient (default: 1.0).
    --max-verts INT   Max vertices for full all-pairs MDS (default: 2000).
    --verbose         Print progress information.

Description
-----------
Parameterises every triangle mesh in the GLTF scene using the algorithm of:

  "Quasi-Isometric Mesh Parameterization Using Heat-Based Geodesics and
   Poisson Surface Fills", MDPI Mathematics 7(8):753, 2019.

The pipeline:
  1. Fill boundary holes with fan triangulation.
  2. Compute pairwise geodesic distances via the heat method (Crane 2013).
  3. Embed vertices in 2-D with classical MDS.
  4. Normalise to [0, 1]² and write back as TEXCOORD_0.
"""

import argparse
import sys

import numpy as np
import trimesh

from src.mesh_parameterization import parameterize_mesh


def _process_mesh(mesh, t_coeff, max_verts, verbose):
    """Return a copy of *mesh* with TEXCOORD_0 UV coordinates added."""
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    if verbose:
        print(f"  vertices={len(vertices)}, faces={len(faces)}", flush=True)

    uv = parameterize_mesh(vertices, faces, t_coeff=t_coeff,
                           max_allpairs=max_verts)

    # Assign UV as TextureVisuals (keeps the existing color, if any)
    new_mesh = mesh.copy()
    new_mesh.visual = trimesh.visual.TextureVisuals(uv=uv)
    return new_mesh


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Parameterise GLTF meshes using heat-based geodesics + MDS."
    )
    parser.add_argument("input",  help="Input GLTF/GLB file path.")
    parser.add_argument("output", help="Output GLTF/GLB file path.")
    parser.add_argument("--t-coeff", type=float, default=1.0,
                        help="Heat time-step coefficient (default: 1.0).")
    parser.add_argument("--max-verts", type=int, default=2000,
                        help="Max vertices for full all-pairs MDS (default: 2000).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress information.")
    args = parser.parse_args(argv)

    if args.verbose:
        print(f"Loading {args.input} …")

    scene = trimesh.load(args.input, force="scene")

    processed = {}
    for name, geom in scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            processed[name] = geom
            continue
        if args.verbose:
            print(f"Parameterising mesh '{name}' …")
        processed[name] = _process_mesh(geom, args.t_coeff, args.max_verts,
                                        args.verbose)

    # Rebuild scene with processed geometry
    new_scene = trimesh.scene.scene.Scene(
        geometry=processed,
        graph=scene.graph,
    )

    if args.verbose:
        print(f"Saving {args.output} …")

    new_scene.export(args.output)

    if args.verbose:
        print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
