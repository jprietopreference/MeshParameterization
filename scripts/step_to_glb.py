"""
Load a STEP file via Gmsh/OCC, mesh with 1mm chord deviation, export as .glb.

Usage: python step_to_glb.py <input.step> <output.glb> [--scale-to-mm FACTOR]
       FACTOR: multiply all coordinates by this to convert to mm (e.g. 25.4 for inches)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_test_meshes import extract_gmsh_mesh, write_glb, mesh_occ_shape

import gmsh
import numpy as np


def main():
    if len(sys.argv) < 3:
        print("Usage: python step_to_glb.py <input.step> <output.glb> [--scale-to-mm FACTOR]")
        sys.exit(1)

    step_path = sys.argv[1]
    output_path = sys.argv[2]
    scale_factor = 1.0

    for i, arg in enumerate(sys.argv):
        if arg == "--scale-to-mm" and i + 1 < len(sys.argv):
            scale_factor = float(sys.argv[i + 1])

    print(f"Loading STEP: {step_path}")
    gmsh.initialize()
    gmsh.model.add("step_import")
    gmsh.model.occ.importShapes(step_path)
    gmsh.model.occ.synchronize()

    # Mesh in native units, then scale coordinates after
    # (avoids OCC BSpline knot issues from pre-scaling)
    if scale_factor != 1.0:
        print(f"Will scale output by {scale_factor}x to mm after meshing...")
        # Adjust meshing parameters for native units
        native_chord = 1.0 / scale_factor  # 1mm in native units
        mesh_occ_shape(max_edge=50.0 / scale_factor)
    else:
        mesh_occ_shape()

    mesh_data = extract_gmsh_mesh()
    gmsh.finalize()

    # Scale vertex positions to mm
    if scale_factor != 1.0:
        for v in mesh_data["shared_vertices"]:
            for i in range(3):
                v[i] = float(v[i]) * scale_factor
        for v in mesh_data["split_vertices"]:
            for i in range(3):
                v[i] = float(v[i]) * scale_factor
        sv = mesh_data["shared_vertices"]
        mn = [min(v[i] for v in sv) for i in range(3)]
        mx = [max(v[i] for v in sv) for i in range(3)]
        print(f"  Scaled extent: {mx[0]-mn[0]:.1f} x {mx[1]-mn[1]:.1f} x {mx[2]-mn[2]:.1f} mm")

    write_glb(output_path,
              mesh_data["shared_vertices"],
              mesh_data["shared_triangles"])

    npz_path = output_path.replace(".glb", ".occmesh.npz")
    np.savez_compressed(npz_path,
        vertices=np.array(mesh_data["split_vertices"], dtype=np.float32),
        triangles=np.array(mesh_data["split_triangles"], dtype=np.int32),
        normals=np.array(mesh_data["split_normals"], dtype=np.float32),
    )

    sv = len(mesh_data["shared_vertices"])
    st = len(mesh_data["shared_triangles"])
    spv = len(mesh_data["split_vertices"])
    print(f"  Shared: {sv} verts, {st} tris | Split: {spv} verts")


if __name__ == "__main__":
    main()
