"""
Generate glTF test meshes using Gmsh (OpenCascade kernel) + isotropic meshing.

Shapes:
  1. 100mm cube (sharp edges)
  2. 100mm cube with 25mm filleted edges
  3. 100mm radius sphere
  4. Torus R=50mm r=15mm
  5. Torus R=200mm r=40mm
  6. Klein bottle (figure-8 immersion, triangulated parametrically)

All meshes are exported as .glb files WITHOUT UV coordinates.
"""

import gmsh
import struct
import json
import math
import os
import sys
import numpy as np


def compute_vertex_normals(vertices, triangles):
    """Compute area-weighted per-vertex normals."""
    normals = [[0.0, 0.0, 0.0] for _ in vertices]
    for tri in triangles:
        v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        e1 = [v1[i] - v0[i] for i in range(3)]
        e2 = [v2[i] - v0[i] for i in range(3)]
        # Cross product (area-weighted normal)
        nx = e1[1]*e2[2] - e1[2]*e2[1]
        ny = e1[2]*e2[0] - e1[0]*e2[2]
        nz = e1[0]*e2[1] - e1[1]*e2[0]
        for vi in tri:
            normals[vi][0] += nx
            normals[vi][1] += ny
            normals[vi][2] += nz
    # Normalize
    for i, n in enumerate(normals):
        length = math.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
        if length > 1e-16:
            normals[i] = [n[0]/length, n[1]/length, n[2]/length]
        else:
            normals[i] = [0.0, 0.0, 1.0]
    return normals


def write_glb(path, vertices, triangles, vertex_normals=None):
    """Write a .glb for the parameterizer + per-vertex OCC normals sidecar.

    The mesh coming from extract_gmsh_mesh() is already split at OCC face
    boundaries. The .glb has POSITION + indices (shared within each OCC face).
    The parameterizer needs connected shared vertices, so we also write a
    .paramesh.bin sidecar with the original Gmsh shared-vertex mesh for it.

    The per-vertex OCC normals are saved as .vertexnormals.bin (n * 3 * float32).
    These normals are per-vertex of the OCC-split mesh (smooth within faces,
    sharp at face boundaries).
    """
    n = len(vertices)
    m = len(triangles)

    # Save per-vertex OCC normals
    if vertex_normals is not None and len(vertex_normals) == n:
        nrm_path = path.replace(".glb", ".vertexnormals.bin")
        with open(nrm_path, "wb") as f:
            for vn in vertex_normals:
                f.write(struct.pack("<3f", vn[0], vn[1], vn[2]))

    pos_data = b""
    min_pos = [float("inf")] * 3
    max_pos = [float("-inf")] * 3
    for v in vertices:
        pos_data += struct.pack("<3f", v[0], v[1], v[2])
        for i in range(3):
            min_pos[i] = min(min_pos[i], v[i])
            max_pos[i] = max(max_pos[i], v[i])

    idx_data = b""
    for tri in triangles:
        idx_data += struct.pack("<3I", tri[0], tri[1], tri[2])

    buf_data = pos_data + idx_data
    while len(buf_data) % 4 != 0:
        buf_data += b"\x00"

    pos_len = len(pos_data)
    idx_offset = pos_len
    idx_len = len(idx_data)

    gltf = {
        "asset": {"version": "2.0", "generator": "generate_test_meshes.py"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3",
             "min": min_pos, "max": max_pos},
            {"bufferView": 1, "componentType": 5125, "count": m * 3, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": pos_len, "target": 34962},
            {"buffer": 0, "byteOffset": idx_offset, "byteLength": idx_len, "target": 34963},
        ],
        "buffers": [{"byteLength": len(buf_data)}],
    }

    json_str = json.dumps(gltf, separators=(",", ":"))
    json_bytes = json_str.encode("utf-8")
    while len(json_bytes) % 4 != 0:
        json_bytes += b" "

    total_length = 12 + 8 + len(json_bytes) + 8 + len(buf_data)

    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total_length))
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        f.write(struct.pack("<II", len(buf_data), 0x004E4942))
        f.write(buf_data)

    print(f"  Wrote {path}: {n} vertices, {m} triangles")


def extract_gmsh_mesh():
    """Extract mesh split at OCC face boundaries with per-vertex OCC normals.

    Within each OCC face, vertices are shared and get the analytical surface
    normal at their parametric position → smooth shading on curved faces.
    At OCC face boundaries, vertices are duplicated with normals from each
    respective face → sharp creases at face edges.

    Also writes a per-triangle face-normal sidecar for the assembly step.

    Returns (vertices, triangles, face_normals) where:
      - vertices: list of [x,y,z] (split at OCC face boundaries)
      - triangles: indexing into vertices
      - face_normals: per-vertex [nx,ny,nz] (one per vertex, NOT one per triangle)
                      Written as sidecar: per-vertex normals, len = len(vertices)
    """
    # Get original shared nodes
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    tag_to_pos = {}
    for i, tag in enumerate(node_tags):
        tag_to_pos[tag] = [coords[3*i], coords[3*i+1], coords[3*i+2]]

    surfaces = gmsh.model.getEntities(dim=2)

    # Build per-surface node parametric coordinates
    node_params = {}  # node_tag → {surf_tag: (u, v)}
    for dim, surf_tag in surfaces:
        surf_node_tags, _, surf_params = gmsh.model.mesh.getNodes(
            dim=2, tag=surf_tag, includeBoundary=True)
        for i, ntag in enumerate(surf_node_tags):
            if ntag not in node_params:
                node_params[ntag] = {}
            node_params[ntag][surf_tag] = (surf_params[2*i], surf_params[2*i+1])

    # For each OCC face, create a separate set of vertices.
    # A node shared between two OCC faces gets duplicated — once per face,
    # each with the normal from that face.
    vertices = []     # exploded vertex positions
    normals = []      # per-vertex OCC normals
    triangles = []    # indices into vertices

    for dim, surf_tag in surfaces:
        # Map: original node_tag → new vertex index for THIS OCC face
        face_vertex_map = {}

        # Get all nodes on this face (including boundary nodes shared with other faces)
        surf_node_tags, _, surf_params = gmsh.model.mesh.getNodes(
            dim=2, tag=surf_tag, includeBoundary=True)

        # Batch query normals for all nodes on this face at once
        if len(surf_node_tags) > 0:
            nrm_flat = gmsh.model.getNormal(surf_tag, surf_params)
        else:
            nrm_flat = []

        for i, ntag in enumerate(surf_node_tags):
            idx = len(vertices)
            face_vertex_map[ntag] = idx
            vertices.append(tag_to_pos[ntag])
            normals.append([nrm_flat[3*i], nrm_flat[3*i+1], nrm_flat[3*i+2]])

        # Get triangles on this face
        elem_types, elem_tags, node_tags_per_elem = gmsh.model.mesh.getElements(
            dim=2, tag=surf_tag)
        for etype, etags, ntags in zip(elem_types, elem_tags, node_tags_per_elem):
            if etype != 2:
                continue
            for i in range(len(etags)):
                nt = [ntags[3*i+j] for j in range(3)]
                tri = [face_vertex_map[t] for t in nt]
                triangles.append(tri)

    # Also extract the original shared-vertex mesh for the parameterizer.
    # The parameterizer needs fully connected vertices (no OCC face splits).
    shared_tag_to_idx = {}
    shared_vertices = []
    for tag in node_tags:
        shared_tag_to_idx[tag] = len(shared_vertices)
        shared_vertices.append(tag_to_pos[tag])

    shared_triangles = []
    for dim, surf_tag in surfaces:
        elem_types, elem_tags, node_tags_per_elem = gmsh.model.mesh.getElements(
            dim=2, tag=surf_tag)
        for etype, etags, ntags in zip(elem_types, elem_tags, node_tags_per_elem):
            if etype != 2:
                continue
            for i in range(len(etags)):
                nt = [ntags[3*i+j] for j in range(3)]
                shared_triangles.append([shared_tag_to_idx[t] for t in nt])

    return {
        "shared_vertices": shared_vertices,
        "shared_triangles": shared_triangles,
        "split_vertices": vertices,
        "split_triangles": triangles,
        "split_normals": normals,  # per-vertex of split mesh
    }


def mesh_occ_shape(chord_error=1.0, max_edge=50.0):
    """Mesh OCC shape with uniform chord deviation (sag).

    chord_error: max allowed deviation from true surface (mm)
    max_edge: max edge length on flat regions (mm)

    For a surface with radius R, the mesh size h for chord error δ is:
      h = sqrt(8 * R * δ)
    Gmsh's MeshSizeFromCurvature=N means h = 2πR/N, so:
      N = 2πR / sqrt(8Rδ) = π * sqrt(2R/δ)
    We use N=40 which gives ~1mm error for R≥13mm (covers all our fillets).
    """
    # Curvature-based sizing: ~40 elements per 2π of curvature
    # This gives chord error ≈ R*(2π/40)²/8 ≈ R*0.025
    # For R=25mm fillet: error ≈ 0.6mm. For R=100mm sphere: error ≈ 2.5mm
    # We cap with max_edge to keep flat areas reasonable.
    #
    # More precise: solve for N from δ = R*(π/N)² → N = π*sqrt(R/δ)
    # For δ=1mm, R=15mm(torus): N=12, R=25mm(fillet): N=16, R=100mm(sphere): N=31
    # Use N=32 as a good compromise.
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 32)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_edge)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", chord_error * 0.5)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.model.mesh.generate(2)


def make_cube_gmsh(size=100.0):
    """100mm cube with isotropic mesh."""
    gmsh.initialize()
    gmsh.model.add("cube")
    gmsh.model.occ.addBox(-size/2, -size/2, -size/2, size, size, size)
    gmsh.model.occ.synchronize()
    mesh_occ_shape()
    mesh_data = extract_gmsh_mesh()
    gmsh.finalize()
    return mesh_data


def make_filleted_cube_gmsh(size=100.0, fillet_radius=25.0):
    """100mm cube with 25mm filleted edges, isotropic mesh."""
    gmsh.initialize()
    gmsh.model.add("filleted_cube")
    box = gmsh.model.occ.addBox(-size/2, -size/2, -size/2, size, size, size)
    # Get all edges of the box and fillet them
    edges = gmsh.model.occ.getEntities(dim=1)
    edge_tags = [e[1] for e in edges]
    gmsh.model.occ.fillet([box], edge_tags, [fillet_radius])
    gmsh.model.occ.synchronize()
    mesh_occ_shape()
    mesh_data = extract_gmsh_mesh()
    gmsh.finalize()
    return mesh_data


def make_sphere_gmsh(radius=100.0):
    """Icosphere-quality sphere using Gmsh OCC kernel."""
    gmsh.initialize()
    gmsh.model.add("sphere")
    gmsh.model.occ.addSphere(0, 0, 0, radius)
    gmsh.model.occ.synchronize()
    mesh_occ_shape()
    mesh_data = extract_gmsh_mesh()
    gmsh.finalize()
    return mesh_data


def make_torus_gmsh(R, r):
    """Torus with major radius R and minor radius r."""
    gmsh.initialize()
    gmsh.model.add("torus")
    gmsh.model.occ.addTorus(0, 0, 0, R, r)
    gmsh.model.occ.synchronize()
    mesh_occ_shape()
    mesh_data = extract_gmsh_mesh()
    gmsh.finalize()
    return mesh_data


def make_klein_bottle(scale=50.0, u_segments=64, v_segments=32):
    """
    Klein bottle (figure-8 immersion in R³).
    No OCC primitive for this — use parametric triangulation.

    x = (a + cos(v/2)*sin(u) - sin(v/2)*sin(2u)) * cos(v)
    y = (a + cos(v/2)*sin(u) - sin(v/2)*sin(2u)) * sin(v)
    z = sin(v/2)*sin(u) + cos(v/2)*sin(2u)
    """
    a = 2.0
    vertices = []
    triangles = []

    for j in range(v_segments + 1):
        v_angle = 2 * math.pi * j / v_segments
        for i in range(u_segments + 1):
            u_angle = 2 * math.pi * i / u_segments

            cos_v2 = math.cos(v_angle / 2)
            sin_v2 = math.sin(v_angle / 2)
            sin_u = math.sin(u_angle)
            sin_2u = math.sin(2 * u_angle)

            r = a + cos_v2 * sin_u - sin_v2 * sin_2u
            x = scale * r * math.cos(v_angle)
            y = scale * r * math.sin(v_angle)
            z = scale * (sin_v2 * sin_u + cos_v2 * sin_2u)
            vertices.append([x, y, z])

    for j in range(v_segments):
        for i in range(u_segments):
            v00 = j * (u_segments + 1) + i
            v10 = v00 + 1
            v01 = v00 + (u_segments + 1)
            v11 = v01 + 1
            triangles.append([v00, v10, v11])
            triangles.append([v00, v11, v01])

    return vertices, triangles


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(out_dir, exist_ok=True)

    print("Generating test meshes (Gmsh + OpenCascade)...")

    def write_occ_mesh(name, mesh_data):
        """Write shared-vertex .glb (for parameterizer) + OCC normals sidecar."""
        glb_path = os.path.join(out_dir, f"{name}.glb")
        # Write shared-vertex .glb for the parameterizer (no normals)
        write_glb(glb_path,
                  mesh_data["shared_vertices"],
                  mesh_data["shared_triangles"])
        # Write OCC-split mesh with per-vertex normals as sidecar .npz
        # The assembly step will use this for the final result.
        split_path = os.path.join(out_dir, f"{name}.occmesh.npz")
        np.savez_compressed(split_path,
            vertices=np.array(mesh_data["split_vertices"], dtype=np.float32),
            triangles=np.array(mesh_data["split_triangles"], dtype=np.int32),
            normals=np.array(mesh_data["split_normals"], dtype=np.float32),
        )
        sv = len(mesh_data["shared_vertices"])
        st = len(mesh_data["shared_triangles"])
        spv = len(mesh_data["split_vertices"])
        print(f"    Shared: {sv} verts, {st} tris | Split: {spv} verts")

    # 1. 100mm cube
    print("\n[1/6] Cube 100mm")
    write_occ_mesh("cube_100mm", make_cube_gmsh(100.0))

    # 2. 100mm cube with 25mm fillet
    print("\n[2/6] Filleted cube 100mm, R=25mm")
    write_occ_mesh("cube_100mm_fillet_25mm", make_filleted_cube_gmsh(100.0, 25.0))

    # 3. 100mm radius sphere
    print("\n[3/6] Sphere R=100mm")
    write_occ_mesh("sphere_R100mm", make_sphere_gmsh(100.0))

    # 4. Torus R=50mm, r=15mm
    print("\n[4/6] Torus R=50mm, r=15mm")
    write_occ_mesh("torus_R50mm_r15mm", make_torus_gmsh(50.0, 15.0))

    # 5. Torus R=200mm, r=40mm
    print("\n[5/6] Torus R=200mm, r=40mm")
    write_occ_mesh("torus_R200mm_r40mm", make_torus_gmsh(200.0, 40.0))

    # 6. Klein bottle (parametric — no OCC primitive)
    print("\n[6/6] Klein bottle")
    v, f = make_klein_bottle(50.0, u_segments=64, v_segments=32)
    write_glb(os.path.join(out_dir, "klein_bottle.glb"), v, f)

    print(f"\nAll meshes generated in: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
