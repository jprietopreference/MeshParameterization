#!/usr/bin/env python3
"""Gmsh isotropic remeshing CLI: GLB → remeshed GLB.

Usage: gmsh_remesh_cli.py input.glb output.glb [--max-tris N] [--feature-angle DEG]
"""

import sys
import os
import struct
import json
import argparse
import numpy as np

def parse_glb(path):
    """Read positions and triangles from a GLB file."""
    with open(path, 'rb') as f:
        data = f.read()
    jl = struct.unpack('<I', data[12:16])[0]
    g = json.loads(data[20:20+jl])
    p = g['meshes'][0]['primitives'][0]

    bin_start = 20 + jl
    while bin_start % 4:
        bin_start += 1
    bin_start += 8

    # Positions
    pa = g['accessors'][p['attributes']['POSITION']]
    pbv = g['bufferViews'][pa['bufferView']]
    nv = pa['count']
    V = np.frombuffer(data, dtype=np.float32,
                      offset=bin_start + pbv.get('byteOffset', 0),
                      count=nv * 3).reshape(-1, 3).copy()

    # Indices
    ia = g['accessors'][p['indices']]
    ibv = g['bufferViews'][ia['bufferView']]
    ct = ia['componentType']
    nf = ia['count'] // 3
    if ct == 5125:
        F = np.frombuffer(data, dtype=np.uint32,
                          offset=bin_start + ibv.get('byteOffset', 0),
                          count=nf * 3).reshape(-1, 3).copy()
    elif ct == 5123:
        F = np.frombuffer(data, dtype=np.uint16,
                          offset=bin_start + ibv.get('byteOffset', 0),
                          count=nf * 3).reshape(-1, 3).astype(np.uint32).copy()
    else:
        raise ValueError(f"Unsupported index type {ct}")

    return V, F


def write_glb(path, V, F):
    """Write a shared-vertex GLB with positions + computed normals."""
    nv = len(V)
    nf = len(F)

    # Compute per-vertex normals (area-weighted face normals)
    N = np.zeros_like(V)
    for fi in range(nf):
        i0, i1, i2 = F[fi]
        e1 = V[i1] - V[i0]
        e2 = V[i2] - V[i0]
        fn = np.cross(e1, e2)
        N[i0] += fn
        N[i1] += fn
        N[i2] += fn
    norms = np.linalg.norm(N, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    N = (N / norms).astype(np.float32)

    # Binary buffer
    pos_data = V.astype(np.float32).tobytes()
    nrm_data = N.tobytes()
    idx_data = F.astype(np.uint32).tobytes()
    buf = pos_data + nrm_data + idx_data
    while len(buf) % 4:
        buf += b'\x00'

    mn = V.min(axis=0).tolist()
    mx = V.max(axis=0).tolist()

    gltf = {
        "asset": {"version": "2.0", "generator": "gmsh_remesh_cli"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "mode": 4}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": nv, "type": "VEC3", "min": mn, "max": mx},
            {"bufferView": 1, "componentType": 5126, "count": nv, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5125, "count": nf * 3, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_data), "target": 34962},
            {"buffer": 0, "byteOffset": len(pos_data), "byteLength": len(nrm_data), "target": 34962},
            {"buffer": 0, "byteOffset": len(pos_data) + len(nrm_data), "byteLength": len(idx_data), "target": 34963},
        ],
        "buffers": [{"byteLength": len(buf)}],
    }

    jb = json.dumps(gltf, separators=(',', ':')).encode()
    while len(jb) % 4:
        jb += b' '
    total = 12 + 8 + len(jb) + 8 + len(buf)

    with open(path, 'wb') as f:
        f.write(struct.pack('<4sII', b'glTF', 2, total))
        f.write(struct.pack('<II', len(jb), 0x4E4F534A))
        f.write(jb)
        f.write(struct.pack('<II', len(buf), 0x004E4942))
        f.write(buf)


def remesh(V, F, max_tris=5000, feature_angle=40.0):
    """Isotropic remesh using Gmsh."""
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("remesh")

    # Add discrete surface
    dim, tag = 2, 1
    gmsh.model.addDiscreteEntity(dim, tag)

    # Add nodes (1-based)
    node_tags = list(range(1, len(V) + 1))
    coords = V.flatten().tolist()
    gmsh.model.mesh.addNodes(dim, tag, node_tags, coords)

    # Add triangles
    nf = len(F)
    elem_tags = list(range(1, nf + 1))
    node_tags_flat = (F + 1).flatten().tolist()
    gmsh.model.mesh.addElementsByType(tag, 2, elem_tags, node_tags_flat)

    # Classify and create geometry for remeshing
    gmsh.model.mesh.classifySurfaces(
        feature_angle * 3.14159265 / 180.0,  # angle
        True,   # boundary
        True,   # forReparametrization
        2.0 * 3.14159265,  # curveAngle
        False   # exportDiscrete
    )
    gmsh.model.mesh.createGeometry()

    # Compute average edge length
    total_len = 0
    edge_count = 0
    for fi in range(nf):
        for e in range(3):
            a, b = F[fi, e], F[fi, (e + 1) % 3]
            d = V[a] - V[b]
            total_len += np.sqrt(d @ d)
            edge_count += 1
    avg_edge = total_len / edge_count

    # Scale up if over triangle limit
    h = avg_edge
    if nf > max_tris:
        h *= np.sqrt(nf / max_tris)

    # Set meshing parameters
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h * 2.0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)

    # Generate
    gmsh.model.mesh.generate(2)

    # Retry if over limit
    for _ in range(5):
        _, _, node_tags_out = gmsh.model.mesh.getElementsByType(2)
        out_nf = len(node_tags_out) // 3
        if out_nf <= max_tris or max_tris <= 0:
            break
        h *= np.sqrt(out_nf / max_tris) * 1.1
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h * 0.5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h * 2.0)
        gmsh.model.mesh.generate(2)

    # Extract result
    node_tags_all, coords_all, _ = gmsh.model.mesh.getNodes(-1, -1)
    max_tag = int(max(node_tags_all))
    tag_to_idx = np.full(max_tag + 1, -1, dtype=np.int32)
    out_nv = len(node_tags_all)
    out_V = np.zeros((out_nv, 3), dtype=np.float32)
    for i, t in enumerate(node_tags_all):
        tag_to_idx[int(t)] = i
        out_V[i] = coords_all[i * 3:(i + 1) * 3]

    _, _, elem_node_tags = gmsh.model.mesh.getElementsByType(2)
    out_nf = len(elem_node_tags) // 3
    out_F = np.zeros((out_nf, 3), dtype=np.uint32)
    for i in range(out_nf):
        for k in range(3):
            out_F[i, k] = tag_to_idx[int(elem_node_tags[i * 3 + k])]

    gmsh.finalize()

    return out_V, out_F


def main():
    parser = argparse.ArgumentParser(description='Gmsh isotropic remeshing')
    parser.add_argument('input', help='Input GLB')
    parser.add_argument('output', help='Output GLB')
    parser.add_argument('--max-tris', type=int, default=5000)
    parser.add_argument('--feature-angle', type=float, default=40.0)
    args = parser.parse_args()

    V, F = parse_glb(args.input)
    print(f"Input: {len(V)} verts, {len(F)} tris")

    out_V, out_F = remesh(V, F, args.max_tris, args.feature_angle)
    print(f"Output: {len(out_V)} verts, {len(out_F)} tris")

    write_glb(args.output, out_V, out_F)
    print(f"Wrote: {args.output}")


if __name__ == '__main__':
    main()
