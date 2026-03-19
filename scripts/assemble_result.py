"""
Assemble the final result .glb from:
  1. Parameterized shared-vertex .glb (POSITION + TEXCOORD_0)
  2. OCC-split mesh .npz (vertices, triangles, per-vertex normals)

The OCC-split mesh has vertices duplicated at OCC face boundaries so that
normals are smooth within each face and sharp at boundaries. The UV coordinates
from the parameterized mesh are mapped onto the split mesh by matching vertex
positions.

Output: .glb with per-vertex OCC normals + UVs + checkerboard material.

Usage:
    python assemble_result.py <param.glb> <occmesh.npz> <output.glb> [--cell-size 25]
"""

import struct
import json
import math
import sys
import zlib
import numpy as np


def make_2x2_checkerboard_png():
    white = (240, 240, 240, 255)
    black = (40, 40, 40, 255)
    pixels = bytearray()
    pixels.append(0); pixels.extend(white); pixels.extend(black)
    pixels.append(0); pixels.extend(black); pixels.extend(white)
    def make_chunk(ct, data):
        chunk = ct + data
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n"
    png += make_chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 6, 0, 0, 0))
    png += make_chunk(b"IDAT", zlib.compress(bytes(pixels), 9))
    png += make_chunk(b"IEND", b"")
    return png


def read_glb(path):
    with open(path, "rb") as f:
        magic, version, length = struct.unpack("<4sII", f.read(12))
        assert magic == b"glTF" and version == 2
        json_len, json_type = struct.unpack("<II", f.read(8))
        assert json_type == 0x4E4F534A
        gltf = json.loads(f.read(json_len))
        bin_data = b""
        if f.tell() < length:
            bin_len, bin_type = struct.unpack("<II", f.read(8))
            assert bin_type == 0x004E4942
            bin_data = f.read(bin_len)
    return gltf, bin_data


def extract_shared_mesh(gltf, buf):
    """Extract shared-vertex positions and UVs from parameterized glb."""
    prim = gltf["meshes"][0]["primitives"][0]

    pos_acc = gltf["accessors"][prim["attributes"]["POSITION"]]
    pos_bv = gltf["bufferViews"][pos_acc["bufferView"]]
    pos_off = pos_bv.get("byteOffset", 0) + pos_acc.get("byteOffset", 0)
    n = pos_acc["count"]

    positions = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        positions[i] = struct.unpack_from("<3f", buf, pos_off + i * 12)

    uvs = None
    if "TEXCOORD_0" in prim["attributes"]:
        uv_acc = gltf["accessors"][prim["attributes"]["TEXCOORD_0"]]
        uv_bv = gltf["bufferViews"][uv_acc["bufferView"]]
        uv_off = uv_bv.get("byteOffset", 0) + uv_acc.get("byteOffset", 0)
        uvs = np.zeros((n, 2), dtype=np.float32)
        for i in range(n):
            uvs[i] = struct.unpack_from("<2f", buf, uv_off + i * 8)

    return positions, uvs


def map_uvs_to_split(shared_pos, shared_uvs, split_pos):
    """Map UVs from shared-vertex mesh to OCC-split mesh by nearest position.

    Each split vertex came from a shared vertex (same position). Build a
    spatial lookup to find the shared vertex for each split vertex.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(shared_pos)
    _, indices = tree.query(split_pos)

    split_uvs = shared_uvs[indices]
    return split_uvs


def write_result_glb(path, vertices, normals, uvs, triangles, cell_size_mm=25.0):
    """Write final .glb with per-vertex OCC normals + UVs + checkerboard."""
    n = len(vertices)
    m = len(triangles)

    min_pos = vertices.min(axis=0).tolist()
    max_pos = vertices.max(axis=0).tolist()
    extent = [max_pos[k] - min_pos[k] for k in range(3)]
    max_extent = max(extent)
    tile_mm = 2.0 * cell_size_mm
    if max_extent > 10:  # mesh in mm
        uv_scale = max_extent / tile_mm
    else:  # small/normalized units — fixed 4 tiles
        uv_scale = 4.0

    # Binary buffer
    pos_data = vertices.astype(np.float32).tobytes()
    nrm_data = normals.astype(np.float32).tobytes()
    uv_data = uvs.astype(np.float32).tobytes() if uvs is not None else b""
    idx_data = triangles.astype(np.uint32).tobytes()
    png_data = make_2x2_checkerboard_png()

    buf = bytearray()
    pos_off = len(buf); buf += pos_data
    nrm_off = len(buf); buf += nrm_data
    uv_off = len(buf); uv_len = len(uv_data)
    if uv_data: buf += uv_data
    idx_off = len(buf); buf += idx_data
    while len(buf) % 4 != 0: buf += b"\x00"
    png_off = len(buf); buf += png_data
    while len(buf) % 4 != 0: buf += b"\x00"

    buffer_views = [
        {"buffer": 0, "byteOffset": pos_off, "byteLength": len(pos_data), "target": 34962},
        {"buffer": 0, "byteOffset": nrm_off, "byteLength": len(nrm_data), "target": 34962},
    ]
    accessors = [
        {"bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3",
         "min": min_pos, "max": max_pos},
        {"bufferView": 1, "componentType": 5126, "count": n, "type": "VEC3"},
    ]
    attributes = {"POSITION": 0, "NORMAL": 1}

    if uvs is not None:
        uv_bv_idx = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": uv_off, "byteLength": uv_len, "target": 34962})
        uv_acc_idx = len(accessors)
        accessors.append({"bufferView": uv_bv_idx, "componentType": 5126, "count": n, "type": "VEC2"})
        attributes["TEXCOORD_0"] = uv_acc_idx

    idx_bv_idx = len(buffer_views)
    buffer_views.append({"buffer": 0, "byteOffset": idx_off, "byteLength": len(idx_data), "target": 34963})
    idx_acc_idx = len(accessors)
    accessors.append({"bufferView": idx_bv_idx, "componentType": 5125, "count": m * 3, "type": "SCALAR"})

    png_bv_idx = len(buffer_views)
    buffer_views.append({"buffer": 0, "byteOffset": png_off, "byteLength": len(png_data)})

    gltf = {
        "asset": {"version": "2.0", "generator": "MeshParameterization"},
        "extensionsUsed": ["KHR_texture_transform"],
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": attributes, "indices": idx_acc_idx, "mode": 4, "material": 0}]}],
        "accessors": accessors, "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(buf)}],
        "images": [{"bufferView": png_bv_idx, "mimeType": "image/png"}],
        "samplers": [{"magFilter": 9728, "minFilter": 9728, "wrapS": 33648, "wrapT": 33648}],
        "textures": [{"sampler": 0, "source": 0}],
        "materials": [{"name": "checkerboard_25mm", "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0, "extensions": {"KHR_texture_transform": {"scale": [uv_scale, uv_scale]}}},
            "metallicFactor": 0.0, "roughnessFactor": 1.0}}],
    }

    json_str = json.dumps(gltf, separators=(",", ":"))
    json_bytes = json_str.encode("utf-8")
    while len(json_bytes) % 4 != 0: json_bytes += b" "

    total = 12 + 8 + len(json_bytes) + 8 + len(buf)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        f.write(struct.pack("<II", len(buf), 0x004E4942))
        f.write(bytes(buf))

    print(f"  Wrote {path} ({n} verts, {m} tris, uv_scale={uv_scale:.2f})")


def main():
    if len(sys.argv) < 4:
        print("Usage: python assemble_result.py <param.glb> <occmesh.npz> <output.glb> [--cell-size 25]")
        sys.exit(1)

    param_path = sys.argv[1]
    occ_path = sys.argv[2]
    output_path = sys.argv[3]
    cell_size = 25.0
    for i, arg in enumerate(sys.argv):
        if arg == "--cell-size" and i + 1 < len(sys.argv):
            cell_size = float(sys.argv[i + 1])

    # Load parameterized shared-vertex mesh (positions + UVs)
    gltf, buf = read_glb(param_path)
    shared_pos, shared_uvs = extract_shared_mesh(gltf, buf)

    # Load OCC-split mesh (split at face boundaries, with per-vertex OCC normals)
    occ = np.load(occ_path)
    split_verts = occ["vertices"]
    split_tris = occ["triangles"]
    split_normals = occ["normals"]

    print(f"  Shared: {len(shared_pos)} verts | Split: {len(split_verts)} verts, {len(split_tris)} tris")

    # Map UVs from shared mesh onto split mesh
    split_uvs = None
    if shared_uvs is not None:
        split_uvs = map_uvs_to_split(shared_pos, shared_uvs, split_verts)

    write_result_glb(output_path, split_verts, split_normals, split_uvs, split_tris, cell_size)


if __name__ == "__main__":
    main()
