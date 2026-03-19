"""
Apply a 25mm checkerboard texture to parameterized .glb meshes.

Uses a single 2x2 pixel B/W texture with MIRRORED_REPEAT wrapping.
One texture tile = 50mm (two 25mm cells), mirroring produces the
alternating pattern. UV coordinates are scaled via KHR_texture_transform
so that 50mm of physical surface = 1 UV tile.

Usage:
    python apply_checkerboard.py <input.glb> <output.glb> [--cell-size 25]
"""

import struct
import json
import sys
import zlib


def make_2x2_checkerboard_png():
    """2x2 pixel B/W checkerboard PNG. ~70 bytes total."""
    white = (240, 240, 240, 255)
    black = (40, 40, 40, 255)

    # 2x2 RGBA image
    pixels = bytearray()
    # Row 0: white, black
    pixels.append(0)  # PNG filter: None
    pixels.extend(white)
    pixels.extend(black)
    # Row 1: black, white
    pixels.append(0)
    pixels.extend(black)
    pixels.extend(white)

    def make_chunk(chunk_type, data):
        chunk = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + chunk + crc

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


def write_glb(path, gltf, bin_data):
    json_str = json.dumps(gltf, separators=(",", ":"))
    json_bytes = json_str.encode("utf-8")
    while len(json_bytes) % 4 != 0:
        json_bytes += b" "
    while len(bin_data) % 4 != 0:
        bin_data += b"\x00"
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        f.write(struct.pack("<II", len(bin_data), 0x004E4942))
        f.write(bin_data)


def get_mesh_extent(gltf):
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            pos_idx = prim["attributes"].get("POSITION")
            if pos_idx is not None:
                acc = gltf["accessors"][pos_idx]
                if "min" in acc and "max" in acc:
                    return [acc["max"][i] - acc["min"][i] for i in range(3)]
    return [100.0, 100.0, 100.0]


def compute_smooth_normals(gltf, buf):
    """Compute area-weighted per-vertex normals from positions + indices."""
    import math
    prim = gltf["meshes"][0]["primitives"][0]
    pos_acc = gltf["accessors"][prim["attributes"]["POSITION"]]
    pos_bv = gltf["bufferViews"][pos_acc["bufferView"]]
    pos_off = pos_bv.get("byteOffset", 0) + pos_acc.get("byteOffset", 0)
    n = pos_acc["count"]

    positions = []
    for i in range(n):
        x, y, z = struct.unpack_from("<3f", buf, pos_off + i * 12)
        positions.append((x, y, z))

    idx_acc = gltf["accessors"][prim["indices"]]
    idx_bv = gltf["bufferViews"][idx_acc["bufferView"]]
    idx_off = idx_bv.get("byteOffset", 0) + idx_acc.get("byteOffset", 0)
    ct = idx_acc["componentType"]
    indices = []
    for i in range(idx_acc["count"]):
        if ct == 5125:
            indices.append(struct.unpack_from("<I", buf, idx_off + i * 4)[0])
        elif ct == 5123:
            indices.append(struct.unpack_from("<H", buf, idx_off + i * 2)[0])
        else:
            indices.append(buf[idx_off + i])

    normals = [[0.0, 0.0, 0.0] for _ in range(n)]
    for i in range(0, len(indices), 3):
        v0, v1, v2 = indices[i], indices[i+1], indices[i+2]
        p0, p1, p2 = positions[v0], positions[v1], positions[v2]
        e1 = [p1[j]-p0[j] for j in range(3)]
        e2 = [p2[j]-p0[j] for j in range(3)]
        nx = e1[1]*e2[2] - e1[2]*e2[1]
        ny = e1[2]*e2[0] - e1[0]*e2[2]
        nz = e1[0]*e2[1] - e1[1]*e2[0]
        for vi in (v0, v1, v2):
            normals[vi][0] += nx; normals[vi][1] += ny; normals[vi][2] += nz
    for i in range(n):
        l = math.sqrt(sum(c*c for c in normals[i]))
        if l > 1e-16:
            normals[i] = [c/l for c in normals[i]]
        else:
            normals[i] = [0, 0, 1]
    return normals


def apply_checkerboard(input_path, output_path, cell_size_mm=25.0):
    gltf, bin_data = read_glb(input_path)
    bin_data = bytearray(bin_data)

    # Add smooth vertex normals if missing
    prim = gltf["meshes"][0]["primitives"][0]
    if "NORMAL" not in prim["attributes"]:
        normals = compute_smooth_normals(gltf, bytes(bin_data))
        while len(bin_data) % 4 != 0:
            bin_data.append(0)
        nrm_offset = len(bin_data)
        nrm_data = b"".join(struct.pack("<3f", *n) for n in normals)
        bin_data.extend(nrm_data)

        gltf["buffers"][0]["byteLength"] = len(bin_data)
        nrm_bv_idx = len(gltf["bufferViews"])
        gltf["bufferViews"].append({
            "buffer": 0, "byteOffset": nrm_offset,
            "byteLength": len(nrm_data), "target": 34962})
        nrm_acc_idx = len(gltf["accessors"])
        n_verts = gltf["accessors"][prim["attributes"]["POSITION"]]["count"]
        gltf["accessors"].append({
            "bufferView": nrm_bv_idx, "componentType": 5126,
            "count": n_verts, "type": "VEC3"})
        prim["attributes"]["NORMAL"] = nrm_acc_idx

    extent = get_mesh_extent(gltf)
    max_extent = max(extent)

    # UV scale: number of texture tile repeats across UV [0,1].
    # 2x2 texture + MIRRORED_REPEAT → 1 tile = 2 checker cells.
    # For meshes in mm: scale so each cell = cell_size_mm.
    # For small meshes (not in mm): use a fixed scale for visibility.
    tile_size_mm = 2.0 * cell_size_mm  # 50mm per tile
    if max_extent > 10:  # mesh is in mm
        uv_scale = max_extent / tile_size_mm
    else:  # mesh in normalized/other units — use fixed 4 tiles (8 cells)
        uv_scale = 4.0

    print(f"  Mesh extent: {extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f}, UV scale = {uv_scale:.2f}")

    # Embed the tiny 2x2 PNG
    png_data = make_2x2_checkerboard_png()
    while len(bin_data) % 4 != 0:
        bin_data.append(0)
    png_offset = len(bin_data)
    bin_data.extend(png_data)
    while len(bin_data) % 4 != 0:
        bin_data.append(0)

    gltf["buffers"][0]["byteLength"] = len(bin_data)

    # BufferView for the PNG
    png_bv_idx = len(gltf["bufferViews"])
    gltf["bufferViews"].append({
        "buffer": 0,
        "byteOffset": png_offset,
        "byteLength": len(png_data),
    })

    # Image
    if "images" not in gltf:
        gltf["images"] = []
    img_idx = len(gltf["images"])
    gltf["images"].append({
        "bufferView": png_bv_idx,
        "mimeType": "image/png",
    })

    # Sampler: NEAREST + MIRRORED_REPEAT
    if "samplers" not in gltf:
        gltf["samplers"] = []
    sampler_idx = len(gltf["samplers"])
    gltf["samplers"].append({
        "magFilter": 9728,  # NEAREST
        "minFilter": 9728,  # NEAREST
        "wrapS": 33648,     # MIRRORED_REPEAT
        "wrapT": 33648,     # MIRRORED_REPEAT
    })

    # Texture
    if "textures" not in gltf:
        gltf["textures"] = []
    tex_idx = len(gltf["textures"])
    gltf["textures"].append({
        "sampler": sampler_idx,
        "source": img_idx,
    })

    # Register KHR_texture_transform extension
    if "extensionsUsed" not in gltf:
        gltf["extensionsUsed"] = []
    if "KHR_texture_transform" not in gltf["extensionsUsed"]:
        gltf["extensionsUsed"].append("KHR_texture_transform")

    # Material with texture transform to scale UVs
    if "materials" not in gltf:
        gltf["materials"] = []
    mat_idx = len(gltf["materials"])
    gltf["materials"].append({
        "name": "checkerboard_25mm",
        "pbrMetallicRoughness": {
            "baseColorTexture": {
                "index": tex_idx,
                "extensions": {
                    "KHR_texture_transform": {
                        "scale": [uv_scale, uv_scale],
                    }
                }
            },
            "metallicFactor": 0.0,
            "roughnessFactor": 1.0,
        },
    })

    # Assign material to all primitives
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            prim["material"] = mat_idx

    write_glb(output_path, gltf, bytes(bin_data))
    print(f"  Wrote {output_path}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python apply_checkerboard.py <input.glb> <output.glb> [--cell-size 25]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    cell_size = 25.0

    for i, arg in enumerate(sys.argv):
        if arg == "--cell-size" and i + 1 < len(sys.argv):
            cell_size = float(sys.argv[i + 1])

    print(f"Applying {cell_size}mm checkerboard to {input_path}...")
    apply_checkerboard(input_path, output_path, cell_size)


if __name__ == "__main__":
    main()
