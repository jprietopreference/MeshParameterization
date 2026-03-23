#!/usr/bin/env python3
"""Refined STEP → GLB pipeline using OCCT + Gmsh.

Pipeline: STEP → OCC import → heal → Gmsh isotropic mesh → OCC normals → GLB

Usage: python occ_gmsh_pipeline.py input.step output.glb [options]
"""

import argparse
import struct
import json
import sys
import os
import math
import numpy as np


def step_to_glb(input_path, output_path, chord_deviation=1.0, min_edge=None, max_edge=None,
                scale=None, heal=True):
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.model.add("pipeline")

    # 1. Import STEP
    print(f"[pipeline] Importing: {input_path}")
    try:
        shapes = gmsh.model.occ.importShapes(input_path)
    except Exception as e:
        print(f"[pipeline] Import failed: {e}")
        gmsh.finalize()
        return False

    gmsh.model.occ.synchronize()

    # Check units — detect if STEP is in inches
    # (Gmsh/OCC reads STEP in its native units)
    entities_3d = gmsh.model.occ.getEntities(dim=3)
    entities_2d = gmsh.model.occ.getEntities(dim=2)
    print(f"[pipeline] Entities: {len(entities_3d)} volumes, {len(entities_2d)} surfaces")

    # 2. Heal shapes
    if heal:
        print("[pipeline] Healing shapes...")
        try:
            gmsh.model.occ.healShapes()
            gmsh.model.occ.synchronize()
            print("[pipeline] Healing done")
        except Exception as e:
            print(f"[pipeline] Healing failed (continuing): {e}")

    # Increase geometry tolerance for problematic models
    gmsh.option.setNumber("Geometry.Tolerance", 1e-4)
    gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-4)

    # 3. Compute bounding box for auto-sizing
    bb = gmsh.model.getBoundingBox(-1, -1)
    extent = [bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]]
    max_extent = max(extent)
    print(f"[pipeline] Bounding box: {extent[0]:.2f} x {extent[1]:.2f} x {extent[2]:.2f}")

    # Auto scale to mm if needed
    if scale is None:
        # Heuristic: if max extent < 1, likely meters → scale to mm
        if max_extent < 0.5:
            scale = 1000.0
            print(f"[pipeline] Auto-scaling to mm (x{scale})")
        elif max_extent < 50:
            # Could be inches
            scale = 25.4
            print(f"[pipeline] Auto-scaling inches->mm (x{scale})")
        else:
            scale = 1.0

    # 4. Configure Gmsh meshing from chord deviation target
    #
    # Chord deviation d for element size s on radius R:
    #   d = R * (1 - cos(s/(2R))) ≈ s² / (8R)   for s << R
    #
    # Gmsh's MeshSizeFromCurvature = N sets element size s = 2πR/N,
    # giving chord deviation d = π²R/(2N²).
    #
    # N must guarantee d ≤ chord_deviation at the transition radius where
    # curvature sizing meets max_edge: R_t = N*max_edge/(2π).
    # Substituting: d_t = π*max_edge/(4N), so N ≥ π*max_edge/(4*chord_deviation).
    #
    # For small radii, element size from curvature = 2πR/N.
    # The chord deviation there is π²R/(2N²), which is SMALLER than the
    # target (over-refined), but that's acceptable — small features need
    # resolution regardless.

    if max_edge is None:
        max_edge = max_extent * 0.5  # 50% of extent — flat faces get large elements

    # Compute N from chord deviation.
    # For radius R, curvature sizing gives element size = 2πR/N.
    # Chord deviation d = π²R/(2N²). For d ≤ target at the largest radius
    # where curvature matters (R_max ≈ max_edge²/(8*d)):
    # N = π*sqrt(R_max/(2*d)) = π*max_edge/(4*d) * sqrt(1/2) ≈ max_edge*0.56/d
    # But cap at a reasonable value to avoid over-refinement.
    r_max = max_edge**2 / (8.0 * chord_deviation)  # transition radius
    curvature_elements = max(4, min(20, math.ceil(math.pi * math.sqrt(r_max / (2.0 * chord_deviation)))))

    # min_edge: the smallest element must achieve chord_deviation on the smallest
    # radius we care about. s = sqrt(8*R_min*d).
    # R_min = chord_deviation gives s = sqrt(8*d*d) = 2*sqrt(2)*d ≈ 2.83*d
    # This ensures even 1mm-radius fillets stay within chord deviation.
    if min_edge is None:
        min_edge = 2.0 * chord_deviation  # conservative: ~0.5mm chord dev on R=1mm

    print(f"[pipeline] Chord deviation target: {chord_deviation:.2f} mm")
    print(f"[pipeline] Derived: N={curvature_elements} elements/2pi, "
          f"min_edge={min_edge:.3f}, max_edge={max_edge:.3f}")

    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature_elements)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_edge)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_edge)
    # Don't propagate small boundary sizes deep into flat face interiors.
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    # Smooth size gradation to limit abrupt size transitions
    gmsh.option.setNumber("Mesh.SmoothRatio", 2.0)
    # Optimize mesh quality after generation (Laplacian smoothing + edge swaps)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.Smoothing", 10)  # Laplacian smoothing passes

    # 5. Generate mesh (with fallback: re-import without heal if heal broke edges)
    print(f"[pipeline] Meshing (min={min_edge:.4f}, max={max_edge:.4f}, curv={curvature_elements})...")
    mesh_ok = False
    try:
        gmsh.model.mesh.generate(2)
        mesh_ok = True
    except Exception as e:
        print(f"[pipeline] Mesh failed: {e}")

    if not mesh_ok and heal:
        # Healing likely broke edge curves — re-import without heal
        print("[pipeline] Retrying without healing...")
        gmsh.clear()
        gmsh.model.add("pipeline_retry")
        gmsh.model.occ.importShapes(input_path)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curvature_elements)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_edge)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_edge)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        try:
            gmsh.model.mesh.generate(2)
            mesh_ok = True
            print("[pipeline] Meshing succeeded without healing")
        except Exception as e2:
            print(f"[pipeline] Mesh without healing also failed: {e2}")

    if not mesh_ok:
        # Last resort: MeshAdapt algorithm, no curvature sizing
        print("[pipeline] Last resort: MeshAdapt, no curvature sizing...")
        gmsh.model.mesh.clear()
        gmsh.option.setNumber("Mesh.Algorithm", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_edge * 2)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_edge * 2)
        try:
            gmsh.model.mesh.generate(2)
            mesh_ok = True
        except Exception as e3:
            print(f"[pipeline] All meshing attempts failed: {e3}")
            gmsh.finalize()
            return False

    # 5b. Find longest Z-perpendicular B-Rep face loop for auto-seam
    seam_edge_tags = set()  # OCC edge tags that form the seam loop
    z_tol_seam = 0.5
    try:
        best_perim = 0
        best_face_tag = -1
        best_edges = []
        for dim_s, tag_s in gmsh.model.occ.getEntities(dim=2):
            bnd = gmsh.model.getBoundary([(dim_s, tag_s)], oriented=True, recursive=False)
            if not bnd:
                continue
            # Get Z coords of all mesh nodes on boundary edges
            zs = []
            edge_perim = 0
            for b in bnd:
                abs_et = abs(b[1])
                try:
                    ntags, coords, _ = gmsh.model.mesh.getNodes(1, abs_et, includeBoundary=True)
                    for i in range(len(ntags)):
                        zs.append(coords[i * 3 + 2] * scale)
                    for i in range(len(ntags) - 1):
                        dx = (coords[(i+1)*3] - coords[i*3]) * scale
                        dy = (coords[(i+1)*3+1] - coords[i*3+1]) * scale
                        dz = (coords[(i+1)*3+2] - coords[i*3+2]) * scale
                        edge_perim += math.sqrt(dx*dx + dy*dy + dz*dz)
                except:
                    pass
            if not zs or (max(zs) - min(zs)) > z_tol_seam:
                continue
            if edge_perim > best_perim:
                best_perim = edge_perim
                best_face_tag = tag_s
                best_edges = [abs(b[1]) for b in bnd]

        if best_edges:
            seam_edge_tags = set(best_edges)
            print(f"[pipeline] Auto-seam: face {best_face_tag}, "
                  f"{len(best_edges)} edges, {best_perim:.1f}mm perimeter")
    except Exception as e:
        print(f"[pipeline] Auto-seam detection failed: {e}")

    # Collect mesh node tags on seam edges
    seam_node_tags = set()
    for edge_tag in seam_edge_tags:
        try:
            ntags, _, _ = gmsh.model.mesh.getNodes(1, edge_tag, includeBoundary=True)
            seam_node_tags.update(int(nt) for nt in ntags)
        except:
            pass
    if seam_node_tags:
        print(f"[pipeline] Seam nodes: {len(seam_node_tags)}")

    # 6. Extract mesh with per-face normals and face IDs
    surfaces = gmsh.model.getEntities(dim=2)
    all_verts = []
    all_normals = []
    all_face_ids = []
    all_seam = []
    all_tris = []

    for face_idx, (dim, tag) in enumerate(surfaces):
        # Get nodes on this surface
        node_tags, coords, param_coords = gmsh.model.mesh.getNodes(dim, tag, includeBoundary=True)

        if len(node_tags) == 0:
            continue

        # Get triangles
        elem_types, elem_tags_list, elem_node_tags_list = gmsh.model.mesh.getElements(dim, tag)

        # Find triangle elements (type 2)
        tri_node_tags = None
        for et, ent in zip(elem_types, elem_node_tags_list):
            if et == 2:  # 3-node triangle
                tri_node_tags = ent
                break

        if tri_node_tags is None or len(tri_node_tags) == 0:
            continue

        # Build local node map (tag → local index)
        base = len(all_verts) // 3
        tag_to_local = {}
        for i, nt in enumerate(node_tags):
            local_idx = base + i
            tag_to_local[int(nt)] = local_idx

            # Position (scaled)
            x = coords[i * 3] * scale
            y = coords[i * 3 + 1] * scale
            z = coords[i * 3 + 2] * scale
            all_verts.extend([x, y, z])

            # Normal from OCC surface at parametric position
            u, v = param_coords[i * 2], param_coords[i * 2 + 1]
            try:
                nx, ny, nz = gmsh.model.getNormal(tag, [u, v])
                length = math.sqrt(nx*nx + ny*ny + nz*nz)
                if length > 1e-15:
                    all_normals.extend([nx/length, ny/length, nz/length])
                else:
                    all_normals.extend([0, 0, 1])
            except:
                all_normals.extend([0, 0, 1])

            # Face ID
            all_face_ids.append(float(face_idx))

            # Seam flag (1.0 if vertex on seam edge, 0.0 otherwise)
            all_seam.append(1.0 if int(nt) in seam_node_tags else 0.0)

        # Triangles
        for i in range(0, len(tri_node_tags), 3):
            n0, n1, n2 = int(tri_node_tags[i]), int(tri_node_tags[i+1]), int(tri_node_tags[i+2])
            if n0 in tag_to_local and n1 in tag_to_local and n2 in tag_to_local:
                all_tris.extend([tag_to_local[n0], tag_to_local[n1], tag_to_local[n2]])

    gmsh.finalize()

    nv = len(all_verts) // 3
    nf = len(all_tris) // 3
    print(f"[pipeline] Mesh: {nv} vertices, {nf} triangles, {len(surfaces)} OCC faces")

    if nv == 0:
        print("[pipeline] Error: empty mesh")
        return False

    # 7. Write GLB
    V = np.array(all_verts, dtype=np.float32)
    N = np.array(all_normals, dtype=np.float32)
    FID = np.array(all_face_ids, dtype=np.float32)
    SEAM = np.array(all_seam, dtype=np.float32)
    F = np.array(all_tris, dtype=np.uint32)

    pos_data = V.tobytes()
    nrm_data = N.tobytes()
    fid_data = FID.tobytes()
    seam_data = SEAM.tobytes()
    idx_data = F.tobytes()
    buf = pos_data + nrm_data + fid_data + seam_data + idx_data
    while len(buf) % 4:
        buf += b'\x00'

    mn = V.reshape(-1, 3).min(axis=0).tolist()
    mx = V.reshape(-1, 3).max(axis=0).tolist()

    nrm_off = len(pos_data)
    fid_off = nrm_off + len(nrm_data)
    seam_off = fid_off + len(fid_data)
    idx_off = seam_off + len(seam_data)

    gltf = json.dumps({
        "asset": {"version": "2.0", "generator": "occ_gmsh_pipeline"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {
            "POSITION": 0, "NORMAL": 1, "_FACE_ID": 2, "_SEAM": 3
        }, "indices": 4, "mode": 4}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": nv, "type": "VEC3", "min": mn, "max": mx},
            {"bufferView": 1, "componentType": 5126, "count": nv, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": nv, "type": "SCALAR"},
            {"bufferView": 3, "componentType": 5126, "count": nv, "type": "SCALAR"},
            {"bufferView": 4, "componentType": 5125, "count": nf * 3, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_data), "target": 34962},
            {"buffer": 0, "byteOffset": nrm_off, "byteLength": len(nrm_data), "target": 34962},
            {"buffer": 0, "byteOffset": fid_off, "byteLength": len(fid_data), "target": 34962},
            {"buffer": 0, "byteOffset": seam_off, "byteLength": len(seam_data), "target": 34962},
            {"buffer": 0, "byteOffset": idx_off, "byteLength": len(idx_data), "target": 34963},
        ],
        "buffers": [{"byteLength": len(buf)}],
    }, separators=(',', ':')).encode()
    while len(gltf) % 4:
        gltf += b' '

    total = 12 + 8 + len(gltf) + 8 + len(buf)
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<4sII', b'glTF', 2, total))
        f.write(struct.pack('<II', len(gltf), 0x4E4F534A))
        f.write(gltf)
        f.write(struct.pack('<II', len(buf), 0x004E4942))
        f.write(buf)

    extent_mm = [mx[i] - mn[i] for i in range(3)]
    print(f"[pipeline] Wrote: {output_path} ({os.path.getsize(output_path)} bytes)")
    print(f"[pipeline] Extent: {extent_mm[0]:.1f} x {extent_mm[1]:.1f} x {extent_mm[2]:.1f} mm")
    return True


def main():
    parser = argparse.ArgumentParser(description="STEP -> GLB pipeline (OCC + Gmsh)")
    parser.add_argument("input", help="Input STEP file")
    parser.add_argument("output", help="Output GLB file")
    parser.add_argument("--chord-deviation", type=float, default=1.0,
                        help="Max chord deviation in mm (default: 1.0)")
    parser.add_argument("--scale", type=float, default=None,
                        help="Scale factor (auto-detect if not set)")
    parser.add_argument("--min-edge", type=float, default=None)
    parser.add_argument("--max-edge", type=float, default=None)
    parser.add_argument("--no-heal", action="store_true")
    args = parser.parse_args()

    ok = step_to_glb(args.input, args.output,
                     chord_deviation=args.chord_deviation,
                     scale=args.scale,
                     min_edge=args.min_edge,
                     max_edge=args.max_edge,
                     heal=not args.no_heal)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
