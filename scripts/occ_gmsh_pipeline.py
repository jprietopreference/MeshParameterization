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
                scale=None, heal=True, output_back_path=None):
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
        best_z = 1e18
        best_edges = []

        # Find closed loops of Z-perpendicular B-Rep edges (not per-face wires).
        # These loops go around the part cross-section, splitting it in two.
        from collections import defaultdict as ddict

        # Collect all B-Rep edges where both endpoints share the same Z
        all_brep_edges = gmsh.model.occ.getEntities(dim=1)
        z_perp_edges = []
        for dim_e, tag_e in all_brep_edges:
            pts = gmsh.model.getBoundary([(1, tag_e)], oriented=False)
            if len(pts) < 2:
                continue  # closed curve (circle)
            p1 = gmsh.model.getValue(0, pts[0][1], [])
            p2 = gmsh.model.getValue(0, pts[1][1], [])
            if abs(p1[2] - p2[2]) * scale > z_tol_seam:
                continue
            # Use OCC analytical curve length for ranking (consistent, mesh-independent)
            edge_len = gmsh.model.occ.getMass(1, tag_e) * scale
            avg_z = (p1[2] + p2[2]) / 2.0 * scale
            z_perp_edges.append((tag_e, pts[0][1], pts[1][1], avg_z, edge_len))

        # Group by Z level (0.1mm buckets)
        z_groups = ddict(list)
        for tag_e, p1, p2, z, length in z_perp_edges:
            z_key = round(z * 10) / 10
            z_groups[z_key].append((tag_e, p1, p2, length))

        # Trace closed loops per Z level, collect all
        all_loops = []  # (z, edges, perimeter)
        for z_key, edge_list in z_groups.items():
            adj = ddict(list)
            for tag_e, p1, p2, length in edge_list:
                adj[p1].append((p2, tag_e, length))
                adj[p2].append((p1, tag_e, length))

            visited_edges = set()
            for tag_e, p1, p2, length in edge_list:
                if tag_e in visited_edges:
                    continue
                loop_edges = []
                loop_perim = 0
                cur, prev, start = p1, -1, p1
                closed = False
                for _ in range(10000):
                    found = False
                    for nb, et, el in adj[cur]:
                        if et in visited_edges or nb == prev:
                            continue
                        visited_edges.add(et)
                        loop_edges.append(et)
                        loop_perim += el
                        prev = cur
                        cur = nb
                        found = True
                        if cur == start and len(loop_edges) >= 3:
                            closed = True
                        break
                    if closed or not found:
                        break
                if closed and len(loop_edges) >= 3:
                    all_loops.append((z_key, loop_edges, loop_perim))

        # Pick best: longest perimeter, within 5% tie pick min Z
        all_loops.sort(key=lambda x: (-x[2], x[0]))
        if all_loops:
            top_perim = all_loops[0][2]
            perim_tol = max(1.0, top_perim * 0.02)  # 2% tolerance — prefer min Z among similar loops
            candidates = [l for l in all_loops if l[2] >= top_perim - perim_tol]
            candidates.sort(key=lambda x: x[0])  # sort by Z ascending
            best_z = candidates[0][0]
            best_edges = candidates[0][1]
            best_perim = candidates[0][2]

        # Topological split: flood-fill face adjacency, seam edges are barriers.
        # 1. Build face adjacency graph: face -> [neighbor faces] via shared edges
        # 2. Don't cross seam edges
        # 3. Find the face with strongest Z+ normal as seed for "front" side
        # 4. Flood from seed -> "front". Everything else -> "back"

        all_faces = gmsh.model.occ.getEntities(dim=2)
        all_face_tags = set(t for _, t in all_faces)
        total_faces = len(all_face_tags)
        min_frac = 0.15

        # Build edge -> faces adjacency
        edge_to_faces = ddict(set)
        face_to_edges = ddict(set)
        for dim_f, tag_f in all_faces:
            bnd = gmsh.model.getBoundary([(dim_f, tag_f)], oriented=False, recursive=False)
            for b in bnd:
                et = abs(b[1])
                edge_to_faces[et].add(tag_f)
                face_to_edges[tag_f].add(et)

        for cand_z, cand_edges, cand_perim in candidates:
            cand_seam_set = set(cand_edges)

            # Build face adjacency (don't cross seam)
            face_adj = ddict(set)
            for et, face_set in edge_to_faces.items():
                if et in cand_seam_set:
                    continue  # seam edge — barrier
                face_list = list(face_set)
                for i in range(len(face_list)):
                    for j in range(i + 1, len(face_list)):
                        face_adj[face_list[i]].add(face_list[j])
                        face_adj[face_list[j]].add(face_list[i])

            # Find seed: face with strongest Z+ normal (most front-facing)
            best_seed = -1
            best_nz = -2
            for dim_f, tag_f in all_faces:
                bb = gmsh.model.getBoundingBox(dim_f, tag_f)
                # Use face center to query normal
                cu = 0.5  # parametric center (approximate)
                cv = 0.5
                try:
                    # Get a point on the face for normal estimation
                    mass_center = gmsh.model.occ.getCenterOfMass(dim_f, tag_f)
                    # Estimate normal from bounding box orientation
                    # Better: check if face has nodes and use their normals
                    node_tags, coords, param_coords = gmsh.model.mesh.getNodes(dim_f, tag_f, includeBoundary=False)
                    if len(node_tags) > 0:
                        # Use first interior node's normal
                        u, v = param_coords[0], param_coords[1]
                        nx, ny, nz = gmsh.model.getNormal(tag_f, [u, v])
                        if nz > best_nz:
                            best_nz = nz
                            best_seed = tag_f
                except:
                    pass

            if best_seed < 0:
                continue  # can't find seed

            # Flood fill from seed
            front_faces = set()
            stack = [best_seed]
            while stack:
                f = stack.pop()
                if f in front_faces:
                    continue
                front_faces.add(f)
                for nb in face_adj.get(f, set()):
                    if nb not in front_faces:
                        stack.append(nb)

            back_faces = all_face_tags - front_faces
            frac_front = len(front_faces) / total_faces
            frac_back = len(back_faces) / total_faces

            if frac_front >= min_frac and frac_back >= min_frac:
                best_edges = cand_edges
                best_perim = cand_perim
                best_z = cand_z
                seam_edge_tags = cand_seam_set
                top_face_tags = front_faces
                bot_face_tags = back_faces
                print(f"[pipeline] Auto-seam: {len(best_edges)} edges, "
                      f"{best_perim:.1f}mm perimeter, z={best_z:.2f}mm")
                print(f"[pipeline] Split: {len(top_face_tags)} front faces ({frac_front:.0%}), "
                      f"{len(bot_face_tags)} back faces ({frac_back:.0%})")
                print(f"[pipeline] Seed face {best_seed} (normal Z={best_nz:.2f})")
                break
        else:
            print(f"[pipeline] No balanced seam found among {len(candidates)} candidates")
    except Exception as e:
        print(f"[pipeline] Auto-seam detection failed: {e}")
        top_face_tags = set()
        bot_face_tags = set()

    # Classify ALL B-Rep edge nodes from OCC topology.
    # Store per-node: edge_type (1=seam, 2=z-perp, 3=other) and edge_id (OCC edge tag).
    # A node at a junction may belong to multiple edges — store all edge IDs.
    # _EDGE_TYPE: highest priority type (1>2>3), 0=interior
    # _EDGE_ID: OCC edge tag (for drawing: connect vertices with same edge_id)
    node_edge_type = {}  # node_tag -> edge_type
    node_edge_ids = {}   # node_tag -> set of edge tags
    seam_node_tags = set()

    all_brep_edges_for_classify = gmsh.model.occ.getEntities(dim=1)
    z_perp_edge_set = set()
    for _, tag_e in all_brep_edges_for_classify:
        pts = gmsh.model.getBoundary([(1, tag_e)], oriented=False)
        if len(pts) >= 2:
            p1 = gmsh.model.getValue(0, pts[0][1], [])
            p2 = gmsh.model.getValue(0, pts[1][1], [])
            if abs(p1[2] - p2[2]) * scale <= z_tol_seam:
                z_perp_edge_set.add(tag_e)

    # Encode edge_type in upper bits and edge_id in lower bits of a float.
    # Use: _EDGE_TYPE = type (1/2/3/0), _EDGE_ID = OCC edge tag
    edge_type_map = {}  # occ_edge_tag -> type (1/2/3)
    for _, tag_e in all_brep_edges_for_classify:
        if tag_e in seam_edge_tags:
            etype = 1
        elif tag_e in z_perp_edge_set:
            etype = 2
        else:
            etype = 3
        edge_type_map[tag_e] = etype
        try:
            ntags, _, _ = gmsh.model.mesh.getNodes(1, tag_e, includeBoundary=True)
            for nt in ntags:
                nt_int = int(nt)
                if nt_int not in node_edge_type or etype < node_edge_type[nt_int]:
                    node_edge_type[nt_int] = etype
                if nt_int not in node_edge_ids:
                    node_edge_ids[nt_int] = set()
                node_edge_ids[nt_int].add(tag_e)
                if etype == 1:
                    seam_node_tags.add(nt_int)
        except:
            pass

    # Build ordered node lists per OCC edge for the GLB
    # Nodes are sorted by parametric coordinate along the edge curve
    edge_node_lists = {}  # occ_edge_tag -> [node_tags in order along curve]
    edge_node_coords = {}  # occ_edge_tag -> {node_tag: (x,y,z)}
    for _, tag_e in all_brep_edges_for_classify:
        try:
            ntags, coords, params = gmsh.model.mesh.getNodes(1, tag_e, includeBoundary=True)
            if len(ntags) < 2:
                continue
            # Sort by parametric coordinate
            order = sorted(range(len(ntags)), key=lambda i: params[i])
            edge_node_lists[tag_e] = [int(ntags[i]) for i in order]
            # Store coordinates keyed by node tag
            node_coords = {}
            for i in range(len(ntags)):
                node_coords[int(ntags[i])] = (coords[i*3], coords[i*3+1], coords[i*3+2])
            edge_node_coords[tag_e] = node_coords
        except:
            pass

    n_seam = sum(1 for v in node_edge_type.values() if v == 1)
    n_zperp = sum(1 for v in node_edge_type.values() if v == 2)
    n_other = sum(1 for v in node_edge_type.values() if v == 3)
    print(f"[pipeline] B-Rep edge nodes: {n_seam} seam, {n_zperp} Z-perp, {n_other} other, "
          f"{len(edge_node_lists)} OCC edges")

    # 6. Extract mesh per face group (top/bottom if seam found, else all together)
    surfaces = gmsh.model.getEntities(dim=2)

    # Global map: gmsh node tag -> GLB vertex index (filled during extraction)
    global_node_to_glb = {}

    def extract_faces(face_tags_filter=None):
        """Extract mesh from a subset of OCC faces. Returns verts, normals, face_ids, seam, edge_type, tris."""
        verts, normals, face_ids, seam_flags, tris = [], [], [], [], []
        for face_idx, (dim, tag) in enumerate(surfaces):
            if face_tags_filter is not None and tag not in face_tags_filter:
                continue
            node_tags, coords, param_coords = gmsh.model.mesh.getNodes(dim, tag, includeBoundary=True)
            if len(node_tags) == 0:
                continue
            elem_types, elem_tags_list, elem_node_tags_list = gmsh.model.mesh.getElements(dim, tag)
            tri_node_tags = None
            for et, ent in zip(elem_types, elem_node_tags_list):
                if et == 2:
                    tri_node_tags = ent
                    break
            if tri_node_tags is None or len(tri_node_tags) == 0:
                continue
            base = len(verts) // 3
            tag_to_local = {}
            for i, nt in enumerate(node_tags):
                local_idx = base + i
                tag_to_local[int(nt)] = local_idx
                global_node_to_glb[int(nt)] = local_idx  # track for line primitives
                verts.extend([coords[i*3]*scale, coords[i*3+1]*scale, coords[i*3+2]*scale])
                u, v = param_coords[i*2], param_coords[i*2+1]
                try:
                    nx, ny, nz = gmsh.model.getNormal(tag, [u, v])
                    ln = math.sqrt(nx*nx+ny*ny+nz*nz)
                    if ln > 1e-15: normals.extend([nx/ln, ny/ln, nz/ln])
                    else: normals.extend([0, 0, 1])
                except:
                    normals.extend([0, 0, 1])
                face_ids.append(float(face_idx))
                seam_flags.append(1.0 if int(nt) in seam_node_tags else 0.0)
            for i in range(0, len(tri_node_tags), 3):
                n0, n1, n2 = int(tri_node_tags[i]), int(tri_node_tags[i+1]), int(tri_node_tags[i+2])
                if n0 in tag_to_local and n1 in tag_to_local and n2 in tag_to_local:
                    tris.extend([tag_to_local[n0], tag_to_local[n1], tag_to_local[n2]])
        return verts, normals, face_ids, seam_flags, tris

    # Determine if we split into two primitives or use one
    has_split = bool(seam_edge_tags) and bool(top_face_tags) and bool(bot_face_tags)
    if has_split:
        top_data = extract_faces(top_face_tags)
        bot_data = extract_faces(bot_face_tags)
        print(f"[pipeline] Top: {len(top_data[0])//3}v {len(top_data[4])//3}f, "
              f"Bottom: {len(bot_data[0])//3}v {len(bot_data[4])//3}f")
    else:
        top_data = extract_faces(None)
        bot_data = None

    # Legacy: keep all_verts etc for the single-mesh path
    all_verts = top_data[0]
    all_normals = top_data[1]
    all_face_ids = top_data[2]
    all_seam = top_data[3]
    all_tris = top_data[4]

    gmsh.finalize()

    def write_glb(path, parts):
        """Write GLB with triangle primitives + line primitives for B-Rep edges.
        parts = [(verts, normals, face_ids, seam, tris), ...]
        """
        # Concatenate all vertex data
        all_v, all_n, all_fid, all_s = [], [], [], []
        part_offsets = []  # (vert_offset, num_verts, num_tris) per part
        all_idx_parts = []
        vert_offset = 0
        for verts, normals, face_ids, seam_flags, tris in parts:
            nv_part = len(verts) // 3
            nf_part = len(tris) // 3
            all_v.extend(verts)
            all_n.extend(normals)
            all_fid.extend(face_ids)
            all_s.extend(seam_flags)
            # Offset indices by accumulated vertex count
            all_idx_parts.append(np.array(tris, dtype=np.uint32) + vert_offset)
            part_offsets.append((vert_offset, nv_part, nf_part))
            vert_offset += nv_part

        total_nv = vert_offset
        if total_nv == 0:
            print("[pipeline] Error: empty mesh")
            return False

        V = np.array(all_v, dtype=np.float32)
        N = np.array(all_n, dtype=np.float32)
        FID = np.array(all_fid, dtype=np.float32)
        SEAM = np.array(all_s, dtype=np.float32)

        pos_data = V.tobytes()
        nrm_data = N.tobytes()
        fid_data = FID.tobytes()
        seam_data = SEAM.tobytes()

        # Build edge line data as JSON extras (BabylonJS doesn't handle line primitives well)
        # Offset each edge point 0.05mm along the vertex normal to avoid Z-fighting
        EDGE_OFFSET = 0.05  # mm
        all_pos_arr = V.reshape(-1, 3)
        all_nrm_arr = N.reshape(-1, 3)
        edge_lines = {}  # "seam" / "zperp" / "other" -> [x,y,z,x,y,z,...]
        etype_keys = {1: "seam", 2: "zperp", 3: "other"}
        for etype in [1, 2, 3]:
            pts = []
            for occ_tag, node_list in edge_node_lists.items():
                if edge_type_map.get(occ_tag, 3) != etype:
                    continue
                coords = edge_node_coords[occ_tag]
                for i in range(len(node_list) - 1):
                    for nt in [node_list[i], node_list[i + 1]]:
                        raw = coords[nt]
                        px, py, pz = raw[0] * scale, raw[1] * scale, raw[2] * scale
                        # Look up vertex normal from GLB data
                        if nt in global_node_to_glb:
                            gi = global_node_to_glb[nt]
                            nx, ny, nz = all_nrm_arr[gi]
                        else:
                            nx, ny, nz = 0, 0, 0
                        pts.extend([round(px + nx * EDGE_OFFSET, 4),
                                    round(py + ny * EDGE_OFFSET, 4),
                                    round(pz + nz * EDGE_OFFSET, 4)])
            if pts:
                edge_lines[etype_keys[etype]] = pts

        # Index buffers per triangle part
        idx_datas = [p.tobytes() for p in all_idx_parts]

        # Build buffer: pos + nrm + fid + seam + tri_idx0 [+ tri_idx1]
        buf = pos_data + nrm_data + fid_data + seam_data
        for idata in idx_datas:
            buf += idata
        while len(buf) % 4:
            buf += b'\x00'

        mn = V.reshape(-1, 3).min(axis=0).tolist()
        mx = V.reshape(-1, 3).max(axis=0).tolist()

        nrm_off = len(pos_data)
        fid_off = nrm_off + len(nrm_data)
        seam_off = fid_off + len(fid_data)

        # Shared vertex attributes: accessors 0-3
        accessors = [
            {"bufferView": 0, "componentType": 5126, "count": total_nv, "type": "VEC3", "min": mn, "max": mx},
            {"bufferView": 1, "componentType": 5126, "count": total_nv, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": total_nv, "type": "SCALAR"},
            {"bufferView": 3, "componentType": 5126, "count": total_nv, "type": "SCALAR"},
        ]
        buffer_views = [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_data), "target": 34962},
            {"buffer": 0, "byteOffset": nrm_off, "byteLength": len(nrm_data), "target": 34962},
            {"buffer": 0, "byteOffset": fid_off, "byteLength": len(fid_data), "target": 34962},
            {"buffer": 0, "byteOffset": seam_off, "byteLength": len(seam_data), "target": 34962},
        ]

        # Per-part triangle index buffers
        idx_base_off = seam_off + len(seam_data)
        primitives = []
        for pi, (vo, nv_p, nf_p) in enumerate(part_offsets):
            acc_idx = len(accessors)
            bv_idx = len(buffer_views)
            buffer_views.append({
                "buffer": 0, "byteOffset": idx_base_off, "byteLength": len(idx_datas[pi]), "target": 34963
            })
            accessors.append({
                "bufferView": bv_idx, "componentType": 5125, "count": nf_p * 3, "type": "SCALAR"
            })
            primitives.append({
                "attributes": {"POSITION": 0, "NORMAL": 1, "_FACE_ID": 2, "_SEAM": 3},
                "indices": acc_idx, "mode": 4
            })
            idx_base_off += len(idx_datas[pi])

        total_nf = sum(po[2] for po in part_offsets)
        n_edge_segs = sum(len(v)//6 for v in edge_lines.values())
        print(f"[pipeline] Mesh: {total_nv} vertices, {total_nf} triangles, "
              f"{len(primitives)} primitive(s), {n_edge_segs} edge segments, {len(surfaces)} OCC faces")

        # Store B-Rep edge lines as extras on the node
        node = {"mesh": 0}
        if edge_lines:
            node["extras"] = {"edgeLines": edge_lines}

        gltf = json.dumps({
            "asset": {"version": "2.0", "generator": "occ_gmsh_pipeline"},
            "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [node],
            "meshes": [{"primitives": primitives}],
            "accessors": accessors,
            "bufferViews": buffer_views,
            "buffers": [{"byteLength": len(buf)}],
        }, separators=(',', ':')).encode()
        while len(gltf) % 4:
            gltf += b' '

        total_size = 12 + 8 + len(gltf) + 8 + len(buf)
        with open(path, 'wb') as f:
            f.write(struct.pack('<4sII', b'glTF', 2, total_size))
            f.write(struct.pack('<II', len(gltf), 0x4E4F534A))
            f.write(gltf)
            f.write(struct.pack('<II', len(buf), 0x004E4942))
            f.write(buf)

        extent_mm = [mx[i] - mn[i] for i in range(3)]
        print(f"[pipeline] Wrote: {path} ({os.path.getsize(path)} bytes)")
        print(f"[pipeline] Extent: {extent_mm[0]:.1f} x {extent_mm[1]:.1f} x {extent_mm[2]:.1f} mm")
        return True

    # Write output: two separate GLBs if split, else one
    if has_split and output_back_path:
        # Two files: front (top) and back (bottom)
        write_glb(output_path, [top_data])
        write_glb(output_back_path, [bot_data])
        return True
    elif has_split:
        # Single file with two primitives
        return write_glb(output_path, [top_data, bot_data])
    else:
        return write_glb(output_path, [top_data])


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
