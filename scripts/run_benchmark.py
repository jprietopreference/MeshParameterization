#!/usr/bin/env python3
"""Run parameterization benchmark on the Stein et al. dataset.

Usage: python run_benchmark.py [--limit N] [--server URL]
Processes Cut/disk-topology meshes from benchmark/Obj_Files/No_UVs/Cut/
Outputs results to benchmark_results/
"""

import os, sys, json, csv, time, struct, argparse, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_DIR = os.path.join(ROOT, "benchmark", "Obj_Files", "No_UVs", "Cut")
RESULTS_DIR = os.path.join(ROOT, "benchmark_results")
SERVER = "http://localhost:8080"


def obj_to_glb(obj_path):
    """Convert OBJ to minimal GLB (positions + indices only)."""
    verts, faces = [], []
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                parts = line.split()[1:]
                idxs = [int(p.split("/")[0]) - 1 for p in parts]
                if len(idxs) >= 3:
                    faces.append(idxs[:3])
                    for i in range(3, len(idxs)):
                        faces.append([idxs[0], idxs[i-1], idxs[i]])

    nv, nf = len(verts), len(faces)
    if nv == 0 or nf == 0:
        return None

    import numpy as np
    V = np.array(verts, dtype=np.float32)
    F = np.array(faces, dtype=np.uint32)

    pos_data = V.tobytes()
    idx_data = F.tobytes()
    buf = pos_data + idx_data
    while len(buf) % 4: buf += b'\x00'

    mn = V.min(axis=0).tolist()
    mx = V.max(axis=0).tolist()

    gltf = json.dumps({
        "asset": {"version": "2.0"}, "scene": 0,
        "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": nv, "type": "VEC3", "min": mn, "max": mx},
            {"bufferView": 1, "componentType": 5125, "count": nf*3, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_data), "target": 34962},
            {"buffer": 0, "byteOffset": len(pos_data), "byteLength": len(idx_data), "target": 34963},
        ],
        "buffers": [{"byteLength": len(buf)}],
    }, separators=(',', ':')).encode()
    while len(gltf) % 4: gltf += b' '

    total = 12 + 8 + len(gltf) + 8 + len(buf)
    glb = bytearray(total)
    struct.pack_into('<4sII', glb, 0, b'glTF', 2, total)
    struct.pack_into('<II', glb, 12, len(gltf), 0x4E4F534A)
    glb[20:20+len(gltf)] = gltf
    struct.pack_into('<II', glb, 20+len(gltf), len(buf), 0x004E4942)
    glb[28+len(gltf):28+len(gltf)+len(buf)] = buf
    return bytes(glb), nv, nf


def parameterize(glb_data, server_url):
    """Send GLB to server, get all method results."""
    req = urllib.request.Request(
        f"{server_url}/api/parameterize",
        data=glb_data,
        headers={"Content-Type": "application/octet-stream"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            headers = dict(resp.headers)
            all_methods = headers.get("X-All-Methods", "[]")
            return json.loads(all_methods)
    except Exception as e:
        return [{"method": "error", "success": False, "error": str(e)}]


def process_mesh(obj_path, server_url):
    """Process one mesh: OBJ → GLB → parameterize → results."""
    name = os.path.basename(obj_path)
    try:
        result = obj_to_glb(obj_path)
        if result is None:
            return name, [{"method": "error", "success": False, "error": "empty mesh"}]
        glb, nv, nf = result
        methods = parameterize(glb, server_url)
        for m in methods:
            m["mesh_name"] = name
            m["input_verts"] = nv
            m["input_faces"] = nf
        return name, methods
    except Exception as e:
        return name, [{"method": "error", "success": False, "error": str(e), "mesh_name": name}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--server", default=SERVER)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Get mesh list
    meshes = sorted([f for f in os.listdir(BENCHMARK_DIR) if f.endswith(".obj")])
    # Sample diverse meshes (every Nth)
    step = max(1, len(meshes) // args.limit)
    selected = meshes[::step][:args.limit]

    print(f"Benchmark: {len(selected)} meshes (from {len(meshes)} total)")
    print(f"Server: {args.server}")
    print(f"Output: {RESULTS_DIR}/")

    all_results = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_mesh, os.path.join(BENCHMARK_DIR, m), args.server): m
            for m in selected
        }
        done = 0
        for future in as_completed(futures):
            name, methods = future.result()
            all_results.extend(methods)
            done += 1
            best = min((m for m in methods if m.get("success")), key=lambda m: m.get("score", 1e18), default=None)
            if best:
                print(f"  [{done}/{len(selected)}] {name}: winner={best['method']} SD={best.get('sym_dirichlet',0):.1f} flips={best.get('flipped_tris',0)}")
            else:
                print(f"  [{done}/{len(selected)}] {name}: ALL FAILED")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.0f}s")

    # Save raw results
    with open(os.path.join(RESULTS_DIR, "benchmark_raw.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Aggregate by method
    from collections import defaultdict
    method_stats = defaultdict(lambda: {"count": 0, "success": 0, "total_sd": 0, "total_flips": 0, "total_time": 0, "best_count": 0})

    # Group by mesh
    mesh_groups = defaultdict(list)
    for r in all_results:
        mesh_groups[r.get("mesh_name", "")].append(r)

    for mesh_name, methods in mesh_groups.items():
        successful = [m for m in methods if m.get("success")]
        if not successful:
            continue
        best = min(successful, key=lambda m: m.get("score", 1e18))
        for m in methods:
            ms = method_stats[m["method"]]
            ms["count"] += 1
            if m.get("success"):
                ms["success"] += 1
                ms["total_sd"] += m.get("sym_dirichlet", 0)
                ms["total_flips"] += m.get("flipped_tris", 0)
                ms["total_time"] += m.get("elapsed_ms", 0)
            if m["method"] == best["method"]:
                ms["best_count"] += 1

    # Print summary
    print(f"\n{'='*90}")
    print(f"{'Method':25s} {'Success':>8s} {'AvgSD':>12s} {'AvgFlips':>10s} {'Wins':>6s} {'AvgTime':>10s}")
    print(f"{'='*90}")
    for method, stats in sorted(method_stats.items(), key=lambda x: -x[1]["best_count"]):
        n = stats["success"] or 1
        print(f"{method:25s} {stats['success']:>5d}/{stats['count']:<3d} {stats['total_sd']/n:12.1f} {stats['total_flips']/n:10.1f} {stats['best_count']:>6d} {stats['total_time']/n:8.0f}ms")
    print(f"{'='*90}")

    # Save summary CSV
    with open(os.path.join(RESULTS_DIR, "benchmark_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Success", "Total", "AvgSymDirichlet", "AvgFlips", "Wins", "AvgTimeMs"])
        for method, stats in sorted(method_stats.items(), key=lambda x: -x[1]["best_count"]):
            n = stats["success"] or 1
            w.writerow([method, stats["success"], stats["count"], f"{stats['total_sd']/n:.1f}",
                       f"{stats['total_flips']/n:.1f}", stats["best_count"], f"{stats['total_time']/n:.0f}"])

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
