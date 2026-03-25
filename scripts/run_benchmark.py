#!/usr/bin/env python3
"""Run parameterization benchmark on the Stein et al. dataset.

Usage: python run_benchmark.py [--limit N] [--server URL] [--resume]
Processes Cut/disk-topology meshes from benchmark/Obj_Files/No_UVs/Cut/
Compares against artist UVs from benchmark/Obj_Files/Artist_UVs/Cut/
Outputs results to benchmark_results/
"""

import os, sys, json, csv, time, struct, argparse, math, subprocess, signal
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_DIR = os.path.join(ROOT, "models", "benchmark", "Obj_Files")
NO_UVS_CUT = os.path.join(BENCHMARK_DIR, "No_UVs", "Cut")
ARTIST_UVS_CUT = os.path.join(BENCHMARK_DIR, "Artist_UVs", "Cut")
RESULTS_DIR = os.path.join(ROOT, "benchmark_results")
SERVER = "http://localhost:8080"


def parse_obj(obj_path):
    """Parse OBJ file, return verts, faces, and optionally UVs."""
    verts, uvs, faces, uv_faces = [], [], [], []
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("vt "):
                uvs.append([float(x) for x in line.split()[1:3]])
            elif line.startswith("f "):
                parts = line.split()[1:]
                vidxs = []
                tidxs = []
                for p in parts:
                    sp = p.split("/")
                    vidxs.append(int(sp[0]) - 1)
                    if len(sp) > 1 and sp[1]:
                        tidxs.append(int(sp[1]) - 1)
                # Triangulate fans
                for i in range(2, len(vidxs)):
                    faces.append([vidxs[0], vidxs[i-1], vidxs[i]])
                    if tidxs:
                        uv_faces.append([tidxs[0], tidxs[i-1], tidxs[i]])
    return verts, faces, uvs, uv_faces


def obj_to_glb(verts, faces):
    """Convert vertex/face arrays to minimal GLB (positions + indices)."""
    nv, nf = len(verts), len(faces)
    if nv == 0 or nf == 0:
        return None

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
    return bytes(glb)


def compute_sym_dirichlet(V, F, UV):
    """Compute per-triangle Symmetric Dirichlet energy from Jacobian SVD."""
    if UV is None or len(UV) == 0:
        return None, None, None
    V = np.array(V, dtype=np.float64)
    UV = np.array(UV, dtype=np.float64)
    F = np.array(F, dtype=np.int32)

    total_sd = 0.0
    flipped = 0
    energies = []

    for tri in F:
        i0, i1, i2 = tri
        # 3D edge vectors
        e1_3d = V[i1] - V[i0]
        e2_3d = V[i2] - V[i0]
        # 2D edge vectors
        e1_uv = UV[i1] - UV[i0]
        e2_uv = UV[i2] - UV[i0]

        # 3D area
        area_3d = 0.5 * np.linalg.norm(np.cross(e1_3d, e2_3d))
        # 2D signed area
        area_2d = 0.5 * (e1_uv[0]*e2_uv[1] - e1_uv[1]*e2_uv[0])

        if area_2d < 0:
            flipped += 1

        if abs(area_3d) < 1e-15 or abs(area_2d) < 1e-15:
            energies.append(1e18)
            continue

        # Build Jacobian (2x2): maps 3D triangle to 2D
        # Use local 2D frame for the 3D triangle
        t1 = e1_3d / (np.linalg.norm(e1_3d) + 1e-15)
        n = np.cross(e1_3d, e2_3d)
        n /= np.linalg.norm(n) + 1e-15
        t2 = np.cross(n, t1)

        # Local 2D coords of triangle vertices
        p1_local = np.array([np.dot(e1_3d, t1), np.dot(e1_3d, t2)])
        p2_local = np.array([np.dot(e2_3d, t1), np.dot(e2_3d, t2)])

        # Inverse of local 2D edge matrix
        M_3d = np.column_stack([p1_local, p2_local])
        M_uv = np.column_stack([e1_uv, e2_uv])

        det_3d = M_3d[0,0]*M_3d[1,1] - M_3d[0,1]*M_3d[1,0]
        if abs(det_3d) < 1e-15:
            energies.append(1e18)
            continue

        M_3d_inv = np.array([[M_3d[1,1], -M_3d[0,1]], [-M_3d[1,0], M_3d[0,0]]]) / det_3d
        J = M_uv @ M_3d_inv

        # SVD
        s = np.linalg.svd(J, compute_uv=False)
        sd = s[0]**2 + s[1]**2 + 1.0/(s[0]**2 + 1e-15) + 1.0/(s[1]**2 + 1e-15)
        energies.append(sd * area_3d)
        total_sd += sd * area_3d

    total_area = sum(0.5 * np.linalg.norm(np.cross(V[F[:,1]]-V[F[:,0]], V[F[:,2]]-V[F[:,0]]), axis=1))
    mean_sd = total_sd / (total_area + 1e-15)

    return mean_sd, flipped, energies


BENCH_CLI = os.path.join(ROOT, "server", "build", "meshparam_bench.exe")
METHODS = ["heat", "lscm", "igl_arap", "slim", "cgal_conformal", "cgal_arap", "cgal_authalic", "cm"]


def count_obj_vertices(obj_path):
    """Quick vertex count from OBJ without full parsing."""
    count = 0
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                count += 1
    return count


def parameterize_cli(obj_path, methods=None, timeout=120):
    """Run each method via CLI subprocess. Returns list of result dicts."""
    if methods is None:
        methods = METHODS

    # Skip large meshes — most methods can't handle 100K+ vertices in reasonable time
    nv = count_obj_vertices(obj_path)
    max_verts = 50000  # practical limit for all methods

    results = []
    for method in methods:
        if nv > max_verts:
            results.append({"method": method, "success": False,
                           "error": f"skipped: {nv} vertices > {max_verts} limit"})
            continue
        if method == "heat" and nv > 5000:
            results.append({"method": "heat", "success": False,
                           "error": f"skipped: {nv} vertices > 5000 limit (O(n^2))"})
            continue
        try:
            proc = subprocess.run(
                [BENCH_CLI, method, obj_path],
                capture_output=True, text=True, timeout=timeout,
            )
            # JSON is the last line of stdout (libraries may print log lines before it)
            stdout_lines = proc.stdout.strip().split('\n')
            json_line = None
            for line in reversed(stdout_lines):
                line = line.strip()
                if line.startswith('{'):
                    json_line = line
                    break
            if json_line:
                r = json.loads(json_line)
                results.append(r)
            elif proc.returncode != 0:
                results.append({"method": method, "success": False,
                               "error": f"exit code {proc.returncode}: {proc.stderr[:200]}"})
            else:
                results.append({"method": method, "success": False,
                               "error": "no JSON output"})
        except subprocess.TimeoutExpired:
            results.append({"method": method, "success": False, "error": "timeout"})
        except Exception as e:
            results.append({"method": method, "success": False, "error": str(e)})
    return results


def parameterize_server(glb_data, server_url, timeout=120):
    """Send GLB to server, get all method results."""
    req = urllib.request.Request(
        f"{server_url}/api/parameterize",
        data=glb_data,
        headers={"Content-Type": "application/octet-stream"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            all_methods = resp.headers.get("X-All-Methods", "[]")
            return json.loads(all_methods)
    except Exception as e:
        return [{"method": "error", "success": False, "error": str(e)}]


def process_mesh(obj_name, server_url=None, timeout=120, use_cli=True):
    """Process one mesh via CLI tools (default) or server."""
    obj_path = os.path.join(NO_UVS_CUT, obj_name)
    try:
        # Quick vertex count to skip huge meshes before full parsing
        nv_quick = count_obj_vertices(obj_path)
        if nv_quick > 50000:
            return obj_name, [{"method": "all", "success": False,
                              "error": f"skipped: {nv_quick} vertices > 50000",
                              "mesh_name": obj_name, "input_verts": nv_quick}]

        verts, faces, _, _ = parse_obj(obj_path)
        nv, nf = len(verts), len(faces)

        if use_cli:
            methods = parameterize_cli(obj_path, timeout=timeout)
        else:
            glb = obj_to_glb(verts, faces)
            if glb is None:
                return obj_name, [{"method": "error", "success": False, "error": "empty mesh"}]
            methods = parameterize_server(glb, server_url, timeout)

        # Compute artist UV metrics for comparison
        artist_path = os.path.join(ARTIST_UVS_CUT, obj_name)
        artist_sd = None
        if os.path.exists(artist_path):
            a_verts, a_faces, a_uvs, a_uv_faces = parse_obj(artist_path)
            if a_uvs and a_uv_faces:
                # Build per-vertex UVs (may need to expand if UV indices differ from vertex indices)
                if a_uv_faces and max(max(f) for f in a_uv_faces) < len(a_uvs):
                    try:
                        sd, flips, _ = compute_sym_dirichlet(
                            np.array(a_verts), np.array(a_uv_faces), np.array(a_uvs))
                        artist_sd = {"sym_dirichlet": sd, "flipped_tris": flips}
                    except:
                        pass

        for m in methods:
            m["mesh_name"] = obj_name
            m["input_verts"] = nv
            m["input_faces"] = nf
            if artist_sd:
                m["artist_sym_dirichlet"] = artist_sd["sym_dirichlet"]
                m["artist_flipped_tris"] = artist_sd["flipped_tris"]

        return obj_name, methods
    except Exception as e:
        return obj_name, [{"method": "error", "success": False, "error": str(e), "mesh_name": obj_name}]


def load_tags():
    """Load dataset_tags.csv into dict keyed by filename."""
    tags = {}
    csv_path = os.path.join(BENCHMARK_DIR, "dataset_tags.csv")
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            tags[row["Filename"]] = row
    return tags


def load_completed(results_path):
    """Load already-completed mesh names from partial results."""
    if not os.path.exists(results_path):
        return set()
    try:
        with open(results_path) as f:
            data = json.load(f)
        return set(r.get("mesh_name", "") for r in data)
    except:
        return set()


def is_server_alive(server_url):
    """Check if server is responding."""
    try:
        req = urllib.request.Request(f"{server_url}/api/health")
        with urllib.request.urlopen(req, timeout=3):
            return True
    except:
        return False


def start_server():
    """Start the parameterization server as a subprocess."""
    server_exe = os.path.join(ROOT, "server", "build", "meshparam_server.exe")
    if not os.path.exists(server_exe):
        return None
    proc = subprocess.Popen(
        [server_exe, "--port", "8080", "--web-root", os.path.join(ROOT, "web"),
         "--gmsh-cli", f"python {os.path.join(ROOT, 'scripts', 'occ_gmsh_pipeline.py')}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0,
    )
    # Wait for server to be ready
    for _ in range(20):
        time.sleep(0.5)
        if is_server_alive(SERVER):
            return proc
    return proc


def ensure_server(server_url, server_proc=None):
    """Ensure server is alive, restart if needed. Returns process handle."""
    if is_server_alive(server_url):
        return server_proc
    print("[benchmark] Server is down, restarting...")
    # Kill old process if any
    if server_proc:
        try:
            server_proc.kill()
            server_proc.wait(timeout=5)
        except:
            pass
    # Also kill any stale server
    if sys.platform == 'win32':
        os.system('taskkill /IM meshparam_server.exe /F >nul 2>&1')
    time.sleep(1)
    proc = start_server()
    if proc and is_server_alive(server_url):
        print("[benchmark] Server restarted successfully")
    else:
        print("[benchmark] WARNING: Server failed to restart!")
    return proc


def main():
    parser = argparse.ArgumentParser(description="Run Stein et al. benchmark")
    parser.add_argument("--limit", type=int, default=100, help="Max meshes to process")
    parser.add_argument("--server", default=SERVER)
    parser.add_argument("--workers", type=int, default=1, help="Parallel requests (1 = sequential, safer)")
    parser.add_argument("--timeout", type=int, default=120, help="Per-mesh timeout in seconds")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed meshes")
    parser.add_argument("--auto-restart", action="store_true", help="Auto-restart server on crash")
    parser.add_argument("--use-server", action="store_true", help="Use server API instead of CLI tools")
    parser.add_argument("--mesh-list", type=str, default=None, help="File with mesh names to process (one per line)")
    parser.add_argument("--subset", default="disk", choices=["disk", "all", "nondisk"],
                        help="Which meshes: disk (Cut+Disk+Manifold+!Small), all, nondisk")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "benchmark_raw.json")

    # Load tags and filter
    tags = load_tags()
    all_cut_files = sorted([f for f in os.listdir(NO_UVS_CUT) if f.endswith(".obj")])

    if args.subset == "disk":
        selected = [f for f in all_cut_files
                    if f in tags
                    and tags[f]["Disk"] == "True"
                    and tags[f]["Manifold"] == "True"
                    and tags[f]["Small"] == "False"]
    elif args.subset == "nondisk":
        selected = [f for f in all_cut_files
                    if f in tags
                    and tags[f]["Disk"] == "False"
                    and tags[f]["Manifold"] == "True"
                    and tags[f]["Small"] == "False"]
    else:
        selected = [f for f in all_cut_files
                    if f in tags and tags[f]["Manifold"] == "True" and tags[f]["Small"] == "False"]

    # Override with explicit mesh list if provided
    if args.mesh_list:
        with open(args.mesh_list) as f:
            mesh_set = set(line.strip() for line in f if line.strip())
        selected = [m for m in selected if m in mesh_set]
        if not selected:
            # Maybe the list has meshes not in the subset filter — use all from list
            selected = sorted([m for m in mesh_set if os.path.exists(os.path.join(NO_UVS_CUT, m))])
        print(f"Using mesh list: {args.mesh_list} ({len(selected)} meshes)")

    # Resume support
    completed = set()
    existing_results = []
    if args.resume:
        completed = load_completed(results_path)
        if completed:
            try:
                with open(results_path) as f:
                    existing_results = json.load(f)
            except:
                existing_results = []
            print(f"Resuming: {len(completed)} meshes already done")
        selected = [f for f in selected if f not in completed]

    # Apply limit (sample evenly)
    if len(selected) > args.limit:
        step = max(1, len(selected) // args.limit)
        selected = selected[::step][:args.limit]

    print(f"Benchmark: {len(selected)} meshes (from {len(all_cut_files)} total, {args.subset} subset)")
    print(f"Server: {args.server}, workers: {args.workers}, timeout: {args.timeout}s")
    print(f"Output: {RESULTS_DIR}/")
    print()

    all_results = list(existing_results)
    t0 = time.time()
    done = 0
    failed = 0
    server_proc = None
    restarts = 0
    consecutive_fails = 0

    for idx, mesh_name in enumerate(selected):
        # Check server health before each mesh (if auto-restart enabled)
        if args.auto_restart:
            server_proc = ensure_server(args.server, server_proc)
            if not is_server_alive(args.server):
                print(f"  [{idx+1}/{len(selected)}] {mesh_name}: SKIPPED (server down)")
                failed += 1
                continue

        name, methods = process_mesh(mesh_name, args.server, args.timeout, use_cli=not args.use_server)
        all_results.extend(methods)
        done += 1

        # Check if this was a connection error (server crash)
        is_conn_error = any("10061" in m.get("error", "") or "Connection refused" in m.get("error", "")
                          for m in methods if not m.get("success"))

        best = min((m for m in methods if m.get("success")),
                  key=lambda m: m.get("score", 1e18), default=None)
        if best:
            sd = best.get("sym_dirichlet", 0)
            flips = best.get("flipped_tris", 0)
            artist = best.get("artist_sym_dirichlet")
            artist_str = f" artist_SD={artist:.1f}" if artist else ""
            print(f"  [{done}/{len(selected)}] {name}: "
                  f"winner={best['method']} SD={sd:.1f} flips={flips}{artist_str}")
            consecutive_fails = 0
        else:
            failed += 1
            err = methods[0].get("error", "?") if methods else "?"
            print(f"  [{done}/{len(selected)}] {name}: FAILED ({err[:80]})")
            consecutive_fails += 1

            # Auto-restart on connection error
            if is_conn_error and args.auto_restart:
                restarts += 1
                print(f"  [benchmark] Server crashed (restart #{restarts}), restarting...")
                server_proc = ensure_server(args.server, server_proc)
                # Retry this mesh
                if is_server_alive(args.server):
                    name2, methods2 = process_mesh(mesh_name, args.server, args.timeout)
                    best2 = min((m for m in methods2 if m.get("success")),
                               key=lambda m: m.get("score", 1e18), default=None)
                    if best2:
                        # Replace failed results with successful retry
                        all_results = [r for r in all_results if r.get("mesh_name") != mesh_name]
                        all_results.extend(methods2)
                        failed -= 1
                        consecutive_fails = 0
                        sd = best2.get("sym_dirichlet", 0)
                        print(f"  [{done}/{len(selected)}] {name}: RETRY OK winner={best2['method']} SD={sd:.1f}")

            if consecutive_fails >= 20 and not args.auto_restart:
                print(f"  [benchmark] 20 consecutive failures, stopping. Use --auto-restart to continue.")
                break

        # Save intermediate results every 10 meshes
            if done % 10 == 0:
                with open(results_path, "w") as f:
                    json.dump(all_results, f)

    elapsed = time.time() - t0
    print(f"\nCompleted {done} meshes in {elapsed:.0f}s ({failed} failed)")

    # Save final results
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # ---- Aggregate by method ----
    from collections import defaultdict
    method_stats = defaultdict(lambda: {
        "count": 0, "success": 0, "wins": 0,
        "sd_values": [], "flip_values": [], "time_values": [],
        "zero_flip_count": 0,
    })

    # Group by mesh
    mesh_groups = defaultdict(list)
    for r in all_results:
        if r.get("method") != "error":
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
                sd = m.get("sym_dirichlet", 0)
                ms["sd_values"].append(sd)
                ms["flip_values"].append(m.get("flipped_tris", 0))
                ms["time_values"].append(m.get("elapsed_ms", 0))
                if m.get("flipped_tris", 0) == 0:
                    ms["zero_flip_count"] += 1
            if m["method"] == best["method"]:
                ms["wins"] += 1

    # Print summary
    print(f"\n{'='*110}")
    print(f"{'Method':20s} {'OK':>6s} {'Rate':>5s} {'MedianSD':>12s} {'MeanSD':>12s} "
          f"{'0-flip%':>7s} {'Wins':>5s} {'MedTime':>8s}")
    print(f"{'='*110}")
    for method, s in sorted(method_stats.items(), key=lambda x: -x[1]["wins"]):
        n = s["success"] or 1
        rate = f"{100*s['success']/s['count']:.0f}%" if s["count"] > 0 else "-"
        med_sd = f"{np.median(s['sd_values']):.1f}" if s["sd_values"] else "-"
        mean_sd = f"{np.mean(s['sd_values']):.1f}" if s["sd_values"] else "-"
        zf = f"{100*s['zero_flip_count']/n:.0f}%" if n > 0 else "-"
        med_t = f"{np.median(s['time_values']):.0f}ms" if s["time_values"] else "-"
        print(f"{method:20s} {s['success']:>3d}/{s['count']:<3d} {rate:>5s} {med_sd:>12s} {mean_sd:>12s} "
              f"{zf:>7s} {s['wins']:>5d} {med_t:>8s}")
    print(f"{'='*110}")

    # Save summary CSV
    with open(os.path.join(RESULTS_DIR, "benchmark_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Success", "Total", "Rate", "MedianSD", "MeanSD",
                     "ZeroFlipPct", "Wins", "MedianTimeMs"])
        for method, s in sorted(method_stats.items(), key=lambda x: -x[1]["wins"]):
            n = s["success"] or 1
            w.writerow([
                method, s["success"], s["count"],
                f"{100*s['success']/s['count']:.1f}" if s["count"] else "",
                f"{np.median(s['sd_values']):.1f}" if s["sd_values"] else "",
                f"{np.mean(s['sd_values']):.1f}" if s["sd_values"] else "",
                f"{100*s['zero_flip_count']/n:.1f}" if n else "",
                s["wins"],
                f"{np.median(s['time_values']):.0f}" if s["time_values"] else "",
            ])

    # Per-mesh CSV with all method scores
    with open(os.path.join(RESULTS_DIR, "benchmark_per_mesh.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Mesh", "Verts", "Faces", "Method", "Success", "SymDirichlet",
                     "FlippedTris", "TimeMs", "Score", "ArtistSD"])
        for r in sorted(all_results, key=lambda x: (x.get("mesh_name",""), x.get("method",""))):
            if r.get("method") == "error":
                continue
            w.writerow([
                r.get("mesh_name", ""), r.get("input_verts", ""), r.get("input_faces", ""),
                r.get("method", ""), r.get("success", False),
                f"{r.get('sym_dirichlet', '')}" if r.get("success") else "",
                r.get("flipped_tris", "") if r.get("success") else "",
                f"{r.get('elapsed_ms', ''):.0f}" if r.get("success") else "",
                f"{r.get('score', '')}" if r.get("success") else "",
                f"{r.get('artist_sym_dirichlet', '')}" if r.get("artist_sym_dirichlet") else "",
            ])

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
