"""
Run all parameterizations in parallel, produce result GLBs with normals + checker.
Collects metrics and prints comparison table.
"""
import subprocess, sys, os, re, concurrent.futures, time

MESHES = [
    "cube_100mm", "cube_100mm_fillet_25mm", "sphere_R100mm",
    "torus_R50mm_r15mm", "torus_R200mm_r40mm",
    "klein_bottle", "klein_bottle_occ", "teapot",
    "cgalref_nefertiti", "cgalref_three_peaks", "cgalref_head", "cgalref_mushroom",
]

# Skip heat for meshes > MAX_HEAT_VERTS (O(n^2) geodesic matrix)
MAX_HEAT_VERTS = 5000

def parse_metrics(stdout):
    m = {}
    for line in stdout.split("\n"):
        if "Loaded" in line:
            match = re.search(r"(\d+) vertices", line)
            if match: m["verts"] = int(match.group(1))
        if "Angle distortion:" in line:
            match = re.search(r"mean=([\d.]+).*max=([\d.]+)", line)
            if match: m["angle_mean"] = float(match.group(1)); m["angle_max"] = float(match.group(2))
        if "Area distortion:" in line:
            match = re.search(r"std=([\d.]+)", line)
            if match: m["area_std"] = float(match.group(1))
        if "L2 stretch:" in line:
            match = re.search(r"mean=([\d.]+)\s+max=([\d.]+)", line)
            if match: m["stretch_mean"] = float(match.group(1)); m["stretch_max"] = float(match.group(2))
    return m

def run_job(mesh, method_tag, cgal_method, root, data_dir):
    heat_cli = os.path.join(root, "build", "meshparam_cli.exe")
    cgal_cli = os.path.join(root, "cgal_param_native", "build", "cgalparam_native_cli.exe")
    assemble = os.path.join(root, "scripts", "assemble_result.py")
    checker = os.path.join(root, "scripts", "apply_checkerboard.py")

    input_glb = os.path.join(data_dir, f"{mesh}.glb")
    occ_npz = os.path.join(data_dir, f"{mesh}.occmesh.npz")
    output_glb = os.path.join(data_dir, f"{mesh}_{method_tag}_result.glb")
    tmp = os.path.join(data_dir, f".{mesh}_{method_tag}_tmp_{os.getpid()}.glb")

    if not os.path.exists(input_glb):
        return mesh, method_tag, None, "input not found"

    # Step 1: Parameterize
    if method_tag == "heat":
        r = subprocess.run([heat_cli, input_glb, tmp, "--no-fill"],
                          capture_output=True, text=True, timeout=600)
    else:
        r = subprocess.run([cgal_cli, input_glb, tmp, "--method", cgal_method],
                          capture_output=True, text=True, timeout=600)

    if r.returncode != 0:
        return mesh, method_tag, None, r.stderr[:200] if r.stderr else "failed"

    metrics = parse_metrics(r.stdout)

    # Step 2: Assemble with normals + checker
    if os.path.exists(occ_npz):
        subprocess.run([sys.executable, assemble, tmp, occ_npz, output_glb, "--cell-size", "25"],
                      capture_output=True, timeout=60)
    else:
        subprocess.run([sys.executable, checker, tmp, output_glb, "--cell-size", "25"],
                      capture_output=True, timeout=60)

    try: os.remove(tmp)
    except: pass

    size_kb = os.path.getsize(output_glb) / 1024 if os.path.exists(output_glb) else 0
    metrics["size_kb"] = size_kb
    return mesh, method_tag, metrics, None


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root, "data")

    # Build job list
    jobs = []
    for mesh in MESHES:
        input_glb = os.path.join(data_dir, f"{mesh}.glb")
        if not os.path.exists(input_glb):
            continue

        # Check vertex count for heat method
        import struct, json
        with open(input_glb, "rb") as f:
            d = f.read()
        jl = struct.unpack("<I", d[12:16])[0]
        g = json.loads(d[20:20+jl])
        n_verts = g["accessors"][0]["count"]

        if n_verts <= MAX_HEAT_VERTS:
            jobs.append((mesh, "heat", None))
        jobs.append((mesh, "cgal_conformal", "conformal"))
        jobs.append((mesh, "cgal_arap", "arap"))

    print(f"Running {len(jobs)} jobs in parallel...")
    t0 = time.time()

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_job, mesh, mt, cm, root, data_dir): (mesh, mt)
            for mesh, mt, cm in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            mesh, method_tag = futures[future]
            try:
                mesh, mt, metrics, err = future.result(timeout=620)
                if err:
                    print(f"  {mesh}/{mt}: FAILED ({err[:60]})")
                    results.append({"name": mesh, "method": mt, "error": True})
                else:
                    print(f"  {mesh}/{mt}: OK ({metrics.get('size_kb',0):.0f} KB)")
                    metrics["name"] = mesh
                    metrics["method"] = mt
                    results.append(metrics)
            except Exception as e:
                print(f"  {mesh}/{method_tag}: EXCEPTION ({e})")
                results.append({"name": mesh, "method": method_tag, "error": True})

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.0f}s")

    # Sort results by mesh order then method
    method_order = {"heat": 0, "cgal_conformal": 1, "cgal_arap": 2}
    mesh_order = {m: i for i, m in enumerate(MESHES)}
    results.sort(key=lambda r: (mesh_order.get(r.get("name",""), 99), method_order.get(r.get("method",""), 99)))

    # Print table
    print(f"\n{'='*115}")
    print(f"{'Mesh':<28} {'Method':<18} {'Verts':>6} {'Angle':>7} {'AngMax':>7} {'AreaStd':>8} {'Stretch':>10} {'StrMax':>12} {'KB':>5}")
    print(f"{'='*115}")
    prev = ""
    for r in results:
        name = r.get("name", "")
        if name != prev and prev: print(f"{'-'*115}")
        prev = name
        if r.get("error"):
            print(f"{name:<28} {r.get('method',''):<18} {'':>6} {'FAIL':>7}")
            continue
        print(f"{name:<28} {r.get('method',''):<18} "
              f"{r.get('verts',0):>6} "
              f"{r.get('angle_mean',0):>6.1f}° {r.get('angle_max',0):>6.1f}° "
              f"{r.get('area_std',0):>8.2f} "
              f"{r.get('stretch_mean',0):>10.1f} {r.get('stretch_max',0):>12.0f} "
              f"{r.get('size_kb',0):>5.0f}")
    print(f"{'='*115}")


if __name__ == "__main__":
    main()
