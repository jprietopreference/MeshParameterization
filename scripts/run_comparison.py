"""
Run both Heat-Geodesic and CGAL (conformal + ARAP) on all test meshes.
Produce one result GLB per method per mesh, all with OCC normals + checker.
Print a summary comparison table at the end.
"""
import subprocess
import sys
import os
import re

MESHES = [
    "cube_100mm",
    "cube_100mm_fillet_25mm",
    "sphere_R100mm",
    "torus_R50mm_r15mm",
    "torus_R200mm_r40mm",
    "klein_bottle",
    "klein_bottle_occ",
    "teapot",
    "cgalref_nefertiti",
    "cgalref_three_peaks",
    "cgalref_head",
    "cgalref_mushroom",
]

METHODS = [
    ("heat", None),
    ("cgal_conformal", "conformal"),
    ("cgal_arap", "arap"),
]


def parse_metrics(stdout):
    """Extract distortion metrics from CLI output."""
    m = {}
    for line in stdout.split("\n"):
        if "Angle distortion:" in line:
            match = re.search(r"mean=([\d.]+).*max=([\d.]+)", line)
            if match:
                m["angle_mean"] = float(match.group(1))
                m["angle_max"] = float(match.group(2))
        if "Area distortion:" in line:
            match = re.search(r"mean=([\d.]+)\s+std=([\d.]+)", line)
            if match:
                m["area_mean"] = float(match.group(1))
                m["area_std"] = float(match.group(2))
        if "L2 stretch:" in line:
            match = re.search(r"mean=([\d.]+)\s+max=([\d.]+)", line)
            if match:
                m["stretch_mean"] = float(match.group(1))
                m["stretch_max"] = float(match.group(2))
        if "Isometric RMS:" in line:
            match = re.search(r"([\d.]+)\s+\(", line)
            if match:
                m["iso_rms"] = float(match.group(1))
    return m


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root, "data")
    heat_cli = os.path.join(root, "build", "meshparam_cli.exe")
    # Prefer native EPICK build if available, fallback to WASM build
    cgal_native = os.path.join(root, "cgal_param_native", "build", "cgalparam_native_cli.exe")
    cgal_wasm = os.path.join(root, "cgal_param", "build", "cgalparam_cli.exe")
    cgal_cli = cgal_native if os.path.exists(cgal_native) else cgal_wasm
    assemble = os.path.join(root, "scripts", "assemble_result.py")
    checker = os.path.join(root, "scripts", "apply_checkerboard.py")

    results = []

    for name in MESHES:
        input_glb = os.path.join(data_dir, f"{name}.glb")
        occ_npz = os.path.join(data_dir, f"{name}.occmesh.npz")

        if not os.path.exists(input_glb):
            continue

        for method_tag, cgal_method in METHODS:
            output_glb = os.path.join(data_dir, f"{name}_{method_tag}_result.glb")
            tmp = os.path.join(data_dir, f".{name}_{method_tag}_tmp.glb")

            print(f"  {name} / {method_tag}...", end=" ", flush=True)

            # Step 1: Parameterize
            if method_tag == "heat":
                r = subprocess.run(
                    [heat_cli, input_glb, tmp, "--no-fill"],
                    capture_output=True, text=True)
            else:
                r = subprocess.run(
                    [cgal_cli, input_glb, tmp, "--method", cgal_method],
                    capture_output=True, text=True)

            if r.returncode != 0:
                print("FAILED")
                continue

            metrics = parse_metrics(r.stdout)

            # Step 2: Assemble with normals + checker
            if os.path.exists(occ_npz):
                subprocess.run(
                    [sys.executable, assemble, tmp, occ_npz, output_glb, "--cell-size", "25"],
                    capture_output=True, text=True)
            else:
                subprocess.run(
                    [sys.executable, checker, tmp, output_glb, "--cell-size", "25"],
                    capture_output=True, text=True)

            try:
                os.remove(tmp)
            except OSError:
                pass

            size_kb = os.path.getsize(output_glb) / 1024 if os.path.exists(output_glb) else 0
            metrics["name"] = name
            metrics["method"] = method_tag
            metrics["size_kb"] = size_kb
            results.append(metrics)
            print(f"OK ({size_kb:.0f} KB)")

    # Print comparison table
    print(f"\n{'='*110}")
    print(f"{'Mesh':<28} {'Method':<18} {'Angle mean':>10} {'Angle max':>10} {'Area std':>9} {'Stretch':>10} {'Str max':>12} {'Iso RMS':>8}")
    print(f"{'='*110}")

    prev_name = ""
    for r in results:
        name = r.get("name", "")
        if name != prev_name and prev_name != "":
            print(f"{'-'*110}")
        prev_name = name

        print(f"{name:<28} {r.get('method',''):<18} "
              f"{r.get('angle_mean',0):>9.1f}° {r.get('angle_max',0):>9.1f}° "
              f"{r.get('area_std',0):>9.2f} "
              f"{r.get('stretch_mean',0):>10.1f} {r.get('stretch_max',0):>12.0f} "
              f"{r.get('iso_rms',0):>8.2f}")

    print(f"{'='*110}")
    print(f"\nResult files in: {data_dir}")


if __name__ == "__main__":
    main()
