"""
Run CGAL parameterization on all test meshes (conformal + ARAP),
then assemble results with OCC face normals + checkerboard.
"""
import subprocess
import sys
import os

MESHES = [
    "cube_100mm",
    "cube_100mm_fillet_25mm",
    "sphere_R100mm",
    "torus_R50mm_r15mm",
    "torus_R200mm_r40mm",
    "klein_bottle",
]

METHODS = ["conformal", "arap"]

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root, "data")
    cli = os.path.join(root, "cgal_param", "build", "cgalparam_cli.exe")
    assemble_script = os.path.join(root, "scripts", "assemble_result.py")
    checker_script = os.path.join(root, "scripts", "apply_checkerboard.py")

    if not os.path.exists(cli):
        print(f"ERROR: CLI not found at {cli}. Build first.")
        sys.exit(1)

    ok = 0
    fail = 0

    for name in MESHES:
        for method in METHODS:
            input_glb = os.path.join(data_dir, f"{name}.glb")
            occ_npz = os.path.join(data_dir, f"{name}.occmesh.npz")
            output_glb = os.path.join(data_dir, f"{name}_cgal_{method}_result.glb")
            tmp_param = os.path.join(data_dir, f".{name}_cgal_{method}_tmp.glb")

            if not os.path.exists(input_glb):
                print(f"SKIP: {input_glb} not found")
                fail += 1
                continue

            print(f"\n{'='*60}")
            print(f" {name} / {method}")
            print(f"{'='*60}")

            # Step 1: CGAL parameterize
            result = subprocess.run(
                [cli, input_glb, tmp_param, "--method", method],
                capture_output=True, text=True
            )
            print(result.stdout.strip())
            if result.returncode != 0:
                print(f"  FAILED: {result.stderr.strip()}")
                fail += 1
                continue

            # Step 2: Assemble with OCC normals + checker (or fallback)
            if os.path.exists(occ_npz):
                result = subprocess.run(
                    [sys.executable, assemble_script, tmp_param, occ_npz, output_glb,
                     "--cell-size", "25"],
                    capture_output=True, text=True
                )
            else:
                result = subprocess.run(
                    [sys.executable, checker_script, tmp_param, output_glb,
                     "--cell-size", "25"],
                    capture_output=True, text=True
                )
            print(result.stdout.strip())
            if result.returncode != 0:
                print(f"  FAILED assembly: {result.stderr.strip()}")
                fail += 1
                continue

            try:
                os.remove(tmp_param)
            except OSError:
                pass

            ok += 1

    # Clean old bare CGAL outputs (without normals/checker)
    for name in MESHES:
        for method in METHODS:
            old = os.path.join(data_dir, f"{name}_cgal_{method}.glb")
            if os.path.exists(old):
                try:
                    os.remove(old)
                except OSError:
                    pass

    print(f"\n{'='*60}")
    print(f" Results: {ok}/{ok+fail} succeeded")
    print(f"{'='*60}")
    for name in MESHES:
        for method in METHODS:
            out = os.path.join(data_dir, f"{name}_cgal_{method}_result.glb")
            if os.path.exists(out):
                size_kb = os.path.getsize(out) / 1024
                print(f"  {name}_cgal_{method}_result.glb  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
