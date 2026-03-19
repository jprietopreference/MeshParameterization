"""
Full pipeline: parameterize + assemble result with OCC face normals + checkerboard.

Input per mesh:
  data/{name}.glb              - shared-vertex mesh (positions + indices)
  data/{name}.facenormals.bin  - per-triangle OCC surface normals (m * 3 * float32)

Output per mesh:
  data/{name}_result.glb       - exploded mesh with per-face OCC normals + UVs + checker
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

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root, "data")
    cli = os.path.join(root, "build", "meshparam_cli.exe")
    assemble_script = os.path.join(root, "scripts", "assemble_result.py")

    if not os.path.exists(cli):
        print(f"ERROR: CLI not found at {cli}. Build first.")
        sys.exit(1)

    ok = 0
    fail = 0

    for name in MESHES:
        input_glb = os.path.join(data_dir, f"{name}.glb")
        occ_npz = os.path.join(data_dir, f"{name}.occmesh.npz")
        output_glb = os.path.join(data_dir, f"{name}_result.glb")
        tmp_param = os.path.join(data_dir, f".{name}_tmp_param.glb")

        if not os.path.exists(input_glb):
            print(f"SKIP: {input_glb} not found")
            fail += 1
            continue

        print(f"\n{'='*60}")
        print(f" {name}")
        print(f"{'='*60}")

        # Step 1: Parameterize (shared-vertex glb → shared-vertex glb with UVs)
        result = subprocess.run(
            [cli, input_glb, tmp_param, "--no-fill"],
            capture_output=True, text=True
        )
        print(result.stdout.strip())
        if result.returncode != 0:
            print(f"  FAILED parameterization: {result.stderr.strip()}")
            fail += 1
            continue

        # Step 2: Assemble (parameterized glb + OCC split mesh → result with normals + checker)
        if os.path.exists(occ_npz):
            result = subprocess.run(
                [sys.executable, assemble_script, tmp_param, occ_npz, output_glb,
                 "--cell-size", "25"],
                capture_output=True, text=True
            )
            print(result.stdout.strip())
            if result.returncode != 0:
                print(f"  FAILED assembly: {result.stderr.strip()}")
                fail += 1
                continue
        else:
            # No OCC data (e.g. Klein bottle) — use old checkerboard approach
            checker_script = os.path.join(root, "scripts", "apply_checkerboard.py")
            result = subprocess.run(
                [sys.executable, checker_script, tmp_param, output_glb,
                 "--cell-size", "25"],
                capture_output=True, text=True
            )
            print(result.stdout.strip())
            if result.returncode != 0:
                print(f"  FAILED checkerboard: {result.stderr.strip()}")
                fail += 1
                continue

        # Clean up temp
        try:
            os.remove(tmp_param)
        except OSError:
            pass

        ok += 1

    print(f"\n{'='*60}")
    print(f" Results: {ok}/{ok+fail} succeeded")
    print(f"{'='*60}")
    print(f"Output files in: {data_dir}")
    for name in MESHES:
        out = os.path.join(data_dir, f"{name}_result.glb")
        if os.path.exists(out):
            size_kb = os.path.getsize(out) / 1024
            print(f"  {name}_result.glb  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
