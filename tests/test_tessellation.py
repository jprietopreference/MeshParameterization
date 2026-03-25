"""Tessellation regression tests for STEP files.

Tests that occ_gmsh_pipeline.py produces consistent results for each STEP file.
Checks: vertex/face count, bounding box, volume, unit detection, seam, triangle quality.

First run creates the baseline. Subsequent runs compare against it.
Usage: pytest tests/test_tessellation.py -v --html=tests/report_tessellation.html
"""
import os
import json
import subprocess
import tempfile
import struct
import numpy as np
import pytest

from conftest import (
    ROOT, STEP_DIR, PIPELINE, BASELINES_DIR,
    step_files, parse_glb, mesh_volume_from_glb, mesh_stats_from_glb,
    load_baseline, save_baseline, check_regression,
    TOL_VERTEX_COUNT, TOL_FACE_COUNT, TOL_VOLUME, TOL_BBOX, TOL_ASPECT_RATIO,
)

BASELINE_FILE = "tessellation.json"


def tessellate_step(step_path):
    """Run occ_gmsh_pipeline.py on a STEP file, return GLB path + stdout."""
    with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as tmp:
        glb_path = tmp.name
    result = subprocess.run(
        ['python', PIPELINE, step_path, glb_path],
        capture_output=True, timeout=120, cwd=ROOT,
        encoding='utf-8', errors='replace'
    )
    return glb_path, result.stdout + result.stderr, result.returncode


def get_step_unit_from_glb(gltf):
    """Read stepUnit from GLB node extras."""
    extras = gltf.get('nodes', [{}])[0].get('extras', {})
    return extras.get('stepUnit', 'unknown'), extras.get('scaleFactor', 1.0)


def get_seam_info_from_glb(gltf, bin_data):
    """Check if GLB has seam (2 primitives) and seam edge count."""
    n_prims = len(gltf['meshes'][0]['primitives'])
    extras = gltf.get('nodes', [{}])[0].get('extras', {})
    seam_pts = extras.get('edgeLines', {}).get('seam', [])
    return {
        'n_primitives': n_prims,
        'has_seam': n_prims >= 2,
        'seam_segments': len(seam_pts) // 6 if seam_pts else 0,
    }


@pytest.fixture(scope="module")
def baseline():
    return load_baseline(BASELINE_FILE)


@pytest.fixture(scope="module")
def results():
    """Collect all test results for baseline update."""
    return {}


STEP_FILES = step_files()


@pytest.mark.parametrize("step_file", STEP_FILES, ids=STEP_FILES)
def test_tessellation(step_file, baseline, results, tmp_path):
    """Test tessellation of a STEP file."""
    step_path = os.path.join(STEP_DIR, step_file)
    glb_path, output, rc = tessellate_step(step_path)

    try:
        assert rc == 0, f"Pipeline failed (rc={rc}):\n{output}"
        assert os.path.exists(glb_path), "GLB not created"
        assert os.path.getsize(glb_path) > 100, "GLB too small"

        gltf, bin_data = parse_glb(glb_path)
        n_prims = len(gltf['meshes'][0]['primitives'])

        # Collect stats from all primitives
        total_verts = 0
        total_faces = 0
        total_volume = 0
        bb_mins, bb_maxs = [], []

        for pi in range(n_prims):
            prim = gltf['meshes'][0]['primitives'][pi]
            # Skip non-triangle primitives
            if prim.get('mode', 4) != 4:
                continue
            stats = mesh_stats_from_glb(gltf, bin_data, pi)
            total_verts += stats['vertices']
            total_faces += stats['faces']
            bb_mins.append(stats['bb_min'])
            bb_maxs.append(stats['bb_max'])
            try:
                vol = mesh_volume_from_glb(gltf, bin_data, pi)
                total_volume += vol
            except:
                pass

        bb_min = [min(b[i] for b in bb_mins) for i in range(3)] if bb_mins else [0, 0, 0]
        bb_max = [max(b[i] for b in bb_maxs) for i in range(3)] if bb_maxs else [0, 0, 0]
        extent = [bb_max[i] - bb_min[i] for i in range(3)]

        # Unit detection
        step_unit, scale_factor = get_step_unit_from_glb(gltf)

        # Seam info
        seam = get_seam_info_from_glb(gltf, bin_data)

        # First primitive aspect ratio
        stats0 = mesh_stats_from_glb(gltf, bin_data, 0)

        # Build result dict
        actual = {
            'vertices': total_verts,
            'faces': total_faces,
            'volume_mm3': round(total_volume, 2),
            'bb_min': [round(x, 2) for x in bb_min],
            'bb_max': [round(x, 2) for x in bb_max],
            'extent': [round(x, 1) for x in extent],
            'step_unit': step_unit,
            'scale_factor': scale_factor,
            'n_primitives': seam['n_primitives'],
            'has_seam': seam['has_seam'],
            'seam_segments': seam['seam_segments'],
            'aspect_mean': round(stats0['aspect_mean'], 3),
            'aspect_max': round(stats0['aspect_max'], 1),
        }

        # Store for baseline update
        results[step_file] = actual

        # Compare with baseline
        expected = baseline.get(step_file)
        if expected is None:
            pytest.skip(f"No baseline for {step_file} — run with --update-baseline")
            return

        checks = []
        checks.append(check_regression(actual['vertices'], expected.get('vertices'), 'vertices', TOL_VERTEX_COUNT))
        checks.append(check_regression(actual['faces'], expected.get('faces'), 'faces', TOL_FACE_COUNT))
        checks.append(check_regression(actual['volume_mm3'], expected.get('volume_mm3'), 'volume', TOL_VOLUME))
        for i, axis in enumerate(['X', 'Y', 'Z']):
            checks.append(check_regression(actual['extent'][i], expected.get('extent', [0, 0, 0])[i], f'extent_{axis}', TOL_BBOX))
        checks.append(check_regression(actual['n_primitives'], expected.get('n_primitives'), 'n_primitives', 0))
        checks.append(check_regression(actual['aspect_mean'], expected.get('aspect_mean'), 'aspect_mean', TOL_ASPECT_RATIO))

        # Check unit detection
        assert actual['step_unit'] == expected.get('step_unit', actual['step_unit']), \
            f"Unit changed: {actual['step_unit']} vs {expected.get('step_unit')}"

        failures = [msg for passed, msg in checks if not passed]
        assert not failures, "Regressions:\n" + "\n".join(failures)

    finally:
        if os.path.exists(glb_path):
            os.remove(glb_path)


def test_save_baseline(results):
    """After all tests, optionally save baseline."""
    if results and os.environ.get('UPDATE_BASELINE'):
        save_baseline(BASELINE_FILE, results)
        print(f"\nBaseline saved to {os.path.join(BASELINES_DIR, BASELINE_FILE)}")
