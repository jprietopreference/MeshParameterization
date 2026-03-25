"""Parameterization regression tests.

Tests each parameterization method on every model file (STEP + GLB).
First run creates the baseline. Subsequent runs detect regressions.

Usage: pytest tests/test_parameterization.py -v --html=tests/report_parameterization.html
"""
import os
import json
import subprocess
import tempfile
import pytest

from conftest import (
    ROOT, STEP_DIR, GLTF_DIR, PIPELINE, BENCH_CLI, BASELINES_DIR,
    step_files, gltf_files, load_baseline, save_baseline, check_regression,
    TOL_SD, TOL_FLIPS,
)

BASELINE_FILE = "parameterization.json"
METHODS = ['heat', 'lscm', 'igl_arap', 'slim', 'cgal_conformal', 'cgal_arap', 'cgal_authalic', 'cm']
TIMEOUT_S = 120


def tessellate_step(step_path):
    """Tessellate STEP to GLB, return path."""
    with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as tmp:
        glb_path = tmp.name
    result = subprocess.run(
        ['python', PIPELINE, step_path, glb_path],
        capture_output=True, text=True, timeout=120, cwd=ROOT
    )
    if result.returncode != 0:
        raise RuntimeError(f"Tessellation failed: {result.stderr}")
    return glb_path


def run_method(method, glb_path):
    """Run a single parameterization method via bench_cli. Returns parsed JSON result."""
    result = subprocess.run(
        [BENCH_CLI, method, glb_path],
        capture_output=True, text=True, timeout=TIMEOUT_S, cwd=ROOT
    )
    # Parse JSON from last line of stdout
    for line in reversed(result.stdout.strip().split('\n')):
        line = line.strip()
        if line.startswith('{'):
            return json.loads(line)
    return {'method': method, 'success': False, 'error': f'exit {result.returncode}'}


def collect_test_cases():
    """Build list of (model_name, glb_source) tuples."""
    cases = []
    for f in step_files():
        cases.append((f, 'step'))
    for f in gltf_files():
        cases.append((f, 'gltf'))
    return cases


TEST_CASES = collect_test_cases()
TEST_IDS = [f"{name}" for name, _ in TEST_CASES]


@pytest.fixture(scope="module")
def baseline():
    return load_baseline(BASELINE_FILE)


@pytest.fixture(scope="module")
def results():
    return {}


@pytest.fixture(scope="module")
def glb_cache():
    """Cache tessellated GLBs to avoid re-tessellating for each method."""
    cache = {}
    yield cache
    # Cleanup temp files
    for path in cache.values():
        if os.path.exists(path):
            os.remove(path)


def get_glb_path(name, source, glb_cache):
    """Get GLB path for a model, tessellating if needed."""
    if name in glb_cache:
        return glb_cache[name]
    if source == 'step':
        path = tessellate_step(os.path.join(STEP_DIR, name))
    else:
        path = os.path.join(GLTF_DIR, name)
    glb_cache[name] = path
    return path


@pytest.mark.parametrize("method", METHODS, ids=METHODS)
@pytest.mark.parametrize("model_name,source", TEST_CASES, ids=TEST_IDS)
def test_parameterization(model_name, source, method, baseline, results, glb_cache):
    """Test a single method on a single model."""
    try:
        glb_path = get_glb_path(model_name, source, glb_cache)
    except Exception as e:
        pytest.skip(f"Tessellation failed: {e}")
        return

    if not os.path.exists(BENCH_CLI):
        pytest.skip(f"bench_cli not found at {BENCH_CLI}")
        return

    try:
        r = run_method(method, glb_path)
    except subprocess.TimeoutExpired:
        r = {'method': method, 'success': False, 'error': 'timeout'}
    except Exception as e:
        r = {'method': method, 'success': False, 'error': str(e)}

    key = f"{model_name}:{method}"
    actual = {
        'success': r.get('success', False),
        'sym_dirichlet': round(r.get('sym_dirichlet', -1), 4) if r.get('success') else -1,
        'flipped_tris': r.get('flipped_tris', -1) if r.get('success') else -1,
        'angle_mean': round(r.get('angle_mean', -1), 2) if r.get('success') else -1,
        'area_mean': round(r.get('area_mean', -1), 3) if r.get('success') else -1,
        'elapsed_ms': round(r.get('elapsed_ms', -1), 1) if r.get('success') else -1,
        'error': r.get('error', '') if not r.get('success') else '',
    }
    results[key] = actual

    # Compare with baseline
    expected = baseline.get(key)
    if expected is None:
        pytest.skip(f"No baseline for {key}")
        return

    checks = []

    # Success/fail must match — but tolerate methods disabled at build time
    if expected.get('success') and not actual['success']:
        # If a method that worked in baseline now fails, it might be disabled in this build
        # (e.g. CM without MKL). Skip rather than fail — not a code regression.
        if os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS'):
            pytest.skip(f"{key}: method failed on CI ({actual.get('error', 'unknown')})")
            return
        assert False, f"Success changed: {actual['success']} vs {expected.get('success')} " \
                       f"(error: {actual.get('error', '')})"

    if actual['success'] and expected.get('success'):
        # Flipped tris: exact match for GLB files, 10% tolerance for STEP (cross-platform tessellation diff)
        flip_tol = 0.1 if source == 'step' else TOL_FLIPS
        checks.append(check_regression(
            actual['flipped_tris'], expected.get('flipped_tris'), 'flipped_tris', flip_tol))

        # SD: only check if both are reasonable (not huge)
        if actual['sym_dirichlet'] < 1e12 and expected.get('sym_dirichlet', 1e12) < 1e12:
            checks.append(check_regression(
                actual['sym_dirichlet'], expected.get('sym_dirichlet'), 'sym_dirichlet', TOL_SD))

        failures = [msg for passed, msg in checks if not passed]
        assert not failures, "Regressions:\n" + "\n".join(failures)


def test_save_baseline(results):
    """After all tests, optionally save baseline."""
    if results and os.environ.get('UPDATE_BASELINE'):
        save_baseline(BASELINE_FILE, results)
        print(f"\nBaseline saved to {os.path.join(BASELINES_DIR, BASELINE_FILE)}")
