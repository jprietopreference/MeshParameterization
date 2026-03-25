"""Shared fixtures and config for tessellation + parameterization tests."""
import os
import json
import pytest
import struct
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
STEP_DIR = os.path.join(MODELS_DIR, "step")
GLTF_DIR = os.path.join(MODELS_DIR, "glTF")
PIPELINE = os.path.join(ROOT, "scripts", "occ_gmsh_pipeline.py")
BENCH_CLI = os.path.join(ROOT, "server", "build", "meshparam_bench.exe")
BASELINES_DIR = os.path.join(os.path.dirname(__file__), "baselines")

# Tolerances for regression checks
TOL_VERTEX_COUNT = 0.0     # exact match
TOL_FACE_COUNT = 0.0       # exact match
TOL_VOLUME = 0.02          # 2% relative
TOL_BBOX = 0.01            # 1% relative
TOL_ASPECT_RATIO = 0.05    # 5% relative
TOL_SD = 0.05              # 5% relative for Symmetric Dirichlet
TOL_FLIPS = 0              # exact match


def step_files():
    """List all STEP files in models/step/."""
    exts = ('.step', '.stp', '.STEP', '.STP')
    if not os.path.isdir(STEP_DIR):
        return []
    return sorted([f for f in os.listdir(STEP_DIR) if any(f.endswith(e) for e in exts)])


def gltf_files():
    """List all GLB files in models/glTF/."""
    if not os.path.isdir(GLTF_DIR):
        return []
    return sorted([f for f in os.listdir(GLTF_DIR) if f.endswith('.glb')])


def parse_glb(path):
    """Parse a GLB file and return gltf JSON + binary data."""
    with open(path, 'rb') as f:
        magic, ver, length = struct.unpack('<4sII', f.read(12))
        json_len, json_type = struct.unpack('<II', f.read(8))
        gltf = json.loads(f.read(json_len))
        bin_len, bin_type = struct.unpack('<II', f.read(8))
        bin_data = f.read(bin_len)
    return gltf, bin_data


def mesh_volume_from_glb(gltf, bin_data, prim_idx=0):
    """Compute signed volume of a mesh primitive using divergence theorem."""
    acc = gltf['accessors']
    bv = gltf['bufferViews']
    prim = gltf['meshes'][0]['primitives'][prim_idx]

    pos_a = acc[prim['attributes']['POSITION']]
    pos_bv = bv[pos_a['bufferView']]
    pos = np.frombuffer(bin_data, dtype=np.float32,
                        offset=pos_bv['byteOffset'], count=pos_a['count'] * 3).reshape(-1, 3)

    idx_a = acc[prim['indices']]
    idx_bv = bv[idx_a['bufferView']]
    idx = np.frombuffer(bin_data, dtype=np.uint32,
                        offset=idx_bv['byteOffset'], count=idx_a['count']).reshape(-1, 3)

    # Signed volume: sum of (v0 . (v1 x v2)) / 6
    v0 = pos[idx[:, 0]]
    v1 = pos[idx[:, 1]]
    v2 = pos[idx[:, 2]]
    vol = np.sum(v0 * np.cross(v1, v2)) / 6.0
    return abs(vol)


def mesh_stats_from_glb(gltf, bin_data, prim_idx=0):
    """Compute mesh statistics for a primitive."""
    acc = gltf['accessors']
    bv = gltf['bufferViews']
    prim = gltf['meshes'][0]['primitives'][prim_idx]

    pos_a = acc[prim['attributes']['POSITION']]
    pos_bv = bv[pos_a['bufferView']]
    pos = np.frombuffer(bin_data, dtype=np.float32,
                        offset=pos_bv['byteOffset'], count=pos_a['count'] * 3).reshape(-1, 3)

    idx_a = acc[prim['indices']]
    idx_bv = bv[idx_a['bufferView']]
    idx = np.frombuffer(bin_data, dtype=np.uint32,
                        offset=idx_bv['byteOffset'], count=idx_a['count']).reshape(-1, 3)

    nv = pos_a['count']
    nf = idx_a['count'] // 3

    # Bounding box
    bb_min = pos.min(axis=0).tolist()
    bb_max = pos.max(axis=0).tolist()

    # Aspect ratio
    v0, v1, v2 = pos[idx[:, 0]], pos[idx[:, 1]], pos[idx[:, 2]]
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)
    longest = np.maximum(e0, np.maximum(e1, e2))
    shortest = np.minimum(e0, np.minimum(e1, e2))
    ratio = np.where(shortest > 1e-10, longest / shortest, 999)

    return {
        'vertices': nv,
        'faces': nf,
        'bb_min': bb_min,
        'bb_max': bb_max,
        'aspect_mean': float(ratio.mean()),
        'aspect_max': float(ratio.max()),
    }


def load_baseline(name):
    """Load a baseline JSON file. Returns {} if not found."""
    path = os.path.join(BASELINES_DIR, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_baseline(name, data):
    """Save baseline JSON file."""
    os.makedirs(BASELINES_DIR, exist_ok=True)
    path = os.path.join(BASELINES_DIR, name)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


def check_regression(actual, expected, key, tol, is_relative=True):
    """Check if actual value regressed vs expected. Returns (passed, message)."""
    if expected is None:
        return True, f"{key}: no baseline (new)"
    if tol == 0:
        if actual != expected:
            return False, f"{key}: {actual} != {expected} (exact)"
        return True, f"{key}: {actual} == {expected}"
    if is_relative:
        if expected == 0:
            diff = abs(actual)
        else:
            diff = abs(actual - expected) / abs(expected)
        if diff > tol:
            return False, f"{key}: {actual} vs {expected} (diff={diff:.4f} > tol={tol})"
        return True, f"{key}: {actual} vs {expected} (diff={diff:.4f})"
    else:
        diff = abs(actual - expected)
        if diff > tol:
            return False, f"{key}: {actual} vs {expected} (diff={diff} > tol={tol})"
        return True, f"{key}: {actual} vs {expected}"
