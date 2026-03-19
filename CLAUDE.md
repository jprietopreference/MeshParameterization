# MeshParameterization

## Project Overview

UV parameterization of triangle meshes from glTF input, targeting both native (C++) and browser (WASM) builds. Based on the paper "Quasi-Isometric Mesh Parameterization Using Heat-Based Geodesics and Poisson Surface Fills" (Mejia-Parra et al., Mathematics 2019).

**Preferred approach: Heat-geodesic method** — best area preservation, no boundary cut needed for closed surfaces, works on all topologies.

## Architecture

### Heat-Geodesic Parameterizer (`/src`, `/include/meshparam`)
- **Pipeline**: Laplace-Beltrami + Mass matrix → Heat equation (implicit Euler) → Normalized flux → Poisson solve for geodesics → MDS eigendecomposition → UV coords
- **Dependencies**: Eigen (sparse solvers), libigl (cotangent Laplacian, gradient operator), tinygltf (glTF I/O), Spectra (eigenvalues)
- **Build**: CMake + FetchContent, MSVC/Ninja. `build/meshparam_cli.exe`
- **Limitation**: O(n²) geodesic matrix — practical limit ~5K vertices

### CGAL Parameterizer (`/cgal_param`, `/cgal_param_native`)
- **Methods**: Discrete Conformal, ARAP, Discrete Authalic, Mean Value Coordinates (LSCM disabled — CGAL 6.x/Boost 1.90 regression)
- **Kernel**: `Simple_cartesian<double>` for WASM, `EPICK` for native — **produces identical results** (verified)
- **Seam cut**: Geodesic path between BFS-diameter poles for closed meshes
- **Native build** uses vcpkg CGAL (C:/vcpkg) with GMP/MPFR
- **No GMP needed** for parameterization quality — confirmed by EPICK vs Simple_cartesian comparison

### Test Mesh Generation (`/scripts/generate_test_meshes.py`)
- Uses **Gmsh + OpenCascade** for geometry and isotropic meshing
- Mesh error target: ~1mm chord deviation (curvature-adaptive)
- OCC surface normals: per-vertex, queried at parametric positions via `gmsh.model.getNormal()`
- Output: shared-vertex .glb + .occmesh.npz sidecar (split mesh with per-vertex OCC normals)

### Result Assembly (`/scripts/assemble_result.py`, `/scripts/apply_checkerboard.py`)
- Combines parameterized UVs + OCC normals (smooth within B-Rep faces, sharp at face boundaries) + 25mm checkerboard texture
- Checkerboard: 2x2 B/W PNG, MIRRORED_REPEAT, KHR_texture_transform for UV scaling
- Non-OCC meshes (CGAL refs, Klein bottle parametric) get face-averaged smooth vertex normals

## Key Decisions
- **C++ over Rust**: better library coverage (libigl, CGAL, Eigen) for geometry processing
- **Gmsh + OpenCascade** for test geometry: isotropic meshing, analytical surface normals
- **Normals from OCC faces**: each triangle's normal comes from the parent B-Rep surface at the triangle centroid, not from the tessellation. Vertices split at OCC face boundaries for sharp creases.
- **All meshes in mm**: STEP files in other units (e.g., teapot in inches) scaled to mm at import
- **Checkerboard UV scale**: `max_extent / 50.0` for mm meshes, fixed 4.0 for small/normalized meshes

## Build Commands

```bash
# Heat-geodesic (native)
# Requires: VS2022, Ninja, CMake
powershell -Command "& { Enter-VsDevShell ...; cmake -G Ninja -B build; cmake --build build }"

# CGAL (WASM-compatible, no GMP)
cd cgal_param && cmake -G Ninja -B build && cmake --build build

# CGAL (native with EPICK, uses vcpkg)
cd cgal_param_native && cmake -G Ninja -B build -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake && cmake --build build

# Generate test meshes
python scripts/generate_test_meshes.py

# Run full comparison (parallel)
python scripts/run_all_parallel.py
```

## Test Data
- OCC meshes: cube, filleted cube, sphere, torus (2 sizes), Klein bottle (OCC STEP), teapot (STEP, scaled from inches)
- CGAL reference meshes: nefertiti, three_peaks, head, mushroom (from CGAL data, scaled to ~100mm)
- Paper: `Documents/mathematics-07-00753.pdf`
- STEP files: `Documents/KleinBottle.STEP`, `Documents/teapot.stp`
