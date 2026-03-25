# MeshParameterization

## Project Overview

UV parameterization of triangle meshes from STEP/glTF input, targeting both native (C++) and browser (WASM) builds. Broker architecture: 8 parameterization methods compete, best result selected automatically by Symmetric Dirichlet energy + flip count.

**Target use case**: Window/door hardware parts (handles, hinges) from STEP files.

## Architecture

### Server (`/server`)
- **HTTP API** (`main.cpp`): `/api/tessellate/step`, `/api/parameterize`, `/api/health`, `/api/docs` (Swagger UI)
- **OpenAPI spec**: `web/openapi.yaml` — full documentation of all endpoints and GLB metadata
- **Broker**: spawns `meshparam_bench.exe` subprocesses per method (process isolation — crashes don't kill the server)
- **8 methods**: heat, lscm, igl_arap, slim, cgal_conformal, cgal_arap, cgal_authalic, cm (Composite Majorization)
- **CLI tools**: `meshparam_bench.exe <method> <input.obj|.glb>` — self-contained per-method parameterizer
- **Dependencies**: Eigen, libigl, CGAL (vcpkg), tinygltf, Spectra, Intel MKL (for CM/Pardiso)
- **MKL DLLs**: Must be in `server/build/` (copy from `C:/vcpkg/installed/x64-windows/bin/mkl_*.dll`)

### Gmsh Pipeline (`/scripts/occ_gmsh_pipeline.py`)
- **STEP → GLB**: OCC import → heal → Gmsh isotropic meshing → OCC normals → B-Rep attributes → GLB
- **Chord deviation target**: 1mm (curvature-adaptive, `MeshSizeFromCurvature`)
- **Auto-seam**: Finds longest Z-perpendicular B-Rep edge loop, splits mesh into front/back at CAD level
- **Split by topology**: Flood-fill from OCC face adjacency graph, seam edges as barrier → 2 glTF primitives
- **B-Rep edge visualization**: Edge polylines in GLB node extras (`edgeLines: {seam, zperp, other}`)
- **Edge offset**: 0.1mm along surface normal to avoid Z-fighting
- **Edge node ordering**: Sorted by parametric coordinate along OCC curves
- **Output**: 1-2 triangle primitives (front/back) sharing one vertex buffer + `edgeLines` JSON extras

### Heat-Geodesic Parameterizer (`/src`, `/include/meshparam`)
- **Pipeline**: Laplace-Beltrami + Mass matrix → Heat equation → Normalized flux → Poisson solve → MDS eigendecomposition → UV coords
- **Limitation**: O(n²) geodesic matrix — practical limit ~5K vertices

### CGAL Parameterizer (`/cgal_param`, `/cgal_param_native`)
- **Methods**: Discrete Conformal, ARAP, Discrete Authalic (LSCM disabled — CGAL 6.x/Boost 1.90 regression)
- **Kernel**: `Simple_cartesian<double>` for WASM, `EPICK` for native
- **Seam cut**: BFS geodesic (fallback when no auto-seam), or auto-seam from Z-perp loop

### Composite Majorization (`/extern/CompMajor`)
- **Algorithm**: Newton-based Symmetric Dirichlet optimizer
- **Dependencies**: Intel MKL (Pardiso sparse solver)
- **Best quality** when it converges (99.4% zero-flip), 86% success rate

### Frontend (`/web`)
- **BabylonJS** viewer with checkerboard texture, seam visualization, B-Rep face edges
- **ViewCube**: Separate BabylonJS engine on overlay canvas (200x200, top-right), face/edge/corner picking, ortho/perspective toggle, animated transitions
- **Edge visualization**: `CreateLineSystem` from `edgeLines` JSON extras, parented to glTF root node (inherits LH Z-flip)
- **Color coding**: Red=seam, Orange=Z-perp edges, Yellow=other B-Rep edges
- **Split mesh display**: Pale yellow (front/Z+), pale green (back) before parameterization
- **Checkerboard textures**: Front=B/W (`checker.png`), Back=pale red/grey (`checker_grey.png`), emissive texture for uniform shading
- **Show parameterization toggle**: Checkbox switches between parameterized GLB (checkerboard) and original tessellated GLB (with correct OCC normals) — full mesh reload, not just material swap
- **Vertex normals**: White lines (2% of bbox diagonal), toggle via checkbox
- **Custom GLB attributes**: `_SEAM`, `_FACE_ID` — injected into BabylonJS via raw GLB parsing
- **Depth precision**: Camera near/far clip tightened to scene bounds (1%-1000% of extent)
- **Swagger UI**: `web/swagger.html` + `swagger-ui-dist` npm package → `/api/docs`

### Benchmark (`/scripts/run_benchmark.py`, `/benchmark`)
- **Dataset**: Stein et al. 2022, 4,826 disk-topology meshes
- **CLI-based**: Each method runs as subprocess with timeout (process isolation)
- **Metrics**: Symmetric Dirichlet energy, flipped triangles, L2/L∞ area distortion

## Benchmark Results (4,819 meshes)

| Method | Success | Median SD | 0-flip% | Wins |
|--------|---------|-----------|---------|------|
| CM | 86% | 0.0055 | 99.4% | 1,423 |
| CGAL ARAP | 94% | 0.0078 | 95.5% | 1,151 |
| SLIM | 100% | 0.0084 | 83.8% | 761 |
| LSCM | 100% | 0.0105 | 85.2% | 725 |
| igl ARAP | 100% | 0.0074 | 63.4% | 585 |
| **Broker** | **100%** | **0.0057** | **94.7%** | |

## Key Decisions
- **C++ over Rust**: better library coverage (libigl, CGAL, Eigen)
- **Gmsh + OpenCascade** for meshing: isotropic, curvature-adaptive, direct from B-Rep (not remeshing)
- **Normals from OCC faces**: queried at parametric positions via `gmsh.model.getNormal()`
- **All meshes in mm**: STEP files auto-detected and scaled at import
- **Process isolation**: Each parameterization method runs as subprocess — crashes don't affect broker
- **BabylonJS left-handed**: glTF loader negates Z; edge lines parented to glTF root node for correct transform
- **Edge data as JSON extras**: Not glTF line primitives (BabylonJS doesn't handle mixed primitives well)
- **Edge normal offset**: 0.1mm along surface normal avoids Z-fighting without depth bias hacks
- **Auto-seam at CAD level**: Split OCC faces into front/back groups before meshing (not post-tessellation)
- **Stein ADMM removed**: Crashed on ~50% of meshes, replaced by CM which is strictly better

## Build Commands

```bash
# Server + bench CLI (native, requires VS2022 + vcpkg + MKL)
cd server && cmake -G Ninja -B build -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake && cmake --build build
# Copy MKL DLLs after build:
cp C:/vcpkg/installed/x64-windows/bin/mkl_*.dll server/build/

# CGAL (WASM-compatible, no GMP)
cd cgal_param && cmake -G Ninja -B build && cmake --build build

# Frontend (Vite dev server)
cd web && npx vite  # → http://localhost:5199

# Run server
server/build/meshparam_server.exe --port 8080 --web-root web --gmsh-cli "python scripts/occ_gmsh_pipeline.py"

# Swagger docs
# → http://localhost:8080/api/docs

# Benchmark (CLI, process-isolated)
python scripts/run_benchmark.py --subset disk --workers 1 --timeout 120
```

## Test Data
- STEP files: `models/step/0627778.step`, `models/step/0618969.step`, `models/step/0617023B.step`
- Original glTF: `models/glTF/` (cube, sphere, torus, teapot, Klein bottle, CGAL refs)
- Benchmark: `models/benchmark/Obj_Files/` (Stein et al. 2022 dataset, 11,913 meshes)
- Results: `models/benchmark_results/benchmark_raw.json`, `benchmark_cm_raw.json`

## Folder Structure
```
models/
├── glTF/               Original tessellated meshes (.glb)
├── step/               STEP CAD files for testing
├── benchmark/          Stein et al. 2022 dataset (gitignored, large)
└── benchmark_results/  Benchmark run outputs (gitignored)
```

### Recovering benchmark data
The benchmark dataset is too large for git. To set up locally:
```bash
# Download Stein et al. 2022 dataset from:
# https://github.com/SteinEtAl/ParameterizationBenchmark
# Extract Obj_Files/ into models/benchmark/
mkdir -p models/benchmark
# Place Obj_Files/ (with Artist_UVs/, No_UVs/, Full/, dataset_tags.csv) inside
```
