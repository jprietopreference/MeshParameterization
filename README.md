# MeshParameterization

Quasi-isometric UV parameterisation for GLTF meshes that lack texture
coordinates, based on the paper:

> **Quasi-Isometric Mesh Parameterization Using Heat-Based Geodesics and
> Poisson Surface Fills**  
> *MDPI Mathematics 7(8):753, 2019.*  
> <https://www.mdpi.com/2227-7390/7/8/753>

---

## Algorithm

The pipeline consists of four stages:

1. **Hole filling** – boundary loops are detected and closed with fan
   triangulation from a centroid vertex (approximating the Poisson
   Surface Fills step from the paper).

2. **Heat-based geodesics** (Crane et al., 2013) – pairwise geodesic
   distances are computed by:
   - solving a single implicit heat-diffusion step,
   - normalising the gradient field,
   - recovering distances via a Poisson solve.

3. **Classical MDS** – the pairwise geodesic distance matrix is
   double-centred and eigendecomposed; the top-2 eigenvectors give a
   quasi-isometric 2-D embedding.

4. **UV normalisation** – embedding coordinates are scaled to
   `[0, 1]²` and written back as `TEXCOORD_0` to the GLTF file.

---

## Project layout

```
src/mesh_parameterization/
    __init__.py         public API
    laplacian.py        cotangent Laplacian + lumped mass matrix
    heat_geodesics.py   heat-based geodesic distance solver
    hole_filling.py     fan-triangulation hole filling
    mds.py              classical multidimensional scaling
    parameterize.py     full UV-parameterisation pipeline
main.py                 CLI entry point
tests/                  pytest test suite
requirements.txt
setup.py
```

---

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

```bash
python main.py <input.gltf|input.glb> <output.gltf|output.glb> [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--t-coeff FLOAT` | `1.0` | Heat time-step coefficient `t = t_coeff · h²` |
| `--max-verts INT` | `2000` | Max vertices for full all-pairs MDS; larger meshes use landmark sampling |
| `--verbose` | off | Print progress |

### Example

```bash
python main.py bunny.glb bunny_uv.glb --verbose
```

---

## Running tests

```bash
python -m pytest tests/ -v
```

---

## References

- Crane K., Weischedel C., Wardetzky M., *Geodesics in Heat*, ACM ToG 32(5), 2013.
- Torgerson W.S., *Multidimensional Scaling: I. Theory and Method*, Psychometrika 17(4), 1952.
- Desbrun et al., *Implicit Fairing of Irregular Meshes*, SIGGRAPH 1999.
