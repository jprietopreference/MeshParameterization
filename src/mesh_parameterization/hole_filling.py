"""
Boundary-loop hole filling using fan triangulation.

Holes are detected as boundary loops (edges belonging to exactly one
triangle) and filled by introducing a centroid vertex at the centre of
each loop and connecting it to all consecutive boundary vertices.

This approach approximates the Poisson Surface Fills step described in:
  "Quasi-Isometric Mesh Parameterization Using Heat-Based Geodesics
   and Poisson Surface Fills", MDPI Mathematics 7(8):753, 2019.
"""

import numpy as np


def fill_holes(vertices, faces):
    """Fill all boundary holes of a mesh with fan triangulation.

    For each closed boundary loop a centroid vertex is appended and
    triangulated against consecutive loop edges.

    Parameters
    ----------
    vertices : array_like, shape (n, 3)
        Vertex positions.
    faces : array_like, shape (m, 3)
        Triangle face indices (zero-based).

    Returns
    -------
    new_vertices : ndarray, shape (n + k, 3)
        Original vertices with *k* centroid vertices appended (one per loop).
    new_faces : ndarray, shape (m + p, 3)
        Original faces with *p* fill triangles appended.
    n_original : int
        Number of original vertices (= ``n``).  Useful for identifying
        which vertices were added during hole filling.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.intp)

    loops = _find_boundary_loops(faces)
    if not loops:
        return vertices.copy(), faces.copy(), len(vertices)

    new_vertices = list(vertices)
    new_faces = list(faces)
    n_original = len(vertices)

    for loop in loops:
        centroid = vertices[loop].mean(axis=0)
        c_idx = len(new_vertices)
        new_vertices.append(centroid)
        for a, b in zip(loop, loop[1:] + [loop[0]]):
            new_faces.append([a, b, c_idx])

    return (np.array(new_vertices, dtype=float),
            np.array(new_faces, dtype=np.intp),
            n_original)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _find_boundary_loops(faces):
    """Return a list of boundary loops as ordered vertex-index lists.

    A boundary edge is one that appears in exactly one triangle.  Loops
    are traced by following the half-edge adjacency around each boundary.

    Parameters
    ----------
    faces : (m, 3) int array

    Returns
    -------
    loops : list of list of int
        Each inner list is the ordered sequence of vertex indices forming
        one closed boundary loop.
    """
    # Build half-edge → opposite-direction half-edge map
    # A boundary edge (u, v) appears only once (as the directed half-edge u→v)
    # while interior edges appear twice with opposite orientations.
    edge_count = {}
    for tri in faces:
        for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            edge_count[(a, b)] = edge_count.get((a, b), 0) + 1

    # Boundary directed half-edges: those with no opposite
    boundary_edges = {(a, b) for (a, b), cnt in edge_count.items()
                      if cnt == 1 and (b, a) not in edge_count}

    if not boundary_edges:
        return []

    # Build adjacency: for boundary half-edge a→b, next starts at b
    next_vertex = {a: b for (a, b) in boundary_edges}

    visited = set()
    loops = []
    for start in list(next_vertex.keys()):
        if start in visited:
            continue
        loop = []
        v = start
        for _ in range(len(next_vertex) + 1):
            if v in visited:
                break
            visited.add(v)
            loop.append(v)
            v = next_vertex.get(v, None)
            if v is None or v == start:
                break
        if len(loop) >= 3:
            loops.append(loop)

    return loops
