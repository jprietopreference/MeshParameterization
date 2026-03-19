#include "meshparam/mesh.h"
#include <igl/boundary_loop.h>
#include <igl/per_vertex_normals.h>
#include <algorithm>
#include <cmath>

namespace meshparam {

bool TriMesh::has_boundary() const {
    std::vector<std::vector<int>> loops;
    igl::boundary_loop(F, loops);
    // Has boundary if there are any boundary loops
    // Has holes if there are more than one boundary loop
    return !loops.empty();
}

double TriMesh::max_edge_length() const {
    double max_len = 0.0;
    for (int f = 0; f < F.rows(); ++f) {
        for (int e = 0; e < 3; ++e) {
            int i = F(f, e);
            int j = F(f, (e + 1) % 3);
            double len = (V.row(i) - V.row(j)).norm();
            max_len = std::max(max_len, len);
        }
    }
    return max_len;
}

std::vector<std::vector<int>> TriMesh::boundary_loops() const {
    std::vector<std::vector<int>> loops;
    igl::boundary_loop(F, loops);
    return loops;
}

void TriMesh::compute_normals() {
    igl::per_vertex_normals(V, F, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA, N);
}

} // namespace meshparam
