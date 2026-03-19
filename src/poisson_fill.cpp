#include "meshparam/poisson_fill.h"
#include <igl/boundary_loop.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/per_vertex_normals.h>
#include <Eigen/Sparse>
#include <vector>
#include <set>
#include <map>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace meshparam {

namespace {

/// Simple hole filling by fan-triangulation of boundary loops.
/// For each boundary loop, compute centroid, add it as a new vertex,
/// and create fan triangles. This is a simplified Poisson fill that
/// works well for small holes and mild concavities.
///
/// A full Poisson surface reconstruction (Kazhdan 2006) operates in
/// R³ with octree discretization. For the MVP we use this simpler
/// approach that achieves the paper's goal: giving geodesics a path
/// across holes so they don't distort.
void fill_boundary_loops(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    Eigen::MatrixXd& V_out,
    Eigen::MatrixXi& F_out,
    std::vector<int>& vertex_map)
{
    std::vector<std::vector<int>> loops;
    igl::boundary_loop(F, loops);

    if (loops.empty()) {
        V_out = V;
        F_out = F;
        vertex_map.resize(V.rows());
        for (int i = 0; i < static_cast<int>(V.rows()); ++i)
            vertex_map[i] = i;
        return;
    }

    // Start with original mesh
    int n_orig = static_cast<int>(V.rows());
    int m_orig = static_cast<int>(F.rows());

    // Count new vertices (one centroid per hole) and new faces
    int n_new_verts = 0;
    int n_new_faces = 0;
    // Skip the largest boundary loop (that's the outer boundary, not a hole)
    // Sort loops by size, largest first
    std::vector<size_t> loop_order(loops.size());
    std::iota(loop_order.begin(), loop_order.end(), 0);
    std::sort(loop_order.begin(), loop_order.end(),
              [&](size_t a, size_t b) { return loops[a].size() > loops[b].size(); });

    // If there's only one boundary loop, the mesh has no holes—just a border
    // Still fill if there are concavities (we fill all loops for simplicity
    // since the paper wants M* to contain M and have no holes)
    std::vector<size_t> loops_to_fill;
    if (loops.size() > 1) {
        // Fill all loops except the largest (outer boundary)
        for (size_t i = 1; i < loop_order.size(); ++i) {
            loops_to_fill.push_back(loop_order[i]);
        }
    }

    for (size_t idx : loops_to_fill) {
        n_new_verts += 1;  // centroid vertex
        n_new_faces += static_cast<int>(loops[idx].size());  // fan triangles
    }

    V_out.resize(n_orig + n_new_verts, 3);
    V_out.topRows(n_orig) = V;

    F_out.resize(m_orig + n_new_faces, 3);
    F_out.topRows(m_orig) = F;

    // Map: original vertex i → M* vertex i (identity for original verts)
    vertex_map.resize(n_orig);
    for (int i = 0; i < n_orig; ++i)
        vertex_map[i] = i;

    int v_cursor = n_orig;
    int f_cursor = m_orig;

    for (size_t idx : loops_to_fill) {
        const auto& loop = loops[idx];
        int loop_size = static_cast<int>(loop.size());

        // Compute centroid of boundary loop
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        for (int vi : loop) {
            centroid += V.row(vi).transpose();
        }
        centroid /= loop_size;

        // Add centroid vertex
        int centroid_idx = v_cursor;
        V_out.row(centroid_idx) = centroid.transpose();
        v_cursor++;

        // Create fan triangles
        for (int i = 0; i < loop_size; ++i) {
            int v0 = loop[i];
            int v1 = loop[(i + 1) % loop_size];
            F_out.row(f_cursor) << v0, v1, centroid_idx;
            f_cursor++;
        }
    }
}

} // anonymous namespace

PoissonFillResult poisson_fill(const TriMesh& mesh) {
    PoissonFillResult result;

    fill_boundary_loops(mesh.V, mesh.F,
                        result.extended_mesh.V,
                        result.extended_mesh.F,
                        result.original_vertex_map);

    return result;
}

Eigen::MatrixXd trim_parameterization(
    const Eigen::MatrixXd& extended_uv,
    const PoissonFillResult& fill_result,
    int original_vertex_count)
{
    Eigen::MatrixXd uv(original_vertex_count, 2);
    for (int i = 0; i < original_vertex_count; ++i) {
        int mapped = fill_result.original_vertex_map[i];
        uv.row(i) = extended_uv.row(mapped);
    }
    return uv;
}

} // namespace meshparam
