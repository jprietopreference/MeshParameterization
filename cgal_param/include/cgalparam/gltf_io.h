#pragma once

#include "types.h"
#include <Eigen/Core>
#include <string>
#include <vector>
#include <cstdint>

namespace cgalparam {

/// Simple triangle mesh for I/O (same as heat project)
struct TriMesh {
    Eigen::MatrixXd V;   // n x 3
    Eigen::MatrixXi F;   // m x 3
    Eigen::MatrixXd UV;  // n x 2 (optional)

    int num_vertices() const { return static_cast<int>(V.rows()); }
    int num_faces() const { return static_cast<int>(F.rows()); }
    bool has_uvs() const { return UV.rows() == V.rows(); }
};

TriMesh load_gltf(const std::string& path);
void save_gltf(const std::string& path, const TriMesh& mesh);

/// Memory-based I/O (for WASM)
TriMesh load_gltf_from_memory(const std::vector<uint8_t>& data);
std::vector<uint8_t> save_gltf_to_memory(const TriMesh& mesh);

/// Convert between Eigen mesh and CGAL Surface_mesh
SurfaceMesh to_cgal_mesh(const TriMesh& mesh);
TriMesh from_cgal_mesh(const SurfaceMesh& sm);

} // namespace cgalparam
