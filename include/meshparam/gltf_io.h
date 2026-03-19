#pragma once

#include "mesh.h"
#include <string>
#include <vector>

namespace meshparam {

/// Load a triangle mesh from a glTF/glb file.
/// Extracts the first mesh/primitive found.
TriMesh load_gltf(const std::string& path);

/// Load from in-memory buffer (for WASM).
TriMesh load_gltf_from_memory(const std::vector<uint8_t>& data);

/// Save mesh with UV parameterization to glTF file.
void save_gltf(const std::string& path, const TriMesh& mesh);

/// Save to in-memory buffer (for WASM).
std::vector<uint8_t> save_gltf_to_memory(const TriMesh& mesh);

} // namespace meshparam
