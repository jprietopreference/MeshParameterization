#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "meshparam/gltf_io.h"
#include <igl/per_vertex_normals.h>
#include <tiny_gltf.h>
#include <stdexcept>
#include <fstream>
#include <cstring>

namespace meshparam {

namespace {

/// Extract typed data from a glTF accessor
template <typename T>
std::vector<T> extract_accessor(const tinygltf::Model& model, int accessor_idx) {
    const auto& accessor = model.accessors[accessor_idx];
    const auto& bufferView = model.bufferViews[accessor.bufferView];
    const auto& buffer = model.buffers[bufferView.buffer];

    const uint8_t* base = buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;
    size_t stride = bufferView.byteStride;
    if (stride == 0) stride = sizeof(T);

    std::vector<T> result(accessor.count);
    for (size_t i = 0; i < accessor.count; ++i) {
        std::memcpy(&result[i], base + i * stride, sizeof(T));
    }
    return result;
}

TriMesh parse_model(const tinygltf::Model& model) {
    if (model.meshes.empty()) {
        throw std::runtime_error("glTF file contains no meshes");
    }

    const auto& mesh = model.meshes[0];
    if (mesh.primitives.empty()) {
        throw std::runtime_error("glTF mesh contains no primitives");
    }

    const auto& primitive = mesh.primitives[0];
    if (primitive.mode != TINYGLTF_MODE_TRIANGLES && primitive.mode != -1) {
        throw std::runtime_error("Only triangle primitives are supported");
    }

    TriMesh result;

    // Extract positions
    auto pos_it = primitive.attributes.find("POSITION");
    if (pos_it == primitive.attributes.end()) {
        throw std::runtime_error("Mesh has no POSITION attribute");
    }

    const auto& pos_accessor = model.accessors[pos_it->second];
    int n = static_cast<int>(pos_accessor.count);
    result.V.resize(n, 3);

    const auto& pos_bv = model.bufferViews[pos_accessor.bufferView];
    const auto& pos_buf = model.buffers[pos_bv.buffer];
    const uint8_t* pos_base = pos_buf.data.data() + pos_bv.byteOffset + pos_accessor.byteOffset;
    size_t pos_stride = pos_bv.byteStride ? pos_bv.byteStride : 3 * sizeof(float);

    for (int i = 0; i < n; ++i) {
        const float* p = reinterpret_cast<const float*>(pos_base + i * pos_stride);
        result.V(i, 0) = p[0];
        result.V(i, 1) = p[1];
        result.V(i, 2) = p[2];
    }

    // Extract indices
    if (primitive.indices >= 0) {
        const auto& idx_accessor = model.accessors[primitive.indices];
        int m = static_cast<int>(idx_accessor.count) / 3;
        result.F.resize(m, 3);

        const auto& idx_bv = model.bufferViews[idx_accessor.bufferView];
        const auto& idx_buf = model.buffers[idx_bv.buffer];
        const uint8_t* idx_base = idx_buf.data.data() + idx_bv.byteOffset + idx_accessor.byteOffset;

        for (int i = 0; i < m * 3; ++i) {
            int idx = 0;
            switch (idx_accessor.componentType) {
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                    uint16_t val;
                    std::memcpy(&val, idx_base + i * sizeof(uint16_t), sizeof(uint16_t));
                    idx = val;
                    break;
                }
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
                    uint32_t val;
                    std::memcpy(&val, idx_base + i * sizeof(uint32_t), sizeof(uint32_t));
                    idx = val;
                    break;
                }
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
                    idx = idx_base[i];
                    break;
                }
                default:
                    throw std::runtime_error("Unsupported index component type");
            }
            result.F(i / 3, i % 3) = idx;
        }
    } else {
        // No indices: sequential triangles
        int m = n / 3;
        result.F.resize(m, 3);
        for (int i = 0; i < m; ++i) {
            result.F(i, 0) = i * 3;
            result.F(i, 1) = i * 3 + 1;
            result.F(i, 2) = i * 3 + 2;
        }
    }

    // Extract existing normals if present
    auto nrm_it = primitive.attributes.find("NORMAL");
    if (nrm_it != primitive.attributes.end()) {
        const auto& nrm_accessor = model.accessors[nrm_it->second];
        result.N.resize(n, 3);

        const auto& nrm_bv = model.bufferViews[nrm_accessor.bufferView];
        const auto& nrm_buf = model.buffers[nrm_bv.buffer];
        const uint8_t* nrm_base = nrm_buf.data.data() + nrm_bv.byteOffset + nrm_accessor.byteOffset;
        size_t nrm_stride = nrm_bv.byteStride ? nrm_bv.byteStride : 3 * sizeof(float);

        for (int i = 0; i < n; ++i) {
            const float* nr = reinterpret_cast<const float*>(nrm_base + i * nrm_stride);
            result.N(i, 0) = nr[0];
            result.N(i, 1) = nr[1];
            result.N(i, 2) = nr[2];
        }
    }

    // Extract existing UVs if present
    auto uv_it = primitive.attributes.find("TEXCOORD_0");
    if (uv_it != primitive.attributes.end()) {
        const auto& uv_accessor = model.accessors[uv_it->second];
        result.UV.resize(n, 2);

        const auto& uv_bv = model.bufferViews[uv_accessor.bufferView];
        const auto& uv_buf = model.buffers[uv_bv.buffer];
        const uint8_t* uv_base = uv_buf.data.data() + uv_bv.byteOffset + uv_accessor.byteOffset;
        size_t uv_stride = uv_bv.byteStride ? uv_bv.byteStride : 2 * sizeof(float);

        for (int i = 0; i < n; ++i) {
            const float* uv = reinterpret_cast<const float*>(uv_base + i * uv_stride);
            result.UV(i, 0) = uv[0];
            result.UV(i, 1) = uv[1];
        }
    }

    return result;
}

} // anonymous namespace

TriMesh load_gltf(const std::string& path) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    bool ok = false;
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".glb") {
        ok = loader.LoadBinaryFromFile(&model, &err, &warn, path);
    } else {
        ok = loader.LoadASCIIFromFile(&model, &err, &warn, path);
    }

    if (!ok) {
        throw std::runtime_error("Failed to load glTF: " + err);
    }

    return parse_model(model);
}

TriMesh load_gltf_from_memory(const std::vector<uint8_t>& data) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    bool ok = loader.LoadBinaryFromMemory(&model, &err, &warn, data.data(),
                                          static_cast<unsigned int>(data.size()));
    if (!ok) {
        throw std::runtime_error("Failed to load glTF from memory: " + err);
    }

    return parse_model(model);
}

void save_gltf(const std::string& path, const TriMesh& mesh) {
    auto data = save_gltf_to_memory(mesh);
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Cannot open file for writing: " + path);
    }
    ofs.write(reinterpret_cast<const char*>(data.data()), data.size());
}

std::vector<uint8_t> save_gltf_to_memory(const TriMesh& mesh) {
    tinygltf::Model model;
    tinygltf::TinyGLTF writer;

    // Write shared-vertex mesh (positions + normals + optional UVs + indices).
    int n = mesh.num_vertices();
    int m = mesh.num_faces();

    // Compute normals if not present
    Eigen::MatrixXd normals = mesh.N;
    if (!mesh.has_normals()) {
        igl::per_vertex_normals(mesh.V, mesh.F, normals);
    }

    std::vector<uint8_t> buffer_data;

    // Positions: n * 3 * float
    size_t pos_offset = 0;
    size_t pos_size = n * 3 * sizeof(float);
    buffer_data.resize(pos_size);
    for (int i = 0; i < n; ++i) {
        float v[3] = {(float)mesh.V(i, 0), (float)mesh.V(i, 1), (float)mesh.V(i, 2)};
        std::memcpy(buffer_data.data() + i * 3 * sizeof(float), v, 3 * sizeof(float));
    }

    // Normals: n * 3 * float
    size_t nrm_offset = buffer_data.size();
    size_t nrm_size = n * 3 * sizeof(float);
    buffer_data.resize(nrm_offset + nrm_size);
    for (int i = 0; i < n; ++i) {
        float nrm[3] = {(float)normals(i, 0), (float)normals(i, 1), (float)normals(i, 2)};
        std::memcpy(buffer_data.data() + nrm_offset + i * 3 * sizeof(float), nrm, 3 * sizeof(float));
    }

    // UVs: n * 2 * float
    size_t uv_offset = buffer_data.size();
    size_t uv_size = 0;
    if (mesh.has_uvs()) {
        uv_size = n * 2 * sizeof(float);
        buffer_data.resize(uv_offset + uv_size);
        for (int i = 0; i < n; ++i) {
            float uv[2] = {(float)mesh.UV(i, 0), (float)mesh.UV(i, 1)};
            std::memcpy(buffer_data.data() + uv_offset + i * 2 * sizeof(float), uv, 2 * sizeof(float));
        }
    }

    // Indices: m * 3 * uint32
    size_t idx_offset = buffer_data.size();
    size_t idx_size = m * 3 * sizeof(uint32_t);
    buffer_data.resize(idx_offset + idx_size);
    for (int i = 0; i < m; ++i) {
        uint32_t tri[3] = {(uint32_t)mesh.F(i, 0), (uint32_t)mesh.F(i, 1), (uint32_t)mesh.F(i, 2)};
        std::memcpy(buffer_data.data() + idx_offset + i * 3 * sizeof(uint32_t), tri, 3 * sizeof(uint32_t));
    }

    // Create glTF structures
    tinygltf::Buffer buf;
    buf.data = buffer_data;
    model.buffers.push_back(buf);

    // BufferView 0: positions
    tinygltf::BufferView pos_bv;
    pos_bv.buffer = 0;
    pos_bv.byteOffset = pos_offset;
    pos_bv.byteLength = pos_size;
    pos_bv.target = TINYGLTF_TARGET_ARRAY_BUFFER;
    model.bufferViews.push_back(pos_bv);

    // BufferView 1: normals
    tinygltf::BufferView nrm_bv;
    nrm_bv.buffer = 0;
    nrm_bv.byteOffset = nrm_offset;
    nrm_bv.byteLength = nrm_size;
    nrm_bv.target = TINYGLTF_TARGET_ARRAY_BUFFER;
    int nrm_bv_idx = static_cast<int>(model.bufferViews.size());
    model.bufferViews.push_back(nrm_bv);

    int uv_bv_idx = -1;
    if (mesh.has_uvs()) {
        tinygltf::BufferView uv_bv;
        uv_bv.buffer = 0;
        uv_bv.byteOffset = uv_offset;
        uv_bv.byteLength = uv_size;
        uv_bv.target = TINYGLTF_TARGET_ARRAY_BUFFER;
        uv_bv_idx = static_cast<int>(model.bufferViews.size());
        model.bufferViews.push_back(uv_bv);
    }

    tinygltf::BufferView idx_bv;
    idx_bv.buffer = 0;
    idx_bv.byteOffset = idx_offset;
    idx_bv.byteLength = idx_size;
    idx_bv.target = TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER;
    int idx_bv_idx = static_cast<int>(model.bufferViews.size());
    model.bufferViews.push_back(idx_bv);

    // Accessor 0: positions
    tinygltf::Accessor pos_acc;
    pos_acc.bufferView = 0;
    pos_acc.byteOffset = 0;
    pos_acc.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
    pos_acc.count = n;
    pos_acc.type = TINYGLTF_TYPE_VEC3;
    pos_acc.minValues = {mesh.V.col(0).minCoeff(), mesh.V.col(1).minCoeff(), mesh.V.col(2).minCoeff()};
    pos_acc.maxValues = {mesh.V.col(0).maxCoeff(), mesh.V.col(1).maxCoeff(), mesh.V.col(2).maxCoeff()};
    model.accessors.push_back(pos_acc);

    // Accessor 1: normals
    tinygltf::Accessor nrm_acc;
    nrm_acc.bufferView = nrm_bv_idx;
    nrm_acc.byteOffset = 0;
    nrm_acc.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
    nrm_acc.count = n;
    nrm_acc.type = TINYGLTF_TYPE_VEC3;
    int nrm_acc_idx = static_cast<int>(model.accessors.size());
    model.accessors.push_back(nrm_acc);

    int uv_acc_idx = -1;
    if (mesh.has_uvs()) {
        tinygltf::Accessor uv_acc;
        uv_acc.bufferView = uv_bv_idx;
        uv_acc.byteOffset = 0;
        uv_acc.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        uv_acc.count = n;
        uv_acc.type = TINYGLTF_TYPE_VEC2;
        uv_acc_idx = static_cast<int>(model.accessors.size());
        model.accessors.push_back(uv_acc);
    }

    tinygltf::Accessor idx_acc;
    idx_acc.bufferView = idx_bv_idx;
    idx_acc.byteOffset = 0;
    idx_acc.componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;
    idx_acc.count = m * 3;
    idx_acc.type = TINYGLTF_TYPE_SCALAR;
    int idx_acc_idx = static_cast<int>(model.accessors.size());
    model.accessors.push_back(idx_acc);

    // Primitive
    tinygltf::Primitive prim;
    prim.attributes["POSITION"] = 0;
    prim.attributes["NORMAL"] = nrm_acc_idx;
    if (uv_acc_idx >= 0) {
        prim.attributes["TEXCOORD_0"] = uv_acc_idx;
    }
    prim.indices = idx_acc_idx;
    prim.mode = TINYGLTF_MODE_TRIANGLES;

    // Mesh
    tinygltf::Mesh gltf_mesh;
    gltf_mesh.name = "parameterized_mesh";
    gltf_mesh.primitives.push_back(prim);
    model.meshes.push_back(gltf_mesh);

    // Node
    tinygltf::Node node;
    node.mesh = 0;
    model.nodes.push_back(node);

    // Scene
    tinygltf::Scene scene;
    scene.nodes.push_back(0);
    model.scenes.push_back(scene);
    model.defaultScene = 0;

    // Asset
    model.asset.version = "2.0";
    model.asset.generator = "MeshParameterization";

    // Write to binary glb
    std::ostringstream oss;
    writer.WriteGltfSceneToStream(&model, oss, false, true); // binary
    std::string str = oss.str();

    return std::vector<uint8_t>(str.begin(), str.end());
}

} // namespace meshparam
