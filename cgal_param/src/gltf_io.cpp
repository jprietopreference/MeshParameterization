#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "cgalparam/gltf_io.h"
#include <tiny_gltf.h>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <cstring>

namespace cgalparam {

namespace {

TriMesh parse_model(const tinygltf::Model& model) {
    if (model.meshes.empty())
        throw std::runtime_error("glTF contains no meshes");
    const auto& prim = model.meshes[0].primitives[0];

    TriMesh result;

    // Positions
    auto pos_it = prim.attributes.find("POSITION");
    if (pos_it == prim.attributes.end())
        throw std::runtime_error("No POSITION attribute");
    const auto& pos_acc = model.accessors[pos_it->second];
    int n = static_cast<int>(pos_acc.count);
    result.V.resize(n, 3);

    const auto& pos_bv = model.bufferViews[pos_acc.bufferView];
    const auto& pos_buf = model.buffers[pos_bv.buffer];
    const uint8_t* pos_base = pos_buf.data.data() + pos_bv.byteOffset + pos_acc.byteOffset;
    size_t pos_stride = pos_bv.byteStride ? pos_bv.byteStride : 3 * sizeof(float);

    for (int i = 0; i < n; ++i) {
        const float* p = reinterpret_cast<const float*>(pos_base + i * pos_stride);
        result.V(i, 0) = p[0]; result.V(i, 1) = p[1]; result.V(i, 2) = p[2];
    }

    // Indices
    if (prim.indices >= 0) {
        const auto& idx_acc = model.accessors[prim.indices];
        int m = static_cast<int>(idx_acc.count) / 3;
        result.F.resize(m, 3);
        const auto& idx_bv = model.bufferViews[idx_acc.bufferView];
        const auto& idx_buf = model.buffers[idx_bv.buffer];
        const uint8_t* idx_base = idx_buf.data.data() + idx_bv.byteOffset + idx_acc.byteOffset;

        for (int i = 0; i < m * 3; ++i) {
            int idx = 0;
            switch (idx_acc.componentType) {
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
                    uint16_t val; std::memcpy(&val, idx_base + i * 2, 2); idx = val; break;
                }
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
                    uint32_t val; std::memcpy(&val, idx_base + i * 4, 4); idx = val; break;
                }
                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: idx = idx_base[i]; break;
                default: throw std::runtime_error("Unsupported index type");
            }
            result.F(i / 3, i % 3) = idx;
        }
    }

    // UVs
    auto uv_it = prim.attributes.find("TEXCOORD_0");
    if (uv_it != prim.attributes.end()) {
        const auto& uv_acc = model.accessors[uv_it->second];
        result.UV.resize(n, 2);
        const auto& uv_bv = model.bufferViews[uv_acc.bufferView];
        const auto& uv_buf = model.buffers[uv_bv.buffer];
        const uint8_t* uv_base = uv_buf.data.data() + uv_bv.byteOffset + uv_acc.byteOffset;
        size_t uv_stride = uv_bv.byteStride ? uv_bv.byteStride : 2 * sizeof(float);
        for (int i = 0; i < n; ++i) {
            const float* uv = reinterpret_cast<const float*>(uv_base + i * uv_stride);
            result.UV(i, 0) = uv[0]; result.UV(i, 1) = uv[1];
        }
    }

    return result;
}

} // anonymous namespace

TriMesh load_gltf(const std::string& path) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;
    bool ok = (path.size() >= 4 && path.substr(path.size() - 4) == ".glb")
        ? loader.LoadBinaryFromFile(&model, &err, &warn, path)
        : loader.LoadASCIIFromFile(&model, &err, &warn, path);
    if (!ok) throw std::runtime_error("Failed to load glTF: " + err);
    return parse_model(model);
}

void save_gltf(const std::string& path, const TriMesh& mesh) {
    tinygltf::Model model;
    tinygltf::TinyGLTF writer;
    int n = mesh.num_vertices();
    int m = mesh.num_faces();

    std::vector<uint8_t> buffer_data;

    // Positions
    size_t pos_size = n * 3 * sizeof(float);
    buffer_data.resize(pos_size);
    for (int i = 0; i < n; ++i) {
        float v[3] = {(float)mesh.V(i,0), (float)mesh.V(i,1), (float)mesh.V(i,2)};
        std::memcpy(buffer_data.data() + i * 12, v, 12);
    }

    // UVs
    size_t uv_offset = buffer_data.size();
    size_t uv_size = 0;
    if (mesh.has_uvs()) {
        uv_size = n * 2 * sizeof(float);
        buffer_data.resize(uv_offset + uv_size);
        for (int i = 0; i < n; ++i) {
            float uv[2] = {(float)mesh.UV(i,0), (float)mesh.UV(i,1)};
            std::memcpy(buffer_data.data() + uv_offset + i * 8, uv, 8);
        }
    }

    // Indices
    size_t idx_offset = buffer_data.size();
    size_t idx_size = m * 3 * sizeof(uint32_t);
    buffer_data.resize(idx_offset + idx_size);
    for (int i = 0; i < m; ++i) {
        uint32_t tri[3] = {(uint32_t)mesh.F(i,0), (uint32_t)mesh.F(i,1), (uint32_t)mesh.F(i,2)};
        std::memcpy(buffer_data.data() + idx_offset + i * 12, tri, 12);
    }

    tinygltf::Buffer buf; buf.data = buffer_data;
    model.buffers.push_back(buf);

    // BufferViews
    tinygltf::BufferView pos_bv; pos_bv.buffer = 0; pos_bv.byteOffset = 0;
    pos_bv.byteLength = pos_size; pos_bv.target = TINYGLTF_TARGET_ARRAY_BUFFER;
    model.bufferViews.push_back(pos_bv);

    int uv_bv_idx = -1;
    if (mesh.has_uvs()) {
        tinygltf::BufferView uv_bv; uv_bv.buffer = 0; uv_bv.byteOffset = uv_offset;
        uv_bv.byteLength = uv_size; uv_bv.target = TINYGLTF_TARGET_ARRAY_BUFFER;
        uv_bv_idx = static_cast<int>(model.bufferViews.size());
        model.bufferViews.push_back(uv_bv);
    }

    tinygltf::BufferView idx_bv; idx_bv.buffer = 0; idx_bv.byteOffset = idx_offset;
    idx_bv.byteLength = idx_size; idx_bv.target = TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER;
    int idx_bv_idx = static_cast<int>(model.bufferViews.size());
    model.bufferViews.push_back(idx_bv);

    // Accessors
    tinygltf::Accessor pos_acc; pos_acc.bufferView = 0; pos_acc.componentType = 5126;
    pos_acc.count = n; pos_acc.type = TINYGLTF_TYPE_VEC3;
    pos_acc.minValues = {mesh.V.col(0).minCoeff(), mesh.V.col(1).minCoeff(), mesh.V.col(2).minCoeff()};
    pos_acc.maxValues = {mesh.V.col(0).maxCoeff(), mesh.V.col(1).maxCoeff(), mesh.V.col(2).maxCoeff()};
    model.accessors.push_back(pos_acc);

    int uv_acc_idx = -1;
    if (mesh.has_uvs()) {
        tinygltf::Accessor uv_acc; uv_acc.bufferView = uv_bv_idx; uv_acc.componentType = 5126;
        uv_acc.count = n; uv_acc.type = TINYGLTF_TYPE_VEC2;
        uv_acc_idx = static_cast<int>(model.accessors.size());
        model.accessors.push_back(uv_acc);
    }

    tinygltf::Accessor idx_acc; idx_acc.bufferView = idx_bv_idx;
    idx_acc.componentType = 5125; idx_acc.count = m * 3; idx_acc.type = TINYGLTF_TYPE_SCALAR;
    int idx_acc_idx = static_cast<int>(model.accessors.size());
    model.accessors.push_back(idx_acc);

    tinygltf::Primitive prim; prim.attributes["POSITION"] = 0;
    if (uv_acc_idx >= 0) prim.attributes["TEXCOORD_0"] = uv_acc_idx;
    prim.indices = idx_acc_idx; prim.mode = TINYGLTF_MODE_TRIANGLES;

    tinygltf::Mesh gm; gm.name = "parameterized"; gm.primitives.push_back(prim);
    model.meshes.push_back(gm);
    tinygltf::Node node; node.mesh = 0; model.nodes.push_back(node);
    tinygltf::Scene scene; scene.nodes.push_back(0);
    model.scenes.push_back(scene); model.defaultScene = 0;
    model.asset.version = "2.0"; model.asset.generator = "CGALParameterization";

    std::ostringstream oss;
    writer.WriteGltfSceneToStream(&model, oss, false, true);
    std::string str = oss.str();

    std::ofstream ofs(path, std::ios::binary);
    ofs.write(str.data(), str.size());
}

TriMesh load_gltf_from_memory(const std::vector<uint8_t>& data) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;
    bool ok = loader.LoadBinaryFromMemory(&model, &err, &warn,
                                          data.data(), static_cast<unsigned int>(data.size()));
    if (!ok) throw std::runtime_error("Failed to load glTF from memory: " + err);
    return parse_model(model);
}

std::vector<uint8_t> save_gltf_to_memory(const TriMesh& mesh) {
    tinygltf::Model model;
    tinygltf::TinyGLTF writer;
    int n = mesh.num_vertices();
    int m = mesh.num_faces();

    std::vector<uint8_t> buffer_data;

    size_t pos_size = n * 3 * sizeof(float);
    buffer_data.resize(pos_size);
    for (int i = 0; i < n; ++i) {
        float v[3] = {(float)mesh.V(i,0), (float)mesh.V(i,1), (float)mesh.V(i,2)};
        std::memcpy(buffer_data.data() + i * 12, v, 12);
    }

    size_t uv_offset = buffer_data.size();
    size_t uv_size = 0;
    if (mesh.has_uvs()) {
        uv_size = n * 2 * sizeof(float);
        buffer_data.resize(uv_offset + uv_size);
        for (int i = 0; i < n; ++i) {
            float uv[2] = {(float)mesh.UV(i,0), (float)mesh.UV(i,1)};
            std::memcpy(buffer_data.data() + uv_offset + i * 8, uv, 8);
        }
    }

    size_t idx_offset = buffer_data.size();
    size_t idx_size = m * 3 * sizeof(uint32_t);
    buffer_data.resize(idx_offset + idx_size);
    for (int i = 0; i < m; ++i) {
        uint32_t tri[3] = {(uint32_t)mesh.F(i,0), (uint32_t)mesh.F(i,1), (uint32_t)mesh.F(i,2)};
        std::memcpy(buffer_data.data() + idx_offset + i * 12, tri, 12);
    }

    tinygltf::Buffer buf; buf.data = buffer_data;
    model.buffers.push_back(buf);

    tinygltf::BufferView pos_bv; pos_bv.buffer = 0; pos_bv.byteOffset = 0;
    pos_bv.byteLength = pos_size; pos_bv.target = TINYGLTF_TARGET_ARRAY_BUFFER;
    model.bufferViews.push_back(pos_bv);

    int uv_bv_idx = -1;
    if (mesh.has_uvs()) {
        tinygltf::BufferView uv_bv; uv_bv.buffer = 0; uv_bv.byteOffset = uv_offset;
        uv_bv.byteLength = uv_size; uv_bv.target = TINYGLTF_TARGET_ARRAY_BUFFER;
        uv_bv_idx = static_cast<int>(model.bufferViews.size());
        model.bufferViews.push_back(uv_bv);
    }

    tinygltf::BufferView idx_bv; idx_bv.buffer = 0; idx_bv.byteOffset = idx_offset;
    idx_bv.byteLength = idx_size; idx_bv.target = TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER;
    int idx_bv_idx = static_cast<int>(model.bufferViews.size());
    model.bufferViews.push_back(idx_bv);

    tinygltf::Accessor pos_acc; pos_acc.bufferView = 0; pos_acc.componentType = 5126;
    pos_acc.count = n; pos_acc.type = TINYGLTF_TYPE_VEC3;
    pos_acc.minValues = {mesh.V.col(0).minCoeff(), mesh.V.col(1).minCoeff(), mesh.V.col(2).minCoeff()};
    pos_acc.maxValues = {mesh.V.col(0).maxCoeff(), mesh.V.col(1).maxCoeff(), mesh.V.col(2).maxCoeff()};
    model.accessors.push_back(pos_acc);

    int uv_acc_idx = -1;
    if (mesh.has_uvs()) {
        tinygltf::Accessor uv_acc; uv_acc.bufferView = uv_bv_idx; uv_acc.componentType = 5126;
        uv_acc.count = n; uv_acc.type = TINYGLTF_TYPE_VEC2;
        uv_acc_idx = static_cast<int>(model.accessors.size());
        model.accessors.push_back(uv_acc);
    }

    tinygltf::Accessor idx_acc; idx_acc.bufferView = idx_bv_idx;
    idx_acc.componentType = 5125; idx_acc.count = m * 3; idx_acc.type = TINYGLTF_TYPE_SCALAR;
    int idx_acc_idx = static_cast<int>(model.accessors.size());
    model.accessors.push_back(idx_acc);

    tinygltf::Primitive prim; prim.attributes["POSITION"] = 0;
    if (uv_acc_idx >= 0) prim.attributes["TEXCOORD_0"] = uv_acc_idx;
    prim.indices = idx_acc_idx; prim.mode = TINYGLTF_MODE_TRIANGLES;

    tinygltf::Mesh gm; gm.name = "parameterized"; gm.primitives.push_back(prim);
    model.meshes.push_back(gm);
    tinygltf::Node node; node.mesh = 0; model.nodes.push_back(node);
    tinygltf::Scene scene; scene.nodes.push_back(0);
    model.scenes.push_back(scene); model.defaultScene = 0;
    model.asset.version = "2.0"; model.asset.generator = "CGALParameterization";

    std::ostringstream oss;
    writer.WriteGltfSceneToStream(&model, oss, false, true);
    std::string str = oss.str();
    return std::vector<uint8_t>(str.begin(), str.end());
}

SurfaceMesh to_cgal_mesh(const TriMesh& mesh) {
    SurfaceMesh sm;
    int n = mesh.num_vertices();
    int m = mesh.num_faces();

    std::vector<vertex_descriptor> vds(n);
    for (int i = 0; i < n; ++i) {
        vds[i] = sm.add_vertex(Kernel::Point_3(mesh.V(i,0), mesh.V(i,1), mesh.V(i,2)));
    }

    for (int i = 0; i < m; ++i) {
        sm.add_face(vds[mesh.F(i,0)], vds[mesh.F(i,1)], vds[mesh.F(i,2)]);
    }

    return sm;
}

TriMesh from_cgal_mesh(const SurfaceMesh& sm) {
    TriMesh mesh;
    int n = static_cast<int>(sm.number_of_vertices());
    int m = static_cast<int>(sm.number_of_faces());

    mesh.V.resize(n, 3);
    for (auto v : sm.vertices()) {
        auto& pt = sm.point(v);
        int i = static_cast<int>(v);
        mesh.V(i, 0) = pt.x(); mesh.V(i, 1) = pt.y(); mesh.V(i, 2) = pt.z();
    }

    mesh.F.resize(m, 3);
    for (auto f : sm.faces()) {
        int fi = static_cast<int>(f);
        int k = 0;
        for (auto v : sm.vertices_around_face(sm.halfedge(f))) {
            mesh.F(fi, k++) = static_cast<int>(v);
        }
    }

    return mesh;
}

} // namespace cgalparam
