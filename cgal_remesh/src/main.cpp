// CGAL Isotropic Remeshing CLI: GLB → remeshed GLB
// Uses CGAL::Polygon_mesh_processing::isotropic_remeshing for uniform triangles.

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <tiny_gltf.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Polygon_mesh_processing/detect_features.h>

#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <string>
#include <sstream>
#include <fstream>
#include <chrono>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Mesh = CGAL::Surface_mesh<K::Point_3>;
namespace PMP = CGAL::Polygon_mesh_processing;

// ============================================================
// GLB I/O (minimal, shared-vertex)
// ============================================================
struct TriMesh {
    std::vector<float> V; // x,y,z interleaved
    std::vector<uint32_t> F; // i,j,k interleaved
    int nv() const { return V.size() / 3; }
    int nf() const { return F.size() / 3; }
};

TriMesh load_glb(const std::string& path) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;
    bool ok = loader.LoadBinaryFromFile(&model, &err, &warn, path);
    if (!ok) throw std::runtime_error("Failed to load: " + err);

    auto& prim = model.meshes[0].primitives[0];
    auto& pos_acc = model.accessors[prim.attributes.at("POSITION")];
    auto& pos_bv = model.bufferViews[pos_acc.bufferView];
    auto& buf = model.buffers[pos_bv.buffer];

    TriMesh m;
    int n = pos_acc.count;
    m.V.resize(n * 3);
    const float* pdata = reinterpret_cast<const float*>(
        buf.data.data() + pos_bv.byteOffset + pos_acc.byteOffset);
    std::memcpy(m.V.data(), pdata, n * 3 * sizeof(float));

    auto& idx_acc = model.accessors[prim.indices];
    auto& idx_bv = model.bufferViews[idx_acc.bufferView];
    int nf = idx_acc.count / 3;
    m.F.resize(nf * 3);
    const uint8_t* idata = buf.data.data() + idx_bv.byteOffset + idx_acc.byteOffset;
    for (int i = 0; i < nf * 3; ++i) {
        if (idx_acc.componentType == 5125)
            m.F[i] = *reinterpret_cast<const uint32_t*>(idata + i * 4);
        else if (idx_acc.componentType == 5123)
            m.F[i] = *reinterpret_cast<const uint16_t*>(idata + i * 2);
        else
            m.F[i] = idata[i];
    }
    return m;
}

void save_glb(const std::string& path, const TriMesh& m) {
    int nv = m.nv(), nf = m.nf();

    // Compute normals
    std::vector<float> N(nv * 3, 0.0f);
    for (int fi = 0; fi < nf; ++fi) {
        uint32_t i0 = m.F[fi*3], i1 = m.F[fi*3+1], i2 = m.F[fi*3+2];
        float e1[3] = {m.V[i1*3]-m.V[i0*3], m.V[i1*3+1]-m.V[i0*3+1], m.V[i1*3+2]-m.V[i0*3+2]};
        float e2[3] = {m.V[i2*3]-m.V[i0*3], m.V[i2*3+1]-m.V[i0*3+1], m.V[i2*3+2]-m.V[i0*3+2]};
        float fn[3] = {e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]};
        for (int k = 0; k < 3; ++k) { N[i0*3+k] += fn[k]; N[i1*3+k] += fn[k]; N[i2*3+k] += fn[k]; }
    }
    for (int i = 0; i < nv; ++i) {
        float len = std::sqrt(N[i*3]*N[i*3] + N[i*3+1]*N[i*3+1] + N[i*3+2]*N[i*3+2]);
        if (len > 1e-15f) { N[i*3] /= len; N[i*3+1] /= len; N[i*3+2] /= len; }
    }

    size_t pos_size = nv * 3 * 4;
    size_t nrm_size = nv * 3 * 4;
    size_t idx_size = nf * 3 * 4;

    std::vector<uint8_t> buf(pos_size + nrm_size + idx_size);
    std::memcpy(buf.data(), m.V.data(), pos_size);
    std::memcpy(buf.data() + pos_size, N.data(), nrm_size);
    std::memcpy(buf.data() + pos_size + nrm_size, m.F.data(), idx_size);

    while (buf.size() % 4) buf.push_back(0);

    float mn[3] = {1e30f,1e30f,1e30f}, mx[3] = {-1e30f,-1e30f,-1e30f};
    for (int i = 0; i < nv; ++i)
        for (int k = 0; k < 3; ++k) {
            mn[k] = std::min(mn[k], m.V[i*3+k]);
            mx[k] = std::max(mx[k], m.V[i*3+k]);
        }

    tinygltf::Model model;
    tinygltf::Buffer b; b.data.assign(buf.begin(), buf.end()); model.buffers.push_back(b);
    tinygltf::BufferView bv0; bv0.buffer=0; bv0.byteOffset=0; bv0.byteLength=pos_size; bv0.target=34962; model.bufferViews.push_back(bv0);
    tinygltf::BufferView bv1; bv1.buffer=0; bv1.byteOffset=pos_size; bv1.byteLength=nrm_size; bv1.target=34962; model.bufferViews.push_back(bv1);
    tinygltf::BufferView bv2; bv2.buffer=0; bv2.byteOffset=pos_size+nrm_size; bv2.byteLength=idx_size; bv2.target=34963; model.bufferViews.push_back(bv2);

    tinygltf::Accessor a0; a0.bufferView=0; a0.componentType=5126; a0.count=nv; a0.type=TINYGLTF_TYPE_VEC3;
    a0.minValues={mn[0],mn[1],mn[2]}; a0.maxValues={mx[0],mx[1],mx[2]}; model.accessors.push_back(a0);
    tinygltf::Accessor a1; a1.bufferView=1; a1.componentType=5126; a1.count=nv; a1.type=TINYGLTF_TYPE_VEC3; model.accessors.push_back(a1);
    tinygltf::Accessor a2; a2.bufferView=2; a2.componentType=5125; a2.count=nf*3; a2.type=TINYGLTF_TYPE_SCALAR; model.accessors.push_back(a2);

    tinygltf::Primitive prim; prim.attributes["POSITION"]=0; prim.attributes["NORMAL"]=1; prim.indices=2; prim.mode=4;
    tinygltf::Mesh gm; gm.primitives.push_back(prim); model.meshes.push_back(gm);
    tinygltf::Node node; node.mesh=0; model.nodes.push_back(node);
    tinygltf::Scene sc; sc.nodes.push_back(0); model.scenes.push_back(sc);
    model.defaultScene=0; model.asset.version="2.0"; model.asset.generator="cgal_remesh_cli";

    tinygltf::TinyGLTF writer;
    std::ostringstream oss;
    writer.WriteGltfSceneToStream(&model, oss, false, true);
    std::string data = oss.str();
    std::ofstream f(path, std::ios::binary);
    f.write(data.data(), data.size());
}

// ============================================================
// Main
// ============================================================
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: cgal_remesh_cli input.glb output.glb [--target-edge-length L] [--iterations N]" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    double target_edge = 0; // 0 = auto
    int iterations = 3;

    for (int i = 3; i < argc; ++i) {
        if (std::string(argv[i]) == "--target-edge-length" && i+1 < argc) target_edge = std::stod(argv[++i]);
        if (std::string(argv[i]) == "--iterations" && i+1 < argc) iterations = std::stoi(argv[++i]);
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    auto tri = load_glb(input_path);
    std::cout << "Input: " << tri.nv() << " verts, " << tri.nf() << " tris" << std::endl;

    // Build CGAL mesh
    Mesh mesh;
    std::vector<Mesh::Vertex_index> vds(tri.nv());
    for (int i = 0; i < tri.nv(); ++i)
        vds[i] = mesh.add_vertex(K::Point_3(tri.V[i*3], tri.V[i*3+1], tri.V[i*3+2]));
    for (int i = 0; i < tri.nf(); ++i)
        mesh.add_face(vds[tri.F[i*3]], vds[tri.F[i*3+1]], vds[tri.F[i*3+2]]);

    // Auto edge length: average edge length
    if (target_edge <= 0) {
        double sum = 0; int cnt = 0;
        for (auto e : mesh.edges()) {
            auto h = mesh.halfedge(e, 0);
            auto p1 = mesh.point(mesh.source(h));
            auto p2 = mesh.point(mesh.target(h));
            sum += std::sqrt(CGAL::squared_distance(p1, p2));
            cnt++;
        }
        target_edge = sum / cnt;
    }

    std::cout << "Remeshing (edge=" << target_edge << ", iters=" << iterations << ")..." << std::endl;

    // Detect and constrain sharp edges (dihedral angle > 60 degrees)
    typedef boost::property_map<Mesh, CGAL::edge_is_feature_t>::type EIFMap;
    EIFMap eif = get(CGAL::edge_is_feature, mesh);
    PMP::detect_sharp_edges(mesh, 60.0, eif);
    int n_sharp = 0;
    for (auto e : mesh.edges()) if (get(eif, e)) n_sharp++;
    std::cout << "Sharp edges: " << n_sharp << std::endl;

    // Isotropic remeshing — protect sharp features to preserve shape
    PMP::isotropic_remeshing(mesh.faces(), target_edge, mesh,
        CGAL::parameters::number_of_iterations(iterations)
        .protect_constraints(true)
        .edge_is_constrained_map(eif));

    // Extract result
    TriMesh out;
    int new_nv = mesh.number_of_vertices();
    int new_nf = mesh.number_of_faces();
    out.V.resize(new_nv * 3);
    out.F.resize(new_nf * 3);

    // Vertex index remap (CGAL may have gaps after remeshing)
    std::map<Mesh::Vertex_index, int> vi_map;
    int idx = 0;
    for (auto v : mesh.vertices()) {
        auto p = mesh.point(v);
        out.V[idx*3] = p.x(); out.V[idx*3+1] = p.y(); out.V[idx*3+2] = p.z();
        vi_map[v] = idx++;
    }

    idx = 0;
    for (auto f : mesh.faces()) {
        for (auto v : mesh.vertices_around_face(mesh.halfedge(f)))
            out.F[idx++] = vi_map[v];
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Output: " << new_nv << " verts, " << new_nf << " tris (" << ms << " ms)" << std::endl;

    save_glb(output_path, out);
    std::cout << "Wrote: " << output_path << std::endl;
    return 0;
}
