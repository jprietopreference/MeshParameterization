// OCC Tessellator CLI: STEP → GLB
// Reads a STEP file, tessellates the B-Rep shape, outputs a shared-vertex GLB.

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <tiny_gltf.h>

#include <STEPControl_Reader.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRep_Tool.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Face.hxx>
#include <TopExp_Explorer.hxx>
#include <TopLoc_Location.hxx>
#include <Poly_Triangulation.hxx>
#include <TopAbs_Orientation.hxx>
#include <gp_Pnt.hxx>
#include <gp_Trsf.hxx>

#include <iostream>
#include <vector>
#include <map>
#include <array>
#include <cstring>
#include <cmath>
#include <string>
#include <sstream>

struct Vec3 {
    float x, y, z;
    bool operator<(const Vec3& o) const {
        if (x != o.x) return x < o.x;
        if (y != o.y) return y < o.y;
        return z < o.z;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: occ_tessellate_cli <input.step> <output.glb> [--deflection <mm>] [--scale <factor>]" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    double deflection = 1.0;
    double scale = 1.0;

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--deflection" && i + 1 < argc) deflection = std::stod(argv[++i]);
        if (arg == "--scale" && i + 1 < argc) scale = std::stod(argv[++i]);
    }

    // Read STEP
    std::cout << "Reading: " << input_path << std::endl;
    STEPControl_Reader reader;
    IFSelect_ReturnStatus status = reader.ReadFile(input_path.c_str());
    if (status != IFSelect_RetDone) {
        std::cerr << "Error: STEP read failed (status " << status << ")" << std::endl;
        return 1;
    }

    reader.TransferRoots();
    TopoDS_Shape shape = reader.OneShape();
    std::cout << "STEP loaded." << std::endl;

    // Tessellate
    BRepMesh_IncrementalMesh mesh(shape, deflection, Standard_False, 0.5, Standard_True);
    mesh.Perform();
    std::cout << "Tessellated (deflection=" << deflection << "mm)." << std::endl;

    // Extract triangulation — shared vertex with welding
    std::vector<float> all_verts;
    std::vector<uint32_t> all_tris;
    std::map<Vec3, uint32_t> vert_map;

    auto get_or_add_vertex = [&](float x, float y, float z) -> uint32_t {
        // Quantize to 1e-6 for welding
        Vec3 key{
            std::round(x * 1e6f) * 1e-6f,
            std::round(y * 1e6f) * 1e-6f,
            std::round(z * 1e6f) * 1e-6f
        };
        auto it = vert_map.find(key);
        if (it != vert_map.end()) return it->second;
        uint32_t idx = static_cast<uint32_t>(all_verts.size() / 3);
        all_verts.push_back(x);
        all_verts.push_back(y);
        all_verts.push_back(z);
        vert_map[key] = idx;
        return idx;
    };

    for (TopExp_Explorer exp(shape, TopAbs_FACE); exp.More(); exp.Next()) {
        TopoDS_Face face = TopoDS::Face(exp.Current());
        TopLoc_Location loc;
        Handle(Poly_Triangulation) tri = BRep_Tool::Triangulation(face, loc);
        if (tri.IsNull()) continue;

        gp_Trsf trsf = loc.Transformation();
        int nNodes = tri->NbNodes();
        int nTris = tri->NbTriangles();
        bool reversed = (face.Orientation() == TopAbs_REVERSED);

        // Map local node indices to global vertex indices
        std::vector<uint32_t> local_to_global(nNodes + 1);
        for (int i = 1; i <= nNodes; ++i) {
            gp_Pnt pt = tri->Node(i).Transformed(trsf);
            float x = static_cast<float>(pt.X() * scale);
            float y = static_cast<float>(pt.Y() * scale);
            float z = static_cast<float>(pt.Z() * scale);
            local_to_global[i] = get_or_add_vertex(x, y, z);
        }

        for (int i = 1; i <= nTris; ++i) {
            int n1, n2, n3;
            tri->Triangle(i).Get(n1, n2, n3);
            if (reversed) std::swap(n2, n3);
            all_tris.push_back(local_to_global[n1]);
            all_tris.push_back(local_to_global[n2]);
            all_tris.push_back(local_to_global[n3]);
        }
    }

    int nv = static_cast<int>(all_verts.size() / 3);
    int nf = static_cast<int>(all_tris.size() / 3);
    std::cout << "Mesh: " << nv << " vertices, " << nf << " triangles" << std::endl;

    if (nv == 0) {
        std::cerr << "Error: No triangulation produced" << std::endl;
        return 1;
    }

    // Write GLB via tinygltf
    tinygltf::Model model;
    std::vector<uint8_t> buffer_data;

    // Positions
    size_t pos_size = nv * 3 * sizeof(float);
    buffer_data.resize(pos_size);
    std::memcpy(buffer_data.data(), all_verts.data(), pos_size);

    // Indices
    size_t idx_offset = buffer_data.size();
    size_t idx_size = nf * 3 * sizeof(uint32_t);
    buffer_data.resize(idx_offset + idx_size);
    std::memcpy(buffer_data.data() + idx_offset, all_tris.data(), idx_size);

    tinygltf::Buffer buf;
    buf.data = buffer_data;
    model.buffers.push_back(buf);

    tinygltf::BufferView pos_bv;
    pos_bv.buffer = 0; pos_bv.byteOffset = 0;
    pos_bv.byteLength = pos_size; pos_bv.target = TINYGLTF_TARGET_ARRAY_BUFFER;
    model.bufferViews.push_back(pos_bv);

    tinygltf::BufferView idx_bv;
    idx_bv.buffer = 0; idx_bv.byteOffset = idx_offset;
    idx_bv.byteLength = idx_size; idx_bv.target = TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER;
    model.bufferViews.push_back(idx_bv);

    // Compute min/max
    float mn[3] = {1e30f, 1e30f, 1e30f}, mx[3] = {-1e30f, -1e30f, -1e30f};
    for (int i = 0; i < nv; ++i) {
        for (int k = 0; k < 3; ++k) {
            mn[k] = std::min(mn[k], all_verts[i * 3 + k]);
            mx[k] = std::max(mx[k], all_verts[i * 3 + k]);
        }
    }

    tinygltf::Accessor pos_acc;
    pos_acc.bufferView = 0; pos_acc.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
    pos_acc.count = nv; pos_acc.type = TINYGLTF_TYPE_VEC3;
    pos_acc.minValues = {mn[0], mn[1], mn[2]};
    pos_acc.maxValues = {mx[0], mx[1], mx[2]};
    model.accessors.push_back(pos_acc);

    tinygltf::Accessor idx_acc;
    idx_acc.bufferView = 1; idx_acc.componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;
    idx_acc.count = nf * 3; idx_acc.type = TINYGLTF_TYPE_SCALAR;
    model.accessors.push_back(idx_acc);

    tinygltf::Primitive prim;
    prim.attributes["POSITION"] = 0;
    prim.indices = 1;
    prim.mode = TINYGLTF_MODE_TRIANGLES;

    tinygltf::Mesh gm;
    gm.primitives.push_back(prim);
    model.meshes.push_back(gm);

    tinygltf::Node node;
    node.mesh = 0;
    model.nodes.push_back(node);

    tinygltf::Scene scene;
    scene.nodes.push_back(0);
    model.scenes.push_back(scene);
    model.defaultScene = 0;
    model.asset.version = "2.0";
    model.asset.generator = "occ_tessellate_cli";

    tinygltf::TinyGLTF writer;
    std::ostringstream oss;
    writer.WriteGltfSceneToStream(&model, oss, false, true); // binary=true
    std::string glb_data = oss.str();

    std::ofstream ofs(output_path, std::ios::binary);
    ofs.write(glb_data.data(), glb_data.size());
    ofs.close();

    std::cout << "Wrote: " << output_path << " (" << glb_data.size() << " bytes)" << std::endl;
    std::cout << "Extent: "
              << (mx[0] - mn[0]) << " x "
              << (mx[1] - mn[1]) << " x "
              << (mx[2] - mn[2]) << std::endl;
    return 0;
}
