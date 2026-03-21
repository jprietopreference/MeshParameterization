// OCC Tessellator CLI: STEP → GLB
// Reads a STEP file, tessellates the B-Rep shape, outputs a shared-vertex GLB.

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <tiny_gltf.h>

#include <STEPControl_Reader.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRepLib_ToolTriangulatedShape.hxx>
#include <BRep_Tool.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Face.hxx>
#include <TopExp_Explorer.hxx>
#include <TopLoc_Location.hxx>
#include <Poly_Triangulation.hxx>
#include <TopAbs_Orientation.hxx>
#include <gp_Pnt.hxx>
#include <gp_Dir.hxx>
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

    // Extract triangulation — split vertices per OCC face for correct normals.
    // Each OCC face gets its own vertices so normals are sharp at face boundaries
    // and smooth within each face (analytical surface normals from B-Rep).
    std::vector<float> all_verts;
    std::vector<float> all_normals;
    std::vector<float> all_face_ids; // per-vertex OCC face ID (stored as float for glTF compat)
    std::vector<uint32_t> all_tris;
    int occ_face_idx = 0;

    for (TopExp_Explorer exp(shape, TopAbs_FACE); exp.More(); exp.Next()) {
        TopoDS_Face face = TopoDS::Face(exp.Current());
        TopLoc_Location loc;
        Handle(Poly_Triangulation) tri = BRep_Tool::Triangulation(face, loc);
        if (tri.IsNull()) continue;

        // Compute analytical surface normals from the B-Rep face
        BRepLib_ToolTriangulatedShape::ComputeNormals(face, tri);

        gp_Trsf trsf = loc.Transformation();
        int nNodes = tri->NbNodes();
        int nTris = tri->NbTriangles();
        bool reversed = (face.Orientation() == TopAbs_REVERSED);

        // Each face gets its own vertex range
        uint32_t base = static_cast<uint32_t>(all_verts.size() / 3);

        for (int i = 1; i <= nNodes; ++i) {
            gp_Pnt pt = tri->Node(i).Transformed(trsf);
            all_verts.push_back(static_cast<float>(pt.X() * scale));
            all_verts.push_back(static_cast<float>(pt.Y() * scale));
            all_verts.push_back(static_cast<float>(pt.Z() * scale));

            all_face_ids.push_back(static_cast<float>(occ_face_idx));

            if (tri->HasNormals()) {
                gp_Dir nrm = tri->Normal(i);
                // Transform normal by rotation part of location
                nrm.Transform(trsf);
                // Flip for reversed faces
                float sign = reversed ? -1.0f : 1.0f;
                all_normals.push_back(static_cast<float>(nrm.X()) * sign);
                all_normals.push_back(static_cast<float>(nrm.Y()) * sign);
                all_normals.push_back(static_cast<float>(nrm.Z()) * sign);
            } else {
                all_normals.push_back(0.0f);
                all_normals.push_back(0.0f);
                all_normals.push_back(1.0f);
            }
        }

        for (int i = 1; i <= nTris; ++i) {
            int n1, n2, n3;
            tri->Triangle(i).Get(n1, n2, n3);
            if (reversed) std::swap(n2, n3);
            all_tris.push_back(base + n1 - 1);
            all_tris.push_back(base + n2 - 1);
            all_tris.push_back(base + n3 - 1);
        }
        occ_face_idx++;
    }

    std::cout << "OCC faces: " << occ_face_idx << std::endl;

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

    // Normals
    size_t nrm_offset = buffer_data.size();
    size_t nrm_size = nv * 3 * sizeof(float);
    buffer_data.resize(nrm_offset + nrm_size);
    std::memcpy(buffer_data.data() + nrm_offset, all_normals.data(), nrm_size);

    // Face IDs (per-vertex, scalar float)
    size_t fid_offset = buffer_data.size();
    size_t fid_size = nv * sizeof(float);
    buffer_data.resize(fid_offset + fid_size);
    std::memcpy(buffer_data.data() + fid_offset, all_face_ids.data(), fid_size);

    // Indices
    size_t idx_offset = buffer_data.size();
    size_t idx_size = nf * 3 * sizeof(uint32_t);
    buffer_data.resize(idx_offset + idx_size);
    std::memcpy(buffer_data.data() + idx_offset, all_tris.data(), idx_size);

    tinygltf::Buffer buf;
    buf.data = buffer_data;
    model.buffers.push_back(buf);

    // BV 0: positions
    tinygltf::BufferView pos_bv;
    pos_bv.buffer = 0; pos_bv.byteOffset = 0;
    pos_bv.byteLength = pos_size; pos_bv.target = TINYGLTF_TARGET_ARRAY_BUFFER;
    model.bufferViews.push_back(pos_bv);

    // BV 1: normals
    tinygltf::BufferView nrm_bv;
    nrm_bv.buffer = 0; nrm_bv.byteOffset = nrm_offset;
    nrm_bv.byteLength = nrm_size; nrm_bv.target = TINYGLTF_TARGET_ARRAY_BUFFER;
    model.bufferViews.push_back(nrm_bv);

    // BV 2: face IDs
    tinygltf::BufferView fid_bv;
    fid_bv.buffer = 0; fid_bv.byteOffset = fid_offset;
    fid_bv.byteLength = fid_size; fid_bv.target = TINYGLTF_TARGET_ARRAY_BUFFER;
    model.bufferViews.push_back(fid_bv);

    // BV 3: indices
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

    // Accessor 0: positions
    tinygltf::Accessor pos_acc;
    pos_acc.bufferView = 0; pos_acc.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
    pos_acc.count = nv; pos_acc.type = TINYGLTF_TYPE_VEC3;
    pos_acc.minValues = {mn[0], mn[1], mn[2]};
    pos_acc.maxValues = {mx[0], mx[1], mx[2]};
    model.accessors.push_back(pos_acc);

    // Accessor 1: normals
    tinygltf::Accessor nrm_acc;
    nrm_acc.bufferView = 1; nrm_acc.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
    nrm_acc.count = nv; nrm_acc.type = TINYGLTF_TYPE_VEC3;
    model.accessors.push_back(nrm_acc);

    // Accessor 2: face IDs
    tinygltf::Accessor fid_acc;
    fid_acc.bufferView = 2; fid_acc.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
    fid_acc.count = nv; fid_acc.type = TINYGLTF_TYPE_SCALAR;
    model.accessors.push_back(fid_acc);

    // Accessor 3: indices
    tinygltf::Accessor idx_acc;
    idx_acc.bufferView = 3; idx_acc.componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;
    idx_acc.count = nf * 3; idx_acc.type = TINYGLTF_TYPE_SCALAR;
    model.accessors.push_back(idx_acc);

    tinygltf::Primitive prim;
    prim.attributes["POSITION"] = 0;
    prim.attributes["NORMAL"] = 1;
    prim.attributes["_FACE_ID"] = 2;
    prim.indices = 3;
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
