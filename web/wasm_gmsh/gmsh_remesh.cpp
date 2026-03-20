// Gmsh WASM remeshing wrapper
// Imports a triangle mesh via Gmsh C API, remeshes isotropically, exports GLB.

#include <emscripten/bind.h>
#include <emscripten/val.h>
extern "C" {
#include <gmshc.h>
}
#include <vector>
#include <cmath>
#include <cstring>
#include <string>
#include <sstream>
#include <cstdint>

using namespace emscripten;

static std::string g_metrics_json;

// Parse a GLB buffer into vertices and triangles
// Assumes our standard GLB layout: accessor 0 = POSITION (VEC3/float),
// accessor 1 = TEXCOORD or INDICES, last SCALAR accessor = indices
struct RawMesh {
    std::vector<double> verts;
    std::vector<int> tris;
    int nv = 0, nf = 0;
};

// Minimal JSON int extraction: find "key": and read the integer after it
static int json_int(const std::string& json, const std::string& key, size_t start = 0) {
    std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle, start);
    if (pos == std::string::npos) return -1;
    pos += needle.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == ':' || json[pos] == '\t' || json[pos] == '\n')) pos++;
    if (pos >= json.size()) return -1;
    bool neg = json[pos] == '-'; if (neg) pos++;
    int val = 0;
    while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') { val = val * 10 + (json[pos] - '0'); pos++; }
    return neg ? -val : val;
}

static RawMesh parse_glb(const uint8_t* data, size_t len) {
    RawMesh m;
    if (len < 20) return m;

    // GLB header
    uint32_t jlen;
    memcpy(&jlen, data + 12, 4);
    std::string json((const char*)(data + 20), jlen);

    // Binary chunk
    size_t bin_start = 20 + jlen;
    while (bin_start % 4) bin_start++;
    bin_start += 8;
    const uint8_t* bin = data + bin_start;

    // Our GLBs have 2 bufferViews: [0]=positions, [1]=indices (or [0]=pos, [1]=uv, [2]=idx)
    // Find bufferView byteOffsets and byteLengths
    struct BV { int off; int len; };
    std::vector<BV> bvs;
    size_t bvp = json.find("\"bufferViews\"");
    if (bvp == std::string::npos) return m;
    size_t arr = json.find('[', bvp);
    // Find each { ... } in the array
    int depth = 0;
    size_t obj_start = 0;
    for (size_t i = arr; i < json.size(); i++) {
        if (json[i] == '{' && depth == 1) obj_start = i;
        if (json[i] == '[' || json[i] == '{') depth++;
        if (json[i] == ']' || json[i] == '}') {
            depth--;
            if (json[i] == '}' && depth == 1) {
                std::string obj = json.substr(obj_start, i - obj_start + 1);
                BV bv;
                bv.off = json_int(obj, "byteOffset");
                if (bv.off < 0) bv.off = 0;
                bv.len = json_int(obj, "byteLength");
                if (bv.len < 0) bv.len = 0;
                bvs.push_back(bv);
            }
            if (depth == 0) break;
        }
    }

    // Find accessors
    struct Acc { int bv; int count; int ct; };
    std::vector<Acc> accs;
    size_t ap = json.find("\"accessors\"");
    if (ap == std::string::npos) return m;
    arr = json.find('[', ap);
    depth = 0;
    for (size_t i = arr; i < json.size(); i++) {
        if (json[i] == '{' && depth == 1) obj_start = i;
        if (json[i] == '[' || json[i] == '{') depth++;
        if (json[i] == ']' || json[i] == '}') {
            depth--;
            if (json[i] == '}' && depth == 1) {
                std::string obj = json.substr(obj_start, i - obj_start + 1);
                Acc a;
                a.bv = json_int(obj, "bufferView");
                a.count = json_int(obj, "count");
                a.ct = json_int(obj, "componentType");
                accs.push_back(a);
            }
            if (depth == 0) break;
        }
    }

    // Find POSITION accessor (VEC3 = first one with componentType 5126 and count>0)
    // and INDICES accessor (SCALAR with componentType 5125 or 5123)
    int pos_idx = -1, idx_idx = -1;
    for (int i = 0; i < (int)accs.size(); i++) {
        // Check if this accessor's section of JSON contains "VEC3" or "SCALAR"
        if (accs[i].ct == 5126 && pos_idx < 0) pos_idx = i;
        if ((accs[i].ct == 5125 || accs[i].ct == 5123) && idx_idx < 0) idx_idx = i;
    }

    if (pos_idx < 0 || idx_idx < 0 || accs[pos_idx].bv >= (int)bvs.size() || accs[idx_idx].bv >= (int)bvs.size()) {
        printf("[gmsh_remesh] parse_glb: pos_idx=%d idx_idx=%d bvs=%d accs=%d\n", pos_idx, idx_idx, (int)bvs.size(), (int)accs.size());
        return m;
    }

    // Read positions
    m.nv = accs[pos_idx].count;
    m.verts.resize(m.nv * 3);
    const float* pdata = (const float*)(bin + bvs[accs[pos_idx].bv].off);
    for (int i = 0; i < m.nv * 3; i++) m.verts[i] = pdata[i];

    // Read indices
    int idx_count = accs[idx_idx].count;
    m.nf = idx_count / 3;
    m.tris.resize(idx_count);
    const uint8_t* idata = bin + bvs[accs[idx_idx].bv].off;
    int ct = accs[idx_idx].ct;
    for (int i = 0; i < idx_count; i++) {
        if (ct == 5125) { uint32_t v; memcpy(&v, idata + i * 4, 4); m.tris[i] = v; }
        else if (ct == 5123) { uint16_t v; memcpy(&v, idata + i * 2, 2); m.tris[i] = v; }
        else m.tris[i] = idata[i];
    }

    printf("[gmsh_remesh] Parsed: %d verts, %d tris\n", m.nv, m.nf);
    return m;
}

// Build a GLB from vertices, triangles, and optional UVs
static std::vector<uint8_t> build_glb(const std::vector<double>& verts, int nv,
                                       const std::vector<int>& tris, int nf) {
    // Binary buffer: positions + indices
    size_t pos_size = nv * 3 * 4;
    size_t idx_size = nf * 3 * 4;
    size_t buf_size = pos_size + idx_size;
    while (buf_size % 4) buf_size++;

    std::vector<uint8_t> buf(buf_size, 0);
    for (int i = 0; i < nv; i++) {
        float v[3] = {(float)verts[i*3], (float)verts[i*3+1], (float)verts[i*3+2]};
        memcpy(buf.data() + i * 12, v, 12);
    }
    for (int i = 0; i < nf; i++) {
        uint32_t t[3] = {(uint32_t)tris[i*3], (uint32_t)tris[i*3+1], (uint32_t)tris[i*3+2]};
        memcpy(buf.data() + pos_size + i * 12, t, 12);
    }

    // Compute min/max
    double mn[3] = {1e30, 1e30, 1e30}, mx[3] = {-1e30, -1e30, -1e30};
    for (int i = 0; i < nv; i++) {
        for (int k = 0; k < 3; k++) {
            mn[k] = std::min(mn[k], verts[i*3+k]);
            mx[k] = std::max(mx[k], verts[i*3+k]);
        }
    }

    std::ostringstream js;
    js << "{\"asset\":{\"version\":\"2.0\",\"generator\":\"GmshWASM\"},"
       << "\"scene\":0,\"scenes\":[{\"nodes\":[0]}],\"nodes\":[{\"mesh\":0}],"
       << "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0},\"indices\":1,\"mode\":4}]}],"
       << "\"accessors\":["
       << "{\"bufferView\":0,\"componentType\":5126,\"count\":" << nv << ",\"type\":\"VEC3\","
       << "\"min\":[" << mn[0] << "," << mn[1] << "," << mn[2] << "],"
       << "\"max\":[" << mx[0] << "," << mx[1] << "," << mx[2] << "]},"
       << "{\"bufferView\":1,\"componentType\":5125,\"count\":" << nf*3 << ",\"type\":\"SCALAR\"}],"
       << "\"bufferViews\":["
       << "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":" << pos_size << ",\"target\":34962},"
       << "{\"buffer\":0,\"byteOffset\":" << pos_size << ",\"byteLength\":" << idx_size << ",\"target\":34963}],"
       << "\"buffers\":[{\"byteLength\":" << buf_size << "}]}";

    std::string json_str = js.str();
    while (json_str.size() % 4) json_str += ' ';

    uint32_t total = 12 + 8 + json_str.size() + 8 + buf_size;
    std::vector<uint8_t> glb(total);
    uint8_t* w = glb.data();

    // Header
    memcpy(w, "glTF", 4); w += 4;
    uint32_t v = 2; memcpy(w, &v, 4); w += 4;
    memcpy(w, &total, 4); w += 4;

    // JSON chunk
    uint32_t jl = json_str.size(); memcpy(w, &jl, 4); w += 4;
    uint32_t jt = 0x4E4F534A; memcpy(w, &jt, 4); w += 4;
    memcpy(w, json_str.data(), jl); w += jl;

    // BIN chunk
    uint32_t bl = buf_size; memcpy(w, &bl, 4); w += 4;
    uint32_t bt = 0x004E4942; memcpy(w, &bt, 4); w += 4;
    memcpy(w, buf.data(), buf_size);

    return glb;
}

val remesh_glb(val input_array, int max_triangles) {
    auto length = input_array["length"].as<unsigned>();
    std::vector<uint8_t> data(length);
    for (unsigned i = 0; i < length; i++) data[i] = input_array[i].as<uint8_t>();

    RawMesh input = parse_glb(data.data(), data.size());
    if (input.nv == 0 || input.nf == 0) {
        g_metrics_json = "{\"error\":\"Failed to parse GLB\",\"nv\":" + std::to_string(input.nv) + ",\"nf\":" + std::to_string(input.nf) + "}";
        return val::null();
    }

    // Log input
    printf("[gmsh_remesh] Input: %d verts, %d tris, max_triangles=%d\n", input.nv, input.nf, max_triangles);

    int ierr = 0;
    gmshInitialize(0, NULL, 0, 0, &ierr);
    if (ierr) { printf("[gmsh_remesh] gmshInitialize failed: %d\n", ierr); g_metrics_json = "{\"error\":\"gmshInitialize\"}"; return val::null(); }
    gmshModelAdd("remesh", &ierr);

    // Add all vertices as discrete mesh nodes
    // First, create a discrete surface entity
    int dim = 2, tag = 1;
    gmshModelAddDiscreteEntity(dim, tag, NULL, 0, &ierr);

    // Set mesh nodes
    std::vector<size_t> nodeTags(input.nv);
    std::vector<double> coords(input.nv * 3);
    for (int i = 0; i < input.nv; i++) {
        nodeTags[i] = i + 1; // 1-based
        coords[i*3]   = input.verts[i*3];
        coords[i*3+1] = input.verts[i*3+1];
        coords[i*3+2] = input.verts[i*3+2];
    }
    gmshModelMeshAddNodes(dim, tag, nodeTags.data(), nodeTags.size(),
                          coords.data(), coords.size(), NULL, 0, &ierr);

    // Set mesh elements (triangles = type 2)
    std::vector<size_t> elemTags(input.nf);
    std::vector<size_t> elemNodeTags(input.nf * 3);
    for (int i = 0; i < input.nf; i++) {
        elemTags[i] = i + 1;
        elemNodeTags[i*3]   = input.tris[i*3] + 1;
        elemNodeTags[i*3+1] = input.tris[i*3+1] + 1;
        elemNodeTags[i*3+2] = input.tris[i*3+2] + 1;
    }
    gmshModelMeshAddElementsByType(tag, 2 /*triangle*/,
                                   elemTags.data(), elemTags.size(),
                                   elemNodeTags.data(), elemNodeTags.size(),
                                   &ierr);

    // Classify mesh to create topology (edges, vertices)
    double angle = 40.0; // feature angle in degrees
    gmshModelMeshClassifySurfaces(angle * M_PI / 180.0, 1, 1, 2.0 * M_PI, 0, &ierr);
    printf("[gmsh_remesh] classifySurfaces: ierr=%d\n", ierr);
    gmshModelMeshCreateGeometry(NULL, 0, &ierr);
    printf("[gmsh_remesh] createGeometry: ierr=%d\n", ierr);

    // Compute average edge length from input mesh for sizing
    double totalLen = 0;
    int edgeCount = 0;
    for (int i = 0; i < input.nf; i++) {
        for (int e = 0; e < 3; e++) {
            int a = input.tris[i*3 + e];
            int b = input.tris[i*3 + ((e+1) % 3)];
            double dx = input.verts[a*3] - input.verts[b*3];
            double dy = input.verts[a*3+1] - input.verts[b*3+1];
            double dz = input.verts[a*3+2] - input.verts[b*3+2];
            totalLen += std::sqrt(dx*dx + dy*dy + dz*dz);
            edgeCount++;
        }
    }
    double avgEdge = totalLen / edgeCount;

    // Estimate: for uniform tris, nTris ~ 2 * area / (sqrt(3)/4 * h^2) ~ 4.62 * area / h^2
    // Start with current average edge length, increase if over limit
    double h = avgEdge;
    if (max_triangles > 0 && input.nf > max_triangles) {
        // Scale up edge length to reduce triangle count
        h *= std::sqrt((double)input.nf / max_triangles);
    }

    // Set meshing parameters for isotropic remeshing
    gmshOptionSetNumber("Mesh.Algorithm", 6, &ierr);        // Frontal-Delaunay
    gmshOptionSetNumber("Mesh.CharacteristicLengthMin", h * 0.5, &ierr);
    gmshOptionSetNumber("Mesh.CharacteristicLengthMax", h * 2.0, &ierr);
    gmshOptionSetNumber("Mesh.CharacteristicLengthFromCurvature", 0, &ierr);
    gmshOptionSetNumber("Mesh.MeshSizeExtendFromBoundary", 1, &ierr);

    // Remesh
    gmshModelMeshGenerate(2, &ierr);

    // If over limit, increase h and retry
    for (int retry = 0; retry < 5 && max_triangles > 0; retry++) {
        size_t *et = NULL, et_n;
        size_t *ent = NULL, ent_n;
        gmshModelMeshGetElementsByType(2 /*triangle*/, &et, &et_n,
                                       &ent, &ent_n, -1, 0, 1, &ierr);
        int total_tris = (int)et_n;
        gmshFree(et); gmshFree(ent);

        if (total_tris <= max_triangles) break;

        h *= std::sqrt((double)total_tris / max_triangles) * 1.1;
        gmshOptionSetNumber("Mesh.CharacteristicLengthMin", h * 0.5, &ierr);
        gmshOptionSetNumber("Mesh.CharacteristicLengthMax", h * 2.0, &ierr);
        gmshModelMeshGenerate(2, &ierr);
    }

    // Extract result mesh
    size_t *nodeTagsOut = NULL, nodeTagsOut_n;
    double *coordsOut = NULL, *paramOut = NULL;
    size_t coordsOut_n, paramOut_n;
    gmshModelMeshGetNodes(&nodeTagsOut, &nodeTagsOut_n,
                          &coordsOut, &coordsOut_n,
                          &paramOut, &paramOut_n,
                          -1, -1, 0, 0, &ierr);

    // Build node tag -> index map
    size_t maxTag = 0;
    for (size_t i = 0; i < nodeTagsOut_n; i++)
        if (nodeTagsOut[i] > maxTag) maxTag = nodeTagsOut[i];

    std::vector<int> tagToIdx(maxTag + 1, -1);
    int outNv = (int)nodeTagsOut_n;
    std::vector<double> outVerts(outNv * 3);
    for (int i = 0; i < outNv; i++) {
        tagToIdx[nodeTagsOut[i]] = i;
        outVerts[i*3]   = coordsOut[i*3];
        outVerts[i*3+1] = coordsOut[i*3+1];
        outVerts[i*3+2] = coordsOut[i*3+2];
    }

    // Get triangles
    size_t *elemTagsOut = NULL, elemTagsOut_n;
    size_t *elemNodeTagsOut = NULL, elemNodeTagsOut_n;
    gmshModelMeshGetElementsByType(2 /*triangle*/, &elemTagsOut, &elemTagsOut_n,
                                   &elemNodeTagsOut, &elemNodeTagsOut_n,
                                   -1, 0, 1, &ierr);

    std::vector<int> outTris(elemNodeTagsOut_n);
    for (size_t i = 0; i < elemNodeTagsOut_n; i++) {
        int idx = (elemNodeTagsOut[i] <= maxTag) ? tagToIdx[elemNodeTagsOut[i]] : 0;
        outTris[i] = (idx >= 0) ? idx : 0;
    }
    int outNf = (int)elemTagsOut_n;

    // Free Gmsh memory
    gmshFree(nodeTagsOut); gmshFree(coordsOut); gmshFree(paramOut);
    gmshFree(elemTagsOut); gmshFree(elemNodeTagsOut);

    gmshFinalize(&ierr);

    // Build metrics
    std::ostringstream oss;
    oss << "{\"vertices\":" << outNv
        << ",\"faces\":" << outNf
        << ",\"input_vertices\":" << input.nv
        << ",\"input_faces\":" << input.nf
        << "}";
    g_metrics_json = oss.str();

    // Build GLB
    auto glb = build_glb(outVerts, outNv, outTris, outNf);

    val out = val::global("Uint8Array").new_(glb.size());
    for (size_t i = 0; i < glb.size(); i++) out.set(i, glb[i]);
    return out;
}

std::string get_metrics() { return g_metrics_json; }

EMSCRIPTEN_BINDINGS(gmsh_remesh) {
    function("remeshGlb", &remesh_glb);
    function("getMetrics", &get_metrics);
}
