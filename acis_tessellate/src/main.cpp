// ACIS Tessellator CLI: STEP → GLB
// Uses Interop for STEP import, ACIS faceting for tessellation.
// Outputs shared-vertex GLB with per-B-Rep-face normals + _FACE_ID.

#define NO_ACIS_IO_REDIRECTION 1
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <tiny_gltf.h>

// System
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WINDOWS_SOURCE
#include <wtypes.h>
#include <winnt.h>
#undef GetMessage
#endif

// Interop
#include "SPAIDocument.h"
#include "SPAIConverter.h"
#include "SPAIOptions.h"
#include "SPAIOptionName.h"
#include "SPAIResult.h"
#include "SPAIValue.h"
#include "SPAISystemInitGuard.h"
#include "SPAIAcisDocument.h"
#include "SPAIUnit.h"

// ACIS
#include "kernapi.hxx"
#include "lists.hxx"
#include "body.hxx"
#include "face.hxx"
#include "lump.hxx"
#include "shell.hxx"
#include "af_api.hxx"
#include "idx_mesh.hxx"
#include "facet_options.hxx"

#include <iostream>
#include <vector>
#include <map>
#include <cstring>
#include <cmath>
#include <string>
#include <sstream>
#include <fstream>
#include <chrono>

extern void unlock_license();

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: acis_tessellate_cli input.step output.glb [--deflection D]" << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    double deflection = 1.0;

    for (int i = 3; i < argc; ++i) {
        if (std::string(argv[i]) == "--deflection" && i + 1 < argc)
            deflection = std::stod(argv[++i]);
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    // Unlock license
    unlock_license();

    // Start ACIS modeller
    api_start_modeller(0);

    std::vector<float> all_verts;
    std::vector<float> all_normals;
    std::vector<float> all_face_ids;
    std::vector<uint32_t> all_tris;
    int occ_face_idx = 0; // B-Rep face counter (named occ for consistency)

    {
        // Initialize Interop
        SPAISystemInitGuard initGuard;

        // Import STEP
        SPAIDocument src(input_path.c_str());
        SPAIAcisDocument dst;

        SPAIOptions options;
        SPAIValue representation("BRep+Assembly");
        options.Add(SPAIOptionName::Representation, representation);

        SPAIConverter converter;
        SPAIResult result = converter.Convert(src, dst);

        if (result != SPAI_S_OK) {
            std::cerr << "Error: STEP import failed: " << result.GetMessage() << std::endl;
            api_stop_modeller();
            return 1;
        }

        ENTITY_LIST* pEntities = NULL;
        dst.GetEntities(pEntities);

        if (!pEntities || pEntities->count() == 0) {
            std::cerr << "Error: No entities in STEP file" << std::endl;
            api_stop_modeller();
            return 1;
        }

        std::cout << "STEP loaded: " << pEntities->count() << " entities" << std::endl;

        // Get unit scale factor (to mm)
        SPAIUnit unit;
        dst.GetUnit(unit);
        double scale_to_mm = unit.GetMMScaleFactor();
        std::cout << "Unit scale to mm: " << scale_to_mm << std::endl;

        // Initialize faceter
        api_initialize_faceter();

        // Set facet options (precise mode for surface tolerance control)
        facet_options_precise fo;
        fo.set_surface_tolerance(deflection);
        fo.set_normal_tolerance(15.0); // 15 degree

        // Iterate over entities, facet each BODY
        for (int ei = 0; ei < pEntities->count(); ++ei) {
            ENTITY* ent = (*pEntities)[ei];
            if (!is_BODY(ent)) continue;

            BODY* body = (BODY*)ent;

            // Facet the body
            outcome o = api_facet_entity(body, &fo);
            if (!o.ok()) {
                std::cerr << "Warning: faceting failed for entity " << ei << std::endl;
                continue;
            }

            // Extract mesh per face (split vertices at B-Rep boundaries)
            ENTITY_LIST face_list;
            api_get_faces(body, face_list);

            for (int fi = 0; fi < face_list.count(); ++fi) {
                FACE* face = (FACE*)face_list[fi];

                INDEXED_MESH* mesh = NULL;
                outcome mo = api_get_indexed_mesh(face, mesh);
                if (!mo.ok() || !mesh) continue;

                int npts = mesh->get_num_vertex();
                int npolys = mesh->get_num_polygon();
                uint32_t base = static_cast<uint32_t>(all_verts.size() / 3);

                // Add vertices + normals (per face = split at boundaries)
                for (int vi = 0; vi < npts; ++vi) {
                    const SPAposition& pt = mesh->get_position(vi);
                    all_verts.push_back(static_cast<float>(pt.x() * scale_to_mm));
                    all_verts.push_back(static_cast<float>(pt.y() * scale_to_mm));
                    all_verts.push_back(static_cast<float>(pt.z() * scale_to_mm));

                    const SPAunit_vector& nrm = mesh->get_normal(vi);
                    all_normals.push_back(static_cast<float>(nrm.x()));
                    all_normals.push_back(static_cast<float>(nrm.y()));
                    all_normals.push_back(static_cast<float>(nrm.z()));

                    all_face_ids.push_back(static_cast<float>(occ_face_idx));
                }

                // Add triangles from polygons
                for (int pi = 0; pi < npolys; ++pi) {
                    indexed_polygon* poly = mesh->get_polygon(pi);
                    int poly_nv = poly->num_vertex();
                    if (poly_nv == 3) {
                        for (int k = 0; k < 3; ++k) {
                            polygon_vertex* pv = poly->get_vertex(k);
                            int vi = mesh->get_vertex_index(pv);
                            all_tris.push_back(base + vi);
                        }
                    } else if (poly_nv > 3) {
                        polygon_vertex* pv0 = poly->get_vertex(0);
                        int v0 = mesh->get_vertex_index(pv0);
                        for (int k = 1; k + 1 < poly_nv; ++k) {
                            polygon_vertex* pv1 = poly->get_vertex(k);
                            polygon_vertex* pv2 = poly->get_vertex(k + 1);
                            all_tris.push_back(base + v0);
                            all_tris.push_back(base + mesh->get_vertex_index(pv1));
                            all_tris.push_back(base + mesh->get_vertex_index(pv2));
                        }
                    }
                }

                occ_face_idx++;
            }
        }

        api_terminate_faceter();

        ACIS_DELETE(pEntities);
    }

    api_stop_modeller();

    int nv = static_cast<int>(all_verts.size() / 3);
    int nf = static_cast<int>(all_tris.size() / 3);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Mesh: " << nv << " vertices, " << nf << " triangles, "
              << occ_face_idx << " B-Rep faces (" << ms << " ms)" << std::endl;

    if (nv == 0) {
        std::cerr << "Error: No triangulation produced" << std::endl;
        return 1;
    }

    // Write GLB (same format as OCC tessellator: POSITION + NORMAL + _FACE_ID + indices)
    tinygltf::Model model;
    std::vector<uint8_t> buffer_data;

    size_t pos_size = nv * 3 * sizeof(float);
    buffer_data.resize(pos_size);
    std::memcpy(buffer_data.data(), all_verts.data(), pos_size);

    size_t nrm_offset = buffer_data.size();
    size_t nrm_size = nv * 3 * sizeof(float);
    buffer_data.resize(nrm_offset + nrm_size);
    std::memcpy(buffer_data.data() + nrm_offset, all_normals.data(), nrm_size);

    size_t fid_offset = buffer_data.size();
    size_t fid_size = nv * sizeof(float);
    buffer_data.resize(fid_offset + fid_size);
    std::memcpy(buffer_data.data() + fid_offset, all_face_ids.data(), fid_size);

    size_t idx_offset = buffer_data.size();
    size_t idx_size = nf * 3 * sizeof(uint32_t);
    buffer_data.resize(idx_offset + idx_size);
    std::memcpy(buffer_data.data() + idx_offset, all_tris.data(), idx_size);

    tinygltf::Buffer buf; buf.data = buffer_data; model.buffers.push_back(buf);

    tinygltf::BufferView bv0; bv0.buffer=0; bv0.byteOffset=0; bv0.byteLength=pos_size; bv0.target=34962; model.bufferViews.push_back(bv0);
    tinygltf::BufferView bv1; bv1.buffer=0; bv1.byteOffset=nrm_offset; bv1.byteLength=nrm_size; bv1.target=34962; model.bufferViews.push_back(bv1);
    tinygltf::BufferView bv2; bv2.buffer=0; bv2.byteOffset=fid_offset; bv2.byteLength=fid_size; bv2.target=34962; model.bufferViews.push_back(bv2);
    tinygltf::BufferView bv3; bv3.buffer=0; bv3.byteOffset=idx_offset; bv3.byteLength=idx_size; bv3.target=34963; model.bufferViews.push_back(bv3);

    float mn[3]={1e30f,1e30f,1e30f}, mx[3]={-1e30f,-1e30f,-1e30f};
    for (int i=0; i<nv; ++i) for (int k=0; k<3; ++k) {
        mn[k]=std::min(mn[k],all_verts[i*3+k]); mx[k]=std::max(mx[k],all_verts[i*3+k]);
    }

    tinygltf::Accessor a0; a0.bufferView=0; a0.componentType=5126; a0.count=nv; a0.type=TINYGLTF_TYPE_VEC3;
    a0.minValues={mn[0],mn[1],mn[2]}; a0.maxValues={mx[0],mx[1],mx[2]}; model.accessors.push_back(a0);
    tinygltf::Accessor a1; a1.bufferView=1; a1.componentType=5126; a1.count=nv; a1.type=TINYGLTF_TYPE_VEC3; model.accessors.push_back(a1);
    tinygltf::Accessor a2; a2.bufferView=2; a2.componentType=5126; a2.count=nv; a2.type=TINYGLTF_TYPE_SCALAR; model.accessors.push_back(a2);
    tinygltf::Accessor a3; a3.bufferView=3; a3.componentType=5125; a3.count=nf*3; a3.type=TINYGLTF_TYPE_SCALAR; model.accessors.push_back(a3);

    tinygltf::Primitive prim; prim.attributes["POSITION"]=0; prim.attributes["NORMAL"]=1; prim.attributes["_FACE_ID"]=2; prim.indices=3; prim.mode=4;
    tinygltf::Mesh gm; gm.primitives.push_back(prim); model.meshes.push_back(gm);
    tinygltf::Node node; node.mesh=0; model.nodes.push_back(node);
    tinygltf::Scene sc; sc.nodes.push_back(0); model.scenes.push_back(sc);
    model.defaultScene=0; model.asset.version="2.0"; model.asset.generator="acis_tessellate_cli";

    tinygltf::TinyGLTF writer;
    std::ostringstream oss;
    writer.WriteGltfSceneToStream(&model, oss, false, true);
    std::string data = oss.str();
    std::ofstream f(output_path, std::ios::binary);
    f.write(data.data(), data.size());

    std::cout << "Wrote: " << output_path << " (" << data.size() << " bytes)" << std::endl;
    std::cout << "Extent: " << (mx[0]-mn[0]) << " x " << (mx[1]-mn[1]) << " x " << (mx[2]-mn[2]) << std::endl;

    return 0;
}
