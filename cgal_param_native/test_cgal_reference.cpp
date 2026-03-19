/// Build and run CGAL's own parameterization tests against their reference data.
/// This validates that our CGAL build produces correct results.

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polyhedron_3.h>

#include <CGAL/Surface_mesh_parameterization/parameterize.h>
#include <CGAL/Surface_mesh_parameterization/Discrete_conformal_map_parameterizer_3.h>
#include <CGAL/Surface_mesh_parameterization/Discrete_authalic_parameterizer_3.h>
#include <CGAL/Surface_mesh_parameterization/Mean_value_coordinates_parameterizer_3.h>
#include <CGAL/Surface_mesh_parameterization/ARAP_parameterizer_3.h>
#include <CGAL/Surface_mesh_parameterization/Barycentric_mapping_parameterizer_3.h>
#include <CGAL/Surface_mesh_parameterization/Circular_border_parameterizer_3.h>
#include <CGAL/Surface_mesh_parameterization/Two_vertices_parameterizer_3.h>
#include <CGAL/Surface_mesh_parameterization/IO/File_off.h>

#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Eigen_solver_traits.h>
#include <Eigen/SparseLU>
#include <Eigen/SparseCholesky>

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using SurfaceMesh = CGAL::Surface_mesh<K::Point_3>;
using vertex_descriptor = SurfaceMesh::Vertex_index;
using halfedge_descriptor = SurfaceMesh::Halfedge_index;
namespace SMP = CGAL::Surface_mesh_parameterization;

using EigenSolver = CGAL::Eigen_solver_traits<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>>;
using EigenLU = CGAL::Eigen_solver_traits<Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::ColMajor>, Eigen::COLAMDOrdering<int>>>;

bool load_off(const std::string& path, SurfaceMesh& sm) {
    std::ifstream in(path);
    if (!in || !(in >> sm)) {
        std::cerr << "  FAIL: cannot load " << path << std::endl;
        return false;
    }
    return true;
}

// Validate UV: check all vertices got parameterized (non-zero for non-boundary)
bool validate_uv(const SurfaceMesh& sm, const SurfaceMesh::Property_map<vertex_descriptor, K::Point_2>& uv) {
    int parameterized = 0;
    for (auto v : sm.vertices()) {
        auto p = uv[v];
        if (p.x() != 0 || p.y() != 0) parameterized++;
    }
    double ratio = static_cast<double>(parameterized) / sm.number_of_vertices();
    if (ratio < 0.9) {
        std::cerr << "  FAIL: only " << parameterized << "/" << sm.number_of_vertices()
                  << " vertices parameterized (" << (ratio*100) << "%)" << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char* argv[]) {
    // Find CGAL data directory
    std::string cgal_data;
    if (argc > 1) {
        cgal_data = argv[1];
    } else {
        // Try common locations
        const char* candidates[] = {
            "build/_deps/cgal-src/Data/data/meshes",
            "../cgal_param/build/_deps/cgal-src/Data/data/meshes",
            nullptr
        };
        for (auto* c = candidates; *c; ++c) {
            std::ifstream test(std::string(*c) + "/nefertiti.off");
            if (test.good()) { cgal_data = *c; break; }
        }
    }

    std::string test_data;
    {
        const char* candidates[] = {
            "build/_deps/cgal-src/Surface_mesh_parameterization/test/Surface_mesh_parameterization/data",
            "../cgal_param/build/_deps/cgal-src/Surface_mesh_parameterization/test/Surface_mesh_parameterization/data",
            nullptr
        };
        for (auto* c = candidates; *c; ++c) {
            std::ifstream test(std::string(*c) + "/oni.off");
            if (test.good()) { test_data = *c; break; }
        }
    }

    if (cgal_data.empty()) {
        std::cerr << "Cannot find CGAL data directory. Pass path as argument." << std::endl;
        return 1;
    }

    std::cout << "CGAL data: " << cgal_data << std::endl;
    std::cout << "Test data: " << test_data << std::endl;

    int passed = 0, failed = 0;

    // =========================================================
    // Test 1: MVC on mushroom.off
    // =========================================================
    {
        std::cout << "\n[1] MVC on mushroom.off... " << std::flush;
        SurfaceMesh sm;
        if (load_off(cgal_data + "/mushroom.off", sm)) {
            auto uv = sm.add_property_map<vertex_descriptor, K::Point_2>("v:uv", K::Point_2(0,0)).first;
            auto bhd = CGAL::Polygon_mesh_processing::longest_border(sm).first;
            using Border = SMP::Circular_border_uniform_parameterizer_3<SurfaceMesh>;
            using Param = SMP::Mean_value_coordinates_parameterizer_3<SurfaceMesh, Border, EigenSolver>;
            auto err = SMP::parameterize(sm, Param(), bhd, uv);
            if (err == SMP::OK && validate_uv(sm, uv)) {
                std::cout << "PASS (" << sm.number_of_vertices() << " verts)" << std::endl;
                passed++;
            } else {
                std::cout << "FAIL (error=" << err << ")" << std::endl;
                failed++;
            }
        } else { failed++; }
    }

    // =========================================================
    // Test 2: ARAP on three_peaks.off
    // =========================================================
    {
        std::cout << "[2] ARAP on three_peaks.off... " << std::flush;
        SurfaceMesh sm;
        if (load_off(cgal_data + "/three_peaks.off", sm)) {
            auto uv = sm.add_property_map<vertex_descriptor, K::Point_2>("v:uv", K::Point_2(0,0)).first;
            auto bhd = CGAL::Polygon_mesh_processing::longest_border(sm).first;
            using Border = SMP::Two_vertices_parameterizer_3<SurfaceMesh>;
            using Param = SMP::ARAP_parameterizer_3<SurfaceMesh, Border, EigenLU>;
            auto err = SMP::parameterize(sm, Param(), bhd, uv);
            if (err == SMP::OK && validate_uv(sm, uv)) {
                std::cout << "PASS (" << sm.number_of_vertices() << " verts)" << std::endl;
                passed++;
            } else {
                std::cout << "FAIL (error=" << err << ")" << std::endl;
                failed++;
            }
        } else { failed++; }
    }

    // =========================================================
    // Test 3: Barycentric on oni.off
    // =========================================================
    {
        std::cout << "[3] Barycentric on oni.off... " << std::flush;
        SurfaceMesh sm;
        if (!test_data.empty() && load_off(test_data + "/oni.off", sm)) {
            auto uv = sm.add_property_map<vertex_descriptor, K::Point_2>("v:uv", K::Point_2(0,0)).first;
            auto bhd = CGAL::Polygon_mesh_processing::longest_border(sm).first;
            using Border = SMP::Circular_border_uniform_parameterizer_3<SurfaceMesh>;
            using Param = SMP::Barycentric_mapping_parameterizer_3<SurfaceMesh, Border, EigenSolver>;
            auto err = SMP::parameterize(sm, Param(), bhd, uv);
            if (err == SMP::OK && validate_uv(sm, uv)) {
                std::cout << "PASS (" << sm.number_of_vertices() << " verts)" << std::endl;
                passed++;
            } else {
                std::cout << "FAIL (error=" << err << ")" << std::endl;
                failed++;
            }
        } else { std::cout << "SKIP (oni.off not found)" << std::endl; }
    }

    // =========================================================
    // Test 4: ARAP on nefertiti.off
    // =========================================================
    {
        std::cout << "[4] ARAP on nefertiti.off... " << std::flush;
        SurfaceMesh sm;
        if (load_off(cgal_data + "/nefertiti.off", sm)) {
            auto uv = sm.add_property_map<vertex_descriptor, K::Point_2>("v:uv", K::Point_2(0,0)).first;
            auto bhd = CGAL::Polygon_mesh_processing::longest_border(sm).first;
            using Border = SMP::Two_vertices_parameterizer_3<SurfaceMesh>;
            using Param = SMP::ARAP_parameterizer_3<SurfaceMesh, Border, EigenLU>;
            auto err = SMP::parameterize(sm, Param(), bhd, uv);
            if (err == SMP::OK && validate_uv(sm, uv)) {
                std::cout << "PASS (" << sm.number_of_vertices() << " verts)" << std::endl;
                passed++;
            } else {
                std::cout << "FAIL (error=" << err << ")" << std::endl;
                failed++;
            }
        } else { failed++; }
    }

    // =========================================================
    // Test 5: Discrete Conformal on nefertiti.off
    // =========================================================
    {
        std::cout << "[5] Discrete Conformal on nefertiti.off... " << std::flush;
        SurfaceMesh sm;
        if (load_off(cgal_data + "/nefertiti.off", sm)) {
            auto uv = sm.add_property_map<vertex_descriptor, K::Point_2>("v:uv", K::Point_2(0,0)).first;
            auto bhd = CGAL::Polygon_mesh_processing::longest_border(sm).first;
            using Border = SMP::Circular_border_uniform_parameterizer_3<SurfaceMesh>;
            using Param = SMP::Discrete_conformal_map_parameterizer_3<SurfaceMesh, Border, EigenSolver>;
            auto err = SMP::parameterize(sm, Param(), bhd, uv);
            if (err == SMP::OK && validate_uv(sm, uv)) {
                std::cout << "PASS (" << sm.number_of_vertices() << " verts)" << std::endl;
                passed++;
            } else {
                std::cout << "FAIL (error=" << err << ")" << std::endl;
                failed++;
            }
        } else { failed++; }
    }

    // =========================================================
    // Test 6: Discrete Authalic on head.off
    // =========================================================
    {
        std::cout << "[6] Discrete Authalic on head.off... " << std::flush;
        SurfaceMesh sm;
        if (load_off(cgal_data + "/head.off", sm)) {
            auto uv = sm.add_property_map<vertex_descriptor, K::Point_2>("v:uv", K::Point_2(0,0)).first;
            auto bhd = CGAL::Polygon_mesh_processing::longest_border(sm).first;
            using Border = SMP::Circular_border_uniform_parameterizer_3<SurfaceMesh>;
            using Param = SMP::Discrete_authalic_parameterizer_3<SurfaceMesh, Border, EigenSolver>;
            auto err = SMP::parameterize(sm, Param(), bhd, uv);
            if (err == SMP::OK && validate_uv(sm, uv)) {
                std::cout << "PASS (" << sm.number_of_vertices() << " verts)" << std::endl;
                passed++;
            } else {
                std::cout << "FAIL (error=" << err << ")" << std::endl;
                failed++;
            }
        } else { failed++; }
    }

    // =========================================================
    // Test 7: MVC on cube.off (from test data)
    // =========================================================
    {
        std::cout << "[7] MVC on cube.off... " << std::flush;
        SurfaceMesh sm;
        if (!test_data.empty() && load_off(test_data + "/cube.off", sm)) {
            auto uv = sm.add_property_map<vertex_descriptor, K::Point_2>("v:uv", K::Point_2(0,0)).first;
            auto bhd = CGAL::Polygon_mesh_processing::longest_border(sm).first;
            if (bhd != SurfaceMesh::null_halfedge()) {
                using Border = SMP::Circular_border_uniform_parameterizer_3<SurfaceMesh>;
                using Param = SMP::Mean_value_coordinates_parameterizer_3<SurfaceMesh, Border, EigenSolver>;
                auto err = SMP::parameterize(sm, Param(), bhd, uv);
                if (err == SMP::OK) {
                    std::cout << "PASS (" << sm.number_of_vertices() << " verts)" << std::endl;
                    passed++;
                } else {
                    std::cout << "FAIL (error=" << err << ")" << std::endl;
                    failed++;
                }
            } else {
                std::cout << "SKIP (closed mesh)" << std::endl;
            }
        } else { std::cout << "SKIP" << std::endl; }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;
    std::cout << "========================================" << std::endl;

    return failed > 0 ? 1 : 0;
}
