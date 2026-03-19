#include "cgalparam/cgal_parameterize.h"
#include "cgalparam/seam_cut.h"
#include "cgalparam/types.h"

#include <CGAL/Surface_mesh_parameterization/parameterize.h>
// LSCM only available with EPICK kernel (native build with GMP).
// With Simple_cartesian it triggers a boost::property_map const issue.
#if defined(CGALPARAM_NATIVE)
    #include <CGAL/Surface_mesh_parameterization/LSCM_parameterizer_3.h>
#endif
#include <CGAL/Surface_mesh_parameterization/ARAP_parameterizer_3.h>
#include <CGAL/Surface_mesh_parameterization/Discrete_conformal_map_parameterizer_3.h>
#include <CGAL/Surface_mesh_parameterization/Discrete_authalic_parameterizer_3.h>
#include <CGAL/Surface_mesh_parameterization/Mean_value_coordinates_parameterizer_3.h>
#include <CGAL/Surface_mesh_parameterization/Square_border_parameterizer_3.h>
#include <CGAL/Surface_mesh_parameterization/Circular_border_parameterizer_3.h>
#include <CGAL/Surface_mesh_parameterization/Two_vertices_parameterizer_3.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/boost/graph/Euler_operations.h>

#include <CGAL/Eigen_solver_traits.h>
#include <Eigen/SparseLU>
#include <Eigen/SparseCholesky>

// Default solver for symmetric positive-definite systems
using EigenSolver = CGAL::Eigen_solver_traits<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>>;
// Solver for general (non-symmetric) systems
using EigenLUSolver = CGAL::Eigen_solver_traits<Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::ColMajor>, Eigen::COLAMDOrdering<int>>>;

#include <iostream>
#include <stdexcept>

namespace SMP = CGAL::Surface_mesh_parameterization;

namespace cgalparam {

std::string method_name(ParamMethod m) {
    switch (m) {
        case ParamMethod::LSCM: return "LSCM";
        case ParamMethod::ARAP: return "ARAP";
        case ParamMethod::DiscreteConformal: return "Discrete Conformal";
        case ParamMethod::DiscreteAuthalic: return "Discrete Authalic";
        case ParamMethod::MeanValue: return "Mean Value Coordinates";
    }
    return "Unknown";
}

Eigen::MatrixXd parameterize(SurfaceMesh& mesh, ParamMethod method) {
    // Step 1: If mesh is closed, cut along a geodesic seam BEFORE creating property maps
    halfedge_descriptor bhd = CGAL::Polygon_mesh_processing::longest_border(mesh).first;
    if (bhd == SurfaceMesh::null_halfedge()) {
        std::cout << "[cgalparam] Mesh is closed — cutting along geodesic seam..." << std::endl;
        auto cut_result = cut_to_disk(mesh);
        mesh = cut_result.cut_mesh;
        bhd = CGAL::Polygon_mesh_processing::longest_border(mesh).first;
        if (bhd == SurfaceMesh::null_halfedge()) {
            throw std::runtime_error("Failed to create boundary after seam cut");
        }
    }

    // Step 2: Create UV property map on the (possibly cut) mesh
    using UV_vpmap = SurfaceMesh::Property_map<vertex_descriptor, Kernel::Point_2>;
    bool use_halfedge_uv = false;
    bool created;
    UV_vpmap uv_vmap;

#if defined(CGALPARAM_NATIVE)
    using UV_hpmap = SurfaceMesh::Property_map<halfedge_descriptor, Kernel::Point_2>;
    UV_hpmap uv_hmap;
    if (method == ParamMethod::LSCM) {
        use_halfedge_uv = true;
        std::tie(uv_hmap, created) = mesh.add_property_map<halfedge_descriptor, Kernel::Point_2>(
            "h:uv", Kernel::Point_2(0, 0));
    } else
#endif
    {
        std::tie(uv_vmap, created) = mesh.add_property_map<vertex_descriptor, Kernel::Point_2>(
            "v:uv", Kernel::Point_2(0, 0));
    }

    std::cout << "[cgalparam] Running " << method_name(method) << "..." << std::endl;

    SMP::Error_code err;

    switch (method) {
        case ParamMethod::LSCM: {
            // LSCM disabled: CGAL 6.x + Boost 1.90 has a property_map regression
            // that prevents compilation with halfedge UV maps.
            throw std::runtime_error("LSCM unavailable (CGAL 6.x/Boost 1.90 property_map issue). Use conformal instead.");
            break;
        }
        case ParamMethod::ARAP: {
            using BorderParam = SMP::Two_vertices_parameterizer_3<SurfaceMesh>;
            using Parameterizer = SMP::ARAP_parameterizer_3<SurfaceMesh, BorderParam, EigenLUSolver>;
            Parameterizer param;
            err = SMP::parameterize(mesh, param, bhd, uv_vmap);
            break;
        }
        case ParamMethod::DiscreteConformal: {
            using BorderParam = SMP::Circular_border_uniform_parameterizer_3<SurfaceMesh>;
            using Parameterizer = SMP::Discrete_conformal_map_parameterizer_3<SurfaceMesh, BorderParam, EigenSolver>;
            Parameterizer param;
            err = SMP::parameterize(mesh, param, bhd, uv_vmap);
            break;
        }
        case ParamMethod::DiscreteAuthalic: {
            using BorderParam = SMP::Circular_border_uniform_parameterizer_3<SurfaceMesh>;
            using Parameterizer = SMP::Discrete_authalic_parameterizer_3<SurfaceMesh, BorderParam, EigenSolver>;
            Parameterizer param;
            err = SMP::parameterize(mesh, param, bhd, uv_vmap);
            break;
        }
        case ParamMethod::MeanValue: {
            using BorderParam = SMP::Circular_border_uniform_parameterizer_3<SurfaceMesh>;
            using Parameterizer = SMP::Mean_value_coordinates_parameterizer_3<SurfaceMesh, BorderParam, EigenSolver>;
            Parameterizer param;
            err = SMP::parameterize(mesh, param, bhd, uv_vmap);
            break;
        }
    }

    if (err != SMP::OK) {
        throw std::runtime_error("Parameterization failed with error code " + std::to_string(err));
    }

    // Extract UV coordinates per vertex
    int n = static_cast<int>(mesh.number_of_vertices());
    Eigen::MatrixXd UV = Eigen::MatrixXd::Zero(n, 2);

#if defined(CGALPARAM_NATIVE)
    if (use_halfedge_uv) {
        // LSCM uses halfedge UV map — average per vertex
        Eigen::VectorXi count = Eigen::VectorXi::Zero(n);
        for (auto f : mesh.faces()) {
            for (auto h : mesh.halfedges_around_face(mesh.halfedge(f))) {
                int vi = static_cast<int>(mesh.target(h));
                auto uv = uv_hmap[h];
                UV(vi, 0) += uv.x();
                UV(vi, 1) += uv.y();
                count(vi) += 1;
            }
        }
        for (int i = 0; i < n; ++i) {
            if (count(i) > 0) { UV(i, 0) /= count(i); UV(i, 1) /= count(i); }
        }
    } else
#endif
    {
        for (auto v : mesh.vertices()) {
            int vi = static_cast<int>(v);
            auto uv = uv_vmap[v];
            UV(vi, 0) = uv.x();
            UV(vi, 1) = uv.y();
        }
    }

    // Normalize to [0,1]
    Eigen::Vector2d uv_min = UV.colwise().minCoeff();
    Eigen::Vector2d uv_max = UV.colwise().maxCoeff();
    Eigen::Vector2d uv_range = uv_max - uv_min;
    for (int i = 0; i < 2; ++i) {
        if (uv_range(i) > 1e-12) {
            UV.col(i) = (UV.col(i).array() - uv_min(i)) / uv_range(i);
        }
    }

    std::cout << "[cgalparam] Done." << std::endl;
    return UV;
}

} // namespace cgalparam
