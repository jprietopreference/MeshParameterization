#pragma once

// Kernel selection:
// - Native (CGALPARAM_NATIVE): EPICK with GMP — robust predicates
// - WASM / restricted: Simple_cartesian<double> — no GMP dependency
#if defined(CGALPARAM_NATIVE)
    #include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
    using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
#else
    #include <CGAL/Simple_cartesian.h>
    using Kernel = CGAL::Simple_cartesian<double>;
#endif

#include <CGAL/Surface_mesh.h>

namespace cgalparam {

using SurfaceMesh = CGAL::Surface_mesh<Kernel::Point_3>;
using halfedge_descriptor = SurfaceMesh::Halfedge_index;
using vertex_descriptor = SurfaceMesh::Vertex_index;
using face_descriptor = SurfaceMesh::Face_index;

} // namespace cgalparam
