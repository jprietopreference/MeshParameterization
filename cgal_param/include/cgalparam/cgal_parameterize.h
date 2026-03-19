#pragma once

#include "types.h"
#include <Eigen/Core>
#include <string>

namespace cgalparam {

enum class ParamMethod {
    LSCM,               // Least Squares Conformal Maps (free border)
    ARAP,               // As Rigid As Possible (free border)
    DiscreteConformal,   // Discrete Conformal Map (fixed border)
    DiscreteAuthalic,    // Discrete Authalic (fixed border)
    MeanValue,           // Floater Mean Value Coordinates (fixed border)
};

std::string method_name(ParamMethod m);

/// Parameterize a CGAL Surface_mesh. Returns UV coordinates as n x 2 matrix.
/// The mesh must be a topological disk (has boundary).
Eigen::MatrixXd parameterize(SurfaceMesh& mesh, ParamMethod method);

} // namespace cgalparam
