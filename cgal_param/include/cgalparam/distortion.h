#pragma once

#include <Eigen/Core>

namespace cgalparam {

struct DistortionMetrics {
    double mean_angle_distortion;
    double max_angle_distortion;
    double mean_area_distortion;
    double std_area_distortion;
    double mean_stretch;
    double max_stretch;

    void print_summary() const;
};

DistortionMetrics compute_distortion(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& UV);

} // namespace cgalparam
