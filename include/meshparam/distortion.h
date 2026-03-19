#pragma once

#include <Eigen/Core>
#include <string>

namespace meshparam {

/// Per-triangle and global distortion metrics for evaluating
/// parameterization quality (Section 3 of the paper).
struct DistortionMetrics {
    /// Per-triangle metrics (m triangles)
    Eigen::VectorXd angle_distortion;   // conformal: max angle error per triangle (degrees)
    Eigen::VectorXd area_distortion;    // authalic: ratio of UV area / 3D area (normalized)
    Eigen::VectorXd stretch;            // L2 stretch: singular value based

    /// Global summary statistics
    double mean_angle_distortion;
    double max_angle_distortion;
    double mean_area_distortion;   // 1.0 = perfect
    double std_area_distortion;
    double mean_stretch;           // 1.0 = perfect isometry
    double max_stretch;

    /// Isometric distortion: RMS of (geodesic_dist - euclidean_dist) / geodesic_dist
    /// Sampled over all vertex pairs (or a subset for large meshes).
    double isometric_rms;

    /// Print a summary to stdout
    void print_summary() const;
};

/// Compute distortion metrics between 3D mesh (V,F) and UV parameterization.
DistortionMetrics compute_distortion(
    const Eigen::MatrixXd& V,   // n x 3 vertices
    const Eigen::MatrixXi& F,   // m x 3 faces
    const Eigen::MatrixXd& UV); // n x 2 UV coordinates

/// Compute isometric distortion using precomputed geodesic matrix.
/// iso_rms = sqrt(mean((g_ij - d_ij)^2 / g_ij^2)) over all pairs i<j
double compute_isometric_distortion(
    const Eigen::MatrixXd& G,   // n x n geodesic distances
    const Eigen::MatrixXd& UV); // n x 2 UV coordinates

} // namespace meshparam
