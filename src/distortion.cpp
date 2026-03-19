#include "meshparam/distortion.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <Eigen/Dense>

namespace meshparam {

namespace {

/// Compute the 3 angles (in degrees) of a triangle given 3 vertices
Eigen::Vector3d triangle_angles(
    const Eigen::Vector3d& a,
    const Eigen::Vector3d& b,
    const Eigen::Vector3d& c)
{
    Eigen::Vector3d ab = b - a, ac = c - a;
    Eigen::Vector3d ba = a - b, bc = c - b;
    Eigen::Vector3d ca = a - c, cb = b - c;

    auto safe_angle = [](const Eigen::Vector3d& u, const Eigen::Vector3d& v) -> double {
        double cos_a = u.dot(v) / (u.norm() * v.norm());
        cos_a = std::max(-1.0, std::min(1.0, cos_a));
        return std::acos(cos_a) * 180.0 / M_PI;
    };

    return {safe_angle(ab, ac), safe_angle(ba, bc), safe_angle(ca, cb)};
}

/// Compute area of a 3D triangle
double triangle_area_3d(
    const Eigen::Vector3d& a,
    const Eigen::Vector3d& b,
    const Eigen::Vector3d& c)
{
    return 0.5 * (b - a).cross(c - a).norm();
}

/// Compute area of a 2D triangle
double triangle_area_2d(
    const Eigen::Vector2d& a,
    const Eigen::Vector2d& b,
    const Eigen::Vector2d& c)
{
    return 0.5 * std::abs((b.x() - a.x()) * (c.y() - a.y()) -
                          (c.x() - a.x()) * (b.y() - a.y()));
}

/// Compute L2 stretch of a triangle (Sander et al. 2001)
/// Returns the L2 stretch metric: sqrt((sigma1^2 + sigma2^2) / 2)
/// where sigma1, sigma2 are the singular values of the Jacobian.
/// Perfect isometry → 1.0
double triangle_stretch(
    const Eigen::Vector3d& p0, const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
    const Eigen::Vector2d& u0, const Eigen::Vector2d& u1, const Eigen::Vector2d& u2)
{
    // 3D edges
    Eigen::Vector3d e1_3d = p1 - p0;
    Eigen::Vector3d e2_3d = p2 - p0;

    // 2D edges
    Eigen::Vector2d e1_2d = u1 - u0;
    Eigen::Vector2d e2_2d = u2 - u0;

    double area_3d = 0.5 * e1_3d.cross(e2_3d).norm();
    double area_2d = 0.5 * std::abs(e1_2d.x() * e2_2d.y() - e1_2d.y() * e2_2d.x());

    if (area_3d < 1e-16 || area_2d < 1e-16) return 1.0;

    // Jacobian of the map from UV to 3D (2x3 matrix conceptually)
    // We compute singular values via the metric tensor.
    // Ss = (e1_3d · e1_3d) and similar
    double a = e1_3d.dot(e1_3d);
    double b = e1_3d.dot(e2_3d);
    double c = e2_3d.dot(e2_3d);

    // Inverse of 2D parameterization
    double det_uv = e1_2d.x() * e2_2d.y() - e1_2d.y() * e2_2d.x();
    if (std::abs(det_uv) < 1e-16) return 1.0;

    // First fundamental form in UV coordinates
    // Using the formula from Sander et al.:
    // The stretch is related to singular values of J
    // sigma^2 = eigenvalues of J^T J

    // J maps from UV to 3D local frame
    // J = [e1_3d, e2_3d] * inv([e1_2d, e2_2d])
    double inv_det = 1.0 / det_uv;
    // Column vectors of inv([e1_2d, e2_2d])
    double i00 =  e2_2d.y() * inv_det, i01 = -e2_2d.x() * inv_det;
    double i10 = -e1_2d.y() * inv_det, i11 =  e1_2d.x() * inv_det;

    // J^T J entries (2x2 symmetric matrix)
    // J col0 = e1_3d * i00 + e2_3d * i10
    // J col1 = e1_3d * i01 + e2_3d * i11
    Eigen::Vector3d j0 = e1_3d * i00 + e2_3d * i10;
    Eigen::Vector3d j1 = e1_3d * i01 + e2_3d * i11;

    double g00 = j0.dot(j0);
    double g01 = j0.dot(j1);
    double g11 = j1.dot(j1);

    // Eigenvalues of [g00 g01; g01 g11]
    double trace = g00 + g11;
    double det = g00 * g11 - g01 * g01;
    double disc = std::max(0.0, trace * trace - 4.0 * det);
    double sigma1_sq = (trace + std::sqrt(disc)) / 2.0;
    double sigma2_sq = (trace - std::sqrt(disc)) / 2.0;
    sigma1_sq = std::max(0.0, sigma1_sq);
    sigma2_sq = std::max(0.0, sigma2_sq);

    // Scale-invariant stretch: sigma_max / sigma_min (= 1.0 for conformal)
    double sigma1 = std::sqrt(std::max(0.0, sigma1_sq));
    double sigma2 = std::sqrt(std::max(0.0, sigma2_sq));
    if (sigma2 < 1e-16) return (sigma1 < 1e-16) ? 1.0 : 1e6;
    return sigma1 / sigma2;  // >= 1.0, equals 1.0 for conformal
}

} // anonymous namespace

DistortionMetrics compute_distortion(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& UV)
{
    int m = static_cast<int>(F.rows());
    DistortionMetrics metrics;

    metrics.angle_distortion.resize(m);
    metrics.area_distortion.resize(m);
    metrics.stretch.resize(m);

    // First pass: compute total areas for global scale factor
    double total_area_3d = 0;
    double total_area_uv = 0;
    for (int f = 0; f < m; ++f) {
        int v0 = F(f, 0), v1 = F(f, 1), v2 = F(f, 2);
        total_area_3d += triangle_area_3d(V.row(v0), V.row(v1), V.row(v2));
        total_area_uv += triangle_area_2d(UV.row(v0).head<2>(), UV.row(v1).head<2>(), UV.row(v2).head<2>());
    }

    for (int f = 0; f < m; ++f) {
        int v0 = F(f, 0), v1 = F(f, 1), v2 = F(f, 2);

        Eigen::Vector3d p0 = V.row(v0), p1 = V.row(v1), p2 = V.row(v2);
        Eigen::Vector2d u0 = UV.row(v0), u1 = UV.row(v1), u2 = UV.row(v2);

        // Angle distortion: max absolute angle difference
        Eigen::Vector3d angles_3d = triangle_angles(p0, p1, p2);
        Eigen::Vector3d angles_2d = triangle_angles(
            Eigen::Vector3d(u0.x(), u0.y(), 0),
            Eigen::Vector3d(u1.x(), u1.y(), 0),
            Eigen::Vector3d(u2.x(), u2.y(), 0));

        double max_angle_err = 0;
        for (int i = 0; i < 3; ++i) {
            max_angle_err = std::max(max_angle_err, std::abs(angles_3d(i) - angles_2d(i)));
        }
        metrics.angle_distortion(f) = max_angle_err;

        // Area distortion: ratio normalized by global scale
        double a3d = triangle_area_3d(p0, p1, p2);
        double a2d = triangle_area_2d(u0, u1, u2);
        double global_scale = (total_area_3d > 1e-16) ? (total_area_uv / total_area_3d) : 1.0;
        // Normalized area ratio: 1.0 = perfect, >1 = stretched, <1 = compressed
        metrics.area_distortion(f) = (a3d > 1e-16 && global_scale > 1e-16)
            ? (a2d / (a3d * global_scale)) : 1.0;

        // L2 stretch
        metrics.stretch(f) = triangle_stretch(p0, p1, p2, u0, u1, u2);
    }

    // Compute summaries
    metrics.mean_angle_distortion = metrics.angle_distortion.mean();
    metrics.max_angle_distortion = metrics.angle_distortion.maxCoeff();

    metrics.mean_area_distortion = metrics.area_distortion.mean();
    double var = (metrics.area_distortion.array() - metrics.mean_area_distortion).square().mean();
    metrics.std_area_distortion = std::sqrt(var);

    metrics.mean_stretch = metrics.stretch.mean();
    metrics.max_stretch = metrics.stretch.maxCoeff();

    metrics.isometric_rms = -1.0; // Not computed here

    return metrics;
}

double compute_isometric_distortion(
    const Eigen::MatrixXd& G,
    const Eigen::MatrixXd& UV)
{
    int n = static_cast<int>(G.rows());
    double sum_sq = 0;
    int count = 0;

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double g_ij = G(i, j);
            if (g_ij < 1e-12) continue;

            double d_ij = (UV.row(i) - UV.row(j)).norm();
            double rel_err = (g_ij - d_ij) / g_ij;
            sum_sq += rel_err * rel_err;
            count++;
        }
    }

    return (count > 0) ? std::sqrt(sum_sq / count) : 0.0;
}

void DistortionMetrics::print_summary() const {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n  === Distortion Metrics ===" << std::endl;
    std::cout << "  Angle distortion:  mean=" << mean_angle_distortion
              << "°  max=" << max_angle_distortion << "°" << std::endl;
    std::cout << "  Area distortion:   mean=" << mean_area_distortion
              << "  std=" << std_area_distortion
              << "  (1.0=perfect)" << std::endl;
    std::cout << "  L2 stretch:        mean=" << mean_stretch
              << "  max=" << max_stretch
              << "  (1.0=isometric)" << std::endl;
    if (isometric_rms >= 0) {
        std::cout << "  Isometric RMS:     " << isometric_rms
                  << "  (0.0=perfect)" << std::endl;
    }
}

} // namespace meshparam
