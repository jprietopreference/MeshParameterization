#pragma once

#include <Eigen/Core>
#include <igl/doublearea.h>
#include <igl/flipped_triangles.h>
#include <cmath>
#include <algorithm>

namespace meshparam {

/// Benchmark metrics following Stein et al. 2022
/// "A Splitting Scheme for Flip-Free Distortion Energies"
struct BenchmarkMetrics {
    int flipped_triangles = 0;       // 0 = bijective (no inversions)
    double symmetric_dirichlet = 0;  // per-triangle average energy
    double l2_area_distortion = 0;   // sqrt(sum(A * |A-AW|^2))
    double linf_area_distortion = 0; // max(|A-AW|)
    int num_vertices = 0;
    int num_faces = 0;
};

/// Compute Symmetric Dirichlet energy per triangle
/// E_SD = sum_i (sigma_i^2 + 1/sigma_i^2) where sigma_i are singular values of J
inline double symmetric_dirichlet_energy(
    const Eigen::MatrixXd& V,   // 3D positions (n x 3)
    const Eigen::MatrixXi& F,   // faces (m x 3)
    const Eigen::MatrixXd& UV)  // UV coords (n x 2)
{
    int m = F.rows();
    double total = 0;
    int valid = 0;

    for (int fi = 0; fi < m; ++fi) {
        int i0 = F(fi, 0), i1 = F(fi, 1), i2 = F(fi, 2);

        // 3D triangle edges
        Eigen::Vector3d e1_3d = V.row(i1) - V.row(i0);
        Eigen::Vector3d e2_3d = V.row(i2) - V.row(i0);

        // UV triangle edges
        Eigen::Vector2d e1_uv = UV.row(i1).head<2>() - UV.row(i0).head<2>();
        Eigen::Vector2d e2_uv = UV.row(i2).head<2>() - UV.row(i0).head<2>();

        // 3D: project to local 2D frame
        double l1 = e1_3d.norm();
        if (l1 < 1e-15) continue;
        Eigen::Vector3d t1 = e1_3d / l1;
        Eigen::Vector3d n = e1_3d.cross(e2_3d);
        double area_3d = n.norm() * 0.5;
        if (area_3d < 1e-15) continue;
        n.normalize();
        Eigen::Vector3d t2 = n.cross(t1);

        // Local 2D coords of 3D triangle
        double x1 = l1, y1 = 0;
        double x2 = e2_3d.dot(t1), y2 = e2_3d.dot(t2);

        // Jacobian: J maps from 3D local frame to UV
        // [e1_uv e2_uv] = J * [e1_local e2_local]
        double det_local = x1 * y2 - x2 * y1;
        if (std::abs(det_local) < 1e-15) continue;

        double a = (e1_uv(0) * y2 - e2_uv(0) * y1) / det_local;
        double b = (-e1_uv(0) * x2 + e2_uv(0) * x1) / det_local;
        double c = (e1_uv(1) * y2 - e2_uv(1) * y1) / det_local;
        double d = (-e1_uv(1) * x2 + e2_uv(1) * x1) / det_local;

        // Singular values from 2x2 Jacobian [a b; c d]
        double s1_sq = 0.5 * (a*a + b*b + c*c + d*d +
            std::sqrt(std::max(0.0, (a*a + b*b + c*c + d*d) * (a*a + b*b + c*c + d*d) -
                4.0 * (a*d - b*c) * (a*d - b*c))));
        double s2_sq = 0.5 * (a*a + b*b + c*c + d*d -
            std::sqrt(std::max(0.0, (a*a + b*b + c*c + d*d) * (a*a + b*b + c*c + d*d) -
                4.0 * (a*d - b*c) * (a*d - b*c))));

        if (s1_sq < 1e-15 || s2_sq < 1e-15) continue;

        // Symmetric Dirichlet: s1^2 + s2^2 + 1/s1^2 + 1/s2^2
        double E = s1_sq + s2_sq + 1.0 / s1_sq + 1.0 / s2_sq;
        total += E * area_3d; // area-weighted
        valid++;
    }

    return (valid > 0) ? total / valid : 1e18;
}

/// Compute all benchmark metrics
inline BenchmarkMetrics compute_benchmark_metrics(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& UV)
{
    BenchmarkMetrics bm;
    bm.num_vertices = V.rows();
    bm.num_faces = F.rows();

    // Flipped triangles
    Eigen::VectorXi flipped = igl::flipped_triangles(UV, F);
    bm.flipped_triangles = flipped.size();

    // Symmetric Dirichlet energy
    bm.symmetric_dirichlet = symmetric_dirichlet_energy(V, F, UV);

    // Area distortion: compare 3D triangle areas vs UV triangle areas
    Eigen::VectorXd A_3d, A_uv;
    igl::doublearea(V, F, A_3d);
    A_3d *= 0.5;

    // UV areas (2D)
    Eigen::MatrixXd UV3(UV.rows(), 3);
    UV3.col(0) = UV.col(0);
    UV3.col(1) = UV.col(1);
    UV3.col(2).setZero();
    igl::doublearea(UV3, F, A_uv);
    A_uv *= 0.5;

    // Normalize so total areas match
    double total_3d = A_3d.sum();
    double total_uv = A_uv.array().abs().sum();
    if (total_uv > 1e-15 && total_3d > 1e-15) {
        A_uv *= (total_3d / total_uv);
    }

    // L2 area distortion
    Eigen::VectorXd diff = (A_3d - A_uv.array().abs().matrix()).array().abs();
    bm.l2_area_distortion = std::sqrt((A_3d.array() * diff.array().pow(2)).sum());

    // L∞ area distortion
    bm.linf_area_distortion = diff.maxCoeff();

    return bm;
}

} // namespace meshparam
