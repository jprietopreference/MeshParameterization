#include "cgalparam/distortion.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <Eigen/Dense>

namespace cgalparam {

namespace {

Eigen::Vector3d tri_angles(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c) {
    auto safe_angle = [](const Eigen::Vector3d& u, const Eigen::Vector3d& v) {
        double cos_a = std::max(-1.0, std::min(1.0, u.dot(v) / (u.norm() * v.norm())));
        return std::acos(cos_a) * 180.0 / M_PI;
    };
    return {safe_angle(b-a, c-a), safe_angle(a-b, c-b), safe_angle(a-c, b-c)};
}

double tri_area_3d(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c) {
    return 0.5 * (b-a).cross(c-a).norm();
}

double tri_area_2d(const Eigen::Vector2d& a, const Eigen::Vector2d& b, const Eigen::Vector2d& c) {
    return 0.5 * std::abs((b.x()-a.x())*(c.y()-a.y()) - (c.x()-a.x())*(b.y()-a.y()));
}

double tri_stretch(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
                   const Eigen::Vector2d& u0, const Eigen::Vector2d& u1, const Eigen::Vector2d& u2) {
    Eigen::Vector3d e1 = p1-p0, e2 = p2-p0;
    Eigen::Vector2d f1 = u1-u0, f2 = u2-u0;
    double det = f1.x()*f2.y() - f1.y()*f2.x();
    if (std::abs(det) < 1e-16) return 1.0;
    double inv = 1.0/det;
    Eigen::Vector3d j0 = e1*(f2.y()*inv) + e2*(-f1.y()*inv);
    Eigen::Vector3d j1 = e1*(-f2.x()*inv) + e2*(f1.x()*inv);
    double g00 = j0.dot(j0), g01 = j0.dot(j1), g11 = j1.dot(j1);
    double tr = g00+g11, det2 = g00*g11-g01*g01;
    double disc = std::max(0.0, tr*tr - 4*det2);
    double s1 = std::sqrt(std::max(0.0, (tr+std::sqrt(disc))/2));
    double s2 = std::sqrt(std::max(0.0, (tr-std::sqrt(disc))/2));
    return (s2 < 1e-16) ? ((s1 < 1e-16) ? 1.0 : 1e6) : s1/s2;
}

} // anonymous namespace

DistortionMetrics compute_distortion(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXd& UV) {
    int m = static_cast<int>(F.rows());
    DistortionMetrics met{};

    double total_a3d = 0, total_a2d = 0;
    for (int f = 0; f < m; ++f) {
        total_a3d += tri_area_3d(V.row(F(f,0)), V.row(F(f,1)), V.row(F(f,2)));
        total_a2d += tri_area_2d(UV.row(F(f,0)).head<2>(), UV.row(F(f,1)).head<2>(), UV.row(F(f,2)).head<2>());
    }
    double gscale = (total_a3d > 1e-16) ? (total_a2d / total_a3d) : 1.0;

    double sum_angle = 0, max_angle = 0;
    double sum_area = 0, sum_area_sq = 0;
    double sum_stretch = 0, max_stretch = 0;

    for (int f = 0; f < m; ++f) {
        Eigen::Vector3d p0=V.row(F(f,0)), p1=V.row(F(f,1)), p2=V.row(F(f,2));
        Eigen::Vector2d u0=UV.row(F(f,0)), u1=UV.row(F(f,1)), u2=UV.row(F(f,2));

        auto a3 = tri_angles(p0,p1,p2);
        auto a2 = tri_angles({u0.x(),u0.y(),0},{u1.x(),u1.y(),0},{u2.x(),u2.y(),0});
        double ae = 0;
        for (int i=0;i<3;++i) ae = std::max(ae, std::abs(a3(i)-a2(i)));
        sum_angle += ae; max_angle = std::max(max_angle, ae);

        double ar3 = tri_area_3d(p0,p1,p2), ar2 = tri_area_2d(u0,u1,u2);
        double ar = (ar3>1e-16 && gscale>1e-16) ? ar2/(ar3*gscale) : 1.0;
        sum_area += ar; sum_area_sq += ar*ar;

        double s = tri_stretch(p0,p1,p2,u0,u1,u2);
        sum_stretch += s; max_stretch = std::max(max_stretch, s);
    }

    met.mean_angle_distortion = sum_angle / m;
    met.max_angle_distortion = max_angle;
    met.mean_area_distortion = sum_area / m;
    double var = sum_area_sq/m - (sum_area/m)*(sum_area/m);
    met.std_area_distortion = std::sqrt(std::max(0.0, var));
    met.mean_stretch = sum_stretch / m;
    met.max_stretch = max_stretch;
    return met;
}

void DistortionMetrics::print_summary() const {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n  === Distortion Metrics ===" << std::endl;
    std::cout << "  Angle distortion:  mean=" << mean_angle_distortion
              << "°  max=" << max_angle_distortion << "°" << std::endl;
    std::cout << "  Area distortion:   mean=" << mean_area_distortion
              << "  std=" << std_area_distortion << "  (1.0=perfect)" << std::endl;
    std::cout << "  L2 stretch:        mean=" << mean_stretch
              << "  max=" << max_stretch << "  (1.0=isometric)" << std::endl;
}

} // namespace cgalparam
