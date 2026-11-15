#include "fix_nvt_nh.h"
#include <cmath>

namespace LAMMPS_NS {

FixNVT::FixNVT(double t_start, double t_stop, double t_damp, double dt)
    : t_start_(t_start),
      t_stop_(t_stop),
      t_freq_(1.0 / t_damp),
      dt_(dt),
      dthalf_(0.5 * dt),
      dt4_(0.25 * dt),
      eta_dot_(0.0),
      eta_mass_(3.0 * t_start / (t_freq_ * t_freq_)) {}

void FixNVT::nh_v_temp(std::vector<double>& v, double factor_eta) const {
  for (double& comp : v) {
    comp *= factor_eta;
  }
}

void FixNVT::initial_integrate(
    std::vector<double>& x,
    std::vector<double>& v,
    const std::vector<double>& f,
    const std::vector<double>& mass,
    double current_temp) {
  const double g_ke = (current_temp - t_start_) / eta_mass_;
  eta_dot_ += dt4_ * g_ke;
  const double factor = std::exp(-dthalf_ * eta_dot_);
  nh_v_temp(v, factor);
  const std::size_t natoms = mass.size();
  for (std::size_t i = 0; i < natoms; ++i) {
    const double dtfm = dthalf_ / mass[i];
    v[3 * i + 0] += dtfm * f[3 * i + 0];
    v[3 * i + 1] += dtfm * f[3 * i + 1];
    v[3 * i + 2] += dtfm * f[3 * i + 2];
    x[3 * i + 0] += dt_ * v[3 * i + 0];
    x[3 * i + 1] += dt_ * v[3 * i + 1];
    x[3 * i + 2] += dt_ * v[3 * i + 2];
  }
}

void FixNVT::final_integrate(
    std::vector<double>& v,
    const std::vector<double>& f,
    const std::vector<double>& mass,
    double current_temp) {
  const std::size_t natoms = mass.size();
  for (std::size_t i = 0; i < natoms; ++i) {
    const double dtfm = dthalf_ / mass[i];
    v[3 * i + 0] += dtfm * f[3 * i + 0];
    v[3 * i + 1] += dtfm * f[3 * i + 1];
    v[3 * i + 2] += dtfm * f[3 * i + 2];
  }
  const double g_ke = (current_temp - t_stop_) / eta_mass_;
  eta_dot_ += dt4_ * g_ke;
  const double factor = std::exp(-dthalf_ * eta_dot_);
  nh_v_temp(v, factor);
}

}  // namespace LAMMPS_NS
