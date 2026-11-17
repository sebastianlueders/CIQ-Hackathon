#include "verlet.h"
#include <algorithm>

namespace LAMMPS_NS {

VerletIntegrator::VerletIntegrator(double timestep)
    : dt_(timestep), dtf_(0.5 * timestep) {}

void VerletIntegrator::setup(std::vector<double>& force) {
  std::fill(force.begin(), force.end(), 0.0);
}

void VerletIntegrator::step(
    std::vector<double>& x,
    std::vector<double>& v,
    std::vector<double>& f,
    const std::vector<double>& mass) const {
  const std::size_t nlocal = mass.size();
  for (std::size_t i = 0; i < nlocal; ++i) {
    const double inv_m = dtf_ / mass[i];
    v[3 * i + 0] += inv_m * f[3 * i + 0];
    v[3 * i + 1] += inv_m * f[3 * i + 1];
    v[3 * i + 2] += inv_m * f[3 * i + 2];
    x[3 * i + 0] += dt_ * v[3 * i + 0];
    x[3 * i + 1] += dt_ * v[3 * i + 1];
    x[3 * i + 2] += dt_ * v[3 * i + 2];
  }
}

}  // namespace LAMMPS_NS
