#include <cstddef>
#include <vector>

namespace LAMMPS_NS {

class ComputeTemp {
 public:
  explicit ComputeTemp(double boltz_constant) : boltz_(boltz_constant) {}

  double compute_scalar(
      const std::vector<double>& v,
      const std::vector<double>& mass,
      int dof_removed = 3) const {
    double ke_twice = 0.0;
    const std::size_t nlocal = mass.size();
    for (std::size_t i = 0; i < nlocal; ++i) {
      const double vx = v[3 * i + 0];
      const double vy = v[3 * i + 1];
      const double vz = v[3 * i + 2];
      ke_twice += mass[i] * (vx * vx + vy * vy + vz * vz);
    }
    const int dof = static_cast<int>(3 * nlocal) - dof_removed;
    if (dof <= 0) return 0.0;
    return ke_twice / (static_cast<double>(dof) * boltz_);
  }

 private:
  double boltz_;
};

}  // namespace LAMMPS_NS
