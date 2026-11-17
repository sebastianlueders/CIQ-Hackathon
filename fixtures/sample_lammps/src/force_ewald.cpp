#include <cmath>
#include <vector>

namespace LAMMPS_NS {

class ForceEwald {
 public:
  ForceEwald(double g_ewald, int kmax) : g_ewald_(g_ewald), kmax_(kmax) {}

  double self_energy(const std::vector<double>& q, double qqrd2e) const {
    double qsqsum = 0.0;
    for (double charge : q) {
      qsqsum += charge * charge;
    }
    constexpr double inv_sqrt_pi = 0.5641895835477563;
    return -qqrd2e * qsqsum * g_ewald_ * inv_sqrt_pi;
  }

  double short_range_damping(double r) const {
    const double grij = g_ewald_ * r;
    const double expm2 = std::exp(-grij * grij);
    return std::erfc(grij) + 2.0 * 0.5641895835477563 * grij * expm2;
  }

 private:
  double g_ewald_;
  int kmax_;
};

}  // namespace LAMMPS_NS
