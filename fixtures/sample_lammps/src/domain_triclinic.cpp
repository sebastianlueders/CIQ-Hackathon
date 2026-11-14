#include <array>
#include <cmath>

namespace LAMMPS_NS {

class DomainTriclinic {
 public:
  DomainTriclinic(std::array<double, 3> lo, std::array<double, 3> hi, std::array<double, 3> tilt)
      : boxlo_(lo), boxhi_(hi), tilt_(tilt) {}

  void x2lamda(const std::array<double, 3>& x, std::array<double, 3>& lamda) const {
    const double lx = boxhi_[0] - boxlo_[0];
    const double ly = boxhi_[1] - boxlo_[1];
    const double lz = boxhi_[2] - boxlo_[2];
    const double dz = x[2] - boxlo_[2];
    const double dy = x[1] - boxlo_[1] - tilt_[2] * (dz / lz);
    const double dx = x[0] - boxlo_[0] - tilt_[1] * (dz / lz) - tilt_[0] * (dy / ly);
    lamda[0] = dx / lx;
    lamda[1] = dy / ly;
    lamda[2] = dz / lz;
  }

  void minimum_image(double& dx, double& dy, double& dz) const {
    const double lx = boxhi_[0] - boxlo_[0];
    const double ly = boxhi_[1] - boxlo_[1];
    const double lz = boxhi_[2] - boxlo_[2];
    if (std::fabs(dz) > 0.5 * lz) dz -= std::copysign(lz, dz);
    if (std::fabs(dy) > 0.5 * ly) dy -= std::copysign(ly, dy);
    if (std::fabs(dx) > 0.5 * lx) dx -= std::copysign(lx, dx);
  }

 private:
  std::array<double, 3> boxlo_;
  std::array<double, 3> boxhi_;
  std::array<double, 3> tilt_;
};

}  // namespace LAMMPS_NS
