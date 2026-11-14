#ifndef LMP_VERLET_H
#define LMP_VERLET_H

#include <cstddef>
#include <vector>

namespace LAMMPS_NS {

class VerletIntegrator {
 public:
  explicit VerletIntegrator(double timestep);
  void setup(std::vector<double>& force);
  void step(
      std::vector<double>& x,
      std::vector<double>& v,
      std::vector<double>& f,
      const std::vector<double>& mass) const;

 private:
  double dt_;
  double dtf_;
};

}  // namespace LAMMPS_NS

#endif
