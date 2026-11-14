#ifndef LMP_FIX_NVT_NH_H
#define LMP_FIX_NVT_NH_H

#include <vector>

namespace LAMMPS_NS {

class FixNVT {
 public:
  FixNVT(double t_start, double t_stop, double t_damp, double dt);
  void initial_integrate(
      std::vector<double>& x,
      std::vector<double>& v,
      const std::vector<double>& f,
      const std::vector<double>& mass,
      double current_temp);
  void final_integrate(
      std::vector<double>& v,
      const std::vector<double>& f,
      const std::vector<double>& mass,
      double current_temp);

 private:
  void nh_v_temp(std::vector<double>& v, double factor_eta) const;
  double t_start_;
  double t_stop_;
  double t_freq_;
  double dt_;
  double dthalf_;
  double dt4_;
  double eta_dot_;
  double eta_mass_;
};

}  // namespace LAMMPS_NS

#endif
