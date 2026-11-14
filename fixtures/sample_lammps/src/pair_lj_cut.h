#ifndef LMP_PAIR_LJ_CUT_H
#define LMP_PAIR_LJ_CUT_H

#include <vector>
#include <cmath>

namespace LAMMPS_NS {

class PairLJCut {
 public:
  explicit PairLJCut(double global_cutoff);
  ~PairLJCut() = default;

  void set_coeff(int i, int j, double epsilon, double sigma, double cutoff);
  double compute_pair_energy(
      const std::vector<double>& x,
      const std::vector<int>& type,
      const std::vector<std::vector<int>>& firstneigh,
      std::vector<double>& force) const;

 private:
  int max_types_;
  double cut_global_;
  std::vector<double> lj1_;
  std::vector<double> lj2_;
  std::vector<double> lj3_;
  std::vector<double> lj4_;
  std::vector<double> cutsq_;
  std::vector<double> offset_;
};

}  // namespace LAMMPS_NS

#endif
