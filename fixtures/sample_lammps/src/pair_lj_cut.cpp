#include "pair_lj_cut.h"

namespace LAMMPS_NS {

PairLJCut::PairLJCut(double global_cutoff)
    : max_types_(8),
      cut_global_(global_cutoff),
      lj1_(64, 0.0),
      lj2_(64, 0.0),
      lj3_(64, 0.0),
      lj4_(64, 0.0),
      cutsq_(64, global_cutoff * global_cutoff),
      offset_(64, 0.0) {}

void PairLJCut::set_coeff(int i, int j, double epsilon, double sigma, double cutoff) {
  const int idx = i * max_types_ + j;
  const double sig2 = sigma * sigma;
  const double sig6 = sig2 * sig2 * sig2;
  const double sig12 = sig6 * sig6;
  lj1_[idx] = 48.0 * epsilon * sig12;
  lj2_[idx] = 24.0 * epsilon * sig6;
  lj3_[idx] = 4.0 * epsilon * sig12;
  lj4_[idx] = 4.0 * epsilon * sig6;
  cutsq_[idx] = cutoff * cutoff;
  const double ratio = sigma / cutoff;
  const double r6 = ratio * ratio * ratio * ratio * ratio * ratio;
  offset_[idx] = 4.0 * epsilon * (r6 * r6 - r6);
}

double PairLJCut::compute_pair_energy(
    const std::vector<double>& x,
    const std::vector<int>& type,
    const std::vector<std::vector<int>>& firstneigh,
    std::vector<double>& force) const {
  double evdwl = 0.0;
  const int nlocal = static_cast<int>(firstneigh.size());
  for (int i = 0; i < nlocal; ++i) {
    const double xtmp = x[3 * i + 0];
    const double ytmp = x[3 * i + 1];
    const double ztmp = x[3 * i + 2];
    const int itype = type[i];
    for (int j : firstneigh[i]) {
      const double delx = xtmp - x[3 * j + 0];
      const double dely = ytmp - x[3 * j + 1];
      const double delz = ztmp - x[3 * j + 2];
      const double rsq = delx * delx + dely * dely + delz * delz;
      const int idx = itype * max_types_ + type[j];
      if (rsq < cutsq_[idx] && rsq > 1e-12) {
        const double r2inv = 1.0 / rsq;
        const double r6inv = r2inv * r2inv * r2inv;
        const double forcelj = r6inv * (lj1_[idx] * r6inv - lj2_[idx]);
        const double fpair = forcelj * r2inv;
        force[3 * i + 0] += delx * fpair;
        force[3 * i + 1] += dely * fpair;
        force[3 * i + 2] += delz * fpair;
        force[3 * j + 0] -= delx * fpair;
        force[3 * j + 1] -= dely * fpair;
        force[3 * j + 2] -= delz * fpair;
        evdwl += r6inv * (lj3_[idx] * r6inv - lj4_[idx]) - offset_[idx];
      }
    }
  }
  return evdwl;
}

}  // namespace LAMMPS_NS
