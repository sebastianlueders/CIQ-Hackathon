#include "neighbor_bin.h"
#include <algorithm>
#include <cmath>

namespace LAMMPS_NS {

NeighborBin::NeighborBin(double box_len, double cutneighmax)
    : box_len_(box_len), cutsq_(cutneighmax * cutneighmax) {
  const int nbins = std::max(1, static_cast<int>(box_len / cutneighmax));
  mbinx_ = nbins;
  mbiny_ = nbins;
  mbinz_ = nbins;
  binhead_.assign(mbinx_ * mbiny_ * mbinz_, -1);
}

int NeighborBin::coord2bin(double px, double py, double pz) const {
  auto clamp_idx = [](double val, double span, int n) {
    int idx = static_cast<int>((val / span) * n);
    return std::clamp(idx, 0, n - 1);
  };
  const int ix = clamp_idx(px, box_len_, mbinx_);
  const int iy = clamp_idx(py, box_len_, mbiny_);
  const int iz = clamp_idx(pz, box_len_, mbinz_);
  return (iz * mbiny_ + iy) * mbinx_ + ix;
}

void NeighborBin::bin_atoms(const std::vector<double>& x, int nall) {
  std::fill(binhead_.begin(), binhead_.end(), -1);
  bins_.assign(nall, -1);
  for (int i = nall - 1; i >= 0; --i) {
    const int ibin = coord2bin(x[3 * i], x[3 * i + 1], x[3 * i + 2]);
    bins_[i] = binhead_[ibin];
    binhead_[ibin] = i;
  }
}

std::vector<std::vector<int>> NeighborBin::build_half_list(
    const std::vector<double>& x,
    int nlocal) const {
  std::vector<std::vector<int>> neigh(nlocal);
  for (int i = 0; i < nlocal; ++i) {
    const int ibin = coord2bin(x[3 * i], x[3 * i + 1], x[3 * i + 2]);
    for (int j = binhead_[ibin]; j >= 0; j = bins_[j]) {
      if (j <= i) continue;
      const double dx = x[3 * i] - x[3 * j];
      const double dy = x[3 * i + 1] - x[3 * j + 1];
      const double dz = x[3 * i + 2] - x[3 * j + 2];
      if (dx * dx + dy * dy + dz * dz < cutsq_) {
        neigh[i].push_back(j);
      }
    }
  }
  return neigh;
}

}  // namespace LAMMPS_NS
