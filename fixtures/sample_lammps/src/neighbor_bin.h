#ifndef LMP_NEIGHBOR_BIN_H
#define LMP_NEIGHBOR_BIN_H

#include <vector>

namespace LAMMPS_NS {

class NeighborBin {
 public:
  NeighborBin(double box_len, double cutneighmax);
  void bin_atoms(const std::vector<double>& x, int nall);
  std::vector<std::vector<int>> build_half_list(
      const std::vector<double>& x,
      int nlocal) const;

 private:
  int coord2bin(double px, double py, double pz) const;
  double box_len_;
  double cutsq_;
  int mbinx_;
  int mbiny_;
  int mbinz_;
  std::vector<int> binhead_;
  std::vector<int> bins_;
};

}  // namespace LAMMPS_NS

#endif
