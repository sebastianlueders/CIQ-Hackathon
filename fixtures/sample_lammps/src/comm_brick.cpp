#include "comm_brick.h"

namespace LAMMPS_NS {

CommBrick::CommBrick(int procx, int procy, int procz, double ghost_cutoff)
    : procgrid_{procx, procy, procz}, cutghost_(ghost_cutoff) {}

std::size_t CommBrick::pack_forward_comm(
    const std::vector<double>& x,
    const std::vector<int>& sendlist,
    std::vector<double>& buf) const {
  buf.resize(sendlist.size() * 3);
  std::size_t m = 0;
  for (int atom_idx : sendlist) {
    buf[m++] = x[3 * atom_idx + 0];
    buf[m++] = x[3 * atom_idx + 1];
    buf[m++] = x[3 * atom_idx + 2];
  }
  return m;
}

void CommBrick::unpack_forward_comm(
    const std::vector<double>& buf,
    int first_ghost,
    int count,
    std::vector<double>& x) const {
  std::size_t m = 0;
  for (int i = 0; i < count; ++i) {
    const int dst = first_ghost + i;
    x[3 * dst + 0] = buf[m++];
    x[3 * dst + 1] = buf[m++];
    x[3 * dst + 2] = buf[m++];
  }
}

}  // namespace LAMMPS_NS
