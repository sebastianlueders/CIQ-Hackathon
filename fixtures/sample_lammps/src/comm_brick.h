#ifndef LMP_COMM_BRICK_H
#define LMP_COMM_BRICK_H

#include <vector>

namespace LAMMPS_NS {

struct GhostHaloBuffer {
  int send_proc;
  int recv_proc;
  std::vector<double> packed_coords;
};

class CommBrick {
 public:
  CommBrick(int procx, int procy, int procz, double ghost_cutoff);
  std::size_t pack_forward_comm(
      const std::vector<double>& x,
      const std::vector<int>& sendlist,
      std::vector<double>& buf) const;
  void unpack_forward_comm(
      const std::vector<double>& buf,
      int first_ghost,
      int count,
      std::vector<double>& x) const;

 private:
  int procgrid_[3];
  double cutghost_;
};

}  // namespace LAMMPS_NS

#endif
