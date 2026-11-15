#include <cstddef>
#include <vector>

namespace LAMMPS_NS {

class AtomVecAtomic {
 public:
  void grow(std::size_t nmax) {
    tag_.resize(nmax, 0);
    type_.resize(nmax, 1);
    x_.resize(nmax * 3, 0.0);
    v_.resize(nmax * 3, 0.0);
    f_.resize(nmax * 3, 0.0);
  }

  void copy_atom(std::size_t src, std::size_t dst) {
    tag_[dst] = tag_[src];
    type_[dst] = type_[src];
    for (std::size_t d = 0; d < 3; ++d) {
      x_[3 * dst + d] = x_[3 * src + d];
      v_[3 * dst + d] = v_[3 * src + d];
      f_[3 * dst + d] = f_[3 * src + d];
    }
  }

 private:
  std::vector<int> tag_;
  std::vector<int> type_;
  std::vector<double> x_;
  std::vector<double> v_;
  std::vector<double> f_;
};

}  // namespace LAMMPS_NS
