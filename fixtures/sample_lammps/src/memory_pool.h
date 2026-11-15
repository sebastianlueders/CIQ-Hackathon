#ifndef LMP_MEMORY_POOL_H
#define LMP_MEMORY_POOL_H

#include <cstdlib>
#include <new>

namespace LAMMPS_NS {

template <typename Scalar>
class AlignedMemoryPool {
 public:
  static constexpr std::size_t kAlignment = 64;

  static Scalar* allocate(std::size_t count) {
    if (count == 0) return nullptr;
    const std::size_t bytes = ((count * sizeof(Scalar) + kAlignment - 1) / kAlignment) * kAlignment;
    void* ptr = std::aligned_alloc(kAlignment, bytes);
    if (!ptr) throw std::bad_alloc();
    return static_cast<Scalar*>(ptr);
  }

  static void release(Scalar* ptr) noexcept {
    std::free(ptr);
  }
};

}  // namespace LAMMPS_NS

#endif
