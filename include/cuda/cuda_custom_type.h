#ifndef GPUSORT_CUDA_CUSTOM_TYPE_H_
#define GPUSORT_CUDA_CUSTOM_TYPE_H_

#include <stdint.h>

namespace gpusort {

#ifdef __CUDA_COMPILER
#define SPECIFIER __host__ __device__
#else
#define SPECIFIER
#endif  // __CUDA_COMPILER

template<typename K>
class CudaCustomType {
 public:
  SPECIFIER CudaCustomType() {
  }

  SPECIFIER CudaCustomType(K key, uint64_t idx) {
    key_ = key;
    idx_ = idx;
  }

  SPECIFIER bool operator == (CudaCustomType const &other) const {
    return this->key_ == other.key_;
  }

  SPECIFIER bool operator < (CudaCustomType const &other) const {
    return this->key_ < other.key_;
  }

  SPECIFIER bool operator <= (CudaCustomType const &other) const {
    return this->key_ <= other.key_;
  }

  SPECIFIER bool operator > (CudaCustomType const &other) const {
    return this->key_ > other.key_;
  }

  SPECIFIER bool operator >= (CudaCustomType const &other) const {
    return this->key_ >= other.key_;
  }

  uint64_t idx() {
    return idx_;
  }

  void set_idx(uint64_t idx) {
    idx_ = idx;
  }

  K key() {
    return key_;
  }

  void set_key(K key) {
    key_ = key;
  }

 private:
  K key_;
  uint64_t idx_;
};

}  // namespace gpusort

#endif  // GPUSORT_CUDA_CUSTOM_TYPE_H_
