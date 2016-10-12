#ifndef GPUSORT_TEST_OFSTREAM_H_
#define GPUSORT_TEST_OFSTREAM_H_

#include <cstdio>
#include <iterator>
#include <string>

#if defined(__USING_CPP_0X) || defined(__USING_CPP_11)
#include <cstdint>
#else
#include <stdint.h>
#endif  // __USING_CPP_0X || __USING_CPP_11

#include "common.h"
#include "serializable.h"

namespace gpusort {

#define BLOCK_SIZE 0xC80000  // 100MB of doubles.

class Serializable;

class OFStream {
 public:
  OFStream(const std::string &filename) {
    handle_ = std::fopen(filename.c_str(), "wb");
  }

  ~OFStream() {
    std::fclose(handle_);
  }

  template<typename T>
  inline void Write(const T &val) {
    if (IsPrimitiveType<T>()) {  // For primitive types.
      std::fwrite(&val, sizeof(T), 1, handle_);
    } else {  // For user-defined type.
      reinterpret_cast<Serializable*>(const_cast<T*>(&val))->Write(this);
    }
  }

  inline void Write(const std::string &str) {
    size_t len = str.size();
    std::fwrite(&len, sizeof(len), 1, handle_);
    // Write the actual string data.
    std::fwrite(str.c_str(), 1, len, handle_);
  }

  template<typename RandomAccessIterator>
  inline void Write(RandomAccessIterator first, RandomAccessIterator last) {
    typedef typename std::iterator_traits<RandomAccessIterator>::value_type T;

    int64_t total = last - first;
    int64_t n_blk = total / BLOCK_SIZE + 1;
    if (total % BLOCK_SIZE == 0) n_blk--;
    for (int64_t i = 1; i <= n_blk; i++) {
      int64_t i1 = (i-1) * BLOCK_SIZE;
      int64_t i2 = (i == n_blk)? total : i1 + BLOCK_SIZE;
      int64_t count = i2 - i1;
      fwrite(first + i1, sizeof(T), count, handle_);
    }
  }

 private:
  FILE *handle_;
};

}  // namespace gpusort

#endif  // GPUSORT_TEST_OFSTREAM_H_
