#ifndef GPUSORT_TEST_IFSTREAM_H_
#define GPUSORT_TEST_IFSTREAM_H_

#include <cstdio>
#include <string>
#include <vector>

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

class IFStream {
 public:
  IFStream(const std::string &filename) {
    handle_ = std::fopen(filename.c_str(), "rb");
  }

  ~IFStream() {
    std::fclose(handle_);
  }

  bool IsEOF() {
    return (std::feof(handle_));
  }

  template<typename T>
  void Read(T *val) {
    if (IsPrimitiveType<T>()) {  // For primitive types.
      std::fread(val, sizeof(T), 1, handle_);
    } else {  // For user-defined type.
      reinterpret_cast<Serializable*>(val)->Read(this);
    }
  }

  std::string Read() {
    std::string str = "";
    size_t len = 0;
    std::fread(&len, sizeof(len), 1, handle_);
    // We can't read to string directly, so instead, create a temporary buffer.
    if (len > 0) {
      char* buf = new char[len];
      std::fread(buf, 1, len, handle_);
      str.append(buf, len);
      delete[] buf;
    }
    return str;
  }

  template<typename T>
  void Read(uint64_t nelem, std::vector<T> *_list) {
    if (nelem == 0) return;
    std::vector<T> &list = *_list;
    list.resize(nelem);
    int64_t n_blk = nelem / BLOCK_SIZE + 1;
    if (nelem % BLOCK_SIZE == 0) n_blk--;
    int64_t read_cnt = 0;
    for (int64_t i = 1; i <= n_blk; i++) {
      int64_t num_read = (i == n_blk)? nelem - read_cnt : BLOCK_SIZE;
      fread(&list[read_cnt], sizeof(T), num_read, handle_);
      read_cnt += num_read;
    }
  }

 private:
  FILE *handle_;
};

}  // namespace gpusort

#endif  // GPUSORT_TEST_IFSTREAM_H_
