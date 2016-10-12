#ifndef GPUSORT_RANDOM_H_
#define GPUSORT_RANDOM_H_

#include <algorithm>

#if defined(__USING_CPP_11)
#include <random>
#else
#include <cstdlib>
#include <ctime>
#endif  // __USING_CPP_11

#include <omp.h>

#include "common.h"
#include "omp/omp_utils.h"

namespace gpusort {

class Random {
 public:
  Random() {
  #if defined(__USING_CPP_11)
    g_engine_ = (std::random_device{}());
  #else
    std::srand(unsigned(std::time(0)));
  #endif  // __USING_CPP_11
  }

  Random(unsigned int seed) {
  #if defined(__USING_CPP_11)
    g_engine_ = (seed);
  #else
    std::srand(seed);
  #endif  // __USING_CPP_11
  }

  template<typename T>
  inline T Next(T min, T max) {
  #if defined(__USING_CPP_11)
    if (std::is_integral<T>::value) {
      std::uniform_int_distribution<T> distribution(min, max);
      return distribution(g_engine_);
    } else {
      std::uniform_real_distribution<T> distribution(min, max);
      return distribution(g_engine_);
    }
  #else
    T range = max - min;
    return min + static_cast<T>(range * (static_cast <double>(std::rand()) /
                                         static_cast <double>(RAND_MAX)));
  #endif  // __USING_CPP_11
  }

#if defined(__USING_CPP_11)
  template<typename Int, typename Distribution, typename T>
  inline void Next(Int count, const Distribution &distribution, T *out) {
    int n_threads = OmpUtils::GetMaxThreads();
    FOR_PARALLEL(n_threads, count, i,
                 {
                   out[i] = distribution(g_engine_);
                 });
  }
#endif  // __USING_CPP_11

  //
  // Generate a vector of count random values on:
  //   + the interval [min, max] with integral values.
  //   + the interval [min, max) with floating-point values.
  //
  template<typename T, typename Int>
  inline void Next(T min, T max, Int count, T *out) {
  #if defined(__USING_CPP_11)
    if (std::is_integral<T>::value) {
      std::uniform_int_distribution<T> distribution(min, max);
      Next(count, distribution, out);
    } else {
      std::uniform_real_distribution<T> distribution(min, max);
      Next(count, distribution, out);
    }
  #else
    T range = max - min;
    int n_threads = OmpUtils::GetMaxThreads();
    FOR_PARALLEL(
        n_threads, count, i,
        {
          out[i] = Next(min, max);
        });
  #endif  // __USING_CPP_11
  }

  template<typename RandomAccessIterator>
  inline void Shuffle(RandomAccessIterator first, RandomAccessIterator last) {
  #if defined(__USING_CPP_11)
    std::shuffle(first, last, g_engine_);
  #else
    std::random_shuffle(first, last);
  #endif
  }

 private:
#if defined(__USING_CPP_11)
  std::default_random_engine g_engine_;
#endif
};

}  // namespace gpusort

#endif  // GPUSORT_RANDOM_H_
