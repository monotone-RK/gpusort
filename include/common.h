#ifndef GPUSORT_COMMON_H_
#define GPUSORT_COMMON_H_

#include <typeinfo>

#if defined(__USING_CPP_0X) || defined(__USING_CPP_11)
#include <cstdint>
#include <cinttypes>
#include <random>
#else
#include <cstdlib>
#include <ctime>
#include <inttypes.h>
#include <stdint.h>
#endif  // __USING_CPP_0X || __USING_CPP_11

#include <omp.h>

namespace gpusort {

#define ARRANGE_VALUES_1(_values) \
  ParUtils::ArrangeValues(comm, g_indexes, old_sizes, _values);

#define ARRANGE_VALUES_1_EX(_values) \
  {\
    int dummy[] = {0, (ParUtils::ArrangeValues(comm, g_indexes, old_sizes,\
                                               std::forward<Vectors*>(_values)),\
                       0)...};\
  }

#define ARRANGE_VALUES_2(_values1, _values2) \
  ARRANGE_VALUES_1(_values1);\
  ARRANGE_VALUES_1(_values2);\

#define ARRANGE_VALUES_3(_values1, _values2, _values3)\
  ARRANGE_VALUES_1(_values1);\
  ARRANGE_VALUES_1(_values2);\
  ARRANGE_VALUES_1(_values3);

#define ARRANGE_VALUES_4(_values1, _values2, _values3, _values4)\
  ARRANGE_VALUES_1(_values1);\
  ARRANGE_VALUES_1(_values2);\
  ARRANGE_VALUES_1(_values3);\
  ARRANGE_VALUES_1(_values4);

#define ARRANGE_VALUES_5(_values1, _values2, _values3, _values4, _values5)\
  ARRANGE_VALUES_1(_values1);\
  ARRANGE_VALUES_1(_values2);\
  ARRANGE_VALUES_1(_values3);\
  ARRANGE_VALUES_1(_values4);\
  ARRANGE_VALUES_1(_values5);

#define ARRANGE_VALUES_6(_values1, _values2, _values3, _values4, _values5,\
                         _values6)\
  ARRANGE_VALUES_1(_values1);\
  ARRANGE_VALUES_1(_values2);\
  ARRANGE_VALUES_1(_values3);\
  ARRANGE_VALUES_1(_values4);\
  ARRANGE_VALUES_1(_values5);\
  ARRANGE_VALUES_1(_values6);

#define ARRANGE_VALUES_7(_values1, _values2, _values3, _values4, _values5,\
                         _values6, _values7)\
  ARRANGE_VALUES_1(_values1);\
  ARRANGE_VALUES_1(_values2);\
  ARRANGE_VALUES_1(_values3);\
  ARRANGE_VALUES_1(_values4);\
  ARRANGE_VALUES_1(_values5);\
  ARRANGE_VALUES_1(_values6);\
  ARRANGE_VALUES_1(_values7);

#define ARRANGE_VALUES_8(_values1, _values2, _values3, _values4, _values5,\
                         _values6, _values7, _values8)\
  ARRANGE_VALUES_1(_values1);\
  ARRANGE_VALUES_1(_values2);\
  ARRANGE_VALUES_1(_values3);\
  ARRANGE_VALUES_1(_values4);\
  ARRANGE_VALUES_1(_values5);\
  ARRANGE_VALUES_1(_values6);\
  ARRANGE_VALUES_1(_values7);\
  ARRANGE_VALUES_1(_values8);

#define ARRANGE_VALUES_9(_values1, _values2, _values3, _values4, _values5,\
                         _values6, _values7, _values8, _values9)\
  ARRANGE_VALUES_1(_values1);\
  ARRANGE_VALUES_1(_values2);\
  ARRANGE_VALUES_1(_values3);\
  ARRANGE_VALUES_1(_values4);\
  ARRANGE_VALUES_1(_values5);\
  ARRANGE_VALUES_1(_values6);\
  ARRANGE_VALUES_1(_values7);\
  ARRANGE_VALUES_1(_values8);\
  ARRANGE_VALUES_1(_values9);

#define ARRANGE_VALUES_10(_values1, _values2, _values3, _values4, _values5,\
                          _values6, _values7, _values8, _values9, _values10)\
  ARRANGE_VALUES_1(_values1);\
  ARRANGE_VALUES_1(_values2);\
  ARRANGE_VALUES_1(_values3);\
  ARRANGE_VALUES_1(_values4);\
  ARRANGE_VALUES_1(_values5);\
  ARRANGE_VALUES_1(_values6);\
  ARRANGE_VALUES_1(_values7);\
  ARRANGE_VALUES_1(_values8);\
  ARRANGE_VALUES_1(_values9);\
  ARRANGE_VALUES_1(_values10);

#define CAT(A, B) A##B

#define SELECT(name, num) CAT(name##_, num)

#define GET_COUNT(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, count, ...) count

#define VA_SIZE(...) GET_COUNT(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)

#define VA_SELECT(name, ...) SELECT(name, VA_SIZE(__VA_ARGS__))(__VA_ARGS__)

#define ARRANGE_VALUES(...)\
  VA_SELECT(ARRANGE_VALUES, __VA_ARGS__)\

#define OMP_PARALLEL_FOR _Pragma("omp parallel for")

#define FOR_PARALLEL(max_threads, n_loops, it, stmts) \
  FOR_PARALLEL_COND(max_threads, n_loops, it, true, stmts)

#define FOR_PARALLEL_COND(max_threads, n_loops, it, cond, stmts) \
  { \
    int64_t it##_each = 1;\
    int it##_threads = n_loops;\
    if (n_loops > max_threads) {\
      it##_each = gpusort::CalcNumElemsEachBlock(n_loops, max_threads);\
      it##_threads = max_threads;\
    }\
    OMP_PARALLEL_FOR \
    for (int it##t = 1; it##t <= it##_threads; it##t++) {\
      int64_t it##_start = (it##t-1) * it##_each;\
      int64_t it##_end = (it##t == it##_threads)? n_loops-1\
                                                  : it##_start + it##_each - 1;\
      for (int64_t it = it##_start; (it <= it##_end) && cond; it++) {\
        stmts;\
      }\
    }\
  }

template<typename Int>
inline Int CalcNumElemsEachBlock(Int total, int p) {
  int r = total % p;
  if (r != 0) {
    Int tmp = (total + (p - r)) / p;
    return (tmp * (p-1) >= total)? tmp-1 : tmp;
  } else {
    return total / p;
  }
}

template<typename T, typename K>
inline std::less<T> ConvertComp(std::less<K> comp) {
  return std::less<T>();
}

template<typename T, typename K>
inline std::greater<T> ConvertComp(std::greater<K> comp) {
  return std::greater<T>();
}

inline void* GetAddress(void *buf, uint64_t idx, size_t obj_size) {
  return reinterpret_cast<int8_t*>(buf) + idx * obj_size;
}

// Compute the next highest power of 2 of 32-bit v.
inline int GetNextHighestPowerOfTwo(unsigned int v) {
  v--;
  v |= (v >> 1);
  v |= (v >> 2);
  v |= (v >> 4);
  v |= (v >> 8);
  v |= (v >> 16);
  v++;
  return v;
}

//
// Compute the prev highest power of 2 of 32-bit v.
//
inline int GetPrevHighestPowerOfTwo(unsigned int v) {
  v--;
  v |= (v >> 1);
  v |= (v >> 2);
  v |= (v >> 4);
  v |= (v >> 8);
  v |= (v >> 16);
  v++;
  return (v >> 1);
}

template<typename Int>
inline void InitGlobalIndexes(Int begin, Int nelem, Int* global_indexes) {
  int n_threads = omp_get_max_threads();
  FOR_PARALLEL(n_threads, nelem, i,
               {
                 global_indexes[i] = begin + i;
               });
}

template<typename T>
inline bool IsIncreasing(std::less<T> comp) {
  return true;
}

template<typename T>
inline bool IsIncreasing(std::greater<T> comp) {
  return false;
}

inline bool IsPowerOfTwo(unsigned int n) {
  return (n && (!(n & (n - 1))));
}

template<typename T>
inline bool IsPrimitiveType() {
  const std::type_info& t = typeid(T);
  if (t == typeid(short)) {
    return true;
  } else if (t == typeid(int)) {
    return true;
  } else if (t == typeid(long)) {
    return true;
  } else if (t == typeid(unsigned short)) {
    return true;
  } else if (t == typeid(unsigned int)) {
    return true;
  } else if (t == typeid(unsigned long)) {
    return true;
  } else if (t == typeid(float)) {
    return true;
  } else if (t == typeid(double)) {
    return true;
  } else if (t == typeid(long double)) {
    return true;
  } else if (t == typeid(long long)) {
    return true;
  } else if (t == typeid(char)) {
    return true;
  } else if (t == typeid(unsigned char)) {
    return true;
  }
  return false;
}

}  // namespace gpusort

#endif  // GPUSORT_COMMON_H_
