#include <climits>
#include <functional>
#include <vector>

#if defined(__USING_CPP_0X) || defined(__USING_CPP_11)
#include <cstdint>
#include <cinttypes>
#else
#include <inttypes.h>
#include <stdint.h>
#endif  // __USING_CPP_0X || __USING_CPP_11

#include "common.h"
#include "exceptions/mpi_exception.h"
#include "exceptions/thrust_exception.h"
#include "helpers/hyksort_helper.h"
#include "mpi/mpi_common.h"
#include "mpi/mpi_utils.h"
#include "par/par_utils.h"
#include "samplesort.h"

#ifdef __DO_BENCHMARK
#include "timer.h"
#endif

namespace gpusort {

// Check whether the number of processes and k_way value are valid
// to be used with the HykSort algorithm. Otherwise, we will use
// the SampleSort algorithm instead.
#define HYKSORT_CHECK(comm, k_way, ...) \
  { \
    int npes = MpiUtils::MpiCommSize(comm);\
    if (!IsPowerOfTwo(npes) || !IsPowerOfTwo(k_way)) {\
      int myrank = MpiUtils::MpiCommRank(comm);\
      if (myrank == 0)\
        std::cout << "The number of processes or k_way value "\
                  << "is not a power of 2. We are using SampleSort "\
                  << "algorithm instead!" << std::endl;\
      SampleSort(comm, __VA_ARGS__);\
      return;\
    }\
  }

#define HYKSORT_WITH_GLOBAL_INDEXES_T(int_t, comm, comp, k_way,\
                                      is_using_gpu, _keys)\
  std::vector<int_t> g_indexes;\
  std::vector<int_t> old_sizes;\
  int_t size = _keys->size();\
  int myrank = MpiUtils::MpiCommRank(comm);\
  ParUtils::GatherNElems(comm, size, &old_sizes);\
  /* Sort the keys and get their global indexes' changes.*/\
  HykSortWithGlobalIndexes(comm, comp, k_way, is_using_gpu,\
                           old_sizes, _keys, &g_indexes);

#define HYKSORT_WITH_GLOBAL_INDEXES(comm, comp, k_way, is_using_gpu, _keys)\
  HYKSORT_WITH_GLOBAL_INDEXES_T(int64_t, comm, comp, k_way,\
                                is_using_gpu, _keys);

template<typename Vector>
void HykSort(MpiComm comm, unsigned int k_way, bool is_using_gpu,
             Vector *_keys) throw(MpiException, ThrustException) {
  typedef typename Vector::value_type T;
  HykSort(comm, std::less<T>(), k_way, is_using_gpu, _keys);
}

template<typename StrictWeakOrdering, typename Vector>
void HykSort(MpiComm comm, StrictWeakOrdering comp,
             unsigned int k_way, bool is_using_gpu,
             Vector *_keys) throw(MpiException, ThrustException) {
  HYKSORT_CHECK(comm, k_way, comp, _keys);

  HykSortHelper helper(comm, k_way, is_using_gpu);
  helper.Sort(comp, _keys);

#ifdef __DO_BENCHMARK
  helper.ShowBenchmark();
#endif
}

template<typename KeyType, typename KeyFunc, typename Vector>
void HykSort(MpiComm comm, unsigned int k_way,
             bool is_using_gpu, KeyFunc key_func,
             Vector *_list) throw(MpiException, ThrustException) {
  typedef typename Vector::value_type T;
  HykSort<KeyType>(comm, std::less<T>(), k_way, is_using_gpu, key_func, _list);
}

template<typename KeyType, typename StrictWeakOrdering,
         typename KeyFunc, typename Vector>
void HykSort(MpiComm comm, StrictWeakOrdering comp,
             unsigned int k_way, bool is_using_gpu, KeyFunc key_func,
             Vector *_list) throw(MpiException, ThrustException) {
  HYKSORT_CHECK(comm, k_way, comp, _list);

  HykSortHelper helper(comm, k_way, is_using_gpu);
  helper.Sort<KeyType>(comp, key_func, _list);

#ifdef __DO_BENCHMARK
  helper.ShowBenchmark();
#endif
}

template<typename VectorK, typename VectorV>
void HykSort(MpiComm comm, unsigned int k_way,
             bool is_using_gpu, VectorK *_keys,
             VectorV *_values) throw(MpiException, ThrustException) {
  typedef typename VectorK::value_type K;
  HykSort(comm, std::less<K>(), k_way, is_using_gpu, _keys, _values);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV>
void HykSort(MpiComm comm, StrictWeakOrdering comp, unsigned int k_way,
             bool is_using_gpu, VectorK *_keys,
             VectorV *_values) throw(MpiException, ThrustException) {
  HYKSORT_CHECK(comm, k_way, comp, _keys, _values);

  HykSortHelper helper(comm, k_way, is_using_gpu);
  helper.SortByKey(comp, _keys, _values);

#ifdef __DO_BENCHMARK
  helper.ShowBenchmark();
#endif
}

#if defined(__USING_CPP_0X) || defined(__USING_CPP_11)
template<typename VectorK, typename... Vectors>
void HykSort(
    MpiComm comm, unsigned int k_way,
    bool is_using_gpu, VectorK *_keys,
    Vectors*... _values) throw(MpiException, ThrustException) {
  typedef typename VectorK::value_type K;
  HykSort(comm, std::less<K>(), k_way, is_using_gpu,
          _keys, _values...);
}

template<typename StrictWeakOrdering, typename VectorK,
         typename... Vectors>
void HykSort(
    MpiComm comm, StrictWeakOrdering comp, unsigned int k_way,
    bool is_using_gpu, VectorK *_keys,
    Vectors*... _values) throw(MpiException, ThrustException) {
  HYKSORT_CHECK(comm, k_way, comp, _keys, _values...);

  // Using keys to sort global indexes.
  HYKSORT_WITH_GLOBAL_INDEXES(comm, comp, k_way, is_using_gpu, _keys);
  // Using the sorted global indexes to re-arrange other values.
#ifdef __DO_BENCHMARK
  Timer arrange_values_tm;
  arrange_values_tm.Start();
#endif

  ARRANGE_VALUES_1_EX(_values);

#ifdef __DO_BENCHMARK
  arrange_values_tm.Stop();
#endif

#ifdef __DO_BENCHMARK
  ParUtils::ShowBenchmark(comm, arrange_values_tm, "Arranging values");
#endif
}

#else  // !__USING_CPP_0X && !__USING_CPP_11
template<typename VectorK, typename VectorV1, typename VectorV2>
void HykSort(MpiComm comm, unsigned int k_way, bool is_using_gpu,
             VectorK *_keys, VectorV1 *_values1,
             VectorV2 *_values2) throw(MpiException, ThrustException) {
  typedef typename VectorK::value_type K;
  HykSort(comm, std::less<K>(), k_way, is_using_gpu,
          _keys, _values1, _values2);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2>
void HykSort(MpiComm comm, StrictWeakOrdering comp, unsigned int k_way,
             bool is_using_gpu, VectorK *_keys, VectorV1 *_values1,
             VectorV2 *_values2) throw(MpiException, ThrustException) {
  HYKSORT_CHECK(comm, k_way, comp, _keys, _values1, _values2);

  // Using keys to sort global indexes.
  HYKSORT_WITH_GLOBAL_INDEXES(comm, comp, k_way, is_using_gpu, _keys);
  // Using the sorted global indexes to re-arrange other values.
#ifdef __DO_BENCHMARK
  Timer arrange_values_tm;
  arrange_values_tm.Start();
#endif

  ARRANGE_VALUES(_values1, _values2);

#ifdef __DO_BENCHMARK
  arrange_values_tm.Stop();
#endif

#ifdef __DO_BENCHMARK
  ParUtils::ShowBenchmark(comm, arrange_values_tm, "Arranging values");
#endif
}

template<typename VectorK, typename VectorV1, typename VectorV2,
         typename VectorV3>
void HykSort(MpiComm comm, unsigned int k_way, bool is_using_gpu,
             VectorK *_keys, VectorV1 *_values1, VectorV2 *_values2,
             VectorV3 *_values3) throw(MpiException, ThrustException) {
  typedef typename VectorK::value_type K;
  HykSort(comm, std::less<K>(), k_way, is_using_gpu, _keys,
          _values1, _values2, _values3);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3>
void HykSort(MpiComm comm, StrictWeakOrdering comp, unsigned int k_way,
             bool is_using_gpu, VectorK *_keys,
             VectorV1 *_values1, VectorV2 *_values2,
             VectorV3 *_values3) throw(MpiException, ThrustException) {
  HYKSORT_CHECK(comm, k_way, comp, _keys, _values1, _values2, _values3);

  // Using keys to sort global indexes.
  HYKSORT_WITH_GLOBAL_INDEXES(comm, comp, k_way, is_using_gpu, _keys);
  // Using the sorted global indexes to re-arrange other values.
#ifdef __DO_BENCHMARK
  Timer arrange_values_tm;
  arrange_values_tm.Start();
#endif

  ARRANGE_VALUES(_values1, _values2, _values3);

#ifdef __DO_BENCHMARK
  arrange_values_tm.Stop();
#endif

#ifdef __DO_BENCHMARK
  ParUtils::ShowBenchmark(comm, arrange_values_tm, "Arranging values");
#endif
}

template<typename VectorK, typename VectorV1, typename VectorV2,
         typename VectorV3, typename VectorV4>
void HykSort(MpiComm comm, unsigned int k_way, bool is_using_gpu,
             VectorK *_keys, VectorV1 *_values1,
             VectorV2 *_values2, VectorV3 *_values3,
             VectorV4 *_values4) throw(MpiException, ThrustException) {
  typedef typename VectorK::value_type K;
  HykSort(comm, std::less<K>(), k_way, is_using_gpu, _keys,
          _values1, _values2, _values3, _values4);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4>
void HykSort(MpiComm comm, StrictWeakOrdering comp, unsigned int k_way,
             bool is_using_gpu, VectorK *_keys, VectorV1 *_values1,
             VectorV2 *_values2, VectorV3 *_values3,
             VectorV4 *_values4) throw(MpiException, ThrustException) {
  HYKSORT_CHECK(comm, k_way, comp, _keys, _values1, _values2,
                _values3, _values4);

  // Using keys to sort global indexes.
  HYKSORT_WITH_GLOBAL_INDEXES(comm, comp, k_way, is_using_gpu, _keys);
  // Using the sorted global indexes to re-arrange other values.
#ifdef __DO_BENCHMARK
  Timer arrange_values_tm;
  arrange_values_tm.Start();
#endif

  ARRANGE_VALUES(_values1, _values2, _values3, _values4);

#ifdef __DO_BENCHMARK
  arrange_values_tm.Stop();
#endif

#ifdef __DO_BENCHMARK
  ParUtils::ShowBenchmark(comm, arrange_values_tm, "Arranging values");
#endif
}

template<typename VectorK, typename VectorV1, typename VectorV2,
         typename VectorV3, typename VectorV4, typename VectorV5>
void HykSort(MpiComm comm, unsigned int k_way, bool is_using_gpu,
             VectorK *_keys, VectorV1 *_values1, VectorV2 *_values2,
             VectorV3 *_values3, VectorV4 *_values4,
             VectorV5 *_values5) throw(MpiException, ThrustException) {
  typedef typename VectorK::value_type K;
  HykSort(comm, std::less<K>(), k_way, is_using_gpu, _keys,
          _values1, _values2, _values3, _values4, _values5);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5>
void HykSort(MpiComm comm, StrictWeakOrdering comp, unsigned int k_way,
             bool is_using_gpu, VectorK *_keys, VectorV1 *_values1,
             VectorV2 *_values2, VectorV3 *_values3, VectorV4 *_values4,
             VectorV5 *_values5) throw(MpiException, ThrustException) {
  HYKSORT_CHECK(comm, k_way, comp, _keys, _values1, _values2,
                _values3, _values4, _values5);

  // Using keys to sort global indexes.
  HYKSORT_WITH_GLOBAL_INDEXES(comm, comp, k_way, is_using_gpu, _keys);
  // Using the sorted global indexes to re-arrange other values.
#ifdef __DO_BENCHMARK
  Timer arrange_values_tm;
  arrange_values_tm.Start();
#endif

  ARRANGE_VALUES(_values1, _values2, _values3, _values4, _values5);

#ifdef __DO_BENCHMARK
  arrange_values_tm.Stop();
#endif

#ifdef __DO_BENCHMARK
  ParUtils::ShowBenchmark(comm, arrange_values_tm, "Arranging values");
#endif
}

template<typename VectorK, typename VectorV1, typename VectorV2,
         typename VectorV3, typename VectorV4, typename VectorV5,
         typename VectorV6>
void HykSort(MpiComm comm, unsigned int k_way, bool is_using_gpu,
             VectorK *_keys, VectorV1 *_values1,
             VectorV2 *_values2, VectorV3 *_values3,
             VectorV4 *_values4, VectorV5 *_values5,
             VectorV6 *_values6) throw(MpiException, ThrustException) {
  typedef typename VectorK::value_type K;
  HykSort(comm, std::less<K>(), k_way, is_using_gpu, _keys,
          _values1, _values2, _values3, _values4, _values5, _values6);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5, typename VectorV6>
void HykSort(MpiComm comm, StrictWeakOrdering comp, unsigned int k_way,
             bool is_using_gpu, VectorK *_keys, VectorV1 *_values1,
             VectorV2 *_values2, VectorV3 *_values3,
             VectorV4 *_values4, VectorV5 *_values5,
             VectorV6 *_values6) throw(MpiException, ThrustException) {
  HYKSORT_CHECK(comm, k_way, comp, _keys, _values1, _values2,
                _values3, _values4, _values5, _values6);

  // Using keys to sort global indexes.
  HYKSORT_WITH_GLOBAL_INDEXES(comm, comp, k_way, is_using_gpu, _keys);
  // Using the sorted global indexes to re-arrange other values.
#ifdef __DO_BENCHMARK
  Timer arrange_values_tm;
  arrange_values_tm.Start();
#endif

  ARRANGE_VALUES(_values1, _values2, _values3, _values4, _values5,
                 _values6);

#ifdef __DO_BENCHMARK
  arrange_values_tm.Stop();
#endif

#ifdef __DO_BENCHMARK
  ParUtils::ShowBenchmark(comm, arrange_values_tm, "Arranging values");
#endif
}

template<typename VectorK, typename VectorV1, typename VectorV2,
         typename VectorV3, typename VectorV4, typename VectorV5,
         typename VectorV6, typename VectorV7>
void HykSort(MpiComm comm, unsigned int k_way, bool is_using_gpu,
             VectorK *_keys, VectorV1 *_values1, VectorV2 *_values2,
             VectorV3 *_values3, VectorV4 *_values4,
             VectorV5 *_values5, VectorV6 *_values6,
             VectorV7 *_values7) throw(MpiException, ThrustException) {
  typedef typename VectorK::value_type K;
  HykSort(comm, std::less<K>(), k_way, is_using_gpu, _keys,
          _values1, _values2, _values3, _values4, _values5,
          _values6, _values7);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5, typename VectorV6, typename VectorV7>
void HykSort(MpiComm comm, StrictWeakOrdering comp, unsigned int k_way,
             bool is_using_gpu, VectorK *_keys, VectorV1 *_values1,
             VectorV2 *_values2, VectorV3 *_values3, VectorV4 *_values4,
             VectorV5 *_values5, VectorV6 *_values6,
             VectorV7 *_values7) throw(MpiException, ThrustException) {
  HYKSORT_CHECK(comm, k_way, comp, _keys, _values1, _values2,
                _values3, _values4, _values5, _values6, _values7);

  // Using keys to sort global indexes.
  HYKSORT_WITH_GLOBAL_INDEXES(comm, comp, k_way, is_using_gpu, _keys);
  // Using the sorted global indexes to re-arrange other values.
#ifdef __DO_BENCHMARK
  Timer arrange_values_tm;
  arrange_values_tm.Start();
#endif

  ARRANGE_VALUES(_values1, _values2, _values3, _values4, _values5,
                 _values6, _values7);

#ifdef __DO_BENCHMARK
  arrange_values_tm.Stop();
#endif

#ifdef __DO_BENCHMARK
  ParUtils::ShowBenchmark(comm, arrange_values_tm, "Arranging values");
#endif
}

template<typename VectorK, typename VectorV1, typename VectorV2,
         typename VectorV3, typename VectorV4, typename VectorV5,
         typename VectorV6, typename VectorV7, typename VectorV8>
void HykSort(MpiComm comm, unsigned int k_way, bool is_using_gpu,
             VectorK *_keys, VectorV1 *_values1, VectorV2 *_values2,
             VectorV3 *_values3, VectorV4 *_values4, VectorV5 *_values5,
             VectorV6 *_values6, VectorV7 *_values7,
             VectorV8 *_values8) throw(MpiException, ThrustException) {
  typedef typename VectorK::value_type K;
  HykSort(comm, std::less<K>(), k_way, is_using_gpu, _keys,
          _values1, _values2, _values3, _values4, _values5,
          _values6, _values7, _values8);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5, typename VectorV6, typename VectorV7,
         typename VectorV8>
void HykSort(MpiComm comm, StrictWeakOrdering comp, unsigned int k_way,
             bool is_using_gpu, VectorK *_keys, VectorV1 *_values1,
             VectorV2 *_values2, VectorV3 *_values3, VectorV4 *_values4,
             VectorV5 *_values5, VectorV6 *_values6, VectorV7 *_values7,
             VectorV8 *_values8) throw(MpiException, ThrustException) {
  HYKSORT_CHECK(comm, k_way, comp, _keys, _values1, _values2,
                _values3, _values4, _values5, _values6, _values7, _values8);

  // Using keys to sort global indexes.
  HYKSORT_WITH_GLOBAL_INDEXES(comm, comp, k_way, is_using_gpu, _keys);
  // Using the sorted global indexes to re-arrange other values.
#ifdef __DO_BENCHMARK
  Timer arrange_values_tm;
  arrange_values_tm.Start();
#endif

  ARRANGE_VALUES(_values1, _values2, _values3, _values4, _values5,
                 _values6, _values7, _values8);

#ifdef __DO_BENCHMARK
  arrange_values_tm.Stop();
#endif

#ifdef __DO_BENCHMARK
  ParUtils::ShowBenchmark(comm, arrange_values_tm, "Arranging values");
#endif
}

template<typename VectorK, typename VectorV1, typename VectorV2,
         typename VectorV3, typename VectorV4, typename VectorV5,
         typename VectorV6, typename VectorV7, typename VectorV8,
         typename VectorV9>
void HykSort(MpiComm comm, unsigned int k_way, bool is_using_gpu,
             VectorK *_keys, VectorV1 *_values1, VectorV2 *_values2,
             VectorV3 *_values3, VectorV4 *_values4, VectorV5 *_values5,
             VectorV6 *_values6, VectorV7 *_values7, VectorV8 *_values8,
             VectorV9 *_values9) throw(MpiException, ThrustException) {
  typedef typename VectorK::value_type K;
  HykSort(comm, std::less<K>(), k_way, is_using_gpu, _keys,
          _values1, _values2, _values3, _values4, _values5,
          _values6, _values7, _values8, _values9);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5, typename VectorV6, typename VectorV7,
         typename VectorV8, typename VectorV9>
void HykSort(MpiComm comm, StrictWeakOrdering comp, unsigned int k_way,
             bool is_using_gpu, VectorK *_keys, VectorV1 *_values1,
             VectorV2 *_values2, VectorV3 *_values3, VectorV4 *_values4,
             VectorV5 *_values5, VectorV6 *_values6,
             VectorV7 *_values7, VectorV8 *_values8,
             VectorV9 *_values9) throw(MpiException, ThrustException) {
  HYKSORT_CHECK(comm, k_way, comp, _keys, _values1, _values2,
                _values3, _values4, _values5, _values6, _values7,
                _values8, _values9);

  // Using keys to sort global indexes.
  HYKSORT_WITH_GLOBAL_INDEXES(comm, comp, k_way, is_using_gpu, _keys);
  // Using the sorted global indexes to re-arrange other values.
#ifdef __DO_BENCHMARK
  Timer arrange_values_tm;
  arrange_values_tm.Start();
#endif

  ARRANGE_VALUES(_values1, _values2, _values3, _values4, _values5,
                 _values6, _values7, _values8, _values9);

#ifdef __DO_BENCHMARK
  arrange_values_tm.Stop();
#endif

#ifdef __DO_BENCHMARK
  ParUtils::ShowBenchmark(comm, arrange_values_tm, "Arranging values");
#endif
}

template<typename VectorK, typename VectorV1, typename VectorV2,
         typename VectorV3, typename VectorV4, typename VectorV5,
         typename VectorV6, typename VectorV7, typename VectorV8,
         typename VectorV9, typename VectorV10>
void HykSort(MpiComm comm, unsigned int k_way, bool is_using_gpu,
             VectorK *_keys, VectorV1 *_values1, VectorV2 *_values2,
             VectorV3 *_values3, VectorV4 *_values4, VectorV5 *_values5,
             VectorV6 *_values6, VectorV7 *_values7,
             VectorV8 *_values8, VectorV9 *_values9,
             VectorV10 *_values10) throw(MpiException, ThrustException) {
  typedef typename VectorK::value_type K;
  HykSort(comm, std::less<K>(), k_way, is_using_gpu, _keys,
          _values1, _values2, _values3, _values4, _values5,
          _values6, _values7, _values8, _values9, _values10);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5, typename VectorV6, typename VectorV7,
         typename VectorV8, typename VectorV9, typename VectorV10>
void HykSort(MpiComm comm, StrictWeakOrdering comp, unsigned int k_way,
             bool is_using_gpu, VectorK *_keys, VectorV1 *_values1,
             VectorV2 *_values2, VectorV3 *_values3, VectorV4 *_values4,
             VectorV5 *_values5, VectorV6 *_values6, VectorV7 *_values7,
             VectorV8 *_values8, VectorV9 *_values9,
             VectorV10 *_values10) throw(MpiException, ThrustException) {
  HYKSORT_CHECK(comm, k_way, comp, _keys, _values1, _values2,
                _values3, _values4, _values5, _values6, _values7,
                _values8, _values9, _values10);

  // Using keys to sort global indexes.
  HYKSORT_WITH_GLOBAL_INDEXES(comm, comp, k_way, is_using_gpu, _keys);
  // Using the sorted global indexes to re-arrange other values.
#ifdef __DO_BENCHMARK
  Timer arrange_values_tm;
  arrange_values_tm.Start();
#endif

  ARRANGE_VALUES(_values1, _values2, _values3, _values4, _values5,
                 _values6, _values7, _values8, _values9, _values10);

#ifdef __DO_BENCHMARK
  arrange_values_tm.Stop();
#endif

#ifdef __DO_BENCHMARK
  ParUtils::ShowBenchmark(comm, arrange_values_tm, "Arranging values");
#endif
}

#endif  // __USING_CPP_0X || __USING_CPP_11

template<typename StrictWeakOrdering, typename VectorK, typename Int>
void HykSortWithGlobalIndexes(
    MpiComm comm, StrictWeakOrdering comp, unsigned int k_way,
    bool is_using_gpu, const std::vector<Int> &old_sizes, VectorK *_keys,
    std::vector<Int> *_g_indexes) throw(MpiException, ThrustException) {
  int myrank = MpiUtils::MpiCommRank(comm);
  Int nelem = _keys->size();
  _g_indexes->resize(nelem);
  Int begin = (myrank == 0)? 0 : old_sizes[myrank - 1];
  InitGlobalIndexes(begin, nelem, &(*_g_indexes)[0]);

  // Sort the keys and the global indexes.
  HykSort(comm, comp, k_way, is_using_gpu, _keys, _g_indexes);
}

}  // namespace gpusort
