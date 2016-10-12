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
#include "helpers/hyperquicksort_helper.h"
#include "mpi/mpi_common.h"
#include "mpi/mpi_utils.h"
#include "par/par_utils.h"
#include "samplesort.h"

#ifdef __DO_BENCHMARK
#include "timer.h"
#endif

namespace gpusort {

// Check whether the number of processes is valid to be used with
// the HyperQuickSort algorithm. Otherwise, we will use
// the SampleSort algorithm instead.
#define HYPERQUICKSORT_CHECK(comm, ...) \
  { \
    int npes = MpiUtils::MpiCommSize(comm);\
    if (!IsPowerOfTwo(npes)) {\
      int myrank = MpiUtils::MpiCommRank(comm);\
      if (myrank == 0)\
        std::cout << "The number of processes is not a power of 2. "\
                  << "We are using SampleSort algorithm instead!" << std::endl;\
      SampleSort(comm, __VA_ARGS__);\
      return;\
    }\
  }

#define HYPERQUICKSORT_WITH_GLOBAL_INDEXES_T(int_t, comm, comp, _keys) \
  std::vector<int_t> g_indexes;\
  std::vector<int_t> old_sizes;\
  int_t size = _keys->size();\
  int myrank = MpiUtils::MpiCommRank(comm);\
  ParUtils::GatherNElems(comm, size, &old_sizes);\
  /* Sort the keys and get their global indexes' changes.*/\
  HyperQuickSortWithGlobalIndexes(comm, comp, old_sizes, _keys, &g_indexes);

#define HYPERQUICKSORT_WITH_GLOBAL_INDEXES(comm, comp, _keys) \
  HYPERQUICKSORT_WITH_GLOBAL_INDEXES_T(int64_t, comm, comp, _keys);

template<typename Vector>
void HyperQuickSort(MpiComm comm,
                    Vector *_keys) throw(MpiException) {
  typedef typename Vector::value_type T;
  HyperQuickSort(comm, std::less<T>(), _keys);
}

template<typename StrictWeakOrdering, typename Vector>
void HyperQuickSort(MpiComm comm, StrictWeakOrdering comp,
                    Vector *_keys) throw(MpiException) {
  HYPERQUICKSORT_CHECK(comm, comp, _keys);

  HyperQuickSortHelper helper(comm);
  helper.Sort(comp, _keys);

#ifdef __DO_BENCHMARK
  helper.ShowBenchmark();
#endif
}

template<typename VectorK, typename VectorV>
void HyperQuickSort(MpiComm comm, VectorK *_keys,
                    VectorV *_values) throw(MpiException) {
  typedef typename VectorK::value_type K;
  HyperQuickSort(comm, std::less<K>(), _keys, _values);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV>
void HyperQuickSort(MpiComm comm, StrictWeakOrdering comp, VectorK *_keys,
                    VectorV *_values) throw(MpiException) {
  HYPERQUICKSORT_CHECK(comm, comp, _keys, _values);

  HyperQuickSortHelper helper(comm);
  helper.SortByKey(comp, _keys, _values);

#ifdef __DO_BENCHMARK
  helper.ShowBenchmark();
#endif
}

#if defined(__USING_CPP_0X) || defined(__USING_CPP_11)
template<typename VectorK, typename... Vectors>
void HyperQuickSort(MpiComm comm, VectorK *_keys,
                    Vectors*... _values) throw(MpiException) {
  typedef typename VectorK::value_type K;
  HyperQuickSort(comm, std::less<K>(), _keys, _values...);
}

template<typename StrictWeakOrdering, typename VectorK,
         typename... Vectors>
void HyperQuickSort(MpiComm comm, StrictWeakOrdering comp, VectorK *_keys,
                    Vectors*... _values) throw(MpiException) {
  HYPERQUICKSORT_CHECK(comm, comp, _keys, _values...);

  // Using keys to sort global indexes.
  HYPERQUICKSORT_WITH_GLOBAL_INDEXES(comm, comp, _keys);
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
void HyperQuickSort(MpiComm comm, VectorK *_keys, VectorV1 *_values1,
                    VectorV2 *_values2) throw(MpiException) {
  typedef typename VectorK::value_type K;
  HyperQuickSort(comm, std::less<K>(), _keys, _values1, _values2);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2>
void HyperQuickSort(MpiComm comm, StrictWeakOrdering comp,
                    VectorK *_keys, VectorV1 *_values1,
                    VectorV2 *_values2) throw(MpiException) {
  HYPERQUICKSORT_CHECK(comm, comp, _keys, _values1, _values2);

  // Using keys to sort global indexes.
  HYPERQUICKSORT_WITH_GLOBAL_INDEXES(comm, comp, _keys);
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
void HyperQuickSort(MpiComm comm, VectorK *_keys,
                    VectorV1 *_values1, VectorV2 *_values2,
                    VectorV3 *_values3) throw(MpiException) {
  typedef typename VectorK::value_type K;
  HyperQuickSort(comm, std::less<K>(), _keys, _values1, _values2, _values3);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3>
void HyperQuickSort(MpiComm comm, StrictWeakOrdering comp, VectorK *_keys,
                    VectorV1 *_values1, VectorV2 *_values2,
                    VectorV3 *_values3) throw(MpiException) {
  HYPERQUICKSORT_CHECK(comm, comp, _keys, _values1, _values2, _values3);

  // Using keys to sort global indexes.
  HYPERQUICKSORT_WITH_GLOBAL_INDEXES(comm, comp, _keys);
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
void HyperQuickSort(MpiComm comm, VectorK *_keys, VectorV1 *_values1,
                    VectorV2 *_values2, VectorV3 *_values3,
                    VectorV4 *_values4) throw(MpiException) {
  typedef typename VectorK::value_type K;
  HyperQuickSort(comm, std::less<K>(), _keys, _values1,
                 _values2, _values3, _values4);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4>
void HyperQuickSort(MpiComm comm, StrictWeakOrdering comp, VectorK *_keys,
                    VectorV1 *_values1, VectorV2 *_values2, VectorV3 *_values3,
                    VectorV4 *_values4) throw(MpiException) {
  HYPERQUICKSORT_CHECK(comm, comp, _keys, _values1, _values2,
                       _values3, _values4);

  // Using keys to sort global indexes.
  HYPERQUICKSORT_WITH_GLOBAL_INDEXES(comm, comp, _keys);
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
void HyperQuickSort(MpiComm comm, VectorK *_keys, VectorV1 *_values1,
                    VectorV2 *_values2, VectorV3 *_values3, VectorV4 *_values4,
                    VectorV5 *_values5) throw(MpiException) {
  typedef typename VectorK::value_type K;
  HyperQuickSort(comm, std::less<K>(), _keys, _values1,
                 _values2, _values3, _values4, _values5);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5>
void HyperQuickSort(MpiComm comm, StrictWeakOrdering comp,
                    VectorK *_keys, VectorV1 *_values1, VectorV2 *_values2,
                    VectorV3 *_values3, VectorV4 *_values4,
                    VectorV5 *_values5) throw(MpiException) {
  HYPERQUICKSORT_CHECK(comm, comp, _keys, _values1, _values2,
                       _values3, _values4, _values5);

  // Using keys to sort global indexes.
  HYPERQUICKSORT_WITH_GLOBAL_INDEXES(comm, comp, _keys);
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
void HyperQuickSort(MpiComm comm, VectorK *_keys, VectorV1 *_values1,
                    VectorV2 *_values2, VectorV3 *_values3,
                    VectorV4 *_values4, VectorV5 *_values5,
                    VectorV6 *_values6) throw(MpiException) {
  typedef typename VectorK::value_type K;
  HyperQuickSort(comm, std::less<K>(), _keys, _values1, _values2,
                 _values3, _values4, _values5, _values6);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5, typename VectorV6>
void HyperQuickSort(MpiComm comm, StrictWeakOrdering comp,
                    VectorK *_keys, VectorV1 *_values1, VectorV2 *_values2,
                    VectorV3 *_values3, VectorV4 *_values4, VectorV5 *_values5,
                    VectorV6 *_values6) throw(MpiException) {
  HYPERQUICKSORT_CHECK(comm, comp, _keys, _values1, _values2,
                       _values3, _values4, _values5, _values6);

  // Using keys to sort global indexes.
  HYPERQUICKSORT_WITH_GLOBAL_INDEXES(comm, comp, _keys);
  // Using the sorted global indexes to re-arrange other values.
#ifdef __DO_BENCHMARK
  Timer arrange_values_tm;
  arrange_values_tm.Start();
#endif

  ARRANGE_VALUES(_values1, _values2, _values3, _values4, _values5, _values6);

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
void HyperQuickSort(MpiComm comm, VectorK *_keys,
                    VectorV1 *_values1, VectorV2 *_values2, VectorV3 *_values3,
                    VectorV4 *_values4, VectorV5 *_values5, VectorV6 *_values6,
                    VectorV7 *_values7) throw(MpiException) {
  typedef typename VectorK::value_type K;
  HyperQuickSort(comm, std::less<K>(), _keys, _values1, _values2,
                 _values3, _values4, _values5, _values6, _values7);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5, typename VectorV6, typename VectorV7>
void HyperQuickSort(MpiComm comm, StrictWeakOrdering comp, VectorK *_keys,
                    VectorV1 *_values1, VectorV2 *_values2, VectorV3 *_values3,
                    VectorV4 *_values4, VectorV5 *_values5, VectorV6 *_values6,
                    VectorV7 *_values7) throw(MpiException) {
  HYPERQUICKSORT_CHECK(comm, comp, _keys, _values1, _values2,
                       _values3, _values4, _values5, _values6, _values7);

  // Using keys to sort global indexes.
  HYPERQUICKSORT_WITH_GLOBAL_INDEXES(comm, comp, _keys);
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
void HyperQuickSort(MpiComm comm, VectorK *_keys, VectorV1 *_values1,
                    VectorV2 *_values2, VectorV3 *_values3, VectorV4 *_values4,
                    VectorV5 *_values5, VectorV6 *_values6, VectorV7 *_values7,
                    VectorV8 *_values8) throw(MpiException) {
  typedef typename VectorK::value_type K;
  HyperQuickSort(comm, std::less<K>(), _keys, _values1, _values2,
                 _values3, _values4, _values5, _values6, _values7, _values8);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5, typename VectorV6, typename VectorV7,
         typename VectorV8>
void HyperQuickSort(MpiComm comm, StrictWeakOrdering comp, VectorK *_keys,
                    VectorV1 *_values1, VectorV2 *_values2,
                    VectorV3 *_values3, VectorV4 *_values4,
                    VectorV5 *_values5, VectorV6 *_values6, VectorV7 *_values7,
                    VectorV8 *_values8) throw(MpiException) {
  HYPERQUICKSORT_CHECK(comm, comp, _keys, _values1, _values2, _values3,
                       _values4, _values5, _values6, _values7, _values8);

  // Using keys to sort global indexes.
  HYPERQUICKSORT_WITH_GLOBAL_INDEXES(comm, comp, _keys);
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
void HyperQuickSort(MpiComm comm, VectorK *_keys,
                    VectorV1 *_values1, VectorV2 *_values2, VectorV3 *_values3,
                    VectorV4 *_values4, VectorV5 *_values5, VectorV6 *_values6,
                    VectorV7 *_values7, VectorV8 *_values8,
                    VectorV9 *_values9) throw(MpiException) {
  typedef typename VectorK::value_type K;
  HyperQuickSort(comm, std::less<K>(), _keys, _values1, _values2, _values3,
                 _values4, _values5, _values6, _values7, _values8, _values9);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5, typename VectorV6, typename VectorV7,
         typename VectorV8, typename VectorV9>
void HyperQuickSort(MpiComm comm, StrictWeakOrdering comp,
                    VectorK *_keys, VectorV1 *_values1, VectorV2 *_values2,
                    VectorV3 *_values3, VectorV4 *_values4, VectorV5 *_values5,
                    VectorV6 *_values6, VectorV7 *_values7, VectorV8 *_values8,
                    VectorV9 *_values9) throw(MpiException) {
  HYPERQUICKSORT_CHECK(comm, comp, _keys, _values1, _values2,
                       _values3, _values4, _values5, _values6,
                       _values7, _values8, _values9);

  // Using keys to sort global indexes.
  HYPERQUICKSORT_WITH_GLOBAL_INDEXES(comm, comp, _keys);
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
void HyperQuickSort(MpiComm comm, VectorK *_keys,
                    VectorV1 *_values1, VectorV2 *_values2, VectorV3 *_values3,
                    VectorV4 *_values4, VectorV5 *_values5, VectorV6 *_values6,
                    VectorV7 *_values7, VectorV8 *_values8, VectorV9 *_values9,
                    VectorV10 *_values10) throw(MpiException) {
  typedef typename VectorK::value_type K;
  HyperQuickSort(comm, std::less<K>(), _keys, _values1,
                 _values2, _values3, _values4, _values5,
                 _values6, _values7, _values8, _values9, _values10);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5, typename VectorV6, typename VectorV7,
         typename VectorV8, typename VectorV9, typename VectorV10>
void HyperQuickSort(MpiComm comm, StrictWeakOrdering comp, VectorK *_keys,
                    VectorV1 *_values1, VectorV2 *_values2, VectorV3 *_values3,
                    VectorV4 *_values4, VectorV5 *_values5, VectorV6 *_values6,
                    VectorV7 *_values7, VectorV8 *_values8, VectorV9 *_values9,
                    VectorV10 *_values10) throw(MpiException) {
  HYPERQUICKSORT_CHECK(comm, comp, _keys, _values1, _values2,
                       _values3, _values4, _values5, _values6,
                       _values7, _values8, _values9, _values10);

  // Using keys to sort global indexes.
  HYPERQUICKSORT_WITH_GLOBAL_INDEXES(comm, comp, _keys);
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
void HyperQuickSortWithGlobalIndexes(
    MpiComm comm, StrictWeakOrdering comp, const std::vector<Int> &old_sizes,
    VectorK *_keys, std::vector<Int> *_g_indexes) throw(MpiException) {
  int myrank = MpiUtils::MpiCommRank(comm);
  Int nelem = _keys->size();
  _g_indexes->resize(nelem);
  Int begin = (myrank == 0)? 0 : old_sizes[myrank - 1];
  InitGlobalIndexes(begin, nelem, &(*_g_indexes)[0]);

  // Sort the keys and the global indexes.
  HyperQuickSort(comm, comp, _keys, _g_indexes);
}

}  // namespace gpusort
