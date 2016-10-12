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
#include "helpers/samplesort_helper.h"
#include "mpi/mpi_common.h"
#include "mpi/mpi_utils.h"
#include "par/par_utils.h"

#ifdef __DO_BENCHMARK
#include "timer.h"
#endif

namespace gpusort {

#define SAMPLESORT_WITH_GLOBAL_INDEXES_T(int_t, comm, comp, _keys) \
  std::vector<int_t> g_indexes;\
  std::vector<int_t> old_sizes;\
  int_t size = _keys->size();\
  int myrank = MpiUtils::MpiCommRank(comm);\
  ParUtils::GatherNElems(comm, size, &old_sizes);\
  /* Sort the keys and get their global indexes' changes.*/\
  SampleSortWithGlobalIndexes(comm, comp, old_sizes, _keys, &g_indexes);

#define SAMPLESORT_WITH_GLOBAL_INDEXES(comm, comp, _keys) \
  SAMPLESORT_WITH_GLOBAL_INDEXES_T(int64_t, comm, comp, _keys);

template<typename Vector>
void SampleSort(MpiComm comm, Vector *_keys) throw(MpiException) {
  typedef typename Vector::value_type T;
  SampleSort(comm, std::less<T>(), _keys);
}

template<typename StrictWeakOrdering, typename Vector>
void SampleSort(MpiComm comm, StrictWeakOrdering comp,
                Vector *_keys) throw(MpiException) {
  SampleSortHelper helper(comm);
  helper.Sort(comp, _keys);

#ifdef __DO_BENCHMARK
  helper.ShowBenchmark();
#endif
}

template<typename VectorK, typename VectorV>
void SampleSort(MpiComm comm, VectorK *_keys,
                VectorV *_values) throw(MpiException) {
  typedef typename VectorK::value_type K;
  SampleSort(comm, std::less<K>(), _keys, _values);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV>
void SampleSort(MpiComm comm, StrictWeakOrdering comp, VectorK *_keys,
                VectorV *_values) throw(MpiException) {
  SampleSortHelper helper(comm);
  helper.SortByKey(comp, _keys, _values);

#ifdef __DO_BENCHMARK
  helper.ShowBenchmark();
#endif
}

#if defined(__USING_CPP_0X) || defined(__USING_CPP_11)
template<typename VectorK, typename... Vectors>
void SampleSort(MpiComm comm, VectorK *_keys,
                Vectors*... _values) throw(MpiException) {
  typedef typename VectorK::value_type K;
  SampleSort(comm, std::less<K>(), _keys, _values...);
}

template<typename StrictWeakOrdering, typename VectorK,
         typename... Vectors>
void SampleSort(MpiComm comm, StrictWeakOrdering comp, VectorK *_keys,
                Vectors*... _values) throw(MpiException) {
  // Using keys to sort global indexes.
  SAMPLESORT_WITH_GLOBAL_INDEXES(comm, comp, _keys);
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
void SampleSort(MpiComm comm, VectorK *_keys, VectorV1 *_values1,
                VectorV2 *_values2) throw(MpiException) {
  typedef typename VectorK::value_type K;
  SampleSort(comm, std::less<K>(), _keys, _values1, _values2);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2>
void SampleSort(MpiComm comm, StrictWeakOrdering comp,
                VectorK *_keys, VectorV1 *_values1,
                VectorV2 *_values2) throw(MpiException) {
  // Using keys to sort global indexes.
  SAMPLESORT_WITH_GLOBAL_INDEXES(comm, comp, _keys);
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
void SampleSort(MpiComm comm, VectorK *_keys,
                VectorV1 *_values1, VectorV2 *_values2,
                VectorV3 *_values3) throw(MpiException) {
  typedef typename VectorK::value_type K;
  SampleSort(comm, std::less<K>(), _keys, _values1, _values2, _values3);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3>
void SampleSort(MpiComm comm, StrictWeakOrdering comp, VectorK *_keys,
                VectorV1 *_values1, VectorV2 *_values2,
                VectorV3 *_values3) throw(MpiException) {
  // Using keys to sort global indexes.
  SAMPLESORT_WITH_GLOBAL_INDEXES(comm, comp, _keys);
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
void SampleSort(MpiComm comm, VectorK *_keys, VectorV1 *_values1,
                VectorV2 *_values2, VectorV3 *_values3,
                VectorV4 *_values4) throw(MpiException) {
  typedef typename VectorK::value_type K;
  SampleSort(comm, std::less<K>(), _keys, _values1,
             _values2, _values3, _values4);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4>
void SampleSort(MpiComm comm, StrictWeakOrdering comp, VectorK *_keys,
                VectorV1 *_values1, VectorV2 *_values2, VectorV3 *_values3,
                VectorV4 *_values4) throw(MpiException) {
  // Using keys to sort global indexes.
  SAMPLESORT_WITH_GLOBAL_INDEXES(comm, comp, _keys);
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
void SampleSort(MpiComm comm, VectorK *_keys, VectorV1 *_values1,
                VectorV2 *_values2, VectorV3 *_values3, VectorV4 *_values4,
                VectorV5 *_values5) throw(MpiException) {
  typedef typename VectorK::value_type K;
  SampleSort(comm, std::less<K>(), _keys, _values1,
             _values2, _values3, _values4, _values5);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5>
void SampleSort(MpiComm comm, StrictWeakOrdering comp,
                VectorK *_keys, VectorV1 *_values1, VectorV2 *_values2,
                VectorV3 *_values3, VectorV4 *_values4,
                VectorV5 *_values5) throw(MpiException) {
  // Using keys to sort global indexes.
  SAMPLESORT_WITH_GLOBAL_INDEXES(comm, comp, _keys);
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
void SampleSort(MpiComm comm, VectorK *_keys, VectorV1 *_values1,
                VectorV2 *_values2, VectorV3 *_values3,
                VectorV4 *_values4, VectorV5 *_values5,
                VectorV6 *_values6) throw(MpiException) {
  typedef typename VectorK::value_type K;
  SampleSort(comm, std::less<K>(), _keys, _values1, _values2,
             _values3, _values4, _values5, _values6);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5, typename VectorV6>
void SampleSort(MpiComm comm, StrictWeakOrdering comp,
                VectorK *_keys, VectorV1 *_values1, VectorV2 *_values2,
                VectorV3 *_values3, VectorV4 *_values4, VectorV5 *_values5,
                VectorV6 *_values6) throw(MpiException) {
  // Using keys to sort global indexes.
  SAMPLESORT_WITH_GLOBAL_INDEXES(comm, comp, _keys);
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
void SampleSort(MpiComm comm, VectorK *_keys,
                VectorV1 *_values1, VectorV2 *_values2, VectorV3 *_values3,
                VectorV4 *_values4, VectorV5 *_values5, VectorV6 *_values6,
                VectorV7 *_values7) throw(MpiException) {
  typedef typename VectorK::value_type K;
  SampleSort(comm, std::less<K>(), _keys, _values1, _values2,
             _values3, _values4, _values5, _values6, _values7);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5, typename VectorV6, typename VectorV7>
void SampleSort(MpiComm comm, StrictWeakOrdering comp, VectorK *_keys,
                VectorV1 *_values1, VectorV2 *_values2, VectorV3 *_values3,
                VectorV4 *_values4, VectorV5 *_values5, VectorV6 *_values6,
                VectorV7 *_values7) throw(MpiException) {
  // Using keys to sort global indexes.
  SAMPLESORT_WITH_GLOBAL_INDEXES(comm, comp, _keys);
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
void SampleSort(MpiComm comm, VectorK *_keys, VectorV1 *_values1,
                VectorV2 *_values2, VectorV3 *_values3, VectorV4 *_values4,
                VectorV5 *_values5, VectorV6 *_values6, VectorV7 *_values7,
                VectorV8 *_values8) throw(MpiException) {
  typedef typename VectorK::value_type K;
  SampleSort(comm, std::less<K>(), _keys, _values1, _values2,
             _values3, _values4, _values5, _values6, _values7, _values8);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5, typename VectorV6, typename VectorV7,
         typename VectorV8>
void SampleSort(MpiComm comm, StrictWeakOrdering comp, VectorK *_keys,
                VectorV1 *_values1, VectorV2 *_values2,
                VectorV3 *_values3, VectorV4 *_values4,
                VectorV5 *_values5, VectorV6 *_values6, VectorV7 *_values7,
                VectorV8 *_values8) throw(MpiException) {
  // Using keys to sort global indexes.
  SAMPLESORT_WITH_GLOBAL_INDEXES(comm, comp, _keys);
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
void SampleSort(MpiComm comm, VectorK *_keys,
                VectorV1 *_values1, VectorV2 *_values2, VectorV3 *_values3,
                VectorV4 *_values4, VectorV5 *_values5, VectorV6 *_values6,
                VectorV7 *_values7, VectorV8 *_values8,
                VectorV9 *_values9) throw(MpiException) {
  typedef typename VectorK::value_type K;
  SampleSort(comm, std::less<K>(), _keys, _values1, _values2, _values3,
             _values4, _values5, _values6, _values7, _values8, _values9);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5, typename VectorV6, typename VectorV7,
         typename VectorV8, typename VectorV9>
void SampleSort(MpiComm comm, StrictWeakOrdering comp,
                VectorK *_keys, VectorV1 *_values1, VectorV2 *_values2,
                VectorV3 *_values3, VectorV4 *_values4, VectorV5 *_values5,
                VectorV6 *_values6, VectorV7 *_values7, VectorV8 *_values8,
                VectorV9 *_values9) throw(MpiException) {
  // Using keys to sort global indexes.
  SAMPLESORT_WITH_GLOBAL_INDEXES(comm, comp, _keys);
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
void SampleSort(MpiComm comm, VectorK *_keys,
                VectorV1 *_values1, VectorV2 *_values2, VectorV3 *_values3,
                VectorV4 *_values4, VectorV5 *_values5, VectorV6 *_values6,
                VectorV7 *_values7, VectorV8 *_values8, VectorV9 *_values9,
                VectorV10 *_values10) throw(MpiException) {
  typedef typename VectorK::value_type K;
  SampleSort(comm, std::less<K>(), _keys, _values1,
             _values2, _values3, _values4, _values5,
             _values6, _values7, _values8, _values9, _values10);
}

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5, typename VectorV6, typename VectorV7,
         typename VectorV8, typename VectorV9, typename VectorV10>
void SampleSort(MpiComm comm, StrictWeakOrdering comp, VectorK *_keys,
                VectorV1 *_values1, VectorV2 *_values2, VectorV3 *_values3,
                VectorV4 *_values4, VectorV5 *_values5, VectorV6 *_values6,
                VectorV7 *_values7, VectorV8 *_values8, VectorV9 *_values9,
                VectorV10 *_values10) throw(MpiException) {
  // Using keys to sort global indexes.
  SAMPLESORT_WITH_GLOBAL_INDEXES(comm, comp, _keys);
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
void SampleSortWithGlobalIndexes(
    MpiComm comm, StrictWeakOrdering comp, const std::vector<Int> &old_sizes,
    VectorK *_keys, std::vector<Int> *_g_indexes) throw(MpiException) {
  int myrank = MpiUtils::MpiCommRank(comm);
  Int nelem = _keys->size();
  _g_indexes->resize(nelem);
  Int begin = (myrank == 0)? 0 : old_sizes[myrank - 1];
  InitGlobalIndexes(begin, nelem, &(*_g_indexes)[0]);

  // Sort the keys and the global indexes.
  SampleSort(comm, comp, _keys, _g_indexes);
}

}  // namespace gpusort
