#ifndef GPUSORT_SAMPLESORT_COMMON_VECTOR_H_
#define GPUSORT_SAMPLESORT_COMMON_VECTOR_H_

#include <vector>

#include "exceptions/mpi_exception.h"
#include "mpi/mpi_common.h"

namespace gpusort {

template<typename Vector>
void SampleSort(MpiComm comm, Vector *_keys) throw(MpiException);

template<typename StrictWeakOrdering, typename Vector>
void SampleSort(MpiComm comm, StrictWeakOrdering comp,
                Vector *_keys) throw(MpiException);

template<typename VectorK, typename VectorV>
void SampleSort(MpiComm comm, VectorK *_keys,
                VectorV *_values) throw(MpiException);

template<typename StrictWeakOrdering, typename VectorK, typename VectorV>
void SampleSort(MpiComm comm, StrictWeakOrdering comp, VectorK *_keys,
                VectorV *_values) throw(MpiException);

#if defined(__USING_CPP_0X) || defined(__USING_CPP_11)
template<typename VectorK, typename... Vectors>
void SampleSort(MpiComm comm, VectorK *_keys,
                Vectors*... _values) throw(MpiException);

template<typename StrictWeakOrdering, typename VectorK,
         typename... Vectors>
void SampleSort(MpiComm comm, StrictWeakOrdering comp,
                VectorK *_keys, Vectors*... _values) throw(MpiException);

#else  // !__USING_CPP_0X && !__USING_CPP_11
template<typename VectorK, typename VectorV1, typename VectorV2>
void SampleSort(MpiComm comm, VectorK *_keys, VectorV1 *_values1,
                VectorV2 *_values2) throw(MpiException);

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2>
void SampleSort(MpiComm comm, StrictWeakOrdering comp, VectorK *_keys,
                VectorV1 *_values1, VectorV2 *_values2) throw(MpiException);

template<typename VectorK, typename VectorV1, typename VectorV2,
         typename VectorV3>
void SampleSort(MpiComm comm, VectorK *_keys, VectorV1 *_values1,
                VectorV2 *_values2, VectorV3 *_values3) throw(MpiException);

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3>
void SampleSort(MpiComm comm, StrictWeakOrdering comp, VectorK *_keys,
                VectorV1 *_values1, VectorV2 *_values2,
                VectorV3 *_values3) throw(MpiException);

template<typename VectorK, typename VectorV1, typename VectorV2,
         typename VectorV3, typename VectorV4>
void SampleSort(MpiComm comm, VectorK *_keys, VectorV1 *_values1,
                VectorV2 *_values2, VectorV3 *_values3,
                VectorV4 *_values4) throw(MpiException);

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4>
void SampleSort(MpiComm comm, StrictWeakOrdering comp, VectorK *_keys,
                VectorV1 *_values1, VectorV2 *_values2, VectorV3 *_values3,
                VectorV4 *_values4) throw(MpiException);

template<typename VectorK, typename VectorV1, typename VectorV2,
         typename VectorV3, typename VectorV4, typename VectorV5>
void SampleSort(MpiComm comm, VectorK *_keys, VectorV1 *_values1,
                VectorV2 *_values2, VectorV3 *_values3, VectorV4 *_values4,
                VectorV5 *_values5) throw(MpiException);

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5>
void SampleSort(MpiComm comm, StrictWeakOrdering comp, VectorK *_keys,
                VectorV1 *_values1, VectorV2 *_values2, VectorV3 *_values3,
                VectorV4 *_values4, VectorV5 *_values5) throw(MpiException);

template<typename VectorK, typename VectorV1, typename VectorV2,
         typename VectorV3, typename VectorV4, typename VectorV5,
         typename VectorV6>
void SampleSort(MpiComm comm, VectorK *_keys, VectorV1 *_values1,
                VectorV2 *_values2, VectorV3 *_values3, VectorV4 *_values4,
                VectorV5 *_values5, VectorV6 *_values6) throw(MpiException);

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5, typename VectorV6>
void SampleSort(MpiComm comm, StrictWeakOrdering comp, VectorK *_keys,
                VectorV1 *_values1, VectorV2 *_values2, VectorV3 *_values3,
                VectorV4 *_values4, VectorV5 *_values5,
                VectorV6 *_values6) throw(MpiException);

template<typename VectorK, typename VectorV1, typename VectorV2,
         typename VectorV3, typename VectorV4, typename VectorV5,
         typename VectorV6, typename VectorV7>
void SampleSort(MpiComm comm, VectorK *_keys_first, VectorV1 *_values1,
                VectorV2 *_values2, VectorV3 *_values3, VectorV4 *_values4,
                VectorV5 *_values5, VectorV6 *_values6,
                VectorV7 *_values7) throw(MpiException);

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5, typename VectorV6, typename VectorV7>
void SampleSort(MpiComm comm, StrictWeakOrdering comp, VectorK *_keys,
                VectorV1 *_values1, VectorV2 *_values2, VectorV3 *_values3,
                VectorV4 *_values4, VectorV5 *_values5, VectorV6 *_values6,
                VectorV7 *_values7) throw(MpiException);

template<typename VectorK, typename VectorV1, typename VectorV2,
         typename VectorV3, typename VectorV4, typename VectorV5,
         typename VectorV6, typename VectorV7, typename VectorV8>
void SampleSort(MpiComm comm, VectorK *_keys, VectorV1 *_values1,
                VectorV2 *_values2, VectorV3 *_values3, VectorV4 *_values4,
                VectorV5 *_values5, VectorV6 *_values6, VectorV7 *_values7,
                VectorV8 *_values8) throw(MpiException);

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5, typename VectorV6, typename VectorV7,
         typename VectorV8>
void SampleSort(MpiComm comm, StrictWeakOrdering comp, VectorK *_keys,
                VectorV1 *_values1, VectorV2 *_values2, VectorV3 *_values3,
                VectorV4 *_values4, VectorV5 *_values5, VectorV6 *_values6,
                VectorV7 *_values7, VectorV8 *_values8) throw(MpiException);

template<typename VectorK, typename VectorV1, typename VectorV2,
         typename VectorV3, typename VectorV4, typename VectorV5,
         typename VectorV6, typename VectorV7, typename VectorV8,
         typename VectorV9>
void SampleSort(MpiComm comm, VectorK *_keys, VectorV1 *_values1,
                VectorV2 *_values2, VectorV3 *_values3, VectorV4 *_values4,
                VectorV5 *_values5, VectorV6 *_values6, VectorV7 *_values7,
                VectorV8 *_values8, VectorV9 *_values9) throw(MpiException);

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5, typename VectorV6, typename VectorV7,
         typename VectorV8, typename VectorV9>
void SampleSort(MpiComm comm, StrictWeakOrdering comp, VectorK *_keys,
                VectorV1 *_values1, VectorV2 *_values2, VectorV3 *_values3,
                VectorV4 *_values4, VectorV5 *_values5, VectorV6 *_values6,
                VectorV7 *_values7, VectorV8 *_values8,
                VectorV9 *_values9) throw(MpiException);

template<typename VectorK, typename VectorV1, typename VectorV2,
         typename VectorV3, typename VectorV4, typename VectorV5,
         typename VectorV6, typename VectorV7, typename VectorV8,
         typename VectorV9, typename VectorV10>
void SampleSort(MpiComm comm, VectorK *_keys, VectorV1 *_values1,
                VectorV2 *_values2, VectorV3 *_values3, VectorV4 *_values4,
                VectorV5 *_values5, VectorV6 *_values6, VectorV7 *_values7,
                VectorV8 *_values8, VectorV9 *_values9,
                VectorV10 *_values10) throw(MpiException);

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5, typename VectorV6, typename VectorV7,
         typename VectorV8, typename VectorV9, typename VectorV10>
void SampleSort(MpiComm comm, StrictWeakOrdering comp, VectorK *_keys,
                VectorV1 *_values1, VectorV2 *_values2, VectorV3 *_values3,
                VectorV4 *_values4, VectorV5 *_values5, VectorV6 *_values6,
                VectorV7 *_values7, VectorV8 *_values8, VectorV9 *_values9,
                VectorV10 *_values10) throw(MpiException);

#endif  // __USING_CPP_0X || __USING_CPP_11

template<typename StrictWeakOrdering, typename VectorK, typename Int>
void SampleSortWithGlobalIndexes(
    MpiComm comm, StrictWeakOrdering comp, const std::vector<Int> &old_sizes,
    VectorK *_keys, std::vector<Int> *_g_indexes) throw(MpiException);

}  // namespace gpusort

#include "samplesort_common_vector.inl"

#endif  // GPUSORT_SAMPLESORT_COMMON_VECTOR_H_
