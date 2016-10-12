#ifndef GPUSORT_HYKSORT_COMMON_VECTOR_H_
#define GPUSORT_HYKSORT_COMMON_VECTOR_H_

#include <vector>

#include "exceptions/mpi_exception.h"
#include "exceptions/thrust_exception.h"
#include "mpi/mpi_common.h"

namespace gpusort {

template<typename Vector>
void HykSort(MpiComm comm, unsigned int k_way, bool is_using_gpu,
             Vector *_keys) throw(MpiException, ThrustException);

template<typename StrictWeakOrdering, typename Vector>
void HykSort(MpiComm comm, StrictWeakOrdering comp,
             unsigned int k_way, bool is_using_gpu,
             Vector *_keys) throw(MpiException, ThrustException);

template<typename KeyType, typename KeyFunc, typename Vector>
void HykSort(MpiComm comm, unsigned int k_way,
             bool is_using_gpu, KeyFunc key_func,
             Vector *_list) throw(MpiException, ThrustException);

template<typename KeyType, typename StrictWeakOrdering,
         typename KeyFunc, typename Vector>
void HykSort(MpiComm comm, StrictWeakOrdering comp,
             unsigned int k_way, bool is_using_gpu, KeyFunc key_func,
             Vector *_list) throw(MpiException, ThrustException);

template<typename VectorK, typename VectorV>
void HykSort(MpiComm comm, unsigned int k_way,
             bool is_using_gpu, VectorK *_keys,
             VectorV *_values) throw(MpiException, ThrustException);

template<typename StrictWeakOrdering, typename VectorK, typename VectorV>
void HykSort(MpiComm comm, StrictWeakOrdering comp, unsigned int k_way,
             bool is_using_gpu, VectorK *_keys,
             VectorV *_values) throw(MpiException, ThrustException);

#if defined(__USING_CPP_0X) || defined(__USING_CPP_11)
template<typename VectorK, typename... Vectors>
void HykSort(
    MpiComm comm, unsigned int k_way,
    bool is_using_gpu, VectorK *_keys,
    Vectors*... _values) throw(MpiException, ThrustException);

template<typename StrictWeakOrdering, typename VectorK,
         typename... Vectors>
void HykSort(
    MpiComm comm, StrictWeakOrdering comp, unsigned int k_way,
    bool is_using_gpu, VectorK *_keys,
    Vectors*... _values) throw(MpiException, ThrustException);

#else  // !__USING_CPP_0X && !__USING_CPP_11
template<typename VectorK, typename VectorV1, typename VectorV2>
void HykSort(MpiComm comm, unsigned int k_way, bool is_using_gpu,
             VectorK *_keys, VectorV1 *_values1,
             VectorV2 *_values2) throw(MpiException, ThrustException);

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2>
void HykSort(MpiComm comm, StrictWeakOrdering comp, unsigned int k_way,
             bool is_using_gpu, VectorK *_keys, VectorV1 *_values1,
             VectorV2 *_values2) throw(MpiException, ThrustException);

template<typename VectorK, typename VectorV1, typename VectorV2,
         typename VectorV3>
void HykSort(MpiComm comm, unsigned int k_way, bool is_using_gpu,
             VectorK *_keys, VectorV1 *_values1, VectorV2 *_values2,
             VectorV3 *_values3) throw(MpiException, ThrustException);

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3>
void HykSort(MpiComm comm, StrictWeakOrdering comp, unsigned int k_way,
             bool is_using_gpu, VectorK *_keys,
             VectorV1 *_values1, VectorV2 *_values2,
             VectorV3 *_values3) throw(MpiException, ThrustException);

template<typename VectorK, typename VectorV1, typename VectorV2,
         typename VectorV3, typename VectorV4>
void HykSort(MpiComm comm, unsigned int k_way, bool is_using_gpu,
             VectorK *_keys, VectorV1 *_values1,
             VectorV2 *_values2, VectorV3 *_values3,
             VectorV4 *_values4) throw(MpiException, ThrustException);

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4>
void HykSort(MpiComm comm, StrictWeakOrdering comp, unsigned int k_way,
             bool is_using_gpu, VectorK *_keys, VectorV1 *_values1,
             VectorV2 *_values2, VectorV3 *_values3,
             VectorV4 *_values4) throw(MpiException, ThrustException);

template<typename VectorK, typename VectorV1, typename VectorV2,
         typename VectorV3, typename VectorV4, typename VectorV5>
void HykSort(MpiComm comm, unsigned int k_way, bool is_using_gpu,
             VectorK *_keys, VectorV1 *_values1, VectorV2 *_values2,
             VectorV3 *_values3, VectorV4 *_values4,
             VectorV5 *_values5) throw(MpiException, ThrustException);

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5>
void HykSort(MpiComm comm, StrictWeakOrdering comp, unsigned int k_way,
             bool is_using_gpu, VectorK *_keys, VectorV1 *_values1,
             VectorV2 *_values2, VectorV3 *_values3, VectorV4 *_values4,
             VectorV5 *_values5) throw(MpiException, ThrustException);

template<typename VectorK, typename VectorV1, typename VectorV2,
         typename VectorV3, typename VectorV4, typename VectorV5,
         typename VectorV6>
void HykSort(MpiComm comm, unsigned int k_way, bool is_using_gpu,
             VectorK *_keys, VectorV1 *_values1, VectorV2 *_values2,
             VectorV3 *_values3, VectorV4 *_values4, VectorV5 *_values5,
             VectorV6 *_values6) throw(MpiException, ThrustException);

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5, typename VectorV6>
void HykSort(MpiComm comm, StrictWeakOrdering comp, unsigned int k_way,
             bool is_using_gpu, VectorK *_keys, VectorV1 *_values1,
             VectorV2 *_values2, VectorV3 *_values3,
             VectorV4 *_values4, VectorV5 *_values5,
             VectorV6 *_values6) throw(MpiException, ThrustException);

template<typename VectorK, typename VectorV1, typename VectorV2,
         typename VectorV3, typename VectorV4, typename VectorV5,
         typename VectorV6, typename VectorV7>
void HykSort(MpiComm comm, unsigned int k_way,
             bool is_using_gpu, VectorK *_keys_first,
             VectorV1 *_values1, VectorV2 *_values2,
             VectorV3 *_values3, VectorV4 *_values4,
             VectorV5 *_values5, VectorV6 *_values6,
             VectorV7 *_values7) throw(MpiException, ThrustException);

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5, typename VectorV6, typename VectorV7>
void HykSort(MpiComm comm, StrictWeakOrdering comp, unsigned int k_way,
             bool is_using_gpu, VectorK *_keys, VectorV1 *_values1,
             VectorV2 *_values2, VectorV3 *_values3, VectorV4 *_values4,
             VectorV5 *_values5, VectorV6 *_values6,
             VectorV7 *_values7) throw(MpiException, ThrustException);

template<typename VectorK, typename VectorV1, typename VectorV2,
         typename VectorV3, typename VectorV4, typename VectorV5,
         typename VectorV6, typename VectorV7, typename VectorV8>
void HykSort(MpiComm comm, unsigned int k_way, bool is_using_gpu,
             VectorK *_keys, VectorV1 *_values1, VectorV2 *_values2,
             VectorV3 *_values3, VectorV4 *_values4, VectorV5 *_values5,
             VectorV6 *_values6, VectorV7 *_values7,
             VectorV8 *_values8) throw(MpiException, ThrustException);

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5, typename VectorV6, typename VectorV7,
         typename VectorV8>
void HykSort(MpiComm comm, StrictWeakOrdering comp, unsigned int k_way,
             bool is_using_gpu, VectorK *_keys, VectorV1 *_values1,
             VectorV2 *_values2, VectorV3 *_values3, VectorV4 *_values4,
             VectorV5 *_values5, VectorV6 *_values6, VectorV7 *_values7,
             VectorV8 *_values8) throw(MpiException, ThrustException);

template<typename VectorK, typename VectorV1, typename VectorV2,
         typename VectorV3, typename VectorV4, typename VectorV5,
         typename VectorV6, typename VectorV7, typename VectorV8,
         typename VectorV9>
void HykSort(MpiComm comm, unsigned int k_way, bool is_using_gpu,
             VectorK *_keys, VectorV1 *_values1, VectorV2 *_values2,
             VectorV3 *_values3, VectorV4 *_values4, VectorV5 *_values5,
             VectorV6 *_values6, VectorV7 *_values7, VectorV8 *_values8,
             VectorV9 *_values9) throw(MpiException, ThrustException);

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5, typename VectorV6, typename VectorV7,
         typename VectorV8, typename VectorV9>
void HykSort(MpiComm comm, StrictWeakOrdering comp, unsigned int k_way,
             bool is_using_gpu, VectorK *_keys, VectorV1 *_values1,
             VectorV2 *_values2, VectorV3 *_values3, VectorV4 *_values4,
             VectorV5 *_values5, VectorV6 *_values6,
             VectorV7 *_values7, VectorV8 *_values8,
             VectorV9 *_values9) throw(MpiException, ThrustException);

template<typename VectorK, typename VectorV1, typename VectorV2,
         typename VectorV3, typename VectorV4, typename VectorV5,
         typename VectorV6, typename VectorV7, typename VectorV8,
         typename VectorV9, typename VectorV10>
void HykSort(MpiComm comm, unsigned int k_way, bool is_using_gpu,
             VectorK *_keys, VectorV1 *_values1, VectorV2 *_values2,
             VectorV3 *_values3, VectorV4 *_values4, VectorV5 *_values5,
             VectorV6 *_values6, VectorV7 *_values7,
             VectorV8 *_values8, VectorV9 *_values9,
             VectorV10 *_values10) throw(MpiException, ThrustException);

template<typename StrictWeakOrdering, typename VectorK, typename VectorV1,
         typename VectorV2, typename VectorV3, typename VectorV4,
         typename VectorV5, typename VectorV6, typename VectorV7,
         typename VectorV8, typename VectorV9, typename VectorV10>
void HykSort(MpiComm comm, StrictWeakOrdering comp, unsigned int k_way,
             bool is_using_gpu, VectorK *_keys, VectorV1 *_values1,
             VectorV2 *_values2, VectorV3 *_values3, VectorV4 *_values4,
             VectorV5 *_values5, VectorV6 *_values6, VectorV7 *_values7,
             VectorV8 *_values8, VectorV9 *_values9,
             VectorV10 *_values10) throw(MpiException, ThrustException);

#endif  // __USING_CPP_0X || __USING_CPP_11

template<typename StrictWeakOrdering, typename VectorK, typename Int>
void HykSortWithGlobalIndexes(
    MpiComm comm, StrictWeakOrdering comp, unsigned int k_way,
    bool is_using_gpu, const std::vector<Int> &old_sizes, VectorK *_keys,
    std::vector<Int> *_g_indexes) throw(MpiException, ThrustException);

}  // namespace gpusort

#include "hyksort_common_vector.inl"

#endif  // GPUSORT_HYKSORT_COMMON_VECTOR_H_
