#include "cuda/cuda_utils.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <inttypes.h>
#include <iterator>
#include <stdint.h>
#include <typeinfo>

#include <omp.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include "common.h"
#include "cuda/cuda_custom_type.h"
#include "omp/omp_utils.h"

#ifdef __DO_BENCHMARK
#include "timer.h"
#endif

namespace gpusort {

//
// Global timers.
//
#ifdef __DO_BENCHMARK
Timer local_sort_mem_getinfo_tm;
Timer local_sort_merge_tm;
Timer local_sort_sort_tm;
Timer local_sort_transfer_tm;
#endif

//
// Declare internal functions.
//
template<typename K>
static inline thrust::less<K> Convert2ThrustComp(std::less<K> comp) {
  return thrust::less<K>();
}

template<typename K>
static inline thrust::greater<K> Convert2ThrustComp(std::greater<K> comp) {
  return thrust::greater<K>();
}

template<typename StrictWeakOrdering, typename DifferenceType,
         typename RandomAccessIterator>
__host__ void ParallelMergeArrays(StrictWeakOrdering comp,
                                  int num_threads, int num_chunks,
                                  const DifferenceType *displs,
                                  RandomAccessIterator in,
                                  RandomAccessIterator out);

void RaiseError(cudaError_t err) throw(ThrustException);

void SelectDevice(int rank) throw(ThrustException);

template<typename StrictWeakOrdering, typename RandomAccessIterator>
void SortByThrust(StrictWeakOrdering comp,
                  RandomAccessIterator first,
                  RandomAccessIterator last,
                  RandomAccessIterator out) throw(ThrustException);

template<typename StrictWeakOrdering, typename RandomAccessIterator1,
         typename RandomAccessIterator2, typename KeyValuePair>
void SortByThrust(StrictWeakOrdering comp,
                  RandomAccessIterator1 key_first,
                  RandomAccessIterator1 key_last,
                  RandomAccessIterator2 val_first,
                  KeyValuePair *out) throw(ThrustException);

void SplitChunks(int64_t num_elems, int64_t size, int64_t *chunk_size,
                 int *num_chunks) throw(ThrustException);

//
// Implement Common API.
//
int CudaUtils::GetDeviceCount() throw(ThrustException) {
  int num_dev = 0;
  cudaError_t cuda_state = cudaGetDeviceCount(&num_dev);
  if (cuda_state != cudaSuccess) {
    RaiseError(cuda_state);
  }
  return num_dev;
}

bool CudaUtils::IsHavingDevices() throw(ThrustException) {
  return (GetDeviceCount() > 0);
}

//
// Implement Sort API.
//
template<typename StrictWeakOrdering, typename RandomAccessIterator>
__host__ void CudaUtils::Sort(
    StrictWeakOrdering comp, int rank, RandomAccessIterator first,
    RandomAccessIterator last) throw(ThrustException) {
  typedef typename std::iterator_traits<RandomAccessIterator>::difference_type
      DiffType;
  typedef typename std::iterator_traits<RandomAccessIterator>::value_type
      ValueType;

#ifdef __DEBUG_MSG
  if (rank == 0) std::cout << "start CudaUtils::Sort" << std::endl;
#endif
  int64_t size = last - first, chunk_size = 0;
  int num_chunks = 0;
  SelectDevice(rank);
  SplitChunks(size, sizeof(ValueType), &chunk_size, &num_chunks);
  int tree_height = ceil(log(num_chunks) / log(2));

  if (num_chunks == 1) {
    SortByThrust(comp, first, last, first);
  } else {
    int64_t remainder = size % chunk_size;
    RandomAccessIterator buffer = new ValueType[size];
    int64_t *displs = new DiffType[num_chunks+1];
    displs[0] = 0;
    displs[num_chunks] = size;

    // sort part
    RandomAccessIterator tmp_first = first;
    RandomAccessIterator tmp_last = first + chunk_size;
    RandomAccessIterator tmp_buf_first = buffer;
    RandomAccessIterator tmp_buf_last = buffer + chunk_size;
    for (int i = 0; i < num_chunks; i++) {
      displs[i+1] = displs[i] + chunk_size;
      if (i == num_chunks-2 && remainder != 0) {
        chunk_size = remainder;
      }
      try {
        if (tree_height % 2 == 0) {
          SortByThrust(comp, tmp_first, tmp_last, tmp_first);
        } else {
          SortByThrust(comp, tmp_first, tmp_last, tmp_buf_first);
        }
      } catch (ThrustException &e) {
        delete[] buffer;
        delete[] displs;
        throw e;
      }
      tmp_first = tmp_last;
      tmp_last = tmp_first + chunk_size;
      tmp_buf_first = tmp_buf_last;
      tmp_buf_last = tmp_buf_first + chunk_size;
    }

    int max_threads = OmpUtils::GetMaxThreads();
  #ifdef __DO_BENCHMARK
    local_sort_merge_tm.Start();
  #endif
    if (tree_height % 2 == 0) {
      ParallelMergeArrays(comp, max_threads, num_chunks, displs, first, buffer);
    } else {
      ParallelMergeArrays(comp, max_threads, num_chunks, displs, buffer, first);
    }
  #ifdef __DO_BENCHMARK
    local_sort_merge_tm.Stop();
  #endif

    delete[] buffer;
    delete[] displs;
  }
}

template<typename StrictWeakOrdering, typename RandomAccessIterator1,
         typename RandomAccessIterator2>
__host__ void CudaUtils::SortByKey(
    StrictWeakOrdering comp, int rank,
    RandomAccessIterator1 key_first, RandomAccessIterator1 key_last,
    RandomAccessIterator2 val_first,
    void *_out) throw(ThrustException) {
  typedef typename std::iterator_traits<RandomAccessIterator1>::difference_type
      DiffType;
  typedef typename std::iterator_traits<RandomAccessIterator1>::value_type
      KeyType;
  typedef typename std::iterator_traits<RandomAccessIterator2>::value_type
      ValueType;
  typedef std::pair<KeyType, ValueType> KeyValuePair;

  KeyValuePair *out = reinterpret_cast<KeyValuePair*>(_out);
  int64_t size = key_last - key_first, chunk_size = 0;
  int num_chunks = 0;
  SelectDevice(rank);
  SplitChunks(size, sizeof(ValueType) + sizeof(KeyType),
              &chunk_size, &num_chunks);
  int tree_height = ceil(log(num_chunks) / log(2));

  if (num_chunks == 1) {
    SortByThrust(comp, key_first, key_last, val_first, out);
  } else {
    int64_t remainder = size % chunk_size;
    KeyValuePair *buffer = new KeyValuePair[size];
    int64_t *displs = new DiffType[num_chunks+1];
    displs[0] = 0;
    displs[num_chunks] = size;

    // sort part
    int64_t d1 = 0;
    int64_t d2 = d1 + chunk_size;
    for (int i = 0; i < num_chunks; i++) {
      try {
        if (tree_height % 2 == 0) {
          SortByThrust(comp, key_first + d1, key_first + d2,
                       val_first + d1, out + d1);
        } else {
          SortByThrust(comp, key_first + d1, key_first + d2,
                       val_first + d1, buffer + d1);
        }
      } catch (ThrustException &e) {
        delete[] buffer;
        delete[] displs;
        throw e;
      }
      if (i < num_chunks - 1)
        displs[i+1] = displs[i] + chunk_size;

      d1 += chunk_size;
      d2 = d1 + ((i == num_chunks-2 && remainder != 0)? remainder : chunk_size);
    }

    int max_threads = OmpUtils::GetMaxThreads();
  #ifdef __DO_BENCHMARK
    local_sort_merge_tm.Start();
  #endif
    if (tree_height % 2 == 0) {
      ParallelMergeArrays(ConvertComp<KeyValuePair>(comp),
                          max_threads, num_chunks, displs, out, buffer);
    } else {
      ParallelMergeArrays(ConvertComp<KeyValuePair>(comp),
                          max_threads, num_chunks, displs, buffer, out);
    }
  #ifdef __DO_BENCHMARK
    local_sort_merge_tm.Stop();
  #endif

    delete[] buffer;
    delete[] displs;
  }
}

//
// Pre-define template parameters for Sort function
// to pass the compiler linkage.
//
#define DECLARE_SORT_TEMPLATE(key_t)\
template void CudaUtils::Sort<std::less<key_t>, key_t*>(\
    std::less<key_t> comp, int rank,\
    key_t* first, key_t* last) throw(ThrustException);\
template void CudaUtils::Sort<std::greater<key_t>, key_t*>(\
    std::greater<key_t> comp, int rank,\
    key_t* first, key_t* last) throw(ThrustException);

DECLARE_SORT_TEMPLATE(int);  // For primitive type: int.
DECLARE_SORT_TEMPLATE(unsigned int);  // For primitive type: unsigned int.
DECLARE_SORT_TEMPLATE(float);  // For primitive type: float.
DECLARE_SORT_TEMPLATE(double);  // For primitive type: double.
DECLARE_SORT_TEMPLATE(int64_t);  // For primitive type: int64_t.
DECLARE_SORT_TEMPLATE(uint64_t);  // For primitive type: uint64_t.
DECLARE_SORT_TEMPLATE(CudaCustomType<int>);  // For custom data type
                                             // (key type is int).
DECLARE_SORT_TEMPLATE(CudaCustomType<unsigned int>);  // For custom data type
                                                      // (key type is uint).
DECLARE_SORT_TEMPLATE(CudaCustomType<float>);  // For custom data type
                                               // (key type is float).
DECLARE_SORT_TEMPLATE(CudaCustomType<double>);  // For custom data type
                                                // (key type is double).
DECLARE_SORT_TEMPLATE(CudaCustomType<int64_t>);  // For custom data type
                                                 // (key type is int64_t).
DECLARE_SORT_TEMPLATE(CudaCustomType<uint64_t>);  // For custom data type
                                                  // (key type is uint64_t).

//
// Pre-define template parameters for SortByKey function
// to pass the compiler linkage.
//
#define DECLARE_SORT_BY_KEY_TEMPLATE(key_t, value_t)\
template void CudaUtils::SortByKey<std::less<key_t>, key_t*, value_t*>(\
    std::less<key_t> comp, int rank, key_t* k_first, key_t* k_last,\
    value_t* v_first, void *_out) throw(ThrustException);\
template void CudaUtils::SortByKey<std::greater<key_t>, key_t*, value_t*>(\
    std::greater<key_t> comp, int rank, key_t* k_first, key_t* k_last,\
    value_t* v_first, void *_out) throw(ThrustException);

DECLARE_SORT_BY_KEY_TEMPLATE(int, int);
DECLARE_SORT_BY_KEY_TEMPLATE(unsigned int, unsigned int);
DECLARE_SORT_BY_KEY_TEMPLATE(float, float);
DECLARE_SORT_BY_KEY_TEMPLATE(double, double);
DECLARE_SORT_BY_KEY_TEMPLATE(int64_t, int64_t);
DECLARE_SORT_BY_KEY_TEMPLATE(uint64_t, uint64_t);

#define DECLARE_SORT_BY_KEY_PAIR_TEMPLATE(type1, type2)\
  DECLARE_SORT_BY_KEY_TEMPLATE(type1, type2);\
  DECLARE_SORT_BY_KEY_TEMPLATE(type2, type1);

// int pair with the remainders.
DECLARE_SORT_BY_KEY_PAIR_TEMPLATE(int, unsigned int);
DECLARE_SORT_BY_KEY_PAIR_TEMPLATE(int, float);
DECLARE_SORT_BY_KEY_PAIR_TEMPLATE(int, double);
DECLARE_SORT_BY_KEY_PAIR_TEMPLATE(int, int64_t);
DECLARE_SORT_BY_KEY_PAIR_TEMPLATE(int, uint64_t);
// unsigned int pair with the remainders.
DECLARE_SORT_BY_KEY_PAIR_TEMPLATE(unsigned int, float);
DECLARE_SORT_BY_KEY_PAIR_TEMPLATE(unsigned int, double);
DECLARE_SORT_BY_KEY_PAIR_TEMPLATE(unsigned int, int64_t);
DECLARE_SORT_BY_KEY_PAIR_TEMPLATE(unsigned int, uint64_t);
// float pair with the remainders.
DECLARE_SORT_BY_KEY_PAIR_TEMPLATE(float, double);
DECLARE_SORT_BY_KEY_PAIR_TEMPLATE(float, int64_t);
DECLARE_SORT_BY_KEY_PAIR_TEMPLATE(float, uint64_t);
// double pair with the remainders.
DECLARE_SORT_BY_KEY_PAIR_TEMPLATE(double, int64_t);
DECLARE_SORT_BY_KEY_PAIR_TEMPLATE(double, uint64_t);
// int64_t pair with the remainders.
DECLARE_SORT_BY_KEY_PAIR_TEMPLATE(int64_t, uint64_t);

//
//  Implement internal functions.
//
template<typename StrictWeakOrdering, typename DifferenceType,
         typename RandomAccessIterator>
__host__ void ParallelMergeArrays(StrictWeakOrdering comp,
                                  int num_threads, int num_chunks,
                                  const DifferenceType *displs,
                                  RandomAccessIterator in,
                                  RandomAccessIterator out) {
  int current_height = 0;
  int num_merge = (num_chunks + 2 - 1) / 2;
  int stride = 1;
  while (num_merge > 0) {
    RandomAccessIterator first1, last1, first2, last2, first3;
    for (int i = 0; i < num_merge; i++) {
      int offset_idx = i * stride * 2;
      int idx_first1 = (offset_idx > num_chunks)? num_chunks : offset_idx;
      int idx_last1 = offset_idx + stride;
      idx_last1 = (idx_last1 > num_chunks)? num_chunks : idx_last1;
      int idx_first2 = offset_idx + stride;
      idx_first2 = (idx_first2 > num_chunks)? num_chunks : idx_first2;
      int idx_last2 = offset_idx + (2 * stride);
      idx_last2 = (idx_last2 > num_chunks)? num_chunks : idx_last2;
      int idx_first3 = offset_idx;

      if (current_height % 2 == 0) {
        first1 = &in[displs[idx_first1]];
        last1 = &in[displs[idx_last1]];
        first2 = &in[displs[idx_first2]];
        last2 = &in[displs[idx_last2]];
        first3 = &out[displs[idx_first3]];
      } else if (current_height % 2 == 1) {
        first1 = &out[displs[idx_first1]];
        last1 = &out[displs[idx_last1]];
        first2 = &out[displs[idx_first2]];
        last2 = &out[displs[idx_last2]];
        first3 = &in[displs[idx_first3]];
      }

      OmpUtils::Merge(comp, first1, last1, first2, last2, first3);
    }

    if (num_merge == 1) {
      num_merge = 0;
    } else {
      num_merge = (num_merge + 2 - 1) / 2;
      stride = stride * 2;
      current_height++;
    }
  }
}

void RaiseError(cudaError_t err) throw(ThrustException) {
  switch (err) {
    case cudaErrorDeviceAlreadyInUse:
      throw ThrustException("A call tried to access an exclusive-thread device"
                            " that is already in use by a different thread");

    case cudaErrorInsufficientDriver:
      throw ThrustException("The installed CUDA driver is older than"
                            " the CUDA runtime library");

    case cudaErrorInvalidDevice:
      throw ThrustException("The device ordinal supplied by the user does not"
                            " correspond to a valid CUDA device");

    case cudaErrorNoDevice:
      throw ThrustException("No CUDA-capable devices were detected");

    default:
      throw ThrustException("Unknown error occurred");
  }
}

void SelectDevice(int rank) throw(ThrustException) {
  int num_dev = CudaUtils::GetDeviceCount();
  cudaError_t cuda_state = cudaSetDevice((rank + 1) % num_dev);
  if (cuda_state != cudaSuccess) {
    RaiseError(cuda_state);
  }
}

template<typename StrictWeakOrdering, typename RandomAccessIterator>
void SortByThrust(StrictWeakOrdering comp,
                  RandomAccessIterator first, RandomAccessIterator last,
                  RandomAccessIterator out) throw(ThrustException) {
  typedef typename std::iterator_traits<RandomAccessIterator>::value_type
      ValueType;
#ifdef __DO_BENCHMARK
  local_sort_transfer_tm.Start();
#endif
  thrust::device_vector<ValueType> d_v(0);
  try {
    d_v.assign(first, last);
  } catch(std::bad_alloc &e) {
    throw ThrustException("Couldn't allocate device vector");
  }
#ifdef __DO_BENCHMARK
  local_sort_transfer_tm.Stop();

  local_sort_sort_tm.Start();
#endif
  try {
    thrust::sort(d_v.begin(), d_v.end(), Convert2ThrustComp(comp));
  } catch (std::bad_alloc &e) {
    throw ThrustException("Ran out of memory while sorting");
  } catch (thrust::system_error &e) {
    throw ThrustException("Some other error happened during sort: "
                          + std::string(e.what()));
  }
#ifdef __DO_BENCHMARK
  local_sort_sort_tm.Stop();

  local_sort_transfer_tm.Start();
#endif
  try {
    thrust::copy(d_v.begin(), d_v.end(), out);
  } catch (thrust::system_error &e) {
    throw ThrustException("Some other error happened during copy: "
                          + std::string(e.what()));
  }
#ifdef __DO_BENCHMARK
  local_sort_transfer_tm.Stop();
#endif
}

template<typename StrictWeakOrdering, typename RandomAccessIterator1,
         typename RandomAccessIterator2, typename KeyValuePair>
void SortByThrust(StrictWeakOrdering comp,
                  RandomAccessIterator1 key_first,
                  RandomAccessIterator1 key_last,
                  RandomAccessIterator2 val_first,
                  KeyValuePair *out) throw(ThrustException) {
  typedef typename std::iterator_traits<RandomAccessIterator1>::value_type
      KeyType;
  typedef typename std::iterator_traits<RandomAccessIterator2>::value_type
      ValueType;

  int64_t nelem = key_last - key_first;
#ifdef __DO_BENCHMARK
  local_sort_transfer_tm.Start();
#endif
  thrust::device_vector<KeyType> d_k(0);
  thrust::device_vector<ValueType> d_v(0);
  try {
    d_k.assign(key_first, key_last);
    d_v.assign(val_first, val_first + nelem);
  } catch(std::bad_alloc &e) {
    throw ThrustException("Couldn't allocate device vector");
  }
#ifdef __DO_BENCHMARK
  local_sort_transfer_tm.Stop();

  local_sort_sort_tm.Start();
#endif
  try {
    thrust::sort_by_key(d_k.begin(), d_k.end(), d_v.begin(),
                        Convert2ThrustComp(comp));
  } catch (std::bad_alloc &e) {
    throw ThrustException("Ran out of memory while sorting");
  } catch (thrust::system_error &e) {
    throw ThrustException("Some other error happened during sort: "
                          + std::string(e.what()));
  }
#ifdef __DO_BENCHMARK
  local_sort_sort_tm.Stop();

  local_sort_transfer_tm.Start();
#endif
  try {
    thrust::copy(d_k.begin(), d_k.end(), key_first);
    thrust::copy(d_v.begin(), d_v.end(), val_first);
  } catch (thrust::system_error &e) {
    throw ThrustException("Some other error happened during copy: "
                          + std::string(e.what()));
  }

  int n_threads = OmpUtils::GetMaxThreads();
  FOR_PARALLEL(n_threads, nelem, i,
               {
                 out[i] = std::make_pair(key_first[i], val_first[i]);
               });
#ifdef __DO_BENCHMARK
  local_sort_transfer_tm.Stop();
#endif
}

void SplitChunks(int64_t num_elems, int64_t size, int64_t *chunk_size,
                 int *num_chunks) throw(ThrustException) {
  size_t mem_avai = 1, mem_total = 1;
  local_sort_mem_getinfo_tm.Start();
  cudaError_t cuda_state = cudaMemGetInfo(&mem_avai, &mem_total);
  local_sort_mem_getinfo_tm.Stop();
  if (cuda_state != cudaSuccess) RaiseError(cuda_state);
  int64_t chunk_size_ = mem_avai / (3 * size);
  int num_chunks_ = (num_elems + chunk_size_ - 1) / chunk_size_;

  *chunk_size = chunk_size_;
  *num_chunks = num_chunks_;
}

}  // namespace gpusort
