#ifndef GPUSORT_HELPERS_SAMPLESORT_H_
#define GPUSORT_HELPERS_SAMPLESORT_H_

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common.h"
#include "exceptions/mpi_exception.h"
#include "mpi/mpi_common.h"
#include "mpi/mpi_utils.h"
#include "omp/omp_utils.h"
#include "par/par_utils.h"

#ifdef __DO_BENCHMARK
#include "timer.h"
#endif

namespace gpusort {

#define KEEP_HIGH 100
#define KEEP_LOW  101

class SampleSortHelper {
 public:
  SampleSortHelper(MpiComm comm) {
    comm_ = comm;
  }

  ~SampleSortHelper() {
  }

#ifdef __DO_BENCHMARK
  void ShowBenchmark() {
    int myrank = MpiUtils::MpiCommRank(comm_);
    if (myrank == 0) {
      const std::string sep = "\t\t\t\t";
      std::cout << "========== Benchmark information: ==========" << std::endl;
      std::cout << "Title:" << sep << "Mean:" << sep
                << "Min:" << sep << "Max:" << std::endl;
    }
    ParUtils::ShowBenchmark(comm_, local_sort_tm_, "Local sort");
    ParUtils::ShowBenchmark(comm_, select_splitters_tm_, "Select splitters");
    ParUtils::ShowBenchmark(comm_, communicate_tm_, "Exchange data");
    ParUtils::ShowBenchmark(comm_, merge_tm_, "Merge data");
    ParUtils::ShowBenchmark(comm_, total_tm_, "Total");
  }
#endif

  //
  // For normal input type.
  //
  template<typename Vector>
  inline void Sort(Vector *_arr) throw(MpiException) {
    typedef typename Vector::value_type T;
    Sort(std::less<T>(), _arr);
  }

  template<typename StrictWeakOrdering, typename Vector>
  void Sort(StrictWeakOrdering comp,
            Vector *_arr) throw(MpiException) {
  #ifdef __DO_BENCHMARK
    MpiUtils::MpiBarrier(comm_);
    total_tm_.Start();
  #endif
    Vector &arr = *_arr;
    int64_t nelem = arr.size();
    if (nelem == 0) throw MpiException("No data to sort.");
    int myrank = MpiUtils::MpiCommRank(comm_);
  #ifdef __DEBUG_MSG
    char dbg_msg[100] = { '\0' };
    sprintf(dbg_msg,
            "Rank %d is sorting data by SampleSort algorithm...",
            myrank);
    std::cout << dbg_msg << std::endl;
  #endif
    // Local sort.
  #ifdef __DEBUG_MSG
    if (myrank == 0) std::cout << "Start Local Sort using CPU." << std::endl;
  #endif
  #ifdef __DO_BENCHMARK
    local_sort_tm_.Start();
  #endif
    OmpUtils::MergeSort(comp, &arr[0], &arr[nelem]);
  #ifdef __DO_BENCHMARK
    local_sort_tm_.Stop();
  #endif
    int npes = MpiUtils::MpiCommSize(comm_);
    if (npes > 1) {
      // Do remaining phases after we finish Local sort.
      SortAmongProcesses(comp, &arr);
    }

  #ifdef __DO_BENCHMARK
    total_tm_.Stop();
  #endif
    MpiUtils::MpiBarrier(comm_);
  }

  //
  // For key-value input type.
  //
  template<typename StrictWeakOrdering, typename VectorK, typename VectorV>
  void SortByKey(StrictWeakOrdering comp, VectorK *_keys,
                 VectorV *_values) throw(MpiException) {
    typedef typename VectorK::value_type K;
    typedef typename VectorV::value_type V;
    typedef std::pair<K, V> KeyValuePair;

  #ifdef __DO_BENCHMARK
    MpiUtils::MpiBarrier(comm_);
    total_tm_.Start();
  #endif
    VectorK &keys = *_keys;
    VectorV &values = *_values;
    int64_t nelem = keys.size();
    if (nelem == 0) throw MpiException("No data to sort.");
    int myrank = MpiUtils::MpiCommRank(comm_);
    int n_threads = OmpUtils::GetMaxThreads();
    std::vector<KeyValuePair> items(nelem);
  #ifdef __DEBUG_MSG
    char dbg_msg[100] = { '\0' };
    sprintf(dbg_msg,
            "Rank %d is sorting key-value data by SampleSort algorithm...",
            myrank);
    std::cout << dbg_msg << std::endl;
  #endif
    // Local sort.
  #ifdef __DEBUG_MSG
    if (myrank == 0) std::cout << "Start Local Sort using CPU." << std::endl;
  #endif
  #ifdef __DO_BENCHMARK
    local_sort_tm_.Start();
  #endif
    FOR_PARALLEL(n_threads, nelem, i,
                 {
                   items[i] = std::make_pair(keys[i], values[i]);
                 });
    OmpUtils::MergeSort(ConvertComp<KeyValuePair>(comp),
                        &items[0], &items[nelem]);

  #ifdef __DO_BENCHMARK
    local_sort_tm_.Stop();
  #endif

    int npes = MpiUtils::MpiCommSize(comm_);
    if (npes > 1) {
      // Do remaining phases after we finish Local sort.
      SortAmongProcesses(ConvertComp<KeyValuePair>(comp), &items);
    }

    // Copy data back to user's buffer.
    int64_t new_size = items.size();
    keys.resize(new_size);
    values.resize(new_size);
    FOR_PARALLEL(n_threads, new_size, i,
                 {
                   keys[i] = items[i].first;
                   values[i] = items[i].second;
                 });

  #ifdef __DO_BENCHMARK
    total_tm_.Stop();
  #endif
    MpiUtils::MpiBarrier(comm_);
  }

 private:  // Private member functions.
  template<typename StrictWeakOrdering, typename Vector>
  void BitonicMerge(MpiComm comm, StrictWeakOrdering comp, int proc_set_size,
                    Vector *_local_list) throw(MpiException) {
    int partner = -1;
    int rank = MpiUtils::MpiCommRank(comm);
    int npes = MpiUtils::MpiCommSize(comm);
    bool is_increasing = IsIncreasing(comp);

    int num_left = GetPrevHighestPowerOfTwo(npes);
    int num_right = npes - num_left;
    // Do merge between the k right processes and the highest k left processes.
    if (rank < num_left && rank >= (num_left - num_right)) {
      partner = rank + num_right;
      int keep_what = (is_increasing)? KEEP_LOW : KEEP_HIGH;
      Merge(comm, comp, partner, keep_what, _local_list);
    } else if (rank >= num_left) {
      partner = rank - num_right;
      int keep_what = (is_increasing)? KEEP_HIGH : KEEP_LOW;
      Merge(comm, comp, partner, keep_what, _local_list);
    }
  }

  template<typename StrictWeakOrdering, typename Vector>
  void BitonicSort(MpiComm comm, StrictWeakOrdering comp,
                   Vector *_arr) throw(MpiException) {
    Vector &arr = *_arr;
    int rank = MpiUtils::MpiCommRank(comm);
    int npes = MpiUtils::MpiCommSize(comm);

    // Local Sort first.
    OmpUtils::MergeSort(comp, &arr[0], &arr[arr.size()]);

    if (npes == 1) return;

    if (IsPowerOfTwo(npes)) {  // If npes is a power of two ...
      BitonicSortBinary(comm, comp, _arr);
    } else {
      MpiComm new_comm;
      // Since npes is not a power of two, we shall split the problem in two ...
      //
      // 1. Create 2 comm groups ... one for the 2^d portion and one for the
      // remainder.
      int splitter = 0;
      ParUtils::SplitCommBinary(comm, &splitter, &new_comm);

      if (rank < splitter) {
        BitonicSortBinary(new_comm, comp, _arr);
      } else {
        BitonicSort(new_comm, comp, _arr);
      }
      MpiUtils::MpiCommFree(&new_comm);

      // 2. Do a special merge of the two segments. (original comm).
      BitonicMerge(comm, comp, GetNextHighestPowerOfTwo(npes), _arr);

      ParUtils::SplitCommBinaryNoFlip(comm, &splitter, &new_comm);

      // 3. Now a final sort on the segments.
      if (rank < splitter) {
        BitonicSortBinary(new_comm, comp, _arr);
      } else {
        BitonicSort(new_comm, comp, _arr);
      }
      MpiUtils::MpiCommFree(&new_comm);
    }
  }

  template<typename StrictWeakOrdering, typename Vector>
  void BitonicSortBinary(MpiComm comm, StrictWeakOrdering comp,
                         Vector *_arr) throw(MpiException) {
    typedef typename Vector::value_type T;

    int rank = MpiUtils::MpiCommRank(comm);
    int npes = MpiUtils::MpiCommSize(comm);
    bool is_increasing = IsIncreasing(comp);
    unsigned int and_bit = 2;
    for (int proc_set_size = 2;
         proc_set_size <= npes;
         proc_set_size *= 2, and_bit <<= 1) {
      if ((rank & and_bit) == 0) {
        BitonicSortBinaryHelper(comm, comp, proc_set_size,
                                is_increasing, _arr);
      } else {
        BitonicSortBinaryHelper(comm, comp, proc_set_size,
                                !is_increasing, _arr);
      }
    }
  }

  template<typename StrictWeakOrdering, typename Vector>
  void BitonicSortBinaryHelper(MpiComm comm, StrictWeakOrdering comp,
                               int proc_set_size, bool incr,
                               Vector *_local_list) throw(MpiException) {
    int myrank = MpiUtils::MpiCommRank(comm);
    int proc_set_dim = 0;
    int x = proc_set_size;
    while (x > 1) {
      x = x >> 1;
      proc_set_dim++;
    }

    int eor_bit = (1 << (proc_set_dim - 1) );
    for (int stage = 0; stage < proc_set_dim; stage++) {
      int partner = (myrank ^ eor_bit);
      bool cond = (incr)? (myrank < partner) : (myrank > partner);
      if (cond) {
        Merge(comm, comp, partner, KEEP_LOW, _local_list);
      } else {
        Merge(comm, comp, partner, KEEP_HIGH, _local_list);
      }
      eor_bit >>= 1;
    }
  }

  // Merge.
  template<typename StrictWeakOrdering, typename Vector>
  void Merge(MpiComm comm, StrictWeakOrdering comp,
             int partner, int keep_what, Vector *_arr) throw(MpiException) {
    Vector &local_list = *_arr;
    MpiStatus status;
    int send_size = local_list.size();
    int recv_size = 0;

    // First communicate how many you will send
    // and how many you will receive ...
    MpiUtils::MpiSendRecv(comm, &send_size , 1, partner, 0,
                          1, partner, 0, &recv_size, &status);

    Vector remote_list(recv_size);
    MpiUtils::MpiSendRecv(comm, &local_list[0], send_size, partner, 1,
                          recv_size, partner, 1, &remote_list[0], &status);

    MergeLists(comp, keep_what, remote_list, &local_list);
  }

  template<typename StrictWeakOrdering, typename Vector>
  void MergeLists(StrictWeakOrdering comp, int keep_what,
                  const Vector &list_b, Vector *_list_a) {
    typedef typename Vector::value_type T;

    Vector &list_a = *_list_a;
    bool is_increasing = IsIncreasing(comp);
    T low  = (is_increasing)? std::max(list_a[0], list_b[0]) :
                              std::max(list_a[list_a.size()-1],
                                       list_b[list_b.size()-1]);
    T high = (is_increasing)? std::min(list_a[list_a.size()-1],
                                       list_b[list_b.size()-1]) :
                              std::min(list_a[0], list_b[0]);
    // We will do a full merge first ...
    size_t list_size = list_a.size() + list_b.size();
    Vector scratch_list(list_size);
    unsigned int index1 = 0;
    unsigned int index2 = 0;

    for (size_t i = 0; i < list_size; i++) {
      // The order of (A || B) is important here,
      // so that index2 remains within bounds.
      bool order_cond = (is_increasing)? (list_a[index1] <= list_b[index2]) :
                                         (list_a[index1] > list_b[index2]);
      if (index1 < list_a.size() &&
          (index2 >= list_b.size() || order_cond)) {
        scratch_list[i] = list_a[index1];
        index1++;
      } else {
        scratch_list[i] = list_b[index2];
        index2++;
      }
    }

    // Scratch list is sorted at this point.
    list_a.clear();
    int half_list_size = list_size / 2;
    if (keep_what == KEEP_LOW) {
      if (is_increasing) {
        int ii = 0;
        while ((scratch_list[ii] < low || ii < half_list_size) &&
               scratch_list[ii] <= high) {
          ii++;
        }
        if (ii) {
          list_a.insert(list_a.end(), scratch_list.begin(),
                        scratch_list.begin() + ii);
        }
      } else {
        int ii = list_size - 1;
        while ((scratch_list[ii] <= high && ii >= half_list_size) ||
               scratch_list[ii] < low) {
          ii--;
        }
        if (ii < list_size - 1) {
          list_a.insert(list_a.end(), scratch_list.begin() + ii + 1,
                        scratch_list.begin() + list_size);
        }
      }
    } else {  // KEEP_HIGH
      if (is_increasing) {
        int ii = list_size - 1;
        while ((ii >= half_list_size && scratch_list[ii] >= low) ||
               scratch_list[ii] > high) {
          ii--;
        }
        if (ii < list_size - 1) {
          list_a.insert(list_a.begin(), scratch_list.begin() + ii + 1,
                        scratch_list.begin() + list_size);
        }
      } else {
        int ii = 0;
        while ((ii < half_list_size && scratch_list[ii] >= low) ||
               scratch_list[ii] > high) {
          ii++;
        }
        if (ii) {
          list_a.insert(list_a.end(), scratch_list.begin(),
                        scratch_list.begin() + ii);
        }
      }
    }
  }

  // Select splitters.
  template<typename StrictWeakOrdering, typename Vector>
  void SelectSplitters(StrictWeakOrdering comp, const Vector &arr,
                       Vector *_splitters) throw(MpiException) {
    Vector &splitters = *_splitters;
    int npes = MpiUtils::MpiCommSize(comm_);
    Vector send_splits(npes - 1);
    splitters.resize(npes);

    int n_threads = OmpUtils::GetMaxThreads();
    int n_loops = npes - 1;
    int64_t nelem = arr.size();

    FOR_PARALLEL(n_threads, n_loops, i,
                 {
                   send_splits[i] = arr[(i+1) * nelem / npes];
                 });

    // Sort send_splits using Bitonic.
    BitonicSort(comm_, comp, &send_splits);

    // All gather with the largest element of send_splits.
    bool is_increasing = IsIncreasing(comp);
    int idx = (is_increasing)? npes - 2 : 0;
    MpiUtils::MpiAllGather(comm_, &send_splits[idx], 1, &splitters[0]);
    typename Vector::iterator rm_it = (is_increasing)? splitters.end() - 1 :
                                                       splitters.begin();
    splitters.erase(rm_it);
  }

  // Do remaining phases after we finish Local sort.
  template<typename StrictWeakOrdering, typename Vector>
  void SortAmongProcesses(StrictWeakOrdering comp,
                          Vector *_arr) throw(MpiException) {
    Vector &arr = *_arr;
    int64_t nelem = arr.size();
    int npes = MpiUtils::MpiCommSize(comm_);
    int myrank = MpiUtils::MpiCommRank(comm_);
    int64_t tot_size = ParUtils::Sum(comm_, nelem);

    if (tot_size < (static_cast<int64_t>(5) * npes * npes)) {
    #ifdef __DEBUG_MSG
      if (myrank == 0) {
        std::cout << "Using Bitonic Sort since tot_size < (5*(npes^2)). "
                  << "tot_size = " << tot_size
                  << ", npes = " << npes << std::endl;
      }
    #endif

      MpiComm new_comm;
      bool need_free_comm = false;
      if (tot_size < npes) {
      #ifdef __DEBUG_MSG
        if (myrank == 0) {
          std::cout << " Input to sort is small. Splitting communicator: "
                    << npes << " -> " << tot_size <<std::endl;
        }
      #endif
        ParUtils::SplitCommUsingSplitter(comm_, tot_size, &new_comm);
        need_free_comm = true;
      } else {
        new_comm = comm_;
      }

      BitonicSort(new_comm, comp, _arr);
      if (need_free_comm) MpiUtils::MpiCommFree(&new_comm);
    } else {
      // Select Splitters.
    #ifdef __DEBUG_MSG
      if (myrank == 0) std::cout << "Selecting splitters..." << std::endl;
    #endif
    #ifdef __DO_BENCHMARK
      select_splitters_tm_.Start();
    #endif
      Vector splitters;
      SelectSplitters(comp, arr, &splitters);
    #ifdef __DO_BENCHMARK
      select_splitters_tm_.Stop();
    #endif
      // Transfer data.
    #ifdef __DEBUG_MSG
      if (myrank == 0) std::cout << "Transferring data..." << std::endl;
    #endif
    #ifdef __DO_BENCHMARK
      communicate_tm_.Start();
    #endif
      TransferData(comp, splitters, _arr);
    #ifdef __DO_BENCHMARK
      communicate_tm_.Stop();
    #endif
      // Sort merged buffer.
    #ifdef __DEBUG_MSG
      if (myrank == 0) std::cout << "Merging..." << std::endl;
    #endif
    #ifdef __DO_BENCHMARK
      merge_tm_.Start();
    #endif
      OmpUtils::MergeSort(comp, &arr[0], &arr[arr.size()]);
    #ifdef __DO_BENCHMARK
      merge_tm_.Stop();
    #endif
    #ifdef __DO_BENCHMARK
      communicate_tm_.Start();
    #endif
      // In some special cases, there might be some processes has an empty
      // result vector due to there are so many equal keys in input data.
      // So we must shift data from it's valid nearest neighbour process.
      ParUtils::KeepNoProcessHasEmptyData(comm_, _arr);
    #ifdef __DO_BENCHMARK
      communicate_tm_.Stop();
    #endif
    }
  }

  // Transfer data.
  template<typename StrictWeakOrdering, typename Vector>
  void TransferData(StrictWeakOrdering comp, const Vector &splitters,
                    Vector *_arr) throw(MpiException) {
    typedef typename Vector::value_type T;

    Vector &arr = *_arr;
    int npes = MpiUtils::MpiCommSize(comm_);
    int n_threads = OmpUtils::GetMaxThreads();
    int64_t nelem = arr.size();
    int *sendcnts = new int[npes];
    int *recvcnts = new int[npes];
    int *sdispls = new int[npes];
    int *rdispls = new int[npes];

    FOR_PARALLEL(n_threads, npes, k,
                 {
                   sendcnts[k] = 0;
                 });
 
    int64_t *loc_splits = new int64_t[npes - 1];
    int n_splits = npes - 1;
    FOR_PARALLEL(n_threads, n_splits, i,
                 {
                   loc_splits[i] = std::upper_bound(&arr[0], &arr[nelem],
                                                    splitters[i], comp) -
                                   &arr[0];
                 });

    FOR_PARALLEL(n_threads, n_splits, i,
                 {
                   int64_t prev_loc = (i == 0)? 0 : loc_splits[i-1];
                   sendcnts[i] = loc_splits[i] - prev_loc;
                 });
    sendcnts[n_splits] = nelem - loc_splits[n_splits-1];

    delete[] loc_splits;
    Vector buff;
    try {
      MpiUtils::MpiAllToAll(comm_, sendcnts, 1, recvcnts);
      sdispls[0] = 0; rdispls[0] = 0;
      OmpUtils::Scan(npes, sendcnts, sdispls);
      OmpUtils::Scan(npes, recvcnts, rdispls);

      int64_t nsorted = rdispls[npes - 1] + recvcnts[npes - 1];
      buff.resize(nsorted);
      MpiUtils::MpiAllToAllV(comm_, &arr[0], sendcnts, sdispls,
                             recvcnts, rdispls, &buff[0]);
    } catch (MpiException &e) {
      delete[] sendcnts;
      delete[] recvcnts;
      delete[] sdispls;
      delete[] rdispls;
      throw e;
    }

    std::swap(arr, buff);
    
    delete[] sendcnts;
    delete[] recvcnts;
    delete[] sdispls;
    delete[] rdispls;
  }

 private:
  MpiComm comm_;

#ifdef __DO_BENCHMARK
  Timer local_sort_tm_;
  Timer select_splitters_tm_;
  Timer communicate_tm_;
  Timer merge_tm_;
  Timer total_tm_;
#endif
};

}  // namespace gpusort

#endif  // GPUSORT_HELPERS_SAMPLESORT_H_
