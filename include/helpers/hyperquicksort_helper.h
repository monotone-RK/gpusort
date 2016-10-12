#ifndef GPUSORT_HELPERS_HYPERQUICKSORT_H_
#define GPUSORT_HELPERS_HYPERQUICKSORT_H_

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
#include "random.h"

#ifdef __DO_BENCHMARK
#include "timer.h"
#endif

namespace gpusort {

class HyperQuickSortHelper {
 public:
  HyperQuickSortHelper(MpiComm comm) {
    comm_ = comm;
    orig_comm_ = comm;
    is_comm_changed_ = false;
    int rank = MpiUtils::MpiCommRank(comm);
    random_generator = (rank);
  }

  ~HyperQuickSortHelper() {
    if (is_comm_changed_) MpiUtils::MpiCommFree(&comm_);
  }

#ifdef __DO_BENCHMARK
  void ShowBenchmark() {
    int myrank = MpiUtils::MpiCommRank(orig_comm_);
    if (myrank == 0) {
      const std::string sep = "\t\t\t\t";
      std::cout << "========== Benchmark information: ==========" << std::endl;
      std::cout << "Title:" << sep << "Mean:" << sep
                << "Min:" << sep << "Max:" << std::endl;
    }
    ParUtils::ShowBenchmark(orig_comm_, local_sort_tm_, "Local sort");
    ParUtils::ShowBenchmark(orig_comm_, select_splitters_tm_,
                            "Select splitters");
    ParUtils::ShowBenchmark(orig_comm_, communicate_tm_, "Exchange data");
    ParUtils::ShowBenchmark(orig_comm_, merge_tm_, "Merge data");
    ParUtils::ShowBenchmark(orig_comm_, comm_split_tm_, "Split comm");
    ParUtils::ShowBenchmark(orig_comm_, total_tm_, "Total");
  }
#endif

  //
  // For normal input type
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
            "Rank %d is sorting data by HyperQuickSort algorithm...",
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
  // For key-value input type
  //
  template<typename StrictWeakOrdering, typename VectorK, typename VectorV>
  inline void SortByKey(
      StrictWeakOrdering comp, VectorK *_keys,
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
            "Rank %d is sorting key-value data by HyperQuickSort algorithm...",
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

 private:  // Private member functions
  inline void ChangeComm(MpiComm new_comm) {
    if (is_comm_changed_) MpiUtils::MpiCommFree(&comm_);
    comm_ = new_comm;
    is_comm_changed_ = true;
  }

   // Merge
  template<typename StrictWeakOrdering, typename Vector>
  void Merge(StrictWeakOrdering comp, int64_t lsize,
             int64_t rsize, typename Vector::value_type *lbuff,
             Vector *_comm_buff, Vector *_merge_buff,
             Vector *_arr) throw(MpiException) {
    typedef typename Vector::value_type T;

    Vector &merge_buff = *_merge_buff;
    Vector &comm_buff = *_comm_buff;
    Vector &arr = *_arr;
    int nbuff_size = lsize + rsize;
    merge_buff.resize(nbuff_size);

    OmpUtils::Merge(comp, lbuff, &lbuff[lsize], &comm_buff[0],
                    &comm_buff[rsize], &merge_buff[0]);

    // Copy new data.
    std::swap(arr, merge_buff);
  }

  // Select splitters
  template<typename StrictWeakOrdering, typename T>
  void SelectSplitters(StrictWeakOrdering comp, int npes, int myrank,
                       const T *arr, int64_t nelem, int64_t *tot_size,
                       T *split_key) throw(MpiException) {
    int n_threads = OmpUtils::GetMaxThreads();
    std::vector<int> glb_splt_cnts(npes);
    std::vector<int> glb_splt_disp(npes, 0);

    // Take random splitters.
    // O( 1 ) -- Let p * splt_count = glb_splt_count = const = 100~1000
    int splt_count = (1000 * nelem) / *tot_size;
    if (npes > 1000) {
      float r = random_generator.Next<float>(0, *tot_size);
      splt_count = (r < 1000 * nelem)? 1 : 0;
    }
    if (splt_count > nelem) splt_count = nelem;

    std::vector<T> splitters(splt_count);
    for (size_t i = 0; i < splt_count; i++) {
      int64_t r = random_generator.Next<int64_t>(0, nelem - 1);
      splitters[i] = arr[r];
    }

    // Gather all splitters. O( log(p) )
    int glb_splt_count;
    MpiUtils::MpiAllGather(comm_, &splt_count, 1, &glb_splt_cnts[0]);
    OmpUtils::Scan(npes, &glb_splt_cnts[0], &glb_splt_disp[0]);
    glb_splt_count = glb_splt_cnts[npes-1] + glb_splt_disp[npes-1];

    std::vector<T> glb_splitters(glb_splt_count);
    MpiUtils::MpiAllGatherV(comm_, &splitters[0], splt_count, &glb_splt_cnts[0],
                            &glb_splt_disp[0], &glb_splitters[0]);

    // Determine split key. O( log(N/p) + log(p) )
    std::vector<int64_t> disp(glb_splt_count, 0);
    if (nelem > 0) {
      FOR_PARALLEL(n_threads, glb_splt_count, i,
                   {
                     disp[i] = std::lower_bound(&arr[0], &arr[nelem],
                                                glb_splitters[i], comp) -
                               &arr[0];
                   });
    }
    std::vector<int64_t> glb_disp;
    ParUtils::Sum(comm_, disp, &glb_disp);

    int64_t* split_disp = &glb_disp[0];
    for (size_t i = 0; i < glb_splt_count; i++)
      if (std::abs(glb_disp[i] - *tot_size/2) <
          std::abs(*split_disp - *tot_size/2))
        split_disp = &glb_disp[i];
    *split_key = glb_splitters[split_disp - &glb_disp[0]];

    *tot_size = (myrank <= (npes-1)/2)? *split_disp : *tot_size - *split_disp;
  }

  // Do remaining phases after we finish Local sort.
  template<typename StrictWeakOrdering, typename Vector>
  void SortAmongProcesses(StrictWeakOrdering comp,
                          Vector *_arr) throw(MpiException) {
    typedef typename Vector::value_type T;

    Vector &arr = *_arr;
    int64_t nelem = arr.size();
    Vector comm_buff;
    Vector merge_buff;
    int npes = MpiUtils::MpiCommSize(comm_);
    int myrank = MpiUtils::MpiCommRank(comm_);
    int global_rank = myrank;
    int64_t tot_size = ParUtils::Sum(comm_, nelem);
    while (npes > 1 && tot_size > 0) {
      // Determine splitters.
    #ifdef __DEBUG_MSG
      if (global_rank == 0) std::cout << "Selecting splitters..." << std::endl;
    #endif
      T split_key;
      T *lbuff = NULL;
      int64_t lsize = 0, rsize = 0;
    #ifdef __DO_BENCHMARK
      select_splitters_tm_.Start();
    #endif
      SelectSplitters(comp, npes, myrank, &arr[0], nelem,
                      &tot_size, &split_key);
    #ifdef __DO_BENCHMARK
      select_splitters_tm_.Stop();
    #endif
      // Transfer data.
      // Split problem into two. O( N/p )
    #ifdef __DEBUG_MSG
      if (global_rank == 0) std::cout << "Transferring data..." << std::endl;
    #endif
    #ifdef __DO_BENCHMARK
      communicate_tm_.Start();
    #endif
      int split_id = (npes - 1) / 2;
      TransferData(comp, npes, myrank, split_id, &arr[0], nelem,
                   split_key, &lsize, &rsize, &lbuff, &comm_buff);
    #ifdef __DO_BENCHMARK
      communicate_tm_.Stop();
    #endif
      // Merge remaining parts.
    #ifdef __DEBUG_MSG
      if (global_rank == 0) std::cout << "Merging..." << std::endl;
    #endif
    #ifdef __DO_BENCHMARK
      merge_tm_.Start();
    #endif
      Merge(comp, lsize, rsize, lbuff, &comm_buff, &merge_buff, _arr);
    #ifdef __DO_BENCHMARK
      merge_tm_.Stop();
    #endif
      // Split comm. O( log(p) ) ??
    #ifdef __DEBUG_MSG
      if (global_rank == 0) std::cout << "Splitting communicator..."
                                      << std::endl;
    #endif
    #ifdef __DO_BENCHMARK
      comm_split_tm_.Start();
    #endif
      MpiComm new_comm;
      MpiUtils::MpiCommSplit(comm_, myrank <= split_id, myrank, &new_comm);
      ChangeComm(new_comm);

      npes = (myrank <= split_id)? split_id + 1 : npes - split_id - 1;
      myrank = (myrank <= split_id)? myrank : myrank - split_id - 1;
      nelem = lsize + rsize;
    #ifdef __DO_BENCHMARK
      comm_split_tm_.Stop();
    #endif
    }

    // In some special cases, there might be some processes has an empty
    // result vector due to there are so many equal keys in input data.
    // So we must shift data from it's valid nearest neighbour process.
  #ifdef __DO_BENCHMARK
    communicate_tm_.Start();
  #endif
    ParUtils::KeepNoProcessHasEmptyData(orig_comm_, _arr);
  #ifdef __DO_BENCHMARK
    communicate_tm_.Stop();
  #endif
  }

  // Transfer data.
  template<typename StrictWeakOrdering, typename Vector>
  void TransferData(StrictWeakOrdering comp, int npes, int myrank,
                    int split_id, const typename Vector::value_type *arr,
                    int64_t nelem, const typename Vector::value_type &split_key,
                    int64_t *lsize, int64_t *rsize,
                    typename Vector::value_type** lbuff,
                    Vector *_comm_buff) throw(MpiException) {
    typedef typename Vector::value_type T;
    Vector &comm_buff = *_comm_buff;
    int new_p0 = (myrank <= split_id)? 0 : split_id + 1;
    int cmp_p0 = (myrank > split_id)? 0 : split_id + 1;

    int partner = myrank + cmp_p0 - new_p0;
    if (partner >= npes) partner = npes - 1;

    // Exchange send sizes.
    *rsize = 0;
    size_t split_indx =
        (nelem > 0)? (std::lower_bound(&arr[0], &arr[nelem], split_key, comp) -
                      &arr[0]) : 0;
    int64_t ssize = (myrank > split_id)? split_indx : nelem - split_indx;
    const T *sbuff = (myrank > split_id)? &arr[0] : &arr[split_indx];
    *lsize = (myrank <= split_id)? split_indx : nelem - split_indx;
    *lbuff = const_cast<T*>((myrank <= split_id)? &arr[0] : &arr[split_indx]);

    MpiStatus status;
    MpiUtils::MpiSendRecv(comm_, &ssize, 1, partner, 0,
                          1, partner, 0, rsize, &status);

    // Exchange data.
    comm_buff.resize(*rsize);
    MpiUtils::MpiSendRecv(comm_, sbuff, ssize, partner, 0,
                          *rsize, partner, 0, &comm_buff[0], &status);
  }

 private:
  MpiComm comm_;
  MpiComm orig_comm_;
  bool is_comm_changed_;
  Random random_generator;

#ifdef __DO_BENCHMARK
  Timer local_sort_tm_;
  Timer select_splitters_tm_;
  Timer communicate_tm_;
  Timer merge_tm_;
  Timer comm_split_tm_;
  Timer total_tm_;
#endif
};

}  // namespace gpusort

#endif  // GPUSORT_HELPERS_HYPERQUICKSORT_H_
