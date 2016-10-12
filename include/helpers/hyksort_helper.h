#ifndef GPUSORT_HELPERS_HYKSORT_H_
#define GPUSORT_HELPERS_HYKSORT_H_

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common.h"
#include "cuda/cuda_custom_type.h"
#include "cuda/cuda_utils.h"
#include "exceptions/mpi_exception.h"
#include "exceptions/thrust_exception.h"
#include "mpi/mpi_common.h"
#include "mpi/mpi_utils.h"
#include "omp/omp_utils.h"
#include "par/par_utils.h"
#include "random.h"

#ifdef __DO_BENCHMARK
#include "timer.h"
#endif

namespace gpusort {

#ifdef __DO_BENCHMARK
// Global local sort timers.
extern Timer local_sort_mem_getinfo_tm;
extern Timer local_sort_merge_tm;
extern Timer local_sort_sort_tm;
extern Timer local_sort_transfer_tm;
#endif

class HykSortHelper {
 public:
  HykSortHelper(MpiComm comm, unsigned int k_way, bool is_using_gpu) {
    comm_ = comm;
    orig_comm_ = comm;
    k_way_ = k_way;
    is_using_gpu_ = is_using_gpu;
    is_comm_changed_ = false;
    int rank = MpiUtils::MpiCommRank(comm);
    random_generator = (rank);

  #ifdef __DO_BENCHMARK
    // Reset local sort timers.
    local_sort_mem_getinfo_tm.Reset();
    local_sort_merge_tm.Reset();
    local_sort_sort_tm.Reset();
    local_sort_transfer_tm.Reset();
  #endif
  }

  ~HykSortHelper() {
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
    ParUtils::ShowBenchmark(orig_comm_, local_sort_sort_tm, "Local sort sort");
    ParUtils::ShowBenchmark(orig_comm_, local_sort_transfer_tm,
                            "Local sort transfer");
    ParUtils::ShowBenchmark(orig_comm_, local_sort_merge_tm,
                            "Local sort merge");
    ParUtils::ShowBenchmark(orig_comm_, local_sort_mem_getinfo_tm,
                            "Local sort get meminfo");

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
  // For normal input type.
  //
  template<typename Vector>
  inline void Sort(Vector *_arr) throw(MpiException, ThrustException) {
    typedef typename Vector::value_type T;
    Sort(std::less<T>(), _arr);
  }

  template<typename StrictWeakOrdering, typename Vector>
  void Sort(StrictWeakOrdering comp,
            Vector *_arr) throw(MpiException, ThrustException) {
    typedef typename Vector::value_type T;

  #ifdef __DO_BENCHMARK
    MpiUtils::MpiBarrier(comm_);
    total_tm_.Start();
  #endif
    Vector &arr = *_arr;
    int64_t nelem = arr.size();
    if (nelem == 0) throw ThrustException("No data to sort.");
    int rank = MpiUtils::MpiCommRank(comm_);

    // Local sort.
  #ifdef __DO_BENCHMARK
    local_sort_tm_.Start();
  #endif
    is_using_gpu_ = (CudaUtils::IsHavingDevices())? is_using_gpu_ : false;
  #ifdef __DEBUG_MSG
    char dbg_msg[100] = { '\0' };
    sprintf(dbg_msg,
            "Rank %d is sorting data by HykSort algorithm...\n"
            "is_using_gpu = %s, kway = %d",
            rank, ((is_using_gpu_)? "true" : "false"), k_way_);
    std::cout << dbg_msg << std::endl;
  #endif
    if (is_using_gpu_) {
    #ifdef __DEBUG_MSG
      if (rank == 0) std::cout << "Start Local Sort using GPU." << std::endl;
    #endif
      CudaUtils::Sort(comp, rank, &arr[0], &arr[nelem]);
    } else {
    #ifdef __DEBUG_MSG
      if (rank == 0) std::cout << "Start Local Sort using CPU." << std::endl;
    #endif
      OmpUtils::MergeSort(comp, &arr[0], &arr[nelem]);
    }
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
  // For user-defined input type.
  //
  template<typename KeyType, typename StrictWeakOrdering,
           typename KeyFunc, typename Vector>
  void Sort(StrictWeakOrdering comp, KeyFunc key_func,
            Vector *_arr) throw(MpiException, ThrustException) {
    typedef typename Vector::value_type T;

  #ifdef __DO_BENCHMARK
    MpiUtils::MpiBarrier(comm_);
    total_tm_.Start();
  #endif
    Vector &arr = *_arr;
    int64_t nelem = arr.size();
    if (nelem == 0) throw ThrustException("No data to sort.");
    int rank = MpiUtils::MpiCommRank(comm_);
    int n_threads = OmpUtils::GetMaxThreads();

    // Local sort.
  #ifdef __DO_BENCHMARK
    local_sort_tm_.Start();
  #endif
    is_using_gpu_ = (CudaUtils::IsHavingDevices())? is_using_gpu_ : false;
  #ifdef __DEBUG_MSG
    char dbg_msg[100] = { '\0' };
    sprintf(dbg_msg,
            "Rank %d is sorting user-defined data by HykSort algorithm...\n"
            "is_using_gpu = %s, kway = %d",
            rank, ((is_using_gpu_)? "true" : "false"), k_way_);
    std::cout << dbg_msg << std::endl;
  #endif
    if (is_using_gpu_) {
      if (key_func == NULL)
        throw ThrustException("Can not determine input keys.");
    #ifdef __DEBUG_MSG
      if (rank == 0) std::cout << "Start Local Sort using GPU." << std::endl;
    #endif
      std::vector<CudaCustomType<KeyType> > c_list(nelem);
      FOR_PARALLEL(n_threads, nelem, i,
                   {
                     KeyType key = key_func(arr[i]);
                     c_list[i] = CudaCustomType<KeyType>(
                                     key, static_cast<uint64_t>(i));
                   });
      CudaUtils::Sort(ConvertComp<CudaCustomType<KeyType> >(comp),
                      rank, &c_list[0], &c_list[nelem]);
      Vector tmp(arr);
      FOR_PARALLEL(n_threads, nelem, i,
                   {
                     uint64_t idx = c_list[i].idx();
                     arr[i] = tmp[idx];
                   });
    } else {
    #ifdef __DEBUG_MSG
      if (rank == 0) std::cout << "Start Local Sort using CPU." << std::endl;
    #endif
      OmpUtils::MergeSort(comp, &arr[0], &arr[nelem]);
    }
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
                 VectorV *_values) throw(MpiException, ThrustException) {
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
    if (nelem == 0) throw ThrustException("No data to sort.");
    int rank = MpiUtils::MpiCommRank(comm_);
    int n_threads = OmpUtils::GetMaxThreads();
    std::vector<KeyValuePair> items(nelem);

    // Local sort.
  #ifdef __DO_BENCHMARK
    local_sort_tm_.Start();
  #endif
    is_using_gpu_ = (CudaUtils::IsHavingDevices())? is_using_gpu_ : false;
  #ifdef __DEBUG_MSG
    char dbg_msg[100] = { '\0' };
    sprintf(dbg_msg,
            "Rank %d is sorting key-value data by HykSort algorithm...\n"
            "is_using_gpu = %s, kway = %d",
            rank, ((is_using_gpu_)? "true" : "false"), k_way_);
    std::cout << dbg_msg << std::endl;
  #endif
    if (is_using_gpu_) {
    #ifdef __DEBUG_MSG
      if (rank == 0) std::cout << "Start Local Sort using GPU." << std::endl;
    #endif
      CudaUtils::SortByKey(comp, rank, &keys[0], &keys[nelem],
                           &values[0], &items[0]);
    } else {
    #ifdef __DEBUG_MSG
      if (rank == 0) std::cout << "Start Local Sort using CPU." << std::endl;
    #endif
      FOR_PARALLEL(n_threads, nelem, i,
                   {
                     items[i] = std::make_pair(keys[i], values[i]);
                   });
      OmpUtils::MergeSort(ConvertComp<KeyValuePair>(comp),
                          &items[0], &items[nelem]);
    }
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
  inline void ChangeComm(MpiComm new_comm) {
    if (is_comm_changed_) MpiUtils::MpiCommFree(&comm_);
    comm_ = new_comm;
    is_comm_changed_ = true;
  }

  // Merge.
  template<typename StrictWeakOrdering, typename Vector>
  void Merge(StrictWeakOrdering comp, const std::vector<int64_t> &recv_disp,
             int merg_indx, std::vector<MpiRequest> *_reqst,
             Vector *_arr1, Vector *_arr2, Vector *_arr) throw(MpiException) {
    typedef typename Vector::value_type ValueType;

    std::vector<MpiRequest> &reqst = *_reqst;
    Vector &arr1 = *_arr1, &arr2 = *_arr2, &arr = *_arr;
    while (merg_indx <= k_way_) {
      MpiUtils::MpiWaitAll(1, &reqst[(merg_indx-1) * 2]);
      MpiUtils::MpiWaitAll(1, &reqst[(merg_indx-2) * 2]);
      ValueType *a = &arr1[0], *b = &arr2[0];
      for (int s = 2; merg_indx % s == 0; s *= 2) {
        OmpUtils::Merge(comp, &a[recv_disp[merg_indx - s/2]],
                        &a[recv_disp[merg_indx]], &a[recv_disp[merg_indx - s]],
                        &a[recv_disp[merg_indx - s/2]],
                        &b[recv_disp[merg_indx - s]]);
        Swap(&a, &b);
      }
      merg_indx += 2;
    }
    // Swap buffers.
    int swap_cond = 0;
    for (int s = 2; k_way_ % s == 0; s *= 2) swap_cond++;
    std::swap(arr, (swap_cond % 2)? arr2 : arr1);
  }

  // Select splitters.
  template<typename StrictWeakOrdering, typename RandomAccessIterator>
  void SelectSplitters(StrictWeakOrdering comp, int n_p,
                       RandomAccessIterator in_first,
                       RandomAccessIterator in_last,
                       RandomAccessIterator out_first) throw(MpiException) {
    unsigned int kway = k_way_ - 1;
    int64_t nelem = in_last - in_first;
    int64_t tot_size = ParUtils::Sum(comm_, nelem);
    double tol = 1e-2 / kway;
    int beta = pow(1.0/tol, 1.0/3.0) * 3.0;
    std::vector<int64_t> start(kway, 0);
    std::vector<int64_t> end(kway, nelem);
    std::vector<int64_t> exp_rank(kway);
    for (int i = 0; i < kway; i++)
      exp_rank[i] = ((i + 1) * tot_size) / k_way_;

    int64_t max_error = tot_size;
    int64_t cond = tot_size * tol;
    while (max_error > cond) {
      SelectSplittersHelper(comp, n_p, exp_rank, beta, in_first, in_last,
                            &start, &end, out_first, &max_error);
    }
  }

  template<typename StrictWeakOrdering, typename RandomAccessIterator>
  void SelectSplittersHelper(StrictWeakOrdering comp, int n_p,
                             const std::vector<int64_t> &exp_rank, int beta,
                             RandomAccessIterator in_first,
                             RandomAccessIterator in_last,
                             std::vector<int64_t> *_start,
                             std::vector<int64_t> *_end,
                             RandomAccessIterator out_first,
                             int64_t *max_err) throw(MpiException) {
    typedef typename std::iterator_traits<RandomAccessIterator>::value_type
        ValueType;

    std::vector<int64_t>& start = *_start;
    std::vector<int64_t>& end = *_end;
    int64_t nelem = in_last - in_first;
    int n_threads = OmpUtils::GetMaxThreads();
    int kway = exp_rank.size();
    std::vector<int64_t> loc_size(kway), tot_size(kway);
    for (int i = 0; i < kway; i++)
      loc_size[i] = end[i] - start[i];
    ParUtils::Sum(comm_, loc_size, &tot_size);

    std::vector<ValueType> loc_splt;
    for (int i = 0; i < kway; i++) {
      int splt_count = (tot_size[i] == 0)? 1 :
                           beta * (end[i]-start[i]) / tot_size[i];
      if (n_p > beta) {
        double r = random_generator.Next<float>(0, tot_size[i]);
        splt_count = (r < beta * loc_size[i])? 1 : 0;
      }
      for (int j = 0; j < splt_count; j++) {
        int64_t r = random_generator.Next<int64_t>(0, loc_size[i]);
        int64_t idx = start[i] + r;
        loc_splt.push_back(in_first[idx]);
      }
      std::sort(&loc_splt[loc_splt.size() - splt_count],
                &loc_splt[loc_splt.size()], comp);
    }

    int splt_count = loc_splt.size();
    // Gather all splitters. O( log(p) )
    int glb_splt_count;
    std::vector<int> glb_splt_cnts(n_p);
    std::vector<int> glb_splt_disp(n_p, 0);
    MpiUtils::MpiAllGather(comm_, &splt_count, 1, &glb_splt_cnts[0]);
    OmpUtils::Scan(n_p, &glb_splt_cnts[0], &glb_splt_disp[0]);
    glb_splt_count = glb_splt_cnts[n_p - 1] + glb_splt_disp[n_p - 1];
    std::vector<ValueType> glb_splt(glb_splt_count);
    MpiUtils::MpiAllGatherV(comm_, &loc_splt[0], splt_count, &glb_splt_cnts[0],
                            &glb_splt_disp[0], &glb_splt[0]);
    std::sort(&glb_splt[0], &glb_splt[glb_splt_count], comp);

    // Rank splitters. O( log(N/p) + log(p) )
    std::vector<int64_t> loc_rank(glb_splt_count, 0);
    if (nelem > 0) {
      FOR_PARALLEL(
          n_threads, glb_splt_count, i,
          {
            loc_rank[i] = std::lower_bound(in_first, in_last,
                                           glb_splt[i], comp) -
                          in_first;
          });
    }
    std::vector<int64_t> glb_rank(glb_splt_count, 0);
    ParUtils::Sum(comm_, loc_rank, &glb_rank);

    int64_t new_max_err = 0;
    // #pragma omp parallel for
    for (int i = 0; i < kway; i++) {
      int ub_indx = std::upper_bound(&glb_rank[0], &glb_rank[glb_splt_count],
                                     exp_rank[i]) -
                    &glb_rank[0];
      int lb_indx = ub_indx - 1;
      if (lb_indx < 0) lb_indx = 0;
      int64_t err = labs(glb_rank[lb_indx] - exp_rank[i]);
      if (err < *max_err) {
        if (glb_rank[lb_indx] > exp_rank[i]) {
          start[i] = 0;
        } else {
          start[i] = loc_rank[lb_indx];
        }
        if (ub_indx == glb_splt_count) {
          end[i] = nelem;
        } else {
          end[i] = loc_rank[ub_indx];
        }
        out_first[i] = glb_splt[lb_indx];
        if (new_max_err < err) new_max_err = err;
      }
    }
    *max_err = new_max_err;
  }

  // Do remaining phases after we finish Local sort.
  template<typename StrictWeakOrdering, typename Vector>
  void SortAmongProcesses(StrictWeakOrdering comp,
                          Vector *_arr) throw(MpiException, ThrustException) {
    int rank = MpiUtils::MpiCommRank(comm_);
    int global_rank = rank;
    Vector &arr = *_arr;
    int64_t nelem = arr.size();
    int n_p = MpiUtils::MpiCommSize(comm_);
    // Dummy arrays for now ...
    Vector arr1(128), arr2;
    while (n_p > 1) {
      if (k_way_ > n_p) k_way_ = n_p;
      int blk_size = n_p / k_way_;
      int blk_id = rank / blk_size;
      // Determine splitters.
    #ifdef __DEBUG_MSG
      if (global_rank == 0) std::cout << "Selecting splitters..." << std::endl;
    #endif
    #ifdef __DO_BENCHMARK
      select_splitters_tm_.Start();
    #endif
      Vector split_keys(k_way_ - 1);
      SelectSplitters(comp, n_p, &arr[0], &arr[arr.size()], &split_keys[0]);
    #ifdef __DO_BENCHMARK
      select_splitters_tm_.Stop();
    #endif

      // Transfer data & Merge.
    #ifdef __DEBUG_MSG
      if (global_rank == 0) std::cout << "Transferring data..." << std::endl;
    #endif
    #ifdef __DO_BENCHMARK
      communicate_tm_.Start();
    #endif
      int merge_idx = 0;
      std::vector<MpiRequest> reqst(k_way_ * 2);
      std::vector<int64_t> recv_disp(k_way_ + 1, 0);
      TransferData(comp, blk_size, blk_id, rank, split_keys, arr,
                   &merge_idx, &recv_disp, &reqst, &arr1, &arr2);
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
      Merge(comp, recv_disp, merge_idx, &reqst, &arr1, &arr2, &arr);
    #ifdef __DO_BENCHMARK
      merge_tm_.Stop();
    #endif
      // Split comm. kway  O( log(p) ) ??
    #ifdef __DEBUG_MSG
      if (global_rank == 0) std::cout << "Splitting communicator..."
                                      << std::endl;
    #endif
    #ifdef __DO_BENCHMARK
      comm_split_tm_.Start();
    #endif
      MpiComm new_comm;
      MpiUtils::MpiCommSplit(comm_, blk_id, rank, &new_comm);
      ChangeComm(new_comm);

      n_p = MpiUtils::MpiCommSize(comm_);
      rank = MpiUtils::MpiCommRank(comm_);
    #ifdef __DO_BENCHMARK
      comm_split_tm_.Stop();
    #endif
      MpiUtils::MpiBarrier(comm_);
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

  template<typename T>
  inline void Swap(T **a, T **b) {
	  T* tmp = *a;
	  *a = *b;
	  *b = tmp;
  }

  // Transfer data and merge.
  template<typename StrictWeakOrdering, typename Vector>
  void TransferData(StrictWeakOrdering comp, int blk_size, int blk_id,
                    int rank, const Vector &split_keys, const Vector &arr,
                    int *merge_idx, std::vector<int64_t> *_recv_disp,
                    std::vector<MpiRequest> *_reqst, Vector *_arr1,
                    Vector *_arr2) throw(MpiException) {
    typedef typename Vector::value_type ValueType;

    Vector &arr1 = *_arr1, &arr2 = *_arr2;
    std::vector<MpiRequest> &reqst = *_reqst;
    std::vector<int64_t> &recv_disp = *_recv_disp;
    int64_t nelem = arr.size();
    int new_pid = rank % blk_size;
    std::vector<int64_t> send_size(k_way_), send_disp(k_way_+1);
    send_disp[0] = 0; send_disp[k_way_] = nelem;
    int n_threads = OmpUtils::GetMaxThreads();
    int n_loops = k_way_ - 1;
    FOR_PARALLEL(n_threads, n_loops, i,
                 {
                   send_disp[i+1] = std::lower_bound(&arr[0], &arr[nelem],
                                                     split_keys[i], comp) -
                                    &arr[0];
                 });
    for (int i = 0; i < k_way_; i++) {
      send_size[i] = send_disp[i+1] - send_disp[i];
    }

    // Get recv_size.
    int recv_iter = 0;
    std::vector<int64_t> recv_size(k_way_);
    int half_kway = k_way_ / 2;
    for (int _i = 0; _i <= half_kway; _i++) {
      int i1 = (blk_id + _i) % k_way_;
      int i2 = (blk_id + k_way_ - _i) % k_way_;
      MpiStatus status;
      int max_j = (_i == 0 || _i == half_kway)? 1 : 2;
      for (int j = 0; j < max_j; j++) {
        int i = (_i == 0)? i1 : (((j + blk_id/_i) % 2)? i1 : i2);
        int partner = blk_size * i + new_pid;
        MpiUtils::MpiSendRecv(comm_, &send_size[i], 1, partner, 0, 1,
                              partner, 0, &recv_size[recv_iter], &status);
        recv_disp[recv_iter + 1] = recv_disp[recv_iter] + recv_size[recv_iter];
        recv_iter++;
      }
    }

    // Communicate data.
    recv_iter = 0;
    int merg_indx = 2;
    std::vector<MpiStatus> status(k_way_ * 2);
    arr1.resize(recv_disp[k_way_]);
    arr2.resize(recv_disp[k_way_]);
    for (int _i = 0; _i <= half_kway; _i++) {
      int i1 = (blk_id + _i) % k_way_;
      int i2 = (blk_id + k_way_ - _i) % k_way_;
      int max_j = (_i == 0 || _i == half_kway)? 1 : 2;
      for (int j = 0; j < max_j; j++) {
        int i = (_i == 0)? i1 : (((j + blk_id/_i) % 2)? i1 : i2);
        int partner = blk_size * i + new_pid;
        int tag = partner * rank;
        MpiUtils::MpiIRecv(comm_, recv_size[recv_iter], partner, tag,
                           &arr1[recv_disp[recv_iter]], &reqst[recv_iter * 2]);
        MpiUtils::MpiISSend(comm_, &arr[send_disp[i]], send_size[i],
                            partner, tag, &reqst[recv_iter * 2 + 1]);
        recv_iter++;

        int flag[2] = {0, 0};
        if (recv_iter > merg_indx) {
          MpiUtils::MpiTest(&reqst[(merg_indx-1) * 2], &flag[0],
                            &status[(merg_indx-1) * 2]);
          MpiUtils::MpiTest(&reqst[(merg_indx-2) * 2], &flag[1],
                            &status[(merg_indx-2) * 2]);
        }
        if (!flag[0] || !flag[1]) continue;
        ValueType *a = &arr1[0], *b = &arr2[0];
        for (int s = 2; merg_indx % s == 0; s *= 2) {
          OmpUtils::Merge(comp, &a[recv_disp[merg_indx - s/2]],
                          &a[recv_disp[merg_indx]], &a[recv_disp[merg_indx-s]],
                          &a[recv_disp[merg_indx - s/2]],
                          &b[recv_disp[merg_indx - s]]);
          Swap(&a, &b);
        }
        merg_indx += 2;
      }
    }

    *merge_idx = merg_indx;
  }

 private:
  MpiComm comm_;
  MpiComm orig_comm_;
  unsigned int k_way_;
  bool is_using_gpu_;
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

#endif  // GPUSORT_HELPERS_HYKSORT_H_
