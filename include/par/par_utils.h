#ifndef GPUSORT_PAR_UTILS_H_
#define GPUSORT_PAR_UTILS_H_

#include <stdint.h>
#include <vector>

#include "common.h"
#include "exceptions/mpi_exception.h"
#include "mpi/mpi_common.h"
#include "mpi/mpi_utils.h"
#include "omp/omp_utils.h"
#include "par/par_utils_helper.h"

#ifdef __DO_BENCHMARK
#include "timer.h"
#endif

namespace gpusort {

class ParUtils {
 public:
  // Arrange values by their global indexes
  template<typename Int, typename Vector>
  static void ArrangeValues(
      MpiComm comm, const std::vector<Int> &global_indexes,
      const std::vector<Int> &orig_sizes, Vector *_values) throw(MpiException) {
    typedef typename Vector::value_type ValueType;
    typedef std::pair<Int, Int> IdxPair;  // A pair of: <index, index>.
    typedef std::vector<IdxPair> IdxVector;  // A vector of IdxPair objects.
    typedef std::pair<Int, ValueType> IdxValuePair;

    Vector &values = *_values;
    Int nelem = global_indexes.size();
    Vector tmp_val(values);
    values.resize(nelem);

    int myrank = MpiUtils::MpiCommRank(comm);  // Rank of this process.
    int n_p = MpiUtils::MpiCommSize(comm);  // The number of MPI processes.
    int n_threads = OmpUtils::GetMaxThreads();
    int *receive_cnt = new int[n_p]();
    IdxVector *senders = new IdxVector[n_p];
    IdxValuePair **recv_buf = new IdxValuePair*[n_p];
    IdxPair **idx_info = new IdxPair*[n_p];
    MpiRequest *requests = new MpiRequest[2 * n_p];
    Int max_newsize = ParUtils::Max(comm, global_indexes.size());
    Int max_loops = (max_newsize / kMaxSend) +
                    ((max_newsize % kMaxSend == 0)? 0 : 1);
    ParUtilsHelper helper(comm);
    for (Int loop = 1; loop <= max_loops; loop++) {
      MpiUtils::MpiBarrier(comm);
      Int sent_cnt = (loop - 1) * kMaxSend;
      const Int *start_idx = &global_indexes[0] + sent_cnt;
      Int remain = nelem - sent_cnt;
      int count = (remain >= kMaxSend)? kMaxSend : remain;
      helper.FindSenders(start_idx, count, loop, &orig_sizes[0],
                         sizeof(ValueType), &tmp_val[0], &values[0], senders);
      helper.SyncSendRecv(senders, requests, receive_cnt, idx_info);
      SendRecv(comm, &tmp_val[0], senders, requests,
               receive_cnt, idx_info, recv_buf);
      // Assign values from receive buffer and free memory when finish.
      for (int rank = 0; rank < n_p; rank++) {
        int cnt = senders[rank].size();
        if (cnt == 0) continue;
        FOR_PARALLEL(n_threads, cnt, i,
                     {
                       IdxValuePair &tmp = recv_buf[rank][i];
                       Int idx = tmp.first;
                       values[idx] = tmp.second;
                     });
        senders[rank].clear();
        delete[] recv_buf[rank];
      }
    }

    // Free memory.
    delete[] receive_cnt;
    delete[] requests;
    delete[] senders;
    delete[] idx_info;
    delete[] recv_buf;

  #ifdef __DEBUG_MSG
    if (myrank == 0) std::cout << "Finish Arranging Values." << std::endl;
  #endif
    MpiUtils::MpiBarrier(comm);
  }

  template<typename T>
  static inline void Gather(MpiComm comm, T val,
                            std::vector<T> *_out) throw(MpiException) {
    int np = MpiUtils::MpiCommSize(comm);
    std::vector<T> &out = *_out;
    out.resize(np);
    ParUtilsHelper(comm).Gather(&val, MpiDataUtils::GetType<T>(), &out[0]);
  }

  template<typename T>
  static inline void GatherNElems(MpiComm comm, T nelem,
                                  std::vector<T> *_nelems) throw(MpiException) {
    ParUtilsHelper(comm).GatherNElems(nelem, _nelems);
  }

  template<typename Vector>
  static void KeepNoProcessHasEmptyData(MpiComm comm,
                                        Vector *_arr) throw(MpiException) {
    Vector &arr = *_arr;
    int n_p = MpiUtils::MpiCommSize(comm);
    int rank = MpiUtils::MpiCommRank(comm);
    int64_t nelem = arr.size();
    std::vector<int64_t> nelems(n_p);
    ParUtils::Gather(comm, nelem, &nelems);
    MpiStatus status;
    for (int i = 0; i < n_p; i++) {
      if (nelems[i] == 0) {
        int partner = i + 1;
        int empty_proc_cnt = 1;
        bool use_prev_proc = false;
        if (i == n_p - 1) {
          partner = n_p - 2;
          use_prev_proc = true;
        } else {
          // Find nearest neighbour which has sorted data.
          while (partner < n_p && nelems[partner] == 0) {
            partner++;
            empty_proc_cnt++;
          }
          // If there is no valid neighbour which has rank is greater than
          // the empty rank then we use the rank (i-1).
          if (partner == n_p) {
            partner = i - 1;
            use_prev_proc = true;
          }
        }
        int64_t size_sync = nelems[partner] / (empty_proc_cnt + 1);
      #ifdef __DEBUG_MSG
        if (rank == 0) {
          std::cout << "nelems: ";
          for (int j = 0; j < n_p; j++) {
            std::cout << nelems[j] << " ";
          }
          std::cout << std::endl;
          std::cout << "Vector in rank " << i << " is empty." << std::endl;
          std::cout << "Shifting " << size_sync << " data elements from rank "
                    << partner << " to it's " << empty_proc_cnt
                    << " nearest processes." << std::endl;
        }
      #endif
        if (rank >= i && (use_prev_proc || rank < partner)) {
          arr.resize(size_sync);
          MpiUtils::MpiRecv(comm, size_sync, partner,
                            0, &arr[0], &status);
        } else if (rank == partner) {
          int max_j = (use_prev_proc)? n_p : partner;
          for (int j = i; j < max_j; j ++) {
            int64_t idx = -1;
            if (use_prev_proc) {
              idx = nelem - (empty_proc_cnt - (j-i)) * size_sync;
            } else {
              idx = (j - i) * size_sync;
            }
            MpiUtils::MpiSend(comm, &arr[idx], size_sync, j, 0);
          }
          if (!use_prev_proc) {
            arr.erase(arr.begin(), arr.begin() + empty_proc_cnt * size_sync);
          } else {
            arr.erase(arr.begin() + nelem - empty_proc_cnt * size_sync,
                      arr.begin() + nelem);
          }
        }
        if (use_prev_proc) i = n_p;  // Break the loop.
        if (i < n_p - 1) i = partner + 1;  // Skip processed ranks.
      }
      MpiUtils::MpiBarrier(comm);
    }
  }

  template<typename T>
  static inline T Max(MpiComm comm, T val) throw(MpiException) {
    return ParUtilsHelper(comm).Max(val);
  }

  template<typename T>
  static inline T Min(MpiComm comm, T val) throw(MpiException) {
    return ParUtilsHelper(comm).Min(val);
  }

#ifdef __DO_BENCHMARK
  static inline void ShowBenchmark(
      MpiComm comm, const Timer &timer,
      const std::string &header) throw(MpiException) {
    ParUtilsHelper(comm).ShowBenchmark(timer, header);
  }
#endif  // __DO_BENCHMARK

  static inline void SplitCommBinary(MpiComm comm, int *splitter_rank,
                                     MpiComm *new_comm) throw(MpiException) {
    ParUtilsHelper(comm).SplitCommBinary(splitter_rank, new_comm);
  }

  static inline void SplitCommBinaryNoFlip(
      MpiComm comm, int *splitter_rank, MpiComm *new_comm) throw(MpiException) {
    ParUtilsHelper(comm).SplitCommBinaryNoFlip(splitter_rank, new_comm);
  }

  static inline void SplitCommUsingSplitter(
      MpiComm comm, int splitter_rank, MpiComm *new_comm) throw(MpiException) {
    ParUtilsHelper(comm).SplitCommUsingSplitter(splitter_rank, new_comm);
  }

  template<typename T>
  static inline T Sum(MpiComm comm, T val) throw(MpiException) {
    return ParUtilsHelper(comm).Sum(val);
  }

  template<typename T>
  static inline void Sum(MpiComm comm, const std::vector<T> &in,
                         std::vector<T> *_out) throw(MpiException) {
    ParUtilsHelper(comm).SumEx(in, _out);
  }

 private:
  template<typename RandomAccessIterator, typename IdxPair,
           typename IdxVector, typename IdxValuePair>
  static void SendRecv(MpiComm comm, RandomAccessIterator value_first,
                       const IdxVector *senders, MpiRequest *requests,
                       int *receive_cnt, IdxPair **idx_info,
                       IdxValuePair **recv_buf) {
    typedef typename std::iterator_traits<RandomAccessIterator>::value_type
        ValueType;

    int n_p = MpiUtils::MpiCommSize(comm);
    int myrank = MpiUtils::MpiCommRank(comm);
    int req_cnt = 0;
    std::vector<IdxValuePair*> data_to_free;
    for (int rank = 0; rank < n_p; rank++) {
      // First we must setup receive buffers.
      int cnt = senders[rank].size();
      if (cnt > 0) {
        recv_buf[rank] = new IdxValuePair[cnt];
        MpiUtils::MpiIRecv(comm, cnt, rank, rank,
                           recv_buf[rank], &requests[req_cnt++]);
      }

      // Send data to remote processes.
      size_t send_cnt = receive_cnt[rank];
      if (send_cnt == 0) continue;  // There is no real elements to send.
      receive_cnt[rank] = 0;  // Reset to be used for next call.
      IdxValuePair *send_data = new IdxValuePair[send_cnt];
      data_to_free.push_back(send_data);
      for (size_t i = 0; i < send_cnt; i++) {
        typename IdxPair::first_type local_idx = idx_info[rank][i].first;
        typename IdxPair::second_type remote_idx = idx_info[rank][i].second;
        send_data[i] = std::make_pair(remote_idx, value_first[local_idx]);
      }
      delete[] idx_info[rank];
      MpiUtils::MpiISSend(comm, &send_data[0], send_cnt,
                          rank, myrank, &requests[req_cnt++]);
    }

    // Wait for all receive requests being complete.
    MpiUtils::MpiWaitAll(req_cnt, requests);
    MpiUtils::MpiBarrier(comm);

    // Free sent data.
    while (!data_to_free.empty()) {
      IdxValuePair* tmp = data_to_free.back();
      data_to_free.pop_back();
      delete[] tmp;
    }
  }
};

}  // namespace gpusort

#endif  // GPUSORT_PAR_UTILS_H_
