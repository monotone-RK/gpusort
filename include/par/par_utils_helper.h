#ifndef GPUSORT_PAR_UTILS_HELPER_H_
#define GPUSORT_PAR_UTILS_HELPER_H_

#include <vector>

#include "exceptions/mpi_exception.h"
#include "mpi/mpi_common.h"

#ifdef __DO_BENCHMARK
#include "timer.h"
#endif

namespace gpusort {

// Max number of elements to be sent each loop.
extern const int kMaxSend;

class ParUtilsHelper {
 public:
  ParUtilsHelper(MpiComm comm) {
    comm_ = comm;
  }

  ~ParUtilsHelper() {
  }

  // IdxPair: A pair of: <Int, Int> (<Remote index, Local index>).
  // IdxVector: A vector of IdxPair objects.
  template<typename Int, typename IdxVector>
  void FindSenders(const Int *idx_first, int count, Int loop,
                   const Int *nelems, size_t obj_size, void *orig_values,
                   void *values, IdxVector *senders);

  void Gather(const void *val, MpiDatatype d_type,
              void *out) throw(MpiException);

  template<typename Int>
  void GatherNElems(Int nelem, std::vector<Int> *_nelems) throw(MpiException);

  template<typename T>
  T Max(T val) throw(MpiException);

  template<typename T>
  T Min(T val) throw(MpiException);

#ifdef __DO_BENCHMARK
  void ShowBenchmark(const Timer &timer,
                     const std::string &header) throw(MpiException);
#endif

  void SplitCommBinary(int *splitter_rank,
                       MpiComm *new_comm) throw(MpiException);

  void SplitCommBinaryNoFlip(int *splitter_rank,
                             MpiComm *new_comm) throw(MpiException);

  void SplitCommUsingSplitter(int splitter_rank,
                              MpiComm *new_comm) throw(MpiException);

  template<typename T>
  T Sum(T val) throw(MpiException);

  template<typename T>
  void SumEx(const std::vector<T> &in,
             std::vector<T> *_out) throw(MpiException);

  template<typename IdxPair, typename IdxVector>
  void SyncSendRecv(const IdxVector *senders, MpiRequest *requests,
                    int *receive_cnt, IdxPair **idx_info) throw(MpiException);

 private:
  MpiComm comm_;
};

}  // namespace gpusort

#endif  // GPUSORT_PAR_UTILS_HELPER_H_
