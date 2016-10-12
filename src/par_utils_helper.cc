#include "par/par_utils_helper.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>

#include "common.h"
#include "mpi/mpi_impl.h"
#include "mpi/mpi_utils.h"
#include "omp/omp_utils.h"

namespace gpusort {

const int kMaxSend = 100000;  // Max number of elements to be sent each loop.

//
// Find out what process is containing a data element
// that idx is it's global index.
//
// PosInfo: A pair of <Rank of sender, Remote index>.
template<typename Int>
inline bool FindSender(const Int *nelems, int np, Int idx, void *pos) {
  // A pair of <Rank of sender, Remote index>.
  typedef std::pair<int, Int> PosInfo;

  int i = std::upper_bound(&nelems[0], &nelems[np], idx) - &nelems[0];
  Int r_idx = (i == 0)? idx : idx - nelems[i-1];
  PosInfo *_pos = reinterpret_cast<PosInfo*>(pos);
  *_pos = static_cast<PosInfo>(std::make_pair(i, r_idx));
  return true;
}

//
// Find out which processes are containing data of mine and which positions
// the data is located in.
//
// IdxPair: A pair of: <Int, Int> (<Remote index, Local index>).
// IdxVector: A vector of IdxPair objects.
template<typename Int, typename IdxVector>
void ParUtilsHelper::FindSenders(const Int *idx_first, int count,
                                 Int loop, const Int *nelems, size_t obj_size,
                                 void *orig_values, void *values,
                                 IdxVector *senders) {
  // A pair of <Rank of sender, Remote index>.
  typedef std::pair<int, Int> PosInfo;

  int np = MpiUtils::MpiCommSize(comm_);
  int myrank = MpiUtils::MpiCommRank(comm_);
  int n_threads = OmpUtils::GetMaxThreads();
  // Create buffers for each searching thread.
  std::vector<std::vector<IdxVector> > tmp_senders(n_threads);
  for (int i = 0; i < n_threads; i++) {
    tmp_senders[i].resize(np);
  }
  // Parallel search and save results to buffers of each thread.
  FOR_PARALLEL(
      n_threads, count, i,
      {
        PosInfo pos;
        FindSender(nelems, np, idx_first[i], &pos);
        int src = pos.first;
        Int remote_idx = pos.second;
        Int local_idx = (loop - 1) * kMaxSend + i;
        if (src == myrank) {
          std::memcpy(GetAddress(values, static_cast<uint64_t>(local_idx),
                                 obj_size),
                      GetAddress(orig_values, static_cast<uint64_t>(remote_idx),
                                 obj_size),
                      obj_size);
          continue;
        }

        int t_id = OmpUtils::GetThreadId();
        tmp_senders[t_id][src].push_back(std::make_pair(remote_idx, local_idx));
      });

  // Merge all threads' buffers to the target buffer.
  for (int i = 0; i < n_threads; i++) {
    FOR_PARALLEL(n_threads, np, p,
                 {
                   IdxVector &target = senders[p];
                   IdxVector &v = tmp_senders[i][p];
                   target.insert(target.end(), &v[0], &v[v.size()]);
                 });
  }
}
//
// Pre-define template parameters for FindSenders function
// to pass the compiler linkage.
//
#define DECLARE_FIND_SENDERS(int_t)\
template void ParUtilsHelper::FindSenders<\
    int_t, std::vector<std::pair<int_t, int_t> > >(\
        const int_t *idx_first, int count, int_t loop, const int_t *nelems,\
        size_t obj_size, void* orig_values, void* values,\
        std::vector<std::pair<int_t, int_t> > *senders);

DECLARE_FIND_SENDERS(int);
DECLARE_FIND_SENDERS(unsigned int);
DECLARE_FIND_SENDERS(int64_t);
DECLARE_FIND_SENDERS(uint64_t);

void ParUtilsHelper::Gather(const void *val, MpiDatatype d_type,
                            void *out) throw(MpiException) {
  MpiImpl::MpiAllGather(comm_, val, 1, d_type, 1, d_type, out);
}

template<typename Int>
void ParUtilsHelper::GatherNElems(
    Int nelem, std::vector<Int> *_nelems) throw(MpiException) {
  int np = MpiUtils::MpiCommSize(comm_);
  std::vector<Int> &nelems = *_nelems;
  nelems.resize(np);
  Gather(&nelem, MpiDataUtils::GetType<Int>(), &nelems[0]);
  for (int i = 1; i < np; i++) {
    nelems[i] += nelems[i - 1];
  }
}
//
// Pre-define template parameters for GatherNElems function
// to pass the compiler linkage.
//
#define DECLARE_GATHER_N_ELEMS(int_t)\
template void ParUtilsHelper::GatherNElems<int_t>(\
    int_t nelem, std::vector<int_t> *_nelems) throw(MpiException);

DECLARE_GATHER_N_ELEMS(int);
DECLARE_GATHER_N_ELEMS(unsigned int);
DECLARE_GATHER_N_ELEMS(int64_t);
DECLARE_GATHER_N_ELEMS(uint64_t);

template<typename T>
T ParUtilsHelper::Max(T val) throw(MpiException) {
  int np = MpiUtils::MpiCommSize(comm_);
  T *vals = new T[np];
  MpiUtils::MpiAllGather(comm_, &val, 1, vals);
  return *std::max_element(&vals[0], &vals[np]);
}

template<typename T>
T ParUtilsHelper::Min(T val) throw(MpiException) {
  int np = MpiUtils::MpiCommSize(comm_);
  T *vals = new T[np];
  MpiUtils::MpiAllGather(comm_, &val, 1, vals);
  return *std::min_element(&vals[0], &vals[np]);
}

template<typename T>
T ParUtilsHelper::Sum(T val) throw(MpiException) {
  T sum = 0;
  MpiUtils::MpiAllReduce(comm_, &val, 1, kMpiSum, &sum);
  return sum;
}

template<typename T>
void ParUtilsHelper::SumEx(const std::vector<T> &in,
                           std::vector<T> *_out) throw(MpiException) {
  std::vector<T> &out = *_out;
  size_t cnt = in.size();
  out.resize(cnt);
  MpiUtils::MpiAllReduce(comm_, &in[0], cnt, kMpiSum, &out[0]);
}

//
// Pre-define template parameters for Math functions: Max, Min, Sum
// to pass the compiler linkage.
//
#define DECLARE_MATH_FUNC(func, data_t)\
template data_t ParUtilsHelper::func(data_t val) throw(MpiException);

#define DECLARE_MATH_FUNCS(data_t)\
  DECLARE_MATH_FUNC(Max, data_t);\
  DECLARE_MATH_FUNC(Min, data_t);\
  DECLARE_MATH_FUNC(Sum, data_t);

DECLARE_MATH_FUNCS(char);
DECLARE_MATH_FUNCS(unsigned char);
DECLARE_MATH_FUNCS(short);
DECLARE_MATH_FUNCS(unsigned short);
DECLARE_MATH_FUNCS(int);
DECLARE_MATH_FUNCS(unsigned int);
DECLARE_MATH_FUNCS(int64_t);
DECLARE_MATH_FUNCS(uint64_t);
DECLARE_MATH_FUNCS(float);
DECLARE_MATH_FUNCS(double);
DECLARE_MATH_FUNCS(long double);

//
// Pre-define template parameters for SumEx function
// to pass the compiler linkage.
//
#define DECLARE_SUM_EX_FUNC(data_t)\
template void ParUtilsHelper::SumEx(\
    const std::vector<data_t> &in,\
    std::vector<data_t> *_out) throw(MpiException);

DECLARE_SUM_EX_FUNC(char);
DECLARE_SUM_EX_FUNC(unsigned char);
DECLARE_SUM_EX_FUNC(short);
DECLARE_SUM_EX_FUNC(unsigned short);
DECLARE_SUM_EX_FUNC(int);
DECLARE_SUM_EX_FUNC(unsigned int);
DECLARE_SUM_EX_FUNC(int64_t);
DECLARE_SUM_EX_FUNC(uint64_t);
DECLARE_SUM_EX_FUNC(float);
DECLARE_SUM_EX_FUNC(double);
DECLARE_SUM_EX_FUNC(long double);

#ifdef __DO_BENCHMARK
// Show benchmark.
void ParUtilsHelper::ShowBenchmark(
    const Timer &timer, const std::string &header) throw(MpiException) {
  int npes = MpiUtils::MpiCommSize(comm_);
  int myrank = MpiUtils::MpiCommRank(comm_);
  double seconds = const_cast<Timer&>(timer).Seconds();
  double total = Sum(seconds);
  double mean = total / npes;
  double min = Min(seconds);
  double max = Max(seconds);
  const int field_width = 35;
  if (myrank != 0) return;
  std::cout << std::left << std::setw(field_width) << header;
  char tmp[field_width] = { '\0' };
  sprintf(tmp, "%.9f", mean);
  std::cout << std::setw(field_width) << tmp;
  sprintf(tmp, "%.9f", min);
  std::cout << std::setw(field_width) << tmp;
  sprintf(tmp, "%.9f", max);
  std::cout << std::setw(field_width) << tmp << std::endl;
}
#endif  // __DO_BENCHMARK

//
// Split communicators.
//
void ParUtilsHelper::SplitCommBinary(int *splitter_rank,
                                     MpiComm *new_comm) throw(MpiException) {
  int npes = MpiUtils::MpiCommSize(comm_);
  int rank = MpiUtils::MpiCommRank(comm_);
  MpiGroup orig_group, new_group;

  *splitter_rank = GetPrevHighestPowerOfTwo(npes);

  // Determine sizes for the 2 groups.
  int *ranks_asc = new int[*splitter_rank];
  int *ranks_desc = new int[npes - *splitter_rank];

  int num_asc = 0;
  int num_desc = npes - *splitter_rank - 1;
  // This is the main mapping between old ranks and new ranks.
  for (int i = 0; i < npes; i++) {
    if (i < *splitter_rank) {
      ranks_asc[num_asc] = i;
      num_asc++;
    } else {
      ranks_desc[num_desc] = i;
      num_desc--;
    }
  }

  try {
    MpiUtils::MpiCommGroup(comm_, &orig_group);

    // Divide tasks into two distinct groups based upon rank.
    if (rank < *splitter_rank) {
      MpiUtils::MpiGroupIncl(orig_group, *splitter_rank, ranks_asc, &new_group);
    }else {
      MpiUtils::MpiGroupIncl(orig_group, npes - *splitter_rank,
                             ranks_desc, &new_group);
    }

    MpiUtils::MpiCommCreate(comm_, new_group, new_comm);
  } catch (MpiException &e) {
    delete[] ranks_asc; 
    delete[] ranks_desc;
    throw e;
  }

  delete[] ranks_asc; 
  delete[] ranks_desc;
}

void ParUtilsHelper::SplitCommBinaryNoFlip(
    int *splitter_rank, MpiComm *new_comm) throw(MpiException) {
  int npes = MpiUtils::MpiCommSize(comm_);
  int rank = MpiUtils::MpiCommRank(comm_);
  MpiGroup orig_group, new_group;

  *splitter_rank = GetPrevHighestPowerOfTwo(npes);

  // Determine sizes for the 2 groups .
  int *ranks_asc = new int[*splitter_rank];
  int *ranks_desc = new int[npes - *splitter_rank];

  int num_asc = 0;
  int num_desc = 0;
  // This is the main mapping between old ranks and new ranks.
  for (int i = 0; i < npes; i++) {
    if (i < *splitter_rank) {
      ranks_asc[num_asc] = i;
      num_asc++;
    } else {
      ranks_desc[num_desc] = i;
      num_desc++;
    }
  }

  try {
    MpiUtils::MpiCommGroup(comm_, &orig_group);

    // Divide tasks into two distinct groups based upon rank.
    if (rank < *splitter_rank) {
      MpiUtils::MpiGroupIncl(orig_group, *splitter_rank, ranks_asc, &new_group);
    } else {
      MpiUtils::MpiGroupIncl(orig_group, npes - *splitter_rank,
                             ranks_desc, &new_group);
    }

    MpiUtils::MpiCommCreate(comm_, new_group, new_comm);
  } catch (MpiException &e) {
    delete[] ranks_asc; 
    delete[] ranks_desc;
    throw e;
  }

  delete[] ranks_asc;
  delete[] ranks_desc;
}

void ParUtilsHelper::SplitCommUsingSplitter(
    int splitter_rank, MpiComm *new_comm) throw(MpiException) {
  int npes = MpiUtils::MpiCommSize(comm_);
  int rank = MpiUtils::MpiCommRank(comm_);
  MpiGroup orig_group, new_group;

  int* ranks_active = new int[splitter_rank];
  int* ranks_idle = new int[npes - splitter_rank];
  for (int i = 0; i < splitter_rank; i++) {
    ranks_active[i] = i;
  }
  for (int i = splitter_rank; i < npes; i++) {
    ranks_idle[i - splitter_rank] = i;
  }

  try {
    // Extract the original group handle.
    MpiUtils::MpiCommGroup(comm_, &orig_group);

    // Divide tasks into two distinct groups based upon rank.
    if (rank < splitter_rank) {
      MpiUtils::MpiGroupIncl(orig_group, splitter_rank,
                             ranks_active, &new_group);
    } else {
      MpiUtils::MpiGroupIncl(orig_group, npes - splitter_rank,
                             ranks_idle, &new_group);
    }

    // Create new communicator.
    MpiUtils::MpiCommCreate(comm_, new_group, new_comm);
  } catch (MpiException &e) {
    delete[] ranks_active; 
    delete[] ranks_idle;
    throw e;
  }

  delete[] ranks_active;
  delete[] ranks_idle;
}

//
// Synchronize information of Send & Receive sites.
//
template<typename IdxPair, typename IdxVector>
void ParUtilsHelper::SyncSendRecv(const IdxVector *senders,
                                  MpiRequest *requests, int *receive_cnt,
                                  IdxPair **idx_info) throw(MpiException) {
  int n_p = MpiUtils::MpiCommSize(comm_);
  int myrank = MpiUtils::MpiCommRank(comm_);
  for (int rank = 0; rank < n_p; rank++) {
    MpiUtils::MpiIRecv(comm_, 1, rank, rank,
                       &receive_cnt[rank], &requests[rank]);
    int cnt = senders[rank].size();
    MpiUtils::MpiISSend(comm_, &cnt, 1, rank,
                        myrank, &requests[n_p + rank]);
  }
  // Wait for all receive requests being complete.
  MpiUtils::MpiWaitAll(2 * n_p, requests);
  MpiUtils::MpiBarrier(comm_);

  int req_cnt = 0;
  for (int rank = 0; rank < n_p; rank++) {
    int cnt = receive_cnt[rank];
    if (cnt > 0) {
      idx_info[rank] = new IdxPair[cnt];
      MpiUtils::MpiIRecv(comm_, cnt, rank, rank,
                         idx_info[rank], &requests[req_cnt++]);
    }

    const IdxVector &sender = senders[rank];
    if (sender.empty()) continue;
    MpiUtils::MpiISSend(comm_, &sender[0], sender.size(), rank,
                        myrank, &requests[req_cnt++]);
  }

  // Wait for all receive requests being complete.
  MpiUtils::MpiWaitAll(req_cnt, requests);
  MpiUtils::MpiBarrier(comm_);
}
//
// Pre-define template parameters for SyncSendRecv function
// to pass the compiler linkage.
//
#define DECLARE_SYNC_SEND_RECV(int_t)\
template void ParUtilsHelper::SyncSendRecv<\
    std::pair<int_t, int_t>, std::vector<std::pair<int_t, int_t> > > (\
        const std::vector<std::pair<int_t, int_t> > *senders,\
        MpiRequest *requests, int *receive_cnt,\
        std::pair<int_t, int_t> **idx_info) throw(MpiException);

DECLARE_SYNC_SEND_RECV(int);
DECLARE_SYNC_SEND_RECV(unsigned int);
DECLARE_SYNC_SEND_RECV(int64_t);
DECLARE_SYNC_SEND_RECV(uint64_t);

}  // namespace gpusort
