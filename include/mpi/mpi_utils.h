#ifndef GPUSORT_MPI_UTILS_H_
#define GPUSORT_MPI_UTILS_H_

#include "exceptions/mpi_exception.h"
#include "mpi/mpi_common.h"
#include "mpi/mpi_impl.h"

namespace gpusort {

class MpiUtils {
 public:
  static inline void MpiAbort(MpiComm comm, int code) throw(MpiException) {
    MpiImpl::MpiAbort(comm, code);
  }

  template<typename T>
  static inline void MpiAllGather(MpiComm comm, const T* send_buf,
                                  int count, T* recv_buf) throw(MpiException) {
    MpiImpl::MpiAllGather(comm, send_buf, count, MpiDataUtils::GetType<T>(),
                          count, MpiDataUtils::GetType<T>(), recv_buf);
  }

  template<typename T>
  static inline void MpiAllGatherV(MpiComm comm, const T* send_buf,
                                   int send_cnt, const int* recv_cnts,
                                   const int* displs,
                                   T* recv_buf) throw(MpiException) {
    MpiImpl::MpiAllGatherV(comm, send_buf, send_cnt,
                           MpiDataUtils::GetType<T>(), recv_cnts, displs,
                           MpiDataUtils::GetType<T>(), recv_buf);
  }

  template<typename T>
  static inline void MpiAllReduce(MpiComm comm, const T* send_buf, int count,
                                  MpiOp op, T* recv_buf) throw(MpiException) {
    MpiImpl::MpiAllReduce(comm, send_buf, count, MpiDataUtils::GetType<T>(),
                          op, recv_buf);
  }

  template<typename T>
  static inline void MpiAllToAll(MpiComm comm, const T* send_buf,
                                 int count, T* recv_buf) throw(MpiException) {
    MpiImpl::MpiAllToAll(comm, send_buf, count, MpiDataUtils::GetType<T>(),
                         count, MpiDataUtils::GetType<T>(), recv_buf);
  }

  template<typename T>
  static inline void MpiAllToAllV(MpiComm comm, const T* send_buf,
                                  const int* send_cnts, const int* sdispls,
                                  const int* recv_cnts, const int* rdispls,
                                  T* recv_buf) throw(MpiException) {
    MpiImpl::MpiAllToAllV(comm, send_buf, send_cnts, sdispls,
                          MpiDataUtils::GetType<T>(), recv_cnts, rdispls,
                          MpiDataUtils::GetType<T>(), recv_buf);
  }

  template<typename T>
  static inline void MpiAllToAllVDense(MpiComm comm, const T* _sbuff,
                                       const int* _s_cnt, const int* _sdisp,
                                       const int* _r_cnt, const int* _rdisp,
                                       T* _rbuff) throw(MpiException) {
    MpiImpl::MpiAllToAllVDense(comm, _sbuff, _s_cnt, _sdisp,
                               MpiDataUtils::GetType<T>(), _r_cnt, _rdisp,
                               MpiDataUtils::GetType<T>(), _rbuff);
  }

  template<typename T>
  static inline void MpiAllToAllVSparse(
      MpiComm comm, const T* send_buf,
      const int* send_cnts, const int* sdispls,
      const int* recv_cnts, const int* rdispls,
      T* recv_buf) throw(MpiException) {
    MpiImpl::MpiAllToAllVSparse(comm, send_buf, send_cnts, sdispls,
                                MpiDataUtils::GetType<T>(), recv_cnts,
                                rdispls, MpiDataUtils::GetType<T>(), recv_buf);
  }

  static inline void MpiBarrier(MpiComm comm) throw(MpiException) {
    MpiImpl::MpiBarrier(comm);
  }
  
  template<typename T>
  static inline void MpiBcast(MpiComm comm, int count,
                              int root, T* buffer) throw(MpiException) {
    MpiImpl::MpiBcast(comm, count, MpiDataUtils::GetType<T>(), root, buffer);
  }

  static inline void MpiCommCreate(MpiComm comm, MpiGroup group,
                                   MpiComm *newcomm) throw(MpiException) {
    MpiImpl::MpiCommCreate(comm, group, newcomm);
  }

  static inline void MpiCommFree(MpiComm *comm) throw(MpiException) {
    MpiImpl::MpiCommFree(comm);
  }

  static inline void MpiCommGroup(MpiComm comm,
                                  MpiGroup *group) throw(MpiException) {
    MpiImpl::MpiCommGroup(comm, group);
  }

  static inline int MpiCommRank(MpiComm comm) throw(MpiException) {
    return MpiImpl::MpiCommRank(comm);
  }

  static inline int MpiCommSize(MpiComm comm) throw(MpiException) {
    return MpiImpl::MpiCommSize(comm);
  }

  static inline void MpiCommSplit(MpiComm comm, int color, int key,
                                  MpiComm *newcomm) throw(MpiException) {
    MpiImpl::MpiCommSplit(comm, color, key, newcomm);
  }

  static inline void MpiFinalize() throw(MpiException) {
     MpiImpl::MpiFinalize();
  }

  template<typename T>
  static inline void MpiGather(MpiComm comm, const T* send_buf, int count,
                               int root, T* recv_buf) throw(MpiException) {
    MpiImpl::MpiGather(comm, send_buf, count, MpiDataUtils::GetType<T>(),
                       count, MpiDataUtils::GetType<T>(), root, recv_buf);
  }

  static inline void MpiGroupIncl(MpiGroup group, int n, const int ranks[],
                                  MpiGroup *newgroup) throw(MpiException) {
    MpiImpl::MpiGroupIncl(group, n, ranks, newgroup);
  }

  static inline void MpiInit(int *argc, char ***argv) throw(MpiException) {
    MpiImpl::MpiInit(argc, argv);
  }

  template<typename T>
  static inline void MpiIRecv(
      MpiComm comm, int count, int source, int tag,
      T* buf, MpiRequest* request) throw(MpiException) {
    MpiImpl::MpiIRecv(comm, count, MpiDataUtils::GetType<T>(),
                      source, tag, buf, request);
  }

  template<typename T>
	static inline void MpiISSend(
      MpiComm comm, const T* buf, int count, int dest,
      int tag, MpiRequest* request) throw(MpiException) {
    MpiImpl::MpiISSend(comm, buf, count, MpiDataUtils::GetType<T>(),
                       dest, tag, request);
  }

  template<typename T>
  static inline void MpiRecv(MpiComm comm, int count, int source, int tag,
                             T* buf, MpiStatus* status) throw(MpiException) {
    MpiImpl::MpiRecv(comm, count, MpiDataUtils::GetType<T>(),
                     source, tag, buf, status);
  }

  template<typename T>
  static inline void MpiScan(MpiComm comm, const T* send_buf, int count,
                             MpiOp op, T* recv_buf) throw(MpiException) {
    MpiImpl::MpiScan(comm, send_buf, count, MpiDataUtils::GetType<T>(),
                     op, recv_buf);
  }

  template<typename T>
  static inline void MpiSend(MpiComm comm, const T* buf,
                             int cnt, int dest, int tag) throw(MpiException) {
    MpiImpl::MpiSend(comm, buf, cnt, MpiDataUtils::GetType<T>(), dest, tag);
  }

  template<typename T>
  static inline void MpiSendRecv(MpiComm comm, const T* send_buf, int send_cnt,
                                 int dest, int send_tag, int recv_cnt,
                                 int source, int recv_tag, T* recv_buf,
                                 MpiStatus* status) throw(MpiException) {
    MpiImpl::MpiSendRecv(comm, send_buf, send_cnt, MpiDataUtils::GetType<T>(),
                         dest, send_tag, recv_cnt, MpiDataUtils::GetType<T>(),
                         source, recv_tag, recv_buf, status);
  }

  static inline void MpiTest(MpiRequest* request, int* flag,
                             MpiStatus* status) throw(MpiException) {
    MpiImpl::MpiTest(request, flag, status);
  }

  static inline void MpiWait(MpiRequest* request) throw(MpiException) {
    MpiImpl::MpiWait(request, MPI_STATUSES_IGNORE);
  }

  static inline void MpiWait(MpiRequest* request,
                             MpiStatus* status) throw(MpiException) {
    MpiImpl::MpiWait(request, status);
  }

  static inline void MpiWaitAll(int count,
                                MpiRequest *requests) throw(MpiException) {
    MpiImpl::MpiWaitAll(count, requests, MPI_STATUSES_IGNORE);
  }

  static inline void MpiWaitAll(int count, MpiRequest *requests,
                                MpiStatus *statuses) throw(MpiException) {
    MpiImpl::MpiWaitAll(count, requests, statuses);
  }
};

}  // namespace gpusort

#endif  // GPUSORT_MPI_UTILS_H_
