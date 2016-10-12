#ifndef GPUSORT_MPI_IMPL_H_
#define GPUSORT_MPI_IMPL_H_

#include <stdint.h>
#include <vector>

#include "exceptions/mpi_exception.h"
#include "mpi/mpi_common.h"

namespace gpusort {

class MpiImpl {
 public:
  static void MpiAbort(MpiComm comm, int error_code) throw(MpiException);

  static void MpiAllGather(MpiComm comm, const void* send_buf,
                           int send_cnt, MpiDatatype send_type,
                           int recv_cnt, MpiDatatype recv_type,
                           void* recv_buf) throw(MpiException);

  static void MpiAllGatherV(MpiComm comm, const void* send_buf, int send_cnt,
                            MpiDatatype send_type, const int* recv_cnts,
                            const int* displs, MpiDatatype recv_type,
                            void* recv_buf) throw(MpiException);

  static void MpiAllReduce(MpiComm comm, const void* send_buf, int count,
                           MpiDatatype data_type, MpiOp op,
                           void* recv_buf) throw(MpiException);

  static void MpiAllToAll(MpiComm comm, const void* send_buf,
                          int send_cnt, MpiDatatype send_type,
                          int recv_cnt, MpiDatatype recv_type,
                          void* recv_buf) throw(MpiException);

  static void MpiAllToAllV(MpiComm comm, const void* send_buf,
                           const int* send_cnts, const int* sdispls,
                           MpiDatatype send_type, const int* recv_cnts,
                           const int* rdispls, MpiDatatype recv_type,
                           void* recv_buf) throw(MpiException);

  static void MpiAllToAllVDense(MpiComm comm, const void* _sbuff,
                                const int* _s_cnt, const int* _sdisp,
                                MpiDatatype send_type, const int* _r_cnt,
                                const int* _rdisp, MpiDatatype recv_type,
                                void* _rbuff) throw(MpiException);

  static void MpiAllToAllVSparse(MpiComm comm, const void* send_buf,
                                 const int* send_cnts, const int* sdispls,
                                 MpiDatatype send_type, const int* recv_cnts,
                                 const int* rdispls, MpiDatatype recv_type,
                                 void* recv_buf) throw(MpiException);

  static void MpiBarrier(MpiComm comm) throw(MpiException);
  
  static void MpiBcast(MpiComm comm, int count, MpiDatatype data_type,
                       int root, void* buffer) throw(MpiException);

  static void MpiCommCreate(MpiComm comm, MpiGroup group,
                            MpiComm *newcomm) throw(MpiException);

  static void MpiCommFree(MpiComm *comm) throw(MpiException);

  static void MpiCommGroup(MpiComm comm, MpiGroup *group) throw(MpiException);

  static int MpiCommRank(MpiComm comm) throw(MpiException);

  static int MpiCommSize(MpiComm comm) throw(MpiException);

  static void MpiCommSplit(MpiComm comm, int color, int key,
                           MpiComm *newcomm) throw(MpiException);

  static void MpiFinalize() throw(MpiException);

  static void MpiGather(MpiComm comm, const void* send_buf,
                        int send_cnt, MpiDatatype send_type,
                        int recv_cnt, MpiDatatype recv_type,
                        int root, void* recv_buf) throw(MpiException);

  static void MpiGroupIncl(MpiGroup group, int n, const int ranks[],
                           MpiGroup *newgroup) throw(MpiException);

  static void MpiInit(int *argc, char ***argv) throw(MpiException);

  static void MpiIRecv(MpiComm comm, int count, MpiDatatype data_type,
                       int source, int tag,
                       void* buf, MpiRequest* request) throw(MpiException);

	static void MpiISSend(MpiComm comm, const void* buf, int count,
	                      MpiDatatype data_type, int dest,
                        int tag, MpiRequest* request) throw(MpiException);

  static void MpiRecv(MpiComm comm, int count, MpiDatatype data_type,
                      int source, int tag, void* buf,
                      MpiStatus* status) throw(MpiException);

  static void MpiScan(MpiComm comm, const void* send_buf, int count,
                      MpiDatatype data_type, MpiOp op,
                      void* recv_buf) throw(MpiException);

  static void MpiSend(MpiComm comm, const void* buf,
                      int cnt, MpiDatatype data_type,
                      int dest, int tag) throw(MpiException);

  static void MpiSendRecv(MpiComm comm, const void* send_buf, int send_cnt,
                          MpiDatatype send_type, int dest, int send_tag,
                          int recv_cnt, MpiDatatype recv_type, int source,
                          int recv_tag, void* recv_buf,
                          MpiStatus* status) throw(MpiException);

  static void MpiTest(MpiRequest* request, int* flag,
                      MpiStatus* status) throw(MpiException);

  static void MpiWait(MpiRequest* request,
                      MpiStatus* status) throw(MpiException);

  static void MpiWaitAll(int count, MpiRequest *requests,
                         MpiStatus *statuses) throw(MpiException);

 private:
  static void MpiCheckError(int state) throw(MpiException);
};

}  // namespace gpusort

#endif  // GPUSORT_MPI_IMPL_H_
