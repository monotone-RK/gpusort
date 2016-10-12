#include "mpi/mpi_impl.h"

#include <sstream>

namespace gpusort {

std::map<std::string, MpiDatatype> MpiDataUtils::types;

template<typename T>
inline std::string ToString(T value) {
  return static_cast<std::ostringstream*>(
      &(std::ostringstream() << value))->str();
}

void MpiImpl::MpiAbort(MpiComm comm, int error_code) throw(MpiException) {
  int state = MPI_Abort(comm, error_code);
  MpiCheckError(state);
}

void MpiImpl::MpiAllGather(MpiComm comm, const void* send_buf,
                           int send_cnt, MpiDatatype send_type,
                           int recv_cnt, MpiDatatype recv_type,
                           void* recv_buf) throw(MpiException) {
  int state = MPI_Allgather(send_buf, send_cnt, send_type,
                            recv_buf, recv_cnt, recv_type, comm);
  MpiCheckError(state);
}

void MpiImpl::MpiAllGatherV(MpiComm comm, const void* send_buf, int send_cnt,
                            MpiDatatype send_type, const int* recv_cnts,
                            const int* displs, MpiDatatype recv_type,
                            void* recv_buf) throw(MpiException) {
  int state = MPI_Allgatherv(send_buf, send_cnt, send_type, recv_buf,
                             recv_cnts, displs, recv_type, comm);
  MpiCheckError(state);
}

void MpiImpl::MpiAllReduce(MpiComm comm, const void* send_buf, int count,
                           MpiDatatype data_type, MpiOp op,
                           void* recv_buf) throw(MpiException) {
  int state = MPI_Allreduce(send_buf, recv_buf, count,
                            data_type, op, comm);
  MpiCheckError(state);
}

void MpiImpl::MpiAllToAll(MpiComm comm, const void* send_buf,
                          int send_cnt, MpiDatatype send_type,
                          int recv_cnt, MpiDatatype recv_type,
                          void* recv_buf) throw(MpiException) {
  int state = MPI_Alltoall(send_buf, send_cnt, send_type,
                           recv_buf, recv_cnt, recv_type, comm);
  MpiCheckError(state);
}

void MpiImpl::MpiAllToAllV(MpiComm comm, const void* send_buf,
                           const int* send_cnts, const int* sdispls,
                           MpiDatatype send_type, const int* recv_cnts,
                           const int* rdispls, MpiDatatype recv_type,
                           void* recv_buf) throw(MpiException) {
  int state = MPI_Alltoallv(send_buf, send_cnts, sdispls, send_type,
                            recv_buf, recv_cnts, rdispls, recv_type, comm);
  MpiCheckError(state);
}

void MpiImpl::MpiAllToAllVDense(MpiComm comm, const void* _sbuff,
                                const int* _s_cnt, const int* _sdisp,
                                MpiDatatype send_type, const int* _r_cnt,
                                const int* _rdisp, MpiDatatype recv_type,
                                void* _rbuff) throw(MpiException) {
  MpiAllToAllV(comm, _sbuff, _s_cnt, _sdisp, send_type,
               _r_cnt, _rdisp, send_type, _rbuff);
}

void MpiImpl::MpiAllToAllVSparse(MpiComm comm, const void* send_buf,
                                 const int* send_cnts, const int* sdispls,
                                 MpiDatatype send_type, const int* recv_cnts,
                                 const int* rdispls, MpiDatatype recv_type,
                                 void* recv_buf) throw(MpiException) {
  MpiAllToAllV(comm, send_buf, send_cnts, sdispls, send_type,
               recv_cnts, rdispls, send_type, recv_buf);
}

void MpiImpl::MpiBarrier(MpiComm comm) throw(MpiException) {
  int state = MPI_Barrier(comm);
  MpiCheckError(state);
}
  
void MpiImpl::MpiBcast(MpiComm comm, int count, MpiDatatype data_type,
                       int root, void* buffer) throw(MpiException) {
  int state = MPI_Bcast(buffer, count, data_type, root, comm);
  MpiCheckError(state);
}

void MpiImpl::MpiCommCreate(MpiComm comm, MpiGroup group,
                            MpiComm *newcomm) throw(MpiException) {
  int state = MPI_Comm_create(comm, group, newcomm);
  MpiCheckError(state);
}

void MpiImpl::MpiCommFree(MpiComm *comm) throw(MpiException) {
  int state = MPI_Comm_free(comm);
  MpiCheckError(state);
}

void MpiImpl::MpiCommGroup(MpiComm comm, MpiGroup *group) throw(MpiException) {
  int state = MPI_Comm_group(comm, group);
  MpiCheckError(state);
}

int MpiImpl::MpiCommRank(MpiComm comm) throw(MpiException) {
  int rank = -1;
  int state = MPI_Comm_rank(comm, &rank);
  MpiCheckError(state);
  return rank;
}

int MpiImpl::MpiCommSize(MpiComm comm) throw(MpiException) {
  int size = 0;
  int state = MPI_Comm_size(comm, &size);
  MpiCheckError(state);
  return size;
}

void MpiImpl::MpiCommSplit(MpiComm comm, int color, int key,
                           MpiComm *newcomm) throw(MpiException) {
  int state = MPI_Comm_split(comm, color, key, newcomm);
  MpiCheckError(state);
}

void MpiImpl::MpiFinalize() throw(MpiException) {
  int state = MPI_Finalize();
  MpiCheckError(state);
}

void MpiImpl::MpiGather(MpiComm comm, const void* send_buf,
                        int send_cnt, MpiDatatype send_type,
                        int recv_cnt, MpiDatatype recv_type,
                        int root, void* recv_buf) throw(MpiException) {
  int state = MPI_Gather(send_buf, send_cnt, send_type, recv_buf,
                         recv_cnt, recv_type, root, comm);
  MpiCheckError(state);
}

void MpiImpl::MpiGroupIncl(MpiGroup group, int n, const int ranks[],
                           MpiGroup *newgroup) throw(MpiException) {
  int state = MPI_Group_incl(group, n, ranks, newgroup);
  MpiCheckError(state);
}

void MpiImpl::MpiInit(int *argc, char ***argv) throw(MpiException) {
  int state = MPI_Init(argc, argv);
  MpiCheckError(state);
}

void MpiImpl::MpiIRecv(MpiComm comm, int count, MpiDatatype data_type,
                       int source, int tag,
                       void* buf, MpiRequest* request) throw(MpiException) {
  int state = MPI_Irecv(buf, count, data_type, source, tag, comm, request);
  MpiCheckError(state);
}

void MpiImpl::MpiISSend(MpiComm comm, const void* buf, int count,
	                      MpiDatatype data_type, int dest,
                        int tag, MpiRequest* request) throw(MpiException) {
  int state = MPI_Issend(buf, count, data_type, dest, tag, comm, request);
  MpiCheckError(state);
}

void MpiImpl::MpiRecv(MpiComm comm, int count, MpiDatatype data_type,
                      int source, int tag, void* buf,
                      MpiStatus* status) throw(MpiException) {
  int state = MPI_Recv(buf, count, data_type, source, tag, comm, status);
  MpiCheckError(state);
}

void MpiImpl::MpiScan(MpiComm comm, const void* send_buf, int count,
                      MpiDatatype data_type, MpiOp op,
                      void* recv_buf) throw(MpiException) {
  int state = MPI_Scan(send_buf, recv_buf, count, data_type, op, comm);
  MpiCheckError(state);
}

void MpiImpl::MpiSend(MpiComm comm, const void* buf,
                      int cnt, MpiDatatype data_type,
                      int dest, int tag) throw(MpiException) {
  int state = MPI_Send(buf, cnt, data_type, dest, tag, comm);
  MpiCheckError(state);
}

void MpiImpl::MpiSendRecv(MpiComm comm, const void* send_buf, int send_cnt,
                          MpiDatatype send_type, int dest, int send_tag,
                          int recv_cnt, MpiDatatype recv_type, int source,
                          int recv_tag, void* recv_buf,
                          MpiStatus* status) throw(MpiException) {
  int state = MPI_Sendrecv(send_buf, send_cnt, send_type, dest, send_tag,
                           recv_buf, recv_cnt, recv_type, source, recv_tag,
                           comm, status);
  MpiCheckError(state);
}

void MpiImpl::MpiTest(MpiRequest* request, int* flag,
                      MpiStatus* status) throw(MpiException) {
  int state = MPI_Test(request, flag, status);
  MpiCheckError(state);
}

void MpiImpl::MpiWait(MpiRequest* request,
                      MpiStatus* status) throw(MpiException) {
  int state = MPI_Wait(request, status);
  MpiCheckError(state);
}

void MpiImpl::MpiWaitAll(int count, MpiRequest *requests,
                         MpiStatus *statuses) throw(MpiException) {
  int state = MPI_Waitall(count, requests, statuses);
  MpiCheckError(state);
}

void MpiImpl::MpiCheckError(int state) throw(MpiException) {
  switch (state) {
    case MPI_ERR_ARG:
      throw MpiException("Invalid argument");

    case MPI_ERR_BUFFER:
      throw MpiException("Invalid buffer pointer");

    case MPI_ERR_COMM:
      throw MpiException("Invalid communicator");

    case MPI_ERR_COUNT:
      throw MpiException("Invalid count argument");

    case MPI_ERR_IN_STATUS:
      throw MpiException("The actual error value is in the MpiStatus argument");

    case MPI_ERR_INTERN:
      throw MpiException("MPI implementation is unable to acquire memory");

    case MPI_ERR_OP:
      throw MpiException("Invalid operation");

    case MPI_ERR_OTHER:
      throw MpiException("An attempt was made to call MpiInit a second time");

    case MPI_ERR_RANK:
      throw MpiException("Invalid source or destination rank");

    case MPI_ERR_REQUEST:
      throw MpiException("Invalid MpiRequest");

    case MPI_ERR_ROOT:
      throw MpiException("Invalid root");

    case MPI_ERR_TAG:
      throw MpiException("Invalid tag argument");

    case MPI_ERR_TYPE:
      throw MpiException("Invalid datatype argument");

    case MPI_SUCCESS:  // Success.
      return;

    default:  // Unspecified error.
      throw MpiException("Unspecified error with code: " + ToString(state));
  }
}

}  // namespace gpusort
