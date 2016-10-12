#ifndef GPUSORT_MPI_COMMON_H_
#define GPUSORT_MPI_COMMON_H_

#include <map>
#include <string>
#include <typeinfo>

#include <mpi.h>

#include "exceptions/mpi_exception.h"

namespace gpusort {

// Re-define data types to fit coding convention
typedef MPI_Comm     MpiComm;
typedef MPI_Datatype MpiDatatype;
typedef MPI_Group    MpiGroup;
typedef MPI_Op       MpiOp;
typedef MPI_Request  MpiRequest;
typedef MPI_Status   MpiStatus;

// Re-define constants to fit coding convention
const int   kMpiAnySource = MPI_ANY_SOURCE;
const MpiOp kMpiMax       = MPI_MAX;
const MpiOp kMpiMaxLoc    = MPI_MAXLOC;
const MpiOp kMpiMin       = MPI_MIN;
const MpiOp kMpiSum       = MPI_SUM;

class MpiDataUtils {
 public:
  template<typename T>
  static inline bool GetMpiType(MpiDatatype *ret) {
    const std::type_info& t = typeid(T);
    bool state = false;
    if (t == typeid(short)) {
      *ret = MPI_SHORT; state = true;
    } else if (t == typeid(int)) {
      *ret = MPI_INT; state = true;
    } else if (t == typeid(long)) {
      *ret = MPI_LONG; state = true;
    } else if (t == typeid(unsigned short)) {
      *ret = MPI_UNSIGNED_SHORT; state = true;
    } else if (t == typeid(unsigned int)) {
      *ret = MPI_UNSIGNED; state = true;
    } else if (t == typeid(unsigned long)) {
      *ret = MPI_UNSIGNED_LONG; state = true;
    } else if (t == typeid(float)) {
      *ret = MPI_FLOAT; state = true;
    } else if (t == typeid(double)) {
      *ret = MPI_DOUBLE; state = true;
    } else if (t == typeid(long double)) {
      *ret = MPI_LONG_DOUBLE; state = true;
    } else if (t == typeid(long long)) {
      *ret = MPI_LONG_LONG_INT; state = true;
    } else if (t == typeid(char)) {
      *ret = MPI_CHAR; state = true;
    } else if (t == typeid(unsigned char)) {
      *ret = MPI_UNSIGNED_CHAR; state = true;
    }
    return state;
  }

  template<typename T>
  static inline MpiDatatype GetType() throw(MpiException) {
    MpiDatatype datatype;
    bool is_succeed = GetMpiType<T>(&datatype);
    if (is_succeed) return datatype;

    const std::type_info& t = typeid(T);
    std::map<std::string, MpiDatatype>::iterator it = types.find(t.name());
    if (it != types.end()) return it->second;

    int state = MPI_Type_contiguous(sizeof(T), MPI_BYTE, &datatype);
    switch (state) {
      case MPI_ERR_TYPE:
        throw MpiException("Invalid datatype argument");
      case MPI_ERR_COUNT:
        throw MpiException("Invalid count argument");
      case MPI_ERR_INTERN:
        throw MpiException("MPI implementation is unable to acquire memory");
      default:  // MPI_SUCCESS
        break;
    }
    
    state = MPI_Type_commit(&datatype);
    if (state != MPI_SUCCESS) {
      throw MpiException("Invalid datatype argument");
    }
    
    types.insert(std::make_pair<std::string, MpiDatatype>(t.name(), datatype));
    return datatype;
  }
  
 private:
  static std::map<std::string, MpiDatatype> types;
};

}  // namespace gpusort

#endif  // GPUSORT_MPI_COMMON_H_
