#ifndef GPUSORT_MPI_EXCEPTION_H_
#define GPUSORT_MPI_EXCEPTION_H_

#include <exception>
#include <string>

namespace gpusort {

class MpiException: public std::exception {
 private:
  std::string message_;

 public:
  ~ MpiException() throw() {}

  explicit MpiException (const std::string& message) : message_(message) {
  }

  virtual const char* what() const throw() {
    return message_.c_str();
  }
};

}  // namespace gpusort

#endif  // GPUSORT_MPI_EXCEPTION_H_
