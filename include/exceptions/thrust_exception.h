#ifndef GPUSORT_THRUST_EXCEPTION_H_
#define GPUSORT_THRUST_EXCEPTION_H_

#include <exception>
#include <string>

namespace gpusort {

class ThrustException: public std::exception {
 private:
  std::string message_;

 public:
  ~ ThrustException() throw() {}

  explicit ThrustException (const std::string& message) : message_(message) {
  }

  virtual const char* what() const throw() {
    return message_.c_str();
  }
};

}  // namespace gpusort

#endif  // GPUSORT_THRUST_EXCEPTION_H_
