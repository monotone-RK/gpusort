#ifndef GPUSORT_TEST_UNIFORM_CHECKER_H_
#define GPUSORT_TEST_UNIFORM_CHECKER_H_

#include <iostream>
#include <string>
#include <vector>

#if defined(__USING_CPP_0X) || defined(__USING_CPP_11)
#include <cstdint>
#else
#include <stdint.h>
#endif  // __USING_CPP_0X || __USING_CPP_11

#include "exceptions/mpi_exception.h"
#include "mpi/mpi_common.h"
#include "mpi/mpi_utils.h"

#include "parameter.h"
#include "utils.h"

namespace gpusort {

template<typename T>
class UniformChecker : public Utils::DataChecker<T> {
 public:
  UniformChecker() : Utils::DataChecker<T>() {
  }

  UniformChecker(MpiComm comm) : Utils::DataChecker<T>(comm) {
  }

  bool Check(const std::vector<T> &list,
             bool is_increasing) throw(MpiException) {
    if (MpiUtils::MpiCommRank(this->comm_) == 0)
      std::cout << "Validating sorted sequence by UniformChecker." << std::endl;
    return Utils::CheckValid(this->comm_, is_increasing,
                             &list[0], &list[list.size()]);
  }

  bool Generate(int64_t nelem, std::vector<T> *list) throw(MpiException) {
    std::string msg = "";
    if (this->comm_ != NULL) {
      int myrank = MpiUtils::MpiCommRank(this->comm_);
      msg = "Rank " + Utils::ToString(myrank) + " is generating an Uniform " +
            "Distribution with n_elements = " + Utils::ToString(nelem) + ".\n";
    } else {
      msg = "Generating an Uniform Distribution with n_elements = " +
            Utils::ToString(nelem) + ".\n";
    }
    std::cout << msg;
    Utils::GenDataUniformDistribution(this->comm_, nelem, list);
    return true;
  }

  bool Generate(const std::string &capacity, const std::string &data_path,
                const std::string &ext) throw(MpiException) {
    return Utils::DataChecker<T>::Generate(capacity, data_path, ext);
  }
};

}  // namespace gpusort

#endif  // GPUSORT_TEST_UNIFORM_CHECKER_H_
