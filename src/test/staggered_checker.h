#ifndef GPUSORT_TEST_STAGGERED_CHECKER_H_
#define GPUSORT_TEST_STAGGERED_CHECKER_H_

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

#define PARAMETER_P_NAME "p"

#define PARAMETER_P_DEFAULT 128

template<typename T>
class StaggeredChecker : public Utils::DataChecker<T> {
 public:
  StaggeredChecker() : Utils::DataChecker<T>() {
  }

  StaggeredChecker(MpiComm comm) : Utils::DataChecker<T>(comm) {
  }

  bool Check(const std::vector<T> &list,
             bool is_increasing) throw(MpiException) {
    if (MpiUtils::MpiCommRank(this->comm_) == 0)
      std::cout << "Validating sorted sequence by StaggeredChecker."
                << std::endl;
    return Utils::CheckValid(this->comm_, is_increasing,
                             &list[0], &list[list.size()]);
  }

  bool Generate(int64_t nelem, std::vector<T> *list) throw(MpiException) {
    int p = this->template GetParameter<int>(PARAMETER_P_NAME,
                                             PARAMETER_P_DEFAULT);
    std::string msg = "";
    if (this->comm_ != NULL) {
      int myrank = MpiUtils::MpiCommRank(this->comm_);
      msg = "Rank " + Utils::ToString(myrank) + " is generating a Staggered " +
            "Distribution with " + std::string(PARAMETER_P_NAME) + " = " +
            Utils::ToString(p) + ", n_elements = " +
            Utils::ToString(nelem) + ".\n";
    } else {
      msg = "Generating a Staggered Distribution with " +
            std::string(PARAMETER_P_NAME) + " = " + Utils::ToString(p) +
            ", n_elements = " + Utils::ToString(nelem) + ".\n";
    }
    std::cout << msg;
    Utils::GenDataStaggeredDistribution(this->comm_, p, nelem, list);
    return true;
  }

  bool Generate(const std::string &capacity, const std::string &data_path,
                const std::string &ext) throw(MpiException) {
    return Utils::DataChecker<T>::Generate(capacity, data_path, ext);
  }

  bool ParseParameters(const std::vector<std::string> &params_cfg) {
    if (this->params_ != NULL) {
      this->params_->Clear();
    } else {
      this->params_ = new ParameterList();
    }
    size_t n_params = params_cfg.size();
    for (size_t i = 0; i < n_params; i++) {
      std::vector<std::string> parts;
      Utils::SplitString(params_cfg[i], PARAMETER_SET_OPERATOR, &parts);
      std::string param_name = Utils::Trim(parts[0]);
      std::string param_value = Utils::Trim(parts[1]);
      if (param_name.compare(PARAMETER_P_NAME) == 0) {
        this->params_->Add(new Parameter<int>(PARAMETER_P_NAME,
                                              Utils::ToInt(param_value)));
      } else {
        return false;
      }
    }

    return true;
  }
};

}  // namespace gpusort

#endif  // GPUSORT_TEST_STAGGERED_CHECKER_H_
