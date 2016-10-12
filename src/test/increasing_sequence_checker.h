#ifndef GPUSORT_TEST_INCREASING_SEQUENCE_CHECKER_H_
#define GPUSORT_TEST_INCREASING_SEQUENCE_CHECKER_H_

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

#define PARAMETER_F1_NAME "f1"

#define PARAMETER_RMAX_NAME "r_max"

#define PARAMETER_F1_DEFAULT 1

#define PARAMETER_RMAX_DEFAULT 5

template<typename T>
class IncreasingSequenceChecker : public Utils::DataChecker<T> {
 public:
  IncreasingSequenceChecker() : Utils::DataChecker<T>() {
  }

  IncreasingSequenceChecker(MpiComm comm) : Utils::DataChecker<T>(comm) {
  }

  bool Check(const std::vector<T> &list,
             bool is_increasing) throw(MpiException) {
    if (MpiUtils::MpiCommRank(this->comm_) == 0)
      std::cout << "Validating sorted sequence by IncreasingSequenceChecker."
                << std::endl;
    return Utils::CheckValid(this->comm_, is_increasing,
                             &list[0], &list[list.size()]);
  }

  bool Generate(int64_t nelem, std::vector<T> *list) throw(MpiException) {
    T f1 = this->template GetParameter<T>(PARAMETER_F1_NAME,
                                          PARAMETER_F1_DEFAULT);
    T r_max = this->template GetParameter<T>(PARAMETER_RMAX_NAME,
                                             PARAMETER_RMAX_DEFAULT);
    std::string msg = "Generating an Increasing Sequence with: " +
                      std::string(PARAMETER_F1_NAME) + " = " +
                      Utils::ToString(f1) + ", " +
                      std::string(PARAMETER_RMAX_NAME) + " = " +
                      Utils::ToString(r_max) + ", n_elements = " +
                      Utils::ToString(nelem) + ".\n";
    std::cout << msg;
    gpusort::Utils::GenDataIncreasing(f1, nelem, r_max, list);
    return true;
  }

  bool Generate(int np, const std::string &capacity,
                const std::string &data_path,
                const std::string &ext) throw(MpiException) {
    return Utils::DataChecker<T>::Generate(np, capacity, data_path, ext);
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
      if (param_name.compare(PARAMETER_F1_NAME) == 0) {
        this->params_->Add(new Parameter<T>(PARAMETER_F1_NAME,
                                            Utils::ToT<T>(param_value)));
      } else if (param_name.compare(PARAMETER_RMAX_NAME) == 0) {
        this->params_->Add(new Parameter<T>(PARAMETER_RMAX_NAME,
                                            Utils::ToT<T>(param_value)));
      } else {
        this->params_->Clear();
        return false;
      }
    }

    return true;
  }
};

}  // namespace gpusort

#endif  // GPUSORT_TEST_INCREASING_SEQUENCE_CHECKER_H_
