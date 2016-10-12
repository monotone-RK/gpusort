#ifndef GPUSORT_TEST_LN_CHECKER_H_
#define GPUSORT_TEST_LN_CHECKER_H_

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#if defined(__USING_CPP_0X) || defined(__USING_CPP_11)
#include <cstdint>
#else
#include <stdint.h>
#endif  // __USING_CPP_0X || __USING_CPP_11

#include <omp.h>

#include "exceptions/mpi_exception.h"
#include "mpi/mpi_common.h"
#include "mpi/mpi_utils.h"
#include "omp/omp_utils.h"
#include "par/par_utils.h"

#include "parameter.h"
#include "utils.h"

namespace gpusort {

#define PARAMETER_BEGIN_NAME "b"

#define PARAMETER_SHUFFLE_NAME "shuffle"

#define PARAMETER_BEGIN_DEFAULT 1

#define PARAMETER_SHUFFLE_DEFAULT false

template<typename T>
class LnChecker : public Utils::DataChecker<T> {
 public:
  LnChecker() : Utils::DataChecker<T>() {
  }

  LnChecker(MpiComm comm) : Utils::DataChecker<T>(comm) {
  }

  bool Check(const std::vector<T> &list,
             bool is_increasing) throw(MpiException) {
    int myrank = MpiUtils::MpiCommRank(this->comm_);
    int np = MpiUtils::MpiCommSize(this->comm_);
    int64_t begin = this->template GetParameter<int64_t>(
                        PARAMETER_BEGIN_NAME, PARAMETER_BEGIN_DEFAULT);
    if (myrank == 0) std::cout << "Validating sorted sequence by LnChecker "
                               << "with: begin = " << begin << std::endl;
    bool state = true;
    int64_t size = list.size();
    std::vector<int64_t> sizes;
    ParUtils::GatherNElems(this->comm_, size, &sizes);
    int64_t total = sizes[np - 1];
    if (size == 0) {
      state = false;
    } else {
      int n_threads = OmpUtils::GetMaxThreads();
      if (is_increasing) {
        // Local check.
        FOR_PARALLEL_COND(
            n_threads, size, i, state,
            {
              int64_t global_idx = (myrank == 0)? i + begin :
                                                  sizes[myrank-1] + i + begin;
              T cur = std::log(global_idx);
              if (cur != list[i]) state = false;
            });
      } else {
        // Local check.
        int64_t end_idx = total + begin - 1;
        FOR_PARALLEL_COND(
            n_threads, size, i, state,
            {
              int64_t global_idx = (myrank == 0)? end_idx - i :
                                                  end_idx - sizes[myrank-1] - i;
              T cur = std::log(global_idx);
              if (cur != list[i]) state = false;
            });
      }
    }
    MpiUtils::MpiBarrier(this->comm_);
    return state;
  }

  bool Generate(int64_t nelem, std::vector<T> *list) throw(MpiException) {
    int64_t begin = this->template GetParameter<int64_t>(
                        PARAMETER_BEGIN_NAME, PARAMETER_BEGIN_DEFAULT);
    bool shuffle = this->template GetParameter<bool>(PARAMETER_SHUFFLE_NAME,
                                                     PARAMETER_SHUFFLE_DEFAULT);
    std::string msg = "Generating a sequence using Ln function with: " +
                      std::string(PARAMETER_BEGIN_NAME) + " = " +
                      Utils::ToString(begin) + ", n_elements = " +
                      Utils::ToString(nelem) + ", " +
                      std::string(PARAMETER_SHUFFLE_NAME) + " = " +
                      ((shuffle)? "true" : "false") + ".\n";
    std::cout << msg;
    Utils::GenDataUsingLnFunc(begin, nelem, shuffle, list);
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
      if (param_name.compare(PARAMETER_BEGIN_NAME) == 0) {
        this->params_->Add(new Parameter<int64_t>(PARAMETER_BEGIN_NAME,
                                                  Utils::ToInt64(param_value)));
      } else if (param_name.compare(PARAMETER_SHUFFLE_NAME) == 0) {
        this->params_->Add(new Parameter<bool>(PARAMETER_SHUFFLE_NAME,
                                               Utils::ToBool(param_value)));
      } else {
        this->params_->Clear();
        return false;
      }
    }

    return true;
  }
};

}  // namespace gpusort

#endif  // GPUSORT_TEST_LN_CHECKER_H_
