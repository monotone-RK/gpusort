#ifndef GPUSORT_TEST_ARITHMETIC_PROGRESSION_CHECKER_H_
#define GPUSORT_TEST_ARITHMETIC_PROGRESSION_CHECKER_H_

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

#define PARAMETER_A1_NAME "a1"

#define PARAMETER_D_NAME "d"

#define PARAMETER_SHUFFLE_NAME "shuffle"

#define PARAMETER_A1_DEFAULT 0

#define PARAMETER_D_DEFAULT 1

#define PARAMETER_SHUFFLE_DEFAULT false

template<typename T>
class ArithmeticProgressionChecker : public Utils::DataChecker<T> {
 public:
  ArithmeticProgressionChecker() : Utils::DataChecker<T>() {
  }

  ArithmeticProgressionChecker(MpiComm comm) : Utils::DataChecker<T>(comm) {
  }

  bool Check(const std::vector<T> &list,
             bool is_increasing) throw(MpiException) {
    int myrank = MpiUtils::MpiCommRank(this->comm_);
    int np = MpiUtils::MpiCommSize(this->comm_);
    T a1 = this->template GetParameter<T>(PARAMETER_A1_NAME,
                                          PARAMETER_A1_DEFAULT);
    T d = this->template GetParameter<T>(PARAMETER_D_NAME, PARAMETER_D_DEFAULT);
    if (myrank == 0)
      std::cout << "Validating sorted sequence by ArithmeticProgressionChecker "
                << "with: a1 = " << a1 << ", d = " << d << std::endl;
    bool state = true;
    int64_t size = list.size();
    int64_t total = ParUtils::Sum(this->comm_, size);
    if (size == 0) {
      state = false;
    } else {
      T first = list[0], last = list[size - 1];
      int n_threads = OmpUtils::GetMaxThreads();
      is_increasing = (d < 0)? !is_increasing : is_increasing;
      if (is_increasing) {
        // Global check.
        if (myrank == np - 1) {
          T _last = a1 + (total - 1) * d;
          if (_last != last) state = false;
        }
        if (myrank == 0) {
          if (first != a1) state = false;
        }
        // Local check.
        if (state) {
          FOR_PARALLEL_COND(n_threads, size, i, state,
                            {
                              T cur = first + i * d;
                              if (cur != list[i]) state = false;
                            });
        }
      } else {
        // Global check.
        if (myrank == np - 1) {
          if (last != a1) state = false;
        }
        if (myrank == 0) {
          T _first = a1 + (total - 1) * d;
          if (first != _first) state = false;
        }
        // Local check.
        if (state) {
          FOR_PARALLEL_COND(n_threads, size, i, state,
                            {
                              T cur = first - i * d;
                              if (cur != list[i]) state = false;
                            });
        }
      }
    }
    MpiUtils::MpiBarrier(this->comm_);
    return state;
  }

  bool Generate(int64_t nelem, std::vector<T> *list) throw(MpiException) {
    T a1 = this->template GetParameter<T>(PARAMETER_A1_NAME,
                                          PARAMETER_A1_DEFAULT);
    T d = this->template GetParameter<T>(PARAMETER_D_NAME, PARAMETER_D_DEFAULT);
    bool shuffle = this->template GetParameter<bool>(PARAMETER_SHUFFLE_NAME,
                                                     PARAMETER_SHUFFLE_DEFAULT);
    std::string msg = "Generating an Arithmetic Progression with: " +
                      std::string(PARAMETER_A1_NAME) + " = " +
                      Utils::ToString(a1) + ", " +
                      std::string(PARAMETER_D_NAME) + " = " +
                      Utils::ToString(d) + ", " +
                      std::string(PARAMETER_SHUFFLE_NAME) + " = " +
                      ((shuffle)? "true" : "false") + ", n_elements = " +
                      Utils::ToString(nelem) + ".\n";
    std::cout << msg;
    Utils::GenDataUsingArithmeticProgression(a1, nelem, d, shuffle, list);
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
      if (param_name.compare(PARAMETER_A1_NAME) == 0) {
        this->params_->Add(new Parameter<T>(PARAMETER_A1_NAME,
                                            Utils::ToT<T>(param_value)));
      } else if (param_name.compare(PARAMETER_D_NAME) == 0) {
        this->params_->Add(new Parameter<T>(PARAMETER_D_NAME,
                                            Utils::ToT<T>(param_value)));
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

#endif  // GPUSORT_TEST_ARITHMETIC_PROGRESSION_CHECKER_H_
