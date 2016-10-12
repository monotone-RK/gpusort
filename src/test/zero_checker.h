#ifndef GPUSORT_TEST_ZERO_CHECKER_H_
#define GPUSORT_TEST_ZERO_CHECKER_H_

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
#include "omp/omp_utils.h"
#include "par/par_utils.h"

#include "parameter.h"
#include "utils.h"

namespace gpusort {

template<typename T>
class ZeroChecker : public Utils::DataChecker<T> {
 public:
  ZeroChecker() : Utils::DataChecker<T>() {
  }

  ZeroChecker(MpiComm comm) : Utils::DataChecker<T>(comm) {
  }

  bool Check(const std::vector<T> &list,
             bool is_increasing) throw(MpiException) {
    int myrank = MpiUtils::MpiCommRank(this->comm_);
    if (myrank == 0)
      std::cout << "Validating sorted sequence by ZeroChecker." << std::endl;
    bool state = true;
    int64_t size = list.size();
    T constant = (size > 0)? list[0] : 0;
    // Global check.
    std::vector<T> out_cons;
    std::vector<int64_t> out_size;
    ParUtils::Gather(this->comm_, size, &out_size);
    ParUtils::Gather(this->comm_, constant, &out_cons);
    if (myrank == 0) {
      int np = out_cons.size();
      for (int i = 0; i < np; i ++) {
        if (out_size[i] != 0) {
          constant = out_cons[i];
          break;
        }
      }
      for (int i = 0; i < np; i ++) {
        if (out_size[i] == 0) continue;
        if (out_cons[i] != constant) {
          state = false;
          break;
        }
      }
    }
    // Local check.
    int n_threads = OmpUtils::GetMaxThreads();
    if (state && size > 0) {
      FOR_PARALLEL_COND(n_threads, size, i, state,
                        {
                          if (constant != list[i]) state = false;
                        });
    }

    MpiUtils::MpiBarrier(this->comm_);
    return state;
  }

  bool Generate(int64_t nelem, std::vector<T> *list) throw(MpiException) {
    std::string msg = "Generating a Zero Distribution with n_elements = " +
                      Utils::ToString(nelem) + ".\n";
    std::cout << msg;
    Utils::GenDataZeroDistribution(nelem, list);
    return true;
  }

  bool Generate(int np, const std::string &capacity,
                const std::string &data_path,
                const std::string &ext) throw(MpiException) {
    return Utils::DataChecker<T>::Generate(np, capacity, data_path, ext);
  }
};

}  // namespace gpusort

#endif  // GPUSORT_TEST_ZERO_CHECKER_H_
