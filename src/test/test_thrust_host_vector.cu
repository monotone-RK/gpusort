#include <iostream>

#include <thrust/host_vector.h>

#include "utils.h"

template<typename StrictWeakOrdering, typename T>
void SortWithThrustHostVector(gpusort::MpiComm comm, StrictWeakOrdering comp,
                              gpusort::Algorithm algo, unsigned int kway,
                              bool is_using_gpu, std::vector<T> *_list) {
  std::vector<T> &list = *_list;
  int myrank = gpusort::MpiUtils::MpiCommRank(comm);
  gpusort::MpiUtils::MpiBarrier(comm);

    // Convert std::vector to thrust::host_vector.
  int n_threads = gpusort::OmpUtils::GetMaxThreads();
  int64_t size = list.size();
  thrust::host_vector<T> host_vec(size);
  FOR_PARALLEL(n_threads, size, i,
               {
                 host_vec[i] = list[i];
               });

  switch (algo) {
    // Using HykSort to sort input data.
    case gpusort::HYK_SORT_ALGORITHM:
      gpusort::HykSort(comm, comp, kway, is_using_gpu, &host_vec);
      break;

    // Using HyperQuickSort to sort input data.
    case gpusort::HYPER_QUICK_SORT_ALGORITHM:
      gpusort::HyperQuickSort(comm, comp, &host_vec);
      break;

    // Using SampleSort to sort input data.
    default:
      gpusort::SampleSort(comm, comp, &host_vec);
      break;
  }

  size = host_vec.size();
  std::cout << "Rank " << myrank << " finished sorting. New data size = "
            << size << std::endl;
  // Convert thrust::host_vector back to std::vector.
  list.resize(size);
  FOR_PARALLEL(n_threads, size, i,
               {
                 list[i] = host_vec[i];
               });
}

#define DECLARE_FUNC(type) \
template void SortWithThrustHostVector<std::less<type>, type>(\
    gpusort::MpiComm, std::less<type>, gpusort::Algorithm,\
    unsigned int, bool, std::vector<type>*);\
template void SortWithThrustHostVector<std::greater<type>, type>(\
    gpusort::MpiComm, std::greater<type>, gpusort::Algorithm,\
    unsigned int, bool, std::vector<type>*);\

DECLARE_FUNC(double);
DECLARE_FUNC(float);
DECLARE_FUNC(long);
DECLARE_FUNC(unsigned long);
DECLARE_FUNC(int);
DECLARE_FUNC(unsigned int);
