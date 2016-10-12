#ifndef GPUSORT_CUDA_UTILS_H_
#define GPUSORT_CUDA_UTILS_H_

#include <vector>

#include "exceptions/thrust_exception.h"

namespace gpusort {

class CudaUtils {
 public:
  static int GetDeviceCount() throw(ThrustException);

  static bool IsHavingDevices() throw(ThrustException);

  template<typename StrictWeakOrdering, typename RandomAccessIterator>
  static void Sort(StrictWeakOrdering comp, int rank,
                   RandomAccessIterator first,
                   RandomAccessIterator last) throw(ThrustException);

  template<typename StrictWeakOrdering, typename RandomAccessIterator1,
           typename RandomAccessIterator2>
  static void SortByKey(StrictWeakOrdering comp, int rank,
                        RandomAccessIterator1 key_first,
                        RandomAccessIterator1 key_last,
                        RandomAccessIterator2 val_first,
                        void *_out) throw(ThrustException);
};

}  // namespace gpusort

#endif  // GPUSORT_CUDA_UTILS_H_
