#ifndef GPUSORT_OMP_UTILS_H_
#define GPUSORT_OMP_UTILS_H_

#include <algorithm>
#include <cstring>

#include <omp.h>

namespace gpusort {

class OmpUtils {
 public:
  class Locker {
   public:
    Locker() {
      locked_ = false;
      omp_init_lock(&lock_);
    }

    ~Locker() {
      if (locked_) UnLock();
      omp_destroy_lock(&lock_);
    }

    void Lock() {
      omp_set_lock(&lock_);
      locked_ = true;
    }

    bool TestLock() {
      bool suc = omp_test_lock(&lock_);
      if (suc) locked_ = true;
      return suc;
    }

    void UnLock() {
      omp_unset_lock(&lock_);
      locked_ = false;
    }

   private:
    omp_lock_t lock_;
    bool locked_;
  };

  static inline int GetMaxThreads() {
    return omp_get_max_threads();
  }
  
  static inline int GetThreadId() {
    return omp_get_thread_num();
  }

  template<typename RandomAccessIterator, typename StrictWeakOrdering>
  static void Merge(StrictWeakOrdering comp,
                    RandomAccessIterator a_first, RandomAccessIterator a_last,
                    RandomAccessIterator b_first, RandomAccessIterator b_last,
                    RandomAccessIterator out_first) {
    typedef typename std::iterator_traits<RandomAccessIterator>::difference_type
        DiffType;
    typedef typename std::iterator_traits<RandomAccessIterator>::value_type
        ValType;

    int p = GetMaxThreads();
    DiffType n1 = a_last - a_first;
    DiffType n2 = b_last - b_first;
    if (n1 == 0 && n2 == 0) return;
    if (n1 == 0 || n2 == 0) {
      ValType* a = (n1 == 0)? &b_first[0] : &a_first[0];
      DiffType n = (n1 == 0)? n2 : n1;
      #pragma omp parallel for
      for (int i = 0; i < p; i++) {
        DiffType indx1 = (i * n) / p;
        DiffType indx2 = ((i+1) * n) / p;
        memcpy(&out_first[indx1], &a[indx1], (indx2-indx1) * sizeof(ValType));
      }
      return;
    }

    // Split both arrays ( a and b ) into n equal parts.
    // Find the position of each split in the final merged array.
    int n = 10;
    ValType* split = new ValType[p * n * 2];
    DiffType* split_size = new DiffType[p * n * 2];
    #pragma omp parallel for
    for (int i = 0; i < p; i++) {
      for (int j = 0; j < n; j++) {
        int indx = i * n + j;
        DiffType indx1 = (indx * n1) / (p * n);
        split[indx] = a_first[indx1];
        split_size[indx] = indx1 + (std::lower_bound(b_first, b_last,
                                                     split[indx], comp) -
                                    b_first);
        indx1 = (indx * n2) / (p * n);
        indx += p * n;
        split[indx] = b_first[indx1];
        split_size[indx] = indx1 + (std::lower_bound(a_first, a_last,
                                                     split[indx], comp) -
                                    a_first);
      }
    }

    // Find the closest split position for each thread that will
    // divide the final array equally between the threads.
    DiffType* split_indx_a = new DiffType[p + 1];
    DiffType* split_indx_b = new DiffType[p + 1];
    split_indx_a[0] = 0;
    split_indx_b[0] = 0;
    split_indx_a[p] = n1;
    split_indx_b[p] = n2;
    #pragma omp parallel for
    for (int i = 1; i < p; i++) {
      DiffType req_size = (i * (n1 + n2)) / p;
      int j = std::lower_bound(&split_size[0], &split_size[p*n], req_size,
                               std::less<DiffType>()) -
              &split_size[0];
      if (j >= p * n) j = p * n - 1;
      ValType split1 = split[j];
      DiffType split_size1 = split_size[j];
      j = p * n + (std::lower_bound(&split_size[p*n], &split_size[p*n*2],
                                    req_size, std::less<DiffType>()) -
                   &split_size[p*n]);
      if (j >= 2 * p * n) j = 2 * p * n - 1;
      if (abs(split_size[j]-req_size) < abs(split_size1-req_size)) {
        split1 = split[j];
        split_size1 = split_size[j];
      }

      split_indx_a[i] = std::lower_bound(a_first, a_last, split1, comp)-a_first;
      split_indx_b[i] = std::lower_bound(b_first, b_last, split1, comp)-b_first;
    }
    delete[] split;
    delete[] split_size;

    // Merge for each thread independently.
    #pragma omp parallel for
    for (int i = 0; i < p; i++) {
      RandomAccessIterator c = out_first + split_indx_a[i] + split_indx_b[i];
      std::merge(a_first + split_indx_a[i], a_first + split_indx_a[i+1],
                 b_first + split_indx_b[i], b_first + split_indx_b[i+1],
                 c, comp);
    }
    delete[] split_indx_a;
    delete[] split_indx_b;
  }

  template<typename RandomAccessIterator, typename StrictWeakOrdering>
  static void MergeSort(StrictWeakOrdering comp,
                        RandomAccessIterator a_first,
                        RandomAccessIterator a_last) {
    typedef typename std::iterator_traits<RandomAccessIterator>::difference_type
        DiffType;
    typedef typename std::iterator_traits<RandomAccessIterator>::value_type
        ValType;

    int p = GetMaxThreads();
    DiffType n = a_last - a_first;
    if (n < 2*p || p == 1) {
      std::sort(a_first, a_last, comp);
      return;
    }

    // Split the array a_first into p equal parts.
    DiffType* split = new DiffType[p + 1];
    split[p] = n;
    #pragma omp parallel for
    for (int id = 0; id < p; id++) {
      split[id] = (id * n) / p;
    }

    // Sort each part independently.
    #pragma omp parallel for
    for (int id = 0; id < p; id++) {
      std::sort(a_first + split[id], a_first + split[id+1], comp);
    }

    // Merge two parts at a time.
    ValType* b = new ValType[n];
    ValType* a = &a_first[0];
    ValType* bb = &b[0];
    for (int j = 1; j < p; j = j*2) {
      for (int i = 0; i < p; i = i+2*j) {
        if (i+j < p) {
          Merge(comp, a + split[i], a + split[i+j], a + split[i+j],
                a + split[(i+2*j <= p)? i+2*j : p], bb + split[i]);
        } else {
          #pragma omp parallel for
          for (int k = split[i]; k < split[p]; k++)
            bb[k] = a[k];
        }
      }
      ValType* tmp_swap = a;
      a = bb;
      bb = tmp_swap;
    }

    // The final result should be in a_first.
    if (a != &a_first[0]) {
      #pragma omp parallel for
      for (int i = 0; i < n; i++)
        a_first[i] = a[i];
    }

    // Free memory.
    delete[] split;
    delete[] b;
  }

  template<typename T, typename I>
  static void Scan(I cnt, const T* a, T* b) {
    int p = GetMaxThreads();
    if (cnt < 100 * p) {
      for (I i = 1; i < cnt; i++)
        b[i] = b[i-1] + a[i-1];
      return;
    }

    I step_size = cnt / p;

    #pragma omp parallel for
    for (int i = 0; i < p; i++) {
      int start = i * step_size;
      int end = start + step_size;
      if (i == p-1) end = cnt;
      if (i != 0) b[start] = 0;
      for (I j = start+1; j < end; j++)
        b[j] = b[j-1] + a[j-1];
    }

    T* sum = new T[p];
    sum[0] = 0;
    for (int i = 1; i < p; i++)
      sum[i] = sum[i-1] + b[i*step_size-1] + a[i*step_size-1];

    #pragma omp parallel for
    for (int i = 1; i < p; i++) {
      int start = i * step_size;
      int end = start + step_size;
      if (i == p-1) end = cnt;
      for (I j = start; j < end; j++)
        b[j] += sum[i];
    }

    delete[] sum;
  }

  static inline void SetNumThreads(int n_threads) {
    return omp_set_num_threads(n_threads);
  }
};

}  // namespace gpusort

#endif  // GPUSORT_OMP_UTILS_H_
