#ifndef GPUSORT_TEST_UTILS_H_
#define GPUSORT_TEST_UTILS_H_

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>

#if defined(__USING_CPP_0X) || defined(__USING_CPP_11)
#include <cstdint>
#include <type_traits>
#else
#include <stdint.h>
#include <cstdlib>
#include <ctime>
#endif  // __USING_CPP_0X || __USING_CPP_11

#include "hyksort.h"
#include "hyperquicksort.h"
#include "samplesort.h"
#include "random.h"

#include "ifstream.h"
#include "ofstream.h"
#include "parameter.h"

namespace gpusort {

#define CALL_FUNC_T(ret, type_str, func, argv, error_stmt) \
  if (type_str.compare("double") == 0) {\
    ret = func<double>argv;\
  } else if (type_str.compare("unsigned long") == 0) {\
    ret = func<uint64_t>argv;\
  } else if (type_str.compare("float") == 0) {\
    ret = func<float>argv;\
  } else if (type_str.compare("long") == 0) {\
    ret = func<int64_t>argv;\
  } else if (type_str.compare("unsigned int") == 0) {\
    ret = func<unsigned int>argv;\
  } else if (type_str.compare("int") == 0) {\
    ret = func<int>argv;\
  } else {\
    error_stmt;\
  }

#define NULL_STMT

#define PARAMETER_SEPARATOR ','

#define PARAMETER_SET_OPERATOR '='

#define VALUE_SEPARATOR ':'

const std::string kDataFileExt = ".dat";

const std::string kErrorAlgorithmMsg = "Unknown sort algorithm.";

const std::string kErrorCapacityMsg =
    "Invalid parameter: Data capacity must be in form: <N>GB or <N>MB\n"
    "\tExample: 1GB, 2.5GB, 300MB,...";

const std::string kErrorCapacityOrDataPathMsg =
    "Invalid data capacity or data path.";

const std::string kErrorCapacityNProcessOrDataPathMsg =
    "Invalid number of processes, data capacity or data path.";

const std::string kErrorDataPathMsg = "Invalid data path.";

const std::string kErrorNProcessesMsg =
    "Invalid parameter: Number of processes must be greater than 0.";

const std::string kErrorTypeMsg = "Invalid data type. Supported types are:\n"
                                  "\tdouble, unsigned long, long,\n"
                                  "\tunsigned int, int.";

const std::string kKeyFileExt = ".key";

const std::string kValueFileExtPrefix = ".v";

#define MAX_INT 0x7fffffff  // 2^31

enum Algorithm {
  UNKNOWN_ALGORITHM          = 0x00,
  HYK_SORT_ALGORITHM         = 0x01,
  HYPER_QUICK_SORT_ALGORITHM = 0x02,
  SAMPLE_SORT_ALGORITHM      = 0x04,
};

class Utils {
 public:
  template<typename T>
  class DataChecker {
   public:
    DataChecker() {
      comm_ = NULL;
      params_ = NULL;
    }

    DataChecker(MpiComm comm) {
      comm_ = comm;
      params_ = NULL;
    }

    virtual ~DataChecker() {
      if (params_ != NULL) delete params_;
    }

    virtual bool Check(const std::vector<T> &list,
                       bool is_increasing) throw(MpiException) = 0;

    virtual bool Generate(int64_t nelem,
                          std::vector<T> *list) throw(MpiException) = 0;

    bool Generate(const std::string &capacity,
                  std::vector<T> *list) throw(MpiException) {
      int64_t nelem = CapacityToNElems<T>(capacity);
      if (nelem == 0) {
        std::cout << kErrorCapacityMsg << std::endl;
        return false;
      }
      return Generate(nelem, list);
    }

    bool Generate(const std::string &capacity, const std::string &data_path,
                  const std::string &ext) throw(MpiException) {
      if (comm_ == NULL) return false;
      std::vector<T> list;
      bool suc = Generate(capacity, &list);
      if (!suc) return false;
      int myrank = MpiUtils::MpiCommRank(comm_);
      SaveMyData(myrank, data_path, ext, &list[0], &list[list.size()]);
      std::string msg = "Rank " + ToString(myrank) +
                        " saved data successfully to " + data_path + ".";
      std::cout << msg << std::endl;
      return true;
    }

    bool Generate(int np, const std::string &capacity,
                  const std::string &data_path,
                  const std::string &ext) throw(MpiException) {
      std::vector<T> list;
      bool suc = Generate(capacity, &list);
      if (!suc) return false;
      SaveData(np, data_path, ext, &list[0], &list[list.size()]);
      std::string msg = "Data is saved successfully to " + data_path + ".";
      std::cout << msg << std::endl;
      return true;
    }

    virtual bool ParseParameters(
        const std::vector<std::string> &params_cfg) {
      return true;
    }

    void set_comm(MpiComm comm) {
      comm_ = comm;
    }

    void set_params(ParameterList *params) {
      if (params_ != NULL) delete params_;
      params_ = params;
    }

   protected:
    template<typename ValueType>
    const ValueType& GetParameter(const std::string &name,
                                  const ValueType &default_val) {
      if (params_ == NULL) return default_val;
      ParameterBase *tmp = (*params_)[name];
      if (tmp == NULL) return default_val;
      return static_cast<Parameter<ValueType>*>(tmp)->Value();
    }

    MpiComm comm_;
    ParameterList *params_;
  };

  typedef DataChecker<double> ValueCheckerBase;
  typedef std::vector<void*> CheckerList;
  typedef std::vector<double> ValueContainer;

  template<typename StrictWeakOrdering, typename KeyList,
           typename ValueContainer>
  static void CallHykSort(MpiComm comm, StrictWeakOrdering comp,
                          unsigned int kway, bool is_using_gpu, size_t num_v,
                          KeyList *_keys, ValueContainer *v_list) {
    if (num_v == 1) {
      HykSort(comm, comp, kway, is_using_gpu, _keys, &v_list[0]);
    } else if (num_v == 2) {
      HykSort(comm, comp, kway, is_using_gpu, _keys, &v_list[0], &v_list[1]);
    } else if (num_v == 3) {
      HykSort(comm, comp, kway, is_using_gpu, _keys, &v_list[0], &v_list[1],
              &v_list[2]);
    } else if (num_v == 4) {
      HykSort(comm, comp, kway, is_using_gpu, _keys, &v_list[0], &v_list[1],
              &v_list[2], &v_list[3]);
    } else if (num_v == 5) {
      HykSort(comm, comp, kway, is_using_gpu, _keys, &v_list[0], &v_list[1],
              &v_list[2], &v_list[3], &v_list[4]);
    } else if (num_v == 6) {
      HykSort(comm, comp, kway, is_using_gpu, _keys, &v_list[0], &v_list[1],
              &v_list[2], &v_list[3], &v_list[4], &v_list[5]);
    } else if (num_v == 7) {
      HykSort(comm, comp, kway, is_using_gpu, _keys, &v_list[0], &v_list[1],
              &v_list[2], &v_list[3], &v_list[4], &v_list[5], &v_list[6]);
    } else if (num_v == 8) {
      HykSort(comm, comp, kway, is_using_gpu, _keys, &v_list[0], &v_list[1],
              &v_list[2], &v_list[3], &v_list[4], &v_list[5], &v_list[6],
              &v_list[7]);
    } else if (num_v == 9) {
      HykSort(comm, comp, kway, is_using_gpu, _keys, &v_list[0], &v_list[1],
              &v_list[2], &v_list[3], &v_list[4], &v_list[5], &v_list[6],
              &v_list[7], &v_list[8]);
    } else if (num_v == 10) {
      HykSort(comm, comp, kway, is_using_gpu, _keys, &v_list[0], &v_list[1],
              &v_list[2], &v_list[3], &v_list[4], &v_list[5], &v_list[6],
              &v_list[7], &v_list[8], &v_list[9]);
    } else {  // Unsupported.
    }
  }

  template<typename StrictWeakOrdering, typename KeyList,
           typename ValueContainer>
  static void CallHyperQuickSort(MpiComm comm, StrictWeakOrdering comp,
                                 size_t num_v, KeyList *_keys,
                                 ValueContainer *v_list) {
    if (num_v == 1) {
      HyperQuickSort(comm, comp, _keys, &v_list[0]);
    } else if (num_v == 2) {
      HyperQuickSort(comm, comp, _keys, &v_list[0], &v_list[1]);
    } else if (num_v == 3) {
      HyperQuickSort(comm, comp, _keys, &v_list[0], &v_list[1], &v_list[2]);
    } else if (num_v == 4) {
      HyperQuickSort(comm, comp, _keys, &v_list[0],
                     &v_list[1], &v_list[2], &v_list[3]);
    } else if (num_v == 5) {
      HyperQuickSort(comm, comp, _keys, &v_list[0],
                     &v_list[1], &v_list[2], &v_list[3], &v_list[4]);
    } else if (num_v == 6) {
      HyperQuickSort(comm, comp, _keys, &v_list[0], &v_list[1],
                     &v_list[2], &v_list[3], &v_list[4], &v_list[5]);
    } else if (num_v == 7) {
      HyperQuickSort(comm, comp, _keys, &v_list[0], &v_list[1], &v_list[2],
                     &v_list[3], &v_list[4], &v_list[5], &v_list[6]);
    } else if (num_v == 8) {
      HyperQuickSort(comm, comp, _keys, &v_list[0], &v_list[1],
                     &v_list[2], &v_list[3], &v_list[4],
                     &v_list[5], &v_list[6], &v_list[7]);
    } else if (num_v == 9) {
      HyperQuickSort(comm, comp, _keys, &v_list[0], &v_list[1],
                     &v_list[2], &v_list[3], &v_list[4],
                     &v_list[5], &v_list[6], &v_list[7], &v_list[8]);
    } else if (num_v == 10) {
      HyperQuickSort(comm, comp, _keys, &v_list[0], &v_list[1],
                     &v_list[2], &v_list[3], &v_list[4], &v_list[5],
                     &v_list[6], &v_list[7], &v_list[8], &v_list[9]);
    } else {  // Unsupported.
    }
  }

  template<typename StrictWeakOrdering, typename KeyList,
           typename ValueContainer>
  static void CallSampleSort(MpiComm comm, StrictWeakOrdering comp,
                             size_t num_v, KeyList *_keys,
                             ValueContainer *v_list) {
    if (num_v == 1) {
      SampleSort(comm, comp, _keys, &v_list[0]);
    } else if (num_v == 2) {
      SampleSort(comm, comp, _keys, &v_list[0], &v_list[1]);
    } else if (num_v == 3) {
      SampleSort(comm, comp, _keys, &v_list[0], &v_list[1], &v_list[2]);
    } else if (num_v == 4) {
      SampleSort(comm, comp, _keys, &v_list[0],
                 &v_list[1], &v_list[2], &v_list[3]);
    } else if (num_v == 5) {
      SampleSort(comm, comp, _keys, &v_list[0],
                 &v_list[1], &v_list[2], &v_list[3], &v_list[4]);
    } else if (num_v == 6) {
      SampleSort(comm, comp, _keys, &v_list[0], &v_list[1],
                 &v_list[2], &v_list[3], &v_list[4], &v_list[5]);
    } else if (num_v == 7) {
      SampleSort(comm, comp, _keys, &v_list[0], &v_list[1], &v_list[2],
                 &v_list[3], &v_list[4], &v_list[5], &v_list[6]);
    } else if (num_v == 8) {
      SampleSort(comm, comp, _keys, &v_list[0], &v_list[1],
                 &v_list[2], &v_list[3], &v_list[4],
                 &v_list[5], &v_list[6], &v_list[7]);
    } else if (num_v == 9) {
      SampleSort(comm, comp, _keys, &v_list[0], &v_list[1],
                 &v_list[2], &v_list[3], &v_list[4],
                 &v_list[5], &v_list[6], &v_list[7], &v_list[8]);
    } else if (num_v == 10) {
      SampleSort(comm, comp, _keys, &v_list[0], &v_list[1],
                 &v_list[2], &v_list[3], &v_list[4], &v_list[5],
                 &v_list[6], &v_list[7], &v_list[8], &v_list[9]);
    } else {  // Unsupported.
    }
  }

  template<typename T>
  static inline int64_t CapacityToNElems(const std::string &cap_str) {
    std::stringstream ss(cap_str);
    float num = 0;
    ss >> num;
    if (num <= 0) return 0;
    std::string postfix;
    ss >> postfix;
    postfix = ToLower(postfix);
    int mul = 0;
    if (postfix.compare("gb") == 0) {  // Gigabytes
      mul = 0x40000000;  // 2^30
    } else if (postfix.compare("mb") == 0) {  // Megabytes
      mul = 0x100000;  // 2^20;
    } else if (postfix.compare("kb") == 0) {  // Kilobytes
      mul = 0x400;  // 2^10;
    } else {  // Unsupported
    }
    return std::ceil(num * mul / sizeof(T));
  }

  template<typename RandomAccessIterator>
  static inline bool CheckValid(bool is_increasing, RandomAccessIterator first,
                                RandomAccessIterator last) {
    int64_t size = last - first;
    bool state = true;
    int64_t last_idx = size - 1;
    int n_threads = OmpUtils::GetMaxThreads();
    if (is_increasing) {
      FOR_PARALLEL_COND(n_threads, last_idx, i, state,
                        {
                          if (first[i] > first[i+1]) state = false;
                        });
    } else {
      FOR_PARALLEL_COND(n_threads, last_idx, i, state,
                        {
                          if (first[i] < first[i+1]) state = false;
                        });
    }
    return state;
  }

  template<typename RandomAccessIterator>
  static inline bool CheckValid(MpiComm comm, bool is_increasing,
                                RandomAccessIterator first,
                                RandomAccessIterator last) {
    typedef typename std::iterator_traits<RandomAccessIterator>::value_type
        ValueType;
    bool state = true;
    int64_t size = last - first;
    if (size == 0) MpiUtils::MpiAbort(comm, 1);

    // Global check.
    std::vector<ValueType> mins;
    std::vector<ValueType> maxs;
    ValueType min = (is_increasing)? first[0] : first[size - 1];
    ValueType max = (is_increasing)? first[size - 1] : first[0];
    ParUtils::Gather(comm, min, &mins);
    ParUtils::Gather(comm, max, &maxs);
    int myrank = MpiUtils::MpiCommRank(comm);
    int np = MpiUtils::MpiCommSize(comm);
    if (myrank == 0) {
      int last_np = np - 1;
      if (is_increasing) {
        for (int i = 0; i < last_np; i++) {
          if (maxs[i] > mins[i+1]) {
            state = false;
            break;
          }
        }
      } else {
        for (int i = 0; i < last_np; i++) {
          if (mins[i] < maxs[i+1]) {
            state = false;
            break;
          }
        }
      }
    }
    // Local check.
    if (state) state = CheckValid(is_increasing, first, last);

    MpiUtils::MpiBarrier(comm);
    return state;
  }

  template<typename T>
  static inline void ClearList(std::vector<T*> *_list) {
    size_t size = _list->size();
    for (size_t i = 0; i < size; i++) {
      delete (*_list)[i];
    }
    _list->clear();
  }

  static inline void DeleteChecker(void *checker, const std::string &type_str) {
    if (type_str.compare("double") == 0) {
      delete reinterpret_cast<Utils::DataChecker<double>*>(checker);
    } else if (type_str.compare("unsigned long") == 0) {
      delete reinterpret_cast<Utils::DataChecker<uint64_t>*>(checker);
    } else if (type_str.compare("float") == 0) {
      delete reinterpret_cast<Utils::DataChecker<float>*>(checker);
    } else if (type_str.compare("long") == 0) {
      delete reinterpret_cast<Utils::DataChecker<int64_t>*>(checker);
    } else if (type_str.compare("unsigned int") == 0) {
      delete reinterpret_cast<Utils::DataChecker<unsigned int>*>(checker);
    } else if (type_str.compare("int") == 0) {
      delete reinterpret_cast<Utils::DataChecker<int>*>(checker);
    } else {  // Unknown checker.
    }
  }

  static inline void DeleteCheckers(Utils::CheckerList *checkers,
                      std::vector<std::string> *types) {
    size_t size = checkers->size();
    for (size_t i = 0; i < size; i++) {
      DeleteChecker((*checkers)[i], (*types)[i]);
    }
    checkers->clear();
    types->clear();
  }

  template<typename T, typename Int>
  static inline void GenDataUsingArithmeticProgression(T a1, Int n, T d,
                                                       bool shuffle,
                                                       std::vector<T> *_out) {
    std::vector<T> &out = *_out;
    out.resize(n);
    out[0] = a1;
    int n_threads = OmpUtils::GetMaxThreads();
    Int total = n - 1;
    FOR_PARALLEL(n_threads, total, i,
                 {
                   T ai = a1 + (i + 1) * d;
                   out[i + 1] = ai;
                 });

    if (shuffle) {
      Random r;
      r.Shuffle(out.begin(), out.end());
    }
  }

  template<typename Int, typename T>
  static inline void GenDataUsingLogFunc(Int start, Int n, bool shuffle,
                                         std::vector<T> *_out) {
    std::vector<T> &out = *_out;
    out.resize(n);
    int n_threads = OmpUtils::GetMaxThreads();
    Int end = n + start - 1;
    FOR_PARALLEL(n_threads, n, i,
                 {
                   out[i] = static_cast<T>(std::log10(i + start));
                 });

    if (shuffle) {
      Random r;
      r.Shuffle(out.begin(), out.end());
    }
  }

  template<typename Int, typename T>
  static inline void GenDataUsingLnFunc(Int start, Int n, bool shuffle,
                                        std::vector<T> *_out) {
    std::vector<T> &out = *_out;
    out.resize(n);
    int n_threads = OmpUtils::GetMaxThreads();
    Int end = n + start - 1;
    FOR_PARALLEL(n_threads, n, i,
                 {
                   out[i] = static_cast<T>(std::log(i + start));
                 });

    if (shuffle) {
      Random r;
      r.Shuffle(out.begin(), out.end());
    }
  }

  template<typename T, typename Int>
  static inline void GenDataIncreasing(T f1, Int n, T r_max,
                                       std::vector<T> *_out) {
    std::vector<T> &out = *_out;
    out.resize(n);
    T f_i_prev = f1;
    out[0] = f1;
    Random r;
    r.Next(static_cast<T>(0), r_max, n-1, &out[1]);
    for (Int i = 2; i <= n; i++) {
      Int cur = i - 1;
      T fi = f_i_prev + out[cur];
      f_i_prev = fi;
      out[cur] = fi;
    }
  }

  template<typename T, typename Int>
  static void GenDataBucketDistribution(MpiComm comm, int p, Int count,
                                        std::vector<T> *_out) {
    std::vector<T> &out = *_out;
    out.resize(count);
    Int n_each_blk = CalcNumElemsEachBlock(count, p);
    Int n_last_blk = count - (p-1) * n_each_blk;
    Int n_each_section = CalcNumElemsEachBlock(n_each_blk, p);
    Int n_last_section = n_each_blk - (p-1) * n_each_section;
    Int n_each_section_of_the_last = CalcNumElemsEachBlock(n_last_blk, p);
    Int n_last_section_of_the_last = n_last_blk -
                                     (p-1) * n_each_section_of_the_last;
    double range = MAX_INT / static_cast<double>(p);  // 2^31 / p.
    Int idx = 0;
    int n_threads = OmpUtils::GetMaxThreads();
    int next_to_last_p = p - 1;
    int myrank = (comm != NULL)? MpiUtils::MpiCommRank(comm) : 0;
    Random r = (comm != NULL)? (myrank) : Random();
    FOR_PARALLEL(n_threads, p, b_it,
                 {
                   for (int s_it = 1; s_it <= p; s_it++) {
                     T min = static_cast<T>((s_it-1) * range);
                     T max = static_cast<T>(s_it * range - 1);
                     Int n_elems = 0;
                     if (b_it == next_to_last_p) {
                       n_elems = (s_it == p)? n_last_section_of_the_last :
                                              n_each_section_of_the_last;
                     } else {
                       n_elems = (s_it == p)? n_last_section : n_each_section;
                     }
                     r.Next(min, max, n_elems, &out[idx]);
                     idx += n_elems;
                   }
                 });
  }

  template<typename T, typename Int>
  static inline void GenDataGaussianDistribution(MpiComm comm, Int count,
                                                 std::vector<T> *_out) {
    std::vector<T> &out = *_out;
    out.resize(count);
    T min = 0;
    T max = MAX_INT;  // 2^31.
    int n_threads = OmpUtils::GetMaxThreads();
    int myrank = (comm != NULL)? MpiUtils::MpiCommRank(comm) : 0;
    Random r = (comm != NULL)? (myrank) : Random();
    FOR_PARALLEL(n_threads, count, i,
                 {
                   T sum = 0;
                   for (int j = 1; j <= 4; j++) {
                     sum += r.Next(min, max);
                   }
                   out[i] = sum / 4;
                 });
  }

  template<typename T, typename Int>
  static void GenDataStaggeredDistribution(MpiComm comm, int p, Int count,
                                           std::vector<T> *_out) {
    std::vector<T> &out = *_out;
    out.resize(count);
    Int n_each_blk = CalcNumElemsEachBlock(count, p);
    Int n_last_blk = count - (p-1) * n_each_blk;
    double range = MAX_INT / static_cast<double>(p);  // 2^31 / p.
    Int idx = 0;
    int n_threads = OmpUtils::GetMaxThreads();
    int half_p = p / 2;
    int myrank = (comm != NULL)? MpiUtils::MpiCommRank(comm) : 0;
    Random r = (comm != NULL)? (myrank) : Random();
    FOR_PARALLEL(n_threads, p, b_it,
                 {
                   int i = b_it + 1;
                   T min = 0; T max = 0;
                   if (i <= half_p) {
                     min = static_cast<T>((2*i + 1) * range);
                     max = static_cast<T>((2*i + 2) * range - 1);
                   } else {
                     min = static_cast<T>((2*i - p) * range);
                     max = static_cast<T>((2*i - p + 1) * range - 1);
                   }
                   Int n_elems = (i == p)? n_last_blk : n_each_blk;
                   r.Next(min, max, n_elems, &out[idx]);
                   idx += n_elems;
                 });
  }

  template<typename T, typename Int>
  static inline void GenDataUniformDistribution(MpiComm comm, Int count,
                                                std::vector<T> *_out) {
    std::vector<T> &out = *_out;
    out.resize(count);
    T min = 0;
    T max = MAX_INT;  // 2^31.
    int myrank = (comm != NULL)? MpiUtils::MpiCommRank(comm) : 0;
    Random r = (comm != NULL)? (myrank) : Random();
    r.Next(min, max, count, &out[0]);
  }

  template<typename T, typename Int>
  static inline void GenDataZeroDistribution(Int count, std::vector<T> *_out) {
    std::vector<T> &out = *_out;
    out.resize(count);
    Random r;
    T constant = r.Next<T>(0, MAX_INT);
    int n_threads = OmpUtils::GetMaxThreads();
    FOR_PARALLEL(n_threads, count, i,
                 {
                   out[i] = constant;
                 });
  }

  static inline std::string GetFilePath(const std::string &parent_path,
                                        int rank,
                                        const std::string &ext = ".dat") {
    std::string path = parent_path;
    path += std::string("/");
    path += ToString(rank) + std::string(ext);
    return path;
  }

  template<typename T>
  static inline void ReadData(int myrank, const std::string &parent_path,
                              const std::string &ext, std::vector<T> *_out) {
    std::string path = GetFilePath(parent_path, myrank, ext);
    IFStream ifs(path);
    int64_t size = 0;
    ifs.Read(&size);
    ifs.Read(size, _out);
  }

  template<typename RandomAccessIterator>
  static inline void SaveData(
      int np, const std::string &parent_path, const std::string &ext,
      RandomAccessIterator first, RandomAccessIterator last) {
    typedef typename std::iterator_traits<RandomAccessIterator>::value_type
        ValueType;
    int64_t total = last - first;
    int64_t nelem = CalcNumElemsEachBlock(total, np);
    int n_threads = OmpUtils::GetMaxThreads();
    FOR_PARALLEL(
        n_threads, np, i,
        {
          std::string path = GetFilePath(parent_path, i, ext);
          OFStream ofs(path);
          int64_t start_idx = nelem * i;
          int64_t end_idx = start_idx + nelem - 1;
          end_idx = std::min(end_idx, total - 1);
          int64_t size = end_idx - start_idx + 1;
          ofs.Write(size);
          ofs.Write(first + start_idx, first + end_idx + 1);
        });
  }

  template<typename RandomAccessIterator>
  static inline void SaveMyData(
      int myrank, const std::string &parent_path, const std::string &ext,
      RandomAccessIterator first, RandomAccessIterator last) {
    std::string path = GetFilePath(parent_path, myrank, ext);
    OFStream ofs(path);
    int64_t nelem = last - first;
    ofs.Write(nelem);
    ofs.Write(first, last);
  }

  static void SetCheckerComm(MpiComm comm,
                             const std::string &type_str, void *checker) {
    if (type_str.compare("double") == 0) {
      reinterpret_cast<Utils::DataChecker<double>*>(checker)->set_comm(comm);
    } else if (type_str.compare("unsigned long") == 0) {
      reinterpret_cast<Utils::DataChecker<uint64_t>*>(checker)->set_comm(comm);
    } else if (type_str.compare("float") == 0) {
      reinterpret_cast<Utils::DataChecker<float>*>(checker)->set_comm(comm);
    } else if (type_str.compare("long") == 0) {
      reinterpret_cast<Utils::DataChecker<int64_t>*>(checker)->set_comm(comm);
    } else if (type_str.compare("unsigned int") == 0) {
      reinterpret_cast<Utils::DataChecker<unsigned int>*>(
          checker)->set_comm(comm);
    } else if (type_str.compare("int") == 0) {
      reinterpret_cast<Utils::DataChecker<int>*>(checker)->set_comm(comm);
    } else {  // Unknown checker.
    }
  }

  static void SetCheckerComm(MpiComm comm,
                             const std::vector<std::string> &types,
                             CheckerList *_checkers) {
    size_t size = _checkers->size();
    for (size_t i = 0; i < size; i++) {
      SetCheckerComm(comm, types[i], (*_checkers)[i]);
    }
  }

  template<typename T, template<typename TT> class Checker>
  static bool Sort(int argc, char *argv[], Algorithm algo, bool reverse,
                   unsigned int kway, bool is_using_gpu,
                   const std::string &data_path, const std::string &ext) {
    return Sort<T, Checker>(argc, argv, algo, reverse, kway,
                            is_using_gpu, data_path, ext, NULL);
  }

  template<typename T, template<typename TT> class Checker>
  static bool Sort(int argc, char *argv[], Algorithm algo, bool reverse,
                   unsigned int kway, bool is_using_gpu,
                   const std::string &data_path, const std::string &ext,
                   ParameterList *params) {
    MpiUtils::MpiInit(&argc, &argv);
    MpiComm comm = MPI_COMM_WORLD;
    Checker<T> checker(comm);
    checker.set_params(params);
    bool suc = false;
    if (!reverse) {
      suc = Sort<T>(comm, std::less<T>(), algo, kway,
                    is_using_gpu, data_path, ext, checker);
    } else {
      suc = Sort<T>(comm, std::greater<T>(), algo, kway,
                    is_using_gpu, data_path, ext, checker);
    }

    gpusort::MpiUtils::MpiFinalize();
    return suc;
  }

  template<typename T>
  static bool Sort(int argc, char *argv[], gpusort::Algorithm algo,
                   bool reverse, unsigned int kway,
                   bool is_using_gpu, const std::string &data_path,
                   const std::vector<std::string> &types,
                   CheckerList *_checkers) {
    MpiUtils::MpiInit(&argc, &argv);
    MpiComm comm = MPI_COMM_WORLD;
    SetCheckerComm(comm, types, _checkers);
    bool suc = false;
    if (!reverse) {
      suc = Sort<T>(comm, std::less<T>(), algo, kway,
                    is_using_gpu, data_path, *_checkers);
    } else {
      suc = Sort<T>(comm, std::greater<T>(), algo, kway,
                    is_using_gpu, data_path, *_checkers);
    }

    gpusort::MpiUtils::MpiFinalize();
    return suc;
  }

  template<typename T, typename StrictWeakOrdering>
  static bool Sort(MpiComm comm, StrictWeakOrdering comp,
                   Algorithm algo, unsigned int kway, bool is_using_gpu,
                   const std::string &data_path, const std::string &ext,
                   const DataChecker<T> &checker) {
    int myrank = MpiUtils::MpiCommRank(comm);
    // Find out number of processes
    int np = MpiUtils::MpiCommSize(comm);
    if (myrank == 0) {
      std::cout << "Number of processes: " << np << std::endl;
      std::cout << "Max threads: " << OmpUtils::GetMaxThreads() << std::endl;
    }

    // Load raw data from a file.
    std::cout << "Rank " << myrank << " is loading input data..." << std::endl;
    std::vector<T> list;
  #ifdef __DO_BENCHMARK
    Timer load_data_tm;
    load_data_tm.Start();
  #endif
    Utils::ReadData(myrank, data_path, ext, &list);
  #ifdef __DO_BENCHMARK
    load_data_tm.Stop();
  #endif
    size_t size = list.size();
    if (size == 0) {
      std::cout << "No data to sort at rank " << myrank << "." << std::endl;
      MpiUtils::MpiAbort(comm, 1);
    }
    std::cout << "Rank " << myrank << " finished loading data."
              << " Data size = " << size << std::endl;
    MpiUtils::MpiBarrier(comm);

    switch (algo) {
      // Using HykSort to sort input data.
      case HYK_SORT_ALGORITHM:
        HykSort(comm, comp, kway, is_using_gpu, &list);
        break;

      // Using HyperQuickSort to sort input data.
      case HYPER_QUICK_SORT_ALGORITHM:
        HyperQuickSort(comm, comp, &list);
        break;

      // Using SampleSort to sort input data.
      default:
        SampleSort(comm, comp, &list);
        break;
    }

  #ifdef __DO_BENCHMARK
    ParUtils::ShowBenchmark(comm, load_data_tm, "Loading data");
  #endif
    size = list.size();
    std::cout << "Rank " << myrank << " finished sorting. New data size = "
              << size << std::endl;
    //
    // Validate the result
    //
    std::cout << "Rank: " << myrank << " is validating..." << std::endl;
    bool is_increasing = IsIncreasing(comp);
    bool suc = const_cast<DataChecker<T>&>(checker).Check(list, is_increasing);
    if (!suc) {
      std::cout << "Fail at Rank: " << myrank << std::endl;
    } else {
      std::cout << "Rank: " << myrank << " is completed!" << std::endl;
    }
    return suc;
  }

  template<typename T, typename StrictWeakOrdering>
  static bool Sort(MpiComm comm, StrictWeakOrdering comp,
                   Algorithm algo, unsigned int kway,
                   bool is_using_gpu, const std::string &data_path,
                   const CheckerList &checkers) {
    int myrank = MpiUtils::MpiCommRank(comm);
    // Find out number of processes
    int np = MpiUtils::MpiCommSize(comm);
    if (myrank == 0) {
      std::cout << "Number of processes: " << np << std::endl;
      std::cout << "Max threads: " << OmpUtils::GetMaxThreads() << std::endl;
    }

    // Load keys from file.
    std::cout << "Rank " << myrank << " is loading input data..." << std::endl;
    std::vector<T> keys;
  #ifdef __DO_BENCHMARK
    Timer load_data_tm;
    load_data_tm.Start();
  #endif
    Utils::ReadData(myrank, data_path, kKeyFileExt, &keys);
  #ifdef __DO_BENCHMARK
    load_data_tm.Stop();
  #endif
    // Load values from file.
    size_t num_v = checkers.size() - 1;
    ValueContainer *v_list = new ValueContainer[num_v];
    bool *states = new bool[num_v];
    for (size_t i = 0; i < num_v; i++) states[i] = true;  // Init states.
    int n_threads = OmpUtils::GetMaxThreads();

  #ifdef __DO_BENCHMARK
    load_data_tm.Start();
  #endif
    FOR_PARALLEL(n_threads, num_v, i,
                 {
                   ValueContainer &v = v_list[i];
                   Utils::ReadData(myrank, data_path,
                                   kValueFileExtPrefix + ToString(i+1), &v);
                   if (keys.size() != v.size()) {
                     std::cout << "Data corrupted." << std::endl;
                     states[i] = false;
                   }
                 });
  #ifdef __DO_BENCHMARK
    load_data_tm.Stop();
  #endif

    for (size_t i = 0; i < num_v; i++)
      if (!states[i]) {
        delete[] v_list;
        delete[] states;
        MpiUtils::MpiAbort(comm, 1);
      }

    std::cout << "Rank " << myrank << " finished loading data."
              << " Data size = " << keys.size() << std::endl;
    MpiUtils::MpiBarrier(comm);

    switch (algo) {
      // Using HykSort to sort input data.
      case HYK_SORT_ALGORITHM:
        CallHykSort(comm, comp, kway, is_using_gpu, num_v, &keys, v_list);
        break;

      // Using HyperQuickSort to sort input data.
      case HYPER_QUICK_SORT_ALGORITHM:
        CallHyperQuickSort(comm, comp, num_v, &keys, v_list);
        break;

      // Using SampleSort to sort input data.
      default:
        CallSampleSort(comm, comp, num_v, &keys, v_list);
        break;
    }
  #ifdef __DO_BENCHMARK
    ParUtils::ShowBenchmark(comm, load_data_tm, "Loading data");
  #endif
    std::cout << "Rank " << myrank << " finished sorting. New data size = "
              << keys.size() << std::endl;
    //
    // Validate the result
    //
    std::cout << "Rank: " << myrank << " is validating..." << std::endl;  
    bool is_increasing = IsIncreasing(comp);
    // Validate keys.
    DataChecker<T> *keys_checker
        = reinterpret_cast<DataChecker<T>*>(checkers[0]);
    bool suc = keys_checker->Check(keys, is_increasing);
    if (!suc) {
      std::cout << "Fail at Rank " << myrank << ". Reason: keys." << std::endl;
    } else {
      // Validate values.
      for (size_t i = 1; i <= num_v; i++) {
        ValueCheckerBase *v_checker
            = reinterpret_cast<ValueCheckerBase*>(checkers[i]);
        states[i-1] = v_checker->Check(v_list[i-1], is_increasing);
        if (!states[i-1]) {
          std::cout << "Fail at Rank " << myrank << ". Reason: v" << i
                    << "." << std::endl;
        }
      }
      for (size_t i = 0; i < num_v; i++)
        if (!states[i]) {
          suc = false;
          break;
        }
    }

    if (suc) std::cout << "Rank: " << myrank << " is completed!" << std::endl;
    delete[] v_list;
    delete[] states;
    return suc;
  }

  static inline void SplitString(const std::string &str, char del,
                                 std::vector<std::string> *out) {
    std::istringstream ss(str);
    std::string s;
    while (std::getline(ss, s, del)) {
        out->push_back(s);
    }
  }

  static inline Algorithm ToAlgorithm(const std::string &str) {
    std::string _str = ToLower(str);
    if (_str.compare("hyk_sort") == 0) {
      return HYK_SORT_ALGORITHM;
    } else if (_str.compare("hyper_quick_sort") == 0) {
      return HYPER_QUICK_SORT_ALGORITHM;
    } else if (_str.compare("sample_sort") == 0) {
      return SAMPLE_SORT_ALGORITHM;
    } else {  // Unknown algorithm.
      return UNKNOWN_ALGORITHM;
    }
  }

  static inline bool ToBool(const std::string &str) {
    std::string _str = ToLower(str);
    std::istringstream is(_str);
    bool ret;
    is >> std::boolalpha >> ret;
    return ret;
  }

  static inline char ToChar(const std::string &str) {
    return static_cast<char>(ToInt(str));
  }

  static inline double ToDouble(const std::string &str) {
  #if defined(__USING_CPP_0X) || defined(__USING_CPP_11)
    return std::stod(str);
  #else
    return std::strtod(str.c_str(), NULL);
  #endif
  }

  static inline float ToFloat(const std::string &str) {
  #if defined(__USING_CPP_0X) || defined(__USING_CPP_11)
    return std::strtof(str.c_str(), NULL);
  #else
    return static_cast<float>(ToDouble(str));
  #endif
  }

  static inline int ToInt(const std::string &str) {
  #if defined(__USING_CPP_0X) || defined(__USING_CPP_11)
    return std::stoi(str);
  #else
    return std::strtol(str.c_str(), NULL, 10);
  #endif
  }

  static inline int64_t ToInt64(const std::string &str) {
  #if defined(__USING_CPP_0X) || defined(__USING_CPP_11)
    return std::stol(str);
  #else
    return static_cast<int64_t>(ToDouble(str));
  #endif
  }

  static inline std::string ToLower(const std::string &str) {
    std::string ret(str);
    std::transform(str.begin(), str.end(), ret.begin(), ::tolower);
    return ret;
  }

  static inline short ToShort(const std::string &str) {
    return static_cast<short>(ToInt(str));
  }

  template<typename T>
  static inline std::string ToString(T value) {
  #if defined(__USING_CPP_11)
    return std::to_string(value);
  #elif defined(__USING_CPP_0X)
    if (std::is_integral<T>::value) {
      if (std::is_unsigned<T>::value) {
        return std::to_string(static_cast<long long unsigned int>(value));
      } else {
        return std::to_string(static_cast<long long int>(value));
      }
    } else {
      return std::to_string(static_cast<long double>(value));
    }
  #else
    return static_cast<std::ostringstream*>(
        &(std::ostringstream() << value))->str();
  #endif
  }

  template<typename T>
  static inline T ToT(const std::string &str) {
    const std::type_info& t = typeid(T);
    if (t == typeid(bool)) {
      return ToBool(str);
    } else if (t == typeid(short)) {
      return ToShort(str);
    } else if (t == typeid(int)) {
      return ToInt(str);
    } else if (t == typeid(long)) {
      return ToInt64(str);
    } else if (t == typeid(unsigned short)) {
      return ToUShort(str);
    } else if (t == typeid(unsigned int)) {
      return ToUInt(str);
    } else if (t == typeid(unsigned long)) {
      return ToUInt64(str);
    } else if (t == typeid(float)) {
      return ToFloat(str);
    } else if (t == typeid(char)) {
      return ToChar(str);
    } else if (t == typeid(unsigned char)) {
      return ToUChar(str);
    } else {  // Default
      return ToDouble(str);
    }
  }

  static inline unsigned char ToUChar(const std::string &str) {
    return static_cast<unsigned char>(ToUInt(str));
  }

  static inline unsigned int ToUInt(const std::string &str) {
  #if defined(__USING_CPP_0X) || defined(__USING_CPP_11)
    return std::stoul(str);
  #else
    return static_cast<unsigned int>(ToDouble(str));
  #endif
  }

  static inline uint64_t ToUInt64(const std::string &str) {
  #if defined(__USING_CPP_0X) || defined(__USING_CPP_11)
    return std::stoull(str);
  #else
    return std::strtoul(str.c_str(), NULL, 10);
  #endif
  }

  static inline std::string ToUpper(const std::string &str) {
    std::string ret(str);
    std::transform(str.begin(), str.end(), ret.begin(), ::toupper);
    return ret;
  }

  static inline unsigned short ToUShort(const std::string &str) {
    return static_cast<unsigned short>(ToUInt(str));
  }

  static inline std::string Trim(const std::string& str,
                                 const std::string& whitespace = " \t") {
    size_t str_begin = str.find_first_not_of(whitespace);
    if (str_begin == std::string::npos) return "";  // No content.
    size_t str_end = str.find_last_not_of(whitespace);
    size_t str_range = str_end - str_begin + 1;
    return str.substr(str_begin, str_range);
  }
};

}  // namespace gpusort

#endif  // GPUSORT_TEST_UTILS_H_
