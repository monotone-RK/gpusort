#include <getopt.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "arithmetic_progression_checker.h"
#include "bucket_checker.h"
#include "gaussian_checker.h"
#include "increasing_sequence_checker.h"
#include "ln_checker.h"
#include "log_checker.h"
#include "staggered_checker.h"
#include "uniform_checker.h"
#include "zero_checker.h"
#include "utils.h"

struct ValueContainer1 {
  double v1_;

  double GetV(int v_idx) {
    if (v_idx == 1) {
      return v1_;
    } else {  // Invalid call.
      return 0;
    }
  }

  void SetV(int v_idx, double val) {
    if (v_idx == 1) {
      v1_ = val;
    } else {  // Invalid call.
    }
  }
};

struct ValueContainer2 {
  double v1_, v2_;

  double GetV(int v_idx) {
    if (v_idx == 1) {
      return v1_;
    } else if (v_idx == 2) {
      return v2_;
    } else {  // Invalid call.
      return 0;
    }
  }

  void SetV(int v_idx, double val) {
    if (v_idx == 1) {
      v1_ = val;
    } else if (v_idx == 2) {
      v2_ = val;
    } else {  // Invalid call.
    }
  }
};

struct ValueContainer3 {
  double v1_, v2_, v3_;

  double GetV(int v_idx) {
    if (v_idx == 1) {
      return v1_;
    } else if (v_idx == 2) {
      return v2_;
    } else if (v_idx == 3) {
      return v3_;
    } else {  // Invalid call.
      return 0;
    }
  }

  void SetV(int v_idx, double val) {
    if (v_idx == 1) {
      v1_ = val;
    } else if (v_idx == 2) {
      v2_ = val;
    } else if (v_idx == 3) {
      v3_ = val;
    } else {  // Invalid call.
    }
  }
};

struct ValueContainer4 {
  double v1_, v2_, v3_, v4_;

  double GetV(int v_idx) {
    if (v_idx == 1) {
      return v1_;
    } else if (v_idx == 2) {
      return v2_;
    } else if (v_idx == 3) {
      return v3_;
    } else if (v_idx == 4) {
      return v4_;
    } else {  // Invalid call.
      return 0;
    }
  }

  void SetV(int v_idx, double val) {
    if (v_idx == 1) {
      v1_ = val;
    } else if (v_idx == 2) {
      v2_ = val;
    } else if (v_idx == 3) {
      v3_ = val;
    } else if (v_idx == 4) {
      v4_ = val;
    } else {  // Invalid call.
    }
  }
};

struct ValueContainer5 {
  double v1_, v2_, v3_, v4_, v5_;

  double GetV(int v_idx) {
    if (v_idx == 1) {
      return v1_;
    } else if (v_idx == 2) {
      return v2_;
    } else if (v_idx == 3) {
      return v3_;
    } else if (v_idx == 4) {
      return v4_;
    } else if (v_idx == 5) {
      return v5_;
    } else {  // Invalid call.
      return 0;
    }
  }

  void SetV(int v_idx, double val) {
    if (v_idx == 1) {
      v1_ = val;
    } else if (v_idx == 2) {
      v2_ = val;
    } else if (v_idx == 3) {
      v3_ = val;
    } else if (v_idx == 4) {
      v4_ = val;
    } else if (v_idx == 5) {
      v5_ = val;
    } else {  // Invalid call.
    }
  }
};

struct ValueContainer6 {
  double v1_, v2_, v3_, v4_, v5_, v6_;

  double GetV(int v_idx) {
    if (v_idx == 1) {
      return v1_;
    } else if (v_idx == 2) {
      return v2_;
    } else if (v_idx == 3) {
      return v3_;
    } else if (v_idx == 4) {
      return v4_;
    } else if (v_idx == 5) {
      return v5_;
    } else if (v_idx == 6) {
      return v6_;
    } else {  // Invalid call.
      return 0;
    }
  }

  void SetV(int v_idx, double val) {
    if (v_idx == 1) {
      v1_ = val;
    } else if (v_idx == 2) {
      v2_ = val;
    } else if (v_idx == 3) {
      v3_ = val;
    } else if (v_idx == 4) {
      v4_ = val;
    } else if (v_idx == 5) {
      v5_ = val;
    } else if (v_idx == 6) {
      v6_ = val;
    } else {  // Invalid call.
    }
  }
};

struct ValueContainer7 {
  double v1_, v2_, v3_, v4_, v5_, v6_, v7_;

  double GetV(int v_idx) {
    if (v_idx == 1) {
      return v1_;
    } else if (v_idx == 2) {
      return v2_;
    } else if (v_idx == 3) {
      return v3_;
    } else if (v_idx == 4) {
      return v4_;
    } else if (v_idx == 5) {
      return v5_;
    } else if (v_idx == 6) {
      return v6_;
    } else if (v_idx == 7) {
      return v7_;
    } else {  // Invalid call.
      return 0;
    }
  }

  void SetV(int v_idx, double val) {
    if (v_idx == 1) {
      v1_ = val;
    } else if (v_idx == 2) {
      v2_ = val;
    } else if (v_idx == 3) {
      v3_ = val;
    } else if (v_idx == 4) {
      v4_ = val;
    } else if (v_idx == 5) {
      v5_ = val;
    } else if (v_idx == 6) {
      v6_ = val;
    } else if (v_idx == 7) {
      v7_ = val;
    } else {  // Invalid call.
    }
  }
};

struct ValueContainer8 {
  double v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_;

  double GetV(int v_idx) {
    if (v_idx == 1) {
      return v1_;
    } else if (v_idx == 2) {
      return v2_;
    } else if (v_idx == 3) {
      return v3_;
    } else if (v_idx == 4) {
      return v4_;
    } else if (v_idx == 5) {
      return v5_;
    } else if (v_idx == 6) {
      return v6_;
    } else if (v_idx == 7) {
      return v7_;
    } else if (v_idx == 8) {
      return v8_;
    } else {  // Invalid call.
      return 0;
    }
  }

  void SetV(int v_idx, double val) {
    if (v_idx == 1) {
      v1_ = val;
    } else if (v_idx == 2) {
      v2_ = val;
    } else if (v_idx == 3) {
      v3_ = val;
    } else if (v_idx == 4) {
      v4_ = val;
    } else if (v_idx == 5) {
      v5_ = val;
    } else if (v_idx == 6) {
      v6_ = val;
    } else if (v_idx == 7) {
      v7_ = val;
    } else if (v_idx == 8) {
      v8_ = val;
    } else {  // Invalid call.
    }
  }
};

struct ValueContainer9 {
  double v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_;

  double GetV(int v_idx) {
    if (v_idx == 1) {
      return v1_;
    } else if (v_idx == 2) {
      return v2_;
    } else if (v_idx == 3) {
      return v3_;
    } else if (v_idx == 4) {
      return v4_;
    } else if (v_idx == 5) {
      return v5_;
    } else if (v_idx == 6) {
      return v6_;
    } else if (v_idx == 7) {
      return v7_;
    } else if (v_idx == 8) {
      return v8_;
    } else if (v_idx == 9) {
      return v9_;
    } else {  // Invalid call.
      return 0;
    }
  }

  void SetV(int v_idx, double val) {
    if (v_idx == 1) {
      v1_ = val;
    } else if (v_idx == 2) {
      v2_ = val;
    } else if (v_idx == 3) {
      v3_ = val;
    } else if (v_idx == 4) {
      v4_ = val;
    } else if (v_idx == 5) {
      v5_ = val;
    } else if (v_idx == 6) {
      v6_ = val;
    } else if (v_idx == 7) {
      v7_ = val;
    } else if (v_idx == 8) {
      v8_ = val;
    } else if (v_idx == 9) {
      v9_ = val;
    } else {  // Invalid call.
    }
  }
};

struct ValueContainer10 {
  double v1_, v2_, v3_, v4_, v5_, v6_, v7_, v8_, v9_, v10_;

  double GetV(int v_idx) {
    if (v_idx == 1) {
      return v1_;
    } else if (v_idx == 2) {
      return v2_;
    } else if (v_idx == 3) {
      return v3_;
    } else if (v_idx == 4) {
      return v4_;
    } else if (v_idx == 5) {
      return v5_;
    } else if (v_idx == 6) {
      return v6_;
    } else if (v_idx == 7) {
      return v7_;
    } else if (v_idx == 8) {
      return v8_;
    } else if (v_idx == 9) {
      return v9_;
    } else if (v_idx == 10) {
      return v10_;
    } else {  // Invalid call.
      return 0;
    }
  }

  void SetV(int v_idx, double val) {
    if (v_idx == 1) {
      v1_ = val;
    } else if (v_idx == 2) {
      v2_ = val;
    } else if (v_idx == 3) {
      v3_ = val;
    } else if (v_idx == 4) {
      v4_ = val;
    } else if (v_idx == 5) {
      v5_ = val;
    } else if (v_idx == 6) {
      v6_ = val;
    } else if (v_idx == 7) {
      v7_ = val;
    } else if (v_idx == 8) {
      v8_ = val;
    } else if (v_idx == 9) {
      v9_ = val;
    } else if (v_idx == 10) {
      v10_ = val;
    } else {  // Invalid call.
    }
  }
};

#define CREATE_CHECKER(_ret, _class, _type_str, _p_cfg) \
  if (_type_str.compare("double") == 0) {\
    _ret = new _class<double>();\
    reinterpret_cast<_class<double>*>(_ret)->ParseParameters(_p_cfg);\
  } else if (_type_str.compare("unsigned long") == 0) {\
    _ret = new _class<uint64_t>();\
    reinterpret_cast<_class<uint64_t>*>(_ret)->ParseParameters(_p_cfg);\
  } else if (_type_str.compare("float") == 0) {\
    _ret = new _class<float>();\
    reinterpret_cast<_class<float>*>(_ret)->ParseParameters(_p_cfg);\
  } else if (_type_str.compare("long") == 0) {\
    _ret = new _class<int64_t>();\
    reinterpret_cast<_class<int64_t>*>(_ret)->ParseParameters(_p_cfg);\
  } else if (_type_str.compare("unsigned int") == 0) {\
    _ret = new _class<unsigned int>();\
    reinterpret_cast<_class<unsigned int>*>(_ret)->ParseParameters(_p_cfg);\
  } else if (_type_str.compare("int") == 0) {\
    _ret = new _class<int>();\
    reinterpret_cast<_class<int>*>(_ret)->ParseParameters(_p_cfg);\
  } else {\
    _ret = NULL;\
  }

void *CreateChecker(const std::string &c_str, std::string *type);

bool CreateCheckers(const std::string &v_str,
                    gpusort::Utils::CheckerList *_checkers,
                    std::vector<std::string> *_types);

template<typename KeyType, typename ValueContainerType>
bool DoWork(int np, const std::string &capacity,
            const std::string &data_path, gpusort::Utils::CheckerList &checkers,
            const std::vector<std::string> &types);

bool DoWork(const std::string &v_str, int np,
            const std::string &capacity, const std::string &data_path);

bool DoWork(int argc, char *argv[], gpusort::Algorithm algo,
            bool reverse, unsigned int kway, bool is_using_gpu,
            const std::string &data_path, const std::string &v_str);

template<typename KeyType>
bool DoWorkWrapper(int np, const std::string &capacity,
                   const std::string &data_path,
                   const gpusort::Utils::CheckerList &checkers,
                   const std::vector<std::string> &types);

int main(int argc, char *argv[]) {
  bool gen_flag = false;
  bool is_using_gpu = true;
  bool reverse_flag = false;
  std::string g_str = "", k_str = "", n_threads_str = "";
  std::string v_str = "ap, a1=1, d=2 : log=double, b=5";
  std::string algo_str = "hyk_sort";
  unsigned int kway = 2;
  int n_threads = 1;
  int c = -1;

  opterr = 0;
  while ((c = getopt(argc, argv, "a:dg:k:n:rv:")) != -1)
    switch (c) {
      case 'a':
        algo_str = std::string(optarg);
        break;

      case 'd':
        gen_flag = true;
        break;

      case 'g':
        g_str = std::string(optarg);
        break;

      case 'k':
        k_str = std::string(optarg);
        break;

      case 'n':
        n_threads_str = std::string(optarg);
        break;

      case 'r':
        reverse_flag = true;
        break;

      case 'v':
        v_str = std::string(optarg);
        break;

      case '?':
        if (optopt == 'a' || optopt == 'e' || optopt == 'g' || optopt == 'k' ||
            optopt == 'n' || optopt == 'v') {
          std::cout << "Option -" << static_cast<char>(optopt)
                    << " requires an argument.\n";
        } else {
          std::cout << "Unknown option -" << static_cast<char>(optopt) << "\n";
        }
        return 1;

      default:
        abort();
    }

  std::string capacity = "";
  int np = 0;
  if (gen_flag) {
    if (optind != argc - 3) {
      std::cout << gpusort::kErrorCapacityNProcessOrDataPathMsg << std::endl;
      return 1;
    }
    std::string np_str = std::string(argv[optind++]);
    capacity = std::string(argv[optind++]);
    np = gpusort::Utils::ToInt(np_str);
    if (np <= 0) {
      std::cout << gpusort::kErrorNProcessesMsg << std::endl;
      return 1;
    }
  } else if (optind != argc - 1) {
    std::cout << gpusort::kErrorDataPathMsg << std::endl;
    return 1;
  }
  std::string data_path = std::string(argv[optind]);

  // Set the maximum number of available threads for using in parallel regions.
  if (n_threads_str.length() != 0) {
    n_threads = gpusort::Utils::ToInt(n_threads_str);
    if (n_threads > 0)
      gpusort::OmpUtils::SetNumThreads(n_threads);
  }

  bool suc = false;
  if (gen_flag) {
    suc = DoWork(v_str, np, capacity, data_path);
  } else {
    if (g_str.length() != 0) is_using_gpu = gpusort::Utils::ToBool(g_str);
    if (k_str.length() != 0) kway = gpusort::Utils::ToUInt(k_str);
    if (kway == 0) kway = 2;
    gpusort::Algorithm algo = gpusort::Utils::ToAlgorithm(algo_str);
    if (algo == gpusort::UNKNOWN_ALGORITHM) {
      std::cout << gpusort::kErrorAlgorithmMsg << std::endl;
      return 1;
    }
    suc = DoWork(argc, argv, algo, reverse_flag, kway,
                 is_using_gpu, data_path, v_str);
  }
  if (!suc) return 1;

  return 0;
}

void *CreateChecker(const std::string &c_str, std::string *_type) {
  std::vector<std::string> p_list;
  gpusort::Utils::SplitString(c_str, PARAMETER_SEPARATOR, &p_list);
  std::vector<std::string> name_and_type;
  gpusort::Utils::SplitString(p_list[0], PARAMETER_SET_OPERATOR,
                              &name_and_type);
  p_list.erase(p_list.begin());
  std::string checker_name = gpusort::Utils::Trim(name_and_type[0]);
  std::string type = (name_and_type.size() > 1)?
                         gpusort::Utils::Trim(name_and_type[1]) : "double";
  void *ret = NULL;
  if (checker_name.compare("ap") == 0) {
    CREATE_CHECKER(ret, gpusort::ArithmeticProgressionChecker, type, p_list);
  } else if (checker_name.compare("bucket") == 0) {
    CREATE_CHECKER(ret, gpusort::BucketChecker, type, p_list);
//  } else if (checker_name.compare("copy") == 0) {
  } else if (checker_name.compare("gaussian") == 0) {
    CREATE_CHECKER(ret, gpusort::GaussianChecker, type, p_list);
  } else if (checker_name.compare("increasing") == 0) {
    CREATE_CHECKER(ret, gpusort::IncreasingSequenceChecker, type, p_list);
  } else if (checker_name.compare("ln") == 0) {
    CREATE_CHECKER(ret, gpusort::LnChecker, type, p_list);
  } else if (checker_name.compare("log") == 0) {
    CREATE_CHECKER(ret, gpusort::LogChecker, type, p_list);
  } else if (checker_name.compare("staggered") == 0) {
    CREATE_CHECKER(ret, gpusort::StaggeredChecker, type, p_list);
  } else if (checker_name.compare("uniform") == 0) {
    CREATE_CHECKER(ret, gpusort::UniformChecker, type, p_list);
  } else if (checker_name.compare("zero") == 0) {
    CREATE_CHECKER(ret, gpusort::ZeroChecker, type, p_list);
  } else {  // Unknown checker.
  }

  *_type = type;
  return ret;
}

bool CreateCheckers(const std::string &v_str,
                    gpusort::Utils::CheckerList *_checkers,
                    std::vector<std::string> *_types) {
  std::vector<void*> &checkers = *_checkers;
  std::vector<std::string> &types = *_types;
  std::vector<std::string> v_list;
  gpusort::Utils::SplitString(v_str, VALUE_SEPARATOR, &v_list);
  int num_v = v_list.size();
  checkers.resize(num_v);
  types.resize(num_v);
  int n_threads = gpusort::OmpUtils::GetMaxThreads();
  if (num_v > 0) {
    FOR_PARALLEL(n_threads, num_v, i,
                 {
                   checkers[i] = CreateChecker(v_list[i], &types[i]);
                   // TODO: Remove this. Now we don't support configuring
                   // value type on command line.
                   if (i > 0) types[i] = "double";
                 });
  }

  return true;
}

template<typename KeyType, typename ValueContainerType>
bool DoWork(int np, const std::string &capacity,
            const std::string &data_path,
            const gpusort::Utils::CheckerList &checkers,
            const std::vector<std::string> &types) {
  typedef std::pair<KeyType, ValueContainerType> PairType;

  int64_t nelem = gpusort::Utils::CapacityToNElems<PairType>(capacity);
  std::vector<KeyType> keys;
  int n_threads = gpusort::OmpUtils::GetMaxThreads();
  // Create a container which will contain all keys and values.
  std::vector<PairType> pair_container(nelem);
  size_t num_v = checkers.size() - 1;
  // Generate keys.
  gpusort::Utils::DataChecker<KeyType> *key_checker
      = reinterpret_cast<gpusort::Utils::DataChecker<KeyType>*>(checkers[0]);
  key_checker->Generate(nelem, &keys);
  // Assign generated keys to the container.
  FOR_PARALLEL(n_threads, nelem, i,
               {
                 pair_container[i].first = keys[i];
               });

  // Generate values and assign to the container.
  std::vector<double> v_list;
  for (size_t v = 1; v <= num_v; v++) {
    gpusort::Utils::DataChecker<double> *checker
        = reinterpret_cast<gpusort::Utils::DataChecker<double>*>(checkers[v]);
    checker->Generate(nelem, &v_list);
    FOR_PARALLEL(n_threads, nelem, j,
                 {
                   ValueContainerType &v_container = pair_container[j].second;
                   v_container.SetV(v, v_list[j]);
                 });
  }
  v_list.clear();
  // Randomly shuffle elements in the container.
  gpusort::Random r;
  r.Shuffle(&pair_container[0], &pair_container[nelem]);
  //
  // Save data in the container to files.
  //
  // Save keys first.
  int64_t n_each = gpusort::CalcNumElemsEachBlock(nelem, np);
  FOR_PARALLEL(n_threads, np, i,
               {
                 std::string path = gpusort::Utils::GetFilePath(
                                        data_path, i, gpusort::kKeyFileExt);
                 gpusort::OFStream out(path.c_str());
                 int64_t start_idx = n_each * i;
                 int64_t end_idx = start_idx + n_each - 1;
                 end_idx = std::min(end_idx, nelem - 1);
                 int64_t size = end_idx - start_idx + 1;
                 out.Write(size);
                 std::vector<double> tmp_k(size);
                 for (int64_t j = start_idx; j <= end_idx; j++) {
                   tmp_k[j - start_idx] = pair_container[j].first;
                 }
                 out.Write(&tmp_k[0], &tmp_k[size]);
               });

  // Save all values.
  for (size_t v = 1; v <= num_v; v++) {
    std::string ext = gpusort::kValueFileExtPrefix
                      + gpusort::Utils::ToString(v);
    FOR_PARALLEL(n_threads, np, i,
                 {
                   std::string path = gpusort::Utils::GetFilePath(data_path,
                                                                  i, ext);
                   gpusort::OFStream out(path.c_str());
                   int64_t start_idx = n_each * i;
                   int64_t end_idx = start_idx + n_each - 1;
                   end_idx = std::min(end_idx, nelem - 1);
                   int64_t size = end_idx - start_idx + 1;
                   out.Write(size);
                   std::vector<double> tmp_v(size);
                   for (int64_t j = start_idx; j <= end_idx; j++) {
                     tmp_v[j - start_idx] = pair_container[j].second.GetV(v);
                   }
                   out.Write(&tmp_v[0], &tmp_v[size]);
                 });
  }

  std::cout << "Data is saved successfully to " << data_path << std::endl;
}

template<typename KeyType>
bool DoWorkWrapper(int np, const std::string &capacity,
                   const std::string &data_path,
                   const gpusort::Utils::CheckerList &checkers,
                   const std::vector<std::string> &types) {
  size_t num_v = checkers.size() - 1;
  if (num_v == 1) {
    return DoWork<KeyType, ValueContainer1>(np, capacity, data_path,
                                            checkers, types);
  } else if (num_v == 2) {
    return DoWork<KeyType, ValueContainer2>(np, capacity, data_path,
                                            checkers, types);
  } else if (num_v == 3) {
    return DoWork<KeyType, ValueContainer3>(np, capacity, data_path,
                                            checkers, types);
  } else if (num_v == 4) {
    return DoWork<KeyType, ValueContainer4>(np, capacity, data_path,
                                            checkers, types);
  } else if (num_v == 5) {
    return DoWork<KeyType, ValueContainer5>(np, capacity, data_path,
                                            checkers, types);
  } else if (num_v == 6) {
    return DoWork<KeyType, ValueContainer6>(np, capacity, data_path,
                                            checkers, types);
  } else if (num_v == 7) {
    return DoWork<KeyType, ValueContainer7>(np, capacity, data_path,
                                            checkers, types);
  } else if (num_v == 8) {
    return DoWork<KeyType, ValueContainer8>(np, capacity, data_path,
                                            checkers, types);
  } else if (num_v == 9) {
    return DoWork<KeyType, ValueContainer9>(np, capacity, data_path,
                                            checkers, types);
  } else if (num_v == 10) {
    return DoWork<KeyType, ValueContainer10>(np, capacity, data_path,
                                             checkers, types);
  } else {  // Unsupported.
    return false;
  }
}

bool DoWork(const std::string &v_str, int np,
            const std::string &capacity, const std::string &data_path) {
  gpusort::Utils::CheckerList checkers;
  std::vector<std::string> types;
  bool suc = CreateCheckers(v_str, &checkers, &types);
  if (!suc) return false;
  CALL_FUNC_T(suc, types[0], DoWorkWrapper,
              (np, capacity, data_path, checkers, types), NULL_STMT);

  gpusort::Utils::DeleteCheckers(&checkers, &types);
  return suc;
}

bool DoWork(int argc, char *argv[], gpusort::Algorithm algo,
            bool reverse, unsigned int kway, bool is_using_gpu,
            const std::string &data_path, const std::string &v_str) {
  gpusort::Utils::CheckerList checkers;
  std::vector<std::string> types;
  bool suc = CreateCheckers(v_str, &checkers, &types);
  if (!suc) return false;
  CALL_FUNC_T(suc, types[0], gpusort::Utils::Sort,
              (argc, argv, algo, reverse, kway,
               is_using_gpu, data_path, types, &checkers),
              NULL_STMT);

  gpusort::Utils::DeleteCheckers(&checkers, &types);
  return suc;
}
