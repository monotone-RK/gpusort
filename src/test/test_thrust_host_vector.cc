#include <getopt.h>
#include <cstring>
#include <iostream>

#include "arithmetic_progression_checker.h"
#include "utils.h"

template<typename T>
bool DoWork(const std::string &a1_str, const std::string &d_str,
            int np, const std::string &capacity,
            const std::string &data_path, const std::string &ext);

template<typename T>
bool DoWork(T a1, T d, int np, const std::string &capacity,
            const std::string &data_path, const std::string &ext);

template<typename T>
bool DoWork(int argc, char *argv[], const std::string &a1_str,
            const std::string &d_str, gpusort::Algorithm algo,
            bool reverse, unsigned int kway, bool is_using_gpu,
            const std::string &data_path, const std::string &ext);

template<typename T>
bool DoWork(int argc, char *argv[], T a1, T d, gpusort::Algorithm algo,
            bool reverse, unsigned int kway, bool is_using_gpu,
            const std::string &data_path, const std::string &ext);

template<typename StrictWeakOrdering, typename T>
void SortWithThrustHostVector(gpusort::MpiComm comm, StrictWeakOrdering comp,
                              gpusort::Algorithm algo, unsigned int kway,
                              bool is_using_gpu, std::vector<T> *_list);

int main(int argc, char *argv[]) {
  bool gen_flag = false;
  bool is_using_gpu = true;
  bool reverse_flag = false;
  std::string a1_str = "0", d_str = "1", g_str = "", k_str = "";
  std::string n_threads_str = "";
  std::string algo_str = "hyk_sort", ext = gpusort::kDataFileExt;
  std::string t_str = "double";
  unsigned int kway = 2;
  int n_threads = 1;
  int c = -1;

  opterr = 0;
  while ((c = getopt(argc, argv, "a:c:de:f:g:k:n:rt:")) != -1)
    switch (c) {
      case 'a':
        algo_str = std::string(optarg);
        break;

      case 'c':
        d_str = std::string(optarg);
        break;

      case 'd':
        gen_flag = true;
        break;

      case 'e':
        ext = std::string(optarg);
        break;

      case 'f':
        a1_str = std::string(optarg);
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

      case 't':
        t_str = std::string(optarg);
        break;

      case '?':
        if (optopt == 'a' || optopt == 'c' || optopt == 'e' || optopt == 'f' ||
            optopt == 'g' || optopt == 'k' || optopt == 'n' || optopt == 't') {
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
    CALL_FUNC_T(suc, t_str, DoWork,
                (a1_str, d_str, np, capacity, data_path, ext),
                {
                  std::cout << gpusort::kErrorTypeMsg << std::endl;
                  return 1;
                });
  } else {
    if (g_str.length() != 0) is_using_gpu = gpusort::Utils::ToBool(g_str);
    if (k_str.length() != 0) kway = gpusort::Utils::ToUInt(k_str);
    if (kway == 0) kway = 2;
    gpusort::Algorithm algo = gpusort::Utils::ToAlgorithm(algo_str);
    if (algo == gpusort::UNKNOWN_ALGORITHM) {
      std::cout << gpusort::kErrorAlgorithmMsg << std::endl;
      return 1;
    }
    CALL_FUNC_T(suc, t_str, DoWork,
                (argc, argv, a1_str, d_str, algo, reverse_flag, kway,
                 is_using_gpu, data_path, ext),
                {
                  std::cout << gpusort::kErrorTypeMsg << std::endl;
                  return 1;
                });
  }
  if (!suc) return 1;

  return 0;
}

template<typename T>
bool DoWork(const std::string &a1_str, const std::string &d_str,
            int np, const std::string &capacity,
            const std::string &data_path, const std::string &ext) {
  T a1 = gpusort::Utils::ToT<T>(a1_str);
  T d = gpusort::Utils::ToT<T>(d_str);
  return DoWork(a1, d, np, capacity, data_path, ext);
}

template<typename T>
bool DoWork(T a1, T d, int np, const std::string &capacity,
            const std::string &data_path, const std::string &ext) {
  gpusort::ParameterList *params = new gpusort::ParameterList();
  params->Add(new gpusort::Parameter<T>(PARAMETER_A1_NAME, a1));
  params->Add(new gpusort::Parameter<T>(PARAMETER_D_NAME, d));
  params->Add(new gpusort::Parameter<bool>(PARAMETER_SHUFFLE_NAME, true));
  gpusort::ArithmeticProgressionChecker<T> checker;
  checker.set_params(params);
  return checker.Generate(np, capacity, data_path, ext);
}

template<typename T>
bool DoWork(int argc, char *argv[], const std::string &a1_str,
            const std::string &d_str, gpusort::Algorithm algo,
            bool reverse, unsigned int kway, bool is_using_gpu,
            const std::string &data_path, const std::string &ext) {
  T a1 = gpusort::Utils::ToT<T>(a1_str);
  T d = gpusort::Utils::ToT<T>(d_str);
  return DoWork(argc, argv, a1, d, algo, reverse, kway,
                is_using_gpu, data_path, ext);
}

template<typename T>
bool DoWork(int argc, char *argv[], T a1, T d, gpusort::Algorithm algo,
            bool reverse, unsigned int kway, bool is_using_gpu,
            const std::string &data_path, const std::string &ext) {
  gpusort::MpiUtils::MpiInit(&argc, &argv);
  gpusort::MpiComm comm = MPI_COMM_WORLD;
  int myrank = gpusort::MpiUtils::MpiCommRank(comm);
  int np = gpusort::MpiUtils::MpiCommSize(comm);
  if (myrank == 0) {
    std::cout << "Number of processes: " << np << std::endl;
  }

  // Init checker.
  gpusort::ParameterList *params = new gpusort::ParameterList();
  params->Add(new gpusort::Parameter<T>(PARAMETER_A1_NAME, a1));
  params->Add(new gpusort::Parameter<T>(PARAMETER_D_NAME, d));
  gpusort::ArithmeticProgressionChecker<T> checker(comm);
  checker.set_params(params);

  // Load raw data from a file.
  std::cout << "Rank " << myrank << " is loading input data..." << std::endl;
  std::vector<T> list;
  gpusort::Utils::ReadData(myrank, data_path, ext, &list);
  size_t size = list.size();
  if (size == 0) {
    std::cout << "No data to sort at rank " << myrank << "." << std::endl;
    gpusort::MpiUtils::MpiAbort(comm, 1);
  }
  std::cout << "Rank " << myrank << " finished loading data. "
            << "Data size = " << size << std::endl;

  // Sort data then check.
  if (!reverse) {
    SortWithThrustHostVector(comm, std::less<T>(), algo,
                             kway, is_using_gpu, &list);
  } else {
    SortWithThrustHostVector(comm, std::greater<T>(), algo,
                             kway, is_using_gpu, &list);
  }

  //
  // Validate the result
  //
  std::cout << "Rank: " << myrank << " is validating..." << std::endl;
  bool is_increasing = !reverse;
  bool suc = checker.Check(list, is_increasing);
  if (!suc) {
    std::cout << "Fail at Rank: " << myrank << std::endl;
  } else {
    std::cout << "Rank: " << myrank << " is completed!" << std::endl;
  }
  gpusort::MpiUtils::MpiFinalize();
  return suc;
}
