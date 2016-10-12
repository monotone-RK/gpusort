#include <getopt.h>
#include <cstring>
#include <iostream>

#include "staggered_checker.h"
#include "utils.h"

template<typename T>
bool DoWork(int argc, char *argv[], int p, const std::string &capacity,
            const std::string &data_path, const std::string &ext);

template<typename T>
bool DoWork(int argc, char *argv[], gpusort::Algorithm algo,
            bool reverse, unsigned int kway, bool is_using_gpu,
            const std::string &data_path, const std::string &ext);

int main(int argc, char *argv[]) {
  bool gen_flag = false;
  bool is_using_gpu = true;
  bool reverse_flag = false;
  std::string g_str = "", k_str = "", p_str = "", algo_str = "hyk_sort";
  std::string n_threads_str = "";
  std::string t_str = "double", ext = gpusort::kDataFileExt;
  unsigned int kway = 2;
  int n_threads = 1;
  int p = 128;
  int c = -1;

  opterr = 0;
  while ((c = getopt(argc, argv, "a:de:g:k:n:p:rt:")) != -1)
    switch (c) {
      case 'a':
        algo_str = std::string(optarg);
        break;

      case 'd':
        gen_flag = true;
        break;

      case 'e':
        ext = std::string(optarg);
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

      case 'p':
        p_str = std::string(optarg);
        break;

      case 'r':
        reverse_flag = true;
        break;

      case 't':
        t_str = std::string(optarg);
        break;

      case '?':
        if (optopt == 'a' || optopt == 'e' || optopt == 'g' || optopt == 'k' ||
            optopt == 'p' || optopt == 'n' || optopt == 't') {
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
  if (gen_flag) {
    if (optind != argc - 2) {
      std::cout << gpusort::kErrorCapacityOrDataPathMsg << std::endl;
      return 1;
    }
    capacity = std::string(argv[optind++]);
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
    if (p_str.length() != 0) p = gpusort::Utils::ToInt(p_str);
    CALL_FUNC_T(suc, t_str, DoWork,
                (argc, argv, p, capacity, data_path, ext),
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
                (argc, argv, algo, reverse_flag, kway,
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
bool DoWork(int argc, char *argv[], int p, const std::string &capacity,
            const std::string &data_path, const std::string &ext) {
  gpusort::MpiUtils::MpiInit(&argc, &argv);
  gpusort::MpiComm comm = MPI_COMM_WORLD;
  int myrank = gpusort::MpiUtils::MpiCommRank(comm);
  gpusort::ParameterList *params = new gpusort::ParameterList();
  params->Add(new gpusort::Parameter<int>(PARAMETER_P_NAME, p));
  gpusort::StaggeredChecker<T> checker(comm);
  checker.set_params(params);

  bool suc = false;
  try {
    suc = checker.Generate(capacity, data_path, ext);
  } catch (gpusort::MpiException &e) {
    std::cout << "Exception at Rank " << myrank
              << ".\n\t Reason: " << e.what() << std::endl;
  }

  gpusort::MpiUtils::MpiFinalize();
  return suc;
}

template<typename T>
bool DoWork(int argc, char *argv[], gpusort::Algorithm algo,
            bool reverse, unsigned int kway, bool is_using_gpu,
            const std::string &data_path, const std::string &ext) {
  return gpusort::Utils::Sort<T, gpusort::StaggeredChecker>(
             argc, argv, algo, reverse, kway, is_using_gpu, data_path, ext);
}
