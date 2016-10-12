#include <getopt.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "arithmetic_progression_checker.h"
#include "increasing_sequence_checker.h"
#include "ln_checker.h"
#include "log_checker.h"

#include "ifstream.h"
#include "ofstream.h"
#include "serializable.h"
#include "utils.h"

#ifdef __DO_BENCHMARK
gpusort::Timer load_data_tm;
#endif

// Example of an User-defined class.
// This class extends Serializable interface for using Disk IO.
class SampleClass : public gpusort::Serializable {
 public:
  SampleClass() {
  }

  SampleClass(int64_t key, double v1, float v2,
              unsigned long v3, float v4, double v5) {
    key_ = key;
    v1_ = v1;
    v2_ = v2;
    v3_ = v3;
    v4_ = v4;
    v5_ = v5;
  }

  //
  // Implement comparison operators are required.
  //
  bool operator == (SampleClass const &other) const {
    return this->key_ == other.key_;
  }

  bool operator < (SampleClass const &other) const {
    return this->key_ < other.key_;
  }

  bool operator <= (SampleClass const &other) const {
    return this->key_ <= other.key_;
  }

  bool operator > (SampleClass const &other) const {
    return this->key_ > other.key_;
  }

  bool operator >= (SampleClass const &other) const {
    return this->key_ >= other.key_;
  }

  //
  // Getters and Setters.
  //
  int64_t key() {
    return key_;
  }

  void set_key(int64_t key) {
    key_ = key;
  }

  double v1() {
    return v1_;
  }

  void set_v1(double v1) {
    v1_ = v1;
  }

  float v2() {
    return v2_;
  }

  void set_v2(float v2) {
    v2_ = v2;
  }

  unsigned long v3() {
    return v3_;
  }

  void set_v3(unsigned long v3) {
    v3_ = v3;
  }

  float v4() {
    return v4_;
  }

  void set_v4(float v4) {
    v4_ = v4;
  }

  double v5() {
    return v5_;
  }

  void set_v5(double v5) {
    v5_ = v5;
  }

  //
  // Implement Serializable API.
  //
  void Read(gpusort::IFStream *s) {
    // Read key.
    s->Read(&key_);
    // Read values.
    s->Read(&v1_);
    s->Read(&v2_);
    s->Read(&v3_);
    s->Read(&v4_);
    s->Read(&v5_);
  }

  void Write(gpusort::OFStream *s) {
    // Write key.
    s->Write(key_);
    // Write values.
    s->Write(v1_);
    s->Write(v2_);
    s->Write(v3_);
    s->Write(v4_);
    s->Write(v5_);
  }

 private:
  int64_t key_;
  double v1_;
  float v2_;
  unsigned long v3_;
  float v4_;
  double v5_;
};

// Example of a function to get key value from an object of User-defined class.
int64_t GetKey(const SampleClass &obj) {
  return const_cast<SampleClass&>(obj).key();
}

//
// Declare internal functions.
//
bool DoWork(int np, const std::string &capacity,
            const std::string &data_path, const std::string &ext);

bool DoWork(int argc, char *argv[], gpusort::Algorithm algo,
            bool reverse, unsigned int kway, bool is_using_gpu,
            const std::string &data_path, const std::string &ext);

void InitCheckers(gpusort::MpiComm comm = NULL);

template<typename StrictWeakOrdering>
bool SortAndCheck(gpusort::MpiComm comm, StrictWeakOrdering comp,
                  gpusort::Algorithm algo, unsigned int kway,
                  bool is_using_gpu, std::vector<SampleClass> *_list);

//
// Declare global variables.
//
gpusort::ArithmeticProgressionChecker<int64_t> key_checker;
gpusort::ArithmeticProgressionChecker<unsigned long> v3_checker;
gpusort::IncreasingSequenceChecker<float> v2_checker;
gpusort::IncreasingSequenceChecker<float> v4_checker;
gpusort::LnChecker<double> v1_checker;
gpusort::LogChecker<double> v5_checker;

// Main function.
int main(int argc, char *argv[]) {
  bool gen_flag = false;
  bool is_using_gpu = true;
  bool reverse_flag = false;
  std::string g_str = "", k_str = "", algo_str = "hyk_sort";
  std::string n_threads_str = "";
  std::string ext = gpusort::kDataFileExt;
  unsigned int kway = 2;
  int n_threads = 1;
  int c = -1;

  opterr = 0;
  while ((c = getopt(argc, argv, "a:de:g:k:n:r")) != -1)
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

      case 'r':
        reverse_flag = true;
        break;

      case '?':
        if (optopt == 'a' || optopt == 'e' || optopt == 'g' ||
            optopt == 'k' || optopt == 'n') {
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

  bool suc = true;
  if (gen_flag) {
    suc = DoWork(np, capacity, data_path, ext);
  } else {
    if (g_str.length() != 0) is_using_gpu = gpusort::Utils::ToBool(g_str);
    if (k_str.length() != 0) kway = gpusort::Utils::ToUInt(k_str);
    if (kway == 0) kway = 2;
    gpusort::Algorithm algo = gpusort::Utils::ToAlgorithm(algo_str);
    if (algo == gpusort::UNKNOWN_ALGORITHM) {
      std::cout << gpusort::kErrorAlgorithmMsg << std::endl;
      return 1;
    }
    suc = DoWork(argc, argv, algo, reverse_flag, kway, is_using_gpu, data_path, ext);
  }

  return (suc)? 0 : 1;
}

bool DoWork(int np, const std::string &capacity,
            const std::string &data_path, const std::string &ext) {
  int n_threads = gpusort::OmpUtils::GetMaxThreads();
  int64_t nelem = gpusort::Utils::CapacityToNElems<SampleClass>(capacity);
  std::vector<SampleClass> list(nelem);

  std::vector<int64_t> keys;
  std::vector<unsigned long> v3_list;
  std::vector<float> v2_list;
  std::vector<float> v4_list;
  std::vector<double> v1_list;
  std::vector<double> v5_list;

  InitCheckers();

  // Generate data.
  bool suc = true;
  FOR_PARALLEL_COND(n_threads, 6, i, suc,
                    {
                      if (i == 0) {
                        suc = key_checker.Generate(nelem, &keys);
                      } else if (i == 1) {
                        suc = v3_checker.Generate(nelem, &v3_list);
                      } else if (i == 2) {
                        suc = v2_checker.Generate(nelem, &v2_list);
                      } else if (i == 3) {
                        suc = v4_checker.Generate(nelem, &v4_list);
                      } else if (i == 4) {
                        suc = v1_checker.Generate(nelem, &v1_list);
                      } else if (i == 5) {
                        suc = v5_checker.Generate(nelem, &v5_list);
                      }
                    });

  if (!suc) return false;
  // Assign generated data to main container.
  FOR_PARALLEL(n_threads, nelem, i,
               {
                 list[i] = SampleClass(keys[i], v1_list[i], v2_list[i],
                                       v3_list[i], v4_list[i], v5_list[i]);
               });
  // Randomly shuffle elements in the container.
  gpusort::Random r;
  r.Shuffle(&list[0], &list[nelem]);
  // Save data in the container to np files.
  gpusort::Utils::SaveData(np, data_path, ext, &list[0], &list[nelem]);
  std::cout << "Data is saved successfully to " << data_path << std::endl;
  return suc;
}

bool DoWork(int argc, char *argv[], gpusort::Algorithm algo,
            bool reverse, unsigned int kway, bool is_using_gpu,
            const std::string &data_path, const std::string &ext) {
  gpusort::MpiUtils::MpiInit(&argc, &argv);
  gpusort::MpiComm comm = MPI_COMM_WORLD;
  int myrank = gpusort::MpiUtils::MpiCommRank(comm);
  int np = gpusort::MpiUtils::MpiCommSize(comm);
  if (myrank == 0) {
    std::cout << "Number of processes: " << np << std::endl;
  }

  // Load raw data from a file.
  std::cout << "Rank " << myrank << " is loading input data..." << std::endl;
  std::vector<SampleClass> list;
#ifdef __DO_BENCHMARK
  load_data_tm.Start();
#endif
  gpusort::Utils::ReadData(myrank, data_path, ext, &list);
#ifdef __DO_BENCHMARK
  load_data_tm.Stop();
#endif
  size_t size = list.size();
  if (size == 0) {
    std::cout << "No data to sort at rank " << myrank << "." << std::endl;
    gpusort::MpiUtils::MpiAbort(comm, 1);
  }
  std::cout << "Rank " << myrank << " finished loading data. "
            << "Data size = " << size << std::endl;

  // Sort data then check.
  bool suc = true;
  if (!reverse) {
    suc = SortAndCheck(comm, std::less<SampleClass>(),
                       algo, kway, is_using_gpu, &list);
  } else {
    suc = SortAndCheck(comm, std::greater<SampleClass>(),
                       algo, kway, is_using_gpu, &list);
  }

  gpusort::MpiUtils::MpiFinalize();
  return suc;
}

void InitCheckers(gpusort::MpiComm comm) {
  //
  // Set communicator to all checkers.
  //
  if (comm != NULL) {
    key_checker.set_comm(comm);
    v1_checker.set_comm(comm);
    v2_checker.set_comm(comm);
    v3_checker.set_comm(comm);
    v4_checker.set_comm(comm);
    v5_checker.set_comm(comm);
  }
  //
  // Create parameters for some checkers.
  //
  gpusort::ParameterList *params_v3 = new gpusort::ParameterList();
  params_v3->Add(new gpusort::Parameter<unsigned long>(PARAMETER_A1_NAME, 2));
  params_v3->Add(new gpusort::Parameter<unsigned long>(PARAMETER_D_NAME, 2));
  params_v3->Add(new gpusort::Parameter<bool>(PARAMETER_SHUFFLE_NAME, false));
  v3_checker.set_params(params_v3);

  gpusort::ParameterList *params_v2 = new gpusort::ParameterList();
  params_v2->Add(new gpusort::Parameter<float>(PARAMETER_F1_NAME, 3.0));
  params_v2->Add(new gpusort::Parameter<float>(PARAMETER_RMAX_NAME, 7.0));
  v2_checker.set_params(params_v2);

  gpusort::ParameterList *params_v1 = new gpusort::ParameterList();
  params_v1->Add(new gpusort::Parameter<int64_t>(PARAMETER_BEGIN_NAME, 6));
  params_v1->Add(new gpusort::Parameter<bool>(PARAMETER_SHUFFLE_NAME, false));
  v1_checker.set_params(params_v1);
}

template<typename StrictWeakOrdering>
bool SortAndCheck(gpusort::MpiComm comm, StrictWeakOrdering comp,
                  gpusort::Algorithm algo, unsigned int kway,
                  bool is_using_gpu, std::vector<SampleClass> *_list) {
  std::vector<SampleClass> &list = *_list;
  int myrank = gpusort::MpiUtils::MpiCommRank(comm);
  gpusort::MpiUtils::MpiBarrier(comm);

  switch (algo) {
    // Using HykSort to sort input data.
    case gpusort::HYK_SORT_ALGORITHM:
      gpusort::HykSort<int64_t>(comm, comp, kway, is_using_gpu, &GetKey, &list);
      break;

    // Using HyperQuickSort to sort input data.
    case gpusort::HYPER_QUICK_SORT_ALGORITHM:
      gpusort::HyperQuickSort(comm, comp, &list);
      break;

    // Using SampleSort to sort input data.
    default:
      gpusort::SampleSort(comm, comp, &list);
      break;
  }
#ifdef __DO_BENCHMARK
  gpusort::ParUtils::ShowBenchmark(comm, load_data_tm, "Loading data");
#endif

  int64_t size = list.size();
  std::cout << "Rank " << myrank << " finished sorting. New data size = "
            << size << std::endl;
  //
  // Validate the result
  //
  InitCheckers(comm);

  std::cout << "Rank: " << myrank << " is validating..." << std::endl;
  bool is_increasing = gpusort::IsIncreasing(comp);
  int n_threads = gpusort::OmpUtils::GetMaxThreads();
  //
  // Assign data to temporary lists for checking.
  //
  std::vector<int64_t> keys(size);
  std::vector<unsigned long> v3_list(size);
  std::vector<float> v2_list(size);
  std::vector<float> v4_list(size);
  std::vector<double> v1_list(size);
  std::vector<double> v5_list(size);
  FOR_PARALLEL(n_threads, size, i,
               {
                 keys[i] = list[i].key();
                 v1_list[i] = list[i].v1();
                 v2_list[i] = list[i].v2();
                 v3_list[i] = list[i].v3();
                 v4_list[i] = list[i].v4();
                 v5_list[i] = list[i].v5();
               });

  // Check all vectors.
  int num_v = 6;
  bool *states = new bool[num_v];
  try {
    states[0] = key_checker.Check(keys, is_increasing);
    states[1] = v1_checker.Check(v1_list, is_increasing);
    states[2] = v2_checker.Check(v2_list, is_increasing);
    states[3] = v3_checker.Check(v3_list, is_increasing);
    states[4] = v4_checker.Check(v4_list, is_increasing);
    states[5] = v5_checker.Check(v5_list, is_increasing);
  } catch (gpusort::MpiException &e) {
    delete[] states;
    std::cout << "Exception at Rank " << myrank
              << ".\n\t Reason: " << e.what() << std::endl;
    throw e;
  }

  bool suc = true;
  for (int i = 0; i < num_v; i++)
    if (!states[i]) {
      suc = false;
      break;
    }

  if (!suc) {
    std::cout << "Fail at Rank: " << myrank << std::endl;
  } else {
    std::cout << "Rank: " << myrank << " is completed!" << std::endl;
  }
  delete[] states;
  return suc;
}
