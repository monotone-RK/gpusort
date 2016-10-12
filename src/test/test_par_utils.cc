#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>

#include "omp/omp_utils.h"
#include "par/par_utils.h"

#define GPU_SORT_HOME "/path/to/tmp"

static int64_t CalcNumElemsEachProcess(int64_t total, int np) {
  int r = total % np;
  if (r != 0) {
	  return (total + (np - r)) / np;
  } else {
	  return total / np;
  }
}

template<typename T>
static std::string ToString(T value) {
  return static_cast<std::ostringstream*>(
      &(std::ostringstream() << value))->str();
}

static std::string GetFilePath(const char* parent_path, int i,
                               const char* ext = ".dat") {
  std::string path = std::string(parent_path);
  path += std::string("/");
  path += ToString(i) + std::string(ext);
  return path;
}

static void GenDataInt(int np, const char* parent_path, int max) {
  std::vector<std::pair<int, int> > list;
  for (int i = 0; i < max; i++) {
    list.push_back(std::make_pair<int, int>(i, max-i));
  }
  std::random_shuffle(list.begin(), list.end());

  for (int i = 0; i < np; i++) {
    std::string path = GetFilePath(parent_path, i);
    std::ofstream out(
        path.c_str(),
        std::ofstream::out | std::ofstream::app | std::ofstream::binary);
    int nelem = CalcNumElemsEachProcess(max, np);
    int start_idx = nelem * i;
    int end_idx = start_idx + nelem - 1;
    for (int j = start_idx; j <= end_idx; j++) {
      out.write(reinterpret_cast<char*>(&list[j].first), sizeof(int));
      out.write(reinterpret_cast<char*>(&list[j].second), sizeof(int));
    }
    out.close();
  }
}

static void ReadDataInt(int myrank, const char* parent_path,
                        std::vector<std::pair<int, int> > *_list) {
  std::vector<std::pair<int, int> > &list = *_list;
  std::string path = GetFilePath(parent_path, myrank);
  std::ifstream in(path.c_str(), std::ifstream::in | std::ifstream::binary);
  if (!in.is_open()) return;
  while (in.good()) {
    int first = 0, second = 0;
    in.read(reinterpret_cast<char*>(&first), sizeof(int));
    if (in.eof()) break;
    in.read(reinterpret_cast<char*>(&second), sizeof(int));
    list.push_back(std::make_pair<int, int>(first, second));
  }
  in.close();
}

static std::string GetDataPath() {
  std::string path = std::string(GPU_SORT_HOME);
  return path;
}

void TestArrangeValues(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  // Find out my identity in the default communicator
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  // Find out number of processes
  int np;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  if (myrank == 0) {
    std::cout << "Number of processes: " << np << std::endl;
    std::cout << "Threads max: " << omp_get_max_threads() << std::endl;
  }

  std::vector<std::pair<int, int> > list;
  std::string path = GetDataPath();
  ReadDataInt(myrank, path.c_str(), &list);

  int64_t size = list.size();
  std::vector<int64_t> keys(size);
  std::vector<int64_t> values(size);
  int sum = list[0].first + list[0].second;
  int nelem = CalcNumElemsEachProcess(sum, np);
  for (int64_t i = 0; i < size; i++) {
    keys[i] = list[i].first;
    values[i] = sum - myrank * nelem - i;
  }
  std::vector<int64_t> old_sizes;
  gpusort::ParUtils::GatherNElems(MPI_COMM_WORLD, size, &old_sizes);
  // Re-arrange values base on global indexes
  gpusort::ParUtils::ArrangeValues(MPI_COMM_WORLD, keys, old_sizes, &values);
  // Validate the result
  int i = 0;
  std::cout << "Rank: " << myrank << " is validating..." << std::endl;
  for (; i < size; i++) {
    if (values[i] != list[i].second) break;
  }
  if (i < size) {
    std::cout << "Validating fail at position: " << i << std::endl;
    std::cout << "first: " << list[i].first << " second: " << list[i].second
        << " value: " << values[i] << std::endl;
  }
  std::cout << "Rank: " << myrank << " is completed!" << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}

int main(int argc, char *argv[]) {
  if (argc > 2) {
    std::string path = GetDataPath();
    int np = atoi(argv[1]);
    int max = atoi(argv[2]);
    GenDataInt(np, path.c_str(), max);
    return 0;
  }

  TestArrangeValues(argc, argv);
  return 0;
}

