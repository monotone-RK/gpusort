#include <fstream>
#include <iostream>
#include <string>

#include "utils.h"

template<typename T>
bool DoWork(const std::string &file_path, uint64_t begin = 0, uint64_t end = 0);

int main(int argc, char *argv[]) {
  if (argc < 3 ) {
    std::cout << "Invalid parameters. Command format:  "
                 "dump_data <data_type> <file_path>" << std:: endl;
    return 1;
  }
  std::string data_type = std::string(argv[1]);
  std::string file_path = std::string(argv[2]);
  std::string begin_str = "";
  std::string end_str = "";
  if (argc > 3) begin_str = std::string(argv[3]);
  if (argc > 4) end_str = std::string(argv[4]);
  uint64_t begin = (begin_str.length() > 0)? gpusort::Utils::ToUInt64(begin_str) : 0;
  uint64_t end = (end_str.length() > 0)? gpusort::Utils::ToUInt64(end_str) : 0;
  bool suc = false;
  CALL_FUNC_T(suc, data_type, DoWork,
              (file_path, begin, end),
              {
                std::cout << gpusort::kErrorTypeMsg << std::endl;
                return 1;
              });

  if (!suc) return 1;
  return 0;
}

template<typename T>
bool DoWork(const std::string &f_path, uint64_t begin = 0, uint64_t end = 0) {
  std::ifstream in(f_path.c_str(), std::ifstream::in | std::ifstream::binary);
  if (!in.is_open()) return false;
  uint64_t count = -1;
  while (in.good()) {
    T tmp;
    in.read(reinterpret_cast<char*>(&tmp), sizeof(T));
    if (in.eof()) break;
    if (count >= begin && (end == 0 || count <= end)) std::cout << (long)tmp << " ";
    count++;
  }
  in.close();

  return true;
}
