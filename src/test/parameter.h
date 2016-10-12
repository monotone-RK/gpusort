#ifndef GPUSORT_TEST_PARAMETER_H_
#define GPUSORT_TEST_PARAMETER_H_

#include <map>
#include <string>

namespace gpusort {

class ParameterBase {
 public:
  ParameterBase(const std::string &name) {
    name_ = name;
  }

  virtual ~ParameterBase() {
  }

  std::string Name() {
    return name_;
  }

 protected:
  std::string name_;
};

template<typename T>
class Parameter : public ParameterBase {
 public:
  Parameter(const std::string &name, T value) : ParameterBase(name) {
    value_ = value;
  }

  ~Parameter() {
  }

  T& Value() {
    return value_;
  }

 private:
  T value_;
};

class ParameterList {
 public:
  ParameterList() {
  }

  ~ParameterList() {
    Clear();
  }

  ParameterBase* operator[](const std::string &name) {
    return this->Get(name);
  }

  void Add(ParameterBase *param) {
    list_[param->Name()] = param;
  }

  void Clear() {
    for (std::map<std::string, ParameterBase*>::iterator it = list_.begin();
         it != list_.end(); it++) {
      delete it->second;
    }
  }

  ParameterBase* Get(const std::string &name) {
    std::map<std::string, ParameterBase*>::iterator it = list_.find(name);
    if (it == list_.end()) return NULL;
    return it->second;
  }

 private:
  std::map<std::string, ParameterBase*> list_;
};

}  // namespace gpusort

#endif  // GPUSORT_TEST_PARAMETER_H_
