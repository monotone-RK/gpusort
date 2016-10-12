#ifndef GPUSORT_TEST_SERIALIZABLE_H_
#define GPUSORT_TEST_SERIALIZABLE_H_

namespace gpusort {

class IFStream;
class OFStream;

class Serializable {
 public:
  virtual void Read(IFStream *s) = 0;

  virtual void Write(OFStream *s) = 0;
};

}  // namespace gpusort

#endif  // GPUSORT_TEST_SERIALIZABLE_H_
