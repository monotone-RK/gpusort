#ifndef GPUSORT_TIMER_H_
#define GPUSORT_TIMER_H_

namespace gpusort {

class Timer {
 public:
  Timer();

  ~Timer();

  void Reset();

  double Seconds();

  void Start();

  void Stop();

 private:
  double seconds_;
  double ref_;
};

}  // namespace gpusort

#endif  // GPUSORT_TIMER_H_
