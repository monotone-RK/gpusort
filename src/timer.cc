#include "timer.h"

#include <omp.h>

namespace gpusort {

Timer::Timer() {
  seconds_ = 0.0;
  ref_ = 0.0;  // OpenMP wall time.
}

Timer::~Timer() {
}

void Timer::Reset() {
  seconds_ = 0.0;
  ref_ = 0.0;
}

double Timer::Seconds() {
  return seconds_;
}

void Timer::Start() {
  ref_ = omp_get_wtime();
}

void Timer::Stop() {
  seconds_ -= ref_;
  ref_ = omp_get_wtime();
  seconds_ += ref_;
}

}  // namespace gpusort
