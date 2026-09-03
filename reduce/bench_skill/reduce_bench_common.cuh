#pragma once

#include "../include/reduce_kernels.cuh"

namespace reduce_bench {

inline reduce_lab::BenchmarkWorkspace& workspace(int n) {
  static reduce_lab::BenchmarkWorkspace value;
  value.ensure(n);
  return value;
}

}  // namespace reduce_bench
