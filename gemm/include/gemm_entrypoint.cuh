#pragma once
#include "gemm_common.cuh"
#ifndef KERNEL_NAME
#define KERNEL_NAME "unnamed"
#endif
int main(int argc, char** argv) {
  return run_benchmark(KERNEL_NAME, launch_gemm, argc, argv);
}
