#include "hgemm_common.cuh"
#include "tensorcore_fp16_cutlass.cuh"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s 128x128|128x64|64x128|64x128s2|64x128s4|64x64 "
                 "[M N K iters warmup --no-verify]\n",
                 argv[0]);
    return 1;
  }
  std::string shape = argv[1];
  HgemmBenchConfig config = parse_hgemm_args(argc, argv, 2);
  if (shape == "128x128") {
    return run_hgemm_benchmark("cutlass_hgemm_128x128x32",
                               tensorcore_fp16_cutlass::launch_128x128, config);
  }
  if (shape == "128x64") {
    return run_hgemm_benchmark("cutlass_hgemm_128x64x32",
                               tensorcore_fp16_cutlass::launch_128x64, config);
  }
  if (shape == "64x128") {
    return run_hgemm_benchmark("cutlass_hgemm_64x128x32",
                               tensorcore_fp16_cutlass::launch_64x128, config);
  }
  if (shape == "64x128s2") {
    return run_hgemm_benchmark(
        "cutlass_hgemm_64x128x32_stages2",
        tensorcore_fp16_cutlass::launch_64x128_stages2, config);
  }
  if (shape == "64x128s4") {
    return run_hgemm_benchmark(
        "cutlass_hgemm_64x128x32_stages4",
        tensorcore_fp16_cutlass::launch_64x128_stages4, config);
  }
  if (shape == "64x64") {
    return run_hgemm_benchmark("cutlass_hgemm_64x64x32",
                               tensorcore_fp16_cutlass::launch_64x64, config);
  }
  std::fprintf(stderr, "unknown CUTLASS shape: %s\n", shape.c_str());
  return 1;
}
