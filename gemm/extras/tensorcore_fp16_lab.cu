#include "hgemm_common.cuh"
#include "tensorcore_fp16_kernels.cuh"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s naive|ada64|ada|cublas [M N K iters warmup "
                 "--no-verify]\n",
                 argv[0]);
    return 1;
  }
  const std::string version = argv[1];
  HgemmBenchConfig config = parse_hgemm_args(argc, argv, 2);
  if (version == "naive") {
    return run_hgemm_benchmark("hgemm_wmma_naive",
                               tensorcore_fp16::launch_naive, config);
  }
  if (version == "ada") {
    return run_hgemm_benchmark("hgemm_mma_ada_pipeline",
                               tensorcore_fp16::launch_ada, config);
  }
  if (version == "ada64") {
    return run_hgemm_benchmark("hgemm_mma_ada_pipeline_64x64",
                               tensorcore_fp16::launch_ada_64x64, config);
  }
  if (version == "cublas") {
    return run_hgemm_benchmark("cublas_hgemm_fp16_fp32",
                               tensorcore_fp16::launch_cublas, config);
  }
  std::fprintf(stderr, "unknown version: %s\n", version.c_str());
  return 1;
}
