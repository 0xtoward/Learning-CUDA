NVCC ?= nvcc
ARCH ?= sm_89
CXXSTD ?= c++17
NVCCFLAGS := -O3 -std=$(CXXSTD) -lineinfo -arch=$(ARCH) -Xptxas=-v -Iinclude
ifeq ($(FAST),1)
NVCCFLAGS += --use_fast_math
endif

KERNELS := v0_naive_bad_mapping v1_coalesced v2_smem_tiled v3_1d_threadtile v4_2d_threadtile v5_2d_xor_swizzle v6_2d_vectorized v7_warptiling
BINS := $(addprefix bin/,$(KERNELS))
NVTX_BINS := $(addprefix bin_nvtx/,$(KERNELS))

all: $(BINS) bin/bank_conflict_demo bin/cublas_baseline bin/cublas_tf32_baseline

nvtx: $(NVTX_BINS) bin_nvtx/cublas_baseline bin_nvtx/cublas_tf32_baseline

bin/%: kernels/%.cu include/gemm_common.cuh include/gemm_entrypoint.cuh
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $< -o $@

bin_nvtx/%: kernels/%.cu include/gemm_common.cuh include/gemm_entrypoint.cuh
	@mkdir -p bin_nvtx
	$(NVCC) $(NVCCFLAGS) -DUSE_NVTX $< -lnvToolsExt -o $@

bin/bank_conflict_demo: extras/bank_conflict_demo.cu
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $< -o $@

bin/cublas_baseline: extras/cublas_baseline.cu include/gemm_common.cuh include/gemm_entrypoint.cuh
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $< -lcublas -o $@

bin/cublas_tf32_baseline: extras/cublas_tf32_baseline.cu include/gemm_common.cuh include/gemm_entrypoint.cuh
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $< -lcublas -o $@

bin_nvtx/cublas_baseline: extras/cublas_baseline.cu include/gemm_common.cuh include/gemm_entrypoint.cuh
	@mkdir -p bin_nvtx
	$(NVCC) $(NVCCFLAGS) -DUSE_NVTX $< -lcublas -lnvToolsExt -o $@

bin_nvtx/cublas_tf32_baseline: extras/cublas_tf32_baseline.cu include/gemm_common.cuh include/gemm_entrypoint.cuh
	@mkdir -p bin_nvtx
	$(NVCC) $(NVCCFLAGS) -DUSE_NVTX $< -lcublas -lnvToolsExt -o $@

clean:
	rm -rf bin/* build/* reports/*

.PHONY: all nvtx clean
