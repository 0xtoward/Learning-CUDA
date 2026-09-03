NVCC ?= nvcc
ARCH ?= sm_89
CXXSTD ?= c++17
CUTLASS_DIR ?= /home/l1q/WSL/CUTLASS
NVCCFLAGS := -O3 -std=$(CXXSTD) -lineinfo -arch=$(ARCH) -Xptxas=-v -Iinclude
ifeq ($(FAST),1)
NVCCFLAGS += --use_fast_math
endif

KERNELS := v0_naive_bad_mapping v1_coalesced v2_smem_tiled v3_1d_threadtile v4_2d_threadtile v5_2d_xor_swizzle v6_2d_vectorized v7_warptiling
BINS := $(addprefix bin/,$(KERNELS))
NVTX_BINS := $(addprefix bin_nvtx/,$(KERNELS))

all: $(BINS) bin/bank_conflict_demo bin/v3_cp_async_double_buffered bin/wmma_tf32_gemm bin/wmma_tf32_block_tiled bin/tensorcore_fp16_lab bin/tensorcore_fp16_cutlass_lab bin/cublas_baseline bin/cublas_tf32_baseline

teaching: bin/cutlass_minimal bin/wmma_minimal bin/cuda_core_gemm_annotated bin/cuda_core_gemm_annotated_ptx

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

bin/v3_cp_async_double_buffered: extras/v3_cp_async_double_buffered.cu include/gemm_common.cuh include/gemm_entrypoint.cuh
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $< -o $@

bin/wmma_tf32_gemm: extras/wmma_tf32_gemm.cu include/gemm_common.cuh include/gemm_entrypoint.cuh
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $< -o $@

bin/wmma_tf32_block_tiled: extras/wmma_tf32_block_tiled.cu include/gemm_common.cuh include/gemm_entrypoint.cuh
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $< -o $@

bin/tensorcore_fp16_lab: extras/tensorcore_fp16_lab.cu extras/tensorcore_fp16_kernels.cuh include/hgemm_common.cuh
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $< -Iextras -lcublas -o $@

bin/tensorcore_fp16_cutlass_lab: extras/tensorcore_fp16_cutlass_lab.cu extras/tensorcore_fp16_cutlass.cuh include/hgemm_common.cuh
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $< -Iextras -I$(CUTLASS_DIR)/include -o $@

bin/cutlass_minimal: examples/cutlass_minimal.cu include/gemm_common.cuh include/gemm_entrypoint.cuh
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $< -I$(CUTLASS_DIR)/include -o $@

bin/wmma_minimal: examples/wmma_minimal.cu include/hgemm_common.cuh
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $< -o $@

bin/cuda_core_gemm_annotated: examples/cuda_core_gemm_annotated.cu include/gemm_common.cuh include/gemm_entrypoint.cuh
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $< -o $@

bin/cuda_core_gemm_annotated_ptx: examples/cuda_core_gemm_annotated.cu include/gemm_common.cuh include/gemm_entrypoint.cuh
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) -DGEMM_DEMO_INLINE_PTX_FMA $< -o $@

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

.PHONY: all teaching nvtx clean
