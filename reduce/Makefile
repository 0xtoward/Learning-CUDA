NVCC ?= /usr/local/cuda/bin/nvcc
ARCH ?= sm_89
NVCCFLAGS := -O3 -std=c++17 -lineinfo -arch=$(ARCH) -Xptxas=-v

WRAPPERS := v0_sum_solve v1_sum_solve v2_sum_solve v3_sum_solve cub_sum_solve
SHARED_LIBS := $(addprefix bench_skill/,$(addsuffix .so,$(WRAPPERS)))

all: bin/reduce_benchmark $(SHARED_LIBS)

bin/reduce_benchmark: reduce_benchmark.cu include/reduce_kernels.cuh
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $< -o $@

bench_skill/%.so: bench_skill/%.cu bench_skill/reduce_bench_common.cuh include/reduce_kernels.cuh
	$(NVCC) $(NVCCFLAGS) -shared -Xcompiler -fPIC $< -o $@

check: all
	./bin/reduce_benchmark v0 sum 1048579 2 1
	./bin/reduce_benchmark v1 max 1048579 2 1
	./bin/reduce_benchmark v2 min 1048579 2 1
	./bin/reduce_benchmark v3 sum 1048579 2 1
	./bin/reduce_benchmark cub sum 1048579 2 1

clean:
	rm -rf bin bench_skill/*.so bench_skill/results profile

.PHONY: all check clean
