#include "reduce_bench_common.cuh"

extern "C" void solve(const float* x, float* out, int n) {
  auto& ws = reduce_bench::workspace(n);
  reduce_lab::launch_v3<reduce_lab::SumOp>(x, out, ws.buffer_a, ws.buffer_b, n,
                                           ws.max_blocks);
}
