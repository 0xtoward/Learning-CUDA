# CUDA Reduce Optimization Lab

An educational reduction ladder with one generic operator interface. The same
kernels support `sum`, `max`, and `min`; the benchmarked table uses FP32 sum.

```text
V0 interleaved shared tree
 -> V1 sequential addressing + two inputs/thread
 -> V2 warp shuffle + one shared value/warp
 -> V3 float4 + grid-stride + hierarchical reduction
 -> CUB DeviceReduce baseline
```

## Build and run

```bash
make ARCH=sm_89 -j
make check

./bin/reduce_benchmark v3 sum 16777216 100 20
./bin/reduce_benchmark v3 max 16777216 100 20
./bin/reduce_benchmark v3 min 16777216 100 20
bash scripts/run_all.sh
```

CLI: `VERSION OP N ITERATIONS WARMUP [--no-verify]`.

See `notes/REDUCE_OPT_TABLE.md` for the optimization table, measurements, and
newer-architecture extensions.

The controlled FP32-sum result at `N=16,777,216` is 0.3523 ms for V3 versus
0.3533 ms for CUB (median, cold-L2 protocol). The matching NCU capture is under
`profile/reduce_v3_sum/`; it reports 96.3% DRAM throughput and an
`LDG.E.128` vector load in SASS.
