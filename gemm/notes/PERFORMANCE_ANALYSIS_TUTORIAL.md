# CUDA GEMM 性能分析实验册：从埋点、计数器到 PTX/SASS

## 0. 先建立正确心智模型

性能分析不是“打开 NCU 看哪个数字红”，而是一条证据链：

1. **形状是真的**：确认 `M/N/K` 来自实际推理执行路径。
2. **结果是对的**：先过 correctness，再谈快慢。
3. **时间是稳的**：warmup、多个独立样本、分位数、固定缓存口径。
4. **先定位大类**：计算、显存、缓存、occupancy、调度/依赖。
5. **再落到指令**：CUDA 源码 → PTX → SASS → 硬件计数器。
6. **最后只改一个假设**：例如“减少 B 的 DRAM 重载”，不要同时改 tile、向量宽度和 block 大小。

## 1. SGLang 中什么叫“真实 GEMM 形状”

以 Llama 3.1 8B、TP=1 为例：`H=4096`、`I=14336`、query heads=32、KV heads=8、head dim=128。SGLang 的 Llama 层使用融合 QKV、融合 gate+up，以及 row-parallel 的 o/down projection。

对线性层 `Y = XW`，SGLang 送入 kernel 的主要形状是：

| 层 | GEMM `(M,N,K)`，TP=1 |
|---|---|
| QKV projection | `(T, 6144, 4096)` |
| attention output projection | `(T, 4096, 4096)` |
| MLP gate+up | `(T, 28672, 4096)` |
| MLP down | `(T, 4096, 14336)` |

这里 `T` 不是抽象的固定 batch：

- extend/prefill：`T = Σ 每个请求本轮新增 token 数`；连续批处理可以是 ragged 的。
- decode：普通单 token decode 中 `T ≈ 活跃请求数`，但 CUDA Graph、DP 或实现策略可能 padding。
- TP>1：column-parallel 的 QKV/gate+up 主要切 `N`；row-parallel 的 o/down 主要切 `K`，并伴随通信。KV heads 少于 TP 数时还可能复制 KV heads，不能机械地除以 TP。

本实验选两点：

- prefill QKV：4 请求 × 128 新 token，`(512,6144,4096)`。
- decode output projection：16 活跃请求，`(16,4096,4096)`。

权威代码入口：

- [SGLang Llama：QKV、o、gate_up、down](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llama.py)
- [SGLang ForwardMode / ForwardBatch](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/forward_batch_info.py)
- [SGLang TP linear 定义](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/linear.py)
- [Meta Llama 3.1 模型参数注册](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py)

## 2. 四版代码应该观察什么

本实验的教学版映射是：

| 版本 | 文件 | 预期证据变化 |
|---|---|---|
| V0 | `v1_coalesced.cu` | 合并访存正确，但每个输出仍反复从 global 读取 A/B |
| V1 | `v2_smem_tiled.cu` | `LDG` 静态指令减少，出现 `LDS/STS/BAR`；数据复用进入 shared memory |
| V2 | `v4_2d_threadtile.cu` | 每线程多输出，独立 accumulator 和 `FFMA` 增多；寄存器压力上升 |
| V3 | `v7_warptiling.cu` | warp/block/thread tile 分层，寄存器从 128 降到 80，occupancy 回升 |

这里所有自写版本都是 FP32 CUDA Core 路径。默认 cuBLAS FP32 也是 `FFMA` 路径；只有显式 TF32 cuBLAS 出现 `HMMA.1688.F32.TF32`。

## 3. 三种“埋点”分别回答什么

### 3.1 CUDA Event：设备时间

项目 harness 的核心形式是：

```cpp
cudaEventRecord(start, stream);
launch(..., stream);
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&ms, start, stop);
```

同一 stream 上，event 时间戳包住 GPU 工作，排除矩阵生成、malloc 和 H2D/D2H。它适合 kernel 延迟/吞吐；它不告诉你为什么慢，也不等于包含 Python、调度器和网络通信的端到端时间。

### 3.2 NVTX：语义区间

`gemm_common.cuh` 已加入可选 RAII 埋点：

```cpp
{
  NvtxRange range("gemm.measure");
  // measured launches
}
```

当前四个区间是：

- `gemm.setup.alloc-copy`
- `gemm.warmup`
- `gemm.measure`
- `gemm.copyback-verify`

构建和采集：

```bash
make -j4 nvtx
bash scripts/profile_nsys.sh bin_nvtx/v7_warptiling 512 6144 4096
```

NVTX 本身不测 GPU 指令。它只是给时间线贴语义标签；只有 CUDA trace 成功时，NSYS 才能把 range 投影到 kernel/memcpy 轨道。

本机 WSL 的 Nsight Systems 2024.5 能看到 NVTX host ranges，但没有记录 CUDA kernel 轨道，因此 `profile/nsys/*.nsys-rep` 只能证明区间边界，不能作为 GPU kernel 时间证据。脚本已设置 WSL 非 root 所需的 `CUDA_INJECTION_SHM_ALLOWED=TRUE`；下一步应升级 Linux 版 Nsight Systems 后重试。NVIDIA 的 [Nsight Systems CUDA trace 文档](https://docs.nvidia.com/nsight-systems/UserGuide/) 和 [CUDA on WSL 工具支持表](https://docs.nvidia.com/cuda/wsl-user-guide/contents.html) 可用于核对版本/驱动条件。

### 3.3 NCU：硬件计数器

NCU 通过性能监控计数器回答：SM 有多忙、DRAM/L2/L1 吞吐多高、warp 为什么没发射、用了多少寄存器/shared memory、真正执行的是 FFMA 还是 HMMA。

一次运行通常无法同时读取所有计数器。NCU 会重放 kernel，多次采集不同 counter；kernel replay 还可能保存/恢复被写内存。因此：

- **NCU 报告里的 duration 用于同一次 profile 内关联，不替代正常 benchmark。**
- section 越多，采集墙钟时间越长，不代表 kernel 变慢。
- kernel 必须近似确定性，且过滤器要精确。

指标名可以按下面拆：

```text
sm__throughput.avg.pct_of_peak_sustained_elapsed
│    │          │   └─ 相对持续峰值，按整个 elapsed 周期归一化
│    │          └──── rollup/submetric
│    └─────────────── quantity
└──────────────────── hardware unit
```

`pct_of_peak_sustained_active` 只在单元活跃周期归一化，`...elapsed` 包含不活跃周期。不要把两者放进同一柱状图后直接比较。

## 4. 可复现采集命令

### 4.1 正确性和分布统计

项目提供了适配 benchmark skill ABI 的 `bench_skill/*_solve.cu`。示例：

```bash
/home/l1q/anaconda3/envs/liger/bin/python \
  /mnt/c/Users/20192/.codex/skills/kernel-benchmark/scripts/benchmark.py \
  bench_skill/v3_warp_solve.cu \
  --implementation=cuda-cpp \
  --baselines=none \
  --ref=bench_skill/gemm_ref.py \
  --output-dir=bench_skill/results/prefill_v3_warp \
  --timing-method=cuda_event \
  --num-warmup=5 --num-trials=15 --discard-first=1 --prewarm-calls=1 \
  --M=512 --N=6144 --K=4096
```

该 skill 每个 trial 前用约 256 MiB 写入冲掉 L2，所以这是 cold-L2 口径。普通 `bin/v7_warptiling ...` 的 20 次连发平均则是 warm/steady-state 口径。报告必须写清口径。

### 4.2 NCU 分阶段采集

先由 skill 生成规范 target：

```bash
python3 /mnt/c/Users/20192/.codex/skills/kernel-profiler-skill/scripts/generate_profile_target.py \
  --target-cmd "./bin/v7_warptiling 512 6144 4096 20 5 --no-verify" \
  --kernel gemm_v7_warptiling \
  --profile-id sglang_prefill_v3_warp \
  --output-root ./profile \
  --requirement "all NCU stages, source, roofline, visual report"
```

再精确过滤并跳过 5 次 warmup，采 3 次：

```bash
bash /mnt/c/Users/20192/.codex/skills/kernel-profiler-skill/scripts/ncu_collect_kernel_profile.sh \
  --target-cmd "./bin/v7_warptiling 512 6144 4096 20 5 --no-verify" \
  --kernel-name gemm_v7_warptiling \
  --kernel-regex ".*gemm_v7_warptiling.*" \
  --launch-skip 5 --launch-count 3 \
  --output-dir ./profile/sglang_prefill_v3_warp \
  --stages all --ncu-bin /usr/local/bin/ncu
```

日常优化不要每次 `all`。推荐证据树：

```text
basic / SpeedOfLight
├─ memory 高 → MemoryWorkloadAnalysis + SourceCounters
├─ SM 高 → ComputeWorkloadAnalysis + InstructionStats
├─ 两边低 → Occupancy + SchedulerStats + WarpStateStats
└─ 判断仍冲突 → roofline + source/SASS
```

## 5. 统计数字怎么读

`mean/median/p20/p80/min/max/std` 各自有不同用途：

- median：默认 headline，抗偶发桌面抢占。
- p20–p80：中间 60% 样本带，适合画误差线。
- mean：总吞吐估计，但会被长尾抬高。
- std：抖动程度；不要只报 `mean ± std` 而隐藏分位数。
- min：最理想下限，不代表常态。
- max：发现桌面合成、时钟、后台任务等长尾。

这台 GPU 同时驱动桌面，Windows 不能保证所有系统组件永远不碰独显。结果仍可用于学习和版本内比较，条件是：

1. 所有版本在同一电源模式、同一会话、相近温度下交错运行。
2. 关闭浏览器视频、录屏、游戏叠加层；观察任务管理器或 `nvidia-smi dmon`。
3. 使用 median 和分位带；差异小于约 2% 不下结论。
4. 对关键结论做 A-B-A 或随机顺序复测。
5. 报告 GPU、驱动、CUDA、功耗/时钟是否锁定，以及 GPU 是否承担显示。

本次主要差异在 20%–10×，远大于普通显示噪声；但精确到小数点后两位的排名不应外推到服务器卡。

## 6. NCU 的古法读图顺序

### 第一步：延迟图

画“median 点 + p20/p80 横线”，prefill/decode 分两个小图。先回答版本是否真的更快、抖动是否重叠。

本次关键现象：prefill 从 V0 到 V3 单调改善；decode 的 V1/V2 反而比 V0 慢。形状适配优先于“版本号更高”。

### 第二步：SOL 小多图

分别画 SM、memory、DRAM、L1/TEX、occupancy；不要堆叠，因为这些百分比不是一个总量的组成部分。

- V0 prefill：memory 92.7%、DRAM 228.6 GB/s、occupancy 99.5%。高 occupancy 没救下性能，因为 warp 大量等 global data。
- V2：128 regs/thread，theoretical occupancy 33.3%。这是寄存器换复用。
- V3：80 regs/thread、occupancy 46.3%、FMA pipe 45.4%，比 V2 更能持续喂 FMA。
- cuBLAS FP32：FMA pipe 63.3%、DRAM 25.9%，偏计算侧。
- cuBLAS TF32：HMMA pipe 45.5%，但 220 regs/thread、occupancy 16.5%；低 occupancy 不等于慢，Tensor Core 单 warp 工作量更大。

### 第三步：occupancy 资源图

理论 occupancy 由每 SM 的 block、warp、register、shared-memory 上限中最紧的一项决定；achieved occupancy 是实际活跃 warp。要问的是“少了哪些 warp，是否因此无法隐藏延迟”，不是盲目追 100%。

decode 中 V2 的 theoretical occupancy 是 33.3%，achieved 只有 7.4%，主因不是又多用了寄存器，而是 `M=16` 对 128-row tile 造成 grid underfill 和大量无效输出位置。

### 第四步：stall 表

`smsp__average_warps_issue_stalled_*_per_issue_active.ratio` 是按 issue-active 周期归一化的 stalled-warps 比率，不是互斥百分比，不能做饼图，也不要求相加为 100。

- `long_scoreboard`：常与 DRAM/L2 长延迟依赖有关。
- `short_scoreboard`：常与 L1/shared/短延迟依赖有关。
- `barrier`：同步等待。
- `not_selected`：有 eligible warp，但调度器选择了别的 warp；本身不一定是坏事。
- `math_pipe_throttle`：目标数学流水线供不应求。

先用最高的 2–3 个 stall 形成假设，再回源代码/SASS；不要按 stall 名字直接改代码。

### 第五步：roofline

手算公式：

```text
F = 2*M*N*K                         # algorithmic FLOPs
P = F / kernel_time                 # FLOP/s
DRAM_bytes = dram_bytes_per_second * kernel_time
AI = F / DRAM_bytes                 # FLOP/byte
roof(AI) = min(compute_peak, AI * memory_peak)
```

本机运行时数据给出 128-bit、约 8.001 GHz memory clock，对应约 256 GB/s 理论带宽。prefill memory-stage 的观测点：

| Variant | AI FLOP/B | measured TFLOP/s | 含义 |
|---|---:|---:|---|
| V0 | 3.88 | 0.88 | 靠近显存斜线，重复读取导致 6.64 GB DRAM traffic |
| V1 | 14.99 | 1.22 | DRAM traffic 降到 1.72 GB，但 barrier/CTA 结构让点远离 roof |
| V2 | 55.37 | 5.39 | 进入寄存器/issue 限制区 |
| V3 | 49.13 | 8.33 | AI 略低于 V2，但 FMA feed 更好，所以更快 |
| cuBLAS FP32 | 143.26 | 10.22 | 明显转向 scalar-FP32 compute roof |
| cuBLAS TF32 | 136.31 | 17.44 | 数值/计算 roof 已换成 Tensor Core，需单独标记 |

roofline 是分类器，不是成绩单。V1 虽在内存斜线下很远，真正下一步可能是 block/同步，而不是继续追 DRAM 带宽。

## 7. 从 CUDA 到 PTX/SASS 怎么对照

生成文件：

```bash
bash scripts/build_ir.sh kernels/v7_warptiling.cu sm_89
```

输出目录包含：

- `.ptx`：CUDA 虚拟 ISA，接近设备侧中间表示；还不是最终机器码。
- `.cubin`：目标 `sm_89` 的设备二进制。
- `.sass` / `.nvdisasm.txt`：GPU 实际执行 ISA。
- `keep/*`：NVCC 各阶段中间文件。

常见指令：

| 层次 | 指令/符号 | 看什么 |
|---|---|---|
| PTX | `ld.global`, `ld.shared`, `st.shared`, `fma.rn.f32`, `bar.sync` | 编译器仍可能继续合并、重排、展开 |
| SASS | `LDG`, `LDS`, `STS`, `FFMA`, `BAR.SYNC` | 最终 global/shared/FMA/同步结构 |
| Tensor Core SASS | `HMMA`, `IMMA`, `WGMMA` | 证明用了 tensor path；函数名写 tensor 不够 |

本次静态 SASS 从 V0 到 V3 的变化：

```text
V0: LDG 多、无 LDS/STS/BAR、FFMA 少
V1: LDG 静态点减少，出现 LDS/STS/BAR
V2: 大量展开的 FFMA，128 registers/thread
V3: 仍是 FFMA，但 warp tile 让寄存器降到 80，调度更健康
cuBLAS FP32: FFMA，没有 HMMA
cuBLAS TF32: HMMA.1688.F32.TF32
```

静态 opcode 次数不等于动态执行次数：循环体的一条 `LDG` 可能执行数千次，展开会把一条源码 FMA 复制成数百条 SASS。动态结论要结合 NCU 的 instruction/pipe metrics。

## 8. 用 Excel/LibreOffice 做“古法图形化”

输入文件：

- `reports/sglang_gemm_benchmark_summary.csv`
- `reports/sglang_gemm_ncu_summary.csv`

建议做四张图：

1. **延迟点图**：x=variant，y=median_ms，误差线下界=`median-p20`、上界=`p80-median`；prefill/decode 分面。
2. **利用率小多图**：分别对 `sm_pct`、`memory_pct`、`achieved_occ_pct` 作 0–100% 柱图。
3. **资源压力图**：regs/thread 和 shared-memory bytes 用两个纵轴或分开画；不要和百分比混在一个轴。
4. **roofline 散点**：x=AI（log 轴），y=TFLOP/s（log 轴）；加 `y=min(20.0,0.256*x)` 的 FP32 参考线，TF32 点用不同形状并注明不同精度。

读图时每张只回答一个问题：

```text
延迟：值不值得继续？
SOL：大方向在哪？
资源/occupancy：并行度被什么卡住？
stall：warp 为什么发不出去？
PTX/SASS：源码改动真的变成预期指令了吗？
roofline：瓶颈属于 bandwidth roof、compute roof，还是两边都没吃满？
```

## 9. 面向 MXFP8/NVFP4 项目的迁移

低精度项目可直接复用这套方法：

- NVTX range：`quantize`、`pack`、`scale-write`、`unpack`、`dequantize`、`error-reduce`。
- CUDA Event：每个 kernel 单独计时，不把文件 I/O、host 解析混入。
- 有效带宽：`(输入字节 + packed 字节 + scale 字节 + 输出字节) / kernel_time`，同时报告公式和计入范围。
- NCU memory：观察 4-bit packed load/store 是否合并、sector/request 比、L1/L2/DRAM throughput。
- SASS：确认 4-bit 路径没有被错误实现成一个 `uint8` 一个元素，检查位运算、向量 load/store 和寄存器膨胀。
- correctness：随机、正态、异常值三组；量化误差与 kernel 实现错误要分别验证。

最终目标不是让每个百分比都高，而是让“格式定义 → 内存布局 → 指令 → counter → 延迟/误差”能互相解释。
