# CUDA Kernel 优化三表与中文面试提纲

这份文件用于面试复习，不从仓库 README 索引。所有性能数字都必须和同一行的
GPU、shape、dtype、缓存策略与计时方法一起陈述，不能跨口径计算 speedup。

## 统一 benchmark 口径

- GPU：RTX 4070 Laptop（Ada，SM89）。
- GEMM prefill shape：`M=512, N=6144, K=4096`。
- CUDA Core GEMM：FP32；Tensor Core GEMM：FP16 A/B、FP32 accumulate/C。
- Reduce：FP32 sum，`N=16,777,216`。
- 正式性能：CUDA event、warmup 后取多次 median、每次计时前 cold L2。
- NCU：用于解释瓶颈，不把 replay 后的单次 duration 当最终 benchmark。
- 正确性：每版先和 CPU/cuBLAS/CUB reference 对比，再讨论速度。

## 表一：Reduce

| 阶段 | 它是什么 / 在解决什么 | 数据路径与核心代码形态 | NCU/代码里会看到什么 | 怎么修、为什么有效 | 本机结果 | 更新架构还可加什么 |
|---|---|---|---|---|---:|---|
| R0 Interleaved | 正确性基线；线程按交错索引做树归约 | GMEM -> SMEM；`index = 2*s*tid`，每轮 barrier | modulo/地址计算多，活跃线程越来越少，分支与同步浪费 | 先保留它讲清多 CTA partial + 第二次归约 | 0.7311 ms / 91.79 GB/s | 原理跨架构不变 |
| R1 Sequential | 去掉交错寻址，并让每线程先吃两个输入 | 两次 GMEM load -> register sum -> 连续 SMEM tree | 分支/索引开销下降；DRAM 利用率明显升高 | 连续线程处理连续元素，第一步就把 block 工作量减半 | 0.3553 ms / 188.88 GB/s | 更宽、对齐的 vector load |
| R2 Warp Shuffle | warp 内不再经过 SMEM 与 `__syncthreads()` | block 前半仍在 SMEM；最后 32 个线程用 `__shfl_down_sync` | barrier 数和 shared traffic 下降；若已 DRAM-bound，时间改善会很小 | warp shuffle 直接交换 lane 寄存器；只写每个 warp 的 partial | 0.3543 ms / 189.42 GB/s | cooperative groups 改善可读性 |
| R3 Vectorized/Hierarchical | 减少 load 指令，并限制 partial 数 | `float4` + grid-stride loop -> warp reduce -> CTA partial -> final kernel | NCU：DRAM 96.3%，34 regs/thread，约 91% occupancy，无 spill；已接近带宽屋顶 | 向量化减少指令，grid-stride 增大每 CTA 工作量，层次归约降低中间流量 | 0.3523 ms / 190.49 GB/s | persistent/cooperative/cluster；实践中常优先做算子融合 |
| CUB | 生产级基线与架构特化 dispatch | `cub::DeviceReduce` | 和 R3 几乎相同表示自写版已触顶，不该继续堆技巧 | 用它判断“优化是否结束”，而不是只和 R0 比 | 0.3533 ms / 189.95 GB/s | 新版 CCCL/CUB 自动演进 |

### Reduce 面试说法

Reduce 不只等于加法。只要操作满足结合律，并有 identity，就可把 `sum` 换成
`max/min/product/bitwise op`；浮点加法在数学上结合、在有限精度下不严格结合，
所以并行树顺序会带来可接受但需验证的数值差异。优化主线是：减少分歧和 barrier，
把 warp 内通信从 SMEM 变成 shuffle，再提高访存粒度；一旦 DRAM 已到 96%，继续
减少几条算术指令通常不会有可见收益，下一步应考虑融合。

## 表二：CUDA Core FP32 GEMM

课程 PPT 的正方形矩阵结果和本机 prefill-shape 结果是两套口径；下表明确标注，
只在各自口径内比较。

| 阶段 | 它是什么 / 在解决什么 | 数据层级与代码形态 | NCU 会怎样暴露问题 | 修复动作 | 结果或证据 | SM90/SM100 方向 |
|---|---|---|---|---|---|---|
| C0 Naive bad mapping | 一线程算一个 `Cij`，但 lane 到 row/col 的映射让 warp 地址跨行 | GMEM；每个线程独立扫 K | Global sectors/request 高、memory throughput 高但有效带宽低、long scoreboard 高 | lane-x 映射连续 N，让 B/C 合并访问；没有真的转置 B | PPT `1024^3`: 274.8 GFLOP/s | 合并访问永远是第一步 |
| C1 Coalesced | 只修 warp 的连续地址 | GMEM；lane-x -> 相邻列 | 访问规整后仍反复从 GMEM 取相同 A/B，算术强度低 | CTA 协作把 K tile 搬进 SMEM | PPT 3229.9 GFLOP/s；本机 compact 0.914 TFLOP/s | Hopper 可把规则 tile copy 换成 TMA |
| C2 SMEM tiling | 让 CTA 内线程复用 A/B tile | GMEM -> SMEM -> scalar FMA | DRAM 等待下降；shared wavefront、barrier/short scoreboard 开始重要 | 调 BM/BN/BK；对 shared leading dimension padding/swizzle | PPT 3953.2；本机 1.484 TFLOP/s | TMA、multicast、cluster SMEM |
| C3 1-D thread tile | 一个线程计算多个同列/同行输出 | SMEM -> registers；`acc[TM]` | shared load 次数下降；寄存器数上升，occupancy 开始下降 | 用一个 B 值更新多个 accumulator，扩大寄存器复用 | PPT 5421.8 GFLOP/s | 继续平衡 tile 与寄存器预算 |
| C4 2-D register tile | 每线程计算 `TM x TN` 微块 | `A_reg x B_reg` outer product | 本机 NCU：4.564 ms，FMA pipe 36.36%，long scoreboard 1.29，128 regs/thread，occupancy 30.69% | 全复用一组 A/B register；下一步不能盲目扩大微块，要降低寄存器压力 | PPT 5654.2；本机 compact 5.709 TFLOP/s | async copy 仍需和寄存器/occupancy 平衡 |
| C5 Padding / XOR swizzle | 修 shared-memory bank 映射 | 改 SMEM leading dimension 或地址置换 | `Shared Wavefronts Excessive`、`L1 Conflicts Shared N-Way` 指向具体 LDS/STS | padding 改 stride；swizzle 在不增加容量时打散 bank。效果可能接近，但地址/容量代价不同 | PPT BCF 后 7279.3 GFLOP/s | 新 MMA/TMA layout 仍需匹配 bank/fragment layout |
| C6 Vectorized | 减少 GMEM load/store 指令 | `float4`/128-bit transaction，带对齐与尾处理 | sectors 合理但 LSU 指令数仍高；小矩阵可能无收益 | 对齐后一次搬 16 B；不能把“vectorized”误解成更多 FLOP | 大矩阵收益更明显 | TMA 是更高层的异步 tile 搬运 |
| C7 Warp tiling | 显式建立 CTA -> warp -> thread 输出所有权 | GMEM -> SMEM -> warp tile -> register micro-tile | 比 C4 更少 regs/thread（本机 80）且更多 resident warps；再看 FMA pipe、issue、barrier | 把 `64x64` CTA 分成四个 `32x32` warp tile，每线程 `4x8` 输出，平衡复用与并行度 | PPT `1024^3`: 8.365 TF；`2048^3`: 14.454 TF；本机 7.508 TF | pipeline、warp specialization、persistent/cluster scheduling |
| cuBLAS | vendor kernel family + shape/架构 dispatch | 全层级 | 用作上限与差距定位，不等于单个固定 kernel | 比较必须保持 math mode/dtype/layout/epilogue 相同 | PPT `2048^3`: 17.500 TF；本机 FP32 10.923 TF | 新库自动选择对应架构路径 |

### CUDA Core GEMM 面试说法

主线不是背八份代码，而是回答三件事：谁拥有哪个输出 tile、A/B 当前在哪层存储、
搬运与计算何时发生。C0 的大问题是 warp 地址映射，不是把 B 从 row-major 变成
col-major。C2 用 SMEM 换取跨线程复用；C3/C4 用寄存器换取单线程复用；C5 修 bank
layout；C7 同时安排 CTA/warp/thread tile。NCU 证明 C4 已经不再主要等 GMEM，问题
转成 128 registers/thread 带来的低 occupancy，于是 C7 缩小每线程微块、显式分 warp。

## 表三：Tensor Core FP16 GEMM

| 阶段 | 它是什么 / 在解决什么 | CTA/warp 与数据路径 | NCU 证据 | 修复动作 | 本机结果 | Hopper/Blackwell 方向 |
|---|---|---|---|---|---:|---|
| T0 Naive WMMA | 证明 warp 级 MMA 正确；一个 warp 算一个 `16x16` C tile | grid `(384,32)`，block 32；每个 K=16 直接 `load_matrix_sync` GMEM -> fragment -> `mma_sync` | 5.171 ms；Tensor pipe 5.26%，Memory 96.75%，L1 97.21%，long scoreboard 47.94，40 regs/thread | 不再让每次 MMA 直接等 GMEM；先做 CTA tile 与数据 staging | 8.8417 ms / 2.91 TFLOP/s（event median） | 只作为教学基线 |
| T1 Handwritten Ada | 手写 warp-cooperative `mma.sync` 与时间流水 | grid `(48,4)`，block 256；`cp.async -> 3-stage permuted SMEM -> ldmatrix -> mma.sync`；CTA `128x128x32` | 1.081 ms；Tensor 38.26%，long scoreboard 1.24；但 DRAM 84.95%，139 regs/thread，50,176 B SMEM，occupancy 16.18%，math throttle 16.88 | 异步预取下一 K tile，并在当前 tile 上 MMA；用 swizzle/ldmatrix 匹配 fragment；下一步缩 CTA 或改善 schedule/epilogue | 1.0199 ms / 25.27 TFLOP/s | SM90：TMA + WGMMA + warp specialization |
| T2 CUTLASS tuned | 完整 production-style iterator/mainloop/epilogue | NCU size `(8,48)x(128)`；CTA `64x128x32`、3 stage、`mma.sync` | 0.707 ms；Tensor 49.44%，Memory 47.7%，long scoreboard 0.19；140 regs/thread、约 37 KiB SMEM、occupancy 15.9% | shape-specific tile 与 stage 选择，把资源花在更高 Tensor 利用率而非追求高 occupancy 本身 | 0.6769 ms / 38.07 TFLOP/s | SM90 collective mainloop、TMA multicast、cluster/persistent scheduler |
| cuBLAS | 闭源生产库与 runtime dispatch | 根据 shape/对齐/math mode 选 kernel family | 用 event benchmark 做稳定基线；可用 NCU/SASS 看行为，不能据此复制闭源实现 | 保持相同输入、FP32 accumulate/C、layout 与缓存策略 | 0.6994 ms / 36.85 TFLOP/s | 新库自动选 SM90/SM100 kernel |

### Tensor Core GEMM 面试说法

`WMMA` 是 CUDA C++ 的 warp-level API；`mma.sync` 是更接近硬件的 PTX 指令；Ada SASS
里常看到 `HMMA`。它们都仍运行在 SIMT 编程模型下，但一次 MMA 是整个 warp 协作的
矩阵操作，不是每个 lane 各做完整矩阵。fragment 的物理元素分布由 ABI/编译器定义，
最终落在各 lane 自己的寄存器中。Tensor Core 只负责矩阵乘加；GMEM 搬运、SMEM layout、
同步、流水、epilogue 和边界仍由 CUDA Core/LSU/warp scheduler 配合完成。

## NCU 到底怎样发现问题

按这个固定顺序，不要在 Summary 看到一条建议就改代码：

1. **先锁合同**：相同 `M/N/K`、dtype、accumulate/output、layout、alpha/beta、prepack、
   warmup 与 cache policy。NCU 顶部的 grid/block `Size` 不要求相同。
2. **Summary / Speed Of Light**：先分成 compute-bound、memory-bound 或 latency-bound。
   单看某个百分比不能下结论。
3. **Compute Workload Analysis**：CUDA Core 看 FP32/FMA pipe；Tensor Core 看 Tensor/HMMA
   pipe。T0 的 Tensor 仅 5.26%，说明“用了 WMMA”不等于“喂饱 Tensor Core”。
4. **Memory Workload Analysis**：看 DRAM/L1/L2、sectors/request、shared wavefronts 与 bank
   conflicts。高 memory throughput 可能是有效搬运，也可能是重复搬运。
5. **Warp State Statistics**：`long scoreboard` 通常对应等待 L1TEX/global/local 依赖；
   `short scoreboard/MIO` 常和 shared/特殊路径有关；`barrier` 是 CTA 同步；
   `math pipe throttle` 表示目标数学流水线或依赖链压力。
6. **Launch Statistics / Occupancy**：把寄存器、SMEM、threads/CTA 与 resident CTA/warp
   连起来。occupancy 低不必然慢，但必须解释为何仍能隐藏延迟。
7. **Source**：最后才定位具体 CUDA/PTX/SASS 行。把热点、stall sampling、LDS/STS、
   `HMMA`、`LDGSTS/cp.async` 对回 loop，不靠猜。
8. **一次只改一个主假设**，用 CUDA-event benchmark 复测；NCU 用于解释，不替代计时。

### 贯穿三表的瓶颈迁移

- Reduce：R0 的控制/同步浪费 -> R1/R2 消除后 -> R3 到 DRAM 96.3%，于是停止微调并考虑融合。
- CUDA Core GEMM：C0 合并访问 -> C2 跨线程复用 -> C4 长等待已低但 128 regs 限 occupancy -> C7 用 warp tile 重平衡。
- Tensor Core GEMM：T0 的 Tensor Core 饿死 -> T1 异步 staging 后 Tensor 利用率大涨，但单 CTA 资源太重 -> T2 用更合适 tile/schedule 提高有效 Tensor 工作。

## Shared wavefront 的面试解释与本次手算

NCU 的 shared-memory `wavefront` 不是 CUDA warp，也不是字节。它是 L1TEX 为完成一条
warp 级 shared-memory 指令而拆出的内部服务工作包；同一请求如果 bank/layout 不理想，
可能被拆成多个串行 wavefront。总数会累加所有动态 shared 指令、warp、CTA 和循环迭代。

截图中的 `M` 是十进制 mega，即一百万次计数：

```text
total      = 15.187964 M = 15,187,964
excessive  =  1.572844 M =  1,572,844
ideal      = total - excessive = 13,615,120
excessive / total = 10.355%
```

还能用该 CUTLASS kernel 的 launch 结构把总量几乎精确对上：

```text
CTA 数          = 8 * 48 = 384
K tile 数       = 4096 / 32 = 128
CTA-K tile 实例 = 384 * 128 = 49,152

实际约 309 wavefront / CTA-K tile:
49,152 * 309 = 15,187,968   # 距截图 4

理想约 277 wavefront / CTA-K tile:
49,152 * 277 = 13,615,104   # 距截图 16

额外约 32 wavefront / CTA-K tile:
49,152 * 32  = 1,572,864    # 距截图 20
```

这说明 15.188M 不是凭空来的，而是 `384 CTA x 128 K-stage` 的动态总和。注意：
`309/277` 仍然来自每条 SASS shared 指令的动态访问模式；不能只看 C++ tile 常量独立预测。
若要从第一性原理得到它，需在 NCU `Source` 中逐条统计 LDS/STS/ldmatrix 的动态执行次数
和每次请求的 wavefront，再求和。`Excessive` 也不总等同于传统 bank conflict，还可能受
active predicate、访问宽度和 layout 影响，所以要同时看 `L1 Conflicts Shared N-Way`。

NCU 的 `Estimated Speedup 10.02%` 是“若消除该规则识别出的额外 wavefront、其他结构
不变”的模型估计，不是性能承诺；它和 10.355% 的计数比例接近，但不要求相等。

## 为什么两个 Size 不同仍可比较

同一个 `512x6144x4096` 输出可以用不同 CTA 分工：

```text
Naive WMMA: grid (384,32), block 32
  6144/16 = 384 个 N tile，512/16 = 32 个 M tile
  一个 warp/CTA 只算 16x16

CUTLASS: grid (8,48), block 128
  512/64 = 8 个 M tile，6144/128 = 48 个 N tile
  一个 CTA 算 64x128，里面有 4 个 warp
```

两者都计算相同的 `2*M*N*K` FLOPs，grid/block 不同正是优化的一部分，所以可比。
真正不可混比的是第三个 `wmma_tf32_block` 报告：它是 TF32 口径，不应参与 FP16 性能排名，
只能用来观察中间架构形态。另需注意 RHS prepack 是否计时；本实验按推理权重复用场景，
反复 GEMM 的 event 时间不包含一次性 prepack。

## NCU 没有哪种“泳道/空泡图”

NCU 是单 kernel 的聚合计数、采样和源代码关联工具。它能把多个 report 作为 current/
baseline 叠加比较，但不会画出“每个 warp 每个 cycle 在哪条流水线、哪里出现空泡”的
真实时序泳道图。`Warp State Statistics` 是空泡原因的统计分布，不是时间线。

Nsight Systems 有 CPU thread、CUDA API、stream、kernel、memcpy 的系统时间线，适合看
launch gap、stream 并发和端到端调度；它也不是逐 warp 的 cycle-accurate 波形。面试时可说：
`nsys` 回答“时间花在哪个算子/stream”，`ncu` 回答“这个 kernel 为什么慢”。

## 60 秒回答模板

> 我先固定 shape、dtype、layout、缓存和计时口径，并验证正确性。然后用 NCU 的 SOL
> 判断算力、带宽还是延迟受限，再把 Warp State、Memory/Compute、Occupancy 和 Source
> 串起来形成一个假设。Reduce 从分歧/同步优化到 DRAM 96%；CUDA Core GEMM 从合并访存、
> SMEM 复用做到寄存器复用，随后发现 128 registers 限制 occupancy，再用 warp tiling
> 重平衡；Tensor Core 版则从 naive WMMA 的 Tensor pipe 5% 和 long scoreboard 48，做到
> `cp.async/ldmatrix/mma.sync` 后 Tensor 38%，最后用 CUTLASS 的更合理 tile/schedule 到 49%。
> 每次只改变一个主因素，并回到 CUDA-event median 验证真实加速。
