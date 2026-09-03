# Learning-CUDA

本项目为 2026 年夏季 InfiniTensor 大模型与人工智能系统训练营 CUDA 方向专业阶段的作业与项目系统。

## 项目结构

```text
Learning-CUDA/
├── Makefile
├── LICENSE
├── README.md
├── src
│   ├── kernels.cu
│   ├── kernels.maca
│   └── kernels.mu
└── tester
    ├── tester_iluvatar.o
    ├── tester_metax.o
    ├── tester_moore.o
    ├── tester_nv.o
    └── utils.h
```

## 环境配置

### 英伟达（NVIDIA）

- 如果你使用的是训练营所提供的服务器，遵照算力文档中的步骤配置好环境即可。
- 如果为本地或其他环境，请确保系统已安装 CUDA Toolkit 11.0 及以上、GNU Make，并支持 C++17。

### 天数智芯（Iluvatar CoreX）

- 如果你使用的是训练营所提供的服务器，遵照算力文档中的步骤配置并使用 BI-150 环境即可。
- 对于非训练营所提供的天数算力，请配置标准的天数 GPU 开放环境。本次作业在天数上默认需支持 C++17，且不保证能在所有其他天数环境上无修改直接运行。

### 沐曦集成电路（MetaX）

- 如果你使用的是训练营所提供的服务器，遵照算力文档中的步骤配置环境即可。
- 对于非训练营所提供的沐曦算力，请配置标准的沐曦 GPU 开放环境。本次作业在沐曦上默认需支持 C++17，且不保证能在所有其他沐曦环境上无修改直接运行。

### 摩尔线程（Moore Threads）

- 如果你使用的是训练营所提供的服务器，请先遵照算力文档中的步骤配置环境。
- 对于非训练营所提供的摩尔算力，请配置标准的摩尔 GPU 开放环境。本次作业在摩尔上默认需支持 C++11，且不保证能在所有其他摩尔环境上无修改直接运行。

## 作业

作业一共有两题。需实现 `src/kernels.cu` 中给定的 **2 个 CUDA 函数**。

1. **rmsNorm**

实现 RMSNorm 算子。给定输入矩阵 `h_input`、权重向量 `h_weight`、输出矩阵 `h_output`、行数 `rows`、隐藏维度 `hidden_dim` 和稳定项 `eps`，对每一行独立计算：

```text
mean_square = sum_j input[i, j]^2 / hidden_dim
output[i, j] = input[i, j] * rsqrt(mean_square + eps) * weight[j]
```

输入和输出均按 row-major 方式展平存储。该函数需支持 `float` 和 `half` 两种类型。

2. **flashAttention**

实现 Flash Attention 算子。需支持 causal masking 和 GQA。具体行为与 [torch.nn.functional.scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 保持一致。接口未提供的参数所代表的功能无需支持和实现。具体参数要求请参考文件中的注释。该函数需支持 `float` 和 `half` 两种类型。

### 国产平台适配

在完成英伟达的基础上，可以将实现适配至天数、沐曦和/或摩尔这三款 GPU 平台上。

- 天数适配需同样在 `src/kernels.cu` 中进行；
- 沐曦适配需在 `src/kernels.maca` 中进行；
- 摩尔适配需在 `src/kernels.mu` 中进行；

具体编译和运行方式以及国产适配对评分的影响，分别可见下面的 **编译与运行** 与 **评分规则** 两部分。

### 注意事项

1. 禁止抄袭与舞弊，包括抄袭其他学员的代码和开源实现。可以讨论和参考思路，但禁止直接看/抄代码。一经发现，成绩作废并失去进入项目阶段和后续实习与推荐等资格；
2. 两个题目都禁止使用任何库函数来直接实现关键功能；
3. 主要计算均需在 GPU 上实现；如有一些信息和程序准备性质的，例如元信息计算、资源准备等，则可以在 CPU/Host 上进行；
4. 代码风格不限，但需保持一致；
5. 需进行适当的代码注释解释重要部分。

### 提交方式

在 InfiniTensor 开源社区作业页面提交 GitHub 链接，无需提交 PR，无需重复提交，评分将以截止日期前的最新提交为准。详细提交方式可见作业提交页面。

## 编译与运行

代码编译与运行可以使用提供的 `Makefile`。

### 构建与运行指令

以下命令需在项目根目录执行：

1. 默认构建并运行测试：

```bash
make
```

2. 构建并运行 verbose 模式测试：

```bash
make VERBOSE=true
```

3. 选择性测试算子：

如果只想测试第一题 `rmsNorm`，可以跳过第二题：

```bash
SKIP_ATTENTION=1 make
```

如果只想测试第二题 Flash Attention，可以跳过第一题：

```bash
SKIP_RMS_NORM=1 make
```

4. 选择编译平台：

```bash
make PLATFORM=nvidia
make PLATFORM=iluvatar
make PLATFORM=metax
make PLATFORM=moore
```

默认平台为英伟达，即不指定 `PLATFORM` 时等价于 `make PLATFORM=nvidia`。

### 环境变量

- `SKIP_RMS_NORM`: 跳过第一题的 `rmsNorm` 测试。
- `SKIP_ATTENTION`: 跳过第二题的 Flash Attention 测试。

## 评分规则

1. 正确性优先：所有提交首先以正确性为前提，需在提供的测试用例中正确输出结果。
2. 性能加分：在正确性的基础上，会对各实现的性能进行排名。
3. 平台适配加分：每道题在英伟达上测例正确的基础上，每多适配一个国产平台可以获得固定得分乘算系数。
4. 综合评判：代码质量、编译与运行问题、是否符合注意事项等会影响最终成绩。

## 有疑问？

可以在群里直接询问助教。

## CUDA 学习实验

`gemm/` 和 `reduce/` 是独立的学习、benchmark 与 profiling 实验，便于
练习面试常见 kernel；它们不替代上面的训练营评分入口 `src/`。

### CUDA Core GEMM：只记四版

| 版本 | 解决的问题 | 核心数据路径 | 本仓库入口 | 更新架构继续加什么 |
|---|---|---|---|---|
| V0 Coalesced | 先让 warp 连续访问 B/C | GMEM | `gemm/kernels/v1_coalesced.cu` | 所有架构都必须先做好映射和合并访问 |
| V1 SMEM Tile | A/B 不要被每个线程反复从显存读取 | GMEM -> SMEM | `gemm/kernels/v2_smem_tiled.cu` | Hopper+ 可用 TMA 搬 tile、cluster multicast |
| V2 Register Tile | 降低 SMEM 流量并扩大 A/B 复用 | SMEM -> registers | `gemm/kernels/v4_2d_threadtile.cu` | 更宽 MMA/异步流水仍要平衡复用、寄存器和 occupancy |
| V3 Warp Tile | 建立 block -> warp -> thread 层次 | GMEM -> SMEM -> warp/register | `gemm/kernels/v7_warptiling.cu` | Hopper/Blackwell：TMA、WGMMA/UMMA、warp specialization、persistent/cluster scheduling |
| cuBLAS | 生产级 shape/架构特化基线 | 全层级 | `gemm/extras/cublas_baseline.cu` | 新版库自动选择对应架构路径 |

完整八级拆分、bank conflict、vectorized load 和 NCU 观察点见
[`gemm/notes/GEMM_OPT_TABLE.md`](gemm/notes/GEMM_OPT_TABLE.md)。

### Tensor Core GEMM：Ada 实测四版

数据契约为 FP16 A/B、FP32 accumulate/C；RTX 4070 Laptop、
`M=512,N=6144,K=4096`、cold-L2、CUDA-event median。

| 版本 | 核心形态 | 实测 | 本仓库入口 | 更新架构继续加什么 |
|---|---|---:|---|---|
| T0 Naive WMMA | 一个 warp 一个 `16x16` tile，直接反复读 GMEM | 8.8417 ms / 2.91 TFLOP/s | `gemm/extras/tensorcore_fp16_kernels.cuh` | 保留为教学基线 |
| T1 Handwritten Ada | `cp.async -> permuted SMEM -> ldmatrix -> mma.sync`，3-stage | 1.0199 ms / 25.27 TFLOP/s | 同上 | Hopper：TMA + WGMMA + warp specialization；Blackwell：UMMA/TMEM |
| T2 CUTLASS tuned | `64x128x32` CTA、3-stage、完整 iterator/epilogue | 0.6769 ms / 38.07 TFLOP/s | `gemm/extras/tensorcore_fp16_cutlass.cuh` | SM90/SM100 collective mainloop、TMA multicast、cluster/persistent scheduler |
| cuBLAS | vendor kernel family + runtime dispatch | 0.6994 ms / 36.85 TFLOP/s | `gemm/extras/tensorcore_fp16_kernels.cuh` | 新版库自动选择架构特化实现 |

这个 3.3% 的 CUTLASS 领先只对该固定 shape 和测试口径成立；完整报告与
NCU 双报告对比方法见
[`gemm/notes/TENSOR_CORE_OPT_TABLE.md`](gemm/notes/TENSOR_CORE_OPT_TABLE.md)。

### Reduce：通用 sum/max/min

RTX 4070 Laptop、FP32 sum、`N=16,777,216`、cold-L2、CUDA-event median。

| 版本 | 解决的问题 | 核心形态 | 实测 / 有效带宽 | 更新架构继续加什么 |
|---|---|---|---:|---|
| V0 Interleaved | 先把多 block reduce 写对 | modulo shared-memory tree | 0.7311 ms / 91.79 GB/s | 无架构依赖 |
| V1 Sequential | 去分歧并把首轮输入翻倍 | two inputs/thread | 0.3553 ms / 188.88 GB/s | 对齐后扩大 vector load |
| V2 Warp Shuffle | warp 内不再用 SMEM/barrier | `__shfl_down_sync` + 每 warp 一个 partial | 0.3543 ms / 189.42 GB/s | cooperative groups 可增强可读性/可移植性 |
| V3 Vectorized | 减 load 指令和中间 kernel 数 | `float4` + grid-stride + hierarchical reduce | 0.3523 ms / 190.49 GB/s | Hopper+ 可研究 persistent/cooperative、cluster/DSM；更常见的收益是算子融合 |
| CUB DeviceReduce | 生产级库基线 | 架构特化 dispatch | 0.3533 ms / 189.95 GB/s | 新版 CCCL/CUB 自动更新策略 |

V3 与 CUB 的中位数仅差约 0.3%；NCU 显示 V3 首轮已达到 96.3% DRAM
吞吐利用率。源码、完整表和复现实验见
[`reduce/README.md`](reduce/README.md)。面试练习顺序见
[`gemm/notes/INTERVIEW_KERNEL_SPRINT.md`](gemm/notes/INTERVIEW_KERNEL_SPRINT.md)。
