# ARM 算法测试封装 — 使用手册

## 概述

本目录包含 9 个 `.h` 封装文件，每个文件提供独立的 `build_xxx()` 和 `xxx_solve()` 接口，
用于在 ARM 服务器上通过 `main.cc` 进行测试。

所有封装均基于本地已完成实验的算法实现。依赖服务器上已有的 `simd_wrapper.h` 和 `flat_simd.h`（通过 `#include "xxx.h"` 引用，需放在同一目录）。

## 在 main.cc 中的使用方法

main.cc 中可修改的 3 处：

```cpp
// 1. 修改 include
#include "arm_algo/algo_xxx.h"

// 2. 修改 build 调用 (第89行)
build_xxx(base, base_number, vecdim);

// 3. 修改 solve 调用 (第100行)
auto res = xxx_solve(base, test_query + i*vecdim, base_number, vecdim, k);
```

## 文件列表与接口

| 文件 | 算法 | build 函数 | solve 函数 | 多线程 |
|------|------|-----------|-----------|--------|
| `algo_flat.h` | Flat-SIMD | `build_flat` | `flat_solve` | 无 |
| `algo_flat_omp.h` | Flat-SIMD + OpenMP base级 | `build_flat_omp` | `flat_omp_solve` | `#pragma omp parallel` |
| `algo_pq.h` | PQ-SIMD (pq_gather) | `build_pq` | `pq_solve` | 无 |
| `algo_pq_omp.h` | PQ-SIMD + OpenMP LUT级 | `build_pq_omp` | `pq_omp_solve` | `#pragma omp parallel for` |
| `algo_ivf.h` | IVF-SIMD | `build_ivf_algo` | `ivf_algo_solve` | 无 |
| `algo_ivf_omp.h` | IVF-SIMD + OpenMP 簇级 | `build_ivf_omp` | `ivf_omp_solve` | `#pragma omp parallel` |
| `algo_ivf_pq.h` | IVF-PQ + rerank | `build_ivf_pq_algo` | `ivf_pq_algo_solve` | 无 |
| `algo_ivf_pq_omp.h` | IVF-PQ+rerank + OpenMP 簇级 | `build_ivf_pq_omp` | `ivf_pq_omp_solve` | `#pragma omp parallel` |
| `algo_hnsw.h` | HNSW 图索引 | `build_hnsw_algo` | `hnsw_algo_solve` | 无 (构建时有 `#pragma omp parallel for`) |

## 参数与建议测试值

### 通用参数

| 环境变量 | 说明 | 默认值 | 建议测试值 |
|---------|------|--------|----------|
| `NUM_THREADS` | OpenMP 线程数 (仅 `_omp` 版本) | 4 | 1, 2, 4, 8 |

### Flat 系列

无额外参数。

**建议测试**:
- `algo_flat.h`: 直接测试，无需调参
- `algo_flat_omp.h`: `NUM_THREADS=1,2,4,8`

### PQ 系列 (algo_pq.h, algo_pq_omp.h)

| 环境变量 | 说明 | 默认值 | 建议测试值 |
|---------|------|--------|----------|
| `PQ_M` | 子空间数量 | 8 | 4, 8, 12, 16, 24 |
| `PQ_P` | 粗筛候选池大小 | 500 | 100, 200, 500 |

**建议测试**:
- `algo_pq.h`: `PQ_M=8,12,16`, `PQ_P=100,500`
- `algo_pq_omp.h`: `PQ_M=8`, `PQ_P=500`, `NUM_THREADS=1,2,4,8`

### IVF 系列 (algo_ivf.h, algo_ivf_omp.h)

| 环境变量 | 说明 | 默认值 | 建议测试值 |
|---------|------|--------|----------|
| `IVF_NLIST` | 聚类数 | 256 | 256 (固定) |
| `IVF_NPROBE` | 搜索簇数 | 8 | 2, 4, 8, 16, 32 |

**建议测试**:
- `algo_ivf.h`: `IVF_NPROBE=4,8,16,32` (绘制 recall-latency 曲线)
- `algo_ivf_omp.h`: `IVF_NPROBE=16,32`, `NUM_THREADS=1,2,4,8`

### IVF-PQ 系列 (algo_ivf_pq.h, algo_ivf_pq_omp.h)

| 环境变量 | 说明 | 默认值 | 建议测试值 |
|---------|------|--------|----------|
| `IVF_NPROBE` | 搜索簇数 | 32 | 16, 32, 48, 64 |
| `IVF_RERANK` | rerank候选数 | 500 | 200, 500 |

**建议测试**:
- `algo_ivf_pq.h`: `IVF_NPROBE=32,64`, `IVF_RERANK=200,500`
- `algo_ivf_pq_omp.h`: `IVF_NPROBE=64`, `IVF_RERANK=500`, `NUM_THREADS=1,2,4,8`

### HNSW 系列 (algo_hnsw.h)

| 环境变量 | 说明 | 默认值 | 建议测试值 |
|---------|------|--------|----------|
| `HNSW_M` | 图连接数 | 16 | 16 (固定) |
| `HNSW_EFC` | 构建时 ef_construction | 200 | 150, 200 |
| `HNSW_EF` | 查询时 ef | 50 | 10, 20, 50, 100, 200 |

**建议测试**:
- `HNSW_EF=10,20,50,100,200` (绘制 recall-latency 曲线)

> 注意: algo_hnsw.h 要求 main.cc 已 include hnswlib 头文件且 `using namespace hnswlib`

## 测试顺序建议

1. **串行 baseline** (全部 `非_omp` 版本) — 测试各算法在 ARM 上的基础性能
2. **Flat OpenMP** (`algo_flat_omp.h`) — 确认 ARM 上 base级并行的加速比
3. **IVF OpenMP** (`algo_ivf_omp.h`) — 测试粗排/精排并行的 ARM 表现
4. **PQ/IVF-PQ OpenMP** — 预计负优化，确认 ARM 上的 memory-bound 结论
5. **HNSW** (`algo_hnsw.h`) — ef sweep，找 ARM 上的最佳 recall-latency 点

## 编译 (ARM 服务器)

```bash
g++ -O2 -march=native -fopenmp -lpthread -std=c++11 main.cc -o main_arm
```

ARM NEON 路径通过 `simd_wrapper.h` 中的 `#ifdef __ARM_NEON` 自动激活。

## 本地验证

在 x86 上亦可编译测试（用于对比），AVX2 路径自动激活：
```bash
# 注意: 本地数据路径为 data/, 需修改 main.cc 或创建 /anndata/ 软链接
g++ -O2 -march=native -fopenmp -lpthread -std=c++11 main.cc -o main_test
```
