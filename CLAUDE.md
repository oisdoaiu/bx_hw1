# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SIMD-optimized Approximate Nearest Neighbor Search (ANNS) on DEEP100K (96-dim vectors, 100k base). Compares brute-force, scalar quantization (SQ), and product quantization (PQ) — each with platform-specific SIMD (SSE/AVX2/ARM NEON). This is a university parallel programming lab assignment.

## Build Commands

No Makefile. Compile manually with g++:

```bash
# SSE (baseline x86-64)
g++ -O2 -fopenmp -lpthread -std=c++11 main.cc -o main_sse

# AVX2 (native, faster on modern x86)
g++ -O2 -march=native -fopenmp -lpthread -std=c++11 main.cc -o main_avx2

# Benchmark tool
g++ -O2 -fopenmp -lpthread -std=c++11 bench_pq.cc -o bench_pq
```

The `-march=native` flag determines which code paths in `#ifdef __AVX2__` / `#elif __ARM_NEON` / `#else` are activated.

## Architecture

**SIMD abstraction layer** (`simd_wrapper.h`): Wraps SSE `__m128` (lo+hi pair), AVX `__m256`, and ARM NEON `float32x4x2_t` behind a uniform `simd8float32` type with `+`, `-`, `*`, `storeu()` operators. All other headers depend on this.

**Search methods** (each header is self-contained, uses `#ifdef` for platform intrinsics):

| Header | Technique | Notes |
|--------|-----------|-------|
| `flat_scan.h` | Scalar brute-force | `InnerProductScalar`, baseline |
| `flat_simd.h` | SIMD brute-force | `InnerProductSIMD` with 4-accumulator unroll |
| `sq_simd.h` | Scalar quantization | Single-accumulator uint8 inner product |
| `sq_simd_improved.h` | Scalar quantization | 4-accumulator unrolled uint8 inner product |
| `pq.h` | Product quantization | AoS layout, scalar build_lut |
| `pq_centroid.h` | Product quantization | AoS + cross-centroid build_lut |
| `pq_simd.h` | Product quantization | SoA layout + cross-centroid + vectorized gather |
| `pq_gather.h` | Product quantization | SoA + cross-centroid + AVX2 gather + movemask threshold filtering |

**Two-stage search pattern** (SQ and PQ): Stage 1 — use quantized codes and a LUT to coarsely filter top-p candidates. Stage 2 — re-rank candidates with float `InnerProductSIMD` for final top-k.

**`main.cc`**: Loads DEEP100K binary data, calls `build_pq()` once, then runs 2000 query vectors through `pq_solve()`, measuring recall@10 and average latency. Currently includes `pq_gather.h`. The function signatures `build_pq()` and `pq_solve()` are the assignment's fixed ABI — only internal implementation may be modified.

**`bench_pq.cc`**: Standalone benchmark that sweeps M × p combinations using `pq.h`, writing results to `pq_results.txt`.

## Data

Symlink `data/` → `/home/oisdoaiu/并行/data` contains binary files:
- `DEEP100K.base.100k.fbin` — base vectors (100k × 96 floats)
- `DEEP100K.query.fbin` — query vectors (10k × 96 floats)
- `DEEP100K.gt.query.100k.top100.bin` — ground truth (10k × 100 ints)

Binary format: first 4 bytes = number of rows (size_t), next 4 bytes = dimension (size_t), then raw float/int data.

## Runtime Configuration

Environment variables (checked at runtime):
- `PQ_M` — number of subspaces for PQ (default 8)
- `PQ_P` — candidate pool size for PQ coarse filter (default varies by file)
- `SQ_P` — candidate pool size for SQ coarse filter (default 50)

## Results

See `benchmark_results.md` for full recall/latency tables across all methods, architectures, and parameter combinations.

---

# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
