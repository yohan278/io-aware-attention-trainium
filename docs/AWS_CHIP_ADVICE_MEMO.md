# AWS Chip Advice Memo: Dual-Die Direction for Trainium-Class Inference

**Date:** March 5, 2026  
**Project:** `io-aware-attention-trainium` dual-die emulation study on Trn2

## Executive Recommendation

For next-generation AWS AI chips, dual-die should be positioned primarily as a **throughput and capacity architecture** (request/KV sharding), not as a default way to reduce single-request latency via tensor-parallel collectives.

Measured data shows:

- Tensor-split dual-die is communication-dominated and loses badly in current form.
- Request/KV-sharded dual-die can improve service throughput in multi-request decode and long-context prefill scenarios.
- The break-even path for tensor-split requires large improvements in **collective latency, effective link bandwidth, and compute/comm overlap**.

## Evidence Base (Runs Used)

The conclusions below are based on Trn2.3xlarge L0 emulation runs with IDs:

1. `run_20260305T032452Z` (kernel strict)
2. `run_20260305T052604Z` (phase ultra-strict)
3. `run_20260303T233006Z`, `run_20260303T234000Z` (extended phase sweeps)

Generated artifacts are intentionally not committed in this public repository; these run IDs are the provenance anchors for archived metrics.

## Key Findings

### 1) Kernel-Level: Tensor-Parallel Dual-Dies Lose in Current Stack

From strict 5-kernel study (`qkv_proj`, `attention`, `mlp`, `rmsnorm`, `out_proj`):

- `dual_die_naive` median speedup vs single-die: **0.17x** (slowdown)
- `dual_die_optimized` median speedup vs single-die: **0.26x** (still slowdown)
- Dual comm share is high (median):
  - naive: about **67%** of latency
  - optimized: about **76%** of latency

Attention is the blocker:

- At `seq_len=1024`, attention p50:
  - single: **1.146 ms**
  - dual naive: **4.480 ms**
  - dual optimized (distributed merge): **231.310 ms**

Collectives for optimized attention (`seq_len=1024`) show many small reductions:

- `all_reduce_max` count p50: **64**
- `all_reduce_sum` count p50: **128**

This indicates latency-dominated collective overhead.

### 2) Prefill/Decode: Dual Benefit Appears in Sharding, Not Tensor Collectives

Ultra-strict run (`measure_iters=2`, `decode_steps=1`) confirms direction but is too short for stable decode claims:

- Prefill tensor-optimized dual:
  - `seq=2048`: **807.1 ms** vs single **14.4 ms**
  - `seq=4096`: **4051.0 ms** vs single **36.7 ms**
- Prefill request-sharded dual:
  - near parity but not consistent win in this strict short run

Extended phase sweeps (`decode_steps=8`) are more representative:

- `dual_die_tensor_optimized` median decode speedup: **0.078x** (major loss)
- `dual_die_request_sharded` median decode speedup: **1.20x** (wins)
- `dual_die_request_sharded` median prefill speedup: **1.58x** (wins)

Interpretation: dual-die value is service-level parallelism and memory partitioning, not single-stream tensor-parallel attention in current software/hardware conditions.

### 3) Break-Even Math: Why Tensor-Split Is Not Winning

In extended run (`run_20260303T234000Z`), tensor-optimized decode rows need very large comm reduction to tie single-die:

- comm fraction median: **91.16%**
- examples of required comm reduction:
  - decode `ctx=2048`, batch 8: roughly **243x**
  - decode `ctx=4096`, batch 8: roughly **32x**
  - decode `ctx=4096`, batch 16: roughly **75x**

In ultra-strict run, measured overlap is effectively **0%**, so communication is not hidden.

## Advice to AWS Silicon/Runtime Teams

### A) Product Positioning

- Market dual-die first for:
  - **decode throughput at SLO** via request/KV sharding
  - **capacity scaling** (larger effective KV/model footprint per service)
- Do not rely on current tensor-parallel collective path as the primary latency story.

### B) Next-Gen Hardware Priorities

1. **Low-latency collectives at small/medium payloads**
   - Current distributed attention patterns are reduction-count heavy.
   - Improve short-message collective latency and launch overhead dramatically.
2. **Higher effective die-to-die bandwidth under real collective patterns**
   - Not only peak bandwidth at 1MB payloads; focus on realistic payload size mix.
3. **True async compute/collective overlap**
   - Independent collective engines + scheduler/runtime support for overlap.
4. **Distributed-attention-friendly primitives**
   - Fast max/sum reductions for online-softmax merge style kernels.
5. **KV-cache aware memory hierarchy**
   - Better per-die KV locality and lower-cost cross-die access paths.

### C) Software/Compiler Priorities

1. Keep tensor collectives off critical single-request latency paths where possible.
2. Default serving strategy toward request/KV sharding for decode.
3. Add compiler/runtime support for tile-wise overlapped collective pipelines.
4. Expose profiling for collective count, payload mix, and overlap to users.

## Instance Recommendation for Final Evaluation

For final, hardware-real evidence, use **`trn2.32xlarge`** (or equivalent multi-chip Trn2 node), because:

- `trn2.3xlarge` L0 runs are valuable for methodology and trends, but ranks can still map within limited resources and do not fully represent cross-chip dual-die behavior.
- Multi-chip placement is required to calibrate real die-to-die alpha/beta and validate the advice quantitatively.

Practical split:

- `trn2.3xlarge`: kernel iteration, correctness, harness development.
- `trn2.32xlarge`: final L1 validation and report-grade numbers.

## Confidence and Caveats

- High confidence:
  - tensor-split dual currently loses due communication dominance
  - request/KV sharding is the most credible dual-die benefit path
- Medium confidence:
  - exact magnitude of decode throughput gain under strict tolerances, because short strict run used `decode_steps=1` and only 2 measured iterations
- Required next run for final claims:
  - multi-chip Trn2 placement
  - `decode_steps >= 16`, `measure_iters >= 5`
  - strict correctness gates retained
