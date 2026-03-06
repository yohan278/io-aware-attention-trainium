# AWS Chip Advice Memo: Dual-Die Direction for Trainium-Class Inference

**Date:** March 6, 2026  
**Project:** `io-aware-attention-trainium` dual-die emulation study on Trn2

## Executive Recommendation

For next-generation AWS AI chips, dual-die should be positioned primarily as a **throughput and capacity architecture** (request/KV sharding), not as a default way to reduce single-request latency via tensor-parallel collectives.

Measured data shows:

- Tensor-split dual-die is communication-dominated and loses badly in current form.
- Request/KV-sharded dual-die can improve service throughput in multi-request decode and long-context prefill scenarios.
- The break-even path for tensor-split requires large improvements in **collective latency, effective link bandwidth, and compute/comm overlap**.

## Evidence Base (Committed Artifacts)

The public repository currently keeps two compact Trn2-backed artifacts as the main evidence base:

1. `results/trn2-phase-inference-quick-fast/run_20260305T224828Z`
2. `results/trn2-moe-stable-small-merged-mask23/run_20260306T100500Z`

Larger sweeps exist as runnable commands in `docs/TRN2_EXECUTED_COMMANDS.md`, but the memo below is written to match the data that is actually committed in-repo.

## Key Findings

### 1) Public Trn2 Phase Artifact: Request Sharding Helps Decode, Tensor Split Loses Hard

From `results/trn2-phase-inference-quick-fast/run_20260305T224828Z`:

- Decode `ctx=2048, C=8`
  - `single_die`: **657 tok/s**, p90 **55.0 ms**
  - `dual_die_request_sharded`: **736.9 tok/s**, p90 **55.3 ms**
  - `dual_die_tensor_optimized`: **184.3 tok/s**, p90 **218.5 ms**
- Decode `ctx=2048, C=16`
  - `single_die`: **1212.5 tok/s**, p90 **58.9 ms**
  - `dual_die_request_sharded`: **1571.6 tok/s**, p90 **40.9 ms**
  - `dual_die_tensor_optimized`: **347.4 tok/s**, p90 **227.8 ms**

Prefill in the same artifact shows the same split:

- `seq=2048`
  - `single_die`: **14.35 ms**
  - `dual_die_request_sharded`: **23.82 ms**
  - `dual_die_tensor_optimized`: **410.75 ms**
- `seq=4096`
  - `single_die`: **36.72 ms**
  - `dual_die_request_sharded`: **40.61 ms**
  - `dual_die_tensor_optimized`: **1646.05 ms**

Interpretation: current dual-die value comes from service-level partitioning, not tensor-parallel collectives on the latency path.

### 2) Public Trn2 MoE Artifact: Locality Helps Communication, But This Is Not Yet a Single-Die Win

From `results/trn2-moe-stable-small-merged-mask23/run_20260306T100500Z`:

- Locality routing cuts remote expert dispatch materially:
  - median remote-dispatch delta vs naive: **-0.1875**
  - roughly **40%** median relative reduction
- Locality improves dual-MoE throughput vs naive modestly:
  - median throughput ratio: **1.034x**
- In one stronger case (`ctx=2048, C=16, skew=0.0`):
  - naive: **1326 tok/s**
  - locality: **2985 tok/s**

Interpretation: locality-aware placement is a real communication improvement, but the public MoE artifact should be pitched as “remote-dispatch reduction and dual-path improvement,” not as proof that dual-die MoE already beats single-die end to end.

### 3) Overlap and Link-Util Claims Need Better Instrumentation Than the Old Repo Used

- Older overlap values in this repo were derived from `compute = total - comm`, which made the overlap metric tautological.
- The current repo fixes that in the phase runner and removes misleading prose that treated those values as direct measurements.
- Link-utilization percentages from kernel artifacts should be treated as calibration-relative, not physical-link utilization, unless calibration covers the workload payload range.

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
  - tensor-split dual currently loses due communication dominance on the public Trn2 phase artifact
  - request/KV sharding is the most credible near-term dual-die inference path
- Medium confidence:
  - exact break-even thresholds for next-gen overlap/bandwidth, because the public repo keeps compact runs rather than the full service-day1 matrix
  - MoE end-to-end upside versus single-die, because the current public Trn2 MoE run validates locality gains more clearly than absolute single-die wins
- Required next run for final claims:
  - a committed multi-context service-day1 Trn2 run
  - multi-chip Trn2 placement for real cross-chip collective costs
  - `decode_steps >= 16`, `measure_iters >= 5`, with strict correctness gates retained
