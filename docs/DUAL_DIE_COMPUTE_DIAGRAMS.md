# Dual-Die Compute Diagrams

This document shows how compute and communication differ across single-die and dual-die execution modes.

## 1) Emulation and Hardware Mapping

```mermaid
flowchart LR
  subgraph L0["L0: Compute-Split Proxy (trn2.3xlarge)"]
    R0L0["Rank 0 (Core group A)"]
    R1L0["Rank 1 (Core group B)"]
    HBM0["Shared physical HBM domain"]
    R0L0 --> HBM0
    R1L0 --> HBM0
  end

  subgraph L1["L1: Real Dual-Chip Placement (multi-chip Trn2)"]
    CHIP0["Die/Chip 0: NCs + Local HBM"]
    CHIP1["Die/Chip 1: NCs + Local HBM"]
    CHIP0 <-- "NeuronLink collectives" --> CHIP1
  end

  subgraph L2["L2: Parametric What-If Model"]
    M1["T_comm = alpha*count + bytes/beta"]
    M2["T_dual = T_compute_split + T_comm*(1-overlap)"]
    M1 --> M2
  end
```

## 2) End-to-End Compute Path Differences

```mermaid
flowchart TB
  subgraph S["Single Die"]
    S0["Input batch"]
    S1["RMSNorm + QKV + Attention + OutProj + MLP"]
    S2["Output"]
    S0 --> S1 --> S2
  end

  subgraph DT["Dual Die: Tensor-Split (dual_die_tensor_optimized)"]
    D0["Replicated input"]
    D1["Rank 0 local shard compute"]
    D2["Rank 1 local shard compute"]
    C1["Collectives (all_reduce/all_gather)"]
    D3["Merged output"]
    D0 --> D1
    D0 --> D2
    D1 --> C1
    D2 --> C1
    C1 --> D3
  end

  subgraph DR["Dual Die: Request/KV Sharded (dual_die_request_sharded)"]
    R0["Request subset A + KV shard A"]
    R1["Request subset B + KV shard B"]
    O0["Output subset A"]
    O1["Output subset B"]
    R0 --> O0
    R1 --> O1
  end
```

## 3) Distributed Attention (Optimized Merge)

Grouped distributed online-softmax merge (new optimization) reduces collective count by merging several K tiles locally before global reductions.

```mermaid
sequenceDiagram
  participant R0 as "Rank 0"
  participant R1 as "Rank 1"

  Note over R0,R1: For one Q tile and one K-group (reduce_group_k tiles)
  R0->>R0: "Compute local logits/prob/output over local K tiles"
  R1->>R1: "Compute local logits/prob/output over local K tiles"
  R0->>R1: "all_reduce(max) for m_group"
  R1->>R0: "all_reduce(max) for m_group"
  R0->>R1: "all_reduce(sum) for l_group"
  R1->>R0: "all_reduce(sum) for l_group"
  R0->>R1: "all_reduce(sum) for o_group"
  R1->>R0: "all_reduce(sum) for o_group"
  R0->>R0: "Merge group state into running online-softmax state"
  R1->>R1: "Merge group state into running online-softmax state"
```

## 4) Kernel-Level Communication Pattern

| Kernel | Single Die | Dual Naive | Dual Optimized |
|---|---|---|---|
| `qkv_proj` | local GEMM | split + gather | split + reduced gather |
| `attention` | local SDPA | gather-heavy logits path | distributed online-softmax merge with reductions |
| `mlp` | local GEMMs | hidden gather path | split + reduction combine |
| `rmsnorm` | local | gather path | gather + small reductions |
| `out_proj` | local GEMM | gather path | split + reduction combine |

## 5) Why Inference Story Uses Different Metrics

- Prefill: long-context latency ratio (`dual/single`) by sequence length.
- Decode: throughput-at-SLO frontier by context length and concurrency.
- Capacity: tokens/s per GiB KV cache footprint.
- Bottleneck attribution: compute vs communication stacked latency.
