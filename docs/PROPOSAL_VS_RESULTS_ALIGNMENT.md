# Proposal vs Results Alignment

## Original intuition

The initial intuition behind dual-die attention was that splitting one request across multiple dies might reduce latency or increase efficiency.

## What the current results actually support

The committed evidence supports a different thesis:

- dual-die helps when used as a serving architecture for concurrent decode
- dual-die hurts when used as tensor-parallel splitting of a single request
- prefill remains better on single-die in the measured regime

## Aligned final claim

The aligned final claim is:

> Dual-die value on Trainium-class accelerators is phase-aware serving value, not tensor-parallel latency reduction.

## Consequences for the paper

### Keep in the main paper

- request-sharded decode throughput/latency gains
- prefill single-die advantage
- hybrid single-prefill/request-decode policy
- communication-structure explanation for tensor-split failure

### Keep as secondary or appendix

- MoE locality routing
- kernel-level calibration details
- any link-utilization claims that depend on calibration beyond the measured payload range

### Do not claim

- “dual-die makes attention faster in general”
- “tensor-split attention is the winning dual-die strategy”
- “composed hybrid plots are direct end-to-end measurements”
