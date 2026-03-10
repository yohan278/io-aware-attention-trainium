# Evidence Index

## Main claim

**Claim:** dual-die value is phase-aware: request-sharded decode helps; tensor-split single-request execution does not.

- measured phase artifact: `results/trn2-phase-inference-quick-fast/run_20260305T224828Z/metrics.csv`
- figure: `results/plots/public_service_decode_slo_frontier.png`
- figure: `results/plots/public_service_prefill_ratio.png`

## Mechanism claim

**Claim:** tensor-split attention loses because of collective structure and reduction overhead.

- measured kernel artifact: `results/trn2-kernel-inference-optimized/run_20260305T221035Z/metrics.csv`
- figure: `results/plots/public_service_comm_breakdown.png`
- figure: `results/plots/public_service_collective_count_vs_latency.png`

## Policy claim

**Claim:** request-aware decode policies dominate end-to-end, and composed/simulated views explain when single-prefill/request-decode is preferred.

- measured direct trace: `results/trn2_direct_policy_trace/run_20260310T044926Z/direct_policy_trace_summary.csv`
- figure: `results/plots/public_service_direct_policy_trace.png`
- composed evidence: `results/plots/public_service_hybrid_e2e.png`
- simulated evidence: `results/plots/public_service_mixed_trace_goodput.png`
- summary table: `results/plots/public_service_service_trace_summary.csv`

## Algorithmic claim

**Claim:** exact partitioned attention can be merged from state only.

- paper data: `paper/data/chiplet_proxy_example.csv`
- figure: `paper/figures/chiplet_comm_scaling.png`

## Secondary claim

**Claim:** locality-aware MoE routing improves the dual path but does not overturn the main single-vs-dual result.

- measured MoE artifact: `results/trn2-moe-stable-small-merged-mask23/run_20260306T100500Z/metrics.csv`
- figure: `results/plots/public_moe_mask23_locality_gain.png`
- figure: `results/plots/public_moe_mask23_remote_dispatch_ratio.png`
