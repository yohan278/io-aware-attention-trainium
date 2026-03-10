# Claims Map

## Main claim

- **Claim:** Dual-die value is phase-aware: request-sharded decode helps serving, tensor-split single-request execution does not.
- **Measured evidence:** `results/trn2-phase-inference-quick-fast/run_20260305T224828Z/metrics.csv`
- **Primary figures:** `public_service_decode_slo_frontier.png`, `public_service_prefill_ratio.png`

## Mechanism claim

- **Claim:** Tensor-split attention loses because communication structure dominates, especially collective count and reduction latency.
- **Measured evidence:** `results/trn2-kernel-inference-optimized/run_20260305T221035Z/metrics.csv`
- **Primary figure:** `public_service_comm_breakdown.png`

## Policy claim

- **Claim:** Request-aware decode policies dominate end-to-end; composed and simulated views explain when single-prefill/request-decode is preferred.
- **Measured direct evidence:** `results/trn2_direct_policy_trace/run_20260310T044926Z/direct_policy_trace_summary.csv`
- **Measured direct figure:** `public_service_direct_policy_trace.png`
- **Composed evidence:** `public_service_hybrid_e2e.png`
- **Simulated evidence:** `public_service_mixed_trace_goodput.png`, `public_service_service_trace_summary.csv`

## Secondary claim

- **Claim:** Locality-aware MoE routing improves the dual-die path by reducing remote dispatch, but it does not overturn the main single-vs-dual result.
- **Measured evidence:** `results/trn2-moe-stable-small-merged-mask23/run_20260306T100500Z/metrics.csv`
- **Primary figure:** `public_moe_mask23_locality_gain.png`

## Algorithmic claim

- **Claim:** Exact partitioned attention can be merged from online-softmax state only.
- **Derived evidence:** `paper/data/chiplet_proxy_example.csv`
- **Primary figure:** `paper/figures/chiplet_comm_scaling.png`
