# Figure Regeneration

The `paper/figures/` directory stores a paper-local copy of every figure currently used by the manuscript, including the regenerated `public_service_*` and `public_moe_mask23_*` plots.

## Service figures

```bash
python scripts/plot_best_graphs.py \
  --phase-metrics-csv results/trn2-phase-inference-quick-fast/run_20260305T224828Z/metrics.csv \
  --phase-collectives-json results/trn2-phase-inference-quick-fast/run_20260305T224828Z/collectives_summary.json \
  --kernel-metrics-csv results/trn2-kernel-inference-optimized/run_20260305T221035Z/metrics.csv \
  --kernel-collectives-json results/trn2-kernel-inference-optimized/run_20260305T221035Z/collectives_summary.json \
  --out-dir results/plots \
  --prefix public_service
```

## Paper-only figures and data

```bash
python scripts/generate_paper_assets.py
```

## Direct end-to-end policy trace

```bash
torchrun --nproc_per_node=2 scripts/run_direct_policy_trace.py \
  --config configs/experiments/trn2_direct_policy_trace.yaml \
  --distributed
```

Latest committed measured run:

- `results/trn2_direct_policy_trace/run_20260310T044926Z/direct_policy_trace_summary.csv`
- `results/trn2_direct_policy_trace/run_20260310T044926Z/direct_policy_trace.png`

## MoE figures

```bash
python scripts/plot_moe_service_study.py \
  --metrics-csv results/trn2-moe-stable-small-merged-mask23/run_20260306T100500Z/metrics.csv \
  --decode-slo-csv results/trn2-moe-stable-small-merged-mask23/run_20260306T100500Z/decode_slo_summary.csv \
  --capacity-csv results/trn2-moe-stable-small-merged-mask23/run_20260306T100500Z/capacity_frontier.csv \
  --out-dir results/plots \
  --prefix public_moe_mask23
```

## Notes

- `public_service_hybrid_e2e.png` is composed from measured prefill latency and measured decode throughput.
- `public_service_direct_policy_trace.png` is directly measured from `run_20260310T044926Z`.
- `public_service_mixed_trace_goodput.png` is simulated from measured phase profiles.
- `public_service_comm_breakdown.png` uses the kernel microbenchmark artifact because the committed phase artifact predates the fixed decode collective-op attribution.
- `paper/figures/project_overview_pipeline.png` is a paper diagram, not a benchmark output.
- `paper/figures/chiplet_comm_scaling.png` is analytical and generated from the state-only communication formulas.
