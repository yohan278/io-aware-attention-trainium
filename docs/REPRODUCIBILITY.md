# Reproducibility Guide

This guide defines what is reproducible in this repo and the exact commands to do it.

## Scope

There are two reproducibility tiers:

1. **Artifact regeneration (local, no Trn2 required):** regenerate committed public plots/tables from committed `results/*` artifacts.
2. **Measurement reruns (Trn2 required):** rerun phase and direct-trace experiments to collect fresh metrics.

## 1) Artifact Regeneration (Local)

Prereqs:

- Python environment with repo dependencies installed.
- Committed run artifacts present under `results/`.

Run:

```bash
bash scripts/reproduce.sh
```

Primary outputs:

- `results/plots/public_service_decode_slo_frontier.png`
- `results/plots/public_service_direct_policy_trace.png`
- `results/plots/public_service_dual_dense_points.png`
- `results/plots/public_service_sharded_dense_analysis.png`
- `results/plots/public_service_mixed_trace_goodput.png`
- `results/plots/public_moe_mask23_decode_slo_frontier.png`

Paper-local sync outputs:

- `paper/figures/public_*.png`
- `paper/data/public_service_dual_dense_points.csv`
- `paper/data/public_service_sharded_dense_queue.csv`
- `paper/data/public_service_sharded_dense_analysis.md`

## 2) Measurement Reruns (Trn2)

### Environment sanity

```bash
python3 scripts/validate_trainium_env.py
```

### Canonical phase study

```bash
torchrun --nproc_per_node=2 scripts/run_phase_study.py \
  --config configs/experiments/trn2_phase_study.yaml \
  --distributed
```

### Direct policy trace (paper baseline)

```bash
torchrun --nproc_per_node=2 scripts/run_direct_policy_trace.py \
  --config configs/experiments/trn2_direct_policy_trace.yaml \
  --distributed
```

### Dense direct policy trace (many measured points)

```bash
torchrun --nproc_per_node=2 scripts/run_direct_policy_trace.py \
  --config configs/experiments/trn2_direct_policy_trace_dense.yaml \
  --distributed
```

Then regenerate dense analysis figures from that run:

```bash
python3 scripts/plot_direct_trace_dense_points.py \
  --summary-csv <dense_run>/direct_policy_trace_summary.csv \
  --samples-json <dense_run>/direct_policy_trace_samples.json

python3 scripts/analyze_sharded_serving_dense.py \
  --summary-csv <dense_run>/direct_policy_trace_summary.csv \
  --samples-json <dense_run>/direct_policy_trace_samples.json
```

## Reproducibility Notes

- Trainium runs can vary due compile cache warm state and host conditions. For paper-quality numbers, use at least one warm rerun and report exact run ID.
- `direct_policy_trace` and `direct_policy_trace_dense` are the canonical sources for end-to-end serving sample distributions.
- Composed and simulated plots are explicitly labeled as such in paper/docs.
