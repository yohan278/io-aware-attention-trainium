# Experiment Guide

This document describes the primary experiment entrypoints and output schema.

## 1) Kernel Study

Compare single-rank vs dual-rank strategies for:

- `qkv_proj`
- `attention`
- `mlp`
- `rmsnorm`
- `out_proj`

Run:

```bash
torchrun --nproc_per_node=2 scripts/run_kernel_study.py \
  --config configs/experiments/trn2_kernel_study.yaml \
  --device trainium \
  --distributed
```

Communication-sensitive MoE variant:

```bash
torchrun --nproc_per_node=2 scripts/run_moe_service_study.py \
  --config configs/experiments/moe_comm_sensitive_cpu.yaml \
  --device cpu \
  --distributed
```

Strict/fast variant:

```bash
torchrun --nproc_per_node=2 scripts/run_kernel_study.py \
  --config configs/experiments/trn2_kernel_quick_fp32_strict.yaml \
  --device trainium \
  --distributed
```

Inference-optimized variant (grouped distributed-merge attention):

```bash
torchrun --nproc_per_node=2 scripts/run_kernel_study.py \
  --config configs/experiments/trn2_kernel_inference_optimized.yaml \
  --device trainium \
  --distributed
```

## 2) Phase Study (Prefill + Decode)

Run:

```bash
torchrun --nproc_per_node=2 scripts/run_phase_study.py \
  --config configs/experiments/trn2_phase_study.yaml \
  --device trainium \
  --distributed
```

Ultra-strict/quick variant:

```bash
torchrun --nproc_per_node=2 scripts/run_phase_study.py \
  --config configs/experiments/trn2_phase_ultra_strict.yaml \
  --device trainium \
  --distributed
```

Inference-focused variant (recommended for chip-advice storyline):

```bash
torchrun --nproc_per_node=2 scripts/run_phase_study.py \
  --config configs/experiments/trn2_inference_story.yaml \
  --device trainium \
  --distributed
```

Lower-cost quick inference variant:

```bash
torchrun --nproc_per_node=2 scripts/run_phase_study.py \
  --config configs/experiments/trn2_inference_story_quick.yaml \
  --device trainium \
  --distributed
```

Expanded service-scale day-1 variant (use when you want the larger matrix, not the small committed artifact set):

```bash
torchrun --nproc_per_node=2 scripts/run_phase_study.py \
  --config configs/experiments/trn2_inference_service_day1.yaml \
  --device trainium \
  --distributed
```

## 3) MoE Service Study (Decode Throughput/Capacity)

This study validates whether dual-die can improve serving optics for sparse models via locality-aware expert placement.

Run:

```bash
torchrun --nproc_per_node=2 scripts/run_moe_service_study.py \
  --config configs/experiments/trn2_moe_service_day1.yaml \
  --device trainium \
  --distributed
```

Recommended on unstable Trn2 hosts:

```bash
NEURON_RT_VISIBLE_CORES=2,3 torchrun --nproc_per_node=2 scripts/run_moe_service_study.py \
  --config configs/experiments/trn2_moe_service_trainium_focus.yaml \
  --device trainium \
  --distributed
```

## 4) Plotting

Kernel study plots:

```bash
python scripts/plot_kernel_study.py \
  --metrics-csv <run_dir>/metrics.csv \
  --out-dir results/plots \
  --prefix <name>
```

Phase study plots:

```bash
python scripts/plot_phase_study.py \
  --metrics-csv <run_dir>/metrics.csv \
  --kernel-phase-csv <run_dir>/kernel_phase_metrics.csv \
  --out-dir results/plots \
  --prefix <name>
```

Capacity frontier plots:

```bash
python scripts/plot_capacity_frontier.py \
  --metrics-csv <run_dir>/metrics.csv \
  --capacity-csv <run_dir>/capacity_frontier.csv \
  --out-dir results/plots \
  --prefix <name>
```

Mixed-traffic simulation:

```bash
python scripts/simulate_mixed_traffic.py \
  --metrics-csv <run_dir>/metrics.csv \
  --out-dir results/plots \
  --prefix <name>
```

The simulator compares deployment policies:

- `single->single`
- `single->request`
- `single->tensor`
- `request->request`

MoE service plots:

```bash
python scripts/plot_moe_service_study.py \
  --metrics-csv <run_dir>/metrics.csv \
  --decode-slo-csv <run_dir>/decode_slo_summary.csv \
  --capacity-csv <run_dir>/capacity_frontier.csv \
  --out-dir results/plots \
  --prefix <name>
```

MoE compact summary tables:

```bash
python scripts/summarize_moe_service.py \
  --metrics-csv <run_dir>/metrics.csv \
  --decode-slo-csv <run_dir>/decode_slo_summary.csv \
  --out-dir results/plots \
  --prefix <name>
```

## 5) What-If Dual-Die Model

```bash
python scripts/what_if_dual_die.py \
  --metrics-csv <run_dir>/metrics.csv \
  --collectives-json <run_dir>/collectives_summary.json \
  --fabric-json <run_dir>/fabric_calibration.json \
  --out-dir results/plots \
  --prefix <name>
```

## 6) Curated Inference Plot Set

Use this to keep only high-signal serving plots:

```bash
python scripts/plot_inference_track.py \
  --metrics-csv <run_dir>/metrics.csv \
  --decode-slo-csv <run_dir>/decode_slo_summary.csv \
  --break-even-csv <run_dir>/break_even_summary.csv \
  --out-dir results/plots \
  --prefix inference_story \
  --purge-stale
```

The script emits:

- `*_prefill_ratio.png`
- `*_decode_slo_frontier.png`
- `*_decode_kv_efficiency.png`
- `*_comm_breakdown.png`
- `*_plot_manifest.md` (why each kept plot is useful)

For the smaller public repo bundle, use:

```bash
python scripts/plot_best_graphs.py \
  --phase-metrics-csv <phase_run>/metrics.csv \
  --decode-slo-csv <phase_run>/decode_slo_summary.csv \
  --phase-collectives-json <phase_run>/collectives_summary.json \
  --kernel-metrics-csv <kernel_run>/metrics.csv \
  --kernel-collectives-json <kernel_run>/collectives_summary.json \
  --out-dir results/plots \
  --prefix public_service
```

## 7) Output Files

Each run directory (`results/<run_id>/`) may include:

- `metrics.csv`, `metrics.jsonl`
- `run_manifest.json`
- `collectives_summary.json`
- `fabric_calibration.csv`, `fabric_calibration.json`
- phase-specific summaries:
  - `break_even_summary.csv`, `break_even_summary.md`
  - `decode_slo_summary.csv`, `decode_slo_summary.md`
  - `capacity_frontier.csv`, `capacity_frontier.md`
  - `kernel_phase_metrics.csv`, `kernel_phase_metrics.jsonl`
- MoE-specific output notes:
  - `decode_slo_summary.*` and `capacity_frontier.*` include `routing_skew`, `num_experts`, `top_k`
  - `remote_dispatch_ratio_p50` in `metrics.csv` quantifies cross-rank expert traffic
- fault-tolerant sweep diagnostics:
  - `runtime_failures.jsonl` (written when enabled and failures occur)

## 8) Public Repo Policy

- Generated results and plots are not committed.
- Keep only source code, configs, and documentation in version control.

## 9) Trn2 Repro Script

For the exact command chain used in this project:

```bash
bash scripts/trn2_repro_inference_track.sh
```

Command reference:

- `docs/TRN2_EXECUTED_COMMANDS.md`
