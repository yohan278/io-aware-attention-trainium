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

Service-scale day-1 variant (throughput/capacity focus):

```bash
torchrun --nproc_per_node=2 scripts/run_phase_study.py \
  --config configs/experiments/trn2_inference_service_day1.yaml \
  --device trainium \
  --distributed
```

## 3) Plotting

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

## 4) What-If Dual-Die Model

```bash
python scripts/what_if_dual_die.py \
  --metrics-csv <run_dir>/metrics.csv \
  --collectives-json <run_dir>/collectives_summary.json \
  --fabric-json <run_dir>/fabric_calibration.json \
  --out-dir results/plots \
  --prefix <name>
```

## 5) Curated Inference Plot Set

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

## 6) Output Files

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
- fault-tolerant sweep diagnostics:
  - `runtime_failures.jsonl` (written when enabled and failures occur)

## 7) Public Repo Policy

- Generated results and plots are not committed.
- Keep only source code, configs, and documentation in version control.

## 8) Trn2 Repro Script

For the exact command chain used in this project:

```bash
bash scripts/trn2_repro_inference_track.sh
```

Command reference:

- `docs/TRN2_EXECUTED_COMMANDS.md`
