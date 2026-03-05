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

## 4) What-If Dual-Die Model

```bash
python scripts/what_if_dual_die.py \
  --metrics-csv <run_dir>/metrics.csv \
  --collectives-json <run_dir>/collectives_summary.json \
  --fabric-json <run_dir>/fabric_calibration.json \
  --out-dir results/plots \
  --prefix <name>
```

## 5) Output Files

Each run directory (`results/<run_id>/`) may include:

- `metrics.csv`, `metrics.jsonl`
- `run_manifest.json`
- `collectives_summary.json`
- `fabric_calibration.csv`, `fabric_calibration.json`
- phase-specific summaries:
  - `break_even_summary.csv`, `break_even_summary.md`
  - `decode_slo_summary.csv`, `decode_slo_summary.md`
  - `kernel_phase_metrics.csv`, `kernel_phase_metrics.jsonl`

## 6) Public Repo Policy

- Generated results and plots are not committed.
- Keep only source code, configs, and documentation in version control.
