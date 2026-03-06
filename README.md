# io-aware-attention-trainium

IO-aware attention benchmarking and dual-die emulation experiments for AWS Trainium2.

## Overview

This repository contains:

- A benchmark harness for single-rank and distributed attention-related kernels.
- A 5-kernel study (`qkv_proj`, `attention`, `mlp`, `rmsnorm`, `out_proj`) with single vs dual-rank setups.
- A phase-aware study for prefill/decode behavior, including request/KV sharding and break-even analysis.
- A MoE service-scale decode study (router dispatch + expert MLP) with naive vs locality-aware dual-die routing.
- Plotting and what-if modeling scripts for communication/overlap sensitivity.

## Repository Layout

```text
src/io_aware_attention/
  bench/          # generic benchmark runner + artifact writing
  kernels/        # kernel implementations (naive, tiled, distributed-merge)
  experiments/    # kernel_study.py and phase_study.py
  runtime/        # Trainium runtime/env helpers
configs/
  benchmark/      # base benchmark configs
  experiments/    # kernel/phase study configs (smoke, strict, full)
scripts/
  run_*.py        # experiment entrypoints
  plot_*.py       # visualization
  what_if_dual_die.py
docs/
  TRAINIUM_SELF_HOSTED.md
  AWS_CHIP_ADVICE_MEMO.md
```

## Quick Start (CPU Smoke)

```bash
conda env create -f conda/environment.cpu.yml
conda activate ioattn-trn2
python scripts/run_bench.py --config configs/benchmark/smoke.yaml --variant naive --device cpu
```

## Trainium Host Setup

```bash
bash scripts/bootstrap_trainium_host.sh
conda activate ioattn-trn2
python scripts/validate_trainium_env.py
```

Detailed host runbook: `docs/TRAINIUM_SELF_HOSTED.md`

## Core Experiments

5-kernel dual-rank study:

```bash
torchrun --nproc_per_node=2 scripts/run_kernel_study.py \
  --config configs/experiments/trn2_kernel_study.yaml \
  --device trainium \
  --distributed
```

5-kernel dual-rank study with grouped distributed-merge attention (reduced collective count):

```bash
torchrun --nproc_per_node=2 scripts/run_kernel_study.py \
  --config configs/experiments/trn2_kernel_inference_optimized.yaml \
  --device trainium \
  --distributed
```

Phase-aware prefill/decode study:

```bash
torchrun --nproc_per_node=2 scripts/run_phase_study.py \
  --config configs/experiments/trn2_phase_study.yaml \
  --device trainium \
  --distributed
```

Inference-focused phase study (recommended for dual-die serving narrative):

```bash
torchrun --nproc_per_node=2 scripts/run_phase_study.py \
  --config configs/experiments/trn2_inference_story.yaml \
  --device trainium \
  --distributed
```

Service-scale day-1 matrix (throughput/capacity headline):

```bash
torchrun --nproc_per_node=2 scripts/run_phase_study.py \
  --config configs/experiments/trn2_inference_service_day1.yaml \
  --device trainium \
  --distributed
```

MoE service-scale decode validation (router dispatch + expert MLP + decode SLO):

```bash
torchrun --nproc_per_node=2 scripts/run_moe_service_study.py \
  --config configs/experiments/trn2_moe_service_day1.yaml \
  --device trainium \
  --distributed
```

Trainium-stable MoE sweep (recommended on flaky hosts):

```bash
NEURON_RT_VISIBLE_CORES=2,3 torchrun --nproc_per_node=2 scripts/run_moe_service_study.py \
  --config configs/experiments/trn2_moe_service_trainium_focus.yaml \
  --device trainium \
  --distributed
```

MoE communication-sensitive validation (smaller expert MLP to stress dispatch/collective effects):

```bash
torchrun --nproc_per_node=2 scripts/run_moe_service_study.py \
  --config configs/experiments/moe_comm_sensitive_cpu.yaml \
  --device cpu \
  --distributed
```

Plot results:

```bash
python scripts/plot_kernel_study.py --metrics-csv <run_dir>/metrics.csv --out-dir results/plots --prefix <name>
python scripts/plot_phase_study.py --metrics-csv <run_dir>/metrics.csv --kernel-phase-csv <run_dir>/kernel_phase_metrics.csv --out-dir results/plots --prefix <name>
```

Curated inference plot set (with optional stale-plot purge):

```bash
python scripts/plot_inference_track.py \
  --metrics-csv <run_dir>/metrics.csv \
  --decode-slo-csv <run_dir>/decode_slo_summary.csv \
  --break-even-csv <run_dir>/break_even_summary.csv \
  --out-dir results/plots \
  --prefix inference_story \
  --purge-stale
```

Capacity frontier plots:

```bash
python scripts/plot_capacity_frontier.py \
  --metrics-csv <run_dir>/metrics.csv \
  --capacity-csv <run_dir>/capacity_frontier.csv \
  --out-dir results/plots \
  --prefix trn2_service_day1
```

Mixed-traffic service simulation:

```bash
python scripts/simulate_mixed_traffic.py \
  --metrics-csv <run_dir>/metrics.csv \
  --out-dir results/plots \
  --prefix trn2_service_day1
```

MoE service plots:

```bash
python scripts/plot_moe_service_study.py \
  --metrics-csv <run_dir>/metrics.csv \
  --decode-slo-csv <run_dir>/decode_slo_summary.csv \
  --capacity-csv <run_dir>/capacity_frontier.csv \
  --out-dir results/plots \
  --prefix trn2_moe_day1 \
  --purge-stale
```

MoE summary table export:

```bash
python scripts/summarize_moe_service.py \
  --metrics-csv <run_dir>/metrics.csv \
  --decode-slo-csv <run_dir>/decode_slo_summary.csv \
  --out-dir results/plots \
  --prefix trn2_moe_day1
```

One-command Trn2 reproduction script:

```bash
bash scripts/trn2_repro_inference_track.sh
```

Break-even what-if sweep:

```bash
python scripts/what_if_dual_die.py \
  --metrics-csv <run_dir>/metrics.csv \
  --collectives-json <run_dir>/collectives_summary.json \
  --fabric-json <run_dir>/fabric_calibration.json \
  --out-dir results/plots \
  --prefix <name>
```

## Outputs

Generated artifacts are intentionally not committed. Each run writes a timestamped directory under `results/` with:

- `metrics.csv`, `metrics.jsonl`
- `run_manifest.json`
- optional fabric and collective summaries
- phase-study derived summaries (`break_even_summary.*`, `decode_slo_summary.*`)
- MoE summaries (`decode_slo_summary.*`, `capacity_frontier.*`) scoped by `routing_skew`, `num_experts`, and `top_k`
- service-capacity artifacts (`capacity_frontier.*`)
- runtime skip logs for fault-tolerant sweeps (`runtime_failures.jsonl`, when enabled and needed)

### Interpretation Note

For this project, the primary claim is **throughput/capacity-first dual-die value**:

- `dual_die_request_sharded` is expected to help decode service throughput/capacity.
- `dual_die_tensor_optimized` remains an explicit communication-bound counterexample.

## Testing

```bash
ruff check .
pytest -q
```

## Documentation

- Contribution guide: `CONTRIBUTING.md`
- Experiment guide: `docs/EXPERIMENT_GUIDE.md`
- Experiment validation flow: `docs/EXPERIMENT_VALIDATION_DIAGRAM.md`
- Trn2 executed commands: `docs/TRN2_EXECUTED_COMMANDS.md`
- Dual-die architecture diagrams: `docs/DUAL_DIE_COMPUTE_DIAGRAMS.md`
- Trainium runbook: `docs/TRAINIUM_SELF_HOSTED.md`
- AWS chip recommendation memo: `docs/AWS_CHIP_ADVICE_MEMO.md`
- MoE plot curation notes: `docs/MOE_RESULTS_CURATION.md`
