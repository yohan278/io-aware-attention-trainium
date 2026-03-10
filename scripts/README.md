# Scripts Directory

Executable entrypoints for experiments, plotting, and artifact reproduction.

## Layout

```
scripts/
  reproduce.sh                       End-to-end artifact regeneration
  run_*.py                           Experiment runners (bench, kernel, phase, MoE, direct trace)
  plot_*.py                          Figure generators for kernel/phase/service/MoE tracks
  simulate_mixed_traffic.py          Queue-level mixed workload simulator
  summarize_moe_service.py           MoE summary table export
  generate_paper_assets.py           Paper-only tables/figures
  analyze_sharded_serving_dense.py   Dense queue analysis from direct-trace samples
  plot_direct_trace_dense_points.py  Dense per-request sample visualization
  bootstrap_trainium_host.sh         Host bootstrap helper for Trn2
  validate_trainium_env.py           Runtime/toolchain validation
```

## One-Command Reproduction

Regenerates the committed public figures and paper-local copies from committed run artifacts:

```bash
bash scripts/reproduce.sh
```

Optional flags:

- `--skip-moe`: skip MoE plot regeneration.
- `--skip-paper-copy`: keep outputs in `results/plots/` only.

## Core Experiment Runners

### Kernel study

```bash
torchrun --nproc_per_node=2 scripts/run_kernel_study.py \
  --config configs/experiments/trn2_kernel_inference_optimized.yaml \
  --distributed
```

### Phase study

```bash
torchrun --nproc_per_node=2 scripts/run_phase_study.py \
  --config configs/experiments/trn2_phase_study.yaml \
  --distributed
```

### Direct end-to-end trace

```bash
torchrun --nproc_per_node=2 scripts/run_direct_policy_trace.py \
  --config configs/experiments/trn2_direct_policy_trace.yaml \
  --distributed
```

### Dense direct trace (many points)

```bash
torchrun --nproc_per_node=2 scripts/run_direct_policy_trace.py \
  --config configs/experiments/trn2_direct_policy_trace_dense.yaml \
  --distributed
```

## Plot and Analysis Helpers

### Public service bundle

```bash
python3 scripts/plot_best_graphs.py \
  --phase-metrics-csv results/trn2-phase-inference-quick-fast/run_20260305T224828Z/metrics.csv \
  --phase-collectives-json results/trn2-phase-inference-quick-fast/run_20260305T224828Z/collectives_summary.json \
  --kernel-metrics-csv results/trn2-kernel-inference-optimized/run_20260305T221035Z/metrics.csv \
  --kernel-collectives-json results/trn2-kernel-inference-optimized/run_20260305T221035Z/collectives_summary.json \
  --out-dir results/plots \
  --prefix public_service
```

### Dense sharded-serving analysis

```bash
python3 scripts/analyze_sharded_serving_dense.py \
  --summary-csv results/trn2_direct_policy_trace_dense/run_20260310T061755Z/direct_policy_trace_summary.csv \
  --samples-json results/trn2_direct_policy_trace_dense/run_20260310T061755Z/direct_policy_trace_samples.json
```
