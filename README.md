# io-aware-attention-trainium

IO-aware FlashAttention experiments targeting self-hosted AWS Trainium (Trn2 single-chip baseline).

This repository is compatible with the CS149-style AWS setup (`ap-southeast-4`, `trn2.3xlarge`, Ubuntu SSH workflow with forwarded profiler ports).

## What this repository provides

- Forward-only SDPA benchmark harness (`naive`, `tiled_online`, and a placeholder `tiled_online_dbuffer` variant).
- Reproducible run artifacts (`metrics.csv`, `metrics.jsonl`, `run_manifest.json`).
- Trainium host bootstrap and environment validation scripts.
- Optional result upload to S3.

## Quickstart (local CPU smoke, conda)

```bash
conda env create -f conda/environment.cpu.yml
conda activate ioattn-trn2
python scripts/run_bench.py --config configs/benchmark/smoke.yaml --variant naive --device cpu
```

If the env already exists:

```bash
conda env update -f conda/environment.cpu.yml --prune
conda activate ioattn-trn2
```

## Self-hosted Trn2 workflow

1. Push your code changes to GitHub.
2. SSH into Trn2 host and pull latest commit.
3. Run bootstrap (conda env update + validation).
4. Run benchmark/profiling command.

```bash
git pull
bash scripts/bootstrap_trainium_host.sh
conda activate ioattn-trn2
python scripts/validate_trainium_env.py
python scripts/run_bench.py --config configs/benchmark/canonical.yaml --variant tiled_online --device trainium
```

See full host instructions in `docs/TRAINIUM_SELF_HOSTED.md`.

## CS149 cloud compatibility notes

- Region default in env template is `ap-southeast-4` (Melbourne).
- Assumed instance type is `trn2.3xlarge`.
- Assumed SSH login user is `ubuntu`.
- For profiler UI access, connect with:

```bash
ssh -i /path/to/key.pem ubuntu@<public_dns_name> -L 3001:localhost:3001 -L 8086:localhost:8086
```

## Conda environment files

- CPU/local: `conda/environment.cpu.yml`
- Trainium host: `conda/environment.trainium.yml`

You can customize env name with `CONDA_ENV_NAME`, for example:

```bash
CONDA_ENV_NAME=fa-trn2 bash scripts/bootstrap_trainium_host.sh
```

## Main interfaces

- Benchmark:
  - `python scripts/run_bench.py --config <yaml> --variant <naive|tiled_online> --device trainium --output-dir <path>`
- 5-kernel scaling study (single-die vs real dual-rank collectives on Trn2):
  - `torchrun --nproc_per_node=2 scripts/run_kernel_study.py --config configs/experiments/trn2_kernel_study.yaml --device trainium --distributed`
- Plot kernel-study outputs:
  - `python scripts/plot_kernel_study.py --metrics-csv <run_dir>/metrics.csv --out-dir results/plots --prefix <name>`
- Validation:
  - `python scripts/validate_trainium_env.py`
- Optional S3 sync:
  - `python scripts/sync_results_s3.py --run-dir <results/run_id> --s3-uri <s3://bucket/prefix>`

## Full experiment: Trn2 single-die vs dual-rank collectives

### Objective

Evaluate five common transformer kernels on AWS Trainium2 and compare:

1. Single-die baseline.
2. Naive dual-die partitioning.
3. Communication-optimized dual-die partitioning.

### Hardware/runtime

- Instance: `trn2.3xlarge`
- Region used: `ap-southeast-4`
- Runtime: `torch-neuronx` + `torch_xla` with `torchrun --nproc_per_node=2`

### Kernel set and shapes

- Kernels: `qkv_proj`, `attention`, `mlp`, `rmsnorm`, `out_proj`
- Shapes in full config:
  - `(batch=1, seq_len=512, model_dim=1024, num_heads=16, mlp_ratio=4)`
  - `(batch=1, seq_len=1024, model_dim=1024, num_heads=16, mlp_ratio=4)`

### Dual-rank implementation details

- Dual paths run as real 2-rank execution over XLA collectives (`all_reduce`, `all_gather`) instead of byte-only emulation.
- Dual setup names:
  - `dual_die_naive`: straightforward partition + heavier collective traffic.
  - `dual_die_optimized`: reduced communication volume via kernel-specific partition/reduction strategy.

### Fabric calibration

Before kernels run, the harness calibrates collectives using:

- `ping_pong`-style roundtrip broadcasts
- `all_reduce`
- `all_gather`

Outputs:

- `fabric_calibration.csv`
- `fabric_calibration.json`

The run on **March 3, 2026** measured a peak calibrated fabric throughput of approximately **1.508537 GB/s**.

### Correctness gate

- Correctness is checked against single-die output per kernel/shape.
- Recorded metrics: `max_abs_err`, robust `max_rel_err` (numerically stable denominator).
- Gate fails only when both thresholds are exceeded:
  - `correctness_abs_tol: 0.05`
  - `correctness_rel_tol: 0.1`

### Key measured metrics

- End-to-end latency: `latency_ms_p50`, `latency_ms_p90`
- Decomposed timing: `compute_ms_p50`, `communication_ms_p50`, `overlap_pct_p50`
- Throughput: `throughput_tokens_per_s`
- Communication volume: `communication_bytes`, `communication_pct_of_hbm`
- Fabric use: `achieved_link_gbps_p50`, `fabric_peak_gbps`, `link_utilization_pct_p50`

### Reproduce full run

```bash
torchrun --nproc_per_node=2 scripts/run_kernel_study.py \
  --config configs/experiments/trn2_kernel_study.yaml \
  --device trainium \
  --distributed \
  --output-dir results/trn2-dual-real
```

```bash
python scripts/plot_kernel_study.py \
  --metrics-csv results/trn2-dual-real/<run_id>/metrics.csv \
  --out-dir results/plots \
  --prefix trn2_dual_real_final
```

### Latest full run summary

Full run id: `run_20260303T221727Z` (executed March 3, 2026).

- `dual_die_naive` median slowdown vs single-die across all kernel/shape points: ~`6.27x`
- `dual_die_optimized` median slowdown vs single-die: ~`5.42x`
- Optimized mode substantially reduced communication bytes for `attention`, `qkv_proj`, and `mlp`, but did not beat single-die latency in this setup.

Interpretation note:

- Some `link_utilization_pct_p50` values can exceed 100% because kernel communication payload patterns differ from calibration microbenchmarks. Treat this as a comparative indicator across setups, not a hard physical utilization bound.

### Committed result plots

![Trn2 dual-real latency](results/plots/trn2_dual_real_final_latency.png)
![Trn2 dual-real throughput](results/plots/trn2_dual_real_final_throughput.png)
![Trn2 dual-real speedup](results/plots/trn2_dual_real_final_speedup.png)

## Artifact schema

Each run directory under `results/` contains:

- `metrics.csv`
- `metrics.jsonl`
- `fabric_calibration.csv` and `fabric_calibration.json` (when distributed calibration is enabled)
- `run_manifest.json` with:
  - `git_commit`
  - `timestamp_utc`
  - `instance_type`
  - `device_target`
  - `torch_version`
  - `torch_neuronx_version`
  - `python_version`
  - `benchmark_config_path`
  - `variant`
  - `seed`

`metrics.csv` includes p50 compute time, p50 communication time, overlap percentage, achieved link bandwidth, and link utilization against calibrated fabric peak.
