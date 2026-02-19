# io-aware-attention-trainium

IO-aware FlashAttention experiments targeting self-hosted AWS Trainium (Trn2 single-chip baseline).

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
- Validation:
  - `python scripts/validate_trainium_env.py`
- Optional S3 sync:
  - `python scripts/sync_results_s3.py --run-dir <results/run_id> --s3-uri <s3://bucket/prefix>`

## Artifact schema

Each run directory under `results/` contains:

- `metrics.csv`
- `metrics.jsonl`
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
