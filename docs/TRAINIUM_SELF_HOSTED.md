# Trainium Self-Hosted Runbook (Trn2 Single Chip)

This guide defines the default push -> pull -> run workflow for this repository.

## 0) Cloud prerequisites (CS149-compatible)

- AWS region: `ap-southeast-4` (Asia Pacific - Melbourne).
- Instance type: `trn2.3xlarge`.
- Login user: `ubuntu`.
- Key permissions on local machine:

```bash
chmod 400 /path/to/key_name.pem
```

- SSH with profiler port forwarding:

```bash
ssh -i /path/to/key_name.pem ubuntu@<public_dns_name> -L 3001:localhost:3001 -L 8086:localhost:8086
```

Those forwarded ports match the standard `neuron-profile` and InfluxDB defaults.

## 1) One-time host setup

```bash
git clone <your-repo-url>
cd io-aware-attention-trainium
bash scripts/bootstrap_trainium_host.sh
```

What bootstrap does:

- creates/reuses conda env `ioattn-trn2` (or `CONDA_ENV_NAME`)
- updates the env from `conda/environment.trainium.yml`
- upgrades `pip/setuptools/wheel` inside the conda env
- installs this project in editable mode (`pip install -e .`)
- runs `scripts/validate_trainium_env.py`

If conda is missing, bootstrap auto-installs Miniconda to `$HOME/miniconda3` by default.

Manual equivalent:

```bash
conda env create -f conda/environment.trainium.yml || conda env update -f conda/environment.trainium.yml --prune
conda activate ioattn-trn2
python -m pip install -e .
python scripts/validate_trainium_env.py
```

## 2) Daily workflow

On local machine:

```bash
git add .
git commit -m "your change"
git push
```

On Trn2 host:

```bash
cd io-aware-attention-trainium
git pull
conda activate ioattn-trn2
python scripts/validate_trainium_env.py
python scripts/run_bench.py --config configs/benchmark/canonical.yaml --variant tiled_online --device trainium
```

## 3) Profiling run

```bash
conda activate ioattn-trn2
python scripts/profile_trainium.py --config configs/benchmark/canonical.yaml --variant tiled_online --set-neuron-profile-env
```

The command writes regular benchmark artifacts while enabling profiling environment hints.

## 4) Results

Every run writes to `results/<run_id>/`:

- `metrics.csv`
- `metrics.jsonl`
- `run_manifest.json`

Optional S3 archival:

```bash
python scripts/sync_results_s3.py --run-dir results/<run_id> --s3-uri s3://your-bucket/flashattention-runs
```

## 5) Troubleshooting

- Validation says Trainium runtime missing:
  - Ensure you are on a Neuron-enabled host/AMI.
  - Re-run `bash scripts/bootstrap_trainium_host.sh`.
  - Confirm the conda env is active: `conda activate ioattn-trn2`.
- Benchmark fails on `trainium` device:
  - Run `python scripts/validate_trainium_env.py` first.
  - Confirm `conda/environment.trainium.yml` and `requirements/trainium.txt` align with your Neuron SDK/AMI.
- S3 sync fails:
  - Check host IAM permissions for `s3:PutObject`.
  - Local artifacts remain intact even when uploads fail.
- Cannot SSH with forwarded ports:
  - Verify the instance security group allows SSH ingress from your IP.
  - Reconnect using the exact `-L 3001` and `-L 8086` flags shown above.
