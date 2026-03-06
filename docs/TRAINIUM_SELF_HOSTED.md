# Trainium Self-Hosted Runbook

This guide describes a clean push -> pull -> run workflow on an AWS Trainium host.

## 1) Prerequisites

- A Neuron-enabled Trn2 instance.
- SSH access (`ubuntu` user is common on Ubuntu AMIs).
- A repository clone on the host.
- Optional: local SSH port forwarding for profiling UIs.

Example SSH with forwarded ports:

```bash
ssh -i /path/to/key.pem ubuntu@<public-dns> -L 3001:localhost:3001 -L 8086:localhost:8086
```

## 2) One-Time Host Bootstrap

```bash
git clone <repo-url>
cd io-aware-attention-trainium
bash scripts/bootstrap_trainium_host.sh
```

Bootstrap will:

- install/reuse Miniconda (if needed)
- create/update conda env (`ioattn-trn2` by default)
- install project dependencies
- install this repository in editable mode
- run environment validation

## 3) Daily Workflow

On local development machine:

```bash
git add .
git commit -m "your change"
git push
```

On Trainium host:

```bash
cd io-aware-attention-trainium
git pull
conda activate ioattn-trn2
python scripts/validate_trainium_env.py
```

Then run one or more experiments:

```bash
python scripts/run_bench.py --config configs/benchmark/canonical.yaml --variant tiled_online --device trainium
torchrun --nproc_per_node=2 scripts/run_kernel_study.py --config configs/experiments/trn2_kernel_study.yaml --device trainium --distributed
torchrun --nproc_per_node=2 scripts/run_phase_study.py --config configs/experiments/trn2_phase_study.yaml --device trainium --distributed
```

## 4) Profiling

```bash
conda activate ioattn-trn2
python scripts/profile_trainium.py --config configs/benchmark/canonical.yaml --variant tiled_online --set-neuron-profile-env
```

## 5) Artifacts

Each run writes `results/<run_id>/` with metrics and manifest files.  
Optional archival:

```bash
python scripts/sync_results_s3.py --run-dir results/<run_id> --s3-uri s3://your-bucket/path
```

## 6) Troubleshooting

- Trainium runtime missing:
  - verify Neuron-enabled AMI/runtime
  - re-run `bash scripts/bootstrap_trainium_host.sh`
- Validation fails:
  - check env activation (`conda activate ioattn-trn2`)
  - check `conda/environment.trainium.yml` compatibility with host runtime
- Runtime flaps with `NC4 init failed` / `NRT_FAILURE`:
  - test per-core health (`NEURON_RT_VISIBLE_CORES=0|1|2|3`) with a tiny XLA op
  - avoid bad cores by pinning two healthy cores, for example:
    - `NEURON_RT_VISIBLE_CORES=2,3 torchrun --nproc_per_node=2 ...`
  - for unstable nodes, disable fabric calibration in run config for MoE sweeps
- SSH/port-forward issues:
  - check security group ingress
  - confirm forwarded port flags are present
