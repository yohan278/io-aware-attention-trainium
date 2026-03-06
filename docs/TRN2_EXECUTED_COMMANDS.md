# Trn2 Executed Commands

This is the command set used for Trn2 experiment runs and plot generation in this project.

## One-command Reproduction

Run the full inference track with:

```bash
bash scripts/trn2_repro_inference_track.sh
```

Optional full (heavier) phase sweep:

```bash
RUN_FULL_PHASE=1 bash scripts/trn2_repro_inference_track.sh
```

Optional service-scale day-1 sweep:

```bash
RUN_SERVICE_DAY1=1 bash scripts/trn2_repro_inference_track.sh
```

Optional MoE service-scale day-1 sweep:

```bash
RUN_MOE_DAY1=1 bash scripts/trn2_repro_inference_track.sh
```

## Expanded Command List

Environment check:

```bash
python scripts/validate_trainium_env.py
```

Kernel study (dual-die inference-optimized attention merge):

```bash
torchrun --nproc_per_node=2 scripts/run_kernel_study.py \
  --config configs/experiments/trn2_kernel_inference_optimized.yaml \
  --device trainium \
  --distributed \
  --output-dir results/trn2-kernel-inference-optimized
```

Kernel plots + what-if analysis:

```bash
python scripts/plot_kernel_study.py \
  --metrics-csv results/trn2-kernel-inference-optimized/<run_id>/metrics.csv \
  --out-dir results/plots \
  --prefix trn2_inference_kernel

python scripts/what_if_dual_die.py \
  --metrics-csv results/trn2-kernel-inference-optimized/<run_id>/metrics.csv \
  --collectives-json results/trn2-kernel-inference-optimized/<run_id>/collectives_summary.json \
  --fabric-json results/trn2-kernel-inference-optimized/<run_id>/fabric_calibration.json \
  --out-dir results/plots \
  --prefix trn2_inference_kernel
```

Phase study (quick inference-focused):

```bash
torchrun --nproc_per_node=2 scripts/run_phase_study.py \
  --config configs/experiments/trn2_inference_story_quick.yaml \
  --device trainium \
  --distributed \
  --output-dir results/trn2-phase-inference-quick
```

Phase plots + curated inference plots:

```bash
python scripts/plot_phase_study.py \
  --metrics-csv results/trn2-phase-inference-quick/<run_id>/metrics.csv \
  --kernel-phase-csv results/trn2-phase-inference-quick/<run_id>/kernel_phase_metrics.csv \
  --decode-slo-csv results/trn2-phase-inference-quick/<run_id>/decode_slo_summary.csv \
  --break-even-csv results/trn2-phase-inference-quick/<run_id>/break_even_summary.csv \
  --out-dir results/plots \
  --prefix trn2_inference_quick

python scripts/plot_inference_track.py \
  --metrics-csv results/trn2-phase-inference-quick/<run_id>/metrics.csv \
  --decode-slo-csv results/trn2-phase-inference-quick/<run_id>/decode_slo_summary.csv \
  --break-even-csv results/trn2-phase-inference-quick/<run_id>/break_even_summary.csv \
  --out-dir results/plots \
  --prefix trn2_inference_quick \
  --purge-stale
```

Phase study (full inference-focused):

```bash
torchrun --nproc_per_node=2 scripts/run_phase_study.py \
  --config configs/experiments/trn2_inference_story.yaml \
  --device trainium \
  --distributed \
  --output-dir results/trn2-phase-inference-full
```

Phase study (service-scale day-1):

```bash
torchrun --nproc_per_node=2 scripts/run_phase_study.py \
  --config configs/experiments/trn2_inference_service_day1.yaml \
  --device trainium \
  --distributed \
  --output-dir results/trn2-phase-inference-service-day1
```

Service-scale post-processing:

```bash
python scripts/plot_capacity_frontier.py \
  --metrics-csv results/trn2-phase-inference-service-day1/<run_id>/metrics.csv \
  --capacity-csv results/trn2-phase-inference-service-day1/<run_id>/capacity_frontier.csv \
  --out-dir results/plots \
  --prefix trn2_service_day1

python scripts/simulate_mixed_traffic.py \
  --metrics-csv results/trn2-phase-inference-service-day1/<run_id>/metrics.csv \
  --out-dir results/plots \
  --prefix trn2_service_day1
```

MoE service study:

```bash
torchrun --nproc_per_node=2 scripts/run_moe_service_study.py \
  --config configs/experiments/trn2_moe_service_day1.yaml \
  --device trainium \
  --distributed \
  --output-dir results/trn2-moe-service-day1
```

MoE communication-sensitive variant (CPU distributed stress test):

```bash
torchrun --nproc_per_node=2 scripts/run_moe_service_study.py \
  --config configs/experiments/moe_comm_sensitive_cpu.yaml \
  --device cpu \
  --distributed \
  --output-dir results/trn2-moe-comm-sensitive-cpu
```

MoE post-processing:

```bash
python scripts/plot_moe_service_study.py \
  --metrics-csv results/trn2-moe-service-day1/<run_id>/metrics.csv \
  --decode-slo-csv results/trn2-moe-service-day1/<run_id>/decode_slo_summary.csv \
  --capacity-csv results/trn2-moe-service-day1/<run_id>/capacity_frontier.csv \
  --out-dir results/plots \
  --prefix trn2_moe_day1 \
  --purge-stale
```

MoE summary table export:

```bash
python scripts/summarize_moe_service.py \
  --metrics-csv results/trn2-moe-service-day1-cpu/<run_id>/metrics.csv \
  --decode-slo-csv results/trn2-moe-service-day1-cpu/<run_id>/decode_slo_summary.csv \
  --out-dir results/plots \
  --prefix trn2_moe_day1_cpu
```
