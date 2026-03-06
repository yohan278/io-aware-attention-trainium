#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DEVICE="${DEVICE:-trainium}"
NPROC="${NPROC:-2}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/results}"
PLOTS_DIR="${PLOTS_DIR:-${OUT_ROOT}/plots}"
RUN_FULL_PHASE="${RUN_FULL_PHASE:-0}"
RUN_SERVICE_DAY1="${RUN_SERVICE_DAY1:-0}"

mkdir -p "${OUT_ROOT}" "${PLOTS_DIR}"

echo "[repro] validate environment"
python scripts/validate_trainium_env.py

echo "[repro] kernel study (inference optimized)"
torchrun --nproc_per_node="${NPROC}" scripts/run_kernel_study.py \
  --config configs/experiments/trn2_kernel_inference_optimized.yaml \
  --device "${DEVICE}" \
  --distributed \
  --output-dir "${OUT_ROOT}/trn2-kernel-inference-optimized"

KERNEL_RUN_DIR="$(ls -dt "${OUT_ROOT}/trn2-kernel-inference-optimized"/run_* | head -n1)"
echo "[repro] kernel run dir: ${KERNEL_RUN_DIR}"

python scripts/plot_kernel_study.py \
  --metrics-csv "${KERNEL_RUN_DIR}/metrics.csv" \
  --out-dir "${PLOTS_DIR}" \
  --prefix trn2_inference_kernel

python scripts/what_if_dual_die.py \
  --metrics-csv "${KERNEL_RUN_DIR}/metrics.csv" \
  --collectives-json "${KERNEL_RUN_DIR}/collectives_summary.json" \
  --fabric-json "${KERNEL_RUN_DIR}/fabric_calibration.json" \
  --out-dir "${PLOTS_DIR}" \
  --prefix trn2_inference_kernel

echo "[repro] phase study (quick inference story)"
torchrun --nproc_per_node="${NPROC}" scripts/run_phase_study.py \
  --config configs/experiments/trn2_inference_story_quick.yaml \
  --device "${DEVICE}" \
  --distributed \
  --output-dir "${OUT_ROOT}/trn2-phase-inference-quick"

PHASE_QUICK_DIR="$(ls -dt "${OUT_ROOT}/trn2-phase-inference-quick"/run_* | head -n1)"
echo "[repro] phase quick run dir: ${PHASE_QUICK_DIR}"

python scripts/plot_phase_study.py \
  --metrics-csv "${PHASE_QUICK_DIR}/metrics.csv" \
  --kernel-phase-csv "${PHASE_QUICK_DIR}/kernel_phase_metrics.csv" \
  --decode-slo-csv "${PHASE_QUICK_DIR}/decode_slo_summary.csv" \
  --break-even-csv "${PHASE_QUICK_DIR}/break_even_summary.csv" \
  --out-dir "${PLOTS_DIR}" \
  --prefix trn2_inference_quick

python scripts/plot_inference_track.py \
  --metrics-csv "${PHASE_QUICK_DIR}/metrics.csv" \
  --decode-slo-csv "${PHASE_QUICK_DIR}/decode_slo_summary.csv" \
  --break-even-csv "${PHASE_QUICK_DIR}/break_even_summary.csv" \
  --out-dir "${PLOTS_DIR}" \
  --prefix trn2_inference_quick \
  --purge-stale

if [[ "${RUN_FULL_PHASE}" == "1" ]]; then
  echo "[repro] phase study (full inference story)"
  torchrun --nproc_per_node="${NPROC}" scripts/run_phase_study.py \
    --config configs/experiments/trn2_inference_story.yaml \
    --device "${DEVICE}" \
    --distributed \
    --output-dir "${OUT_ROOT}/trn2-phase-inference-full"

  PHASE_FULL_DIR="$(ls -dt "${OUT_ROOT}/trn2-phase-inference-full"/run_* | head -n1)"
  echo "[repro] phase full run dir: ${PHASE_FULL_DIR}"

  python scripts/plot_phase_study.py \
    --metrics-csv "${PHASE_FULL_DIR}/metrics.csv" \
    --kernel-phase-csv "${PHASE_FULL_DIR}/kernel_phase_metrics.csv" \
    --decode-slo-csv "${PHASE_FULL_DIR}/decode_slo_summary.csv" \
    --break-even-csv "${PHASE_FULL_DIR}/break_even_summary.csv" \
    --out-dir "${PLOTS_DIR}" \
    --prefix trn2_inference_full

  python scripts/plot_inference_track.py \
    --metrics-csv "${PHASE_FULL_DIR}/metrics.csv" \
    --decode-slo-csv "${PHASE_FULL_DIR}/decode_slo_summary.csv" \
    --break-even-csv "${PHASE_FULL_DIR}/break_even_summary.csv" \
    --out-dir "${PLOTS_DIR}" \
    --prefix trn2_inference_full \
    --purge-stale
fi

if [[ "${RUN_SERVICE_DAY1}" == "1" ]]; then
  echo "[repro] phase study (service-scale day1)"
  torchrun --nproc_per_node="${NPROC}" scripts/run_phase_study.py \
    --config configs/experiments/trn2_inference_service_day1.yaml \
    --device "${DEVICE}" \
    --distributed \
    --output-dir "${OUT_ROOT}/trn2-phase-inference-service-day1"

  PHASE_DAY1_DIR="$(ls -dt "${OUT_ROOT}/trn2-phase-inference-service-day1"/run_* | head -n1)"
  echo "[repro] phase day1 run dir: ${PHASE_DAY1_DIR}"

  python scripts/plot_phase_study.py \
    --metrics-csv "${PHASE_DAY1_DIR}/metrics.csv" \
    --kernel-phase-csv "${PHASE_DAY1_DIR}/kernel_phase_metrics.csv" \
    --decode-slo-csv "${PHASE_DAY1_DIR}/decode_slo_summary.csv" \
    --break-even-csv "${PHASE_DAY1_DIR}/break_even_summary.csv" \
    --out-dir "${PLOTS_DIR}" \
    --prefix trn2_service_day1

  python scripts/plot_inference_track.py \
    --metrics-csv "${PHASE_DAY1_DIR}/metrics.csv" \
    --decode-slo-csv "${PHASE_DAY1_DIR}/decode_slo_summary.csv" \
    --break-even-csv "${PHASE_DAY1_DIR}/break_even_summary.csv" \
    --out-dir "${PLOTS_DIR}" \
    --prefix trn2_service_day1

  python scripts/plot_capacity_frontier.py \
    --metrics-csv "${PHASE_DAY1_DIR}/metrics.csv" \
    --capacity-csv "${PHASE_DAY1_DIR}/capacity_frontier.csv" \
    --out-dir "${PLOTS_DIR}" \
    --prefix trn2_service_day1

  python scripts/simulate_mixed_traffic.py \
    --metrics-csv "${PHASE_DAY1_DIR}/metrics.csv" \
    --out-dir "${PLOTS_DIR}" \
    --prefix trn2_service_day1 \
    --seed 123 \
    --duration-s 120 \
    --arrival-rate-rps 12 \
    --prefill-ratio 0.30 \
    --decode-slo-ms 250 \
    --drop-wait-ms 2000 \
    --decode-tokens 64 \
    --context-weights "2048:0.5,4096:0.35,8192:0.15"
fi

echo "[repro] complete"
echo "  kernel: ${KERNEL_RUN_DIR}"
echo "  quick phase: ${PHASE_QUICK_DIR}"
if [[ "${RUN_SERVICE_DAY1}" == "1" ]]; then
  echo "  service day1: ${PHASE_DAY1_DIR}"
fi
