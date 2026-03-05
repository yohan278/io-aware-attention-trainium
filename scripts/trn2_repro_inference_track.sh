#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DEVICE="${DEVICE:-trainium}"
NPROC="${NPROC:-2}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/results}"
PLOTS_DIR="${PLOTS_DIR:-${OUT_ROOT}/plots}"
RUN_FULL_PHASE="${RUN_FULL_PHASE:-0}"

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

echo "[repro] complete"
echo "  kernel: ${KERNEL_RUN_DIR}"
echo "  quick phase: ${PHASE_QUICK_DIR}"
