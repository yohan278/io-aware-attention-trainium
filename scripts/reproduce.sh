#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_MOE=0

for arg in "$@"; do
  case "$arg" in
    --skip-moe) SKIP_MOE=1 ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: bash scripts/reproduce.sh [--skip-moe]" >&2
      exit 2
      ;;
  esac
done

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

PHASE_RUN="results/trn2-phase-inference-quick-fast/run_20260305T224828Z"
KERNEL_RUN="results/trn2-kernel-inference-optimized/run_20260305T221035Z"
MOE_RUN="results/trn2-moe-stable-small-merged-mask23/run_20260306T100500Z"
DIRECT_RUN="results/trn2_direct_policy_trace/run_20260310T044926Z"
DIRECT_DENSE_RUN="results/trn2_direct_policy_trace_dense/run_20260310T061755Z"

require_file "$PHASE_RUN/metrics.csv"
require_file "$PHASE_RUN/decode_slo_summary.csv"
require_file "$PHASE_RUN/collectives_summary.json"
require_file "$PHASE_RUN/break_even_summary.csv"
require_file "$KERNEL_RUN/metrics.csv"
require_file "$KERNEL_RUN/collectives_summary.json"
require_file "$DIRECT_RUN/direct_policy_trace.png"
require_file "$DIRECT_RUN/direct_policy_trace_summary.csv"
require_file "$DIRECT_DENSE_RUN/direct_policy_trace_summary.csv"
require_file "$DIRECT_DENSE_RUN/direct_policy_trace_samples.json"

if [[ "$SKIP_MOE" -eq 0 ]]; then
  require_file "$MOE_RUN/metrics.csv"
  require_file "$MOE_RUN/decode_slo_summary.csv"
  require_file "$MOE_RUN/capacity_frontier.csv"
fi

mkdir -p results/plots

echo "[1/4] Regenerate core service figures"
"$PYTHON_BIN" scripts/plot_best_graphs.py \
  --phase-metrics-csv "$PHASE_RUN/metrics.csv" \
  --decode-slo-csv "$PHASE_RUN/decode_slo_summary.csv" \
  --phase-collectives-json "$PHASE_RUN/collectives_summary.json" \
  --kernel-metrics-csv "$KERNEL_RUN/metrics.csv" \
  --kernel-collectives-json "$KERNEL_RUN/collectives_summary.json" \
  --out-dir results/plots \
  --prefix public_service

echo "[2/4] Regenerate mixed-traffic policy simulation"
"$PYTHON_BIN" scripts/simulate_mixed_traffic.py \
  --metrics-csv "$PHASE_RUN/metrics.csv" \
  --out-dir results/plots \
  --prefix public_service

echo "[3/4] Refresh direct-trace and dense sharded-serving analyses"
cp "$DIRECT_RUN/direct_policy_trace.png" results/plots/public_service_direct_policy_trace.png

"$PYTHON_BIN" scripts/plot_direct_trace_dense_points.py \
  --summary-csv "$DIRECT_DENSE_RUN/direct_policy_trace_summary.csv" \
  --samples-json "$DIRECT_DENSE_RUN/direct_policy_trace_samples.json" \
  --out-path results/plots/public_service_dual_dense_points.png \
  --points-csv results/plots/public_service_dual_dense_points.csv

"$PYTHON_BIN" scripts/analyze_sharded_serving_dense.py \
  --summary-csv "$DIRECT_DENSE_RUN/direct_policy_trace_summary.csv" \
  --samples-json "$DIRECT_DENSE_RUN/direct_policy_trace_samples.json" \
  --request-slo-ms 500 \
  --output-tokens 128 \
  --duration-s 180 \
  --trials 120 \
  --arrival-rates 6,7,8,9,10,11,12,13,14,15,16 \
  --out-plot results/plots/public_service_sharded_dense_analysis.png \
  --out-csv results/plots/public_service_sharded_dense_queue.csv \
  --out-md results/plots/public_service_sharded_dense_analysis.md

if [[ "$SKIP_MOE" -eq 0 ]]; then
  echo "[4/4] Regenerate MoE figures"
  "$PYTHON_BIN" scripts/plot_moe_service_study.py \
    --metrics-csv "$MOE_RUN/metrics.csv" \
    --decode-slo-csv "$MOE_RUN/decode_slo_summary.csv" \
    --capacity-csv "$MOE_RUN/capacity_frontier.csv" \
    --out-dir results/plots \
    --prefix public_moe_mask23
else
  echo "[4/4] Skipping MoE figure regeneration (--skip-moe)"
fi

echo "Reproduction complete."
echo "Core outputs:"
echo "  - results/plots/public_service_decode_slo_frontier.png"
echo "  - results/plots/public_service_direct_policy_trace.png"
echo "  - results/plots/public_service_dual_dense_points.png"
echo "  - results/plots/public_service_sharded_dense_analysis.png"
echo "  - results/plots/public_service_mixed_trace_goodput.png"
