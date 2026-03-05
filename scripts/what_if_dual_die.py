#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build dual-die break-even what-if curves from metrics, collectives, and fabric calibration."
    )
    parser.add_argument("--metrics-csv", type=Path, required=True, help="Path to metrics.csv.")
    parser.add_argument(
        "--collectives-json",
        type=Path,
        default=None,
        help="Path to collectives_summary.json (defaults to metrics sibling).",
    )
    parser.add_argument(
        "--fabric-json",
        type=Path,
        default=None,
        help="Path to fabric_calibration.json (defaults to metrics sibling).",
    )
    parser.add_argument(
        "--overlap-fracs",
        default="0,0.25,0.5,0.75,0.9",
        help="Comma-separated overlap fractions to sweep.",
    )
    parser.add_argument(
        "--target-overlap",
        type=float,
        default=0.5,
        help="Overlap fraction for required-bandwidth calculation.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for CSV/plots. Defaults to metrics parent.",
    )
    parser.add_argument("--prefix", default="dual_die_what_if", help="Output filename prefix.")
    return parser.parse_args()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _scenario_key_from_metric(row: dict[str, Any]) -> tuple[Any, ...]:
    phase = str(row.get("phase", "kernel"))
    kernel = str(row.get("kernel", "")) if row.get("kernel") is not None else ""
    return (
        phase,
        kernel,
        _safe_int(row.get("batch"), 0),
        _safe_int(row.get("seq_len"), 0),
        _safe_int(row.get("context_len"), 0),
        _safe_int(row.get("decode_steps"), 0),
        _safe_int(row.get("model_dim"), 0),
        _safe_int(row.get("num_heads"), 0),
    )


def _scenario_key_from_collective(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("phase", "kernel")),
        str(row.get("kernel", "")),
        _safe_int(row.get("batch"), 0),
        _safe_int(row.get("seq_len"), 0),
        _safe_int(row.get("context_len"), 0),
        _safe_int(row.get("decode_steps"), 0),
        _safe_int(row.get("model_dim"), 0),
        _safe_int(row.get("num_heads"), 0),
    )


def _op_family(op_name: str) -> str:
    if op_name.startswith("all_reduce"):
        return "all_reduce"
    if op_name.startswith("all_gather"):
        return "all_gather"
    if op_name.startswith("broadcast"):
        return "broadcast"
    return "all_reduce"


def _calibration_params(fabric_json: dict[str, Any]) -> tuple[dict[str, dict[str, float]], float]:
    collectives = fabric_json.get("collectives", {}) if isinstance(fabric_json, dict) else {}
    default = {"alpha_ms": 0.0, "beta_gbps": 1e-9}
    out: dict[str, dict[str, float]] = {
        "all_reduce": dict(default),
        "all_gather": dict(default),
        "broadcast": dict(default),
    }

    peak = _safe_float(fabric_json.get("peak_gbps"), 0.0) if isinstance(fabric_json, dict) else 0.0

    for key, dst_key in [("all_reduce", "all_reduce"), ("all_gather", "all_gather"), ("ping_pong", "broadcast")]:
        payload = collectives.get(key, {})
        entries = payload.get("entries", []) if isinstance(payload, dict) else []
        if not entries:
            continue
        latencies = [_safe_float(item.get("latency_ms_p50"), 0.0) for item in entries]
        gbps = [_safe_float(item.get("effective_gbps_p50"), 0.0) for item in entries]
        alpha_ms = min([x for x in latencies if x > 0.0], default=0.0)
        beta_gbps = max([x for x in gbps if x > 0.0], default=1e-9)
        if key == "ping_pong":
            alpha_ms *= 0.5
        out[dst_key] = {
            "alpha_ms": float(alpha_ms),
            "beta_gbps": float(beta_gbps),
        }
    if peak <= 0.0:
        peak = max([v["beta_gbps"] for v in out.values()], default=0.0)
    return out, float(peak)


def _scenario_label(row: dict[str, Any]) -> str:
    phase = str(row.get("phase", "kernel"))
    setup = str(row.get("setup", "dual"))
    kernel = str(row.get("kernel", ""))
    if phase == "kernel":
        return (
            f"{setup}|{kernel}|B{_safe_int(row.get('batch'))}|S{_safe_int(row.get('seq_len'))}|"
            f"D{_safe_int(row.get('model_dim'))}"
        )
    return (
        f"{phase}|{setup}|B{_safe_int(row.get('batch'))}|S{_safe_int(row.get('seq_len'))}|"
        f"C{_safe_int(row.get('context_len'))}|T{_safe_int(row.get('decode_steps'))}"
    )


def _plot_break_even(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    ordered = df.sort_values("required_overlap_pct")
    x = np.arange(len(ordered))
    y = ordered["required_overlap_pct"].to_numpy()
    colors = ["#2ca02c" if val <= 100.0 else "#d62728" for val in y]

    fig, ax = plt.subplots(figsize=(max(10, len(ordered) * 0.3), 5), constrained_layout=True)
    ax.bar(x, np.clip(y, 0.0, 200.0), color=colors)
    ax.axhline(100.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_ylabel("Required overlap (%) for dual to tie single")
    ax.set_title("Dual-Die Break-Even Overlap Requirement")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["scenario_label"].tolist(), rotation=70, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_speedup_vs_overlap(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    grouped = (
        df.groupby(["phase", "setup", "overlap_frac"], as_index=False)["predicted_speedup_single_over_dual"].median()
    )
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for (phase, setup), sub in grouped.groupby(["phase", "setup"]):
        sub = sub.sort_values("overlap_frac")
        ax.plot(
            sub["overlap_frac"].to_numpy(),
            sub["predicted_speedup_single_over_dual"].to_numpy(),
            marker="o",
            label=f"{phase}:{setup}",
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Overlap fraction")
    ax.set_ylabel("Predicted speedup (single/dual)")
    ax.set_title("Predicted Speedup vs Overlap")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not args.metrics_csv.exists():
        raise SystemExit(f"metrics.csv not found: {args.metrics_csv}")
    run_dir = args.metrics_csv.parent
    collectives_path = args.collectives_json or (run_dir / "collectives_summary.json")
    fabric_path = args.fabric_json or (run_dir / "fabric_calibration.json")
    out_dir = args.out_dir or run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    overlap_fracs = sorted({_safe_float(x, 0.0) for x in args.overlap_fracs.split(",")})
    overlap_fracs = [min(max(x, 0.0), 0.99) for x in overlap_fracs]
    target_overlap = min(max(float(args.target_overlap), 0.0), 0.99)

    metrics_df = pd.read_csv(args.metrics_csv)
    collectives_json = _load_json(collectives_path)
    fabric_json = _load_json(fabric_path)
    cal_params, fabric_peak_gbps = _calibration_params(fabric_json)

    collectives_rows = collectives_json.get("rows", []) if isinstance(collectives_json, dict) else []
    collectives_index: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in collectives_rows:
        key = _scenario_key_from_collective(row)
        collectives_index[key] = row

    metric_rows = metrics_df.to_dict(orient="records")
    single_index: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in metric_rows:
        if str(row.get("setup")) == "single_die":
            single_index[_scenario_key_from_metric(row)] = row

    sweep_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for row in metric_rows:
        setup = str(row.get("setup", ""))
        if "dual" not in setup:
            continue

        key = _scenario_key_from_metric(row)
        single = single_index.get(key)
        if single is None:
            continue
        coll_row = collectives_index.get(key, {})
        ops = coll_row.get("ops", {}) if isinstance(coll_row, dict) else {}

        alpha_term_ms = 0.0
        data_term_ms = 0.0
        total_bytes = 0.0
        for op_name, payload in ops.items():
            family = _op_family(str(op_name))
            alpha_ms = _safe_float(cal_params.get(family, {}).get("alpha_ms"), 0.0)
            beta_gbps = max(_safe_float(cal_params.get(family, {}).get("beta_gbps"), 0.0), 1e-9)
            count_total = _safe_float(payload.get("count_total"), 0.0)
            bytes_total = _safe_float(payload.get("bytes_total"), 0.0)
            alpha_term_ms += count_total * alpha_ms
            data_term_ms += (bytes_total / (beta_gbps * 1e9)) * 1000.0
            total_bytes += bytes_total

        comm_modeled_ms = alpha_term_ms + data_term_ms
        if comm_modeled_ms <= 0.0:
            comm_modeled_ms = _safe_float(row.get("communication_ms_p50"), 0.0)

        single_ms = _safe_float(single.get("latency_ms_p50"), 0.0)
        dual_ms = _safe_float(row.get("latency_ms_p50"), 0.0)
        compute_ms = _safe_float(row.get("compute_ms_p50"), 0.0)
        phase = str(row.get("phase", "kernel"))
        kernel = str(row.get("kernel", ""))
        scenario_label = _scenario_label(row)

        if comm_modeled_ms > 0.0:
            required_overlap = 1.0 - ((single_ms - compute_ms) / comm_modeled_ms)
            required_overlap_pct = max(0.0, required_overlap * 100.0)
        else:
            required_overlap_pct = 0.0 if compute_ms <= single_ms else 1000.0

        denom = max(1e-9, (1.0 - target_overlap))
        comm_budget_ms = (single_ms - compute_ms) / denom
        if comm_budget_ms <= alpha_term_ms:
            required_bw_multiplier = float("inf")
            required_bw_gbps = float("inf")
        else:
            required_bw_multiplier = data_term_ms / (comm_budget_ms - alpha_term_ms) if data_term_ms > 0 else 0.0
            required_bw_gbps = required_bw_multiplier * max(fabric_peak_gbps, 1e-9)

        summary_rows.append(
            {
                "phase": phase,
                "setup": setup,
                "kernel": kernel,
                "batch": _safe_int(row.get("batch"), 0),
                "seq_len": _safe_int(row.get("seq_len"), 0),
                "context_len": _safe_int(row.get("context_len"), 0),
                "decode_steps": _safe_int(row.get("decode_steps"), 0),
                "model_dim": _safe_int(row.get("model_dim"), 0),
                "num_heads": _safe_int(row.get("num_heads"), 0),
                "single_latency_ms_p50": single_ms,
                "dual_latency_ms_p50": dual_ms,
                "dual_compute_ms_p50": compute_ms,
                "dual_comm_modeled_ms": comm_modeled_ms,
                "dual_comm_observed_ms": _safe_float(row.get("communication_ms_p50"), 0.0),
                "collective_bytes_total": total_bytes,
                "required_overlap_pct": required_overlap_pct,
                "target_overlap_frac": target_overlap,
                "required_bw_multiplier_at_target_overlap": required_bw_multiplier,
                "required_bw_gbps_at_target_overlap": required_bw_gbps,
                "scenario_label": scenario_label,
            }
        )

        for overlap in overlap_fracs:
            pred_dual = compute_ms + comm_modeled_ms * (1.0 - overlap)
            speedup = (single_ms / pred_dual) if pred_dual > 0 else 0.0
            sweep_rows.append(
                {
                    "phase": phase,
                    "setup": setup,
                    "kernel": kernel,
                    "batch": _safe_int(row.get("batch"), 0),
                    "seq_len": _safe_int(row.get("seq_len"), 0),
                    "context_len": _safe_int(row.get("context_len"), 0),
                    "decode_steps": _safe_int(row.get("decode_steps"), 0),
                    "model_dim": _safe_int(row.get("model_dim"), 0),
                    "num_heads": _safe_int(row.get("num_heads"), 0),
                    "overlap_frac": overlap,
                    "single_latency_ms_p50": single_ms,
                    "predicted_dual_latency_ms_p50": pred_dual,
                    "predicted_speedup_single_over_dual": speedup,
                    "dual_wins": bool(pred_dual <= single_ms),
                    "scenario_label": scenario_label,
                }
            )

    sweep_df = pd.DataFrame(sweep_rows)
    summary_df = pd.DataFrame(summary_rows)
    sweep_csv = out_dir / f"{args.prefix}_sweep.csv"
    summary_csv = out_dir / f"{args.prefix}_summary.csv"
    sweep_df.to_csv(sweep_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    break_even_path = out_dir / f"{args.prefix}_break_even_overlap.png"
    speedup_path = out_dir / f"{args.prefix}_speedup_vs_overlap.png"
    _plot_break_even(summary_df, break_even_path)
    _plot_speedup_vs_overlap(sweep_df, speedup_path)

    print(sweep_csv)
    print(summary_csv)
    if break_even_path.exists():
        print(break_even_path)
    if speedup_path.exists():
        print(speedup_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
