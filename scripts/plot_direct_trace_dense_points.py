#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "single->single": "#1f77b4",
    "single->request": "#2ca02c",
    "request->request": "#1b7f3a",
}
LABELS = {
    "single->single": "single→single",
    "single->request": "single→request",
    "request->request": "request→request",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot dense per-request sample points from direct policy trace to "
            "show dual-die serving gains with many measurements."
        )
    )
    parser.add_argument("--summary-csv", type=Path, required=True, help="direct_policy_trace_summary.csv")
    parser.add_argument("--samples-json", type=Path, required=True, help="direct_policy_trace_samples.json")
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("results/plots/public_service_dual_dense_points.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--points-csv",
        type=Path,
        default=Path("results/plots/public_service_dual_dense_points.csv"),
        help="Output long-form sample points CSV.",
    )
    return parser.parse_args()


def _load_points(samples_json: Path) -> pd.DataFrame:
    raw = json.loads(samples_json.read_text(encoding="utf-8"))
    rows: list[dict[str, float | str | int]] = []
    for policy, payload in raw.items():
        total_ms = payload.get("total_ms", [])
        on_time = payload.get("on_time", [])
        for idx, latency_ms in enumerate(total_ms):
            rows.append(
                {
                    "policy": str(policy),
                    "sample_idx": int(idx),
                    "latency_ms": float(latency_ms),
                    "requests_per_s": float(16.0 / (float(latency_ms) / 1000.0)) if float(latency_ms) > 0 else 0.0,
                    "on_time": float(on_time[idx]) if idx < len(on_time) else float("nan"),
                }
            )
    points = pd.DataFrame(rows)
    if points.empty:
        raise ValueError("No sample points found in samples JSON.")
    return points


def _plot(points: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    policies = [policy for policy in summary["policy"].tolist() if policy in points["policy"].unique()]
    x_positions = np.arange(len(policies), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    left, right = axes

    for idx, policy in enumerate(policies):
        sub = points[points["policy"] == policy].sort_values("sample_idx")
        rng = np.random.default_rng(1000 + idx)
        jitter = rng.uniform(-0.18, 0.18, size=len(sub))
        left.scatter(
            np.full(len(sub), x_positions[idx]) + jitter,
            sub["latency_ms"],
            s=18,
            alpha=0.65,
            color=COLORS.get(policy, "#666666"),
            edgecolors="none",
            label=LABELS.get(policy, policy),
        )
        median_latency = float(np.median(sub["latency_ms"]))
        left.hlines(
            y=median_latency,
            xmin=x_positions[idx] - 0.26,
            xmax=x_positions[idx] + 0.26,
            color="#111111",
            linewidth=2.0,
        )

        lat_sorted = np.sort(sub["latency_ms"].to_numpy(dtype=float))
        cdf_y = np.linspace(0.0, 1.0, len(lat_sorted), endpoint=True)
        right.plot(
            lat_sorted,
            cdf_y,
            linewidth=2.2,
            color=COLORS.get(policy, "#666666"),
            label=f"{LABELS.get(policy, policy)} (n={len(lat_sorted)})",
        )

    left.set_xticks(x_positions)
    left.set_xticklabels([LABELS.get(policy, policy) for policy in policies], rotation=15, ha="right")
    left.set_ylabel("End-to-end request latency (ms)")
    left.set_title("Per-request measured points (jittered)")
    left.grid(axis="y", alpha=0.25)

    right.set_xlabel("End-to-end request latency (ms)")
    right.set_ylabel("Empirical CDF")
    right.set_title("Latency distribution comparison")
    right.grid(alpha=0.25)
    right.legend(loc="lower right", fontsize=8)

    if "single->single" in points["policy"].unique() and "single->request" in points["policy"].unique():
        base = points[points["policy"] == "single->single"]["latency_ms"].to_numpy(dtype=float)
        dual = points[points["policy"] == "single->request"]["latency_ms"].to_numpy(dtype=float)
        base_med = float(np.median(base))
        dual_med = float(np.median(dual))
        speedup = base_med / dual_med if dual_med > 0 else 0.0
        win_rate = float((dual < base_med).mean())
        title_suffix = f"median latency speedup={speedup:.2f}x, dual sample win-rate={win_rate:.1%}"
    else:
        title_suffix = "dense direct-trace points"

    fig.suptitle(
        f"Inference serving is better with dual-die request sharding ({title_suffix})",
        fontsize=13,
        y=1.03,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    summary = pd.read_csv(args.summary_csv)
    points = _load_points(args.samples_json)

    args.points_csv.parent.mkdir(parents=True, exist_ok=True)
    points.to_csv(args.points_csv, index=False)
    _plot(points=points, summary=summary, out_path=args.out_path)

    print(args.out_path)
    print(args.points_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
