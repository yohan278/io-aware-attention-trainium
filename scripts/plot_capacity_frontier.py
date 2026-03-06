#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

SETUP_ORDER = ["single_die", "dual_die_tensor_optimized", "dual_die_request_sharded"]
COLORS = {
    "single_die": "#1f77b4",
    "dual_die_tensor_optimized": "#ff7f0e",
    "dual_die_request_sharded": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot service-capacity frontier and decode concurrency scaling from phase-study artifacts."
        )
    )
    parser.add_argument("--metrics-csv", type=Path, required=True, help="Path to phase-study metrics.csv.")
    parser.add_argument(
        "--capacity-csv",
        type=Path,
        default=None,
        help="Path to capacity_frontier.csv (defaults to sibling of metrics.csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/plots"),
        help="Output plot directory.",
    )
    parser.add_argument("--prefix", default="service_capacity", help="Output file prefix.")
    return parser.parse_args()


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _plot_capacity_frontier(capacity: pd.DataFrame, out_path: Path) -> bool:
    if capacity.empty:
        return False

    contexts = sorted(int(x) for x in capacity["context_len"].unique())
    fig, ax = plt.subplots(figsize=(max(8.5, len(contexts) * 2.6), 5), constrained_layout=True)

    x = np.arange(len(contexts), dtype=float)
    width = 0.25
    plotted_any = False
    for idx, setup in enumerate(SETUP_ORDER):
        subset = capacity[capacity["setup"] == setup]
        if subset.empty:
            continue
        ys = []
        for context in contexts:
            row = subset[subset["context_len"] == context]
            ys.append(float(row.iloc[0]["max_feasible_concurrency"]) if not row.empty else np.nan)
        offset = (idx - 1) * width
        ax.bar(x + offset, ys, width=width, label=setup, color=COLORS[setup], alpha=0.9)
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return False

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in contexts])
    ax.set_xlabel("Decode context length")
    ax.set_ylabel("Max feasible concurrency")
    ax.set_title("Capacity Frontier (Feasible Concurrency by Context)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_concurrency_scaling(metrics: pd.DataFrame, capacity: pd.DataFrame, out_path: Path) -> bool:
    decode = metrics[metrics["phase"] == "decode"].copy()
    if decode.empty:
        return False

    contexts = sorted(int(x) for x in decode["context_len"].unique())
    fig, axes = plt.subplots(1, len(contexts), figsize=(6 * len(contexts), 4.8), constrained_layout=True)
    if len(contexts) == 1:
        axes = [axes]

    for ax, context in zip(axes, contexts):
        panel = decode[decode["context_len"] == context]
        if panel.empty:
            continue
        for setup in SETUP_ORDER:
            sub = panel[panel["setup"] == setup].sort_values("batch")
            if sub.empty:
                continue
            xvals = sub["batch"].astype(int).to_numpy()
            yvals = sub["throughput_tokens_per_s"].astype(float).to_numpy()
            ax.plot(xvals, yvals, marker="o", linewidth=2, color=COLORS[setup], label=setup)

            max_feasible = None
            cap_row = capacity[
                (capacity["setup"] == setup) & (capacity["context_len"].astype(int) == int(context))
            ]
            if not cap_row.empty:
                max_feasible = int(cap_row.iloc[0]["max_feasible_concurrency"])
            if max_feasible is not None:
                infeasible_mask = xvals > max_feasible
                if np.any(infeasible_mask):
                    ax.scatter(
                        xvals[infeasible_mask],
                        yvals[infeasible_mask],
                        marker="x",
                        color=COLORS[setup],
                        s=55,
                        linewidths=2,
                    )
                    ax.axvline(max_feasible, color=COLORS[setup], linestyle="--", alpha=0.25)

        ax.set_title(f"context={context}")
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("Throughput (tokens/s)")
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.1))
    fig.suptitle("Decode Throughput Scaling (X marks infeasible points)", y=1.15)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> int:
    args = parse_args()
    if not args.metrics_csv.exists():
        raise SystemExit(f"Missing metrics CSV: {args.metrics_csv}")

    capacity_csv = args.capacity_csv or (args.metrics_csv.parent / "capacity_frontier.csv")
    metrics = _safe_read_csv(args.metrics_csv)
    capacity = _safe_read_csv(capacity_csv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    capacity_png = args.out_dir / f"{args.prefix}_capacity_frontier.png"
    scaling_png = args.out_dir / f"{args.prefix}_concurrency_scaling.png"

    generated: list[Path] = []
    if _plot_capacity_frontier(capacity, capacity_png):
        generated.append(capacity_png)
    if _plot_concurrency_scaling(metrics, capacity, scaling_png):
        generated.append(scaling_png)

    if not generated:
        raise SystemExit("No plots generated (missing capacity/decode data).")
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
