#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

SETUP_ORDER = ["single_die", "dual_die_moe_naive", "dual_die_moe_locality"]
COLORS = {
    "single_die": "#1f77b4",
    "dual_die_moe_naive": "#ff7f0e",
    "dual_die_moe_locality": "#2ca02c",
}
LABELS = {
    "single_die": "single",
    "dual_die_moe_naive": "dual-naive",
    "dual_die_moe_locality": "dual-locality",
}


def _display_setup(name: str) -> str:
    return LABELS.get(str(name), str(name).replace("_", "-"))


def _annotate_totals(
    ax: plt.Axes,
    *,
    x: np.ndarray,
    totals: np.ndarray,
    labels: list[str],
    y_pad: float,
) -> None:
    for idx, total in enumerate(totals):
        ax.text(float(x[idx]), float(total) + float(y_pad), labels[idx], ha="center", va="bottom", fontsize=8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot MoE service-study outputs: decode SLO frontier, capacity frontier, "
            "locality gain, remote dispatch ratio, and comm breakdown."
        )
    )
    parser.add_argument("--metrics-csv", type=Path, required=True, help="Path to MoE metrics.csv")
    parser.add_argument(
        "--decode-slo-csv",
        type=Path,
        default=None,
        help="Path to decode_slo_summary.csv (defaults to sibling of metrics.csv).",
    )
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
    parser.add_argument("--prefix", default="moe_service", help="Output file prefix.")
    parser.add_argument(
        "--purge-stale",
        action="store_true",
        help="Delete <prefix>_*.png files in out-dir that were not regenerated.",
    )
    return parser.parse_args()


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _plot_decode_slo_frontier(df: pd.DataFrame, out_path: Path) -> bool:
    if df.empty:
        return False

    contexts = sorted(int(x) for x in df["context_len"].unique())
    skews = sorted(float(x) for x in df["routing_skew"].unique())
    if not contexts or not skews:
        return False

    fig, axes = plt.subplots(
        len(skews),
        len(contexts),
        figsize=(6 * len(contexts), 4.5 * len(skews)),
        constrained_layout=True,
        squeeze=False,
    )

    for r, skew in enumerate(skews):
        for c, context in enumerate(contexts):
            ax = axes[r][c]
            panel = df[(df["routing_skew"] == skew) & (df["context_len"].astype(int) == int(context))]
            if panel.empty:
                ax.set_visible(False)
                continue
            slos = sorted(float(x) for x in panel["slo_ms"].unique())
            for setup in SETUP_ORDER:
                sub = panel[panel["setup"] == setup]
                if sub.empty:
                    continue
                ys = []
                for slo in slos:
                    row = sub[sub["slo_ms"] == slo]
                    ys.append(float(row.iloc[0]["best_throughput_tokens_per_s"]) if not row.empty else np.nan)
                ax.plot(slos, ys, marker="o", linewidth=2, color=COLORS[setup], label=_display_setup(setup))

            ax.set_title(f"ctx={context}, skew={skew:.2f}")
            ax.set_xlabel("SLO p90 latency (ms)")
            ax.set_ylabel("Best throughput (tokens/s)")
            ax.grid(alpha=0.25)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("MoE Decode Throughput under SLO", y=1.05)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_capacity_frontier(df: pd.DataFrame, out_path: Path) -> bool:
    if df.empty:
        return False

    contexts = sorted(int(x) for x in df["context_len"].unique())
    skews = sorted(float(x) for x in df["routing_skew"].unique())
    if not contexts or not skews:
        return False

    fig, axes = plt.subplots(1, len(skews), figsize=(6 * len(skews), 5), constrained_layout=True)
    if len(skews) == 1:
        axes = [axes]

    width = 0.24
    x = np.arange(len(contexts))
    for ax, skew in zip(axes, skews):
        panel = df[df["routing_skew"] == skew]
        for idx, setup in enumerate(SETUP_ORDER):
            sub = panel[panel["setup"] == setup]
            if sub.empty:
                continue
            ys = []
            for context in contexts:
                row = sub[sub["context_len"].astype(int) == int(context)]
                ys.append(float(row.iloc[0]["max_feasible_concurrency"]) if not row.empty else np.nan)
            ax.bar(x + (idx - 1) * width, ys, width=width, color=COLORS[setup], label=_display_setup(setup))

        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in contexts])
        ax.set_xlabel("Context length")
        ax.set_ylabel("Max SLO-feasible concurrency")
        ax.set_title(f"routing_skew={skew:.2f}")
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("MoE Capacity Frontier", y=1.08)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_locality_gain(metrics: pd.DataFrame, out_path: Path) -> bool:
    decode = metrics[metrics["phase"] == "decode"].copy()
    if decode.empty:
        return False

    key_cols = ["context_len", "batch", "decode_steps", "num_experts", "top_k", "routing_skew"]
    naive = decode[decode["setup"] == "dual_die_moe_naive"][key_cols + ["throughput_tokens_per_s"]]
    locality = decode[decode["setup"] == "dual_die_moe_locality"][key_cols + ["throughput_tokens_per_s"]]
    if naive.empty or locality.empty:
        return False

    merged = locality.merge(naive, on=key_cols, how="inner", suffixes=("_locality", "_naive"))
    if merged.empty:
        return False

    merged["locality_speedup"] = merged["throughput_tokens_per_s_locality"] / merged[
        "throughput_tokens_per_s_naive"
    ].replace(0.0, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    by_skew = merged.groupby("routing_skew", as_index=False)["locality_speedup"].median().sort_values("routing_skew")
    axes[0].plot(by_skew["routing_skew"], by_skew["locality_speedup"], marker="o", linewidth=2, color="#2ca02c")
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Routing skew")
    axes[0].set_ylabel("Throughput ratio (dual-locality / dual-naive)")
    axes[0].set_title("Locality Gain vs Routing Skew")
    axes[0].grid(alpha=0.25)

    by_batch = merged.groupby("batch", as_index=False)["locality_speedup"].median().sort_values("batch")
    axes[1].plot(by_batch["batch"], by_batch["locality_speedup"], marker="o", linewidth=2, color="#2ca02c")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Concurrency")
    axes[1].set_ylabel("Throughput ratio (dual-locality / dual-naive)")
    axes[1].set_title("Locality Gain vs Concurrency")
    axes[1].grid(alpha=0.25)

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_remote_dispatch_ratio(metrics: pd.DataFrame, out_path: Path) -> bool:
    decode = metrics[metrics["phase"] == "decode"].copy()
    if decode.empty or "remote_dispatch_ratio_p50" not in decode.columns:
        return False

    dual = decode[decode["setup"].isin(["dual_die_moe_naive", "dual_die_moe_locality"])].copy()
    if dual.empty:
        return False

    grouped = (
        dual.groupby(["setup", "routing_skew"], as_index=False)["remote_dispatch_ratio_p50"]
        .median()
        .sort_values(["routing_skew", "setup"])
    )
    skews = sorted(float(x) for x in grouped["routing_skew"].unique())
    x = np.arange(len(skews), dtype=float)
    width = 0.34

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    plotted = False
    for idx, setup in enumerate(["dual_die_moe_naive", "dual_die_moe_locality"]):
        sub = grouped[grouped["setup"] == setup]
        if sub.empty:
            continue
        ys = []
        for skew in skews:
            row = sub[sub["routing_skew"] == skew]
            ys.append(float(row.iloc[0]["remote_dispatch_ratio_p50"]) if not row.empty else np.nan)
        ax.bar(x + (idx - 0.5) * width, ys, width=width, color=COLORS[setup], label=_display_setup(setup))
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:.2f}" for s in skews])
    ax.set_xlabel("Routing skew")
    ax.set_ylabel("Median remote-dispatch fraction")
    ax.set_title("Locality Reduces Remote Expert Dispatch")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_comm_breakdown(metrics: pd.DataFrame, out_path: Path) -> bool:
    decode = metrics[metrics["phase"] == "decode"].copy()
    if decode.empty:
        return False

    points: list[pd.Series] = []
    for setup in SETUP_ORDER:
        sub = decode[decode["setup"] == setup]
        if sub.empty:
            continue
        # Pick representative high-pressure point by max context, then max concurrency.
        max_ctx = int(sub["context_len"].max())
        ctx_sub = sub[sub["context_len"].astype(int) == max_ctx]
        max_batch = int(ctx_sub["batch"].max())
        chosen = ctx_sub[ctx_sub["batch"].astype(int) == max_batch]
        if chosen.empty:
            continue
        points.append(chosen.sort_values("throughput_tokens_per_s", ascending=False).iloc[0])

    if not points:
        return False

    labels = []
    compute_vals = []
    comm_vals = []
    for row in points:
        labels.append(f"{_display_setup(str(row['setup']))}\nctx={int(row['context_len'])},C={int(row['batch'])}")
        compute_vals.append(float(row["compute_ms_p50"]))
        comm_vals.append(float(row["communication_ms_p50"]))

    x = np.arange(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2.4), 4.8), constrained_layout=True)
    ax.bar(x, compute_vals, color="#4C78A8", label="compute_ms_p50")
    ax.bar(x, comm_vals, bottom=compute_vals, color="#F58518", label="communication_ms_p50")
    totals = np.asarray(compute_vals, dtype=float) + np.asarray(comm_vals, dtype=float)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency contribution (ms)")
    ax.set_title("Representative MoE Compute vs Communication Breakdown")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    y_pad = max(1.0, float(totals.max(initial=0.0)) * 0.03)
    total_labels = []
    for compute_ms, comm_ms in zip(compute_vals, comm_vals):
        total = float(compute_ms + comm_ms)
        comm_share = (float(comm_ms) / total * 100.0) if total > 0 else 0.0
        total_labels.append(f"{total:.1f} ms\n{comm_share:.0f}% comm")
    _annotate_totals(ax, x=x, totals=totals, labels=total_labels, y_pad=y_pad)
    ax.set_ylim(0.0, float(totals.max(initial=0.0)) + y_pad * 3.6)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> int:
    args = parse_args()
    if not args.metrics_csv.exists():
        raise SystemExit(f"Missing metrics CSV: {args.metrics_csv}")

    decode_slo_csv = args.decode_slo_csv or (args.metrics_csv.parent / "decode_slo_summary.csv")
    capacity_csv = args.capacity_csv or (args.metrics_csv.parent / "capacity_frontier.csv")

    metrics = _safe_read_csv(args.metrics_csv)
    decode_slo = _safe_read_csv(decode_slo_csv)
    capacity = _safe_read_csv(capacity_csv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        "decode_slo_frontier": args.out_dir / f"{args.prefix}_decode_slo_frontier.png",
        "capacity_frontier": args.out_dir / f"{args.prefix}_capacity_frontier.png",
        "locality_gain": args.out_dir / f"{args.prefix}_locality_gain.png",
        "remote_dispatch_ratio": args.out_dir / f"{args.prefix}_remote_dispatch_ratio.png",
        "comm_breakdown": args.out_dir / f"{args.prefix}_comm_breakdown.png",
    }

    generated: list[Path] = []
    if _plot_decode_slo_frontier(decode_slo, targets["decode_slo_frontier"]):
        generated.append(targets["decode_slo_frontier"])
    if _plot_capacity_frontier(capacity, targets["capacity_frontier"]):
        generated.append(targets["capacity_frontier"])
    if _plot_locality_gain(metrics, targets["locality_gain"]):
        generated.append(targets["locality_gain"])
    if _plot_remote_dispatch_ratio(metrics, targets["remote_dispatch_ratio"]):
        generated.append(targets["remote_dispatch_ratio"])
    if _plot_comm_breakdown(metrics, targets["comm_breakdown"]):
        generated.append(targets["comm_breakdown"])

    if not generated:
        raise SystemExit("No plots generated (missing decode/SLO/capacity data).")

    purged: list[Path] = []
    if args.purge_stale:
        keep = {path.resolve() for path in generated}
        for path in sorted(args.out_dir.glob(f"{args.prefix}_*.png")):
            if path.resolve() in keep:
                continue
            path.unlink(missing_ok=True)
            purged.append(path)

    for path in generated:
        print(path)
    for path in purged:
        print(f"purged: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
