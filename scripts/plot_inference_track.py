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
REASONS = {
    "prefill_ratio": "Shows whether dual setup improves long-context prefill latency relative to single.",
    "decode_slo_frontier": "Shows throughput-at-SLO, the right lens for serving value.",
    "decode_kv_efficiency": "Shows tokens/s achieved per GiB KV footprint to capture capacity efficiency.",
    "comm_breakdown": "Shows whether dual performance is compute-limited or communication-limited.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a curated inference-focused plot set from phase-study outputs and optionally purge stale plots."
        )
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        required=True,
        help="Path to phase-study metrics.csv",
    )
    parser.add_argument(
        "--decode-slo-csv",
        type=Path,
        default=None,
        help="Path to decode_slo_summary.csv (defaults to sibling of metrics.csv).",
    )
    parser.add_argument(
        "--break-even-csv",
        type=Path,
        default=None,
        help="Path to break_even_summary.csv (defaults to sibling of metrics.csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/plots"),
        help="Output plot directory.",
    )
    parser.add_argument("--prefix", default="inference_track", help="Output file prefix.")
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


def _plot_prefill_ratio(df: pd.DataFrame, out_path: Path) -> bool:
    prefill = df[df["phase"] == "prefill"].copy()
    if prefill.empty:
        return False

    single = prefill[prefill["setup"] == "single_die"][["seq_len", "latency_ms_p50"]].rename(
        columns={"latency_ms_p50": "single_latency"}
    )
    merged = prefill.merge(single, on="seq_len", how="left")
    merged["latency_ratio_vs_single"] = merged["latency_ms_p50"] / merged["single_latency"].replace(0, np.nan)

    seqs = sorted(int(x) for x in merged["seq_len"].unique())
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for setup in SETUP_ORDER:
        if setup == "single_die":
            continue
        sub = merged[merged["setup"] == setup]
        if sub.empty:
            continue
        ys = []
        for seq in seqs:
            row = sub[sub["seq_len"] == seq]
            ys.append(float(row.iloc[0]["latency_ratio_vs_single"]) if not row.empty else np.nan)
        ax.plot(seqs, ys, marker="o", linewidth=2, color=COLORS[setup], label=setup)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Dual latency / single latency (p50)")
    ax.set_title("Prefill Latency Ratio vs Single")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_decode_slo_frontier(df: pd.DataFrame, out_path: Path) -> bool:
    if df.empty:
        return False
    contexts = sorted(int(x) for x in df["context_len"].unique())
    if not contexts:
        return False

    fig, axes = plt.subplots(1, len(contexts), figsize=(6 * len(contexts), 4.8), constrained_layout=True)
    if len(contexts) == 1:
        axes = [axes]

    for ax, ctx in zip(axes, contexts):
        panel = df[df["context_len"] == ctx]
        slos = sorted(float(x) for x in panel["slo_ms"].unique())
        for setup in SETUP_ORDER:
            sub = panel[panel["setup"] == setup]
            if sub.empty:
                continue
            ys = []
            for slo in slos:
                row = sub[sub["slo_ms"] == slo]
                ys.append(float(row.iloc[0]["best_throughput_tokens_per_s"]) if not row.empty else np.nan)
            ax.plot(slos, ys, marker="o", linewidth=2, color=COLORS[setup], label=setup)
        ax.set_title(f"context={ctx}")
        ax.set_xlabel("SLO p90 latency (ms)")
        ax.set_ylabel("Best throughput (tokens/s)")
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Decode Throughput-at-SLO Frontier", y=1.13)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_decode_kv_efficiency(df: pd.DataFrame, out_path: Path) -> bool:
    decode = df[df["phase"] == "decode"].copy()
    if decode.empty:
        return False

    decode["kv_gib_per_rank"] = decode["kv_cache_bytes_per_rank"] / float(2**30)
    decode["tokens_per_s_per_kv_gib"] = decode["throughput_tokens_per_s"] / decode["kv_gib_per_rank"].replace(
        0, np.nan
    )
    contexts = sorted(int(x) for x in decode["context_len"].unique())

    fig, axes = plt.subplots(1, len(contexts), figsize=(6 * len(contexts), 4.8), constrained_layout=True)
    if len(contexts) == 1:
        axes = [axes]

    for ax, ctx in zip(axes, contexts):
        panel = decode[decode["context_len"] == ctx]
        concurrencies = sorted(int(x) for x in panel["batch"].unique())
        for setup in SETUP_ORDER:
            sub = panel[panel["setup"] == setup]
            if sub.empty:
                continue
            ys = []
            for c in concurrencies:
                row = sub[sub["batch"] == c]
                ys.append(float(row.iloc[0]["tokens_per_s_per_kv_gib"]) if not row.empty else np.nan)
            ax.plot(concurrencies, ys, marker="o", linewidth=2, color=COLORS[setup], label=setup)
        ax.set_title(f"context={ctx}")
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("tokens/s per GiB KV (per rank)")
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Decode KV Capacity Efficiency", y=1.13)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_comm_breakdown(df: pd.DataFrame, out_path: Path) -> bool:
    if df.empty:
        return False
    use = df[df["setup"].isin(SETUP_ORDER)].copy()
    if use.empty:
        return False

    labels = []
    compute_vals = []
    comm_vals = []
    for row in use.sort_values(["phase", "setup", "seq_len", "context_len", "batch"]).itertuples(index=False):
        if row.phase == "prefill":
            label = f"prefill S={int(row.seq_len)}\n{row.setup}"
        else:
            label = f"decode C={int(row.batch)} ctx={int(row.context_len)}\n{row.setup}"
        labels.append(label)
        compute_vals.append(float(row.compute_ms_p50))
        comm_vals.append(float(row.communication_ms_p50))

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(11, len(labels) * 0.45), 5.5), constrained_layout=True)
    ax.bar(x, compute_vals, label="compute_ms_p50", color="#4C78A8")
    ax.bar(x, comm_vals, bottom=compute_vals, label="comm_ms_p50", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right")
    ax.set_ylabel("Latency contribution (ms)")
    ax.set_title("Compute vs Communication Breakdown")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _write_manifest(
    *,
    out_path: Path,
    generated: list[Path],
    purged: list[Path],
) -> None:
    lines = ["# Inference Plot Manifest", "", "This set keeps only plots that directly support inference chip advice.", ""]
    for path in generated:
        reason = "Curated inference signal plot."
        for key, explanation in REASONS.items():
            if path.stem.endswith(key):
                reason = explanation
                break
        lines.append(f"- `{path.name}`: {reason}")
    if purged:
        lines.append("")
        lines.append("## Purged stale files")
        for path in purged:
            lines.append(f"- `{path.name}`")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.metrics_csv.exists():
        raise SystemExit(f"Missing metrics CSV: {args.metrics_csv}")

    decode_slo_csv = args.decode_slo_csv or (args.metrics_csv.parent / "decode_slo_summary.csv")
    break_even_csv = args.break_even_csv or (args.metrics_csv.parent / "break_even_summary.csv")

    metrics = pd.read_csv(args.metrics_csv)
    decode_slo = _safe_read_csv(decode_slo_csv)
    break_even = _safe_read_csv(break_even_csv)
    _ = break_even  # reserved for future extensions; metrics and SLO already capture key narrative.

    args.out_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        "prefill_ratio": args.out_dir / f"{args.prefix}_prefill_ratio.png",
        "decode_slo_frontier": args.out_dir / f"{args.prefix}_decode_slo_frontier.png",
        "decode_kv_efficiency": args.out_dir / f"{args.prefix}_decode_kv_efficiency.png",
        "comm_breakdown": args.out_dir / f"{args.prefix}_comm_breakdown.png",
    }

    generated: list[Path] = []
    if _plot_prefill_ratio(metrics, targets["prefill_ratio"]):
        generated.append(targets["prefill_ratio"])
    if _plot_decode_slo_frontier(decode_slo, targets["decode_slo_frontier"]):
        generated.append(targets["decode_slo_frontier"])
    if _plot_decode_kv_efficiency(metrics, targets["decode_kv_efficiency"]):
        generated.append(targets["decode_kv_efficiency"])
    if _plot_comm_breakdown(metrics, targets["comm_breakdown"]):
        generated.append(targets["comm_breakdown"])

    purged: list[Path] = []
    if args.purge_stale:
        keep = {path.resolve() for path in generated}
        for path in sorted(args.out_dir.glob(f"{args.prefix}_*.png")):
            if path.resolve() in keep:
                continue
            path.unlink(missing_ok=True)
            purged.append(path)

    manifest_path = args.out_dir / f"{args.prefix}_plot_manifest.md"
    _write_manifest(out_path=manifest_path, generated=generated, purged=purged)

    for path in generated:
        print(path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
