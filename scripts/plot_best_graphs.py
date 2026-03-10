#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

SETUP_ORDER = ["single_die", "dual_die_tensor_optimized", "dual_die_request_sharded"]
POLICY_ORDER = [
    ("single->single", "single_die", "single_die"),
    ("single->request", "single_die", "dual_die_request_sharded"),
    ("single->tensor", "single_die", "dual_die_tensor_optimized"),
    ("request->request", "dual_die_request_sharded", "dual_die_request_sharded"),
]
SETUP_COLORS = {
    "single_die": "#1f77b4",
    "single_die_native": "#9c9c9c",
    "dual_die_tensor_optimized": "#ff7f0e",
    "dual_die_request_sharded": "#2ca02c",
    "dual_die_naive": "#e15759",
    "dual_die_optimized": "#76b7b2",
}
SETUP_LABELS = {
    "single_die": "single",
    "single_die_native": "single-baseline",
    "dual_die_tensor_optimized": "dual-tensor",
    "dual_die_request_sharded": "dual-request",
    "dual_die_naive": "dual-naive",
    "dual_die_optimized": "dual-tensor",
}
POLICY_COLORS = {
    "single->single": "#1f77b4",
    "single->tensor": "#ff7f0e",
    "single->request": "#2ca02c",
    "request->request": "#1b7f3a",
}
POLICY_LABELS = {
    "single->single": "single→single",
    "single->request": "single→request",
    "single->tensor": "single→tensor",
    "request->request": "request→request",
}


def _display_setup(name: str) -> str:
    return SETUP_LABELS.get(str(name), str(name).replace("_", "-"))


def _display_policy(name: str) -> str:
    return POLICY_LABELS.get(str(name), str(name))


def _annotate_totals(
    ax: Any,
    *,
    x: np.ndarray,
    totals: np.ndarray,
    labels: list[str],
    y_pad: float,
) -> None:
    for idx, total in enumerate(totals):
        ax.text(
            float(x[idx]),
            float(total) + float(y_pad),
            labels[idx],
            ha="center",
            va="bottom",
            fontsize=8,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paper-best dual-die figures: decode SLO frontier, crossover heatmap, "
            "prefill ratio, hybrid end-to-end bars, comm breakdown, and collective-count scatter."
        )
    )
    parser.add_argument("--phase-metrics-csv", type=Path, required=True, help="Path to phase-study metrics.csv")
    parser.add_argument(
        "--decode-slo-csv",
        type=Path,
        default=None,
        help="Path to decode_slo_summary.csv (defaults to sibling of phase metrics).",
    )
    parser.add_argument(
        "--phase-collectives-json",
        type=Path,
        default=None,
        help="Path to phase collectives_summary.json (defaults to sibling of phase metrics).",
    )
    parser.add_argument(
        "--kernel-metrics-csv",
        type=Path,
        default=None,
        help="Path to kernel-study metrics.csv for collective scatter.",
    )
    parser.add_argument(
        "--kernel-collectives-json",
        type=Path,
        default=None,
        help="Path to kernel collectives_summary.json for collective scatter.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/plots"),
        help="Output directory for plots.",
    )
    parser.add_argument("--prefix", default="best_graphs", help="Output file prefix.")
    parser.add_argument(
        "--heatmap-slo-ms",
        type=float,
        default=250.0,
        help="p90 latency SLO threshold used to determine best setup in crossover heatmap.",
    )
    parser.add_argument(
        "--hybrid-output-tokens",
        type=int,
        default=128,
        help="Decode output token count for hybrid end-to-end policy bar chart.",
    )
    parser.add_argument(
        "--hybrid-context-len",
        type=int,
        default=None,
        help="Context length used for hybrid bar chart (default=max available decode context).",
    )
    parser.add_argument(
        "--hybrid-concurrency",
        type=int,
        default=None,
        help="Concurrency used for hybrid bar chart (default=max common concurrency across setups).",
    )
    parser.add_argument(
        "--top3-only",
        action="store_true",
        help="Generate only the top 3 figures: decode SLO frontier, crossover heatmap, comm breakdown.",
    )
    return parser.parse_args()


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _plot_decode_slo_frontier(metrics: pd.DataFrame, out_path: Path, slo_ms: float) -> bool:
    decode = metrics[metrics["phase"] == "decode"].copy()
    if decode.empty:
        return False

    contexts = sorted(int(x) for x in decode["context_len"].unique())
    if not contexts:
        return False

    fig, axes = plt.subplots(1, len(contexts), figsize=(6 * len(contexts), 4.8), constrained_layout=True)
    if len(contexts) == 1:
        axes = [axes]

    for ax, context in zip(axes, contexts):
        panel = decode[decode["context_len"].astype(int) == int(context)].copy()
        if panel.empty:
            ax.set_visible(False)
            continue
        for setup in SETUP_ORDER:
            sub = panel[panel["setup"] == setup]
            if sub.empty:
                continue
            sub = sub.sort_values("latency_ms_p90")
            ax.plot(
                sub["latency_ms_p90"].to_numpy(dtype=float),
                sub["throughput_tokens_per_s"].to_numpy(dtype=float),
                marker="o",
                linewidth=2,
                color=SETUP_COLORS[setup],
                label=_display_setup(setup),
            )
            for _, row in sub.iterrows():
                ax.annotate(
                    f"C={int(row['batch'])}",
                    (float(row["latency_ms_p90"]), float(row["throughput_tokens_per_s"])),
                    fontsize=7,
                    xytext=(4, 4),
                    textcoords="offset points",
                    alpha=0.8,
                )
        ax.axvline(float(slo_ms), color="#666666", linestyle="--", linewidth=1.1, alpha=0.8)
        ax.set_xlabel("p90 latency (ms)")
        ax.set_ylabel("Throughput (tokens/s)")
        ax.set_title(f"context={context}")
        ax.grid(alpha=0.25)
        ax.margins(x=0.08, y=0.08)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(f"Measured Decode Throughput vs p90 Latency ({slo_ms:.0f} ms SLO line)", y=1.08)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_crossover_heatmap(metrics: pd.DataFrame, out_path: Path, slo_ms: float) -> bool:
    decode = metrics[metrics["phase"] == "decode"].copy()
    if decode.empty:
        return False

    contexts = sorted(int(x) for x in decode["context_len"].unique())
    concurrencies = sorted(int(x) for x in decode["batch"].unique())
    if len(contexts) < 2 or len(concurrencies) < 2:
        return False

    setup_to_id = {"none": 0, "single_die": 1, "dual_die_tensor_optimized": 2, "dual_die_request_sharded": 3}
    id_to_label = {
        0: "none",
        1: "single",
        2: "dual-tensor",
        3: "dual-request",
    }

    grid = np.zeros((len(concurrencies), len(contexts)), dtype=int)

    for yi, conc in enumerate(concurrencies):
        for xi, context in enumerate(contexts):
            panel = decode[(decode["batch"].astype(int) == int(conc)) & (decode["context_len"].astype(int) == int(context))]
            if panel.empty:
                continue
            feasible = panel[panel["latency_ms_p90"].astype(float) <= float(slo_ms)]
            if feasible.empty:
                grid[yi, xi] = 0
                continue
            best = feasible.sort_values("throughput_tokens_per_s", ascending=False).iloc[0]
            grid[yi, xi] = setup_to_id.get(str(best["setup"]), 0)

    cmap = ListedColormap(["#bdbdbd", SETUP_COLORS["single_die"], SETUP_COLORS["dual_die_tensor_optimized"], SETUP_COLORS["dual_die_request_sharded"]])

    fig, ax = plt.subplots(figsize=(1.8 * len(contexts) + 2, 0.8 * len(concurrencies) + 2), constrained_layout=True)
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=3)
    _ = im

    ax.set_xticks(np.arange(len(contexts)))
    ax.set_xticklabels([str(c) for c in contexts])
    ax.set_yticks(np.arange(len(concurrencies)))
    ax.set_yticklabels([str(c) for c in concurrencies])
    ax.set_xlabel("Context length")
    ax.set_ylabel("Concurrency")
    ax.set_title(f"Best Setup Regime Map (p90 <= {slo_ms:.0f} ms)")

    for yi in range(len(concurrencies)):
        for xi in range(len(contexts)):
            val = int(grid[yi, xi])
            ax.text(xi, yi, id_to_label[val], ha="center", va="center", fontsize=8, color="black")

    legend_labels = ["none", "single", "dual-tensor", "dual-request"]
    legend_colors = ["#bdbdbd", SETUP_COLORS["single_die"], SETUP_COLORS["dual_die_tensor_optimized"], SETUP_COLORS["dual_die_request_sharded"]]
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in legend_colors]
    ax.legend(handles, legend_labels, loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.14))

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_prefill_latency_ratio(metrics: pd.DataFrame, out_path: Path) -> bool:
    prefill = metrics[metrics["phase"] == "prefill"].copy()
    if prefill.empty:
        return False

    base = prefill[prefill["setup"] == "single_die"][["batch", "seq_len", "latency_ms_p50"]].rename(
        columns={"latency_ms_p50": "single_latency_ms_p50"}
    )
    dual = prefill[prefill["setup"] != "single_die"].copy()
    if dual.empty or base.empty:
        return False

    merged = dual.merge(base, on=["batch", "seq_len"], how="inner")
    if merged.empty:
        return False

    merged["ratio"] = merged["latency_ms_p50"] / merged["single_latency_ms_p50"].replace(0, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    seqs = sorted(int(x) for x in merged["seq_len"].unique())

    full_ax, zoom_ax = axes
    request_annotations: list[tuple[int, float]] = []

    for setup in [s for s in SETUP_ORDER if s != "single_die"]:
        for batch in sorted(int(x) for x in merged[merged["setup"] == setup]["batch"].unique()):
            sub = merged[(merged["setup"] == setup) & (merged["batch"].astype(int) == int(batch))]
            ys = []
            for seq in seqs:
                row = sub[sub["seq_len"].astype(int) == int(seq)]
                value = float(row.iloc[0]["ratio"]) if not row.empty else np.nan
                ys.append(value)
                if setup == "dual_die_request_sharded" and not np.isnan(value):
                    request_annotations.append((int(seq), float(value)))

            label = f"{SETUP_LABELS.get(setup, setup)} B={batch}"
            full_ax.plot(
                seqs,
                ys,
                marker="o",
                linewidth=2,
                color=SETUP_COLORS.get(setup, "#333333"),
                label=label,
            )
            if setup == "dual_die_request_sharded":
                zoom_ax.plot(
                    seqs,
                    ys,
                    marker="o",
                    linewidth=2.2,
                    color=SETUP_COLORS.get(setup, "#333333"),
                    label=label,
                )

    for ax in axes:
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Sequence length (tokens)")
        ax.set_xticks(seqs)
        ax.set_xticklabels([str(seq) for seq in seqs])
        ax.grid(alpha=0.25)

    full_ax.set_ylabel("dual/single latency ratio (p50)")
    full_ax.set_title("Full Range")
    full_ax.set_ylim(0.0, 50.0)
    full_ax.legend()

    request_values = [value for _, value in request_annotations]
    zoom_lower = max(0.95, (min(request_values) - 0.08) if request_values else 0.95)
    zoom_upper = min(2.0, (max(request_values) + 0.08) if request_values else 1.25)
    zoom_ax.set_title("Request-Sharded Detail")
    zoom_ax.set_ylim(zoom_lower, zoom_upper)
    zoom_ax.legend()
    for seq, value in request_annotations:
        zoom_ax.annotate(
            f"{value:.2f}×",
            (float(seq), float(value)),
            fontsize=8,
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )

    fig.suptitle("Prefill Latency Ratio Relative to Single-Die", y=1.03)

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_phase_tradeoff(metrics: pd.DataFrame, out_path: Path) -> bool:
    prefill = metrics[metrics["phase"] == "prefill"].copy()
    decode = metrics[metrics["phase"] == "decode"].copy()
    if prefill.empty or decode.empty:
        return False

    # Pick a single, representative operating point:
    # - Prefill: longest committed sequence length (hardest case).
    # - Decode: largest committed concurrency at the committed context.
    seq_len = int(prefill["seq_len"].astype(int).max())
    context_len = int(decode["context_len"].astype(int).max())

    prefill_panel = prefill[prefill["seq_len"].astype(int) == seq_len]
    decode_panel = decode[decode["context_len"].astype(int) == context_len]
    if prefill_panel.empty or decode_panel.empty:
        return False

    prefill_batch = int(prefill_panel["batch"].astype(int).max())
    decode_conc = int(decode_panel["batch"].astype(int).max())

    base_prefill = prefill_panel[(prefill_panel["setup"] == "single_die") & (prefill_panel["batch"].astype(int) == prefill_batch)]
    base_decode = decode_panel[(decode_panel["setup"] == "single_die") & (decode_panel["batch"].astype(int) == decode_conc)]
    if base_prefill.empty or base_decode.empty:
        return False

    base_prefill_ms = float(base_prefill.iloc[0]["latency_ms_p50"])
    base_decode_toks = float(base_decode.iloc[0]["throughput_tokens_per_s"])
    if base_prefill_ms <= 0 or base_decode_toks <= 0:
        return False

    setups = ["single_die", "dual_die_request_sharded", "dual_die_tensor_optimized"]
    rows: list[dict[str, float | str]] = []
    for setup in setups:
        p_row = prefill_panel[(prefill_panel["setup"] == setup) & (prefill_panel["batch"].astype(int) == prefill_batch)]
        d_row = decode_panel[(decode_panel["setup"] == setup) & (decode_panel["batch"].astype(int) == decode_conc)]
        if p_row.empty or d_row.empty:
            continue
        prefill_ratio = float(p_row.iloc[0]["latency_ms_p50"]) / base_prefill_ms
        decode_ratio = float(d_row.iloc[0]["throughput_tokens_per_s"]) / base_decode_toks
        rows.append(
            {
                "setup": setup,
                "prefill_ratio": prefill_ratio,
                "decode_throughput_ratio": decode_ratio,
            }
        )

    if not rows:
        return False

    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)

    for _, row in plot_df.iterrows():
        setup = str(row["setup"])
        x = float(row["prefill_ratio"])
        y = float(row["decode_throughput_ratio"])
        ax.scatter(
            x,
            y,
            s=160.0,
            color=SETUP_COLORS.get(setup, "#808080"),
            edgecolors="black",
            linewidths=0.35,
            alpha=0.9,
        )
        ax.annotate(
            _display_setup(setup),
            (x, y),
            fontsize=9,
            xytext=(8, 6),
            textcoords="offset points",
        )

    ax.axvline(1.0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(f"Prefill latency ratio vs single (p50) at S={seq_len}, B={prefill_batch}")
    ax.set_ylabel(f"Decode throughput ratio vs single at ctx={context_len}, C={decode_conc}")
    ax.set_title("Phase Tradeoff: Prefill Penalty vs Decode Benefit")
    ax.grid(alpha=0.25, which="both")

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _pick_hybrid_shape(decode: pd.DataFrame, context_override: int | None, conc_override: int | None) -> tuple[int, int] | None:
    contexts = sorted(int(x) for x in decode["context_len"].unique())
    if not contexts:
        return None
    context = int(context_override) if context_override is not None else max(contexts)
    panel = decode[decode["context_len"].astype(int) == context]
    if panel.empty:
        return None

    if conc_override is not None:
        return context, int(conc_override)

    setups = sorted(panel["setup"].unique().tolist())
    common = []
    for c in sorted(int(x) for x in panel["batch"].unique()):
        has_all = True
        for setup in setups:
            if panel[(panel["setup"] == setup) & (panel["batch"].astype(int) == int(c))].empty:
                has_all = False
                break
        if has_all:
            common.append(c)
    if not common:
        return None
    return context, max(common)


def _build_hybrid_policy_rows(
    *,
    metrics: pd.DataFrame,
    output_tokens: int,
    context_override: int | None,
    conc_override: int | None,
) -> tuple[pd.DataFrame, int, int] | None:
    prefill = metrics[metrics["phase"] == "prefill"].copy()
    decode = metrics[metrics["phase"] == "decode"].copy()
    if prefill.empty or decode.empty:
        return None

    shape = _pick_hybrid_shape(decode, context_override=context_override, conc_override=conc_override)
    if shape is None:
        return None
    context, concurrency = shape

    rows: list[dict[str, float | str]] = []
    for policy_name, prefill_setup, decode_setup in POLICY_ORDER:
        decode_row = decode[
            (decode["setup"] == decode_setup)
            & (decode["context_len"].astype(int) == int(context))
            & (decode["batch"].astype(int) == int(concurrency))
        ]
        if decode_row.empty:
            continue
        decode_row = decode_row.iloc[0]

        prefill_sub = prefill[prefill["setup"] == prefill_setup].copy()
        if prefill_sub.empty:
            continue
        prefill_sub["seq_dist"] = (prefill_sub["seq_len"].astype(int) - int(context)).abs()
        prefill_row = prefill_sub.sort_values(["seq_dist", "seq_len"]).iloc[0]

        prefill_latency_total = float(prefill_row["latency_ms_p50"])
        prefill_batch = max(1, int(prefill_row["batch"]))
        prefill_per_request_ms = prefill_latency_total / float(prefill_batch)

        throughput = float(decode_row["throughput_tokens_per_s"])
        decode_ms_per_token_per_request = (1000.0 * float(concurrency) / throughput) if throughput > 0 else np.nan
        decode_tail_ms = float(output_tokens) * decode_ms_per_token_per_request
        total_request_ms = prefill_per_request_ms + decode_tail_ms
        req_per_s = 1000.0 / total_request_ms if total_request_ms > 0 else np.nan

        rows.append(
            {
                "policy": policy_name,
                "prefill_setup": prefill_setup,
                "decode_setup": decode_setup,
                "prefill_per_request_ms": prefill_per_request_ms,
                "decode_tail_ms": decode_tail_ms,
                "decode_ms_per_token_per_request": decode_ms_per_token_per_request,
                "total_request_ms": total_request_ms,
                "requests_per_s": req_per_s,
            }
        )

    if not rows:
        return None
    return pd.DataFrame(rows), context, concurrency


def _plot_hybrid_end_to_end(
    metrics: pd.DataFrame,
    out_path: Path,
    output_tokens: int,
    context_override: int | None,
    conc_override: int | None,
) -> bool:
    hybrid = _build_hybrid_policy_rows(
        metrics=metrics,
        output_tokens=output_tokens,
        context_override=context_override,
        conc_override=conc_override,
    )
    if hybrid is None:
        return False
    plot_df, context, concurrency = hybrid
    plot_df["label"] = plot_df["policy"].map(_display_policy)
    baseline = plot_df[plot_df["policy"] == "single->single"]
    baseline_total = (
        float(baseline.iloc[0]["total_request_ms"])
        if not baseline.empty
        else float(plot_df["total_request_ms"].min())
    )
    plot_df["speedup_vs_single"] = baseline_total / plot_df["total_request_ms"].replace(0.0, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    x = np.arange(len(plot_df), dtype=float)
    colors = [POLICY_COLORS.get(policy, "#666666") for policy in plot_df["policy"]]

    axes[0].bar(x, plot_df["prefill_per_request_ms"], color="#9ecae1", label="prefill")
    axes[0].bar(
        x,
        plot_df["decode_tail_ms"],
        bottom=plot_df["prefill_per_request_ms"],
        color=colors,
        label=f"decode tail ({output_tokens} tokens)",
    )
    totals = (
        plot_df["prefill_per_request_ms"].to_numpy(dtype=float)
        + plot_df["decode_tail_ms"].to_numpy(dtype=float)
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(plot_df["label"], rotation=15, ha="right")
    axes[0].set_ylabel("Per-request time estimate (ms)")
    axes[0].set_title("Composed Request-Time Estimate")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()
    y_pad = max(10.0, float(totals.max(initial=0.0)) * 0.03)
    _annotate_totals(
        axes[0],
        x=x,
        totals=totals,
        labels=[f"{total:.0f} ms" for total in totals],
        y_pad=y_pad,
    )
    axes[0].set_ylim(0.0, float(totals.max(initial=0.0)) + y_pad * 4.0)

    axes[1].bar(x, plot_df["speedup_vs_single"], color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(plot_df["label"], rotation=15, ha="right")
    axes[1].axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Speedup vs single→single (×)")
    axes[1].set_title("Relative Steady-State Throughput")
    axes[1].grid(axis="y", alpha=0.25)
    speedups = plot_df["speedup_vs_single"].to_numpy(dtype=float)
    speedup_pad = max(0.02, float(speedups.max(initial=0.0)) * 0.03)
    _annotate_totals(
        axes[1],
        x=x,
        totals=speedups,
        labels=[f"{speedup:.2f}×" for speedup in speedups],
        y_pad=speedup_pad,
    )
    axes[1].set_ylim(0.0, max(1.1, float(speedups.max(initial=0.0)) + speedup_pad * 4.0))

    fig.suptitle(
        (
            "Composed Phase-Aware Policy Estimate "
            f"(context={context}, concurrency={concurrency}, output_tokens={output_tokens})"
        ),
        y=1.07,
    )
    fig.text(
        0.5,
        1.01,
        "Composed from measured phase statistics",
        ha="center",
        va="top",
        fontsize=9,
        style="italic",
    )
    fig.text(
        0.5,
        -0.01,
        "Uses measured prefill latency plus measured decode throughput; not a direct end-to-end trace.",
        ha="center",
        va="top",
        fontsize=9,
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _collective_row_lookup(collectives_json: dict[str, Any]) -> dict[tuple[str, str, int, int, int], dict[str, Any]]:
    rows = collectives_json.get("rows", []) if isinstance(collectives_json, dict) else []
    lookup: dict[tuple[str, str, int, int, int], dict[str, Any]] = {}
    for row in rows:
        try:
            # Prefill rows store seq_len with context_len=0; decode rows store context_len.
            context_or_seq = int(row.get("context_len", 0) or 0)
            if context_or_seq == 0:
                context_or_seq = int(row.get("seq_len", 0) or 0)
            key = (
                str(row.get("phase", "")),
                str(row.get("setup", "")),
                context_or_seq,
                int(row.get("batch", 0) or 0),
                int(row.get("decode_steps", 0) or 0),
            )
            lookup[key] = row
        except Exception:
            continue
    return lookup


def _plot_comm_breakdown(
    metrics: pd.DataFrame,
    phase_collectives_json: dict[str, Any],
    out_path: Path,
    kernel_metrics: pd.DataFrame | None = None,
    kernel_collectives_json: dict[str, Any] | None = None,
) -> bool:
    if kernel_metrics is not None and not kernel_metrics.empty and kernel_collectives_json is not None:
        attention = kernel_metrics[kernel_metrics["kernel"] == "attention"].copy()
        if not attention.empty:
            key_lookup = _kernel_collectives_lookup(kernel_collectives_json)
            chosen_seq = int(attention["seq_len"].astype(int).max())
            chosen = attention[attention["seq_len"].astype(int) == chosen_seq].copy()
            bars: list[dict[str, float | str]] = []

            for setup in ["single_die", "dual_die_naive", "dual_die_optimized"]:
                row_df = chosen[chosen["setup"] == setup]
                if row_df.empty:
                    continue
                row = row_df.sort_values(["batch", "model_dim"], ascending=[False, False]).iloc[0]
                key = (
                    str(setup),
                    "attention",
                    int(row.get("batch", 0) or 0),
                    int(row.get("seq_len", 0) or 0),
                    int(row.get("model_dim", 0) or 0),
                )
                collectives_row = key_lookup.get(key, {})
                ops = collectives_row.get("ops", {}) if isinstance(collectives_row, dict) else {}

                label = _display_setup(setup)
                bars.append(
                    {
                        "setup": setup,
                        "label": f"{label}\nattn S={chosen_seq}",
                        "compute_ms": float(row.get("compute_ms_p50", 0.0)),
                        "all_gather_ms": float(((ops.get("all_gather") or {}).get("time_ms_p50", 0.0) or 0.0)),
                        "all_reduce_max_ms": float(((ops.get("all_reduce_max") or {}).get("time_ms_p50", 0.0) or 0.0)),
                        "all_reduce_sum_ms": float(((ops.get("all_reduce_sum") or {}).get("time_ms_p50", 0.0) or 0.0)),
                    }
                )

            if bars:
                plot_df = pd.DataFrame(bars)
                x = np.arange(len(plot_df), dtype=float)

                fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
                bottom = np.zeros(len(plot_df), dtype=float)

                components = [
                    ("compute_ms", "compute", "#4C78A8"),
                    ("all_gather_ms", "all_gather", "#F58518"),
                    ("all_reduce_max_ms", "all_reduce_max", "#54A24B"),
                    ("all_reduce_sum_ms", "all_reduce_sum", "#E45756"),
                ]
                for col, label, color in components:
                    vals = plot_df[col].to_numpy(dtype=float)
                    ax.bar(x, vals, bottom=bottom, label=label, color=color)
                    bottom += vals

                totals = bottom.copy()
                ax.set_xticks(x)
                ax.set_xticklabels(plot_df["label"], rotation=15, ha="right")
                ax.set_ylabel("Latency contribution (ms)")
                ax.set_title(f"Representative Attention Microbenchmark Breakdown (seq={chosen_seq})")
                ax.grid(axis="y", alpha=0.25)
                ax.legend(ncols=2)
                y_pad = max(1.0, float(totals.max(initial=0.0)) * 0.04)
                for idx, bar_row in plot_df.iterrows():
                    total = float(
                        bar_row["compute_ms"]
                        + bar_row["all_gather_ms"]
                        + bar_row["all_reduce_max_ms"]
                        + bar_row["all_reduce_sum_ms"]
                    )
                    comm = float(
                        bar_row["all_gather_ms"]
                        + bar_row["all_reduce_max_ms"]
                        + bar_row["all_reduce_sum_ms"]
                    )
                    comm_share = (comm / total * 100.0) if total > 0 else 0.0
                    ax.text(
                        float(x[idx]),
                        float(total) + y_pad,
                        f"{total:.1f} ms",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        fontweight="bold",
                    )
                    ax.text(
                        float(x[idx]),
                        float(total) + y_pad * 2.1,
                        f"{comm_share:.0f}% comm",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
                ax.set_ylim(0.0, float(totals.max(initial=0.0)) + y_pad * 5.0)
                fig.savefig(out_path, dpi=180, bbox_inches="tight")
                plt.close(fig)
                return True

    decode = metrics[metrics["phase"] == "decode"].copy()
    if decode.empty:
        return False

    lookup = _collective_row_lookup(phase_collectives_json)
    bars: list[dict[str, float | str]] = []

    for setup in SETUP_ORDER:
        sub = decode[decode["setup"] == setup]
        if sub.empty:
            continue
        # Representative high-pressure point: largest context, then highest concurrency.
        sub = sub.sort_values(["context_len", "batch"], ascending=[False, False])
        row = sub.iloc[0]

        key = (
            "decode",
            setup,
            int(row.get("context_len", 0) or 0),
            int(row.get("batch", 0) or 0),
            int(row.get("decode_steps", 0) or 0),
        )
        collectives_row = lookup.get(key, {})
        ops = collectives_row.get("ops", {}) if isinstance(collectives_row, dict) else {}

        all_gather_ms = float(((ops.get("all_gather") or {}).get("time_ms_p50", 0.0) or 0.0))
        all_reduce_max_ms = float(((ops.get("all_reduce_max") or {}).get("time_ms_p50", 0.0) or 0.0))
        all_reduce_sum_ms = float(((ops.get("all_reduce_sum") or {}).get("time_ms_p50", 0.0) or 0.0))

        bars.append(
            {
                "setup": setup,
                "label": f"{_display_setup(setup)}\nctx={int(row['context_len'])},C={int(row['batch'])}",
                "compute_ms": float(row.get("compute_ms_p50", 0.0)),
                "all_gather_ms": all_gather_ms,
                "all_reduce_max_ms": all_reduce_max_ms,
                "all_reduce_sum_ms": all_reduce_sum_ms,
            }
        )

    if not bars:
        return False

    plot_df = pd.DataFrame(bars)
    x = np.arange(len(plot_df), dtype=float)

    fig, ax = plt.subplots(figsize=(max(8.5, len(plot_df) * 2.4), 5), constrained_layout=True)
    bottom = np.zeros(len(plot_df), dtype=float)

    components = [
        ("compute_ms", "compute", "#4C78A8"),
        ("all_gather_ms", "all_gather", "#F58518"),
        ("all_reduce_max_ms", "all_reduce_max", "#54A24B"),
        ("all_reduce_sum_ms", "all_reduce_sum", "#E45756"),
    ]
    for col, label, color in components:
        vals = plot_df[col].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, label=label, color=color)
        bottom += vals

    totals = bottom.copy()
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=20, ha="right")
    ax.set_ylabel("Latency contribution (ms)")
    ax.set_title("Representative Decode-Phase Breakdown")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=2)
    y_pad = max(1.0, float(totals.max(initial=0.0)) * 0.03)
    total_labels = []
    for _, bar_row in plot_df.iterrows():
        total = float(
            bar_row["compute_ms"]
            + bar_row["all_gather_ms"]
            + bar_row["all_reduce_max_ms"]
            + bar_row["all_reduce_sum_ms"]
        )
        comm = float(bar_row["all_gather_ms"] + bar_row["all_reduce_max_ms"] + bar_row["all_reduce_sum_ms"])
        comm_share = (comm / total * 100.0) if total > 0 else 0.0
        total_labels.append(f"{total:.1f} ms\n{comm_share:.0f}% comm")
    _annotate_totals(ax, x=x, totals=totals, labels=total_labels, y_pad=y_pad)
    ax.set_ylim(0.0, float(totals.max(initial=0.0)) + y_pad * 3.6)

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _kernel_collectives_lookup(collectives_json: dict[str, Any]) -> dict[tuple[str, str, int, int, int], dict[str, Any]]:
    rows = collectives_json.get("rows", []) if isinstance(collectives_json, dict) else []
    out: dict[tuple[str, str, int, int, int], dict[str, Any]] = {}
    for row in rows:
        try:
            key = (
                str(row.get("setup", "")),
                str(row.get("kernel", "")),
                int(row.get("batch", 0) or 0),
                int(row.get("seq_len", 0) or 0),
                int(row.get("model_dim", 0) or 0),
            )
            out[key] = row
        except Exception:
            continue
    return out


def _plot_collective_scatter(
    kernel_metrics: pd.DataFrame,
    kernel_collectives_json: dict[str, Any],
    out_path: Path,
) -> bool:
    if kernel_metrics.empty:
        return False

    lookup = _kernel_collectives_lookup(kernel_collectives_json)

    rows: list[dict[str, float | str]] = []
    for _, row in kernel_metrics.iterrows():
        key = (
            str(row.get("setup", "")),
            str(row.get("kernel", "")),
            int(row.get("batch", 0) or 0),
            int(row.get("seq_len", 0) or 0),
            int(row.get("model_dim", 0) or 0),
        )
        c_row = lookup.get(key, {})
        ops = c_row.get("ops", {}) if isinstance(c_row, dict) else {}

        count = 0.0
        bytes_p50 = 0.0
        for op in ops.values() if isinstance(ops, dict) else []:
            count += float((op or {}).get("count_p50", 0.0) or 0.0)
            bytes_p50 += float((op or {}).get("bytes_p50", 0.0) or 0.0)

        rows.append(
            {
                "setup": str(row.get("setup", "")),
                "kernel": str(row.get("kernel", "")),
                "seq_len": int(row.get("seq_len", 0) or 0),
                "collective_count": count,
                "latency_ms_p50": float(row.get("latency_ms_p50", 0.0) or 0.0),
                "bytes_p50": bytes_p50,
            }
        )

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return False

    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)

    for setup, sub in plot_df.groupby("setup"):
        color = SETUP_COLORS.get(setup, "#888888")
        sizes = 25.0 + (sub["bytes_p50"].to_numpy(dtype=float) / 200000.0)
        ax.scatter(
            sub["collective_count"],
            sub["latency_ms_p50"],
            s=sizes,
            alpha=0.7,
            label=_display_setup(str(setup)),
            color=color,
            edgecolors="black",
            linewidths=0.25,
        )

    hot_rows: list[pd.Series] = []
    tensor_rows = plot_df[
        (plot_df["setup"] == "dual_die_optimized") & (plot_df["collective_count"] > 0)
    ].sort_values(["latency_ms_p50", "collective_count"], ascending=False)
    naive_rows = plot_df[
        (plot_df["setup"] == "dual_die_naive") & (plot_df["collective_count"] > 0)
    ].sort_values(["latency_ms_p50", "collective_count"], ascending=False)
    if not tensor_rows.empty:
        hot_rows.append(tensor_rows.iloc[0])
    if not naive_rows.empty:
        hot_rows.append(naive_rows.iloc[0])
    for row in hot_rows:
        ax.annotate(
            f"{_display_setup(str(row['setup']))}:{row['kernel']}@S{int(row['seq_len'])}",
            (float(row["collective_count"]), float(row["latency_ms_p50"])),
            fontsize=7,
            alpha=0.85,
            xytext=(5, 4),
            textcoords="offset points",
        )

    ax.set_xlabel("Collective count (p50 per iteration)")
    ax.set_ylabel("Kernel latency p50 (ms)")
    ax.set_title("Attention Latency Rises with Collective Count, Not Just Bytes Moved")
    if (plot_df["collective_count"] > 0).any():
        ax.set_xscale("symlog", linthresh=1.0)
    ax.grid(alpha=0.25)
    legend1 = ax.legend(ncols=2, fontsize=8)
    # Bubble-size cue: bytes moved. This makes the "count vs bytes" point legible.
    size_bytes = [1e6, 1e7, 1e8]
    size_handles = [
        ax.scatter(
            [],
            [],
            s=25.0 + (b / 200000.0),
            alpha=0.35,
            color="#9c9c9c",
            edgecolors="black",
            linewidths=0.25,
        )
        for b in size_bytes
    ]
    size_labels = ["1 MB", "10 MB", "100 MB"]
    _ = ax.legend(
        size_handles,
        size_labels,
        title="Bytes (p50)",
        loc="lower right",
        fontsize=7,
        title_fontsize=7,
        frameon=True,
    )
    ax.add_artist(legend1)

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_prefill_collective_bytes_vs_time(
    phase_collectives_json: dict[str, Any],
    out_path: Path,
) -> bool:
    rows = phase_collectives_json.get("rows", []) if isinstance(phase_collectives_json, dict) else []
    candidates = [
        row
        for row in rows
        if str(row.get("phase", "")) == "prefill"
        and str(row.get("setup", "")) == "dual_die_tensor_optimized"
        and isinstance(row.get("ops", {}), dict)
        and bool(row.get("ops", {}))
    ]
    if not candidates:
        return False

    def _key(row: dict[str, Any]) -> tuple[int, int]:
        return (
            int(row.get("seq_len", 0) or 0),
            int(row.get("batch", 0) or 0),
        )

    chosen = sorted(candidates, key=_key, reverse=True)[0]
    seq_len = int(chosen.get("seq_len", 0) or 0)
    batch = int(chosen.get("batch", 0) or 0)
    ops = chosen.get("ops", {}) if isinstance(chosen.get("ops", {}), dict) else {}
    if not ops:
        return False

    color_map = {
        "all_gather": "#F58518",
        "all_reduce_max": "#54A24B",
        "all_reduce_sum": "#E45756",
    }

    points: list[dict[str, float | str]] = []
    for op_name, payload in ops.items():
        if not isinstance(payload, dict):
            continue
        bytes_total = float(payload.get("bytes_total", 0.0) or 0.0)
        time_ms_total = float(payload.get("time_ms_total", 0.0) or 0.0)
        count_total = float(payload.get("count_total", 0.0) or 0.0)
        if bytes_total <= 0.0 or time_ms_total <= 0.0:
            continue
        eff_gbps = (bytes_total / (time_ms_total / 1000.0) / 1e9) if time_ms_total > 0 else 0.0
        points.append(
            {
                "op": str(op_name),
                "bytes_mb": bytes_total / (1024.0 * 1024.0),
                "time_ms": time_ms_total,
                "count": count_total,
                "eff_gbps": eff_gbps,
            }
        )

    if not points:
        return False

    plot_df = pd.DataFrame(points).sort_values("time_ms", ascending=False)
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)

    for _, row in plot_df.iterrows():
        op = str(row["op"])
        x = float(row["bytes_mb"])
        y = float(row["time_ms"])
        ax.scatter(
            x,
            y,
            s=140.0,
            color=color_map.get(op, "#808080"),
            alpha=0.85,
            edgecolors="black",
            linewidths=0.35,
        )
        ax.annotate(
            f"{op}\n{row['count']:.0f} calls\n{row['eff_gbps']:.3f} GB/s",
            (x, y),
            fontsize=8,
            xytext=(8, 6),
            textcoords="offset points",
        )

    # Reference bandwidth lines (GB/s): time_ms = bytes_mb * 1000 / (1024 * bw)
    x_min = max(1e-3, float(plot_df["bytes_mb"].min()))
    x_max = float(plot_df["bytes_mb"].max()) * 1.2
    xs = np.logspace(np.log10(x_min), np.log10(x_max), 100)
    for bw in (0.01, 0.1, 1.0, 10.0):
        ys = xs * (1000.0 / (1024.0 * bw))
        ax.plot(xs, ys, linestyle="--", linewidth=1.0, color="#999999", alpha=0.6)
        ax.text(xs[-1], ys[-1], f"{bw:g} GB/s", fontsize=7, color="#777777", ha="left", va="center")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total collective bytes (MB)")
    ax.set_ylabel("Total collective time (ms)")
    ax.set_title(f"Prefill Tensor-Split Collective Cost (seq={seq_len}, B={batch})")
    ax.grid(alpha=0.25, which="both")

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> int:
    args = parse_args()
    if not args.phase_metrics_csv.exists():
        raise SystemExit(f"Missing phase metrics CSV: {args.phase_metrics_csv}")

    phase_collectives_json = args.phase_collectives_json or (args.phase_metrics_csv.parent / "collectives_summary.json")

    phase_metrics = _safe_read_csv(args.phase_metrics_csv)
    phase_collectives = _safe_read_json(phase_collectives_json)

    kernel_metrics = _safe_read_csv(args.kernel_metrics_csv) if args.kernel_metrics_csv else pd.DataFrame()
    if args.kernel_collectives_json:
        kernel_collectives = _safe_read_json(args.kernel_collectives_json)
    elif args.kernel_metrics_csv:
        kernel_collectives = _safe_read_json(args.kernel_metrics_csv.parent / "collectives_summary.json")
    else:
        kernel_collectives = {}

    args.out_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        "decode_slo_frontier": args.out_dir / f"{args.prefix}_decode_slo_frontier.png",
        "crossover_heatmap": args.out_dir / f"{args.prefix}_crossover_heatmap.png",
        "prefill_ratio": args.out_dir / f"{args.prefix}_prefill_ratio.png",
        "phase_tradeoff": args.out_dir / f"{args.prefix}_phase_tradeoff.png",
        "hybrid_e2e": args.out_dir / f"{args.prefix}_hybrid_e2e.png",
        "comm_breakdown": args.out_dir / f"{args.prefix}_comm_breakdown.png",
        "collective_scatter": args.out_dir / f"{args.prefix}_collective_count_vs_latency.png",
        "prefill_collectives": args.out_dir / f"{args.prefix}_prefill_collectives_bytes_vs_time.png",
    }

    generated: list[Path] = []
    if _plot_decode_slo_frontier(phase_metrics, targets["decode_slo_frontier"], args.heatmap_slo_ms):
        generated.append(targets["decode_slo_frontier"])
    if _plot_crossover_heatmap(phase_metrics, targets["crossover_heatmap"], args.heatmap_slo_ms):
        generated.append(targets["crossover_heatmap"])
    if _plot_comm_breakdown(
        phase_metrics,
        phase_collectives,
        targets["comm_breakdown"],
        kernel_metrics=kernel_metrics,
        kernel_collectives_json=kernel_collectives,
    ):
        generated.append(targets["comm_breakdown"])

    if not args.top3_only:
        if _plot_prefill_latency_ratio(phase_metrics, targets["prefill_ratio"]):
            generated.append(targets["prefill_ratio"])
        if _plot_phase_tradeoff(phase_metrics, targets["phase_tradeoff"]):
            generated.append(targets["phase_tradeoff"])
        if _plot_prefill_collective_bytes_vs_time(phase_collectives, targets["prefill_collectives"]):
            generated.append(targets["prefill_collectives"])
        if _plot_hybrid_end_to_end(
            phase_metrics,
            targets["hybrid_e2e"],
            output_tokens=args.hybrid_output_tokens,
            context_override=args.hybrid_context_len,
            conc_override=args.hybrid_concurrency,
        ):
            generated.append(targets["hybrid_e2e"])
        if _plot_collective_scatter(kernel_metrics, kernel_collectives, targets["collective_scatter"]):
            generated.append(targets["collective_scatter"])

    if not generated:
        raise SystemExit("No best-graph figures were generated.")

    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
