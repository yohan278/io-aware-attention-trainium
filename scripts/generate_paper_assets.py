#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
FIG_DIR = PAPER_DIR / "figures"
DATA_DIR = PAPER_DIR / "data"
TABLE_DIR = PAPER_DIR / "tables"
PLOTS_DIR = ROOT / "results" / "plots"

SERVICE_METRICS = ROOT / "results/trn2-phase-inference-quick-fast/run_20260305T224828Z/metrics.csv"
TRACE_SUMMARY = ROOT / "results/plots/public_service_service_trace_summary.csv"
MOE_METRICS = ROOT / "results/trn2-moe-stable-small-merged-mask23/run_20260306T100500Z/metrics.csv"


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _compute_headline_metrics() -> dict[str, object]:
    phase_rows = _load_csv(SERVICE_METRICS)
    trace_rows = _load_csv(TRACE_SUMMARY)
    moe_rows = _load_csv(MOE_METRICS)

    phase = {
        (row["phase"], row["setup"], int(row["batch"]), int(row["seq_len"]), int(row["context_len"])): row
        for row in phase_rows
    }

    out: dict[str, object] = {
        "chiplet_proxy_example": {
            "seq_len": 16,
            "model_dim": 8,
            "value_dim": 8,
            "seed": 7,
            "max_abs_error": 2.082e-17,
            "state_words": 320,
            "score_only_words": 256,
            "score_value_words": 2304,
            "score_only_ratio_vs_state": 0.80,
            "score_value_ratio_vs_state": 7.20,
        }
    }

    for concurrency in (8, 16):
        single = phase[("decode", "single_die", concurrency, 1, 2048)]
        request = phase[("decode", "dual_die_request_sharded", concurrency, 1, 2048)]
        tensor = phase[("decode", "dual_die_tensor_optimized", concurrency, 1, 2048)]
        out[f"decode_c{concurrency}"] = {
            "single_p50_ms": float(single["latency_ms_p50"]),
            "single_tok_s": float(single["throughput_tokens_per_s"]),
            "request_p50_ms": float(request["latency_ms_p50"]),
            "request_tok_s": float(request["throughput_tokens_per_s"]),
            "tensor_p50_ms": float(tensor["latency_ms_p50"]),
            "tensor_tok_s": float(tensor["throughput_tokens_per_s"]),
            "request_latency_ratio_vs_single": float(request["latency_ms_p50"]) / float(single["latency_ms_p50"]),
            "request_throughput_ratio_vs_single": float(request["throughput_tokens_per_s"])
            / float(single["throughput_tokens_per_s"]),
            "tensor_latency_ratio_vs_single": float(tensor["latency_ms_p50"]) / float(single["latency_ms_p50"]),
            "tensor_throughput_ratio_vs_single": float(tensor["throughput_tokens_per_s"])
            / float(single["throughput_tokens_per_s"]),
        }

    single = phase[("prefill", "single_die", 2, 4096, 0)]
    request = phase[("prefill", "dual_die_request_sharded", 2, 4096, 0)]
    tensor = phase[("prefill", "dual_die_tensor_optimized", 2, 4096, 0)]
    out["prefill_s4096"] = {
        "single_p50_ms": float(single["latency_ms_p50"]),
        "request_p50_ms": float(request["latency_ms_p50"]),
        "tensor_p50_ms": float(tensor["latency_ms_p50"]),
        "request_latency_ratio_vs_single": float(request["latency_ms_p50"]) / float(single["latency_ms_p50"]),
        "tensor_latency_ratio_vs_single": float(tensor["latency_ms_p50"]) / float(single["latency_ms_p50"]),
    }

    trace = {row["policy"]: row for row in trace_rows}
    single_policy = trace["single->single"]
    out["single->single"] = {
        "goodput_tokens_per_s": float(single_policy["goodput_tokens_per_s"]),
        "goodput_ratio_vs_single": 1.0,
        "on_time_ratio": float(single_policy["on_time_ratio"]),
        "p90_ms": float(single_policy["latency_ms_p90"]),
    }
    for policy in ("single->request", "request->request", "single->tensor"):
        row = trace[policy]
        out[policy] = {
            "goodput_tokens_per_s": float(row["goodput_tokens_per_s"]),
            "goodput_ratio_vs_single": float(row["goodput_tokens_per_s"])
            / float(single_policy["goodput_tokens_per_s"]),
            "on_time_ratio": float(row["on_time_ratio"]),
            "p90_ms": float(row["latency_ms_p90"]),
        }

    moe_pairs: dict[tuple[int, int, int, int, int, float], dict[str, dict[str, str]]] = {}
    for row in moe_rows:
        if row["phase"] != "decode":
            continue
        key = (
            int(row["context_len"]),
            int(row["batch"]),
            int(row["decode_steps"]),
            int(row["num_experts"]),
            int(row["top_k"]),
            float(row["routing_skew"]),
        )
        moe_pairs.setdefault(key, {})[row["setup"]] = row

    locality_speedups: list[float] = []
    remote_reductions: list[float] = []
    for pair in moe_pairs.values():
        if "dual_die_moe_locality" not in pair or "dual_die_moe_naive" not in pair:
            continue
        locality = pair["dual_die_moe_locality"]
        naive = pair["dual_die_moe_naive"]
        locality_speedups.append(
            float(locality["throughput_tokens_per_s"]) / float(naive["throughput_tokens_per_s"])
        )
        remote_reductions.append(
            float(naive["remote_dispatch_ratio_p50"]) - float(locality["remote_dispatch_ratio_p50"])
        )

    out["moe_secondary"] = {
        "median_locality_gain": float(np.median(locality_speedups)),
        "median_remote_dispatch_reduction_abs": float(np.median(remote_reductions)),
    }
    return out


def _save_metrics(metrics: dict[str, object]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "headline_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    example = metrics["chiplet_proxy_example"]
    with (DATA_DIR / "chiplet_proxy_example.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in example.items():
            writer.writerow([key, value])


def _draw_box(
    ax: plt.Axes,
    *,
    xy: tuple[float, float],
    width: float,
    height: float,
    color: str,
    title: str,
    lines: list[str],
    edgecolor: str = "#3a3a3a",
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.4,
        facecolor=color,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    left, bottom = xy
    ax.text(left + width / 2, bottom + height * 0.78, title, ha="center", va="center", fontsize=13, weight="bold")
    ax.text(left + width / 2, bottom + height * 0.44, "\n".join(lines), ha="center", va="center", fontsize=10)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], text: str = "") -> None:
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, linewidth=1.5, color="#444444")
    ax.add_patch(arrow)
    if text:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.025, text, ha="center", va="bottom", fontsize=9)


def _plot_project_overview() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.03,
        0.96,
        "End-to-End Paper Evidence Pipeline",
        fontsize=22,
        weight="bold",
        ha="left",
        va="top",
    )

    _draw_box(
        ax,
        xy=(0.04, 0.67),
        width=0.24,
        height=0.18,
        color="#E8DDF4",
        title="Algorithmic Layer",
        lines=[
            "Exact SDPA baseline",
            "State-only chiplet proxy",
            "Communication formulas",
        ],
    )
    _draw_box(
        ax,
        xy=(0.37, 0.67),
        width=0.26,
        height=0.18,
        color="#DCEAF7",
        title="Trainium Phase Harness",
        lines=[
            "single / dual-request / dual-tensor",
            "prefill vs decode",
            "latency, throughput, collectives",
        ],
    )
    _draw_box(
        ax,
        xy=(0.72, 0.67),
        width=0.24,
        height=0.18,
        color="#D9F1E3",
        title="Policy Layer",
        lines=[
            "Composed hybrid estimate",
            "Mixed-traffic simulation",
            "Measured / composed / simulated",
        ],
    )

    _arrow(ax, (0.28, 0.76), (0.37, 0.76), "state-minimal attention")
    _arrow(ax, (0.63, 0.76), (0.72, 0.76), "phase-aware policy")

    story_patch = FancyBboxPatch(
        (0.04, 0.19),
        0.92,
        0.34,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        facecolor="#F8EFD9",
        edgecolor="#7A6B39",
        linestyle="--",
    )
    ax.add_patch(story_patch)
    ax.text(0.06, 0.5, "Paper Outputs", fontsize=18, weight="bold", ha="left", va="center")

    _draw_box(
        ax,
        xy=(0.08, 0.28),
        width=0.21,
        height=0.17,
        color="#FFF6E8",
        title="Measured Results",
        lines=[
            "Decode frontier",
            "Prefill ratio",
            "Comm breakdown",
        ],
        edgecolor="#9A7C39",
    )
    _draw_box(
        ax,
        xy=(0.39, 0.28),
        width=0.22,
        height=0.17,
        color="#FFF6E8",
        title="Composed Results",
        lines=[
            "Hybrid request-time estimate",
            "Policy throughput speedup",
        ],
        edgecolor="#9A7C39",
    )
    _draw_box(
        ax,
        xy=(0.70, 0.28),
        width=0.22,
        height=0.17,
        color="#FFF6E8",
        title="Simulated Results",
        lines=[
            "Mixed-traffic goodput",
            "On-time service ratio",
        ],
        edgecolor="#9A7C39",
    )

    _arrow(ax, (0.5, 0.67), (0.5, 0.53), "evidence")
    _arrow(ax, (0.19, 0.45), (0.19, 0.28))
    _arrow(ax, (0.5, 0.45), (0.5, 0.28))
    _arrow(ax, (0.81, 0.45), (0.81, 0.28))

    ax.text(
        0.5,
        0.12,
        "Core paper thesis: dual-die value comes from request-sharded decode serving, not tensor-parallel splitting of one request.",
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
    )

    fig.savefig(FIG_DIR / "project_overview_pipeline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_chiplet_comm_scaling() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    seq = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096], dtype=float)
    value_dim = 8.0
    state_words = seq * (2.0 * (2.0 + value_dim))
    score_only_words = seq**2
    score_value_words = seq**2 * (1.0 + value_dim)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    axes[0].plot(seq, state_words, marker="o", linewidth=2.4, label="state only", color="#2ca02c")
    axes[0].plot(seq, score_only_words, marker="o", linewidth=2.0, label="score only", color="#1f77b4")
    axes[0].plot(seq, score_value_words, marker="o", linewidth=2.0, label="score + value", color="#d62728")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log", base=10)
    axes[0].set_xlabel("Sequence length S")
    axes[0].set_ylabel("Cross-partition words exchanged")
    axes[0].set_title("Absolute Communication Scaling (Dv=8, two partitions)")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend()

    axes[1].plot(seq, score_only_words / state_words, marker="o", linewidth=2.0, label="score-only / state", color="#1f77b4")
    axes[1].plot(
        seq,
        score_value_words / state_words,
        marker="o",
        linewidth=2.4,
        label="score+value / state",
        color="#d62728",
    )
    axes[1].axhline(1.0, linestyle="--", linewidth=1.0, color="#333333")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log", base=10)
    axes[1].set_xlabel("Sequence length S")
    axes[1].set_ylabel("Communication ratio")
    axes[1].set_title("How Much Worse Naive Exchange Becomes")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend()

    fig.savefig(FIG_DIR / "chiplet_comm_scaling.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_headline_metrics(metrics: dict[str, object]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    decode16 = metrics["decode_c16"]
    prefill = metrics["prefill_s4096"]
    single_request = metrics["single->request"]
    request_request = metrics["request->request"]
    moe = metrics["moe_secondary"]

    labels = [
        "decode\nrequest/single\nthroughput",
        "decode\ntensor/single\nthroughput",
        "prefill\nrequest/single\nlatency",
        "mixed traffic\nsingle→request\ngoodput",
        "mixed traffic\nrequest→request\ngoodput",
        "MoE\nlocality/naive\nthroughput",
    ]
    values = np.array(
        [
            float(decode16["request_throughput_ratio_vs_single"]),
            float(decode16["tensor_throughput_ratio_vs_single"]),
            float(prefill["request_latency_ratio_vs_single"]),
            float(single_request["goodput_ratio_vs_single"]),
            float(request_request["goodput_ratio_vs_single"]),
            float(moe["median_locality_gain"]),
        ],
        dtype=float,
    )
    colors = ["#2ca02c", "#ff7f0e", "#1f77b4", "#2ca02c", "#1b7f3a", "#76b7b2"]

    fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    x = np.arange(len(labels), dtype=float)
    ax.bar(x, values, color=colors)
    ax.axhline(1.0, linestyle="--", linewidth=1.0, color="#333333")
    ax.set_ylabel("Ratio relative to comparison baseline")
    ax.set_title("Headline Ratios Used in the Paper Story")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)

    for idx, value in enumerate(values):
        ax.text(float(x[idx]), float(value) + 0.03, f"{value:.2f}×", ha="center", va="bottom", fontsize=9)

    fig.savefig(FIG_DIR / "headline_metrics_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _fmt(x: float, *, digits: int = 1) -> str:
    return f"{x:.{digits}f}"


def _write_headline_table_tex(metrics: dict[str, object]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    d8 = metrics["decode_c8"]
    d16 = metrics["decode_c16"]
    prefill = metrics["prefill_s4096"]
    sim_single = metrics["single->single"]
    sim_sr = metrics["single->request"]
    sim_rr = metrics["request->request"]
    sim_st = metrics["single->tensor"]

    # One table environment containing two side-by-side tabular blocks.
    tex = r"""\begin{table*}[t]
  \centering
  \caption{Headline results from committed artifacts. Phase results are measured on Trn2. Mixed-traffic results are simulated from measured phase profiles. Ratios are relative to the baseline in the same block.}
  \label{tab:headline}
  \footnotesize
  \begin{minipage}{0.62\textwidth}
    \centering
    \begin{tabular}{lcccc}
      \toprule
      Phase & Metric & Single & Dual-request & Dual-tensor \\
      \midrule
      Decode (ctx=2048, C=16) & tok/s & %(d16_single_tok)s & %(d16_req_tok)s (\textbf{%(d16_req_tok_ratio)s}$\times$) & %(d16_ten_tok)s (\textbf{%(d16_ten_tok_ratio)s}$\times$) \\
      & p50 ms & %(d16_single_p50)s & %(d16_req_p50)s (\textbf{%(d16_req_p50_ratio)s}$\times$) & %(d16_ten_p50)s (\textbf{%(d16_ten_p50_ratio)s}$\times$) \\
      Decode (ctx=2048, C=8) & tok/s & %(d8_single_tok)s & %(d8_req_tok)s (%(d8_req_tok_ratio)s$\times$) & %(d8_ten_tok)s (\textbf{%(d8_ten_tok_ratio)s}$\times$) \\
      & p50 ms & %(d8_single_p50)s & %(d8_req_p50)s (%(d8_req_p50_ratio)s$\times$) & %(d8_ten_p50)s (\textbf{%(d8_ten_p50_ratio)s}$\times$) \\
      Prefill (S=4096, B=2) & p50 ms & %(p_single_p50)s & %(p_req_p50)s (%(p_req_p50_ratio)s$\times$) & %(p_ten_p50)s (\textbf{%(p_ten_p50_ratio)s}$\times$) \\
      \bottomrule
    \end{tabular}
  \end{minipage}
  \hfill
  \begin{minipage}{0.36\textwidth}
    \centering
    \begin{tabular}{lcccc}
      \toprule
      Policy (sim) & Goodput tok/s & vs single & On-time & p90 ms \\
      \midrule
      single$\rightarrow$single & %(sim_single_goodput)s & 1.00$\times$ & %(sim_single_ontime)s\%% & %(sim_single_p90)s \\
      single$\rightarrow$request & %(sim_sr_goodput)s & \textbf{%(sim_sr_ratio)s}$\times$ & \textbf{%(sim_sr_ontime)s}\%% & %(sim_sr_p90)s \\
      request$\rightarrow$request & %(sim_rr_goodput)s & %(sim_rr_ratio)s$\times$ & %(sim_rr_ontime)s\%% & %(sim_rr_p90)s \\
      single$\rightarrow$tensor & %(sim_st_goodput)s & \textbf{%(sim_st_ratio)s}$\times$ & %(sim_st_ontime)s\%% & %(sim_st_p90)s \\
      \bottomrule
    \end{tabular}
  \end{minipage}
\end{table*}
"""

    payload = {
        "d16_single_tok": _fmt(float(d16["single_tok_s"]), digits=1),
        "d16_req_tok": _fmt(float(d16["request_tok_s"]), digits=1),
        "d16_ten_tok": _fmt(float(d16["tensor_tok_s"]), digits=1),
        "d16_req_tok_ratio": _fmt(float(d16["request_throughput_ratio_vs_single"]), digits=2),
        "d16_ten_tok_ratio": _fmt(float(d16["tensor_throughput_ratio_vs_single"]), digits=2),
        "d16_single_p50": _fmt(float(d16["single_p50_ms"]), digits=1),
        "d16_req_p50": _fmt(float(d16["request_p50_ms"]), digits=1),
        "d16_ten_p50": _fmt(float(d16["tensor_p50_ms"]), digits=1),
        "d16_req_p50_ratio": _fmt(float(d16["request_latency_ratio_vs_single"]), digits=2),
        "d16_ten_p50_ratio": _fmt(float(d16["tensor_latency_ratio_vs_single"]), digits=2),
        "d8_single_tok": _fmt(float(d8["single_tok_s"]), digits=1),
        "d8_req_tok": _fmt(float(d8["request_tok_s"]), digits=1),
        "d8_ten_tok": _fmt(float(d8["tensor_tok_s"]), digits=1),
        "d8_req_tok_ratio": _fmt(float(d8["request_throughput_ratio_vs_single"]), digits=2),
        "d8_ten_tok_ratio": _fmt(float(d8["tensor_throughput_ratio_vs_single"]), digits=2),
        "d8_single_p50": _fmt(float(d8["single_p50_ms"]), digits=1),
        "d8_req_p50": _fmt(float(d8["request_p50_ms"]), digits=1),
        "d8_ten_p50": _fmt(float(d8["tensor_p50_ms"]), digits=1),
        "d8_req_p50_ratio": _fmt(float(d8["request_latency_ratio_vs_single"]), digits=2),
        "d8_ten_p50_ratio": _fmt(float(d8["tensor_latency_ratio_vs_single"]), digits=2),
        "p_single_p50": _fmt(float(prefill["single_p50_ms"]), digits=1),
        "p_req_p50": _fmt(float(prefill["request_p50_ms"]), digits=1),
        "p_ten_p50": _fmt(float(prefill["tensor_p50_ms"]), digits=1),
        "p_req_p50_ratio": _fmt(float(prefill["request_latency_ratio_vs_single"]), digits=2),
        "p_ten_p50_ratio": _fmt(float(prefill["tensor_latency_ratio_vs_single"]), digits=1),
        "sim_single_goodput": _fmt(float(sim_single["goodput_tokens_per_s"]), digits=0),
        "sim_single_ontime": _fmt(float(sim_single["on_time_ratio"]) * 100.0, digits=1),
        "sim_single_p90": _fmt(float(sim_single["p90_ms"]), digits=1),
        "sim_sr_goodput": _fmt(float(sim_sr["goodput_tokens_per_s"]), digits=0),
        "sim_sr_ratio": _fmt(float(sim_sr["goodput_ratio_vs_single"]), digits=2),
        "sim_sr_ontime": _fmt(float(sim_sr["on_time_ratio"]) * 100.0, digits=1),
        "sim_sr_p90": _fmt(float(sim_sr["p90_ms"]), digits=1),
        "sim_rr_goodput": _fmt(float(sim_rr["goodput_tokens_per_s"]), digits=0),
        "sim_rr_ratio": _fmt(float(sim_rr["goodput_ratio_vs_single"]), digits=2),
        "sim_rr_ontime": _fmt(float(sim_rr["on_time_ratio"]) * 100.0, digits=1),
        "sim_rr_p90": _fmt(float(sim_rr["p90_ms"]), digits=1),
        "sim_st_goodput": _fmt(float(sim_st["goodput_tokens_per_s"]), digits=0),
        "sim_st_ratio": _fmt(float(sim_st["goodput_ratio_vs_single"]), digits=2),
        "sim_st_ontime": _fmt(float(sim_st["on_time_ratio"]) * 100.0, digits=1),
        "sim_st_p90": _fmt(float(sim_st["p90_ms"]), digits=1),
    }
    (TABLE_DIR / "headline_table.tex").write_text(tex % payload, encoding="utf-8")


def _plot_headline_table_png(metrics: dict[str, object]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    d8 = metrics["decode_c8"]
    d16 = metrics["decode_c16"]
    prefill = metrics["prefill_s4096"]
    sim_single = metrics["single->single"]
    sim_sr = metrics["single->request"]
    sim_rr = metrics["request->request"]
    sim_st = metrics["single->tensor"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.4), constrained_layout=True)
    for ax in axes:
        ax.axis("off")

    phase_cols = ["Phase", "Metric", "Single", "Dual-request", "Dual-tensor"]
    phase_rows = [
        [
            "Decode (ctx=2048,C=16)",
            "tok/s",
            _fmt(float(d16["single_tok_s"]), digits=1),
            f'{_fmt(float(d16["request_tok_s"]),digits=1)} ({_fmt(float(d16["request_throughput_ratio_vs_single"]),digits=2)}x)',
            f'{_fmt(float(d16["tensor_tok_s"]),digits=1)} ({_fmt(float(d16["tensor_throughput_ratio_vs_single"]),digits=2)}x)',
        ],
        [
            "",
            "p50 ms",
            _fmt(float(d16["single_p50_ms"]), digits=1),
            f'{_fmt(float(d16["request_p50_ms"]),digits=1)} ({_fmt(float(d16["request_latency_ratio_vs_single"]),digits=2)}x)',
            f'{_fmt(float(d16["tensor_p50_ms"]),digits=1)} ({_fmt(float(d16["tensor_latency_ratio_vs_single"]),digits=2)}x)',
        ],
        [
            "Decode (ctx=2048,C=8)",
            "tok/s",
            _fmt(float(d8["single_tok_s"]), digits=1),
            f'{_fmt(float(d8["request_tok_s"]),digits=1)} ({_fmt(float(d8["request_throughput_ratio_vs_single"]),digits=2)}x)',
            f'{_fmt(float(d8["tensor_tok_s"]),digits=1)} ({_fmt(float(d8["tensor_throughput_ratio_vs_single"]),digits=2)}x)',
        ],
        [
            "",
            "p50 ms",
            _fmt(float(d8["single_p50_ms"]), digits=1),
            f'{_fmt(float(d8["request_p50_ms"]),digits=1)} ({_fmt(float(d8["request_latency_ratio_vs_single"]),digits=2)}x)',
            f'{_fmt(float(d8["tensor_p50_ms"]),digits=1)} ({_fmt(float(d8["tensor_latency_ratio_vs_single"]),digits=2)}x)',
        ],
        [
            "Prefill (S=4096,B=2)",
            "p50 ms",
            _fmt(float(prefill["single_p50_ms"]), digits=1),
            f'{_fmt(float(prefill["request_p50_ms"]),digits=1)} ({_fmt(float(prefill["request_latency_ratio_vs_single"]),digits=2)}x)',
            f'{_fmt(float(prefill["tensor_p50_ms"]),digits=1)} ({_fmt(float(prefill["tensor_latency_ratio_vs_single"]),digits=1)}x)',
        ],
    ]

    table0 = axes[0].table(
        cellText=phase_rows,
        colLabels=phase_cols,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table0.auto_set_font_size(False)
    table0.set_fontsize(8.5)
    table0.scale(1.1, 1.45)
    axes[0].set_title("Measured Phase Headline Metrics", fontsize=12, weight="bold", pad=10)

    # Bold the key cells (entire cell, since matplotlib doesn't support partial bold).
    # Decode C=16 request/tensor cells (tok/s and p50)
    for r in (1, 2):  # data rows are 1-indexed in the table (row 0 is header)
        for c in (3, 4):
            table0[(r, c)].get_text().set_weight("bold")
    # Prefill tensor cell
    table0[(5, 4)].get_text().set_weight("bold")

    sim_cols = ["Policy (sim)", "Goodput tok/s", "vs single", "On-time %", "p90 ms"]
    sim_rows = [
        [
            "single→single",
            _fmt(float(sim_single["goodput_tokens_per_s"]), digits=0),
            "1.00x",
            _fmt(float(sim_single["on_time_ratio"]) * 100.0, digits=1),
            _fmt(float(sim_single["p90_ms"]), digits=1),
        ],
        [
            "single→request",
            _fmt(float(sim_sr["goodput_tokens_per_s"]), digits=0),
            _fmt(float(sim_sr["goodput_ratio_vs_single"]), digits=2) + "x",
            _fmt(float(sim_sr["on_time_ratio"]) * 100.0, digits=1),
            _fmt(float(sim_sr["p90_ms"]), digits=1),
        ],
        [
            "request→request",
            _fmt(float(sim_rr["goodput_tokens_per_s"]), digits=0),
            _fmt(float(sim_rr["goodput_ratio_vs_single"]), digits=2) + "x",
            _fmt(float(sim_rr["on_time_ratio"]) * 100.0, digits=1),
            _fmt(float(sim_rr["p90_ms"]), digits=1),
        ],
        [
            "single→tensor",
            _fmt(float(sim_st["goodput_tokens_per_s"]), digits=0),
            _fmt(float(sim_st["goodput_ratio_vs_single"]), digits=2) + "x",
            _fmt(float(sim_st["on_time_ratio"]) * 100.0, digits=1),
            _fmt(float(sim_st["p90_ms"]), digits=1),
        ],
    ]

    table1 = axes[1].table(
        cellText=sim_rows,
        colLabels=sim_cols,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(8.5)
    table1.scale(1.05, 1.55)
    axes[1].set_title("Simulated Mixed-Traffic Policy Outcomes", fontsize=12, weight="bold", pad=10)

    # Bold key outcomes in the single->request row and the single->tensor ratio cell.
    # Row indices: header=0, then rows 1..4.
    for c in (2, 3):  # vs single, on-time %
        table1[(2, c)].get_text().set_weight("bold")
    table1[(4, 2)].get_text().set_weight("bold")

    out_png = PLOTS_DIR / "public_headline_table.png"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(FIG_DIR / "public_headline_table.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    metrics = _compute_headline_metrics()
    _save_metrics(metrics)
    _plot_project_overview()
    _plot_chiplet_comm_scaling()
    _plot_headline_metrics(metrics)
    _write_headline_table_tex(metrics)
    _plot_headline_table_png(metrics)
    print(FIG_DIR / "project_overview_pipeline.png")
    print(FIG_DIR / "chiplet_comm_scaling.png")
    print(FIG_DIR / "headline_metrics_summary.png")
    print(TABLE_DIR / "headline_table.tex")
    print(FIG_DIR / "public_headline_table.png")
    print(PLOTS_DIR / "public_headline_table.png")
    print(DATA_DIR / "headline_metrics.json")
    print(DATA_DIR / "chiplet_proxy_example.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
