#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SETUP_ORDER = ["single_die", "dual_die_tensor_optimized", "dual_die_request_sharded"]
POLICY_ORDER = [
    ("single->single", "single_die", "single_die"),
    ("single->request", "single_die", "dual_die_request_sharded"),
    ("single->tensor", "single_die", "dual_die_tensor_optimized"),
    ("request->request", "dual_die_request_sharded", "dual_die_request_sharded"),
]
COLORS = {
    "single->single": "#1f77b4",
    "single->tensor": "#ff7f0e",
    "single->request": "#2ca02c",
    "request->request": "#1b7f3a",
}
POLICY_LABELS = {
    "single->single": "single→single",
    "single->tensor": "single→tensor",
    "single->request": "single→request",
    "request->request": "request→request",
}


@dataclass(frozen=True)
class Request:
    arrival_s: float
    phase: str
    context_len: int
    tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate mixed prefill/decode traffic using measured phase-study throughput profiles."
        )
    )
    parser.add_argument("--metrics-csv", type=Path, required=True, help="Path to phase-study metrics.csv.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to metrics parent).",
    )
    parser.add_argument("--prefix", default="service_trace", help="Output file prefix.")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed for deterministic request stream.")
    parser.add_argument("--duration-s", type=float, default=120.0, help="Simulation duration in seconds.")
    parser.add_argument("--arrival-rate-rps", type=float, default=12.0, help="Request arrival rate (Poisson).")
    parser.add_argument("--prefill-ratio", type=float, default=0.3, help="Prefill request fraction in [0, 1].")
    parser.add_argument("--decode-slo-ms", type=float, default=250.0, help="Latency SLO for on-time goodput.")
    parser.add_argument(
        "--drop-wait-ms",
        type=float,
        default=2000.0,
        help="Drop request if queue wait exceeds this threshold (ms).",
    )
    parser.add_argument("--decode-tokens", type=int, default=64, help="Tokens attributed to one decode request.")
    parser.add_argument(
        "--context-weights",
        default="2048:0.5,4096:0.35,8192:0.15",
        help="Comma-separated context weights, e.g. 2048:0.5,4096:0.35,8192:0.15",
    )
    return parser.parse_args()


def _parse_context_weights(raw: str) -> tuple[list[int], np.ndarray]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("context-weights must contain at least one entry")
    contexts: list[int] = []
    weights: list[float] = []
    for item in items:
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid context weight entry: {item}")
        context = int(parts[0])
        weight = float(parts[1])
        if context < 1 or weight <= 0:
            raise ValueError(f"Invalid context/weight pair: {item}")
        contexts.append(context)
        weights.append(weight)
    probs = np.array(weights, dtype=np.float64)
    probs = probs / probs.sum()
    return contexts, probs


def _build_requests(
    *,
    seed: int,
    duration_s: float,
    arrival_rate_rps: float,
    prefill_ratio: float,
    decode_tokens: int,
    contexts: list[int],
    context_probs: np.ndarray,
) -> list[Request]:
    rng = np.random.default_rng(seed)
    requests: list[Request] = []
    now = 0.0
    while now < duration_s:
        now += float(rng.exponential(1.0 / arrival_rate_rps))
        if now >= duration_s:
            break
        is_prefill = bool(rng.random() < prefill_ratio)
        context = int(rng.choice(contexts, p=context_probs))
        tokens = int(context if is_prefill else decode_tokens)
        requests.append(
            Request(
                arrival_s=float(now),
                phase="prefill" if is_prefill else "decode",
                context_len=context,
                tokens=tokens,
            )
        )
    return requests


def _build_setup_profiles(metrics: pd.DataFrame, decode_slo_ms: float) -> dict[str, dict[str, dict[int, float]]]:
    profiles: dict[str, dict[str, dict[int, float]]] = {}
    for setup in SETUP_ORDER:
        prefill_setup = metrics[(metrics["phase"] == "prefill") & (metrics["setup"] == setup)]
        decode_setup = metrics[(metrics["phase"] == "decode") & (metrics["setup"] == setup)]

        prefill_map: dict[int, float] = {}
        for context in sorted(prefill_setup["seq_len"].astype(int).unique().tolist()):
            sub = prefill_setup[prefill_setup["seq_len"].astype(int) == context]
            if sub.empty:
                continue
            best = float(sub["throughput_tokens_per_s"].astype(float).max())
            if best > 0:
                prefill_map[int(context)] = best

        decode_map: dict[int, float] = {}
        for context in sorted(decode_setup["context_len"].astype(int).unique().tolist()):
            sub = decode_setup[decode_setup["context_len"].astype(int) == context]
            if sub.empty:
                continue
            feasible = sub[sub["latency_ms_p90"].astype(float) <= float(decode_slo_ms)]
            pick = feasible if not feasible.empty else sub
            best = float(pick["throughput_tokens_per_s"].astype(float).max()) if not pick.empty else 0.0
            if best > 0:
                decode_map[int(context)] = best

        profiles[setup] = {"prefill": prefill_map, "decode": decode_map}
    return profiles


def _build_policy_profiles(
    setup_profiles: dict[str, dict[str, dict[int, float]]]
) -> dict[str, dict[str, dict[int, float]]]:
    policies: dict[str, dict[str, dict[int, float]]] = {}
    for policy_name, prefill_setup, decode_setup in POLICY_ORDER:
        policies[policy_name] = {
            "prefill": dict(setup_profiles.get(prefill_setup, {}).get("prefill", {})),
            "decode": dict(setup_profiles.get(decode_setup, {}).get("decode", {})),
        }
    return policies


def _nearest_profile(profile: dict[int, float], context_len: int) -> float:
    if not profile:
        return 0.0
    if context_len in profile:
        return float(profile[context_len])
    keys = sorted(profile)
    nearest = min(keys, key=lambda key: abs(int(key) - int(context_len)))
    return float(profile[nearest])


def _simulate_setup(
    *,
    requests: list[Request],
    profile: dict[str, dict[int, float]],
    duration_s: float,
    slo_ms: float,
    drop_wait_ms: float,
) -> dict[str, float]:
    server_free_s = 0.0
    latencies_ms: list[float] = []
    total_requests = len(requests)
    completed = 0
    on_time = 0
    late = 0
    dropped = 0
    dropped_no_profile = 0
    served_tokens = 0.0
    goodput_tokens = 0.0

    for req in requests:
        phase_profile = profile.get(req.phase, {})
        throughput = _nearest_profile(phase_profile, req.context_len)
        if throughput <= 0:
            dropped += 1
            dropped_no_profile += 1
            continue

        wait_s = max(0.0, server_free_s - req.arrival_s)
        if wait_s * 1000.0 > drop_wait_ms:
            dropped += 1
            continue

        service_s = float(req.tokens) / float(throughput)
        start_s = max(req.arrival_s, server_free_s)
        end_s = start_s + service_s
        server_free_s = end_s

        latency_ms = (end_s - req.arrival_s) * 1000.0
        latencies_ms.append(latency_ms)
        completed += 1
        served_tokens += float(req.tokens)
        if latency_ms <= slo_ms:
            on_time += 1
            goodput_tokens += float(req.tokens)
        else:
            late += 1

    p50_latency = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
    p90_latency = float(np.percentile(latencies_ms, 90)) if latencies_ms else 0.0

    return {
        "total_requests": float(total_requests),
        "completed_requests": float(completed),
        "on_time_requests": float(on_time),
        "late_requests": float(late),
        "dropped_requests": float(dropped),
        "dropped_no_profile": float(dropped_no_profile),
        "on_time_ratio": (float(on_time) / float(completed)) if completed > 0 else 0.0,
        "drop_ratio": (float(dropped) / float(total_requests)) if total_requests > 0 else 0.0,
        "latency_ms_p50": p50_latency,
        "latency_ms_p90": p90_latency,
        "served_tokens_per_s": served_tokens / duration_s if duration_s > 0 else 0.0,
        "goodput_tokens_per_s": goodput_tokens / duration_s if duration_s > 0 else 0.0,
    }


def _plot_goodput(rows: list[dict[str, float]], out_path: Path) -> None:
    policies = [str(row["policy"]) for row in rows]
    goodput = [float(row["goodput_tokens_per_s"]) for row in rows]
    on_time_ratio = [float(row["on_time_ratio"]) * 100.0 for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    x = np.arange(len(policies))
    bars = ax.bar(x, goodput, color=[COLORS.get(policy, "#808080") for policy in policies], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS.get(policy, policy) for policy in policies])
    ax.set_ylabel("Goodput (tokens/s under SLO)")
    ax.set_title("Mixed-Traffic Service Goodput by Deployment Policy")
    ax.grid(axis="y", alpha=0.25)

    y_pad = max(40.0, max(goodput, default=0.0) * 0.012)
    for bar, tokens_per_s, ratio in zip(bars, goodput, on_time_ratio):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_pad,
            f"{tokens_per_s:.0f} tok/s\n{ratio:.1f}% on-time",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0.0, max(goodput, default=0.0) + y_pad * 5.0)

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_policy_frontier(rows: list[dict[str, float]], out_path: Path) -> None:
    policies = [str(row["policy"]) for row in rows]
    goodput = np.array([float(row["goodput_tokens_per_s"]) for row in rows], dtype=float)
    on_time_pct = np.array([float(row["on_time_ratio"]) * 100.0 for row in rows], dtype=float)
    label_offsets = {
        "single->single": (8, 8),
        "single->request": (14, 18),
        "request->request": (14, -10),
        "single->tensor": (10, 8),
    }

    fig, ax = plt.subplots(figsize=(8.6, 5), constrained_layout=True)
    for policy, x, y in zip(policies, on_time_pct, goodput):
        ax.scatter(
            float(x),
            float(y),
            s=160.0,
            color=COLORS.get(policy, "#808080"),
            edgecolors="black",
            linewidths=0.35,
            alpha=0.9,
        )
        ax.annotate(
            POLICY_LABELS.get(policy, policy),
            (float(x), float(y)),
            fontsize=9,
            xytext=label_offsets.get(policy, (8, 6)),
            textcoords="offset points",
        )

    ax.set_xlabel("On-time service (%)")
    ax.set_ylabel("Goodput (tokens/s under SLO)")
    ax.set_title("Policy Frontier: Goodput vs On-Time Service")
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, max(1.0, float(goodput.max(initial=0.0)) * 1.12))
    ax.grid(alpha=0.25)

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not args.metrics_csv.exists():
        raise SystemExit(f"Missing metrics CSV: {args.metrics_csv}")
    if not (0.0 <= args.prefill_ratio <= 1.0):
        raise SystemExit("--prefill-ratio must be in [0, 1].")
    if args.duration_s <= 0 or args.arrival_rate_rps <= 0 or args.decode_tokens <= 0:
        raise SystemExit("duration-s, arrival-rate-rps, and decode-tokens must be > 0.")

    contexts, context_probs = _parse_context_weights(args.context_weights)
    metrics = pd.read_csv(args.metrics_csv)
    setup_profiles = _build_setup_profiles(metrics, decode_slo_ms=float(args.decode_slo_ms))
    profiles = _build_policy_profiles(setup_profiles)
    requests = _build_requests(
        seed=int(args.seed),
        duration_s=float(args.duration_s),
        arrival_rate_rps=float(args.arrival_rate_rps),
        prefill_ratio=float(args.prefill_ratio),
        decode_tokens=int(args.decode_tokens),
        contexts=contexts,
        context_probs=context_probs,
    )
    if not requests:
        raise SystemExit("No requests generated; increase duration or arrival rate.")

    out_dir = args.out_dir or args.metrics_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, float]] = []
    for policy_name, _, _ in POLICY_ORDER:
        profile = profiles.get(policy_name, {"prefill": {}, "decode": {}})
        result = _simulate_setup(
            requests=requests,
            profile=profile,
            duration_s=float(args.duration_s),
            slo_ms=float(args.decode_slo_ms),
            drop_wait_ms=float(args.drop_wait_ms),
        )
        row = {"policy": policy_name}
        row.update(result)
        summary_rows.append(row)

    csv_path = out_dir / f"{args.prefix}_service_trace_summary.csv"
    md_path = out_dir / f"{args.prefix}_service_trace_summary.md"
    plot_path = out_dir / f"{args.prefix}_mixed_trace_goodput.png"
    frontier_path = out_dir / f"{args.prefix}_mixed_trace_frontier.png"
    req_stream_path = out_dir / f"{args.prefix}_service_trace_requests.json"

    fields = [
        "policy",
        "total_requests",
        "completed_requests",
        "on_time_requests",
        "late_requests",
        "dropped_requests",
        "dropped_no_profile",
        "on_time_ratio",
        "drop_ratio",
        "latency_ms_p50",
        "latency_ms_p90",
        "served_tokens_per_s",
        "goodput_tokens_per_s",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    lines = [
        "# Mixed Traffic Summary",
        "",
        f"- duration_s: {float(args.duration_s):.2f}",
        f"- arrival_rate_rps: {float(args.arrival_rate_rps):.3f}",
        f"- prefill_ratio: {float(args.prefill_ratio):.3f}",
        f"- decode_slo_ms: {float(args.decode_slo_ms):.2f}",
        f"- drop_wait_ms: {float(args.drop_wait_ms):.2f}",
        "",
        "| Policy | Goodput (tokens/s) | Served (tokens/s) | On-time % | Drop % | p50 latency (ms) | p90 latency (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['policy']} | {float(row['goodput_tokens_per_s']):.2f} | "
            f"{float(row['served_tokens_per_s']):.2f} | {float(row['on_time_ratio']) * 100.0:.2f} | "
            f"{float(row['drop_ratio']) * 100.0:.2f} | {float(row['latency_ms_p50']):.2f} | "
            f"{float(row['latency_ms_p90']):.2f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    request_payload = {
        "seed": int(args.seed),
        "duration_s": float(args.duration_s),
        "arrival_rate_rps": float(args.arrival_rate_rps),
        "prefill_ratio": float(args.prefill_ratio),
        "decode_slo_ms": float(args.decode_slo_ms),
        "drop_wait_ms": float(args.drop_wait_ms),
        "decode_tokens": int(args.decode_tokens),
        "context_weights": args.context_weights,
        "request_count": len(requests),
    }
    req_stream_path.write_text(json.dumps(request_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _plot_goodput(summary_rows, plot_path)
    _plot_policy_frontier(summary_rows, frontier_path)

    print(csv_path)
    print(md_path)
    print(plot_path)
    print(frontier_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
