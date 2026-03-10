#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POLICY_COLORS = {
    "single->single": "#1f77b4",
    "single->request": "#2ca02c",
    "request->request": "#1b7f3a",
}
POLICY_LABELS = {
    "single->single": "single→single",
    "single->request": "single→request",
    "request->request": "request→request",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dense analysis for sharded serving: decode improvements plus "
            "multi-request queue simulation from measured direct-trace samples."
        )
    )
    parser.add_argument("--summary-csv", type=Path, required=True, help="direct_policy_trace_summary.csv")
    parser.add_argument("--samples-json", type=Path, required=True, help="direct_policy_trace_samples.json")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["single->single", "single->request"],
        help="Policies to include from samples JSON.",
    )
    parser.add_argument("--request-slo-ms", type=float, default=500.0, help="Request latency SLO in ms.")
    parser.add_argument("--output-tokens", type=int, default=128, help="Output tokens per request for goodput.")
    parser.add_argument("--duration-s", type=float, default=180.0, help="Simulation duration per trial.")
    parser.add_argument("--trials", type=int, default=60, help="Number of Monte Carlo trials per arrival-rate point.")
    parser.add_argument(
        "--arrival-rates",
        type=str,
        default="6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0",
        help="Comma-separated arrival rates (req/s) for queue simulation.",
    )
    parser.add_argument(
        "--out-plot",
        type=Path,
        default=Path("results/plots/public_service_sharded_dense_analysis.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/plots/public_service_sharded_dense_queue.csv"),
        help="Output CSV for simulation aggregates.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("results/plots/public_service_sharded_dense_analysis.md"),
        help="Output markdown summary.",
    )
    return parser.parse_args()


def _parse_rates(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    values = [value for value in values if value > 0]
    if not values:
        raise ValueError("arrival-rates must contain at least one positive value.")
    return values


def _load_samples(
    samples_json: Path,
    policies: list[str],
    batch_by_policy: dict[str, int],
) -> dict[str, dict[str, np.ndarray]]:
    raw = json.loads(samples_json.read_text(encoding="utf-8"))
    out: dict[str, dict[str, np.ndarray]] = {}
    for policy in policies:
        payload = raw.get(policy)
        if payload is None:
            continue
        total = np.asarray(payload.get("total_ms", []), dtype=float)
        decode = np.asarray(payload.get("decode_ms", []), dtype=float)
        prefill = np.asarray(payload.get("prefill_ms", []), dtype=float)
        batch_size = int(batch_by_policy.get(policy, 1))
        if batch_size < 1:
            batch_size = 1
        if total.size == 0:
            continue
        out[policy] = {
            "total_ms": total,
            "total_ms_per_request": total / float(batch_size),
            "decode_ms": decode,
            "prefill_ms": prefill,
        }
    if not out:
        raise ValueError("No requested policies found in samples JSON.")
    return out


def _simulate_queue(
    *,
    service_ms_samples: np.ndarray,
    arrival_rate_rps: float,
    duration_s: float,
    request_slo_ms: float,
    output_tokens: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    arrivals: list[float] = []
    now = 0.0
    while now < duration_s:
        now += float(rng.exponential(1.0 / arrival_rate_rps))
        if now >= duration_s:
            break
        arrivals.append(now)

    server_free_ms = 0.0
    latencies_ms: list[float] = []
    on_time = 0
    for arrival_s in arrivals:
        arrival_ms = float(arrival_s * 1000.0)
        service_ms = float(rng.choice(service_ms_samples))
        start_ms = max(arrival_ms, server_free_ms)
        end_ms = start_ms + service_ms
        server_free_ms = end_ms
        latency_ms = end_ms - arrival_ms
        latencies_ms.append(latency_ms)
        if latency_ms <= request_slo_ms:
            on_time += 1

    completed = len(latencies_ms)
    on_time_ratio = float(on_time / completed) if completed > 0 else 0.0
    goodput_req_s = float(on_time / duration_s) if duration_s > 0 else 0.0
    goodput_tok_s = goodput_req_s * float(output_tokens)
    p90_ms = float(np.percentile(latencies_ms, 90)) if latencies_ms else 0.0
    return {
        "arrivals": float(len(arrivals)),
        "completed": float(completed),
        "on_time_ratio": on_time_ratio,
        "goodput_req_s": goodput_req_s,
        "goodput_tok_s": goodput_tok_s,
        "latency_p90_ms": p90_ms,
    }


def _aggregate_simulation(
    *,
    samples_by_policy: dict[str, dict[str, np.ndarray]],
    arrival_rates: list[float],
    duration_s: float,
    request_slo_ms: float,
    output_tokens: int,
    trials: int,
    seed: int = 123,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | str]] = []
    for policy, payload in samples_by_policy.items():
        service_samples = payload["total_ms_per_request"]
        for rate in arrival_rates:
            trial_metrics: list[dict[str, float]] = []
            for _ in range(trials):
                trial_rng = np.random.default_rng(int(rng.integers(0, 2_000_000_000)))
                trial_metrics.append(
                    _simulate_queue(
                        service_ms_samples=service_samples,
                        arrival_rate_rps=float(rate),
                        duration_s=duration_s,
                        request_slo_ms=request_slo_ms,
                        output_tokens=output_tokens,
                        rng=trial_rng,
                    )
                )

            keys = ["on_time_ratio", "goodput_req_s", "goodput_tok_s", "latency_p90_ms"]
            stats: dict[str, float] = {}
            for key in keys:
                values = np.asarray([item[key] for item in trial_metrics], dtype=float)
                stats[f"{key}_mean"] = float(values.mean())
                stats[f"{key}_std"] = float(values.std(ddof=0))

            rows.append(
                {
                    "policy": policy,
                    "arrival_rate_rps": float(rate),
                    **stats,
                }
            )
    df = pd.DataFrame(rows).sort_values(["policy", "arrival_rate_rps"]).reset_index(drop=True)
    return df


def _plot(
    *,
    sim_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    samples_by_policy: dict[str, dict[str, np.ndarray]],
    out_plot: Path,
) -> None:
    policies = [policy for policy in summary_df["policy"].tolist() if policy in sim_df["policy"].unique()]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=True)

    for policy in policies:
        sub = sim_df[sim_df["policy"] == policy].sort_values("arrival_rate_rps")
        x = sub["arrival_rate_rps"].to_numpy(dtype=float)
        on_time = sub["on_time_ratio_mean"].to_numpy(dtype=float) * 100.0
        on_time_std = sub["on_time_ratio_std"].to_numpy(dtype=float) * 100.0
        goodput = sub["goodput_tok_s_mean"].to_numpy(dtype=float)
        goodput_std = sub["goodput_tok_s_std"].to_numpy(dtype=float)
        p90 = sub["latency_p90_ms_mean"].to_numpy(dtype=float)
        p90_std = sub["latency_p90_ms_std"].to_numpy(dtype=float)
        color = POLICY_COLORS.get(policy, "#666666")
        label = POLICY_LABELS.get(policy, policy)

        axes[0].plot(x, on_time, marker="o", linewidth=2, color=color, label=label)
        axes[0].fill_between(x, on_time - on_time_std, on_time + on_time_std, color=color, alpha=0.15)

        axes[1].plot(x, goodput, marker="o", linewidth=2, color=color, label=label)
        axes[1].fill_between(x, goodput - goodput_std, goodput + goodput_std, color=color, alpha=0.15)

        axes[2].plot(x, p90, marker="o", linewidth=2, color=color, label=label)
        axes[2].fill_between(x, p90 - p90_std, p90 + p90_std, color=color, alpha=0.15)

    axes[0].set_title("On-time fraction under load")
    axes[0].set_ylabel("On-time requests (%)")
    axes[0].set_xlabel("Arrival rate (req/s)")
    axes[0].set_ylim(0.0, 105.0)
    axes[0].grid(alpha=0.25)

    axes[1].set_title("On-time goodput under load")
    axes[1].set_ylabel("Goodput (tokens/s)")
    axes[1].set_xlabel("Arrival rate (req/s)")
    axes[1].grid(alpha=0.25)

    axes[2].set_title("Tail latency under load")
    axes[2].set_ylabel("Request p90 latency (ms)")
    axes[2].set_xlabel("Arrival rate (req/s)")
    axes[2].grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, ncols=len(handles), loc="upper center", bbox_to_anchor=(0.5, 1.08))

    baseline = samples_by_policy.get("single->single", {}).get("decode_ms", np.asarray([], dtype=float))
    sharded = samples_by_policy.get("single->request", {}).get("decode_ms", np.asarray([], dtype=float))
    if baseline.size > 0 and sharded.size > 0:
        decode_speedup = float(np.median(baseline) / np.median(sharded))
    else:
        decode_speedup = float("nan")

    fig.suptitle(
        f"Dense sharded-serving analysis (decode median speedup={decode_speedup:.2f}x)",
        y=1.14,
        fontsize=13,
    )

    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_markdown(
    *,
    out_md: Path,
    summary_df: pd.DataFrame,
    samples_by_policy: dict[str, dict[str, np.ndarray]],
    sim_df: pd.DataFrame,
) -> None:
    baseline_total = np.asarray(samples_by_policy.get("single->single", {}).get("total_ms", []), dtype=float)
    sharded_total = np.asarray(samples_by_policy.get("single->request", {}).get("total_ms", []), dtype=float)
    baseline_decode = np.asarray(samples_by_policy.get("single->single", {}).get("decode_ms", []), dtype=float)
    sharded_decode = np.asarray(samples_by_policy.get("single->request", {}).get("decode_ms", []), dtype=float)

    lines = [
        "# Dense Sharded Serving Analysis",
        "",
        "## Direct measured decode and end-to-end comparison",
        "",
    ]

    if baseline_total.size > 0 and sharded_total.size > 0:
        total_speedup = float(np.median(baseline_total) / np.median(sharded_total))
        total_delta = float(np.median(baseline_total) - np.median(sharded_total))
        lines.append(
            f"- End-to-end median latency: `{np.median(baseline_total):.2f} ms` (single) vs "
            f"`{np.median(sharded_total):.2f} ms` (single->request), speedup `{total_speedup:.2f}x`, "
            f"absolute drop `{total_delta:.2f} ms`."
        )
    if baseline_decode.size > 0 and sharded_decode.size > 0:
        decode_speedup = float(np.median(baseline_decode) / np.median(sharded_decode))
        decode_delta = float(np.median(baseline_decode) - np.median(sharded_decode))
        lines.append(
            f"- Decode median latency component: `{np.median(baseline_decode):.2f} ms` (single) vs "
            f"`{np.median(sharded_decode):.2f} ms` (single->request), speedup `{decode_speedup:.2f}x`, "
            f"absolute drop `{decode_delta:.2f} ms`."
        )
    lines.extend(
        [
            "",
            "## Multi-user queue simulation from measured service-time samples",
            "",
            "| Policy | Max arrival rate with >=90% on-time | Goodput at that point (tok/s) |",
            "| --- | ---: | ---: |",
        ]
    )

    for policy in summary_df["policy"].tolist():
        sub = sim_df[sim_df["policy"] == policy].sort_values("arrival_rate_rps")
        feasible = sub[sub["on_time_ratio_mean"] >= 0.90]
        if feasible.empty:
            lines.append(f"| {policy} | n/a | n/a |")
            continue
        best = feasible.iloc[-1]
        lines.append(
            f"| {policy} | {float(best['arrival_rate_rps']):.1f} req/s | "
            f"{float(best['goodput_tok_s_mean']):.1f} |"
        )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    arrival_rates = _parse_rates(args.arrival_rates)
    summary_df = pd.read_csv(args.summary_csv)
    batch_by_policy = {
        str(row["policy"]): int(row["batch"])
        for _, row in summary_df.iterrows()
        if int(row.get("batch", 1)) > 0
    }
    samples_by_policy = _load_samples(args.samples_json, args.policies, batch_by_policy)

    sim_df = _aggregate_simulation(
        samples_by_policy=samples_by_policy,
        arrival_rates=arrival_rates,
        duration_s=args.duration_s,
        request_slo_ms=args.request_slo_ms,
        output_tokens=args.output_tokens,
        trials=args.trials,
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    sim_df.to_csv(args.out_csv, index=False)
    _plot(
        sim_df=sim_df,
        summary_df=summary_df,
        samples_by_policy=samples_by_policy,
        out_plot=args.out_plot,
    )
    _write_markdown(
        out_md=args.out_md,
        summary_df=summary_df,
        samples_by_policy=samples_by_policy,
        sim_df=sim_df,
    )

    print(args.out_plot)
    print(args.out_csv)
    print(args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
