#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize MoE service-study metrics into compact CSV/Markdown tables for paper text."
        )
    )
    parser.add_argument("--metrics-csv", type=Path, required=True, help="Path to metrics.csv")
    parser.add_argument(
        "--decode-slo-csv",
        type=Path,
        default=None,
        help="Path to decode_slo_summary.csv (defaults to sibling of metrics.csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/plots"),
        help="Output directory for summary tables.",
    )
    parser.add_argument("--prefix", default="moe_service", help="Output file prefix.")
    return parser.parse_args()


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return float("nan")
    return float(a) / float(b)


def _median_or_nan(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.median())


def main() -> int:
    args = parse_args()
    if not args.metrics_csv.exists():
        raise SystemExit(f"Missing metrics CSV: {args.metrics_csv}")

    decode_slo_csv = args.decode_slo_csv or (args.metrics_csv.parent / "decode_slo_summary.csv")

    metrics = pd.read_csv(args.metrics_csv)
    decode_slo = pd.read_csv(decode_slo_csv) if decode_slo_csv.exists() else pd.DataFrame()

    key_cols = ["context_len", "batch", "decode_steps", "num_experts", "top_k", "routing_skew"]
    single = metrics[metrics["setup"] == "single_die"][key_cols + ["throughput_tokens_per_s"]]
    naive = metrics[metrics["setup"] == "dual_die_moe_naive"][
        key_cols + ["throughput_tokens_per_s", "communication_ms_p50", "remote_dispatch_ratio_p50"]
    ]
    locality = metrics[metrics["setup"] == "dual_die_moe_locality"][
        key_cols + ["throughput_tokens_per_s", "communication_ms_p50", "remote_dispatch_ratio_p50"]
    ]

    loc_vs_naive = locality.merge(naive, on=key_cols, how="inner", suffixes=("_loc", "_naive"))
    naive_vs_single = naive.merge(single, on=key_cols, how="inner", suffixes=("_naive", "_single"))
    loc_vs_single = locality.merge(single, on=key_cols, how="inner", suffixes=("_loc", "_single"))

    rows: list[dict[str, float | str]] = []
    skews = sorted(float(x) for x in metrics["routing_skew"].unique())

    for skew in skews:
        row: dict[str, float | str] = {"routing_skew": float(skew)}
        lv = loc_vs_naive[loc_vs_naive["routing_skew"] == skew]
        ns = naive_vs_single[naive_vs_single["routing_skew"] == skew]
        ls = loc_vs_single[loc_vs_single["routing_skew"] == skew]

        if not lv.empty:
            tp_ratio = lv["throughput_tokens_per_s_loc"] / lv["throughput_tokens_per_s_naive"].replace(
                0, np.nan
            )
            comm_ratio = lv["communication_ms_p50_loc"] / lv["communication_ms_p50_naive"].replace(0, np.nan)
            remote_delta = lv["remote_dispatch_ratio_p50_loc"] - lv["remote_dispatch_ratio_p50_naive"]
            row["locality_vs_naive_throughput_ratio_median"] = _median_or_nan(tp_ratio)
            row["locality_vs_naive_comm_ratio_median"] = _median_or_nan(comm_ratio)
            row["locality_vs_naive_remote_delta_median"] = _median_or_nan(remote_delta)
        else:
            row["locality_vs_naive_throughput_ratio_median"] = float("nan")
            row["locality_vs_naive_comm_ratio_median"] = float("nan")
            row["locality_vs_naive_remote_delta_median"] = float("nan")

        if not ns.empty:
            naive_ratio = ns["throughput_tokens_per_s_naive"] / ns["throughput_tokens_per_s_single"].replace(
                0, np.nan
            )
            row["naive_vs_single_throughput_ratio_median"] = _median_or_nan(naive_ratio)
        else:
            row["naive_vs_single_throughput_ratio_median"] = float("nan")

        if not ls.empty:
            loc_ratio = ls["throughput_tokens_per_s_loc"] / ls["throughput_tokens_per_s_single"].replace(0, np.nan)
            row["locality_vs_single_throughput_ratio_median"] = _median_or_nan(loc_ratio)
        else:
            row["locality_vs_single_throughput_ratio_median"] = float("nan")

        if not decode_slo.empty:
            feasible_counts = (
                decode_slo[decode_slo["routing_skew"] == skew]
                .groupby("setup")["feasible"]
                .sum()
                .to_dict()
            )
            row["feasible_points_single"] = float(feasible_counts.get("single_die", 0.0))
            row["feasible_points_naive"] = float(feasible_counts.get("dual_die_moe_naive", 0.0))
            row["feasible_points_locality"] = float(feasible_counts.get("dual_die_moe_locality", 0.0))

        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values("routing_skew")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"{args.prefix}_summary.csv"
    md_path = args.out_dir / f"{args.prefix}_summary.md"

    out_df.to_csv(csv_path, index=False)

    md_lines = [
        "# MoE Service Summary",
        "",
        "| Routing skew | locality/naive throughput (median) | locality/naive comm ratio (median) | locality-naive remote dispatch delta (median) | naive/single throughput (median) | locality/single throughput (median) | feasible(single) | feasible(naive) | feasible(locality) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in out_df.to_dict(orient="records"):
        md_lines.append(
            f"| {float(row['routing_skew']):.2f} | "
            f"{float(row.get('locality_vs_naive_throughput_ratio_median', float('nan'))):.4f} | "
            f"{float(row.get('locality_vs_naive_comm_ratio_median', float('nan'))):.4f} | "
            f"{float(row.get('locality_vs_naive_remote_delta_median', float('nan'))):.4f} | "
            f"{float(row.get('naive_vs_single_throughput_ratio_median', float('nan'))):.4f} | "
            f"{float(row.get('locality_vs_single_throughput_ratio_median', float('nan'))):.4f} | "
            f"{int(row.get('feasible_points_single', 0.0))} | "
            f"{int(row.get('feasible_points_naive', 0.0))} | "
            f"{int(row.get('feasible_points_locality', 0.0))} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(csv_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
