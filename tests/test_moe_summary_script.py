from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys


def _write_metrics(path: Path) -> None:
    fields = [
        "phase",
        "setup",
        "batch",
        "context_len",
        "decode_steps",
        "num_experts",
        "top_k",
        "routing_skew",
        "throughput_tokens_per_s",
        "communication_ms_p50",
        "remote_dispatch_ratio_p50",
    ]
    rows = [
        {
            "phase": "decode",
            "setup": "single_die",
            "batch": 16,
            "context_len": 2048,
            "decode_steps": 8,
            "num_experts": 8,
            "top_k": 2,
            "routing_skew": 0.0,
            "throughput_tokens_per_s": 1000.0,
            "communication_ms_p50": 0.0,
            "remote_dispatch_ratio_p50": 0.0,
        },
        {
            "phase": "decode",
            "setup": "dual_die_moe_naive",
            "batch": 16,
            "context_len": 2048,
            "decode_steps": 8,
            "num_experts": 8,
            "top_k": 2,
            "routing_skew": 0.0,
            "throughput_tokens_per_s": 1200.0,
            "communication_ms_p50": 20.0,
            "remote_dispatch_ratio_p50": 0.5,
        },
        {
            "phase": "decode",
            "setup": "dual_die_moe_locality",
            "batch": 16,
            "context_len": 2048,
            "decode_steps": 8,
            "num_experts": 8,
            "top_k": 2,
            "routing_skew": 0.0,
            "throughput_tokens_per_s": 1210.0,
            "communication_ms_p50": 18.0,
            "remote_dispatch_ratio_p50": 0.3,
        },
        {
            "phase": "decode",
            "setup": "single_die",
            "batch": 16,
            "context_len": 2048,
            "decode_steps": 8,
            "num_experts": 8,
            "top_k": 2,
            "routing_skew": 2.0,
            "throughput_tokens_per_s": 900.0,
            "communication_ms_p50": 0.0,
            "remote_dispatch_ratio_p50": 0.0,
        },
        {
            "phase": "decode",
            "setup": "dual_die_moe_naive",
            "batch": 16,
            "context_len": 2048,
            "decode_steps": 8,
            "num_experts": 8,
            "top_k": 2,
            "routing_skew": 2.0,
            "throughput_tokens_per_s": 1100.0,
            "communication_ms_p50": 24.0,
            "remote_dispatch_ratio_p50": 0.55,
        },
        {
            "phase": "decode",
            "setup": "dual_die_moe_locality",
            "batch": 16,
            "context_len": 2048,
            "decode_steps": 8,
            "num_experts": 8,
            "top_k": 2,
            "routing_skew": 2.0,
            "throughput_tokens_per_s": 1105.0,
            "communication_ms_p50": 20.0,
            "remote_dispatch_ratio_p50": 0.2,
        },
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_decode_slo(path: Path) -> None:
    fields = [
        "setup",
        "context_len",
        "num_experts",
        "top_k",
        "routing_skew",
        "slo_ms",
        "best_throughput_tokens_per_s",
        "best_concurrency",
        "best_latency_ms_p90",
        "feasible",
    ]
    rows = [
        {
            "setup": "single_die",
            "context_len": 2048,
            "num_experts": 8,
            "top_k": 2,
            "routing_skew": 0.0,
            "slo_ms": 120.0,
            "best_throughput_tokens_per_s": 1000.0,
            "best_concurrency": 16,
            "best_latency_ms_p90": 110.0,
            "feasible": True,
        },
        {
            "setup": "dual_die_moe_naive",
            "context_len": 2048,
            "num_experts": 8,
            "top_k": 2,
            "routing_skew": 0.0,
            "slo_ms": 120.0,
            "best_throughput_tokens_per_s": 1200.0,
            "best_concurrency": 16,
            "best_latency_ms_p90": 100.0,
            "feasible": True,
        },
        {
            "setup": "dual_die_moe_locality",
            "context_len": 2048,
            "num_experts": 8,
            "top_k": 2,
            "routing_skew": 0.0,
            "slo_ms": 120.0,
            "best_throughput_tokens_per_s": 1210.0,
            "best_concurrency": 16,
            "best_latency_ms_p90": 98.0,
            "feasible": True,
        },
        {
            "setup": "single_die",
            "context_len": 2048,
            "num_experts": 8,
            "top_k": 2,
            "routing_skew": 2.0,
            "slo_ms": 120.0,
            "best_throughput_tokens_per_s": 900.0,
            "best_concurrency": 16,
            "best_latency_ms_p90": 119.0,
            "feasible": True,
        },
        {
            "setup": "dual_die_moe_naive",
            "context_len": 2048,
            "num_experts": 8,
            "top_k": 2,
            "routing_skew": 2.0,
            "slo_ms": 120.0,
            "best_throughput_tokens_per_s": 1100.0,
            "best_concurrency": 16,
            "best_latency_ms_p90": 117.0,
            "feasible": True,
        },
        {
            "setup": "dual_die_moe_locality",
            "context_len": 2048,
            "num_experts": 8,
            "top_k": 2,
            "routing_skew": 2.0,
            "slo_ms": 120.0,
            "best_throughput_tokens_per_s": 1105.0,
            "best_concurrency": 16,
            "best_latency_ms_p90": 116.0,
            "feasible": True,
        },
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_summarize_moe_service_script_outputs_tables(tmp_path: Path) -> None:
    metrics_csv = tmp_path / "metrics.csv"
    slo_csv = tmp_path / "decode_slo_summary.csv"
    out_dir = tmp_path / "out"

    _write_metrics(metrics_csv)
    _write_decode_slo(slo_csv)

    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_moe_service.py",
            "--metrics-csv",
            str(metrics_csv),
            "--decode-slo-csv",
            str(slo_csv),
            "--out-dir",
            str(out_dir),
            "--prefix",
            "unit_moe",
        ],
        check=True,
    )

    summary_csv = out_dir / "unit_moe_summary.csv"
    summary_md = out_dir / "unit_moe_summary.md"
    assert summary_csv.exists()
    assert summary_md.exists()

    with summary_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert "locality_vs_naive_throughput_ratio_median" in rows[0]
    assert "feasible_points_locality" in rows[0]
