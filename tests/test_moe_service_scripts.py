from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys


def _write_moe_metrics_csv(path: Path) -> None:
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
        "compute_ms_p50",
        "communication_ms_p50",
        "remote_dispatch_ratio_p50",
    ]
    rows = [
        {
            "phase": "decode",
            "setup": "single_die",
            "batch": 8,
            "context_len": 2048,
            "decode_steps": 8,
            "num_experts": 8,
            "top_k": 2,
            "routing_skew": 0.0,
            "throughput_tokens_per_s": 900.0,
            "compute_ms_p50": 8.0,
            "communication_ms_p50": 0.0,
            "remote_dispatch_ratio_p50": 0.0,
        },
        {
            "phase": "decode",
            "setup": "single_die",
            "batch": 16,
            "context_len": 2048,
            "decode_steps": 8,
            "num_experts": 8,
            "top_k": 2,
            "routing_skew": 1.8,
            "throughput_tokens_per_s": 1200.0,
            "compute_ms_p50": 12.0,
            "communication_ms_p50": 0.0,
            "remote_dispatch_ratio_p50": 0.0,
        },
        {
            "phase": "decode",
            "setup": "dual_die_moe_naive",
            "batch": 8,
            "context_len": 2048,
            "decode_steps": 8,
            "num_experts": 8,
            "top_k": 2,
            "routing_skew": 0.0,
            "throughput_tokens_per_s": 820.0,
            "compute_ms_p50": 6.0,
            "communication_ms_p50": 4.0,
            "remote_dispatch_ratio_p50": 0.52,
        },
        {
            "phase": "decode",
            "setup": "dual_die_moe_locality",
            "batch": 8,
            "context_len": 2048,
            "decode_steps": 8,
            "num_experts": 8,
            "top_k": 2,
            "routing_skew": 0.0,
            "throughput_tokens_per_s": 840.0,
            "compute_ms_p50": 6.2,
            "communication_ms_p50": 3.9,
            "remote_dispatch_ratio_p50": 0.49,
        },
        {
            "phase": "decode",
            "setup": "dual_die_moe_naive",
            "batch": 16,
            "context_len": 2048,
            "decode_steps": 8,
            "num_experts": 8,
            "top_k": 2,
            "routing_skew": 1.8,
            "throughput_tokens_per_s": 920.0,
            "compute_ms_p50": 7.5,
            "communication_ms_p50": 8.5,
            "remote_dispatch_ratio_p50": 0.56,
        },
        {
            "phase": "decode",
            "setup": "dual_die_moe_locality",
            "batch": 16,
            "context_len": 2048,
            "decode_steps": 8,
            "num_experts": 8,
            "top_k": 2,
            "routing_skew": 1.8,
            "throughput_tokens_per_s": 1180.0,
            "compute_ms_p50": 8.0,
            "communication_ms_p50": 5.0,
            "remote_dispatch_ratio_p50": 0.26,
        },
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_decode_slo_csv(path: Path) -> None:
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
            "slo_ms": 250.0,
            "best_throughput_tokens_per_s": 900.0,
            "best_concurrency": 8,
            "best_latency_ms_p90": 140.0,
            "feasible": True,
        },
        {
            "setup": "dual_die_moe_naive",
            "context_len": 2048,
            "num_experts": 8,
            "top_k": 2,
            "routing_skew": 0.0,
            "slo_ms": 250.0,
            "best_throughput_tokens_per_s": 820.0,
            "best_concurrency": 8,
            "best_latency_ms_p90": 170.0,
            "feasible": True,
        },
        {
            "setup": "dual_die_moe_locality",
            "context_len": 2048,
            "num_experts": 8,
            "top_k": 2,
            "routing_skew": 0.0,
            "slo_ms": 250.0,
            "best_throughput_tokens_per_s": 840.0,
            "best_concurrency": 8,
            "best_latency_ms_p90": 165.0,
            "feasible": True,
        },
        {
            "setup": "single_die",
            "context_len": 2048,
            "num_experts": 8,
            "top_k": 2,
            "routing_skew": 1.8,
            "slo_ms": 250.0,
            "best_throughput_tokens_per_s": 1200.0,
            "best_concurrency": 16,
            "best_latency_ms_p90": 180.0,
            "feasible": True,
        },
        {
            "setup": "dual_die_moe_naive",
            "context_len": 2048,
            "num_experts": 8,
            "top_k": 2,
            "routing_skew": 1.8,
            "slo_ms": 250.0,
            "best_throughput_tokens_per_s": 920.0,
            "best_concurrency": 16,
            "best_latency_ms_p90": 220.0,
            "feasible": True,
        },
        {
            "setup": "dual_die_moe_locality",
            "context_len": 2048,
            "num_experts": 8,
            "top_k": 2,
            "routing_skew": 1.8,
            "slo_ms": 250.0,
            "best_throughput_tokens_per_s": 1180.0,
            "best_concurrency": 16,
            "best_latency_ms_p90": 190.0,
            "feasible": True,
        },
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_capacity_csv(path: Path) -> None:
    fields = [
        "setup",
        "context_len",
        "num_experts",
        "top_k",
        "routing_skew",
        "slo_ms",
        "max_tested_concurrency",
        "max_feasible_concurrency",
        "best_throughput_tokens_per_s",
        "best_concurrency",
        "best_latency_ms_p90",
        "has_feasible",
    ]
    rows = [
        {
            "setup": "single_die",
            "context_len": 2048,
            "num_experts": 8,
            "top_k": 2,
            "routing_skew": 0.0,
            "slo_ms": 250.0,
            "max_tested_concurrency": 8,
            "max_feasible_concurrency": 8,
            "best_throughput_tokens_per_s": 900.0,
            "best_concurrency": 8,
            "best_latency_ms_p90": 140.0,
            "has_feasible": True,
        },
        {
            "setup": "dual_die_moe_naive",
            "context_len": 2048,
            "num_experts": 8,
            "top_k": 2,
            "routing_skew": 0.0,
            "slo_ms": 250.0,
            "max_tested_concurrency": 8,
            "max_feasible_concurrency": 8,
            "best_throughput_tokens_per_s": 820.0,
            "best_concurrency": 8,
            "best_latency_ms_p90": 170.0,
            "has_feasible": True,
        },
        {
            "setup": "dual_die_moe_locality",
            "context_len": 2048,
            "num_experts": 8,
            "top_k": 2,
            "routing_skew": 0.0,
            "slo_ms": 250.0,
            "max_tested_concurrency": 8,
            "max_feasible_concurrency": 8,
            "best_throughput_tokens_per_s": 840.0,
            "best_concurrency": 8,
            "best_latency_ms_p90": 165.0,
            "has_feasible": True,
        },
        {
            "setup": "single_die",
            "context_len": 2048,
            "num_experts": 8,
            "top_k": 2,
            "routing_skew": 1.8,
            "slo_ms": 250.0,
            "max_tested_concurrency": 16,
            "max_feasible_concurrency": 16,
            "best_throughput_tokens_per_s": 1200.0,
            "best_concurrency": 16,
            "best_latency_ms_p90": 180.0,
            "has_feasible": True,
        },
        {
            "setup": "dual_die_moe_naive",
            "context_len": 2048,
            "num_experts": 8,
            "top_k": 2,
            "routing_skew": 1.8,
            "slo_ms": 250.0,
            "max_tested_concurrency": 16,
            "max_feasible_concurrency": 16,
            "best_throughput_tokens_per_s": 920.0,
            "best_concurrency": 16,
            "best_latency_ms_p90": 220.0,
            "has_feasible": True,
        },
        {
            "setup": "dual_die_moe_locality",
            "context_len": 2048,
            "num_experts": 8,
            "top_k": 2,
            "routing_skew": 1.8,
            "slo_ms": 250.0,
            "max_tested_concurrency": 16,
            "max_feasible_concurrency": 16,
            "best_throughput_tokens_per_s": 1180.0,
            "best_concurrency": 16,
            "best_latency_ms_p90": 190.0,
            "has_feasible": True,
        },
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_plot_moe_service_study_generates_outputs(tmp_path: Path) -> None:
    metrics_csv = tmp_path / "metrics.csv"
    decode_slo_csv = tmp_path / "decode_slo_summary.csv"
    capacity_csv = tmp_path / "capacity_frontier.csv"
    out_dir = tmp_path / "plots"

    _write_moe_metrics_csv(metrics_csv)
    _write_decode_slo_csv(decode_slo_csv)
    _write_capacity_csv(capacity_csv)

    subprocess.run(
        [
            sys.executable,
            "scripts/plot_moe_service_study.py",
            "--metrics-csv",
            str(metrics_csv),
            "--decode-slo-csv",
            str(decode_slo_csv),
            "--capacity-csv",
            str(capacity_csv),
            "--out-dir",
            str(out_dir),
            "--prefix",
            "unit_moe",
        ],
        check=True,
    )

    assert (out_dir / "unit_moe_decode_slo_frontier.png").exists()
    assert (out_dir / "unit_moe_capacity_frontier.png").exists()
    assert (out_dir / "unit_moe_locality_gain.png").exists()
    assert (out_dir / "unit_moe_remote_dispatch_ratio.png").exists()
    assert (out_dir / "unit_moe_comm_breakdown.png").exists()
