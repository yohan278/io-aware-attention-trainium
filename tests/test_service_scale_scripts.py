from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys


def _write_metrics_csv(path: Path) -> None:
    fields = [
        "phase",
        "setup",
        "seq_len",
        "context_len",
        "batch",
        "latency_ms_p90",
        "throughput_tokens_per_s",
    ]
    rows = [
        # Prefill rows
        {"phase": "prefill", "setup": "single_die", "seq_len": 2048, "context_len": 0, "batch": 2, "latency_ms_p90": 20, "throughput_tokens_per_s": 220000},
        {"phase": "prefill", "setup": "single_die", "seq_len": 4096, "context_len": 0, "batch": 2, "latency_ms_p90": 40, "throughput_tokens_per_s": 200000},
        {"phase": "prefill", "setup": "single_die", "seq_len": 8192, "context_len": 0, "batch": 2, "latency_ms_p90": 85, "throughput_tokens_per_s": 180000},
        {"phase": "prefill", "setup": "dual_die_tensor_optimized", "seq_len": 2048, "context_len": 0, "batch": 2, "latency_ms_p90": 400, "throughput_tokens_per_s": 10000},
        {"phase": "prefill", "setup": "dual_die_tensor_optimized", "seq_len": 4096, "context_len": 0, "batch": 2, "latency_ms_p90": 1600, "throughput_tokens_per_s": 5000},
        {"phase": "prefill", "setup": "dual_die_tensor_optimized", "seq_len": 8192, "context_len": 0, "batch": 2, "latency_ms_p90": 3200, "throughput_tokens_per_s": 3000},
        {"phase": "prefill", "setup": "dual_die_request_sharded", "seq_len": 2048, "context_len": 0, "batch": 2, "latency_ms_p90": 25, "throughput_tokens_per_s": 210000},
        {"phase": "prefill", "setup": "dual_die_request_sharded", "seq_len": 4096, "context_len": 0, "batch": 2, "latency_ms_p90": 45, "throughput_tokens_per_s": 190000},
        {"phase": "prefill", "setup": "dual_die_request_sharded", "seq_len": 8192, "context_len": 0, "batch": 2, "latency_ms_p90": 90, "throughput_tokens_per_s": 170000},
        # Decode rows
        {"phase": "decode", "setup": "single_die", "seq_len": 1, "context_len": 2048, "batch": 8, "latency_ms_p90": 80, "throughput_tokens_per_s": 700},
        {"phase": "decode", "setup": "single_die", "seq_len": 1, "context_len": 2048, "batch": 16, "latency_ms_p90": 180, "throughput_tokens_per_s": 1200},
        {"phase": "decode", "setup": "single_die", "seq_len": 1, "context_len": 4096, "batch": 8, "latency_ms_p90": 90, "throughput_tokens_per_s": 650},
        {"phase": "decode", "setup": "single_die", "seq_len": 1, "context_len": 4096, "batch": 16, "latency_ms_p90": 210, "throughput_tokens_per_s": 1000},
        {"phase": "decode", "setup": "single_die", "seq_len": 1, "context_len": 8192, "batch": 8, "latency_ms_p90": 140, "throughput_tokens_per_s": 500},
        {"phase": "decode", "setup": "dual_die_tensor_optimized", "seq_len": 1, "context_len": 2048, "batch": 8, "latency_ms_p90": 300, "throughput_tokens_per_s": 200},
        {"phase": "decode", "setup": "dual_die_tensor_optimized", "seq_len": 1, "context_len": 2048, "batch": 16, "latency_ms_p90": 500, "throughput_tokens_per_s": 300},
        {"phase": "decode", "setup": "dual_die_request_sharded", "seq_len": 1, "context_len": 2048, "batch": 8, "latency_ms_p90": 85, "throughput_tokens_per_s": 800},
        {"phase": "decode", "setup": "dual_die_request_sharded", "seq_len": 1, "context_len": 2048, "batch": 16, "latency_ms_p90": 160, "throughput_tokens_per_s": 1500},
        {"phase": "decode", "setup": "dual_die_request_sharded", "seq_len": 1, "context_len": 4096, "batch": 8, "latency_ms_p90": 95, "throughput_tokens_per_s": 700},
        {"phase": "decode", "setup": "dual_die_request_sharded", "seq_len": 1, "context_len": 4096, "batch": 16, "latency_ms_p90": 190, "throughput_tokens_per_s": 1300},
        {"phase": "decode", "setup": "dual_die_request_sharded", "seq_len": 1, "context_len": 8192, "batch": 8, "latency_ms_p90": 170, "throughput_tokens_per_s": 520},
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_plot_capacity_frontier_script_generates_outputs(tmp_path: Path) -> None:
    metrics_csv = tmp_path / "metrics.csv"
    capacity_csv = tmp_path / "capacity_frontier.csv"
    _write_metrics_csv(metrics_csv)

    with capacity_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "setup",
                "context_len",
                "slo_ms",
                "max_tested_concurrency",
                "max_feasible_concurrency",
                "best_throughput_tokens_per_s",
                "best_concurrency",
                "best_latency_ms_p90",
                "feasible_points",
                "total_points",
                "has_feasible",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "setup": "single_die",
                "context_len": 2048,
                "slo_ms": 250.0,
                "max_tested_concurrency": 16,
                "max_feasible_concurrency": 16,
                "best_throughput_tokens_per_s": 1200.0,
                "best_concurrency": 16,
                "best_latency_ms_p90": 180.0,
                "feasible_points": 2,
                "total_points": 2,
                "has_feasible": True,
            }
        )

    out_dir = tmp_path / "plots"
    subprocess.run(
        [
            sys.executable,
            "scripts/plot_capacity_frontier.py",
            "--metrics-csv",
            str(metrics_csv),
            "--capacity-csv",
            str(capacity_csv),
            "--out-dir",
            str(out_dir),
            "--prefix",
            "unit",
        ],
        check=True,
    )

    assert (out_dir / "unit_capacity_frontier.png").exists()
    assert (out_dir / "unit_concurrency_scaling.png").exists()


def test_simulate_mixed_traffic_is_deterministic(tmp_path: Path) -> None:
    metrics_csv = tmp_path / "metrics.csv"
    _write_metrics_csv(metrics_csv)

    out_a = tmp_path / "sim_a"
    out_b = tmp_path / "sim_b"
    out_a.mkdir(parents=True, exist_ok=True)
    out_b.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        sys.executable,
        "scripts/simulate_mixed_traffic.py",
        "--metrics-csv",
        str(metrics_csv),
        "--seed",
        "7",
        "--duration-s",
        "60",
        "--arrival-rate-rps",
        "8",
        "--prefill-ratio",
        "0.3",
        "--decode-slo-ms",
        "250",
        "--drop-wait-ms",
        "2000",
        "--decode-tokens",
        "64",
        "--context-weights",
        "2048:0.5,4096:0.35,8192:0.15",
        "--prefix",
        "unit",
    ]

    subprocess.run(base_cmd + ["--out-dir", str(out_a)], check=True)
    subprocess.run(base_cmd + ["--out-dir", str(out_b)], check=True)

    summary_a = (out_a / "service_trace_summary.csv").read_text(encoding="utf-8")
    summary_b = (out_b / "service_trace_summary.csv").read_text(encoding="utf-8")
    assert summary_a == summary_b
    assert (out_a / "unit_mixed_trace_goodput.png").exists()
