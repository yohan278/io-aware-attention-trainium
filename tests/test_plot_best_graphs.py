from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_plot_best_graphs_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "plot_best_graphs.py"
    spec = importlib.util.spec_from_file_location("plot_best_graphs", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_hybrid_policy_rows_uses_mixed_prefill_decode_setups() -> None:
    mod = _load_plot_best_graphs_module()
    metrics = pd.DataFrame(
        [
            {
                "phase": "prefill",
                "setup": "single_die",
                "seq_len": 2048,
                "context_len": 0,
                "batch": 2,
                "latency_ms_p50": 20.0,
                "throughput_tokens_per_s": 200000.0,
            },
            {
                "phase": "prefill",
                "setup": "dual_die_request_sharded",
                "seq_len": 2048,
                "context_len": 0,
                "batch": 2,
                "latency_ms_p50": 22.0,
                "throughput_tokens_per_s": 190000.0,
            },
            {
                "phase": "prefill",
                "setup": "dual_die_tensor_optimized",
                "seq_len": 2048,
                "context_len": 0,
                "batch": 2,
                "latency_ms_p50": 120.0,
                "throughput_tokens_per_s": 10000.0,
            },
            {
                "phase": "decode",
                "setup": "single_die",
                "seq_len": 1,
                "context_len": 2048,
                "batch": 16,
                "latency_ms_p90": 180.0,
                "throughput_tokens_per_s": 1200.0,
            },
            {
                "phase": "decode",
                "setup": "dual_die_request_sharded",
                "seq_len": 1,
                "context_len": 2048,
                "batch": 16,
                "latency_ms_p90": 160.0,
                "throughput_tokens_per_s": 1500.0,
            },
            {
                "phase": "decode",
                "setup": "dual_die_tensor_optimized",
                "seq_len": 1,
                "context_len": 2048,
                "batch": 16,
                "latency_ms_p90": 500.0,
                "throughput_tokens_per_s": 300.0,
            },
        ]
    )

    hybrid = mod._build_hybrid_policy_rows(
        metrics=metrics,
        output_tokens=128,
        context_override=2048,
        conc_override=16,
    )

    assert hybrid is not None
    frame, context, concurrency = hybrid
    assert context == 2048
    assert concurrency == 16

    by_policy = {str(row["policy"]): row for row in frame.to_dict(orient="records")}
    assert by_policy["single->request"]["prefill_setup"] == "single_die"
    assert by_policy["single->request"]["decode_setup"] == "dual_die_request_sharded"
    assert by_policy["single->request"]["prefill_per_request_ms"] == 10.0
    assert by_policy["single->request"]["decode_tail_ms"] < by_policy["single->single"]["decode_tail_ms"]
    assert by_policy["single->tensor"]["total_request_ms"] > by_policy["single->single"]["total_request_ms"]
