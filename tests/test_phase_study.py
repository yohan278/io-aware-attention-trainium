from __future__ import annotations

import csv
from pathlib import Path

from io_aware_attention.experiments import phase_study as ps
from io_aware_attention.experiments.phase_study import (
    KERNEL_PHASE_COLUMNS,
    REQUIRED_COLUMNS,
    _build_capacity_frontier,
    load_phase_study_config,
    run_phase_study,
)


def test_load_phase_study_smoke_config() -> None:
    cfg = load_phase_study_config(Path("configs/experiments/phase_smoke.yaml"))
    assert cfg.device == "cpu"
    assert cfg.dtype == "fp32"
    assert cfg.setups == ["single_die"]
    assert cfg.distributed is False
    assert cfg.dual_world_size == 2
    assert cfg.enable_fabric_calibration is False
    assert cfg.tensor_attention_naive_threshold == 0
    assert cfg.tensor_attention_tile_q == 64
    assert cfg.tensor_attention_tile_k == 128
    assert cfg.tensor_attention_reduce_group_k == 1
    assert cfg.tensor_attention_pipelined_prefill is True
    assert cfg.tensor_attention_pipelined_decode is False
    assert cfg.continue_on_runtime_error is True
    assert cfg.capacity_slo_ms == 250.0
    assert cfg.record_runtime_failures is True
    assert len(cfg.prefill_shapes) == 1
    assert len(cfg.decode_shapes) == 1


def test_phase_study_writes_expected_artifacts(tmp_path: Path) -> None:
    config_path = Path("configs/experiments/phase_smoke.yaml")
    cfg = load_phase_study_config(config_path)

    run_dir, records = run_phase_study(
        config=cfg,
        config_path=config_path,
        output_dir=tmp_path,
        device_override="cpu",
        distributed_override=False,
    )

    assert records
    metrics_csv = run_dir / "metrics.csv"
    metrics_jsonl = run_dir / "metrics.jsonl"
    kernel_phase_csv = run_dir / "kernel_phase_metrics.csv"
    collectives_json = run_dir / "collectives_summary.json"
    manifest_json = run_dir / "run_manifest.json"
    decode_slo_csv = run_dir / "decode_slo_summary.csv"
    break_even_csv = run_dir / "break_even_summary.csv"
    capacity_frontier_csv = run_dir / "capacity_frontier.csv"

    assert metrics_csv.exists()
    assert metrics_jsonl.exists()
    assert kernel_phase_csv.exists()
    assert collectives_json.exists()
    assert manifest_json.exists()
    assert decode_slo_csv.exists()
    assert break_even_csv.exists()
    assert capacity_frontier_csv.exists()

    with metrics_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert rows
        for col in REQUIRED_COLUMNS:
            assert col in rows[0]

    with kernel_phase_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert rows
        for col in KERNEL_PHASE_COLUMNS:
            assert col in rows[0]


def test_build_capacity_frontier_summary() -> None:
    rows = _build_capacity_frontier(
        [
            {
                "phase": "decode",
                "setup": "single_die",
                "context_len": 2048,
                "batch": 8,
                "latency_ms_p90": 120.0,
                "throughput_tokens_per_s": 700.0,
            },
            {
                "phase": "decode",
                "setup": "single_die",
                "context_len": 2048,
                "batch": 16,
                "latency_ms_p90": 320.0,
                "throughput_tokens_per_s": 1200.0,
            },
            {
                "phase": "decode",
                "setup": "dual_die_request_sharded",
                "context_len": 2048,
                "batch": 8,
                "latency_ms_p90": 130.0,
                "throughput_tokens_per_s": 900.0,
            },
            {
                "phase": "decode",
                "setup": "dual_die_request_sharded",
                "context_len": 2048,
                "batch": 16,
                "latency_ms_p90": 170.0,
                "throughput_tokens_per_s": 1500.0,
            },
        ],
        capacity_slo_ms=250.0,
    )
    keyed = {(row["setup"], int(row["context_len"])): row for row in rows}
    single = keyed[("single_die", 2048)]
    req = keyed[("dual_die_request_sharded", 2048)]

    assert int(single["max_feasible_concurrency"]) == 8
    assert int(single["max_tested_concurrency"]) == 16
    assert bool(single["has_feasible"]) is True
    assert float(single["best_throughput_tokens_per_s"]) == 700.0

    assert int(req["max_feasible_concurrency"]) == 16
    assert int(req["max_tested_concurrency"]) == 16
    assert bool(req["has_feasible"]) is True
    assert float(req["best_throughput_tokens_per_s"]) == 1500.0


def test_phase_study_runtime_failure_logging_continues(tmp_path: Path, monkeypatch) -> None:
    config_path = Path("configs/experiments/phase_smoke.yaml")
    cfg = load_phase_study_config(config_path)
    original = ps._prefill_step_single
    state = {"calls": 0}

    def flaky_prefill(*args: object, **kwargs: object):
        state["calls"] += 1
        # First call builds reference output; second call is setup execution.
        if state["calls"] == 2:
            raise RuntimeError("intentional prefill failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(ps, "_prefill_step_single", flaky_prefill)

    run_dir, records = run_phase_study(
        config=cfg,
        config_path=config_path,
        output_dir=tmp_path,
        device_override="cpu",
        distributed_override=False,
    )

    assert records
    failures_path = run_dir / "runtime_failures.jsonl"
    assert failures_path.exists()
    lines = [line for line in failures_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines
    assert any("intentional prefill failure" in line for line in lines)
