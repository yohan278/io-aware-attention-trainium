from __future__ import annotations

import csv
from pathlib import Path

from io_aware_attention.experiments.moe_service_study import (
    REQUIRED_COLUMNS,
    _build_capacity_frontier,
    load_moe_study_config,
    run_moe_service_study,
)


def test_load_moe_smoke_config() -> None:
    cfg = load_moe_study_config(Path("configs/experiments/moe_smoke.yaml"))
    assert cfg.device == "cpu"
    assert cfg.dtype == "fp32"
    assert cfg.distributed is False
    assert cfg.dual_world_size == 2
    assert cfg.setups == ["single_die"]
    assert cfg.capacity_slo_ms == 100.0
    assert cfg.record_runtime_failures is True
    assert len(cfg.decode_shapes) == 1


def test_moe_study_writes_expected_artifacts(tmp_path: Path) -> None:
    config_path = Path("configs/experiments/moe_smoke.yaml")
    cfg = load_moe_study_config(config_path)

    run_dir, records = run_moe_service_study(
        config=cfg,
        config_path=config_path,
        output_dir=tmp_path,
        device_override="cpu",
        distributed_override=False,
    )

    assert records
    metrics_csv = run_dir / "metrics.csv"
    metrics_jsonl = run_dir / "metrics.jsonl"
    collectives_json = run_dir / "collectives_summary.json"
    decode_slo_csv = run_dir / "decode_slo_summary.csv"
    capacity_csv = run_dir / "capacity_frontier.csv"
    manifest_json = run_dir / "run_manifest.json"

    assert metrics_csv.exists()
    assert metrics_jsonl.exists()
    assert collectives_json.exists()
    assert decode_slo_csv.exists()
    assert capacity_csv.exists()
    assert manifest_json.exists()

    with metrics_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    for col in REQUIRED_COLUMNS:
        assert col in rows[0]

    with decode_slo_csv.open("r", encoding="utf-8") as handle:
        slo_rows = list(csv.DictReader(handle))
    assert slo_rows
    assert "routing_skew" in slo_rows[0]
    assert "num_experts" in slo_rows[0]
    assert "top_k" in slo_rows[0]

    with capacity_csv.open("r", encoding="utf-8") as handle:
        cap_rows = list(csv.DictReader(handle))
    assert cap_rows
    assert "routing_skew" in cap_rows[0]
    assert "num_experts" in cap_rows[0]
    assert "top_k" in cap_rows[0]


def test_capacity_frontier_keeps_routing_skew_dimension() -> None:
    rows = _build_capacity_frontier(
        [
            {
                "setup": "dual_die_moe_naive",
                "context_len": 2048,
                "batch": 8,
                "latency_ms_p90": 120.0,
                "throughput_tokens_per_s": 500.0,
                "num_experts": 8,
                "top_k": 2,
                "routing_skew": 0.0,
            },
            {
                "setup": "dual_die_moe_naive",
                "context_len": 2048,
                "batch": 8,
                "latency_ms_p90": 320.0,
                "throughput_tokens_per_s": 400.0,
                "num_experts": 8,
                "top_k": 2,
                "routing_skew": 1.8,
            },
        ],
        capacity_slo_ms=250.0,
    )

    keyed = {
        (
            str(row["setup"]),
            int(row["context_len"]),
            float(row["routing_skew"]),
        ): row
        for row in rows
    }

    assert keyed[("dual_die_moe_naive", 2048, 0.0)]["has_feasible"] is True
    assert keyed[("dual_die_moe_naive", 2048, 1.8)]["has_feasible"] is False
