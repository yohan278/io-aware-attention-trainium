from __future__ import annotations

import csv
from pathlib import Path

from io_aware_attention.experiments.phase_study import (
    KERNEL_PHASE_COLUMNS,
    REQUIRED_COLUMNS,
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

    assert metrics_csv.exists()
    assert metrics_jsonl.exists()
    assert kernel_phase_csv.exists()
    assert collectives_json.exists()
    assert manifest_json.exists()
    assert decode_slo_csv.exists()
    assert break_even_csv.exists()

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
