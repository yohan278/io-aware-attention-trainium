from __future__ import annotations

import csv
from pathlib import Path

from io_aware_attention.experiments.phase_study import (
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
    manifest_json = run_dir / "run_manifest.json"
    decode_slo_csv = run_dir / "decode_slo_summary.csv"
    break_even_csv = run_dir / "break_even_summary.csv"

    assert metrics_csv.exists()
    assert metrics_jsonl.exists()
    assert manifest_json.exists()
    assert decode_slo_csv.exists()
    assert break_even_csv.exists()

    with metrics_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert rows
        for col in REQUIRED_COLUMNS:
            assert col in rows[0]
