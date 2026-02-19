from __future__ import annotations

import csv
from pathlib import Path

from io_aware_attention.bench.artifacts import REQUIRED_METRIC_COLUMNS
from io_aware_attention.bench.runner import run_benchmark
from io_aware_attention.config import load_benchmark_config


def test_runner_writes_expected_artifacts(tmp_path: Path) -> None:
    config = load_benchmark_config(Path("configs/benchmark/smoke.yaml"))
    run_dir, records = run_benchmark(
        config=config,
        config_path=Path("configs/benchmark/smoke.yaml"),
        output_dir=tmp_path,
        variant_override="naive",
        device_override="cpu",
    )

    assert records
    metrics_csv = run_dir / "metrics.csv"
    metrics_jsonl = run_dir / "metrics.jsonl"
    manifest_json = run_dir / "run_manifest.json"
    assert metrics_csv.exists()
    assert metrics_jsonl.exists()
    assert manifest_json.exists()

    with metrics_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert rows
        for col in REQUIRED_METRIC_COLUMNS:
            assert col in rows[0]

