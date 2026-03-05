from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from io_aware_attention.runtime.trainium import (
    get_instance_type,
    get_torch_neuronx_version,
    get_visible_core_mask,
)

REQUIRED_METRIC_COLUMNS = [
    "timestamp",
    "variant",
    "device",
    "dtype",
    "batch",
    "heads",
    "seq_len",
    "head_dim",
    "latency_ms_p50",
    "latency_ms_p90",
    "throughput_tokens_per_s",
    "estimated_flops",
    "estimated_bytes",
    "arithmetic_intensity",
    "max_abs_err",
    "max_rel_err",
]


def utc_now_timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def create_run_dir(output_root: str | Path, run_id: str | None = None) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    resolved_run_id = run_id or f"run_{utc_now_timestamp()}"
    run_dir = root / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_metrics(run_dir: Path, records: list[dict[str, Any]]) -> tuple[Path, Path]:
    csv_path = run_dir / "metrics.csv"
    jsonl_path = run_dir / "metrics.jsonl"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_METRIC_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    return csv_path, jsonl_path


def get_git_commit(repo_root: Path) -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return output.strip()
    except Exception:
        return "unknown"


def build_run_manifest(
    repo_root: Path,
    benchmark_config_path: Path,
    variant: str,
    seed: int,
    emulation_level: str | None = None,
    distributed_enabled: bool | None = None,
    distributed_world_size: int | None = None,
    distributed_rank: int | None = None,
    rank_core_masks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    instance_type = get_instance_type()
    visible = get_visible_core_mask()
    local_rank = int(os.getenv("RANK", "0"))
    local_world_size = int(os.getenv("WORLD_SIZE", "1"))
    resolved_rank = int(local_rank if distributed_rank is None else distributed_rank)
    resolved_world_size = int(local_world_size if distributed_world_size is None else distributed_world_size)
    resolved_distributed = bool(resolved_world_size > 1 if distributed_enabled is None else distributed_enabled)

    if emulation_level is None:
        if resolved_distributed and instance_type.startswith("trn2.32xlarge"):
            resolved_emulation = "L1"
        elif resolved_distributed:
            resolved_emulation = "L0"
        else:
            resolved_emulation = "L2"
    else:
        resolved_emulation = emulation_level

    if rank_core_masks is None:
        rank_core_masks = [
            {
                "rank": resolved_rank,
                "visible_cores_raw": visible["raw"],
                "visible_cores": visible["parsed"],
                "chip_id": "unknown",
            }
        ]

    return {
        "git_commit": get_git_commit(repo_root),
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "instance_type": instance_type,
        "device_target": "trn2",
        "torch_version": torch.__version__,
        "torch_neuronx_version": get_torch_neuronx_version(),
        "python_version": ".".join(str(x) for x in __import__("sys").version_info[:3]),
        "benchmark_config_path": str(benchmark_config_path),
        "variant": variant,
        "seed": seed,
        "emulation_level": resolved_emulation,
        "distributed_enabled": resolved_distributed,
        "distributed_world_size": resolved_world_size,
        "distributed_rank": resolved_rank,
        "visible_cores_raw": visible["raw"],
        "visible_cores": visible["parsed"],
        "rank_core_masks": rank_core_masks,
    }


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    manifest_path = run_dir / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest_path
