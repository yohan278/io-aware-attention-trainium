from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

KernelVariant = Literal[
    "naive",
    "tiled_online",
    "tiled_online_dbuffer",
    "tiled_online_dist_merge_sync",
    "tiled_online_dist_merge_pipelined",
]
DeviceType = Literal["cpu", "trainium"]
DType = Literal["bf16", "fp32"]


@dataclass(frozen=True)
class AttentionShape:
    batch: int
    heads: int
    seq_len: int
    head_dim: int

    @classmethod
    def from_dict(cls, raw: dict) -> "AttentionShape":
        return cls(
            batch=int(raw["batch"]),
            heads=int(raw["heads"]),
            seq_len=int(raw["seq_len"]),
            head_dim=int(raw["head_dim"]),
        )

    def validate(self) -> None:
        if self.batch < 1 or self.heads < 1 or self.seq_len < 1 or self.head_dim < 1:
            raise ValueError(f"Invalid shape values: {self}")


@dataclass(frozen=True)
class BenchmarkConfig:
    variant: KernelVariant
    device: DeviceType
    dtype: DType
    causal: bool
    warmup_iters: int
    measure_iters: int
    seed: int
    shapes: list[AttentionShape]

    @classmethod
    def from_dict(cls, raw: dict) -> "BenchmarkConfig":
        shapes = [AttentionShape.from_dict(item) for item in raw["shapes"]]
        cfg = cls(
            variant=str(raw.get("variant", "naive")),
            device=str(raw.get("device", "cpu")),
            dtype=str(raw.get("dtype", "bf16")),
            causal=bool(raw.get("causal", False)),
            warmup_iters=int(raw.get("warmup_iters", 2)),
            measure_iters=int(raw.get("measure_iters", 10)),
            seed=int(raw.get("seed", 0)),
            shapes=shapes,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.variant not in {
            "naive",
            "tiled_online",
            "tiled_online_dbuffer",
            "tiled_online_dist_merge_sync",
            "tiled_online_dist_merge_pipelined",
        }:
            raise ValueError(f"Unsupported variant: {self.variant}")
        if self.device not in {"cpu", "trainium"}:
            raise ValueError(f"Unsupported device: {self.device}")
        if self.dtype not in {"bf16", "fp32"}:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        if self.warmup_iters < 0:
            raise ValueError("warmup_iters must be >= 0")
        if self.measure_iters < 1:
            raise ValueError("measure_iters must be >= 1")
        for shape in self.shapes:
            shape.validate()


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if "shapes" not in raw:
        raise ValueError(f"Missing required key 'shapes' in {config_path}")
    return BenchmarkConfig.from_dict(raw)
