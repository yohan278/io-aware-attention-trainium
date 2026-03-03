from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml

from io_aware_attention.bench.artifacts import (
    build_run_manifest,
    create_run_dir,
    write_manifest,
)
from io_aware_attention.runtime.trainium import mark_step_if_needed, resolve_device, sync_if_needed

KernelName = Literal["qkv_proj", "attention", "mlp", "rmsnorm", "out_proj"]
SetupName = Literal["single_die", "dual_die_naive", "dual_die_optimized"]
DTypeName = Literal["bf16", "fp32"]
DeviceName = Literal["cpu", "trainium"]

ALL_KERNELS: tuple[KernelName, ...] = ("qkv_proj", "attention", "mlp", "rmsnorm", "out_proj")
ALL_SETUPS: tuple[SetupName, ...] = ("single_die", "dual_die_naive", "dual_die_optimized")

REQUIRED_COLUMNS = [
    "timestamp",
    "setup",
    "kernel",
    "device",
    "dtype",
    "batch",
    "seq_len",
    "model_dim",
    "num_heads",
    "latency_ms_p50",
    "latency_ms_p90",
    "throughput_tokens_per_s",
    "estimated_flops",
    "estimated_hbm_bytes",
    "arithmetic_intensity",
    "communication_bytes",
    "communication_pct_of_hbm",
    "max_abs_err",
    "max_rel_err",
]


@dataclass(frozen=True)
class KernelShape:
    batch: int
    seq_len: int
    model_dim: int
    num_heads: int
    mlp_ratio: int = 4

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "KernelShape":
        return cls(
            batch=int(raw["batch"]),
            seq_len=int(raw["seq_len"]),
            model_dim=int(raw["model_dim"]),
            num_heads=int(raw["num_heads"]),
            mlp_ratio=int(raw.get("mlp_ratio", 4)),
        )

    def validate(self) -> None:
        if self.batch < 1 or self.seq_len < 1 or self.model_dim < 2 or self.num_heads < 1:
            raise ValueError(f"Invalid kernel shape: {self}")
        if self.model_dim % 2 != 0:
            raise ValueError(f"model_dim must be divisible by 2 for dual-die emulation: {self.model_dim}")
        if self.model_dim % self.num_heads != 0:
            raise ValueError(
                f"model_dim ({self.model_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.mlp_ratio < 1:
            raise ValueError(f"mlp_ratio must be >= 1, got {self.mlp_ratio}")


@dataclass(frozen=True)
class KernelStudyConfig:
    device: DeviceName
    dtype: DTypeName
    warmup_iters: int
    measure_iters: int
    seed: int
    kernels: list[KernelName]
    setups: list[SetupName]
    causal_attention: bool
    shapes: list[KernelShape]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "KernelStudyConfig":
        shapes = [KernelShape.from_dict(item) for item in raw["shapes"]]
        kernels = [str(item) for item in raw.get("kernels", list(ALL_KERNELS))]
        setups = [str(item) for item in raw.get("setups", list(ALL_SETUPS))]
        cfg = cls(
            device=str(raw.get("device", "cpu")),
            dtype=str(raw.get("dtype", "bf16")),
            warmup_iters=int(raw.get("warmup_iters", 2)),
            measure_iters=int(raw.get("measure_iters", 8)),
            seed=int(raw.get("seed", 0)),
            kernels=kernels,  # type: ignore[arg-type]
            setups=setups,  # type: ignore[arg-type]
            causal_attention=bool(raw.get("causal_attention", False)),
            shapes=shapes,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.device not in {"cpu", "trainium"}:
            raise ValueError(f"Unsupported device: {self.device}")
        if self.dtype not in {"bf16", "fp32"}:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        if self.warmup_iters < 0:
            raise ValueError("warmup_iters must be >= 0")
        if self.measure_iters < 1:
            raise ValueError("measure_iters must be >= 1")
        for kernel in self.kernels:
            if kernel not in ALL_KERNELS:
                raise ValueError(f"Unsupported kernel in config: {kernel}")
        for setup in self.setups:
            if setup not in ALL_SETUPS:
                raise ValueError(f"Unsupported setup in config: {setup}")
        for shape in self.shapes:
            shape.validate()


def load_kernel_study_config(path: str | Path) -> KernelStudyConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if "shapes" not in raw:
        raise ValueError(f"Missing required key 'shapes' in {config_path}")
    return KernelStudyConfig.from_dict(raw)


def _dtype_from_name(name: DTypeName) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype name: {name}")


def _dtype_bytes(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 2
    if dtype == torch.float32:
        return 4
    raise ValueError(f"Unsupported dtype: {dtype}")


def _percentile_ms(latencies_seconds: list[float], percentile: float) -> float:
    if not latencies_seconds:
        return 0.0
    return float(np.percentile(np.array(latencies_seconds) * 1000.0, percentile))


def _max_relative_error(reference: torch.Tensor, output: torch.Tensor) -> float:
    denom = reference.abs().clamp_min(1e-9)
    return float(((reference - output).abs() / denom).max().item())


def _timestamp_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _softmax_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    causal: bool,
    k_start: int,
) -> torch.Tensor:
    d_model = q.size(-1)
    scale_value = 1.0 / math.sqrt(float(d_model))
    logits = torch.matmul(q, k.transpose(-1, -2)) * scale_value
    if causal:
        seq_len = q.size(-2)
        tile_len = k.size(-2)
        q_pos = torch.arange(seq_len, device=q.device).view(1, 1, seq_len, 1)
        k_pos = torch.arange(k_start, k_start + tile_len, device=q.device).view(1, 1, 1, tile_len)
        logits = logits.masked_fill(k_pos > q_pos, float("-inf"))
    return logits


def _attention_single(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    logits = _softmax_logits(q.float(), k.float(), causal=causal, k_start=0)
    probs = torch.softmax(logits, dim=-1)
    out = torch.matmul(probs, v.float())
    return out.to(dtype=q.dtype)


def _attention_dual_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    dtype_bytes: int,
) -> tuple[torch.Tensor, float]:
    seq_len = k.size(-2)
    split = seq_len // 2
    k0, k1 = k[:, :, :split, :], k[:, :, split:, :]
    v0, v1 = v[:, :, :split, :], v[:, :, split:, :]

    logits0 = _softmax_logits(q.float(), k0.float(), causal=causal, k_start=0)
    logits1 = _softmax_logits(q.float(), k1.float(), causal=causal, k_start=split)
    logits = torch.cat([logits0, logits1], dim=-1)
    probs = torch.softmax(logits, dim=-1)
    p0, p1 = probs[:, :, :, :split], probs[:, :, :, split:]
    out = torch.matmul(p0, v0.float()) + torch.matmul(p1, v1.float())

    # Naive dual-die exchange of full logits across partitions.
    communication_bytes = float(logits.numel() * dtype_bytes)
    return out.to(dtype=q.dtype), communication_bytes


def _attention_dual_optimized(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    dtype_bytes: int,
) -> tuple[torch.Tensor, float]:
    seq_len = k.size(-2)
    split = seq_len // 2
    k0, k1 = k[:, :, :split, :], k[:, :, split:, :]
    v0, v1 = v[:, :, :split, :], v[:, :, split:, :]

    logits0 = _softmax_logits(q.float(), k0.float(), causal=causal, k_start=0)
    logits1 = _softmax_logits(q.float(), k1.float(), causal=causal, k_start=split)

    m0 = torch.max(logits0, dim=-1).values
    m1 = torch.max(logits1, dim=-1).values
    p0 = torch.exp(logits0 - m0.unsqueeze(-1))
    p1 = torch.exp(logits1 - m1.unsqueeze(-1))
    l0 = p0.sum(dim=-1)
    l1 = p1.sum(dim=-1)
    o0 = torch.matmul(p0, v0.float())
    o1 = torch.matmul(p1, v1.float())

    m = torch.maximum(m0, m1)
    a0 = torch.exp(m0 - m)
    a1 = torch.exp(m1 - m)
    l = a0 * l0 + a1 * l1
    out = (a0.unsqueeze(-1) * o0 + a1.unsqueeze(-1) * o1) / l.unsqueeze(-1).clamp_min(1e-9)

    # State exchange (m, l, o_partial) from each partition.
    b, h, s, d = q.shape
    scalars_per_query = 2 + d
    communication_bytes = float(2 * b * h * s * scalars_per_query * dtype_bytes)
    return out.to(dtype=q.dtype), communication_bytes


def _qkv_single(x: torch.Tensor, w_qkv: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, w_qkv)


def _qkv_dual_naive(
    x: torch.Tensor,
    w_qkv: torch.Tensor,
    dtype_bytes: int,
) -> tuple[torch.Tensor, float]:
    w0, w1 = torch.chunk(w_qkv, 2, dim=1)
    y0 = torch.matmul(x, w0)
    y1 = torch.matmul(x, w1)
    out = torch.cat([y0, y1], dim=-1)
    communication_bytes = float(out.numel() * dtype_bytes)
    return out, communication_bytes


def _qkv_dual_optimized(
    x: torch.Tensor,
    w_qkv: torch.Tensor,
    dtype_bytes: int,
) -> tuple[torch.Tensor, float]:
    w0, w1 = torch.chunk(w_qkv, 2, dim=1)
    y0 = torch.matmul(x, w0)
    y1 = torch.matmul(x, w1)
    out = torch.cat([y0, y1], dim=-1)
    # Optimized path keeps shards local and only gathers once at stage boundary.
    communication_bytes = float((out.numel() * dtype_bytes) / 2.0)
    return out, communication_bytes


def _mlp_single(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    hidden = torch.nn.functional.gelu(torch.matmul(x, w1), approximate="tanh")
    return torch.matmul(hidden, w2)


def _mlp_dual_naive(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    dtype_bytes: int,
) -> tuple[torch.Tensor, float]:
    w1_0, w1_1 = torch.chunk(w1, 2, dim=1)
    hidden0 = torch.nn.functional.gelu(torch.matmul(x, w1_0), approximate="tanh")
    hidden1 = torch.nn.functional.gelu(torch.matmul(x, w1_1), approximate="tanh")
    hidden_full = torch.cat([hidden0, hidden1], dim=-1)
    out = torch.matmul(hidden_full, w2)
    communication_bytes = float(hidden_full.numel() * dtype_bytes)
    return out, communication_bytes


def _mlp_dual_optimized(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    dtype_bytes: int,
) -> tuple[torch.Tensor, float]:
    w1_0, w1_1 = torch.chunk(w1, 2, dim=1)
    w2_0, w2_1 = torch.chunk(w2, 2, dim=0)

    out0 = torch.matmul(torch.nn.functional.gelu(torch.matmul(x, w1_0), approximate="tanh"), w2_0)
    out1 = torch.matmul(torch.nn.functional.gelu(torch.matmul(x, w1_1), approximate="tanh"), w2_1)
    out = out0 + out1
    communication_bytes = float(out.numel() * dtype_bytes)
    return out, communication_bytes


def _rmsnorm_single(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_f32 = x.float()
    inv_rms = torch.rsqrt(torch.mean(x_f32 * x_f32, dim=-1, keepdim=True) + eps)
    out = x_f32 * inv_rms * gamma.float()
    return out.to(dtype=x.dtype)


def _rmsnorm_dual_naive(
    x: torch.Tensor,
    gamma: torch.Tensor,
    dtype_bytes: int,
) -> tuple[torch.Tensor, float]:
    out = _rmsnorm_single(x, gamma)
    communication_bytes = float(x.numel() * dtype_bytes)
    return out, communication_bytes


def _rmsnorm_dual_optimized(
    x: torch.Tensor,
    gamma: torch.Tensor,
    dtype_bytes: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, float]:
    x0, x1 = torch.chunk(x.float(), 2, dim=-1)
    g0, g1 = torch.chunk(gamma.float(), 2, dim=-1)

    local0 = torch.sum(x0 * x0, dim=-1, keepdim=True)
    local1 = torch.sum(x1 * x1, dim=-1, keepdim=True)
    global_mean = (local0 + local1) / float(x.size(-1))
    inv_rms = torch.rsqrt(global_mean + eps)

    out0 = x0 * inv_rms * g0
    out1 = x1 * inv_rms * g1
    out = torch.cat([out0, out1], dim=-1).to(dtype=x.dtype)

    token_count = x.size(0) * x.size(1)
    communication_bytes = float(2 * token_count * dtype_bytes)
    return out, communication_bytes


def _out_proj_single(x: torch.Tensor, w_out: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, w_out)


def _out_proj_dual_naive(
    x: torch.Tensor,
    w_out: torch.Tensor,
    dtype_bytes: int,
) -> tuple[torch.Tensor, float]:
    out = torch.matmul(x, w_out)
    communication_bytes = float(x.numel() * dtype_bytes)
    return out, communication_bytes


def _out_proj_dual_optimized(
    x: torch.Tensor,
    w_out: torch.Tensor,
    dtype_bytes: int,
) -> tuple[torch.Tensor, float]:
    x0, x1 = torch.chunk(x, 2, dim=-1)
    w0, w1 = torch.chunk(w_out, 2, dim=0)
    out = torch.matmul(x0, w0) + torch.matmul(x1, w1)
    communication_bytes = float(out.numel() * dtype_bytes)
    return out, communication_bytes


def _estimate_flops(kernel: KernelName, shape: KernelShape) -> float:
    b, s, d = shape.batch, shape.seq_len, shape.model_dim
    h = shape.num_heads
    dh = d // h
    mlp_dim = shape.mlp_ratio * d
    if kernel == "qkv_proj":
        return 2.0 * b * s * d * (3 * d)
    if kernel == "attention":
        qk = 2.0 * b * h * s * s * dh
        av = 2.0 * b * h * s * s * dh
        softmax = 5.0 * b * h * s * s
        return qk + av + softmax
    if kernel == "mlp":
        first = 2.0 * b * s * d * mlp_dim
        second = 2.0 * b * s * mlp_dim * d
        gelu = 8.0 * b * s * mlp_dim
        return first + second + gelu
    if kernel == "rmsnorm":
        return 5.0 * b * s * d
    if kernel == "out_proj":
        return 2.0 * b * s * d * d
    raise ValueError(f"Unsupported kernel: {kernel}")


def _estimate_hbm_bytes(kernel: KernelName, shape: KernelShape, dtype_bytes: int) -> float:
    b, s, d = shape.batch, shape.seq_len, shape.model_dim
    h = shape.num_heads
    dh = d // h
    mlp_dim = shape.mlp_ratio * d
    if kernel == "qkv_proj":
        x = b * s * d
        w = d * (3 * d)
        y = b * s * (3 * d)
        return float((x + w + y) * dtype_bytes)
    if kernel == "attention":
        qkv = 3 * b * h * s * dh
        out = b * h * s * dh
        logits = b * h * s * s
        return float((qkv + out + logits) * dtype_bytes)
    if kernel == "mlp":
        x = b * s * d
        w = d * mlp_dim + mlp_dim * d
        hidden = b * s * mlp_dim
        out = b * s * d
        return float((x + w + hidden + out) * dtype_bytes)
    if kernel == "rmsnorm":
        x = b * s * d
        gamma = d
        out = b * s * d
        return float((x + gamma + out) * dtype_bytes)
    if kernel == "out_proj":
        x = b * s * d
        w = d * d
        out = b * s * d
        return float((x + w + out) * dtype_bytes)
    raise ValueError(f"Unsupported kernel: {kernel}")


def _benchmark_fn(
    fn: Any,
    warmup_iters: int,
    measure_iters: int,
    device: Any,
) -> tuple[torch.Tensor, list[float], float]:
    for _ in range(warmup_iters):
        _ = fn()
        mark_step_if_needed(device)
    sync_if_needed(device)

    latencies: list[float] = []
    total_comm_bytes = 0.0
    output = None

    for _ in range(measure_iters):
        start = time.perf_counter()
        output, comm_bytes = fn()
        mark_step_if_needed(device)
        sync_if_needed(device)
        end = time.perf_counter()
        latencies.append(end - start)
        total_comm_bytes += float(comm_bytes)

    assert output is not None
    avg_comm_bytes = total_comm_bytes / float(measure_iters)
    return output, latencies, avg_comm_bytes


def _init_kernel_tensors(
    kernel: KernelName,
    shape: KernelShape,
    dtype: torch.dtype,
    device: Any,
) -> dict[str, torch.Tensor]:
    b, s, d = shape.batch, shape.seq_len, shape.model_dim

    if kernel == "attention":
        h = shape.num_heads
        dh = d // h
        return {
            "q": torch.randn((b, h, s, dh), dtype=dtype, device=device),
            "k": torch.randn((b, h, s, dh), dtype=dtype, device=device),
            "v": torch.randn((b, h, s, dh), dtype=dtype, device=device),
        }

    x = torch.randn((b, s, d), dtype=dtype, device=device)
    if kernel == "qkv_proj":
        return {
            "x": x,
            "w_qkv": torch.randn((d, 3 * d), dtype=dtype, device=device),
        }
    if kernel == "mlp":
        mlp_dim = shape.mlp_ratio * d
        return {
            "x": x,
            "w1": torch.randn((d, mlp_dim), dtype=dtype, device=device),
            "w2": torch.randn((mlp_dim, d), dtype=dtype, device=device),
        }
    if kernel == "rmsnorm":
        return {
            "x": x,
            "gamma": torch.randn((d,), dtype=dtype, device=device),
        }
    if kernel == "out_proj":
        return {
            "x": x,
            "w_out": torch.randn((d, d), dtype=dtype, device=device),
        }

    raise ValueError(f"Unsupported kernel: {kernel}")


def _build_runner(
    kernel: KernelName,
    setup: SetupName,
    tensors: dict[str, torch.Tensor],
    causal_attention: bool,
    dtype_bytes: int,
) -> Any:
    if kernel == "qkv_proj":
        x = tensors["x"]
        w_qkv = tensors["w_qkv"]
        if setup == "single_die":
            return lambda: (_qkv_single(x, w_qkv), 0.0)
        if setup == "dual_die_naive":
            return lambda: _qkv_dual_naive(x, w_qkv, dtype_bytes)
        return lambda: _qkv_dual_optimized(x, w_qkv, dtype_bytes)

    if kernel == "attention":
        q, k, v = tensors["q"], tensors["k"], tensors["v"]
        if setup == "single_die":
            return lambda: (_attention_single(q, k, v, causal_attention), 0.0)
        if setup == "dual_die_naive":
            return lambda: _attention_dual_naive(q, k, v, causal_attention, dtype_bytes)
        return lambda: _attention_dual_optimized(q, k, v, causal_attention, dtype_bytes)

    if kernel == "mlp":
        x, w1, w2 = tensors["x"], tensors["w1"], tensors["w2"]
        if setup == "single_die":
            return lambda: (_mlp_single(x, w1, w2), 0.0)
        if setup == "dual_die_naive":
            return lambda: _mlp_dual_naive(x, w1, w2, dtype_bytes)
        return lambda: _mlp_dual_optimized(x, w1, w2, dtype_bytes)

    if kernel == "rmsnorm":
        x, gamma = tensors["x"], tensors["gamma"]
        if setup == "single_die":
            return lambda: (_rmsnorm_single(x, gamma), 0.0)
        if setup == "dual_die_naive":
            return lambda: _rmsnorm_dual_naive(x, gamma, dtype_bytes)
        return lambda: _rmsnorm_dual_optimized(x, gamma, dtype_bytes)

    if kernel == "out_proj":
        x, w_out = tensors["x"], tensors["w_out"]
        if setup == "single_die":
            return lambda: (_out_proj_single(x, w_out), 0.0)
        if setup == "dual_die_naive":
            return lambda: _out_proj_dual_naive(x, w_out, dtype_bytes)
        return lambda: _out_proj_dual_optimized(x, w_out, dtype_bytes)

    raise ValueError(f"Unsupported kernel/setup: {kernel}/{setup}")


def _write_metrics(run_dir: Path, records: list[dict[str, Any]]) -> tuple[Path, Path]:
    csv_path = run_dir / "metrics.csv"
    jsonl_path = run_dir / "metrics.jsonl"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    return csv_path, jsonl_path


def _write_summary(run_dir: Path, records: list[dict[str, Any]]) -> Path:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in records:
        key = (str(row["kernel"]), str(row["setup"]))
        by_key.setdefault(key, []).append(row)

    lines = [
        "# Kernel Study Summary",
        "",
        "| Kernel | Setup | Avg p50 latency (ms) | Avg throughput (tokens/s) | Avg communication (bytes) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]

    for (kernel, setup), rows in sorted(by_key.items()):
        p50 = float(np.mean([float(item["latency_ms_p50"]) for item in rows]))
        thr = float(np.mean([float(item["throughput_tokens_per_s"]) for item in rows]))
        comm = float(np.mean([float(item["communication_bytes"]) for item in rows]))
        lines.append(f"| {kernel} | {setup} | {p50:.4f} | {thr:.2f} | {comm:.2f} |")

    summary_path = run_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def run_kernel_study(
    config: KernelStudyConfig,
    config_path: Path,
    output_dir: Path,
    device_override: str | None = None,
    setups_override: list[SetupName] | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    device_name = str(device_override or config.device)
    dtype = _dtype_from_name(config.dtype)
    dtype_bytes = _dtype_bytes(dtype)
    device = resolve_device(device_name)

    run_dir = create_run_dir(output_dir)
    records: list[dict[str, Any]] = []

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    selected_setups = setups_override or config.setups
    for setup in selected_setups:
        if setup not in ALL_SETUPS:
            raise ValueError(f"Unsupported setup override: {setup}")

    for shape_idx, shape in enumerate(config.shapes):
        for kernel_idx, kernel in enumerate(config.kernels):
            # Deterministic parameter generation per shape/kernel across setups.
            torch.manual_seed(config.seed + shape_idx * 1000 + kernel_idx)
            tensors = _init_kernel_tensors(kernel, shape, dtype=dtype, device=device)

            baseline_runner = _build_runner(
                kernel=kernel,
                setup="single_die",
                tensors=tensors,
                causal_attention=config.causal_attention,
                dtype_bytes=dtype_bytes,
            )
            baseline_out, _, _ = _benchmark_fn(
                baseline_runner,
                warmup_iters=1,
                measure_iters=1,
                device=device,
            )
            baseline_cpu = baseline_out.detach().to("cpu", dtype=torch.float32)

            for setup in selected_setups:
                runner = _build_runner(
                    kernel=kernel,
                    setup=setup,
                    tensors=tensors,
                    causal_attention=config.causal_attention,
                    dtype_bytes=dtype_bytes,
                )
                output, latencies, avg_comm_bytes = _benchmark_fn(
                    runner,
                    warmup_iters=config.warmup_iters,
                    measure_iters=config.measure_iters,
                    device=device,
                )

                output_cpu = output.detach().to("cpu", dtype=torch.float32)
                max_abs_err = float((baseline_cpu - output_cpu).abs().max().item())
                max_rel_err = _max_relative_error(baseline_cpu, output_cpu)

                p50_ms = _percentile_ms(latencies, 50)
                p90_ms = _percentile_ms(latencies, 90)
                throughput = (shape.batch * shape.seq_len) / (p50_ms / 1000.0) if p50_ms > 0 else 0.0

                estimated_flops = _estimate_flops(kernel, shape)
                estimated_hbm_bytes = _estimate_hbm_bytes(kernel, shape, dtype_bytes=dtype_bytes)
                arithmetic_intensity = (
                    estimated_flops / estimated_hbm_bytes if estimated_hbm_bytes > 0 else 0.0
                )
                comm_pct = (avg_comm_bytes / estimated_hbm_bytes * 100.0) if estimated_hbm_bytes > 0 else 0.0

                record = {
                    "timestamp": _timestamp_utc(),
                    "setup": setup,
                    "kernel": kernel,
                    "device": device_name,
                    "dtype": config.dtype,
                    "batch": shape.batch,
                    "seq_len": shape.seq_len,
                    "model_dim": shape.model_dim,
                    "num_heads": shape.num_heads,
                    "latency_ms_p50": round(float(p50_ms), 6),
                    "latency_ms_p90": round(float(p90_ms), 6),
                    "throughput_tokens_per_s": round(float(throughput), 6),
                    "estimated_flops": round(float(estimated_flops), 2),
                    "estimated_hbm_bytes": round(float(estimated_hbm_bytes), 2),
                    "arithmetic_intensity": round(float(arithmetic_intensity), 6),
                    "communication_bytes": round(float(avg_comm_bytes), 2),
                    "communication_pct_of_hbm": round(float(comm_pct), 6),
                    "max_abs_err": round(float(max_abs_err), 8),
                    "max_rel_err": round(float(max_rel_err), 8),
                }
                missing = [col for col in REQUIRED_COLUMNS if col not in record]
                if missing:
                    raise RuntimeError(f"Metric record missing required columns: {missing}")
                records.append(record)

    _write_metrics(run_dir, records)
    _write_summary(run_dir, records)

    repo_root = Path(__file__).resolve().parents[3]
    manifest = build_run_manifest(
        repo_root=repo_root,
        benchmark_config_path=config_path.resolve(),
        variant="kernel_study",
        seed=config.seed,
    )
    manifest.update(
        {
            "setups": list(selected_setups),
            "kernels": list(config.kernels),
            "causal_attention": config.causal_attention,
        }
    )
    write_manifest(run_dir, manifest)

    return run_dir, records


def run_kernel_once_for_testing(
    kernel: KernelName,
    setup: SetupName,
    shape: KernelShape,
    dtype_name: DTypeName = "fp32",
    causal_attention: bool = False,
    seed: int = 0,
) -> torch.Tensor:
    """Small deterministic helper used in unit tests."""
    dtype = _dtype_from_name(dtype_name)
    dtype_bytes = _dtype_bytes(dtype)
    device = torch.device("cpu")
    torch.manual_seed(seed)
    tensors = _init_kernel_tensors(kernel, shape, dtype=dtype, device=device)
    runner = _build_runner(
        kernel=kernel,
        setup=setup,
        tensors=tensors,
        causal_attention=causal_attention,
        dtype_bytes=dtype_bytes,
    )
    out, _ = runner()
    return out.detach().to("cpu", dtype=torch.float32)
