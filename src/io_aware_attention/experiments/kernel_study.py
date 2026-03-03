from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, field
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
from io_aware_attention.runtime.trainium import (
    DistributedContext,
    distributed_barrier,
    finalize_distributed_context,
    init_distributed_context,
    mark_step_if_needed,
    resolve_device,
    sync_if_needed,
)

KernelName = Literal["qkv_proj", "attention", "mlp", "rmsnorm", "out_proj"]
SetupName = Literal["single_die", "dual_die_naive", "dual_die_optimized"]
DTypeName = Literal["bf16", "fp32"]
DeviceName = Literal["cpu", "trainium"]

ALL_KERNELS: tuple[KernelName, ...] = ("qkv_proj", "attention", "mlp", "rmsnorm", "out_proj")
ALL_SETUPS: tuple[SetupName, ...] = ("single_die", "dual_die_naive", "dual_die_optimized")

DEFAULT_FABRIC_MESSAGE_SIZES = [1024, 4096, 16384, 65536, 262144, 1048576]

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
    "compute_ms_p50",
    "communication_ms_p50",
    "overlap_pct_p50",
    "throughput_tokens_per_s",
    "estimated_flops",
    "estimated_hbm_bytes",
    "arithmetic_intensity",
    "communication_bytes",
    "communication_pct_of_hbm",
    "achieved_link_gbps_p50",
    "link_utilization_pct_p50",
    "fabric_peak_gbps",
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

    def validate(self, dual_world_size: int = 2) -> None:
        if self.batch < 1 or self.seq_len < 1 or self.model_dim < 2 or self.num_heads < 1:
            raise ValueError(f"Invalid kernel shape: {self}")
        if self.model_dim % dual_world_size != 0:
            raise ValueError(
                "model_dim must be divisible by dual_world_size for dual-die partitioning: "
                f"{self.model_dim} % {dual_world_size} != 0"
            )
        if self.seq_len % dual_world_size != 0:
            raise ValueError(
                "seq_len must be divisible by dual_world_size for sequence partitioning: "
                f"{self.seq_len} % {dual_world_size} != 0"
            )
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
    distributed: bool
    dual_world_size: int
    enable_fabric_calibration: bool
    fabric_message_sizes: list[int]
    fabric_warmup_iters: int
    fabric_measure_iters: int
    enforce_correctness: bool
    correctness_abs_tol: float | None
    correctness_rel_tol: float | None

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
            distributed=bool(raw.get("distributed", False)),
            dual_world_size=int(raw.get("dual_world_size", 2)),
            enable_fabric_calibration=bool(raw.get("enable_fabric_calibration", True)),
            fabric_message_sizes=[
                int(item) for item in raw.get("fabric_message_sizes", DEFAULT_FABRIC_MESSAGE_SIZES)
            ],
            fabric_warmup_iters=int(raw.get("fabric_warmup_iters", 2)),
            fabric_measure_iters=int(raw.get("fabric_measure_iters", 8)),
            enforce_correctness=bool(raw.get("enforce_correctness", True)),
            correctness_abs_tol=(
                None if raw.get("correctness_abs_tol") is None else float(raw["correctness_abs_tol"])
            ),
            correctness_rel_tol=(
                None if raw.get("correctness_rel_tol") is None else float(raw["correctness_rel_tol"])
            ),
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
        if self.dual_world_size != 2:
            raise ValueError(f"This experiment currently supports dual_world_size=2 only, got {self.dual_world_size}")
        if self.fabric_warmup_iters < 0 or self.fabric_measure_iters < 1:
            raise ValueError("fabric calibration warmup/measure iters must be >= 0 / >= 1")
        if not self.fabric_message_sizes:
            raise ValueError("fabric_message_sizes must not be empty")
        for size in self.fabric_message_sizes:
            if size < 1:
                raise ValueError(f"fabric message sizes must be >= 1 bytes, got {size}")
        for kernel in self.kernels:
            if kernel not in ALL_KERNELS:
                raise ValueError(f"Unsupported kernel in config: {kernel}")
        for setup in self.setups:
            if setup not in ALL_SETUPS:
                raise ValueError(f"Unsupported setup in config: {setup}")
        for shape in self.shapes:
            shape.validate(self.dual_world_size)


@dataclass
class CommMetrics:
    bytes_total: float = 0.0
    time_s_total: float = 0.0
    bytes_by_op: dict[str, float] = field(default_factory=dict)
    time_by_op: dict[str, float] = field(default_factory=dict)

    def add(self, op_name: str, payload_bytes: float, elapsed_s: float) -> None:
        self.bytes_total += float(payload_bytes)
        self.time_s_total += float(elapsed_s)
        self.bytes_by_op[op_name] = self.bytes_by_op.get(op_name, 0.0) + float(payload_bytes)
        self.time_by_op[op_name] = self.time_by_op.get(op_name, 0.0) + float(elapsed_s)


@dataclass
class BenchmarkResult:
    output: torch.Tensor
    latencies_s: list[float]
    compute_s: list[float]
    communication_s: list[float]
    overlap_pct: list[float]
    communication_bytes: list[float]
    achieved_link_gbps: list[float]


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


def _correctness_thresholds(config: KernelStudyConfig) -> tuple[float, float]:
    if config.correctness_abs_tol is not None and config.correctness_rel_tol is not None:
        return float(config.correctness_abs_tol), float(config.correctness_rel_tol)
    if config.dtype == "fp32":
        return 1e-4, 5e-4
    return 1e-2, 2e-2


def _relative_error_eps(dtype_name: DTypeName) -> float:
    if dtype_name == "fp32":
        return 1e-6
    return 1e-3


def _percentile_ms(latencies_seconds: list[float], percentile: float) -> float:
    if not latencies_seconds:
        return 0.0
    return float(np.percentile(np.array(latencies_seconds) * 1000.0, percentile))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values), percentile))


def _max_relative_error(reference: torch.Tensor, output: torch.Tensor, eps: float) -> float:
    scale = torch.maximum(reference.abs(), output.abs()).clamp_min(eps)
    return float(((reference - output).abs() / scale).max().item())


def _timestamp_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _matmul_fp32(a: torch.Tensor, b: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    return torch.matmul(a.float(), b.float()).to(dtype=out_dtype)


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


def _attention_dual_naive_local(
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

    communication_bytes = float(logits.numel() * dtype_bytes)
    return out.to(dtype=q.dtype), communication_bytes


def _attention_dual_optimized_local(
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
    m = torch.maximum(m0, m1)

    stable0 = logits0 - m.unsqueeze(-1)
    stable1 = logits1 - m.unsqueeze(-1)
    stable0 = torch.where(torch.isfinite(stable0), stable0, torch.full_like(stable0, -1e9))
    stable1 = torch.where(torch.isfinite(stable1), stable1, torch.full_like(stable1, -1e9))

    p0 = torch.exp(stable0)
    p1 = torch.exp(stable1)
    l = p0.sum(dim=-1) + p1.sum(dim=-1)
    out = (torch.matmul(p0, v0.float()) + torch.matmul(p1, v1.float())) / l.unsqueeze(-1).clamp_min(1e-9)

    b, h, s, d = q.shape
    scalars_per_query = 2 + d
    communication_bytes = float(2 * b * h * s * scalars_per_query * dtype_bytes)
    return out.to(dtype=q.dtype), communication_bytes


def _qkv_single(x: torch.Tensor, w_qkv: torch.Tensor) -> torch.Tensor:
    return _matmul_fp32(x, w_qkv, out_dtype=x.dtype)


def _qkv_dual_naive_local(
    x: torch.Tensor,
    w_qkv: torch.Tensor,
    dtype_bytes: int,
) -> tuple[torch.Tensor, float]:
    w0, w1 = torch.chunk(w_qkv, 2, dim=1)
    y0 = _matmul_fp32(x, w0, out_dtype=x.dtype)
    y1 = _matmul_fp32(x, w1, out_dtype=x.dtype)
    out = torch.cat([y0, y1], dim=-1)
    communication_bytes = float(2 * out.numel() * dtype_bytes)
    return out, communication_bytes


def _qkv_dual_optimized_local(
    x: torch.Tensor,
    w_qkv: torch.Tensor,
    dtype_bytes: int,
) -> tuple[torch.Tensor, float]:
    w0, w1 = torch.chunk(w_qkv, 2, dim=1)
    y0 = _matmul_fp32(x, w0, out_dtype=x.dtype)
    y1 = _matmul_fp32(x, w1, out_dtype=x.dtype)
    out = torch.cat([y0, y1], dim=-1)
    communication_bytes = float(out.numel() * dtype_bytes)
    return out, communication_bytes


def _mlp_single(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    hidden = torch.nn.functional.gelu(_matmul_fp32(x, w1, out_dtype=torch.float32), approximate="tanh")
    out = torch.matmul(hidden, w2.float())
    return out.to(dtype=x.dtype)


def _mlp_dual_naive_local(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    dtype_bytes: int,
) -> tuple[torch.Tensor, float]:
    w1_0, w1_1 = torch.chunk(w1, 2, dim=1)
    hidden0 = torch.nn.functional.gelu(_matmul_fp32(x, w1_0, out_dtype=torch.float32), approximate="tanh")
    hidden1 = torch.nn.functional.gelu(_matmul_fp32(x, w1_1, out_dtype=torch.float32), approximate="tanh")
    hidden_full = torch.cat([hidden0, hidden1], dim=-1)
    out = torch.matmul(hidden_full, w2.float()).to(dtype=x.dtype)
    communication_bytes = float(hidden_full.numel() * dtype_bytes)
    return out, communication_bytes


def _mlp_dual_optimized_local(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    dtype_bytes: int,
) -> tuple[torch.Tensor, float]:
    w1_0, w1_1 = torch.chunk(w1, 2, dim=1)
    w2_0, w2_1 = torch.chunk(w2, 2, dim=0)

    h0 = torch.nn.functional.gelu(_matmul_fp32(x, w1_0, out_dtype=torch.float32), approximate="tanh")
    h1 = torch.nn.functional.gelu(_matmul_fp32(x, w1_1, out_dtype=torch.float32), approximate="tanh")
    out0 = torch.matmul(h0, w2_0.float())
    out1 = torch.matmul(h1, w2_1.float())
    out = (out0 + out1).to(dtype=x.dtype)

    communication_bytes = float(out.numel() * dtype_bytes)
    return out, communication_bytes


def _rmsnorm_single(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_f32 = x.float()
    inv_rms = torch.rsqrt(torch.mean(x_f32 * x_f32, dim=-1, keepdim=True) + eps)
    out = x_f32 * inv_rms * gamma.float()
    return out.to(dtype=x.dtype)


def _rmsnorm_dual_naive_local(
    x: torch.Tensor,
    gamma: torch.Tensor,
    dtype_bytes: int,
) -> tuple[torch.Tensor, float]:
    out = _rmsnorm_single(x, gamma)
    communication_bytes = float(x.numel() * dtype_bytes)
    return out, communication_bytes


def _rmsnorm_dual_optimized_local(
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
    return _matmul_fp32(x, w_out, out_dtype=x.dtype)


def _out_proj_dual_naive_local(
    x: torch.Tensor,
    w_out: torch.Tensor,
    dtype_bytes: int,
) -> tuple[torch.Tensor, float]:
    out = _out_proj_single(x, w_out)
    communication_bytes = float(2 * x.numel() * dtype_bytes)
    return out, communication_bytes


def _out_proj_dual_optimized_local(
    x: torch.Tensor,
    w_out: torch.Tensor,
    dtype_bytes: int,
) -> tuple[torch.Tensor, float]:
    x0, x1 = torch.chunk(x, 2, dim=-1)
    w0, w1 = torch.chunk(w_out, 2, dim=0)
    out0 = _matmul_fp32(x0, w0, out_dtype=torch.float32)
    out1 = _matmul_fp32(x1, w1, out_dtype=torch.float32)
    out = (out0 + out1).to(dtype=x.dtype)
    communication_bytes = float(out.numel() * dtype_bytes)
    return out, communication_bytes


def _all_reduce_payload_bytes(message_bytes: int, world_size: int) -> float:
    if world_size <= 1:
        return 0.0
    return float(message_bytes) * (2.0 * (world_size - 1) / world_size)


def _all_gather_payload_bytes(local_message_bytes: int, world_size: int) -> float:
    if world_size <= 1:
        return 0.0
    return float(local_message_bytes) * float(world_size - 1)


def _collective_all_reduce_(
    tensor: torch.Tensor,
    *,
    op_name: str,
    metrics: CommMetrics,
    ctx: DistributedContext,
    device: Any,
    dtype_bytes: int,
) -> None:
    if not ctx.enabled:
        return
    import torch.distributed as dist

    if op_name == "sum":
        reduce_op = dist.ReduceOp.SUM
    elif op_name == "max":
        reduce_op = dist.ReduceOp.MAX
    else:
        raise ValueError(f"Unsupported all-reduce op: {op_name}")

    start = time.perf_counter()
    dist.all_reduce(tensor, op=reduce_op)
    mark_step_if_needed(device)
    sync_if_needed(device)
    elapsed = time.perf_counter() - start

    payload = _all_reduce_payload_bytes(
        message_bytes=int(tensor.numel()) * int(dtype_bytes),
        world_size=ctx.world_size,
    )
    metrics.add("all_reduce", payload, elapsed)


def _collective_all_gather(
    local_tensor: torch.Tensor,
    *,
    gather_dim: int,
    metrics: CommMetrics,
    ctx: DistributedContext,
    device: Any,
    dtype_bytes: int,
) -> torch.Tensor:
    if not ctx.enabled:
        return local_tensor
    import torch.distributed as dist

    gathered = [torch.zeros_like(local_tensor) for _ in range(ctx.world_size)]
    start = time.perf_counter()
    dist.all_gather(gathered, local_tensor)
    mark_step_if_needed(device)
    sync_if_needed(device)
    elapsed = time.perf_counter() - start

    payload = _all_gather_payload_bytes(
        local_message_bytes=int(local_tensor.numel()) * int(dtype_bytes),
        world_size=ctx.world_size,
    )
    metrics.add("all_gather", payload, elapsed)
    return torch.cat(gathered, dim=gather_dim)


def _collective_broadcast_(
    tensor: torch.Tensor,
    *,
    src: int,
    metrics: CommMetrics,
    ctx: DistributedContext,
    device: Any,
    dtype_bytes: int,
) -> None:
    if not ctx.enabled:
        return
    import torch.distributed as dist

    start = time.perf_counter()
    dist.broadcast(tensor, src=src)
    mark_step_if_needed(device)
    sync_if_needed(device)
    elapsed = time.perf_counter() - start

    payload = _all_gather_payload_bytes(
        local_message_bytes=int(tensor.numel()) * int(dtype_bytes),
        world_size=ctx.world_size,
    )
    metrics.add("broadcast", payload, elapsed)


def _distributed_max_scalar(value: float, *, device: Any, ctx: DistributedContext) -> float:
    if not ctx.enabled:
        return float(value)
    import torch.distributed as dist

    scalar = torch.tensor([float(value)], dtype=torch.float32, device=device)
    dist.all_reduce(scalar, op=dist.ReduceOp.MAX)
    mark_step_if_needed(device)
    sync_if_needed(device)
    return float(scalar.detach().to("cpu").item())


def _seq_partition_start(tensor: torch.Tensor, rank: int) -> int:
    chunks = torch.chunk(tensor, 2, dim=-2)
    if rank == 0:
        return 0
    return int(chunks[0].size(-2))


def _qkv_dual_dist(
    x: torch.Tensor,
    w_qkv: torch.Tensor,
    *,
    optimized: bool,
    dtype_bytes: int,
    ctx: DistributedContext,
    device: Any,
) -> tuple[torch.Tensor, CommMetrics]:
    _ = device
    metrics = CommMetrics()
    w_parts = torch.chunk(w_qkv, 2, dim=1)
    y_local = _matmul_fp32(x, w_parts[ctx.rank], out_dtype=x.dtype)
    if optimized:
        out = _collective_all_gather(
            y_local,
            gather_dim=-1,
            metrics=metrics,
            ctx=ctx,
            device=device,
            dtype_bytes=dtype_bytes,
        )
        return out, metrics

    first = _collective_all_gather(
        y_local,
        gather_dim=-1,
        metrics=metrics,
        ctx=ctx,
        device=device,
        dtype_bytes=dtype_bytes,
    )
    reshard = torch.chunk(first, 2, dim=-1)[ctx.rank]
    second = _collective_all_gather(
        reshard,
        gather_dim=-1,
        metrics=metrics,
        ctx=ctx,
        device=device,
        dtype_bytes=dtype_bytes,
    )
    return second, metrics


def _attention_dual_dist_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    dtype_bytes: int,
    ctx: DistributedContext,
    device: Any,
) -> tuple[torch.Tensor, CommMetrics]:
    metrics = CommMetrics()
    k_local = torch.chunk(k, 2, dim=-2)[ctx.rank]
    v_local = torch.chunk(v, 2, dim=-2)[ctx.rank]
    k_start = _seq_partition_start(k, ctx.rank)

    logits_local = _softmax_logits(q.float(), k_local.float(), causal=causal, k_start=k_start)
    logits_full = _collective_all_gather(
        logits_local,
        gather_dim=-1,
        metrics=metrics,
        ctx=ctx,
        device=device,
        dtype_bytes=dtype_bytes,
    )
    v_full = _collective_all_gather(
        v_local.float(),
        gather_dim=-2,
        metrics=metrics,
        ctx=ctx,
        device=device,
        dtype_bytes=dtype_bytes,
    )

    probs = torch.softmax(logits_full, dim=-1)
    out = torch.matmul(probs, v_full)
    return out.to(dtype=q.dtype), metrics


def _attention_dual_dist_optimized(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    dtype_bytes: int,
    ctx: DistributedContext,
    device: Any,
) -> tuple[torch.Tensor, CommMetrics]:
    metrics = CommMetrics()
    k_local = torch.chunk(k, 2, dim=-2)[ctx.rank]
    v_local = torch.chunk(v, 2, dim=-2)[ctx.rank]
    k_start = _seq_partition_start(k, ctx.rank)

    logits_local = _softmax_logits(q.float(), k_local.float(), causal=causal, k_start=k_start)
    m_local = torch.max(logits_local, dim=-1).values

    m_global = m_local.clone()
    _collective_all_reduce_(
        m_global,
        op_name="max",
        metrics=metrics,
        ctx=ctx,
        device=device,
        dtype_bytes=dtype_bytes,
    )

    stable = logits_local - m_global.unsqueeze(-1)
    stable = torch.where(torch.isfinite(stable), stable, torch.full_like(stable, -1e9))
    p_local = torch.exp(stable)

    l_local = p_local.sum(dim=-1)
    o_local = torch.matmul(p_local, v_local.float())

    l_global = l_local.clone()
    _collective_all_reduce_(
        l_global,
        op_name="sum",
        metrics=metrics,
        ctx=ctx,
        device=device,
        dtype_bytes=dtype_bytes,
    )

    o_global = o_local.clone()
    _collective_all_reduce_(
        o_global,
        op_name="sum",
        metrics=metrics,
        ctx=ctx,
        device=device,
        dtype_bytes=dtype_bytes,
    )

    out = o_global / l_global.unsqueeze(-1).clamp_min(1e-9)
    return out.to(dtype=q.dtype), metrics


def _mlp_dual_dist(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    optimized: bool,
    dtype_bytes: int,
    ctx: DistributedContext,
    device: Any,
) -> tuple[torch.Tensor, CommMetrics]:
    metrics = CommMetrics()
    w1_local = torch.chunk(w1, 2, dim=1)[ctx.rank]
    hidden_local = torch.nn.functional.gelu(_matmul_fp32(x, w1_local, out_dtype=torch.float32), approximate="tanh")

    if optimized:
        w2_local = torch.chunk(w2, 2, dim=0)[ctx.rank]
        out_local = torch.matmul(hidden_local, w2_local.float())
        _collective_all_reduce_(
            out_local,
            op_name="sum",
            metrics=metrics,
            ctx=ctx,
            device=device,
            dtype_bytes=dtype_bytes,
        )
        return out_local.to(dtype=x.dtype), metrics

    hidden_full = _collective_all_gather(
        hidden_local,
        gather_dim=-1,
        metrics=metrics,
        ctx=ctx,
        device=device,
        dtype_bytes=dtype_bytes,
    )
    out = torch.matmul(hidden_full, w2.float())
    return out.to(dtype=x.dtype), metrics


def _rmsnorm_dual_dist(
    x: torch.Tensor,
    gamma: torch.Tensor,
    *,
    optimized: bool,
    dtype_bytes: int,
    ctx: DistributedContext,
    device: Any,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, CommMetrics]:
    metrics = CommMetrics()
    if not optimized:
        x_local = torch.chunk(x.float(), 2, dim=-1)[ctx.rank]
        x_full = _collective_all_gather(
            x_local,
            gather_dim=-1,
            metrics=metrics,
            ctx=ctx,
            device=device,
            dtype_bytes=dtype_bytes,
        )
        inv_rms = torch.rsqrt(torch.mean(x_full * x_full, dim=-1, keepdim=True) + eps)
        out = x_full * inv_rms * gamma.float()
        return out.to(dtype=x.dtype), metrics

    x_local = torch.chunk(x.float(), 2, dim=-1)[ctx.rank]
    g_local = torch.chunk(gamma.float(), 2, dim=-1)[ctx.rank]

    local_sum = torch.sum(x_local * x_local, dim=-1, keepdim=True)
    _collective_all_reduce_(
        local_sum,
        op_name="sum",
        metrics=metrics,
        ctx=ctx,
        device=device,
        dtype_bytes=dtype_bytes,
    )
    inv_rms = torch.rsqrt(local_sum / float(x.size(-1)) + eps)
    out_local = x_local * inv_rms * g_local
    out_full = _collective_all_gather(
        out_local,
        gather_dim=-1,
        metrics=metrics,
        ctx=ctx,
        device=device,
        dtype_bytes=dtype_bytes,
    )
    return out_full.to(dtype=x.dtype), metrics


def _out_proj_dual_dist(
    x: torch.Tensor,
    w_out: torch.Tensor,
    *,
    optimized: bool,
    dtype_bytes: int,
    ctx: DistributedContext,
    device: Any,
) -> tuple[torch.Tensor, CommMetrics]:
    metrics = CommMetrics()
    x_local = torch.chunk(x, 2, dim=-1)[ctx.rank]
    w_local = torch.chunk(w_out, 2, dim=0)[ctx.rank]

    if optimized:
        out_local = _matmul_fp32(x_local, w_local, out_dtype=torch.float32)
        _collective_all_reduce_(
            out_local,
            op_name="sum",
            metrics=metrics,
            ctx=ctx,
            device=device,
            dtype_bytes=dtype_bytes,
        )
        return out_local.to(dtype=x.dtype), metrics

    x_full = _collective_all_gather(
        x_local,
        gather_dim=-1,
        metrics=metrics,
        ctx=ctx,
        device=device,
        dtype_bytes=dtype_bytes,
    )
    w_full = _collective_all_gather(
        w_local,
        gather_dim=0,
        metrics=metrics,
        ctx=ctx,
        device=device,
        dtype_bytes=dtype_bytes,
    )
    out = _matmul_fp32(x_full, w_full, out_dtype=x.dtype)
    return out, metrics


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
) -> BenchmarkResult:
    for _ in range(warmup_iters):
        _ = fn()
        mark_step_if_needed(device)
    sync_if_needed(device)

    latencies: list[float] = []
    compute_s: list[float] = []
    communication_s: list[float] = []
    overlap_pct: list[float] = []
    communication_bytes: list[float] = []
    achieved_link_gbps: list[float] = []

    output: torch.Tensor | None = None

    for _ in range(measure_iters):
        start = time.perf_counter()
        output, comm = fn()
        mark_step_if_needed(device)
        sync_if_needed(device)
        end = time.perf_counter()

        total_s = end - start
        comm_s = float(max(comm.time_s_total, 0.0))
        compute_time_s = max(total_s - comm_s, 0.0)
        hidden_overlap_s = max(0.0, compute_time_s + comm_s - total_s)
        overlap_ratio = (hidden_overlap_s / comm_s * 100.0) if comm_s > 0 else 0.0
        achieved_bw = (comm.bytes_total / comm_s / 1e9) if comm_s > 0 else 0.0

        latencies.append(total_s)
        compute_s.append(compute_time_s)
        communication_s.append(comm_s)
        overlap_pct.append(float(overlap_ratio))
        communication_bytes.append(float(comm.bytes_total))
        achieved_link_gbps.append(float(achieved_bw))

    assert output is not None
    return BenchmarkResult(
        output=output,
        latencies_s=latencies,
        compute_s=compute_s,
        communication_s=communication_s,
        overlap_pct=overlap_pct,
        communication_bytes=communication_bytes,
        achieved_link_gbps=achieved_link_gbps,
    )


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
    dist_ctx: DistributedContext,
    device: Any,
) -> Any:
    if setup == "single_die":
        if kernel == "qkv_proj":
            x = tensors["x"]
            w_qkv = tensors["w_qkv"]
            return lambda: (_qkv_single(x, w_qkv), CommMetrics())
        if kernel == "attention":
            q, k, v = tensors["q"], tensors["k"], tensors["v"]
            return lambda: (_attention_single(q, k, v, causal_attention), CommMetrics())
        if kernel == "mlp":
            x, w1, w2 = tensors["x"], tensors["w1"], tensors["w2"]
            return lambda: (_mlp_single(x, w1, w2), CommMetrics())
        if kernel == "rmsnorm":
            x, gamma = tensors["x"], tensors["gamma"]
            return lambda: (_rmsnorm_single(x, gamma), CommMetrics())
        if kernel == "out_proj":
            x, w_out = tensors["x"], tensors["w_out"]
            return lambda: (_out_proj_single(x, w_out), CommMetrics())
        raise ValueError(f"Unsupported kernel: {kernel}")

    if dist_ctx.enabled:
        if dist_ctx.world_size != 2:
            raise RuntimeError(
                "Real dual-die execution requires world_size=2. "
                f"Current world_size={dist_ctx.world_size}."
            )
        if kernel == "qkv_proj":
            x = tensors["x"]
            w_qkv = tensors["w_qkv"]
            if setup == "dual_die_naive":
                return lambda: _qkv_dual_dist(
                    x,
                    w_qkv,
                    optimized=False,
                    dtype_bytes=dtype_bytes,
                    ctx=dist_ctx,
                    device=device,
                )
            return lambda: _qkv_dual_dist(
                x,
                w_qkv,
                optimized=True,
                dtype_bytes=dtype_bytes,
                ctx=dist_ctx,
                device=device,
            )

        if kernel == "attention":
            q, k, v = tensors["q"], tensors["k"], tensors["v"]
            if setup == "dual_die_naive":
                return lambda: _attention_dual_dist_naive(
                    q,
                    k,
                    v,
                    causal=causal_attention,
                    dtype_bytes=dtype_bytes,
                    ctx=dist_ctx,
                    device=device,
                )
            return lambda: _attention_dual_dist_optimized(
                q,
                k,
                v,
                causal=causal_attention,
                dtype_bytes=dtype_bytes,
                ctx=dist_ctx,
                device=device,
            )

        if kernel == "mlp":
            x, w1, w2 = tensors["x"], tensors["w1"], tensors["w2"]
            return lambda: _mlp_dual_dist(
                x,
                w1,
                w2,
                optimized=(setup == "dual_die_optimized"),
                dtype_bytes=dtype_bytes,
                ctx=dist_ctx,
                device=device,
            )

        if kernel == "rmsnorm":
            x, gamma = tensors["x"], tensors["gamma"]
            return lambda: _rmsnorm_dual_dist(
                x,
                gamma,
                optimized=(setup == "dual_die_optimized"),
                dtype_bytes=dtype_bytes,
                ctx=dist_ctx,
                device=device,
            )

        if kernel == "out_proj":
            x, w_out = tensors["x"], tensors["w_out"]
            return lambda: _out_proj_dual_dist(
                x,
                w_out,
                optimized=(setup == "dual_die_optimized"),
                dtype_bytes=dtype_bytes,
                ctx=dist_ctx,
                device=device,
            )

        raise ValueError(f"Unsupported kernel/setup: {kernel}/{setup}")

    if kernel == "qkv_proj":
        x = tensors["x"]
        w_qkv = tensors["w_qkv"]
        if setup == "dual_die_naive":
            def run_qkv_dual_naive_local() -> tuple[torch.Tensor, CommMetrics]:
                out, comm_bytes = _qkv_dual_naive_local(x, w_qkv, dtype_bytes)
                return out, CommMetrics(bytes_total=comm_bytes)

            return run_qkv_dual_naive_local

        def run_qkv_dual_opt_local() -> tuple[torch.Tensor, CommMetrics]:
            out, comm_bytes = _qkv_dual_optimized_local(x, w_qkv, dtype_bytes)
            return out, CommMetrics(bytes_total=comm_bytes)

        return run_qkv_dual_opt_local

    if kernel == "attention":
        q, k, v = tensors["q"], tensors["k"], tensors["v"]
        if setup == "dual_die_naive":
            def run_attention_dual_naive_local() -> tuple[torch.Tensor, CommMetrics]:
                out, comm_bytes = _attention_dual_naive_local(q, k, v, causal_attention, dtype_bytes)
                return out, CommMetrics(bytes_total=comm_bytes)

            return run_attention_dual_naive_local

        def run_attention_dual_opt_local() -> tuple[torch.Tensor, CommMetrics]:
            out, comm_bytes = _attention_dual_optimized_local(q, k, v, causal_attention, dtype_bytes)
            return out, CommMetrics(bytes_total=comm_bytes)

        return run_attention_dual_opt_local

    if kernel == "mlp":
        x, w1, w2 = tensors["x"], tensors["w1"], tensors["w2"]
        if setup == "dual_die_naive":
            def run_mlp_dual_naive_local() -> tuple[torch.Tensor, CommMetrics]:
                out, comm_bytes = _mlp_dual_naive_local(x, w1, w2, dtype_bytes)
                return out, CommMetrics(bytes_total=comm_bytes)

            return run_mlp_dual_naive_local

        def run_mlp_dual_opt_local() -> tuple[torch.Tensor, CommMetrics]:
            out, comm_bytes = _mlp_dual_optimized_local(x, w1, w2, dtype_bytes)
            return out, CommMetrics(bytes_total=comm_bytes)

        return run_mlp_dual_opt_local

    if kernel == "rmsnorm":
        x, gamma = tensors["x"], tensors["gamma"]
        if setup == "dual_die_naive":
            def run_rmsnorm_dual_naive_local() -> tuple[torch.Tensor, CommMetrics]:
                out, comm_bytes = _rmsnorm_dual_naive_local(x, gamma, dtype_bytes)
                return out, CommMetrics(bytes_total=comm_bytes)

            return run_rmsnorm_dual_naive_local

        def run_rmsnorm_dual_opt_local() -> tuple[torch.Tensor, CommMetrics]:
            out, comm_bytes = _rmsnorm_dual_optimized_local(x, gamma, dtype_bytes)
            return out, CommMetrics(bytes_total=comm_bytes)

        return run_rmsnorm_dual_opt_local

    if kernel == "out_proj":
        x, w_out = tensors["x"], tensors["w_out"]
        if setup == "dual_die_naive":
            def run_out_proj_dual_naive_local() -> tuple[torch.Tensor, CommMetrics]:
                out, comm_bytes = _out_proj_dual_naive_local(x, w_out, dtype_bytes)
                return out, CommMetrics(bytes_total=comm_bytes)

            return run_out_proj_dual_naive_local

        def run_out_proj_dual_opt_local() -> tuple[torch.Tensor, CommMetrics]:
            out, comm_bytes = _out_proj_dual_optimized_local(x, w_out, dtype_bytes)
            return out, CommMetrics(bytes_total=comm_bytes)

        return run_out_proj_dual_opt_local

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
        "| Kernel | Setup | Avg p50 latency (ms) | Avg p50 compute (ms) | Avg p50 comm (ms) | Avg link util (%) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for (kernel, setup), rows in sorted(by_key.items()):
        p50 = float(np.mean([float(item["latency_ms_p50"]) for item in rows]))
        comp = float(np.mean([float(item["compute_ms_p50"]) for item in rows]))
        comm = float(np.mean([float(item["communication_ms_p50"]) for item in rows]))
        util = float(np.mean([float(item["link_utilization_pct_p50"]) for item in rows]))
        lines.append(f"| {kernel} | {setup} | {p50:.4f} | {comp:.4f} | {comm:.4f} | {util:.2f} |")

    summary_path = run_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def _write_fabric_calibration(run_dir: Path, calibration: dict[str, Any]) -> tuple[Path, Path]:
    csv_path = run_dir / "fabric_calibration.csv"
    json_path = run_dir / "fabric_calibration.json"

    rows: list[dict[str, Any]] = []
    collectives = calibration.get("collectives", {})
    for name, payload in collectives.items():
        for entry in payload.get("entries", []):
            rows.append(
                {
                    "collective": name,
                    "message_bytes": int(entry["message_bytes"]),
                    "latency_ms_p50": float(entry["latency_ms_p50"]),
                    "latency_ms_p90": float(entry["latency_ms_p90"]),
                    "effective_gbps_p50": float(entry["effective_gbps_p50"]),
                    "effective_gbps_p90": float(entry["effective_gbps_p90"]),
                }
            )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "collective",
                "message_bytes",
                "latency_ms_p50",
                "latency_ms_p90",
                "effective_gbps_p50",
                "effective_gbps_p90",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    json_path.write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return csv_path, json_path


def _calibrate_ping_pong(
    *,
    device: Any,
    dtype: torch.dtype,
    dtype_bytes: int,
    message_sizes: list[int],
    warmup_iters: int,
    measure_iters: int,
    ctx: DistributedContext,
) -> list[dict[str, float]]:
    entries: list[dict[str, float]] = []
    for message_bytes in message_sizes:
        distributed_barrier(ctx)
        elems = max(1, int(math.ceil(float(message_bytes) / float(dtype_bytes))))
        payload = torch.zeros((elems,), dtype=dtype, device=device)

        for _ in range(warmup_iters):
            warmup_metrics = CommMetrics()
            if ctx.rank == 0:
                payload.fill_(1)
            _collective_broadcast_(
                payload,
                src=0,
                metrics=warmup_metrics,
                ctx=ctx,
                device=device,
                dtype_bytes=dtype_bytes,
            )
            if ctx.rank == 1:
                payload.fill_(2)
            _collective_broadcast_(
                payload,
                src=1,
                metrics=warmup_metrics,
                ctx=ctx,
                device=device,
                dtype_bytes=dtype_bytes,
            )

        latencies: list[float] = []
        gbps: list[float] = []
        for _ in range(measure_iters):
            metrics = CommMetrics()
            start = time.perf_counter()
            if ctx.rank == 0:
                payload.fill_(3)
            _collective_broadcast_(
                payload,
                src=0,
                metrics=metrics,
                ctx=ctx,
                device=device,
                dtype_bytes=dtype_bytes,
            )
            if ctx.rank == 1:
                payload.fill_(4)
            _collective_broadcast_(
                payload,
                src=1,
                metrics=metrics,
                ctx=ctx,
                device=device,
                dtype_bytes=dtype_bytes,
            )
            end = time.perf_counter()
            latencies.append(end - start)
            bw = (metrics.bytes_total / metrics.time_s_total / 1e9) if metrics.time_s_total > 0 else 0.0
            gbps.append(float(bw))

        entries.append(
            {
                "message_bytes": float(payload.numel() * dtype_bytes),
                "latency_ms_p50": _percentile_ms(latencies, 50),
                "latency_ms_p90": _percentile_ms(latencies, 90),
                "effective_gbps_p50": _percentile(gbps, 50),
                "effective_gbps_p90": _percentile(gbps, 90),
            }
        )
    return entries


def _calibrate_all_reduce(
    *,
    device: Any,
    dtype: torch.dtype,
    dtype_bytes: int,
    message_sizes: list[int],
    warmup_iters: int,
    measure_iters: int,
    ctx: DistributedContext,
) -> list[dict[str, float]]:
    entries: list[dict[str, float]] = []
    for message_bytes in message_sizes:
        distributed_barrier(ctx)
        elems = max(1, int(math.ceil(float(message_bytes) / float(dtype_bytes))))
        tensor = torch.randn((elems,), dtype=dtype, device=device)

        for _ in range(warmup_iters):
            warmup_metrics = CommMetrics()
            _collective_all_reduce_(
                tensor,
                op_name="sum",
                metrics=warmup_metrics,
                ctx=ctx,
                device=device,
                dtype_bytes=dtype_bytes,
            )

        latencies: list[float] = []
        gbps: list[float] = []
        for _ in range(measure_iters):
            metrics = CommMetrics()
            start = time.perf_counter()
            _collective_all_reduce_(
                tensor,
                op_name="sum",
                metrics=metrics,
                ctx=ctx,
                device=device,
                dtype_bytes=dtype_bytes,
            )
            end = time.perf_counter()
            latencies.append(end - start)
            bw = (metrics.bytes_total / metrics.time_s_total / 1e9) if metrics.time_s_total > 0 else 0.0
            gbps.append(float(bw))

        entries.append(
            {
                "message_bytes": float(tensor.numel() * dtype_bytes),
                "latency_ms_p50": _percentile_ms(latencies, 50),
                "latency_ms_p90": _percentile_ms(latencies, 90),
                "effective_gbps_p50": _percentile(gbps, 50),
                "effective_gbps_p90": _percentile(gbps, 90),
            }
        )
    return entries


def _calibrate_all_gather(
    *,
    device: Any,
    dtype: torch.dtype,
    dtype_bytes: int,
    message_sizes: list[int],
    warmup_iters: int,
    measure_iters: int,
    ctx: DistributedContext,
) -> list[dict[str, float]]:
    entries: list[dict[str, float]] = []
    for message_bytes in message_sizes:
        distributed_barrier(ctx)
        elems = max(1, int(math.ceil(float(message_bytes) / float(dtype_bytes))))
        local = torch.randn((elems,), dtype=dtype, device=device)

        for _ in range(warmup_iters):
            warmup_metrics = CommMetrics()
            _ = _collective_all_gather(
                local,
                gather_dim=0,
                metrics=warmup_metrics,
                ctx=ctx,
                device=device,
                dtype_bytes=dtype_bytes,
            )

        latencies: list[float] = []
        gbps: list[float] = []
        for _ in range(measure_iters):
            metrics = CommMetrics()
            start = time.perf_counter()
            _ = _collective_all_gather(
                local,
                gather_dim=0,
                metrics=metrics,
                ctx=ctx,
                device=device,
                dtype_bytes=dtype_bytes,
            )
            end = time.perf_counter()
            latencies.append(end - start)
            bw = (metrics.bytes_total / metrics.time_s_total / 1e9) if metrics.time_s_total > 0 else 0.0
            gbps.append(float(bw))

        entries.append(
            {
                "message_bytes": float(local.numel() * dtype_bytes),
                "latency_ms_p50": _percentile_ms(latencies, 50),
                "latency_ms_p90": _percentile_ms(latencies, 90),
                "effective_gbps_p50": _percentile(gbps, 50),
                "effective_gbps_p90": _percentile(gbps, 90),
            }
        )
    return entries


def run_fabric_calibration(
    *,
    device: Any,
    dtype: torch.dtype,
    dtype_name: DTypeName,
    dtype_bytes: int,
    config: KernelStudyConfig,
    dist_ctx: DistributedContext,
    run_dir: Path | None,
) -> dict[str, Any]:
    if not dist_ctx.enabled:
        return {
            "enabled": False,
            "dtype": dtype_name,
            "world_size": dist_ctx.world_size,
            "collectives": {},
            "peak_gbps": 0.0,
        }

    message_sizes = sorted(set(int(x) for x in config.fabric_message_sizes))
    ping_pong_entries = _calibrate_ping_pong(
        device=device,
        dtype=dtype,
        dtype_bytes=dtype_bytes,
        message_sizes=message_sizes,
        warmup_iters=config.fabric_warmup_iters,
        measure_iters=config.fabric_measure_iters,
        ctx=dist_ctx,
    )
    all_reduce_entries = _calibrate_all_reduce(
        device=device,
        dtype=dtype,
        dtype_bytes=dtype_bytes,
        message_sizes=message_sizes,
        warmup_iters=config.fabric_warmup_iters,
        measure_iters=config.fabric_measure_iters,
        ctx=dist_ctx,
    )
    all_gather_entries = _calibrate_all_gather(
        device=device,
        dtype=dtype,
        dtype_bytes=dtype_bytes,
        message_sizes=message_sizes,
        warmup_iters=config.fabric_warmup_iters,
        measure_iters=config.fabric_measure_iters,
        ctx=dist_ctx,
    )

    collectives = {
        "ping_pong": {
            "entries": ping_pong_entries,
            "max_gbps": max([float(x["effective_gbps_p50"]) for x in ping_pong_entries], default=0.0),
        },
        "all_reduce": {
            "entries": all_reduce_entries,
            "max_gbps": max([float(x["effective_gbps_p50"]) for x in all_reduce_entries], default=0.0),
        },
        "all_gather": {
            "entries": all_gather_entries,
            "max_gbps": max([float(x["effective_gbps_p50"]) for x in all_gather_entries], default=0.0),
        },
    }
    peak = max(float(payload["max_gbps"]) for payload in collectives.values())
    calibration = {
        "enabled": True,
        "dtype": dtype_name,
        "world_size": dist_ctx.world_size,
        "fabric_message_sizes": message_sizes,
        "collectives": collectives,
        "peak_gbps": round(float(peak), 6),
        "timestamp": _timestamp_utc(),
    }

    if dist_ctx.is_primary and run_dir is not None:
        _write_fabric_calibration(run_dir, calibration)
    distributed_barrier(dist_ctx)
    return calibration


def run_kernel_study(
    config: KernelStudyConfig,
    config_path: Path,
    output_dir: Path,
    device_override: str | None = None,
    setups_override: list[SetupName] | None = None,
    distributed_override: bool | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    device_name = str(device_override or config.device)
    dtype = _dtype_from_name(config.dtype)
    dtype_bytes = _dtype_bytes(dtype)
    distributed_requested = bool(config.distributed if distributed_override is None else distributed_override)
    abs_tol, rel_tol = _correctness_thresholds(config)

    dist_ctx = init_distributed_context(
        device_name=device_name,
        enable_distributed=distributed_requested,
        expected_world_size=(config.dual_world_size if distributed_requested else None),
    )

    run_dir: Path
    if dist_ctx.is_primary:
        run_dir = create_run_dir(output_dir)
    else:
        run_dir = output_dir / "_distributed_non_primary"

    try:
        device = resolve_device(device_name)
        selected_setups = setups_override or config.setups
        for setup in selected_setups:
            if setup not in ALL_SETUPS:
                raise ValueError(f"Unsupported setup override: {setup}")

        if distributed_requested and not dist_ctx.enabled:
            raise RuntimeError("Distributed execution was requested but distributed context is not enabled.")

        if dist_ctx.enabled:
            distributed_barrier(dist_ctx)

        fabric_summary = {
            "enabled": False,
            "peak_gbps": 0.0,
            "collectives": {},
        }
        if config.enable_fabric_calibration:
            fabric_summary = run_fabric_calibration(
                device=device,
                dtype=dtype,
                dtype_name=config.dtype,
                dtype_bytes=dtype_bytes,
                config=config,
                dist_ctx=dist_ctx,
                run_dir=run_dir if dist_ctx.is_primary else None,
            )

        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        records: list[dict[str, Any]] = []

        for shape_idx, shape in enumerate(config.shapes):
            for kernel_idx, kernel in enumerate(config.kernels):
                torch.manual_seed(config.seed + shape_idx * 1000 + kernel_idx)
                tensors = _init_kernel_tensors(kernel, shape, dtype=dtype, device=device)

                baseline_runner = _build_runner(
                    kernel=kernel,
                    setup="single_die",
                    tensors=tensors,
                    causal_attention=config.causal_attention,
                    dtype_bytes=dtype_bytes,
                    dist_ctx=dist_ctx,
                    device=device,
                )
                baseline_result = _benchmark_fn(
                    baseline_runner,
                    warmup_iters=1,
                    measure_iters=1,
                    device=device,
                )
                baseline_cpu = baseline_result.output.detach().to("cpu", dtype=torch.float32)

                for setup in selected_setups:
                    runner = _build_runner(
                        kernel=kernel,
                        setup=setup,
                        tensors=tensors,
                        causal_attention=config.causal_attention,
                        dtype_bytes=dtype_bytes,
                        dist_ctx=dist_ctx,
                        device=device,
                    )
                    result = _benchmark_fn(
                        runner,
                        warmup_iters=config.warmup_iters,
                        measure_iters=config.measure_iters,
                        device=device,
                    )

                    output_cpu = result.output.detach().to("cpu", dtype=torch.float32)
                    max_abs_err = float((baseline_cpu - output_cpu).abs().max().item())
                    max_rel_err = _max_relative_error(
                        baseline_cpu,
                        output_cpu,
                        eps=_relative_error_eps(config.dtype),
                    )

                    max_abs_err = _distributed_max_scalar(max_abs_err, device=device, ctx=dist_ctx)
                    max_rel_err = _distributed_max_scalar(max_rel_err, device=device, ctx=dist_ctx)

                    if config.enforce_correctness and setup != "single_die":
                        if max_abs_err > abs_tol or max_rel_err > rel_tol:
                            raise RuntimeError(
                                f"Correctness gate failed for {kernel}/{setup}: "
                                f"max_abs_err={max_abs_err:.6g} (tol={abs_tol:.6g}), "
                                f"max_rel_err={max_rel_err:.6g} (tol={rel_tol:.6g})"
                            )

                    p50_ms = _percentile_ms(result.latencies_s, 50)
                    p90_ms = _percentile_ms(result.latencies_s, 90)
                    compute_ms_p50 = _percentile_ms(result.compute_s, 50)
                    comm_ms_p50 = _percentile_ms(result.communication_s, 50)
                    overlap_pct_p50 = _percentile(result.overlap_pct, 50)
                    throughput = (shape.batch * shape.seq_len) / (p50_ms / 1000.0) if p50_ms > 0 else 0.0

                    avg_comm_bytes = float(np.mean(result.communication_bytes)) if result.communication_bytes else 0.0
                    achieved_link_gbps_p50 = _percentile(result.achieved_link_gbps, 50)
                    fabric_peak = float(fabric_summary.get("peak_gbps", 0.0))
                    link_util_pct = (achieved_link_gbps_p50 / fabric_peak * 100.0) if fabric_peak > 0 else 0.0

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
                        "compute_ms_p50": round(float(compute_ms_p50), 6),
                        "communication_ms_p50": round(float(comm_ms_p50), 6),
                        "overlap_pct_p50": round(float(overlap_pct_p50), 6),
                        "throughput_tokens_per_s": round(float(throughput), 6),
                        "estimated_flops": round(float(estimated_flops), 2),
                        "estimated_hbm_bytes": round(float(estimated_hbm_bytes), 2),
                        "arithmetic_intensity": round(float(arithmetic_intensity), 6),
                        "communication_bytes": round(float(avg_comm_bytes), 2),
                        "communication_pct_of_hbm": round(float(comm_pct), 6),
                        "achieved_link_gbps_p50": round(float(achieved_link_gbps_p50), 6),
                        "link_utilization_pct_p50": round(float(link_util_pct), 6),
                        "fabric_peak_gbps": round(float(fabric_peak), 6),
                        "max_abs_err": round(float(max_abs_err), 8),
                        "max_rel_err": round(float(max_rel_err), 8),
                    }
                    missing = [col for col in REQUIRED_COLUMNS if col not in record]
                    if missing:
                        raise RuntimeError(f"Metric record missing required columns: {missing}")
                    if dist_ctx.is_primary:
                        records.append(record)

        if dist_ctx.is_primary:
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
                    "distributed_requested": distributed_requested,
                    "distributed_enabled": dist_ctx.enabled,
                    "distributed_backend": dist_ctx.backend,
                    "distributed_world_size": dist_ctx.world_size,
                    "correctness_abs_tol": abs_tol,
                    "correctness_rel_tol": rel_tol,
                    "fabric_peak_gbps": float(fabric_summary.get("peak_gbps", 0.0)),
                }
            )
            write_manifest(run_dir, manifest)

        distributed_barrier(dist_ctx)
        return run_dir, records if dist_ctx.is_primary else []
    finally:
        finalize_distributed_context(dist_ctx)


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
        dist_ctx=DistributedContext(enabled=False),
        device=device,
    )
    out, _ = runner()
    return out.detach().to("cpu", dtype=torch.float32)
