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
from io_aware_attention.experiments import kernel_study as ks
from io_aware_attention.runtime.trainium import (
    DistributedContext,
    distributed_barrier,
    finalize_distributed_context,
    gather_rank_strings,
    get_visible_core_mask,
    init_distributed_context,
    mark_step_if_needed,
    parse_visible_cores,
    resolve_device,
    sync_if_needed,
)

PhaseName = Literal["prefill", "decode"]
SetupName = Literal["single_die", "dual_die_tensor_optimized", "dual_die_request_sharded"]
DTypeName = Literal["bf16", "fp32"]
DeviceName = Literal["cpu", "trainium"]

ALL_SETUPS: tuple[SetupName, ...] = (
    "single_die",
    "dual_die_tensor_optimized",
    "dual_die_request_sharded",
)

KERNELS: tuple[str, ...] = ("qkv_proj", "attention", "mlp", "rmsnorm", "out_proj")

REQUIRED_COLUMNS = [
    "timestamp",
    "phase",
    "setup",
    "device",
    "dtype",
    "batch",
    "seq_len",
    "context_len",
    "decode_steps",
    "model_dim",
    "num_heads",
    "latency_ms_p50",
    "latency_ms_p90",
    "compute_ms_p50",
    "communication_ms_p50",
    "overlap_pct_p50",
    "throughput_tokens_per_s",
    "communication_bytes",
    "achieved_link_gbps_p50",
    "link_utilization_pct_p50",
    "fabric_peak_gbps",
    "kv_cache_bytes_per_rank",
    "max_abs_err",
    "max_rel_err",
]

KERNEL_PHASE_COLUMNS = [
    "timestamp",
    "phase",
    "setup",
    "kernel",
    "device",
    "dtype",
    "batch",
    "seq_len",
    "context_len",
    "decode_steps",
    "model_dim",
    "num_heads",
    "latency_ms_p50",
    "latency_ms_p90",
    "compute_ms_p50",
    "communication_ms_p50",
    "overlap_pct_p50",
    "communication_bytes",
    "achieved_link_gbps_p50",
    "link_utilization_pct_p50",
    "fabric_peak_gbps",
]


@dataclass(frozen=True)
class PrefillShape:
    batch: int
    seq_len: int
    model_dim: int
    num_heads: int
    mlp_ratio: int = 4
    setups: list[SetupName] | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PrefillShape":
        setups_raw = raw.get("setups")
        setups = [str(item) for item in setups_raw] if setups_raw is not None else None
        return cls(
            batch=int(raw["batch"]),
            seq_len=int(raw["seq_len"]),
            model_dim=int(raw["model_dim"]),
            num_heads=int(raw["num_heads"]),
            mlp_ratio=int(raw.get("mlp_ratio", 4)),
            setups=setups,  # type: ignore[arg-type]
        )

    def validate(self) -> None:
        if self.batch < 1 or self.seq_len < 1:
            raise ValueError(f"Invalid prefill shape: {self}")
        if self.model_dim < 2 or self.model_dim % self.num_heads != 0:
            raise ValueError(f"model_dim/num_heads mismatch: {self}")
        if self.model_dim % 2 != 0:
            raise ValueError(f"model_dim must be divisible by 2 for dual split: {self.model_dim}")


@dataclass(frozen=True)
class DecodeShape:
    concurrency: int
    context_len: int
    decode_steps: int
    model_dim: int
    num_heads: int
    mlp_ratio: int = 4
    setups: list[SetupName] | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DecodeShape":
        setups_raw = raw.get("setups")
        setups = [str(item) for item in setups_raw] if setups_raw is not None else None
        return cls(
            concurrency=int(raw["concurrency"]),
            context_len=int(raw["context_len"]),
            decode_steps=int(raw["decode_steps"]),
            model_dim=int(raw["model_dim"]),
            num_heads=int(raw["num_heads"]),
            mlp_ratio=int(raw.get("mlp_ratio", 4)),
            setups=setups,  # type: ignore[arg-type]
        )

    def validate(self) -> None:
        if self.concurrency < 1 or self.context_len < 1 or self.decode_steps < 1:
            raise ValueError(f"Invalid decode shape: {self}")
        if self.model_dim < 2 or self.model_dim % self.num_heads != 0:
            raise ValueError(f"model_dim/num_heads mismatch: {self}")
        if self.model_dim % 2 != 0:
            raise ValueError(f"model_dim must be divisible by 2 for dual split: {self.model_dim}")


@dataclass(frozen=True)
class PhaseStudyConfig:
    device: DeviceName
    dtype: DTypeName
    seed: int
    warmup_iters: int
    measure_iters: int
    distributed: bool
    dual_world_size: int
    setups: list[SetupName]
    prefill_shapes: list[PrefillShape]
    decode_shapes: list[DecodeShape]
    decode_slo_ms: list[float]
    enable_fabric_calibration: bool
    fabric_message_sizes: list[int]
    fabric_warmup_iters: int
    fabric_measure_iters: int
    enforce_correctness: bool
    correctness_abs_tol: float
    correctness_rel_tol: float
    tensor_attention_naive_threshold: int
    tensor_attention_tile_q: int
    tensor_attention_tile_k: int
    tensor_attention_reduce_group_k: int
    tensor_attention_pipelined_prefill: bool
    tensor_attention_pipelined_decode: bool

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PhaseStudyConfig":
        prefill_shapes = [PrefillShape.from_dict(item) for item in raw.get("prefill", [])]
        decode_shapes = [DecodeShape.from_dict(item) for item in raw.get("decode", [])]
        setups = [str(item) for item in raw.get("setups", list(ALL_SETUPS))]
        cfg = cls(
            device=str(raw.get("device", "cpu")),
            dtype=str(raw.get("dtype", "bf16")),
            seed=int(raw.get("seed", 0)),
            warmup_iters=int(raw.get("warmup_iters", 2)),
            measure_iters=int(raw.get("measure_iters", 8)),
            distributed=bool(raw.get("distributed", False)),
            dual_world_size=int(raw.get("dual_world_size", 2)),
            setups=setups,  # type: ignore[arg-type]
            prefill_shapes=prefill_shapes,
            decode_shapes=decode_shapes,
            decode_slo_ms=[float(x) for x in raw.get("decode_slo_ms", [2.0, 4.0, 8.0, 16.0])],
            enable_fabric_calibration=bool(raw.get("enable_fabric_calibration", True)),
            fabric_message_sizes=[
                int(item) for item in raw.get("fabric_message_sizes", ks.DEFAULT_FABRIC_MESSAGE_SIZES)
            ],
            fabric_warmup_iters=int(raw.get("fabric_warmup_iters", 2)),
            fabric_measure_iters=int(raw.get("fabric_measure_iters", 8)),
            enforce_correctness=bool(raw.get("enforce_correctness", True)),
            correctness_abs_tol=float(raw.get("correctness_abs_tol", 0.05)),
            correctness_rel_tol=float(raw.get("correctness_rel_tol", 0.1)),
            tensor_attention_naive_threshold=int(raw.get("tensor_attention_naive_threshold", 0)),
            tensor_attention_tile_q=int(raw.get("tensor_attention_tile_q", 64)),
            tensor_attention_tile_k=int(raw.get("tensor_attention_tile_k", 128)),
            tensor_attention_reduce_group_k=int(raw.get("tensor_attention_reduce_group_k", 1)),
            tensor_attention_pipelined_prefill=bool(raw.get("tensor_attention_pipelined_prefill", True)),
            tensor_attention_pipelined_decode=bool(raw.get("tensor_attention_pipelined_decode", False)),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.device not in {"cpu", "trainium"}:
            raise ValueError(f"Unsupported device: {self.device}")
        if self.dtype not in {"bf16", "fp32"}:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        if self.warmup_iters < 0 or self.measure_iters < 1:
            raise ValueError("warmup/measure iters must be >= 0 / >= 1")
        if self.fabric_warmup_iters < 0 or self.fabric_measure_iters < 1:
            raise ValueError("fabric calibration warmup/measure iters must be >= 0 / >= 1")
        if self.tensor_attention_naive_threshold < 0:
            raise ValueError("tensor_attention_naive_threshold must be >= 0")
        if (
            self.tensor_attention_tile_q < 1
            or self.tensor_attention_tile_k < 1
            or self.tensor_attention_reduce_group_k < 1
        ):
            raise ValueError(
                "tensor_attention_tile_q, tensor_attention_tile_k, and "
                "tensor_attention_reduce_group_k must be >= 1"
            )
        if not self.fabric_message_sizes:
            raise ValueError("fabric_message_sizes must not be empty")
        for size in self.fabric_message_sizes:
            if size < 1:
                raise ValueError(f"fabric message sizes must be >= 1 bytes, got {size}")
        if self.dual_world_size != 2:
            raise ValueError("Phase study currently supports dual_world_size=2 only")
        if not self.prefill_shapes and not self.decode_shapes:
            raise ValueError("At least one of prefill/decode sections must be configured")
        for setup in self.setups:
            if setup not in ALL_SETUPS:
                raise ValueError(f"Unsupported setup: {setup}")
        for shape in self.prefill_shapes:
            shape.validate()
            active_setups = shape.setups or self.setups
            for setup in active_setups:
                if setup not in ALL_SETUPS:
                    raise ValueError(f"Unsupported setup in prefill shape: {setup}")
                if setup not in self.setups:
                    raise ValueError(
                        f"Prefill shape setup '{setup}' is not included in top-level setups {self.setups}"
                    )
            if self.distributed and shape.batch % self.dual_world_size != 0 and "dual_die_request_sharded" in active_setups:
                raise ValueError(
                    "For request-sharded prefill, batch must be divisible by world size. "
                    f"Got batch={shape.batch}, world_size={self.dual_world_size}."
                )
        for shape in self.decode_shapes:
            shape.validate()
            active_setups = shape.setups or self.setups
            for setup in active_setups:
                if setup not in ALL_SETUPS:
                    raise ValueError(f"Unsupported setup in decode shape: {setup}")
                if setup not in self.setups:
                    raise ValueError(
                        f"Decode shape setup '{setup}' is not included in top-level setups {self.setups}"
                    )
            if self.distributed and shape.concurrency % self.dual_world_size != 0 and "dual_die_request_sharded" in active_setups:
                raise ValueError(
                    "For request-sharded decode, concurrency must be divisible by world size. "
                    f"Got concurrency={shape.concurrency}, world_size={self.dual_world_size}."
                )


def load_phase_study_config(path: str | Path) -> PhaseStudyConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return PhaseStudyConfig.from_dict(raw)


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


def _sample_on_cpu_then_move(device: Any) -> bool:
    device_str = str(device).lower()
    return device_str.startswith("xla") or "privateuseone" in device_str


def _randn(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: Any,
) -> torch.Tensor:
    if _sample_on_cpu_then_move(device):
        return torch.randn(shape, dtype=dtype, device="cpu").to(device)
    return torch.randn(shape, dtype=dtype, device=device)


def _timestamp_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _percentile_ms(values_s: list[float], q: float) -> float:
    if not values_s:
        return 0.0
    return float(np.percentile(np.array(values_s) * 1000.0, q))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values), q))


def _max_relative_error(reference: torch.Tensor, output: torch.Tensor, eps: float) -> float:
    scale = torch.maximum(reference.abs(), output.abs())
    dynamic_floor = float(scale.mean().item()) * 1e-3
    scale = scale.clamp_min(max(float(eps), dynamic_floor))
    return float(((reference - output).abs() / scale).max().item())


def _split_qkv(qkv: torch.Tensor, num_heads: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, s, three_d = qkv.shape
    d = three_d // 3
    dh = d // num_heads
    q, k, v = torch.chunk(qkv, 3, dim=-1)
    q = q.view(b, s, num_heads, dh).transpose(1, 2)
    k = k.view(b, s, num_heads, dh).transpose(1, 2)
    v = v.view(b, s, num_heads, dh).transpose(1, 2)
    return q, k, v


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    b, h, s, dh = x.shape
    return x.transpose(1, 2).reshape(b, s, h * dh)


def _decode_attention_step(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
    # q: [B, H, 1, Dh], k/v cache: [B, H, T, Dh]
    scale = 1.0 / math.sqrt(float(q.size(-1)))
    logits = torch.matmul(q.float(), k_cache.transpose(-1, -2).float()) * scale
    probs = torch.softmax(logits, dim=-1)
    return torch.matmul(probs, v_cache.float()).to(dtype=q.dtype)


def _append_cache_with_window(
    cache: torch.Tensor,
    update: torch.Tensor,
    *,
    max_cache_len: int,
) -> torch.Tensor:
    merged = torch.cat([cache, update], dim=-2)
    if merged.size(-2) > max_cache_len:
        merged = merged[..., -max_cache_len:, :]
    return merged


def _empty_kernel_breakdown() -> dict[str, dict[str, float]]:
    return {name: {"total_s": 0.0, "comm_s": 0.0, "bytes": 0.0} for name in KERNELS}


def _empty_kernel_series() -> dict[str, dict[str, list[float]]]:
    return {
        name: {
            "total_s": [],
            "compute_s": [],
            "comm_s": [],
            "overlap_pct": [],
            "bytes": [],
            "link_gbps": [],
        }
        for name in KERNELS
    }


def _merge_kernel_breakdown(
    *,
    dst: dict[str, dict[str, float]],
    src: dict[str, dict[str, float]],
) -> None:
    for name in KERNELS:
        payload = src.get(name)
        if payload is None:
            continue
        dst[name]["total_s"] += float(payload.get("total_s", 0.0))
        dst[name]["comm_s"] += float(payload.get("comm_s", 0.0))
        dst[name]["bytes"] += float(payload.get("bytes", 0.0))


def _timed_kernel(
    *,
    name: str,
    run: Any,
    device: Any,
    comm: ks.CommMetrics | None = None,
) -> tuple[Any, ks.CommMetrics, dict[str, float]]:
    start = time.perf_counter()
    raw_out = run()
    mark_step_if_needed(device)
    sync_if_needed(device)
    elapsed = time.perf_counter() - start
    out = raw_out
    op_comm = comm or ks.CommMetrics()
    if isinstance(raw_out, tuple) and len(raw_out) == 2 and isinstance(raw_out[1], ks.CommMetrics):
        out = raw_out[0]
        op_comm = raw_out[1]
    stats = {
        "total_s": float(elapsed),
        "comm_s": float(max(op_comm.time_s_total, 0.0)),
        "bytes": float(op_comm.bytes_total),
        "op_count": {name: int(count) for name, count in op_comm.counts_by_op.items()},
        "op_bytes": {name: float(payload) for name, payload in op_comm.bytes_by_op.items()},
        "op_time_s": {name: float(elapsed_s) for name, elapsed_s in op_comm.time_by_op.items()},
    }
    return out, op_comm, stats


def _make_weights(
    *,
    model_dim: int,
    mlp_ratio: int,
    dtype: torch.dtype,
    device: Any,
) -> dict[str, torch.Tensor]:
    mlp_dim = model_dim * mlp_ratio
    return {
        "w_qkv": _randn((model_dim, 3 * model_dim), dtype=dtype, device=device),
        "w_out": _randn((model_dim, model_dim), dtype=dtype, device=device),
        "w1": _randn((model_dim, mlp_dim), dtype=dtype, device=device),
        "w2": _randn((mlp_dim, model_dim), dtype=dtype, device=device),
        "gamma": _randn((model_dim,), dtype=dtype, device=device),
    }


def _prefill_step_single(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    *,
    num_heads: int,
) -> tuple[torch.Tensor, ks.CommMetrics, dict[str, dict[str, float]]]:
    device = x.device
    kernels = _empty_kernel_breakdown()

    x_norm, c0, s0 = _timed_kernel(
        name="rmsnorm",
        run=lambda: ks._rmsnorm_single(x, weights["gamma"]),
        device=device,
    )
    kernels["rmsnorm"] = s0

    qkv, c1, s1 = _timed_kernel(
        name="qkv_proj",
        run=lambda: ks._qkv_single(x_norm, weights["w_qkv"]),
        device=device,
    )
    kernels["qkv_proj"] = s1

    q, k, v = _split_qkv(qkv, num_heads=num_heads)
    attn, c2, s2 = _timed_kernel(
        name="attention",
        run=lambda: ks._attention_single(q, k, v, causal=False),
        device=device,
    )
    kernels["attention"] = s2

    attn_tokens = _merge_heads(attn)
    y_proj, c3, s3 = _timed_kernel(
        name="out_proj",
        run=lambda: ks._out_proj_single(attn_tokens, weights["w_out"]),
        device=device,
    )
    kernels["out_proj"] = s3

    y_mlp, c4, s4 = _timed_kernel(
        name="mlp",
        run=lambda: ks._mlp_single(x_norm, weights["w1"], weights["w2"]),
        device=device,
    )
    kernels["mlp"] = s4

    out = y_proj + y_mlp + x_norm
    comm_total = ks.CommMetrics()
    for payload in (c0, c1, c2, c3, c4):
        comm_total.merge_(payload)
    return out, comm_total, kernels


def _prefill_step_tensor(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    *,
    num_heads: int,
    attention_optimized: bool,
    attention_pipelined: bool,
    attention_tile_q: int,
    attention_tile_k: int,
    attention_reduce_group_k: int,
    dtype_bytes: int,
    ctx: DistributedContext,
    device: Any,
) -> tuple[torch.Tensor, ks.CommMetrics, dict[str, dict[str, float]]]:
    comm = ks.CommMetrics()
    kernels = _empty_kernel_breakdown()

    x_norm, c0, s0 = _timed_kernel(
        name="rmsnorm",
        run=lambda: ks._rmsnorm_dual_dist(
            x,
            weights["gamma"],
            optimized=True,
            dtype_bytes=dtype_bytes,
            ctx=ctx,
            device=device,
        ),
        device=device,
    )
    kernels["rmsnorm"] = s0

    qkv, c1, s1 = _timed_kernel(
        name="qkv_proj",
        run=lambda: ks._qkv_dual_dist(
            x_norm,
            weights["w_qkv"],
            optimized=True,
            dtype_bytes=dtype_bytes,
            ctx=ctx,
            device=device,
        ),
        device=device,
    )
    kernels["qkv_proj"] = s1

    q, k, v = _split_qkv(qkv, num_heads=num_heads)
    if attention_optimized:
        def run_attention() -> tuple[torch.Tensor, ks.CommMetrics]:
            return ks._attention_dual_dist_tiled_merge(
                q,
                k,
                v,
                causal=False,
                dtype_bytes=dtype_bytes,
                ctx=ctx,
                device=device,
                pipelined=attention_pipelined,
                tile_q=attention_tile_q,
                tile_k=attention_tile_k,
                reduce_group_k=attention_reduce_group_k,
            )
    else:
        def run_attention() -> tuple[torch.Tensor, ks.CommMetrics]:
            return ks._attention_dual_dist_naive(
                q,
                k,
                v,
                causal=False,
                dtype_bytes=dtype_bytes,
                ctx=ctx,
                device=device,
            )

    attn, c2, s2 = _timed_kernel(
        name="attention",
        run=run_attention,
        device=device,
    )
    kernels["attention"] = s2

    attn_tokens = _merge_heads(attn)
    y_proj, c3, s3 = _timed_kernel(
        name="out_proj",
        run=lambda: ks._out_proj_dual_dist(
            attn_tokens,
            weights["w_out"],
            optimized=True,
            dtype_bytes=dtype_bytes,
            ctx=ctx,
            device=device,
        ),
        device=device,
    )
    kernels["out_proj"] = s3

    y_mlp, c4, s4 = _timed_kernel(
        name="mlp",
        run=lambda: ks._mlp_dual_dist(
            x_norm,
            weights["w1"],
            weights["w2"],
            optimized=True,
            dtype_bytes=dtype_bytes,
            ctx=ctx,
            device=device,
        ),
        device=device,
    )
    kernels["mlp"] = s4

    for payload in (c0, c1, c2, c3, c4):
        comm.merge_(payload)
    out = y_proj + y_mlp + x_norm
    return out, comm, kernels


def _prefill_step_request_sharded(
    x_local: torch.Tensor,
    weights: dict[str, torch.Tensor],
    *,
    num_heads: int,
) -> tuple[torch.Tensor, ks.CommMetrics, dict[str, dict[str, float]]]:
    return _prefill_step_single(x_local, weights, num_heads=num_heads)


def _decode_step_single(
    x_t: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    weights: dict[str, torch.Tensor],
    *,
    num_heads: int,
    max_cache_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ks.CommMetrics, dict[str, dict[str, float]]]:
    kernels = _empty_kernel_breakdown()
    device = x_t.device

    x_norm, c0, s0 = _timed_kernel(
        name="rmsnorm",
        run=lambda: ks._rmsnorm_single(x_t, weights["gamma"]),
        device=device,
    )
    kernels["rmsnorm"] = s0

    qkv, c1, s1 = _timed_kernel(
        name="qkv_proj",
        run=lambda: ks._qkv_single(x_norm, weights["w_qkv"]),
        device=device,
    )
    kernels["qkv_proj"] = s1

    q, k_new, v_new = _split_qkv(qkv, num_heads=num_heads)
    k_cache = _append_cache_with_window(k_cache, k_new, max_cache_len=max_cache_len)
    v_cache = _append_cache_with_window(v_cache, v_new, max_cache_len=max_cache_len)

    attn, c2, s2 = _timed_kernel(
        name="attention",
        run=lambda: _decode_attention_step(q, k_cache, v_cache),
        device=device,
    )
    kernels["attention"] = s2

    attn_tokens = _merge_heads(attn)
    y_proj, c3, s3 = _timed_kernel(
        name="out_proj",
        run=lambda: ks._out_proj_single(attn_tokens, weights["w_out"]),
        device=device,
    )
    kernels["out_proj"] = s3

    y_mlp, c4, s4 = _timed_kernel(
        name="mlp",
        run=lambda: ks._mlp_single(x_norm, weights["w1"], weights["w2"]),
        device=device,
    )
    kernels["mlp"] = s4

    out = y_proj + y_mlp + x_norm
    comm_total = ks.CommMetrics()
    for payload in (c0, c1, c2, c3, c4):
        comm_total.merge_(payload)
    return out, k_cache, v_cache, comm_total, kernels


def _decode_step_tensor(
    x_t: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    weights: dict[str, torch.Tensor],
    *,
    num_heads: int,
    attention_optimized: bool,
    attention_pipelined: bool,
    attention_tile_q: int,
    attention_tile_k: int,
    attention_reduce_group_k: int,
    dtype_bytes: int,
    ctx: DistributedContext,
    device: Any,
    max_cache_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ks.CommMetrics, dict[str, dict[str, float]]]:
    comm = ks.CommMetrics()
    kernels = _empty_kernel_breakdown()

    x_norm, c0, s0 = _timed_kernel(
        name="rmsnorm",
        run=lambda: ks._rmsnorm_dual_dist(
            x_t,
            weights["gamma"],
            optimized=True,
            dtype_bytes=dtype_bytes,
            ctx=ctx,
            device=device,
        ),
        device=device,
    )
    kernels["rmsnorm"] = s0

    qkv, c1, s1 = _timed_kernel(
        name="qkv_proj",
        run=lambda: ks._qkv_dual_dist(
            x_norm,
            weights["w_qkv"],
            optimized=True,
            dtype_bytes=dtype_bytes,
            ctx=ctx,
            device=device,
        ),
        device=device,
    )
    kernels["qkv_proj"] = s1
    q, k_new, v_new = _split_qkv(qkv, num_heads=num_heads)
    k_cache = _append_cache_with_window(k_cache, k_new, max_cache_len=max_cache_len)
    v_cache = _append_cache_with_window(v_cache, v_new, max_cache_len=max_cache_len)

    if attention_optimized:
        def run_attention() -> tuple[torch.Tensor, ks.CommMetrics]:
            return ks._attention_dual_dist_tiled_merge(
                q,
                k_cache,
                v_cache,
                causal=False,
                dtype_bytes=dtype_bytes,
                ctx=ctx,
                device=device,
                pipelined=attention_pipelined,
                tile_q=attention_tile_q,
                tile_k=attention_tile_k,
                reduce_group_k=attention_reduce_group_k,
            )
    else:
        def run_attention() -> tuple[torch.Tensor, ks.CommMetrics]:
            return ks._attention_dual_dist_naive(
                q,
                k_cache,
                v_cache,
                causal=False,
                dtype_bytes=dtype_bytes,
                ctx=ctx,
                device=device,
            )

    attn, c2, s2 = _timed_kernel(
        name="attention",
        run=run_attention,
        device=device,
    )
    kernels["attention"] = s2
    attn_tokens = _merge_heads(attn)
    y_proj, c3, s3 = _timed_kernel(
        name="out_proj",
        run=lambda: ks._out_proj_dual_dist(
            attn_tokens,
            weights["w_out"],
            optimized=True,
            dtype_bytes=dtype_bytes,
            ctx=ctx,
            device=device,
        ),
        device=device,
    )
    kernels["out_proj"] = s3

    y_mlp, c4, s4 = _timed_kernel(
        name="mlp",
        run=lambda: ks._mlp_dual_dist(
            x_norm,
            weights["w1"],
            weights["w2"],
            optimized=True,
            dtype_bytes=dtype_bytes,
            ctx=ctx,
            device=device,
        ),
        device=device,
    )
    kernels["mlp"] = s4
    for payload in (c0, c1, c2, c3, c4):
        comm.merge_(payload)
    out = y_proj + y_mlp + x_norm
    return out, k_cache, v_cache, comm, kernels


def _bench_runner(
    fn: Any,
    *,
    warmup_iters: int,
    measure_iters: int,
    device: Any,
) -> tuple[
    torch.Tensor,
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    dict[str, dict[str, list[float]]],
    list[dict[str, Any]],
]:
    for _ in range(warmup_iters):
        _, _, _ = fn()
        mark_step_if_needed(device)
    sync_if_needed(device)

    out: torch.Tensor | None = None
    total_s: list[float] = []
    compute_s: list[float] = []
    comm_s: list[float] = []
    overlap_pct: list[float] = []
    comm_bytes: list[float] = []
    link_gbps: list[float] = []
    kernel_series = _empty_kernel_series()
    collective_samples: list[dict[str, Any]] = []

    for _ in range(measure_iters):
        start = time.perf_counter()
        out, comm, kernel_breakdown = fn()
        mark_step_if_needed(device)
        sync_if_needed(device)
        end = time.perf_counter()

        total = end - start
        comm_t = float(max(comm.time_s_total, 0.0))
        compute_t = max(total - comm_t, 0.0)
        hidden_overlap = max(0.0, compute_t + comm_t - total)
        overlap = (hidden_overlap / comm_t * 100.0) if comm_t > 0 else 0.0
        bw = (comm.bytes_total / comm_t / 1e9) if comm_t > 0 else 0.0

        total_s.append(total)
        compute_s.append(compute_t)
        comm_s.append(comm_t)
        overlap_pct.append(overlap)
        comm_bytes.append(float(comm.bytes_total))
        link_gbps.append(float(bw))

        for kernel in KERNELS:
            payload = kernel_breakdown.get(kernel, {})
            k_total = float(payload.get("total_s", 0.0))
            k_comm = float(max(payload.get("comm_s", 0.0), 0.0))
            k_compute = max(k_total - k_comm, 0.0)
            k_hidden_overlap = max(0.0, k_compute + k_comm - k_total)
            k_overlap = (k_hidden_overlap / k_comm * 100.0) if k_comm > 0 else 0.0
            k_bytes = float(payload.get("bytes", 0.0))
            k_bw = (k_bytes / k_comm / 1e9) if k_comm > 0 else 0.0

            kernel_series[kernel]["total_s"].append(k_total)
            kernel_series[kernel]["compute_s"].append(k_compute)
            kernel_series[kernel]["comm_s"].append(k_comm)
            kernel_series[kernel]["overlap_pct"].append(k_overlap)
            kernel_series[kernel]["bytes"].append(k_bytes)
            kernel_series[kernel]["link_gbps"].append(k_bw)
            op_count = payload.get("op_count", {})
            op_bytes = payload.get("op_bytes", {})
            op_time_s = payload.get("op_time_s", {})
            for op_name in set(op_count) | set(op_bytes) | set(op_time_s):
                collective_samples.append(
                    {
                        "kernel": kernel,
                        "op": op_name,
                        "count": float(op_count.get(op_name, 0.0)),
                        "bytes": float(op_bytes.get(op_name, 0.0)),
                        "time_s": float(op_time_s.get(op_name, 0.0)),
                    }
                )

    assert out is not None
    return (
        out,
        total_s,
        compute_s,
        comm_s,
        overlap_pct,
        comm_bytes,
        link_gbps,
        kernel_series,
        collective_samples,
    )


def _gather_batch_tensor(local: torch.Tensor, ctx: DistributedContext, device: Any) -> torch.Tensor:
    if not ctx.enabled:
        return local
    import torch.distributed as dist

    gathered = [torch.zeros_like(local) for _ in range(ctx.world_size)]
    dist.all_gather(gathered, local)
    mark_step_if_needed(device)
    sync_if_needed(device)
    return torch.cat(gathered, dim=0)


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


def _build_kernel_phase_rows(
    *,
    phase: PhaseName,
    setup: SetupName,
    device_name: str,
    dtype_name: str,
    batch: int,
    seq_len: int,
    context_len: int,
    decode_steps: int,
    model_dim: int,
    num_heads: int,
    fabric_peak_gbps: float,
    kernel_series: dict[str, dict[str, list[float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kernel in KERNELS:
        series = kernel_series.get(kernel, {})
        total_s = [float(x) for x in series.get("total_s", [])]
        compute_s = [float(x) for x in series.get("compute_s", [])]
        comm_s = [float(x) for x in series.get("comm_s", [])]
        overlap_pct = [float(x) for x in series.get("overlap_pct", [])]
        comm_bytes = [float(x) for x in series.get("bytes", [])]
        link_gbps = [float(x) for x in series.get("link_gbps", [])]
        achieved_link_gbps_p50 = _percentile(link_gbps, 50)
        link_util_pct = (achieved_link_gbps_p50 / fabric_peak_gbps * 100.0) if fabric_peak_gbps > 0 else 0.0

        row = {
            "timestamp": _timestamp_utc(),
            "phase": phase,
            "setup": setup,
            "kernel": kernel,
            "device": device_name,
            "dtype": dtype_name,
            "batch": int(batch),
            "seq_len": int(seq_len),
            "context_len": int(context_len),
            "decode_steps": int(decode_steps),
            "model_dim": int(model_dim),
            "num_heads": int(num_heads),
            "latency_ms_p50": round(float(_percentile_ms(total_s, 50)), 6),
            "latency_ms_p90": round(float(_percentile_ms(total_s, 90)), 6),
            "compute_ms_p50": round(float(_percentile_ms(compute_s, 50)), 6),
            "communication_ms_p50": round(float(_percentile_ms(comm_s, 50)), 6),
            "overlap_pct_p50": round(float(_percentile(overlap_pct, 50)), 6),
            "communication_bytes": round(float(np.mean(comm_bytes)) if comm_bytes else 0.0, 2),
            "achieved_link_gbps_p50": round(float(achieved_link_gbps_p50), 6),
            "link_utilization_pct_p50": round(float(link_util_pct), 6),
            "fabric_peak_gbps": round(float(fabric_peak_gbps), 6),
        }
        missing = [col for col in KERNEL_PHASE_COLUMNS if col not in row]
        if missing:
            raise RuntimeError(f"Missing kernel phase metrics columns: {missing}")
        rows.append(row)
    return rows


def _write_kernel_phase_metrics(run_dir: Path, records: list[dict[str, Any]]) -> tuple[Path, Path]:
    csv_path = run_dir / "kernel_phase_metrics.csv"
    jsonl_path = run_dir / "kernel_phase_metrics.jsonl"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=KERNEL_PHASE_COLUMNS)
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    return csv_path, jsonl_path


def _summarize_collective_samples(samples: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_op: dict[str, dict[str, list[float]]] = {}
    for sample in samples:
        op = str(sample.get("op", "unknown"))
        bucket = by_op.setdefault(op, {"count": [], "bytes": [], "time_s": []})
        bucket["count"].append(float(sample.get("count", 0.0)))
        bucket["bytes"].append(float(sample.get("bytes", 0.0)))
        bucket["time_s"].append(float(sample.get("time_s", 0.0)))

    out: dict[str, dict[str, float]] = {}
    for op_name, values in by_op.items():
        total_count = float(np.sum(values["count"])) if values["count"] else 0.0
        total_bytes = float(np.sum(values["bytes"])) if values["bytes"] else 0.0
        total_time = float(np.sum(values["time_s"])) if values["time_s"] else 0.0
        out[op_name] = {
            "count_total": total_count,
            "bytes_total": total_bytes,
            "time_ms_total": total_time * 1000.0,
            "count_p50": _percentile(values["count"], 50),
            "bytes_p50": _percentile(values["bytes"], 50),
            "time_ms_p50": _percentile(values["time_s"], 50) * 1000.0,
            "effective_gbps_p50": ((total_bytes / total_time) / 1e9) if total_time > 0 else 0.0,
        }
    return out


def _write_collectives_summary(run_dir: Path, rows: list[dict[str, Any]]) -> Path:
    payload = {
        "generated_at_utc": _timestamp_utc(),
        "rows": rows,
    }
    out_path = run_dir / "collectives_summary.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _build_decode_slo_summary(records: list[dict[str, Any]], slo_values: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    decode_records = [r for r in records if r["phase"] == "decode"]
    keys = sorted({(str(r["setup"]), int(r["context_len"])) for r in decode_records})

    for setup, context_len in keys:
        subset = [r for r in decode_records if r["setup"] == setup and int(r["context_len"]) == context_len]
        for slo in slo_values:
            eligible = [r for r in subset if float(r["latency_ms_p90"]) <= float(slo)]
            if not eligible:
                rows.append(
                    {
                        "setup": setup,
                        "context_len": context_len,
                        "slo_ms": float(slo),
                        "best_throughput_tokens_per_s": 0.0,
                        "best_concurrency": 0,
                        "best_latency_ms_p90": 0.0,
                        "feasible": False,
                    }
                )
                continue
            best = max(eligible, key=lambda row: float(row["throughput_tokens_per_s"]))
            rows.append(
                {
                    "setup": setup,
                    "context_len": context_len,
                    "slo_ms": float(slo),
                    "best_throughput_tokens_per_s": float(best["throughput_tokens_per_s"]),
                    "best_concurrency": int(best["batch"]),
                    "best_latency_ms_p90": float(best["latency_ms_p90"]),
                    "feasible": True,
                }
            )

    return rows


def _write_decode_slo_summary(run_dir: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    csv_path = run_dir / "decode_slo_summary.csv"
    md_path = run_dir / "decode_slo_summary.md"

    if not rows:
        csv_path.write_text("", encoding="utf-8")
        md_path.write_text("# Decode SLO Summary\n\nNo decode records found.\n", encoding="utf-8")
        return csv_path, md_path

    fields = [
        "setup",
        "context_len",
        "slo_ms",
        "best_throughput_tokens_per_s",
        "best_concurrency",
        "best_latency_ms_p90",
        "feasible",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "# Decode Throughput At SLO",
        "",
        "| Setup | Context | SLO (ms) | Best throughput (tokens/s) | Concurrency | p90 latency (ms) | Feasible |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['setup']} | {int(row['context_len'])} | {float(row['slo_ms']):.2f} | "
            f"{float(row['best_throughput_tokens_per_s']):.2f} | {int(row['best_concurrency'])} | "
            f"{float(row['best_latency_ms_p90']):.4f} | {bool(row['feasible'])} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def _build_break_even_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline: dict[tuple[Any, ...], dict[str, Any]] = {}

    for row in records:
        if row["setup"] != "single_die":
            continue
        key = (
            row["phase"],
            int(row["batch"]),
            int(row["seq_len"]),
            int(row["context_len"]),
            int(row["decode_steps"]),
            int(row["model_dim"]),
            int(row["num_heads"]),
        )
        baseline[key] = row

    for row in records:
        if row["setup"] == "single_die":
            continue
        key = (
            row["phase"],
            int(row["batch"]),
            int(row["seq_len"]),
            int(row["context_len"]),
            int(row["decode_steps"]),
            int(row["model_dim"]),
            int(row["num_heads"]),
        )
        single = baseline.get(key)
        if single is None:
            continue

        single_latency = float(single["latency_ms_p50"])
        dual_latency = float(row["latency_ms_p50"])
        dual_compute = float(row["compute_ms_p50"])
        dual_comm = float(row["communication_ms_p50"])
        measured_overlap = max(0.0, dual_compute + dual_comm - dual_latency)
        required_overlap = max(0.0, dual_compute + dual_comm - single_latency)
        overlap_gap = max(0.0, required_overlap - measured_overlap)
        speedup = (single_latency / dual_latency) if dual_latency > 0 else 0.0
        comm_fraction = (dual_comm / dual_latency * 100.0) if dual_latency > 0 else 0.0
        comm_budget = max(0.0, single_latency - dual_compute + measured_overlap)

        rows.append(
            {
                "phase": str(row["phase"]),
                "setup": str(row["setup"]),
                "batch": int(row["batch"]),
                "seq_len": int(row["seq_len"]),
                "context_len": int(row["context_len"]),
                "decode_steps": int(row["decode_steps"]),
                "single_latency_ms_p50": single_latency,
                "dual_latency_ms_p50": dual_latency,
                "dual_compute_ms_p50": dual_compute,
                "dual_comm_ms_p50": dual_comm,
                "measured_overlap_ms_p50": measured_overlap,
                "required_overlap_ms_to_tie_single": required_overlap,
                "additional_overlap_needed_ms": overlap_gap,
                "comm_budget_ms_to_tie_single": comm_budget,
                "dual_speedup_vs_single": speedup,
                "dual_wins": bool(dual_latency < single_latency),
                "comm_fraction_pct": comm_fraction,
            }
        )

    return rows


def _write_break_even_summary(run_dir: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    csv_path = run_dir / "break_even_summary.csv"
    md_path = run_dir / "break_even_summary.md"

    if not rows:
        csv_path.write_text("", encoding="utf-8")
        md_path.write_text("# Break-Even Summary\n\nNo comparable rows found.\n", encoding="utf-8")
        return csv_path, md_path

    fields = [
        "phase",
        "setup",
        "batch",
        "seq_len",
        "context_len",
        "decode_steps",
        "single_latency_ms_p50",
        "dual_latency_ms_p50",
        "dual_compute_ms_p50",
        "dual_comm_ms_p50",
        "measured_overlap_ms_p50",
        "required_overlap_ms_to_tie_single",
        "additional_overlap_needed_ms",
        "comm_budget_ms_to_tie_single",
        "dual_speedup_vs_single",
        "dual_wins",
        "comm_fraction_pct",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "# Break-Even Summary",
        "",
        "Dual wins when `compute + comm - overlap <= single latency`.",
        "",
        "| Phase | Setup | Batch | Seq | Context | Steps | Single p50 (ms) | Dual p50 (ms) | Dual compute (ms) | Dual comm (ms) | Measured overlap (ms) | Required overlap to tie (ms) | Additional overlap needed (ms) | Comm budget to tie (ms) | Dual speedup | Dual wins |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['phase']} | {row['setup']} | {int(row['batch'])} | {int(row['seq_len'])} | "
            f"{int(row['context_len'])} | {int(row['decode_steps'])} | "
            f"{float(row['single_latency_ms_p50']):.4f} | {float(row['dual_latency_ms_p50']):.4f} | "
            f"{float(row['dual_compute_ms_p50']):.4f} | {float(row['dual_comm_ms_p50']):.4f} | "
            f"{float(row['measured_overlap_ms_p50']):.4f} | "
            f"{float(row['required_overlap_ms_to_tie_single']):.4f} | "
            f"{float(row['additional_overlap_needed_ms']):.4f} | "
            f"{float(row['comm_budget_ms_to_tie_single']):.4f} | "
            f"{float(row['dual_speedup_vs_single']):.4f} | {bool(row['dual_wins'])} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def _validate_correctness(
    *,
    setup: SetupName,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    enforce: bool,
    abs_tol: float,
    rel_tol: float,
    eps: float,
) -> tuple[float, float]:
    ref_cpu = reference.detach().to("cpu", dtype=torch.float32)
    cand_cpu = candidate.detach().to("cpu", dtype=torch.float32)
    max_abs_err = float((ref_cpu - cand_cpu).abs().max().item())
    max_rel_err = _max_relative_error(ref_cpu, cand_cpu, eps=eps)

    if enforce and setup != "single_die" and (max_abs_err > abs_tol or max_rel_err > rel_tol):
        raise RuntimeError(
            f"Correctness gate failed for {setup}: "
            f"max_abs_err={max_abs_err:.6g} (tol={abs_tol:.6g}), "
            f"max_rel_err={max_rel_err:.6g} (tol={rel_tol:.6g})"
        )
    return max_abs_err, max_rel_err


def run_phase_study(
    *,
    config: PhaseStudyConfig,
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

    ctx = init_distributed_context(
        device_name=device_name,
        enable_distributed=distributed_requested,
        expected_world_size=(config.dual_world_size if distributed_requested else None),
    )

    run_dir = create_run_dir(output_dir) if ctx.is_primary else output_dir / "_distributed_non_primary"

    try:
        device = resolve_device(device_name)
        selected_setups = setups_override or config.setups
        for setup in selected_setups:
            if setup not in ALL_SETUPS:
                raise ValueError(f"Unsupported setup override: {setup}")
            if setup != "single_die" and not ctx.enabled:
                raise RuntimeError(
                    f"Setup {setup} requires distributed execution. Use torchrun + --distributed."
                )
        if distributed_requested and not ctx.enabled:
            raise RuntimeError("Distributed execution was requested but distributed context is not enabled.")

        if ctx.enabled:
            distributed_barrier(ctx)

        fabric_summary: dict[str, Any] = {
            "enabled": False,
            "peak_gbps": 0.0,
            "collectives": {},
        }
        if config.enable_fabric_calibration:
            fabric_summary = ks.run_fabric_calibration(
                device=device,
                dtype=dtype,
                dtype_name=config.dtype,
                dtype_bytes=dtype_bytes,
                config=config,
                dist_ctx=ctx,
                run_dir=run_dir if ctx.is_primary else None,
            )

        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        records: list[dict[str, Any]] = []
        kernel_phase_records: list[dict[str, Any]] = []
        collectives_rows: list[dict[str, Any]] = []
        local_mask = get_visible_core_mask()
        rank_masks_raw = [str(local_mask["raw"])]
        if ctx.enabled:
            rank_masks_raw = gather_rank_strings(
                local_value=str(local_mask["raw"]),
                ctx=ctx,
                device=device,
            )
        rank_core_masks = [
            {
                "rank": int(rank),
                "visible_cores_raw": str(raw),
                "visible_cores": parse_visible_cores(None if raw == "all" else str(raw)),
                "chip_id": "unknown",
            }
            for rank, raw in enumerate(rank_masks_raw)
        ]
        eps = 1e-3 if config.dtype == "bf16" else 1e-6

        # Prefill phase
        for idx, shape in enumerate(config.prefill_shapes):
            seed = config.seed + 1000 + idx
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            x_global = _randn((shape.batch, shape.seq_len, shape.model_dim), dtype=dtype, device=device)
            weights = _make_weights(
                model_dim=shape.model_dim,
                mlp_ratio=shape.mlp_ratio,
                dtype=dtype,
                device=device,
            )

            # Single-die reference output (rank0 authoritative in distributed mode).
            reference_out_full, _, _ = _prefill_step_single(x_global, weights, num_heads=shape.num_heads)
            sync_if_needed(device)
            tensor_attention_optimized_prefill = True
            if config.tensor_attention_naive_threshold > 0:
                tensor_attention_optimized_prefill = shape.seq_len < config.tensor_attention_naive_threshold
            shape_setups = [setup for setup in (shape.setups or selected_setups) if setup in selected_setups]
            if not shape_setups:
                if ctx.is_primary:
                    print(
                        "[phase-study] skipping prefill shape because no setups remain after overrides: "
                        f"batch={shape.batch} seq_len={shape.seq_len} model_dim={shape.model_dim} "
                        f"num_heads={shape.num_heads}",
                        flush=True,
                    )
                continue

            for setup in shape_setups:
                if setup == "single_die" and ctx.enabled and not ctx.is_primary:
                    distributed_barrier(ctx)
                    continue

                if setup == "single_die":
                    def runner() -> tuple[torch.Tensor, ks.CommMetrics, dict[str, dict[str, float]]]:
                        return _prefill_step_single(x_global, weights, num_heads=shape.num_heads)
                    tokens = shape.batch * shape.seq_len
                    kv_cache_bytes = 0.0
                elif setup == "dual_die_tensor_optimized":
                    def runner() -> tuple[torch.Tensor, ks.CommMetrics, dict[str, dict[str, float]]]:
                        return _prefill_step_tensor(
                            x_global,
                            weights,
                            num_heads=shape.num_heads,
                            attention_optimized=tensor_attention_optimized_prefill,
                            attention_pipelined=config.tensor_attention_pipelined_prefill,
                            attention_tile_q=config.tensor_attention_tile_q,
                            attention_tile_k=config.tensor_attention_tile_k,
                            attention_reduce_group_k=config.tensor_attention_reduce_group_k,
                            dtype_bytes=dtype_bytes,
                            ctx=ctx,
                            device=device,
                        )
                    tokens = shape.batch * shape.seq_len
                    kv_cache_bytes = 0.0
                else:
                    local_batch = shape.batch // ctx.world_size
                    b_start = ctx.rank * local_batch
                    b_end = b_start + local_batch
                    x_local = x_global[b_start:b_end]
                    def runner() -> tuple[torch.Tensor, ks.CommMetrics, dict[str, dict[str, float]]]:
                        return _prefill_step_request_sharded(x_local, weights, num_heads=shape.num_heads)
                    tokens = shape.batch * shape.seq_len
                    kv_cache_bytes = 0.0

                if ctx.is_primary:
                    attention_mode = "single_rank"
                    if setup == "dual_die_tensor_optimized":
                        attention_mode = (
                            "optimized" if tensor_attention_optimized_prefill else "naive_fallback"
                        )
                    elif setup == "dual_die_request_sharded":
                        attention_mode = "request_sharded"
                    print(
                        "[phase-study] prefill "
                        f"setup={setup} batch={shape.batch} seq_len={shape.seq_len} "
                        f"model_dim={shape.model_dim} num_heads={shape.num_heads} "
                        f"attention={attention_mode}",
                        flush=True,
                    )

                (
                    out,
                    total_s,
                    compute_s,
                    comm_s,
                    overlap_pct,
                    comm_bytes,
                    link_gbps,
                    kernel_series,
                    collective_samples,
                ) = _bench_runner(
                    runner,
                    warmup_iters=config.warmup_iters,
                    measure_iters=config.measure_iters,
                    device=device,
                )

                if setup == "dual_die_request_sharded":
                    out_full = _gather_batch_tensor(out, ctx=ctx, device=device)
                else:
                    out_full = out

                max_abs, max_rel = _validate_correctness(
                    setup=setup,
                    reference=reference_out_full,
                    candidate=out_full,
                    enforce=config.enforce_correctness,
                    abs_tol=config.correctness_abs_tol,
                    rel_tol=config.correctness_rel_tol,
                    eps=eps,
                )

                if not ctx.is_primary:
                    distributed_barrier(ctx)
                    continue

                p50_ms = _percentile_ms(total_s, 50)
                p90_ms = _percentile_ms(total_s, 90)
                throughput = tokens / (p50_ms / 1000.0) if p50_ms > 0 else 0.0
                achieved_link_gbps_p50 = _percentile(link_gbps, 50)
                fabric_peak = float(fabric_summary.get("peak_gbps", 0.0))
                link_util_pct = (achieved_link_gbps_p50 / fabric_peak * 100.0) if fabric_peak > 0 else 0.0

                record = {
                    "timestamp": _timestamp_utc(),
                    "phase": "prefill",
                    "setup": setup,
                    "device": device_name,
                    "dtype": config.dtype,
                    "batch": shape.batch,
                    "seq_len": shape.seq_len,
                    "context_len": 0,
                    "decode_steps": 0,
                    "model_dim": shape.model_dim,
                    "num_heads": shape.num_heads,
                    "latency_ms_p50": round(float(p50_ms), 6),
                    "latency_ms_p90": round(float(p90_ms), 6),
                    "compute_ms_p50": round(float(_percentile_ms(compute_s, 50)), 6),
                    "communication_ms_p50": round(float(_percentile_ms(comm_s, 50)), 6),
                    "overlap_pct_p50": round(float(_percentile(overlap_pct, 50)), 6),
                    "throughput_tokens_per_s": round(float(throughput), 6),
                    "communication_bytes": round(float(np.mean(comm_bytes)), 2),
                    "achieved_link_gbps_p50": round(float(achieved_link_gbps_p50), 6),
                    "link_utilization_pct_p50": round(float(link_util_pct), 6),
                    "fabric_peak_gbps": round(float(fabric_peak), 6),
                    "kv_cache_bytes_per_rank": round(float(kv_cache_bytes), 2),
                    "max_abs_err": round(float(max_abs), 8),
                    "max_rel_err": round(float(max_rel), 8),
                }
                missing = [col for col in REQUIRED_COLUMNS if col not in record]
                if missing:
                    raise RuntimeError(f"Missing metrics columns: {missing}")
                records.append(record)
                kernel_phase_records.extend(
                    _build_kernel_phase_rows(
                        phase="prefill",
                        setup=setup,
                        device_name=device_name,
                        dtype_name=config.dtype,
                        batch=shape.batch,
                        seq_len=shape.seq_len,
                        context_len=0,
                        decode_steps=0,
                        model_dim=shape.model_dim,
                        num_heads=shape.num_heads,
                        fabric_peak_gbps=fabric_peak,
                        kernel_series=kernel_series,
                    )
                )
                collectives_rows.append(
                    {
                        "timestamp": _timestamp_utc(),
                        "phase": "prefill",
                        "setup": setup,
                        "batch": shape.batch,
                        "seq_len": shape.seq_len,
                        "context_len": 0,
                        "decode_steps": 0,
                        "model_dim": shape.model_dim,
                        "num_heads": shape.num_heads,
                        "ops": _summarize_collective_samples(collective_samples),
                    }
                )
                if ctx.enabled:
                    distributed_barrier(ctx)

        # Decode phase
        for idx, shape in enumerate(config.decode_shapes):
            seed = config.seed + 10000 + idx
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            weights = _make_weights(
                model_dim=shape.model_dim,
                mlp_ratio=shape.mlp_ratio,
                dtype=dtype,
                device=device,
            )
            dh = shape.model_dim // shape.num_heads

            x_steps_global = _randn(
                (shape.decode_steps, shape.concurrency, 1, shape.model_dim),
                dtype=dtype,
                device=device,
            )
            k_cache_global = _randn(
                (shape.concurrency, shape.num_heads, shape.context_len, dh),
                dtype=dtype,
                device=device,
            )
            v_cache_global = _randn(
                (shape.concurrency, shape.num_heads, shape.context_len, dh),
                dtype=dtype,
                device=device,
            )

            # Single-die reference for correctness.
            ref_k = k_cache_global.clone()
            ref_v = v_cache_global.clone()
            ref_out = None
            for step in range(shape.decode_steps):
                x_t = x_steps_global[step]
                ref_out, ref_k, ref_v, _, _ = _decode_step_single(
                    x_t,
                    ref_k,
                    ref_v,
                    weights,
                    num_heads=shape.num_heads,
                    max_cache_len=shape.context_len,
                )
            assert ref_out is not None
            sync_if_needed(device)
            tensor_attention_optimized_decode = True
            if config.tensor_attention_naive_threshold > 0:
                tensor_attention_optimized_decode = shape.context_len < config.tensor_attention_naive_threshold
            shape_setups = [setup for setup in (shape.setups or selected_setups) if setup in selected_setups]
            if not shape_setups:
                if ctx.is_primary:
                    print(
                        "[phase-study] skipping decode shape because no setups remain after overrides: "
                        f"concurrency={shape.concurrency} context_len={shape.context_len} "
                        f"decode_steps={shape.decode_steps} model_dim={shape.model_dim} "
                        f"num_heads={shape.num_heads}",
                        flush=True,
                    )
                continue

            for setup in shape_setups:
                if setup == "single_die" and ctx.enabled and not ctx.is_primary:
                    distributed_barrier(ctx)
                    continue

                if setup == "single_die":
                    local_x_steps = x_steps_global
                    k_cache_init = k_cache_global.clone()
                    v_cache_init = v_cache_global.clone()

                    def decode_runner_single() -> tuple[torch.Tensor, ks.CommMetrics, dict[str, dict[str, float]]]:
                        k_cache = k_cache_init.clone()
                        v_cache = v_cache_init.clone()
                        out = torch.zeros((shape.concurrency, 1, shape.model_dim), dtype=dtype, device=device)
                        comm_total = ks.CommMetrics()
                        kernel_total = _empty_kernel_breakdown()
                        for step in range(shape.decode_steps):
                            out, k_cache, v_cache, comm, kernel_breakdown = _decode_step_single(
                                local_x_steps[step],
                                k_cache,
                                v_cache,
                                weights,
                                num_heads=shape.num_heads,
                                max_cache_len=shape.context_len,
                            )
                            comm_total.merge_(comm)
                            _merge_kernel_breakdown(dst=kernel_total, src=kernel_breakdown)
                        return out, comm_total, kernel_total

                    runner = decode_runner_single
                    kv_cache_bytes = float(
                        k_cache_init.numel() * dtype_bytes + v_cache_init.numel() * dtype_bytes
                    )
                elif setup == "dual_die_tensor_optimized":
                    local_x_steps = x_steps_global
                    k_cache_init = k_cache_global.clone()
                    v_cache_init = v_cache_global.clone()

                    def decode_runner_tensor() -> tuple[torch.Tensor, ks.CommMetrics, dict[str, dict[str, float]]]:
                        k_cache = k_cache_init.clone()
                        v_cache = v_cache_init.clone()
                        out = torch.zeros((shape.concurrency, 1, shape.model_dim), dtype=dtype, device=device)
                        comm_total = ks.CommMetrics()
                        kernel_total = _empty_kernel_breakdown()
                        for step in range(shape.decode_steps):
                            out, k_cache, v_cache, comm, kernel_breakdown = _decode_step_tensor(
                                local_x_steps[step],
                                k_cache,
                                v_cache,
                                weights,
                                num_heads=shape.num_heads,
                                attention_optimized=tensor_attention_optimized_decode,
                                attention_pipelined=config.tensor_attention_pipelined_decode,
                                attention_tile_q=config.tensor_attention_tile_q,
                                attention_tile_k=config.tensor_attention_tile_k,
                                attention_reduce_group_k=config.tensor_attention_reduce_group_k,
                                dtype_bytes=dtype_bytes,
                                ctx=ctx,
                                device=device,
                                max_cache_len=shape.context_len,
                            )
                            comm_total.merge_(comm)
                            _merge_kernel_breakdown(dst=kernel_total, src=kernel_breakdown)
                        return out, comm_total, kernel_total

                    runner = decode_runner_tensor
                    kv_cache_bytes = float(
                        k_cache_init.numel() * dtype_bytes + v_cache_init.numel() * dtype_bytes
                    )
                else:
                    local_batch = shape.concurrency // ctx.world_size
                    b_start = ctx.rank * local_batch
                    b_end = b_start + local_batch
                    local_x_steps = x_steps_global[:, b_start:b_end]
                    k_cache_init = k_cache_global[b_start:b_end].clone()
                    v_cache_init = v_cache_global[b_start:b_end].clone()

                    def decode_runner_request_sharded() -> tuple[
                        torch.Tensor, ks.CommMetrics, dict[str, dict[str, float]]
                    ]:
                        k_cache = k_cache_init.clone()
                        v_cache = v_cache_init.clone()
                        out = torch.zeros((local_batch, 1, shape.model_dim), dtype=dtype, device=device)
                        comm_total = ks.CommMetrics()
                        kernel_total = _empty_kernel_breakdown()
                        for step in range(shape.decode_steps):
                            out, k_cache, v_cache, comm, kernel_breakdown = _decode_step_single(
                                local_x_steps[step],
                                k_cache,
                                v_cache,
                                weights,
                                num_heads=shape.num_heads,
                                max_cache_len=shape.context_len,
                            )
                            comm_total.merge_(comm)
                            _merge_kernel_breakdown(dst=kernel_total, src=kernel_breakdown)
                        return out, comm_total, kernel_total

                    runner = decode_runner_request_sharded
                    kv_cache_bytes = float(
                        k_cache_init.numel() * dtype_bytes + v_cache_init.numel() * dtype_bytes
                    )

                if ctx.is_primary:
                    attention_mode = "single_rank"
                    if setup == "dual_die_tensor_optimized":
                        attention_mode = (
                            "optimized" if tensor_attention_optimized_decode else "naive_fallback"
                        )
                    elif setup == "dual_die_request_sharded":
                        attention_mode = "request_sharded"
                    print(
                        "[phase-study] decode "
                        f"setup={setup} concurrency={shape.concurrency} context_len={shape.context_len} "
                        f"decode_steps={shape.decode_steps} model_dim={shape.model_dim} "
                        f"num_heads={shape.num_heads} attention={attention_mode}",
                        flush=True,
                    )

                (
                    out,
                    total_s,
                    compute_s,
                    comm_s,
                    overlap_pct,
                    comm_bytes,
                    link_gbps,
                    kernel_series,
                    collective_samples,
                ) = _bench_runner(
                    runner,
                    warmup_iters=config.warmup_iters,
                    measure_iters=config.measure_iters,
                    device=device,
                )

                if setup == "dual_die_request_sharded":
                    out_full = _gather_batch_tensor(out, ctx=ctx, device=device)
                else:
                    out_full = out

                max_abs, max_rel = _validate_correctness(
                    setup=setup,
                    reference=ref_out,
                    candidate=out_full,
                    enforce=config.enforce_correctness,
                    abs_tol=config.correctness_abs_tol,
                    rel_tol=config.correctness_rel_tol,
                    eps=eps,
                )

                if not ctx.is_primary:
                    distributed_barrier(ctx)
                    continue

                p50_ms = _percentile_ms(total_s, 50)
                p90_ms = _percentile_ms(total_s, 90)
                tokens = shape.concurrency * shape.decode_steps
                throughput = tokens / (p50_ms / 1000.0) if p50_ms > 0 else 0.0
                achieved_link_gbps_p50 = _percentile(link_gbps, 50)
                fabric_peak = float(fabric_summary.get("peak_gbps", 0.0))
                link_util_pct = (achieved_link_gbps_p50 / fabric_peak * 100.0) if fabric_peak > 0 else 0.0

                record = {
                    "timestamp": _timestamp_utc(),
                    "phase": "decode",
                    "setup": setup,
                    "device": device_name,
                    "dtype": config.dtype,
                    "batch": shape.concurrency,
                    "seq_len": 1,
                    "context_len": shape.context_len,
                    "decode_steps": shape.decode_steps,
                    "model_dim": shape.model_dim,
                    "num_heads": shape.num_heads,
                    "latency_ms_p50": round(float(p50_ms), 6),
                    "latency_ms_p90": round(float(p90_ms), 6),
                    "compute_ms_p50": round(float(_percentile_ms(compute_s, 50)), 6),
                    "communication_ms_p50": round(float(_percentile_ms(comm_s, 50)), 6),
                    "overlap_pct_p50": round(float(_percentile(overlap_pct, 50)), 6),
                    "throughput_tokens_per_s": round(float(throughput), 6),
                    "communication_bytes": round(float(np.mean(comm_bytes)), 2),
                    "achieved_link_gbps_p50": round(float(achieved_link_gbps_p50), 6),
                    "link_utilization_pct_p50": round(float(link_util_pct), 6),
                    "fabric_peak_gbps": round(float(fabric_peak), 6),
                    "kv_cache_bytes_per_rank": round(float(kv_cache_bytes), 2),
                    "max_abs_err": round(float(max_abs), 8),
                    "max_rel_err": round(float(max_rel), 8),
                }
                missing = [col for col in REQUIRED_COLUMNS if col not in record]
                if missing:
                    raise RuntimeError(f"Missing metrics columns: {missing}")
                records.append(record)
                kernel_phase_records.extend(
                    _build_kernel_phase_rows(
                        phase="decode",
                        setup=setup,
                        device_name=device_name,
                        dtype_name=config.dtype,
                        batch=shape.concurrency,
                        seq_len=1,
                        context_len=shape.context_len,
                        decode_steps=shape.decode_steps,
                        model_dim=shape.model_dim,
                        num_heads=shape.num_heads,
                        fabric_peak_gbps=fabric_peak,
                        kernel_series=kernel_series,
                    )
                )
                collectives_rows.append(
                    {
                        "timestamp": _timestamp_utc(),
                        "phase": "decode",
                        "setup": setup,
                        "batch": shape.concurrency,
                        "seq_len": 1,
                        "context_len": shape.context_len,
                        "decode_steps": shape.decode_steps,
                        "model_dim": shape.model_dim,
                        "num_heads": shape.num_heads,
                        "ops": _summarize_collective_samples(collective_samples),
                    }
                )
                if ctx.enabled:
                    distributed_barrier(ctx)

        if ctx.is_primary:
            _write_metrics(run_dir, records)
            _write_kernel_phase_metrics(run_dir, kernel_phase_records)
            _write_collectives_summary(run_dir, collectives_rows)
            decode_slo = _build_decode_slo_summary(records, config.decode_slo_ms)
            _write_decode_slo_summary(run_dir, decode_slo)
            break_even = _build_break_even_summary(records)
            _write_break_even_summary(run_dir, break_even)

            repo_root = Path(__file__).resolve().parents[3]
            manifest = build_run_manifest(
                repo_root=repo_root,
                benchmark_config_path=config_path.resolve(),
                variant="phase_study",
                seed=config.seed,
                distributed_enabled=ctx.enabled,
                distributed_world_size=ctx.world_size,
                distributed_rank=ctx.rank,
                rank_core_masks=rank_core_masks,
            )
            manifest.update(
                {
                    "setups": list(selected_setups),
                    "distributed_requested": distributed_requested,
                    "distributed_enabled": ctx.enabled,
                    "distributed_backend": ctx.backend,
                    "distributed_world_size": ctx.world_size,
                    "decode_slo_ms": list(config.decode_slo_ms),
                    "correctness_abs_tol": config.correctness_abs_tol,
                    "correctness_rel_tol": config.correctness_rel_tol,
                    "fabric_peak_gbps": float(fabric_summary.get("peak_gbps", 0.0)),
                    "enable_fabric_calibration": config.enable_fabric_calibration,
                    "fabric_message_sizes": list(config.fabric_message_sizes),
                    "fabric_warmup_iters": config.fabric_warmup_iters,
                    "fabric_measure_iters": config.fabric_measure_iters,
                    "tensor_attention_naive_threshold": config.tensor_attention_naive_threshold,
                    "tensor_attention_tile_q": config.tensor_attention_tile_q,
                    "tensor_attention_tile_k": config.tensor_attention_tile_k,
                    "tensor_attention_reduce_group_k": config.tensor_attention_reduce_group_k,
                    "tensor_attention_pipelined_prefill": config.tensor_attention_pipelined_prefill,
                    "tensor_attention_pipelined_decode": config.tensor_attention_pipelined_decode,
                }
            )
            write_manifest(run_dir, manifest)

        distributed_barrier(ctx)
        return run_dir, records if ctx.is_primary else []
    finally:
        finalize_distributed_context(ctx)
