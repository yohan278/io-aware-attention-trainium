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
    init_distributed_context,
    mark_step_if_needed,
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


@dataclass(frozen=True)
class PrefillShape:
    batch: int
    seq_len: int
    model_dim: int
    num_heads: int
    mlp_ratio: int = 4

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PrefillShape":
        return cls(
            batch=int(raw["batch"]),
            seq_len=int(raw["seq_len"]),
            model_dim=int(raw["model_dim"]),
            num_heads=int(raw["num_heads"]),
            mlp_ratio=int(raw.get("mlp_ratio", 4)),
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

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DecodeShape":
        return cls(
            concurrency=int(raw["concurrency"]),
            context_len=int(raw["context_len"]),
            decode_steps=int(raw["decode_steps"]),
            model_dim=int(raw["model_dim"]),
            num_heads=int(raw["num_heads"]),
            mlp_ratio=int(raw.get("mlp_ratio", 4)),
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
            if self.distributed and shape.batch % self.dual_world_size != 0 and "dual_die_request_sharded" in self.setups:
                raise ValueError(
                    "For request-sharded prefill, batch must be divisible by world size. "
                    f"Got batch={shape.batch}, world_size={self.dual_world_size}."
                )
        for shape in self.decode_shapes:
            shape.validate()
            if self.distributed and shape.concurrency % self.dual_world_size != 0 and "dual_die_request_sharded" in self.setups:
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
    scale = torch.maximum(reference.abs(), output.abs()).clamp_min(eps)
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
    }


def _prefill_step_single(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, ks.CommMetrics]:
    qkv = ks._qkv_single(x, weights["w_qkv"])
    y_proj = ks._out_proj_single(x, weights["w_out"])
    y_mlp = ks._mlp_single(x, weights["w1"], weights["w2"])
    out = y_proj + y_mlp + qkv[..., : x.size(-1)]
    return out, ks.CommMetrics()


def _prefill_step_tensor(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    *,
    dtype_bytes: int,
    ctx: DistributedContext,
    device: Any,
) -> tuple[torch.Tensor, ks.CommMetrics]:
    comm = ks.CommMetrics()
    qkv, c1 = ks._qkv_dual_dist(
        x,
        weights["w_qkv"],
        optimized=True,
        dtype_bytes=dtype_bytes,
        ctx=ctx,
        device=device,
    )
    y_proj, c2 = ks._out_proj_dual_dist(
        x,
        weights["w_out"],
        optimized=True,
        dtype_bytes=dtype_bytes,
        ctx=ctx,
        device=device,
    )
    y_mlp, c3 = ks._mlp_dual_dist(
        x,
        weights["w1"],
        weights["w2"],
        optimized=True,
        dtype_bytes=dtype_bytes,
        ctx=ctx,
        device=device,
    )
    for payload in (c1, c2, c3):
        comm.bytes_total += payload.bytes_total
        comm.time_s_total += payload.time_s_total
    out = y_proj + y_mlp + qkv[..., : x.size(-1)]
    return out, comm


def _prefill_step_request_sharded(
    x_local: torch.Tensor,
    weights: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, ks.CommMetrics]:
    return _prefill_step_single(x_local, weights)


def _decode_step_single(
    x_t: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    weights: dict[str, torch.Tensor],
    *,
    num_heads: int,
    max_cache_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ks.CommMetrics]:
    qkv = ks._qkv_single(x_t, weights["w_qkv"])
    q, k_new, v_new = _split_qkv(qkv, num_heads=num_heads)
    k_cache = _append_cache_with_window(k_cache, k_new, max_cache_len=max_cache_len)
    v_cache = _append_cache_with_window(v_cache, v_new, max_cache_len=max_cache_len)

    attn = _decode_attention_step(q, k_cache, v_cache)
    attn_tokens = _merge_heads(attn)
    y_proj = ks._out_proj_single(attn_tokens, weights["w_out"])
    y_mlp = ks._mlp_single(x_t, weights["w1"], weights["w2"])
    out = y_proj + y_mlp
    return out, k_cache, v_cache, ks.CommMetrics()


def _decode_step_tensor(
    x_t: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    weights: dict[str, torch.Tensor],
    *,
    num_heads: int,
    dtype_bytes: int,
    ctx: DistributedContext,
    device: Any,
    max_cache_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ks.CommMetrics]:
    comm = ks.CommMetrics()
    qkv, c1 = ks._qkv_dual_dist(
        x_t,
        weights["w_qkv"],
        optimized=True,
        dtype_bytes=dtype_bytes,
        ctx=ctx,
        device=device,
    )
    q, k_new, v_new = _split_qkv(qkv, num_heads=num_heads)
    k_cache = _append_cache_with_window(k_cache, k_new, max_cache_len=max_cache_len)
    v_cache = _append_cache_with_window(v_cache, v_new, max_cache_len=max_cache_len)

    attn, c2 = ks._attention_dual_dist_optimized(
        q,
        k_cache,
        v_cache,
        causal=False,
        dtype_bytes=dtype_bytes,
        ctx=ctx,
        device=device,
    )
    attn_tokens = _merge_heads(attn)
    y_proj, c3 = ks._out_proj_dual_dist(
        attn_tokens,
        weights["w_out"],
        optimized=True,
        dtype_bytes=dtype_bytes,
        ctx=ctx,
        device=device,
    )
    y_mlp, c4 = ks._mlp_dual_dist(
        x_t,
        weights["w1"],
        weights["w2"],
        optimized=True,
        dtype_bytes=dtype_bytes,
        ctx=ctx,
        device=device,
    )
    for payload in (c1, c2, c3, c4):
        comm.bytes_total += payload.bytes_total
        comm.time_s_total += payload.time_s_total
    out = y_proj + y_mlp
    return out, k_cache, v_cache, comm


def _bench_runner(
    fn: Any,
    *,
    warmup_iters: int,
    measure_iters: int,
    device: Any,
) -> tuple[torch.Tensor, list[float], list[float], list[float], list[float], list[float], list[float]]:
    for _ in range(warmup_iters):
        _ = fn()
        mark_step_if_needed(device)
    sync_if_needed(device)

    out: torch.Tensor | None = None
    total_s: list[float] = []
    compute_s: list[float] = []
    comm_s: list[float] = []
    overlap_pct: list[float] = []
    comm_bytes: list[float] = []
    link_gbps: list[float] = []

    for _ in range(measure_iters):
        start = time.perf_counter()
        out, comm = fn()
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

    assert out is not None
    return out, total_s, compute_s, comm_s, overlap_pct, comm_bytes, link_gbps


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

    if enforce and setup != "single_die" and (max_abs_err > abs_tol and max_rel_err > rel_tol):
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
            reference_out_full, _ = _prefill_step_single(x_global, weights)
            sync_if_needed(device)

            for setup in selected_setups:
                if setup == "single_die" and ctx.enabled and not ctx.is_primary:
                    distributed_barrier(ctx)
                    continue

                if setup == "single_die":
                    runner = lambda: _prefill_step_single(x_global, weights)
                    tokens = shape.batch * shape.seq_len
                    kv_cache_bytes = 0.0
                elif setup == "dual_die_tensor_optimized":
                    runner = lambda: _prefill_step_tensor(
                        x_global,
                        weights,
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
                    runner = lambda: _prefill_step_request_sharded(x_local, weights)
                    tokens = shape.batch * shape.seq_len
                    kv_cache_bytes = 0.0

                out, total_s, compute_s, comm_s, overlap_pct, comm_bytes, link_gbps = _bench_runner(
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
                ref_out, ref_k, ref_v, _ = _decode_step_single(
                    x_t,
                    ref_k,
                    ref_v,
                    weights,
                    num_heads=shape.num_heads,
                    max_cache_len=shape.context_len,
                )
            assert ref_out is not None
            sync_if_needed(device)

            for setup in selected_setups:
                if setup == "single_die" and ctx.enabled and not ctx.is_primary:
                    distributed_barrier(ctx)
                    continue

                if setup == "single_die":
                    local_x_steps = x_steps_global
                    k_cache_init = k_cache_global.clone()
                    v_cache_init = v_cache_global.clone()

                    def decode_runner_single() -> tuple[torch.Tensor, ks.CommMetrics]:
                        k_cache = k_cache_init.clone()
                        v_cache = v_cache_init.clone()
                        out = torch.zeros((shape.concurrency, 1, shape.model_dim), dtype=dtype, device=device)
                        comm_total = ks.CommMetrics()
                        for step in range(shape.decode_steps):
                            out, k_cache, v_cache, comm = _decode_step_single(
                                local_x_steps[step],
                                k_cache,
                                v_cache,
                                weights,
                                num_heads=shape.num_heads,
                                max_cache_len=shape.context_len,
                            )
                            comm_total.bytes_total += comm.bytes_total
                            comm_total.time_s_total += comm.time_s_total
                        return out, comm_total

                    runner = decode_runner_single
                    kv_cache_bytes = float(
                        k_cache_init.numel() * dtype_bytes + v_cache_init.numel() * dtype_bytes
                    )
                elif setup == "dual_die_tensor_optimized":
                    local_x_steps = x_steps_global
                    k_cache_init = k_cache_global.clone()
                    v_cache_init = v_cache_global.clone()

                    def decode_runner_tensor() -> tuple[torch.Tensor, ks.CommMetrics]:
                        k_cache = k_cache_init.clone()
                        v_cache = v_cache_init.clone()
                        out = torch.zeros((shape.concurrency, 1, shape.model_dim), dtype=dtype, device=device)
                        comm_total = ks.CommMetrics()
                        for step in range(shape.decode_steps):
                            out, k_cache, v_cache, comm = _decode_step_tensor(
                                local_x_steps[step],
                                k_cache,
                                v_cache,
                                weights,
                                num_heads=shape.num_heads,
                                dtype_bytes=dtype_bytes,
                                ctx=ctx,
                                device=device,
                                max_cache_len=shape.context_len,
                            )
                            comm_total.bytes_total += comm.bytes_total
                            comm_total.time_s_total += comm.time_s_total
                        return out, comm_total

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

                    def decode_runner_request_sharded() -> tuple[torch.Tensor, ks.CommMetrics]:
                        k_cache = k_cache_init.clone()
                        v_cache = v_cache_init.clone()
                        out = torch.zeros((local_batch, 1, shape.model_dim), dtype=dtype, device=device)
                        comm_total = ks.CommMetrics()
                        for step in range(shape.decode_steps):
                            out, k_cache, v_cache, comm = _decode_step_single(
                                local_x_steps[step],
                                k_cache,
                                v_cache,
                                weights,
                                num_heads=shape.num_heads,
                                max_cache_len=shape.context_len,
                            )
                            comm_total.bytes_total += comm.bytes_total
                            comm_total.time_s_total += comm.time_s_total
                        return out, comm_total

                    runner = decode_runner_request_sharded
                    kv_cache_bytes = float(
                        k_cache_init.numel() * dtype_bytes + v_cache_init.numel() * dtype_bytes
                    )

                out, total_s, compute_s, comm_s, overlap_pct, comm_bytes, link_gbps = _bench_runner(
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
                if ctx.enabled:
                    distributed_barrier(ctx)

        if ctx.is_primary:
            _write_metrics(run_dir, records)
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
                }
            )
            write_manifest(run_dir, manifest)

        distributed_barrier(ctx)
        return run_dir, records if ctx.is_primary else []
    finally:
        finalize_distributed_context(ctx)
