from __future__ import annotations

import csv
import json
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
from io_aware_attention.experiments import kernel_study as ks
from io_aware_attention.runtime.trainium import (
    DistributedContext,
    distributed_barrier,
    finalize_distributed_context,
    get_visible_core_mask,
    init_distributed_context,
    mark_step_if_needed,
    parse_visible_cores,
    resolve_device,
    sync_if_needed,
)

SetupName = Literal["single_die", "dual_die_moe_naive", "dual_die_moe_locality"]
DeviceName = Literal["cpu", "trainium"]
DTypeName = Literal["bf16", "fp32"]

ALL_SETUPS: tuple[SetupName, ...] = (
    "single_die",
    "dual_die_moe_naive",
    "dual_die_moe_locality",
)

REQUIRED_COLUMNS = [
    "timestamp",
    "phase",
    "setup",
    "device",
    "dtype",
    "batch",
    "context_len",
    "decode_steps",
    "model_dim",
    "hidden_dim",
    "num_experts",
    "top_k",
    "routing_skew",
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
    "remote_dispatch_ratio_p50",
    "max_abs_err",
    "max_rel_err",
]


@dataclass(frozen=True)
class DecodeShape:
    concurrency: int
    context_len: int
    decode_steps: int
    model_dim: int
    hidden_dim: int
    num_experts: int
    top_k: int
    routing_skew: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DecodeShape":
        return cls(
            concurrency=int(raw["concurrency"]),
            context_len=int(raw["context_len"]),
            decode_steps=int(raw["decode_steps"]),
            model_dim=int(raw["model_dim"]),
            hidden_dim=int(raw["hidden_dim"]),
            num_experts=int(raw["num_experts"]),
            top_k=int(raw.get("top_k", 2)),
            routing_skew=float(raw.get("routing_skew", 1.2)),
        )

    def validate(self, world_size: int, distributed: bool) -> None:
        if self.concurrency < 1 or self.context_len < 1 or self.decode_steps < 1:
            raise ValueError(f"Invalid decode shape: {self}")
        if self.model_dim < 2 or self.hidden_dim < 2:
            raise ValueError(f"model_dim/hidden_dim invalid: {self}")
        if self.num_experts < 2 or self.num_experts % world_size != 0:
            raise ValueError(
                "num_experts must be >=2 and divisible by dual_world_size; "
                f"got num_experts={self.num_experts}, world_size={world_size}"
            )
        if self.top_k < 1 or self.top_k > self.num_experts:
            raise ValueError(f"top_k must be in [1, num_experts], got {self.top_k}")
        if self.routing_skew < 0:
            raise ValueError(f"routing_skew must be >= 0, got {self.routing_skew}")
        if distributed and self.concurrency % world_size != 0:
            raise ValueError(
                "For distributed MoE study, concurrency must be divisible by world size. "
                f"Got concurrency={self.concurrency}, world_size={world_size}."
            )


@dataclass(frozen=True)
class MoEStudyConfig:
    device: DeviceName
    dtype: DTypeName
    seed: int
    warmup_iters: int
    measure_iters: int
    distributed: bool
    dual_world_size: int
    setups: list[SetupName]
    decode_shapes: list[DecodeShape]
    decode_slo_ms: list[float]
    capacity_slo_ms: float
    continue_on_runtime_error: bool
    record_runtime_failures: bool
    enable_fabric_calibration: bool
    fabric_message_sizes: list[int]
    fabric_warmup_iters: int
    fabric_measure_iters: int
    enforce_correctness: bool
    correctness_abs_tol: float
    correctness_rel_tol: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MoEStudyConfig":
        decode_shapes = [DecodeShape.from_dict(item) for item in raw.get("decode", [])]
        cfg = cls(
            device=str(raw.get("device", "cpu")),
            dtype=str(raw.get("dtype", "fp32")),
            seed=int(raw.get("seed", 0)),
            warmup_iters=int(raw.get("warmup_iters", 1)),
            measure_iters=int(raw.get("measure_iters", 5)),
            distributed=bool(raw.get("distributed", False)),
            dual_world_size=int(raw.get("dual_world_size", 2)),
            setups=[str(item) for item in raw.get("setups", list(ALL_SETUPS))],
            decode_shapes=decode_shapes,
            decode_slo_ms=[float(x) for x in raw.get("decode_slo_ms", [100.0, 250.0, 500.0])],
            capacity_slo_ms=float(raw.get("capacity_slo_ms", 250.0)),
            continue_on_runtime_error=bool(raw.get("continue_on_runtime_error", True)),
            record_runtime_failures=bool(raw.get("record_runtime_failures", True)),
            enable_fabric_calibration=bool(raw.get("enable_fabric_calibration", True)),
            fabric_message_sizes=[int(x) for x in raw.get("fabric_message_sizes", ks.DEFAULT_FABRIC_MESSAGE_SIZES)],
            fabric_warmup_iters=int(raw.get("fabric_warmup_iters", 2)),
            fabric_measure_iters=int(raw.get("fabric_measure_iters", 8)),
            enforce_correctness=bool(raw.get("enforce_correctness", True)),
            correctness_abs_tol=float(raw.get("correctness_abs_tol", 0.02)),
            correctness_rel_tol=float(raw.get("correctness_rel_tol", 0.025)),
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
        if self.dual_world_size != 2:
            raise ValueError("MoE service study currently supports dual_world_size=2 only")
        if not self.decode_shapes:
            raise ValueError("At least one decode shape is required")
        if self.capacity_slo_ms <= 0:
            raise ValueError("capacity_slo_ms must be > 0")
        for setup in self.setups:
            if setup not in ALL_SETUPS:
                raise ValueError(f"Unsupported setup: {setup}")
        for shape in self.decode_shapes:
            shape.validate(self.dual_world_size, self.distributed)


@dataclass
class CommMetrics:
    bytes_total: float = 0.0
    time_s_total: float = 0.0
    counts_by_op: dict[str, int] = field(default_factory=dict)
    bytes_by_op: dict[str, float] = field(default_factory=dict)
    time_by_op: dict[str, float] = field(default_factory=dict)

    def add(self, op_name: str, payload_bytes: float, elapsed_s: float) -> None:
        self.bytes_total += float(payload_bytes)
        self.time_s_total += float(elapsed_s)
        self.counts_by_op[op_name] = self.counts_by_op.get(op_name, 0) + 1
        self.bytes_by_op[op_name] = self.bytes_by_op.get(op_name, 0.0) + float(payload_bytes)
        self.time_by_op[op_name] = self.time_by_op.get(op_name, 0.0) + float(elapsed_s)

    def merge_(self, other: "CommMetrics") -> None:
        self.bytes_total += float(other.bytes_total)
        self.time_s_total += float(other.time_s_total)
        for k, v in other.counts_by_op.items():
            self.counts_by_op[k] = self.counts_by_op.get(k, 0) + int(v)
        for k, v in other.bytes_by_op.items():
            self.bytes_by_op[k] = self.bytes_by_op.get(k, 0.0) + float(v)
        for k, v in other.time_by_op.items():
            self.time_by_op[k] = self.time_by_op.get(k, 0.0) + float(v)


def load_moe_study_config(path: str | Path) -> MoEStudyConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return MoEStudyConfig.from_dict(raw)


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


def _randn(shape: tuple[int, ...], *, dtype: torch.dtype, device: Any) -> torch.Tensor:
    if _sample_on_cpu_then_move(device):
        out = torch.randn(shape, dtype=dtype, device="cpu")
        return out.to(device)
    return torch.randn(shape, dtype=dtype, device=device)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _percentile_ms(values_s: list[float], q: float) -> float:
    return _percentile(values_s, q) * 1000.0


def _max_relative_error(reference: torch.Tensor, candidate: torch.Tensor, *, eps: float) -> float:
    reference = reference.detach().to(dtype=torch.float32)
    candidate = candidate.detach().to(dtype=torch.float32)
    denom = torch.maximum(reference.abs(), torch.full_like(reference, float(eps)))
    rel = (reference - candidate).abs() / denom
    return float(rel.max().item())


def _all_gather_payload_bytes(local_message_bytes: int, world_size: int) -> float:
    if world_size <= 1:
        return 0.0
    return float(local_message_bytes) * float(world_size - 1)


def _all_reduce_payload_bytes(local_message_bytes: int, world_size: int) -> float:
    if world_size <= 1:
        return 0.0
    # Approximate effective bytes traversing links for all-reduce.
    return float(local_message_bytes) * float(2 * (world_size - 1))


def _collective_all_gather(
    local_tensor: torch.Tensor,
    *,
    op_name: str,
    metrics: CommMetrics,
    ctx: DistributedContext,
    device: Any,
) -> list[torch.Tensor]:
    if not ctx.enabled:
        return [local_tensor]
    import torch.distributed as dist

    gathered = [torch.zeros_like(local_tensor) for _ in range(ctx.world_size)]
    start = time.perf_counter()
    dist.all_gather(gathered, local_tensor)
    mark_step_if_needed(device)
    sync_if_needed(device)
    elapsed = time.perf_counter() - start

    payload = _all_gather_payload_bytes(
        local_message_bytes=int(local_tensor.numel()) * int(local_tensor.element_size()),
        world_size=ctx.world_size,
    )
    metrics.add(op_name, payload, elapsed)
    return gathered


def _collective_all_reduce_sum(
    tensor: torch.Tensor,
    *,
    op_name: str,
    metrics: CommMetrics,
    ctx: DistributedContext,
    device: Any,
) -> torch.Tensor:
    if not ctx.enabled:
        return tensor
    import torch.distributed as dist

    out = tensor.clone()
    start = time.perf_counter()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    mark_step_if_needed(device)
    sync_if_needed(device)
    elapsed = time.perf_counter() - start

    payload = _all_reduce_payload_bytes(
        local_message_bytes=int(out.numel()) * int(out.element_size()),
        world_size=ctx.world_size,
    )
    metrics.add(op_name, payload, elapsed)
    return out


def _gather_batch_tensor(local: torch.Tensor, ctx: DistributedContext, device: Any) -> torch.Tensor:
    if not ctx.enabled:
        return local
    import torch.distributed as dist

    gathered = [torch.zeros_like(local) for _ in range(ctx.world_size)]
    dist.all_gather(gathered, local)
    mark_step_if_needed(device)
    sync_if_needed(device)
    return torch.cat(gathered, dim=0)


def _make_expert_weights(
    *,
    num_experts: int,
    model_dim: int,
    hidden_dim: int,
    dtype: torch.dtype,
    device: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    w1 = _randn((num_experts, model_dim, hidden_dim), dtype=dtype, device=device) * 0.03
    w2 = _randn((num_experts, hidden_dim, model_dim), dtype=dtype, device=device) * 0.03
    return w1, w2


def _generate_router_assignments(
    *,
    decode_steps: int,
    concurrency: int,
    num_experts: int,
    top_k: int,
    routing_skew: float,
    device: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = _randn((decode_steps, concurrency, num_experts), dtype=torch.float32, device=device)
    if routing_skew > 0:
        # Higher skew biases traffic toward lower-index experts.
        bias = torch.linspace(0.0, -routing_skew, num_experts, dtype=torch.float32, device=device)
        logits = logits + bias.view(1, 1, num_experts)
    top_vals, top_idx = torch.topk(logits, k=top_k, dim=-1)
    gates = torch.softmax(top_vals, dim=-1)
    return top_idx.to(dtype=torch.int64), gates.to(dtype=torch.float32)


def _build_naive_placement(num_experts: int, world_size: int, device: Any) -> torch.Tensor:
    placement = torch.arange(num_experts, dtype=torch.int64, device=device) % int(world_size)
    return placement


def _build_locality_placement(
    *,
    expert_idx_steps: torch.Tensor,
    num_experts: int,
    world_size: int,
    device: Any,
) -> torch.Tensor:
    if world_size != 2:
        raise ValueError("Locality placement currently supports world_size=2")
    counts = torch.bincount(expert_idx_steps.reshape(-1), minlength=num_experts)
    per_rank = num_experts // world_size
    placement = torch.empty((num_experts,), dtype=torch.int64, device=device)
    # Avoid sort-based ops (unsupported on Trn2) and use top-k to pick the busiest experts.
    _, top_idx = torch.topk(counts.to(dtype=torch.float32), k=per_rank, largest=True)
    top_idx = top_idx.to(dtype=torch.int64)
    placement.fill_(1)
    placement[top_idx] = 0
    return placement


def _apply_experts(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    *,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    if x.numel() == 0:
        return torch.zeros_like(x)
    num_experts = int(w1.size(0))
    expert_ids_safe = expert_ids.to(dtype=torch.int64).clamp_min(0).clamp_max(num_experts - 1)
    w1_sel = w1.index_select(0, expert_ids_safe).to(dtype=torch.float32)
    w2_sel = w2.index_select(0, expert_ids_safe).to(dtype=torch.float32)

    x_fp32 = x.to(dtype=torch.float32)
    h = torch.bmm(x_fp32.unsqueeze(1), w1_sel).squeeze(1)
    h = torch.nn.functional.silu(h)
    y = torch.bmm(h.unsqueeze(1), w2_sel).squeeze(1)
    return y


def _moe_step_single(
    *,
    x: torch.Tensor,
    expert_idx: torch.Tensor,
    gates: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> tuple[torch.Tensor, CommMetrics, float]:
    bsz, top_k = expert_idx.shape
    token_idx = torch.arange(bsz, dtype=torch.int64, device=x.device).unsqueeze(1).expand(bsz, top_k)

    flat_token = token_idx.reshape(-1)
    flat_expert = expert_idx.reshape(-1)
    flat_gate = gates.reshape(-1)
    flat_x = x.index_select(0, flat_token)

    contrib = _apply_experts(flat_x, flat_expert, w1=w1, w2=w2)
    contrib = contrib * flat_gate.to(dtype=contrib.dtype).unsqueeze(-1)

    y = torch.zeros((bsz, x.size(-1)), dtype=torch.float32, device=x.device)
    y.index_add_(0, flat_token, contrib)
    return y.to(dtype=x.dtype), CommMetrics(), 0.0


def _moe_step_dual(
    *,
    x_local: torch.Tensor,
    expert_idx_local: torch.Tensor,
    gates_local: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    placement: torch.Tensor,
    ctx: DistributedContext,
    device: Any,
) -> tuple[torch.Tensor, CommMetrics, float]:
    if not ctx.enabled:
        raise RuntimeError("Dual MoE step requires distributed context")
    if ctx.world_size != 2:
        raise RuntimeError("Dual MoE step supports world_size=2 only")

    bsz, top_k = expert_idx_local.shape

    metrics = CommMetrics()
    # Robust distributed formulation for Trainium:
    # 1) all-gather token states + routing metadata across ranks,
    # 2) each rank computes only contributions for experts it owns,
    # 3) all-reduce partial outputs to exact full output, then slice local batch.
    gathered_x = _collective_all_gather(
        x_local.to(dtype=torch.float32),
        op_name="all_gather_tokens",
        metrics=metrics,
        ctx=ctx,
        device=device,
    )
    gathered_idx = _collective_all_gather(
        expert_idx_local.to(dtype=torch.float32),
        op_name="all_gather_router_idx",
        metrics=metrics,
        ctx=ctx,
        device=device,
    )
    gathered_gates = _collective_all_gather(
        gates_local.to(dtype=torch.float32),
        op_name="all_gather_router_gates",
        metrics=metrics,
        ctx=ctx,
        device=device,
    )

    x_full = torch.cat(gathered_x, dim=0)
    idx_full = torch.cat(gathered_idx, dim=0).to(dtype=torch.int64)
    gates_full = torch.cat(gathered_gates, dim=0).to(dtype=torch.float32)

    global_bsz = int(x_full.size(0))
    token_idx = torch.arange(global_bsz, dtype=torch.int64, device=x_local.device).unsqueeze(1).expand(global_bsz, top_k)
    flat_token = token_idx.reshape(-1)
    flat_expert = idx_full.reshape(-1)
    flat_gate = gates_full.reshape(-1)
    flat_x = x_full.index_select(0, flat_token)

    owner = placement.index_select(0, flat_expert)
    local_weight = (owner == int(ctx.rank)).to(dtype=torch.float32)

    local_contrib = _apply_experts(flat_x, flat_expert, w1=w1, w2=w2)
    local_contrib = local_contrib * flat_gate.unsqueeze(-1) * local_weight.unsqueeze(-1)

    y_partial = torch.zeros((global_bsz, x_local.size(-1)), dtype=torch.float32, device=x_local.device)
    y_partial.index_add_(0, flat_token, local_contrib)

    y_full = _collective_all_reduce_sum(
        y_partial,
        op_name="all_reduce_output_sum",
        metrics=metrics,
        ctx=ctx,
        device=device,
    )

    y_local = y_full[int(ctx.rank) * bsz : int(ctx.rank + 1) * bsz]
    return y_local.to(dtype=x_local.dtype), metrics, 0.0


def _bench_runner(
    fn,
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
    list[float],
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
    remote_ratio: list[float] = []
    collective_samples: list[dict[str, Any]] = []

    for _ in range(measure_iters):
        start = time.perf_counter()
        out, comm, remote_ratio_step = fn()
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
        remote_ratio.append(float(remote_ratio_step))

        for op_name in set(comm.counts_by_op) | set(comm.bytes_by_op) | set(comm.time_by_op):
            collective_samples.append(
                {
                    "op": str(op_name),
                    "count": float(comm.counts_by_op.get(op_name, 0)),
                    "bytes": float(comm.bytes_by_op.get(op_name, 0.0)),
                    "time_s": float(comm.time_by_op.get(op_name, 0.0)),
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
        remote_ratio,
        collective_samples,
    )


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


def _write_metrics(run_dir: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    csv_path = run_dir / "metrics.csv"
    jsonl_path = run_dir / "metrics.jsonl"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    return csv_path, jsonl_path


def _write_collectives_summary(run_dir: Path, rows: list[dict[str, Any]]) -> Path:
    out_path = run_dir / "collectives_summary.json"
    payload = {"generated_at_utc": _timestamp_utc(), "rows": rows}
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _build_decode_slo_summary(records: list[dict[str, Any]], slo_values: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = sorted(
        {
            (
                str(r["setup"]),
                int(r["context_len"]),
                int(r["num_experts"]),
                int(r["top_k"]),
                float(r["routing_skew"]),
            )
            for r in records
        }
    )

    for setup, context_len, num_experts, top_k, routing_skew in keys:
        subset = [
            r
            for r in records
            if r["setup"] == setup
            and int(r["context_len"]) == context_len
            and int(r["num_experts"]) == num_experts
            and int(r["top_k"]) == top_k
            and float(r["routing_skew"]) == routing_skew
        ]
        for slo in slo_values:
            eligible = [r for r in subset if float(r["latency_ms_p90"]) <= float(slo)]
            if not eligible:
                rows.append(
                    {
                        "setup": setup,
                        "context_len": context_len,
                        "num_experts": num_experts,
                        "top_k": top_k,
                        "routing_skew": routing_skew,
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
                    "num_experts": num_experts,
                    "top_k": top_k,
                    "routing_skew": routing_skew,
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

    fields = [
        "setup",
        "context_len",
        "num_experts",
        "top_k",
        "routing_skew",
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
        "# MoE Decode Throughput At SLO",
        "",
        "| Setup | Context | Experts | Top-k | Routing skew | SLO (ms) | Best throughput (tokens/s) | Concurrency | p90 latency (ms) | Feasible |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['setup']} | {int(row['context_len'])} | {int(row['num_experts'])} | {int(row['top_k'])} | "
            f"{float(row['routing_skew']):.2f} | {float(row['slo_ms']):.2f} | "
            f"{float(row['best_throughput_tokens_per_s']):.2f} | {int(row['best_concurrency'])} | "
            f"{float(row['best_latency_ms_p90']):.4f} | {bool(row['feasible'])} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def _build_capacity_frontier(records: list[dict[str, Any]], capacity_slo_ms: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = sorted(
        {
            (
                str(r["setup"]),
                int(r["context_len"]),
                int(r["num_experts"]),
                int(r["top_k"]),
                float(r["routing_skew"]),
            )
            for r in records
        }
    )
    for setup, context_len, num_experts, top_k, routing_skew in keys:
        subset = [
            r
            for r in records
            if r["setup"] == setup
            and int(r["context_len"]) == context_len
            and int(r["num_experts"]) == num_experts
            and int(r["top_k"]) == top_k
            and float(r["routing_skew"]) == routing_skew
        ]
        feasible = [r for r in subset if float(r["latency_ms_p90"]) <= float(capacity_slo_ms)]
        best = max(feasible, key=lambda row: float(row["throughput_tokens_per_s"])) if feasible else None
        rows.append(
            {
                "setup": setup,
                "context_len": int(context_len),
                "num_experts": int(num_experts),
                "top_k": int(top_k),
                "routing_skew": float(routing_skew),
                "slo_ms": float(capacity_slo_ms),
                "max_tested_concurrency": int(max((int(r["batch"]) for r in subset), default=0)),
                "max_feasible_concurrency": int(max((int(r["batch"]) for r in feasible), default=0)),
                "best_throughput_tokens_per_s": float(best["throughput_tokens_per_s"]) if best else 0.0,
                "best_concurrency": int(best["batch"]) if best else 0,
                "best_latency_ms_p90": float(best["latency_ms_p90"]) if best else 0.0,
                "has_feasible": bool(best is not None),
            }
        )
    return rows


def _write_capacity_frontier(run_dir: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    csv_path = run_dir / "capacity_frontier.csv"
    md_path = run_dir / "capacity_frontier.md"

    fields = [
        "setup",
        "context_len",
        "num_experts",
        "top_k",
        "routing_skew",
        "slo_ms",
        "max_tested_concurrency",
        "max_feasible_concurrency",
        "best_throughput_tokens_per_s",
        "best_concurrency",
        "best_latency_ms_p90",
        "has_feasible",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "# MoE Capacity Frontier",
        "",
        "| Setup | Context | Experts | Top-k | Routing skew | SLO (ms) | Max tested conc | Max feasible conc | Best throughput (tokens/s) | Best conc | Best p90 latency (ms) | Has feasible |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['setup']} | {int(row['context_len'])} | {int(row['num_experts'])} | {int(row['top_k'])} | "
            f"{float(row['routing_skew']):.2f} | {float(row['slo_ms']):.2f} | "
            f"{int(row['max_tested_concurrency'])} | {int(row['max_feasible_concurrency'])} | "
            f"{float(row['best_throughput_tokens_per_s']):.2f} | {int(row['best_concurrency'])} | "
            f"{float(row['best_latency_ms_p90']):.4f} | {bool(row['has_feasible'])} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def _write_runtime_failures(run_dir: Path, rows: list[dict[str, Any]]) -> Path:
    out_path = run_dir / "runtime_failures.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return out_path


def _timestamp_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def _estimate_remote_ratio(
    *,
    expert_idx_steps_local: torch.Tensor,
    placement: torch.Tensor,
    rank: int,
) -> float:
    expert_cpu = expert_idx_steps_local.detach().to("cpu", dtype=torch.int64)
    placement_cpu = placement.detach().to("cpu", dtype=torch.int64)
    owners = placement_cpu.index_select(0, expert_cpu.reshape(-1))
    if owners.numel() == 0:
        return 0.0
    remote = owners != int(rank)
    return float(remote.to(dtype=torch.float32).mean().item())


def run_moe_service_study(
    *,
    config: MoEStudyConfig,
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
        collectives_rows: list[dict[str, Any]] = []
        runtime_failures: list[dict[str, Any]] = []
        abort_distributed_run = False

        local_mask = get_visible_core_mask()
        rank_masks_raw = [str(local_mask["raw"])]
        if ctx.enabled:
            # Avoid an extra startup collective on unstable runtimes.
            rank_masks_raw = [str(local_mask["raw"]) for _ in range(ctx.world_size)]
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

        for idx, shape in enumerate(config.decode_shapes):
            if abort_distributed_run:
                break
            seed = config.seed + 20000 + idx
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            x_steps_global = _randn(
                (shape.decode_steps, shape.concurrency, shape.model_dim),
                dtype=dtype,
                device=device,
            )
            expert_idx_steps, gates_steps = _generate_router_assignments(
                decode_steps=shape.decode_steps,
                concurrency=shape.concurrency,
                num_experts=shape.num_experts,
                top_k=shape.top_k,
                routing_skew=shape.routing_skew,
                device=device,
            )
            w1, w2 = _make_expert_weights(
                num_experts=shape.num_experts,
                model_dim=shape.model_dim,
                hidden_dim=shape.hidden_dim,
                dtype=dtype,
                device=device,
            )

            ref_out = None
            for step in range(shape.decode_steps):
                ref_out, _, _ = _moe_step_single(
                    x=x_steps_global[step],
                    expert_idx=expert_idx_steps[step],
                    gates=gates_steps[step],
                    w1=w1,
                    w2=w2,
                )
            assert ref_out is not None
            sync_if_needed(device)

            placement_naive = _build_naive_placement(shape.num_experts, ctx.world_size if ctx.enabled else 2, device)
            placement_local = _build_locality_placement(
                expert_idx_steps=expert_idx_steps,
                num_experts=shape.num_experts,
                world_size=ctx.world_size if ctx.enabled else 2,
                device=device,
            )

            for setup in selected_setups:
                if abort_distributed_run:
                    break
                try:
                    if setup == "single_die" and ctx.enabled and not ctx.is_primary:
                        continue

                    if setup == "single_die":
                        local_x_steps = x_steps_global
                        local_expert_idx = expert_idx_steps
                        local_gates = gates_steps

                        def runner() -> tuple[torch.Tensor, CommMetrics, float]:
                            out_last = torch.zeros((shape.concurrency, shape.model_dim), dtype=dtype, device=device)
                            comm_total = CommMetrics()
                            remote_total = 0.0
                            for step in range(shape.decode_steps):
                                out_last, comm_step, remote_step = _moe_step_single(
                                    x=local_x_steps[step],
                                    expert_idx=local_expert_idx[step],
                                    gates=local_gates[step],
                                    w1=w1,
                                    w2=w2,
                                )
                                comm_total.merge_(comm_step)
                                remote_total += float(remote_step)
                            remote_avg = remote_total / float(shape.decode_steps)
                            return out_last, comm_total, remote_avg

                    else:
                        local_batch = shape.concurrency // ctx.world_size
                        b_start = ctx.rank * local_batch
                        b_end = b_start + local_batch
                        local_x_steps = x_steps_global[:, b_start:b_end]
                        local_expert_idx = expert_idx_steps[:, b_start:b_end]
                        local_gates = gates_steps[:, b_start:b_end]
                        placement = placement_naive if setup == "dual_die_moe_naive" else placement_local
                        remote_ratio_expected = _estimate_remote_ratio(
                            expert_idx_steps_local=local_expert_idx,
                            placement=placement,
                            rank=int(ctx.rank),
                        )

                        def runner() -> tuple[torch.Tensor, CommMetrics, float]:
                            out_last = torch.zeros((local_batch, shape.model_dim), dtype=dtype, device=device)
                            comm_total = CommMetrics()
                            for step in range(shape.decode_steps):
                                out_last, comm_step, _ = _moe_step_dual(
                                    x_local=local_x_steps[step],
                                    expert_idx_local=local_expert_idx[step],
                                    gates_local=local_gates[step],
                                    w1=w1,
                                    w2=w2,
                                    placement=placement,
                                    ctx=ctx,
                                    device=device,
                                )
                                comm_total.merge_(comm_step)
                            return out_last, comm_total, float(remote_ratio_expected)


                    if ctx.is_primary:
                        print(
                            "[moe-study] decode "
                            f"setup={setup} concurrency={shape.concurrency} context_len={shape.context_len} "
                            f"decode_steps={shape.decode_steps} model_dim={shape.model_dim} hidden_dim={shape.hidden_dim} "
                            f"experts={shape.num_experts} top_k={shape.top_k} skew={shape.routing_skew}",
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
                        remote_ratio,
                        collective_samples,
                    ) = _bench_runner(
                        runner,
                        warmup_iters=config.warmup_iters,
                        measure_iters=config.measure_iters,
                        device=device,
                    )

                    if setup == "single_die":
                        out_full = out
                    else:
                        out_full = _gather_batch_tensor(out, ctx=ctx, device=device)

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
                        continue

                    p50_ms = _percentile_ms(total_s, 50)
                    p90_ms = _percentile_ms(total_s, 90)
                    tokens = shape.concurrency * shape.decode_steps
                    throughput = tokens / (p50_ms / 1000.0) if p50_ms > 0 else 0.0
                    achieved_link_gbps_p50 = _percentile(link_gbps, 50)
                    fabric_peak = float(fabric_summary.get("peak_gbps", 0.0))
                    link_util_pct = (achieved_link_gbps_p50 / fabric_peak * 100.0) if fabric_peak > 0 else 0.0

                    row = {
                        "timestamp": _timestamp_utc(),
                        "phase": "decode",
                        "setup": setup,
                        "device": device_name,
                        "dtype": config.dtype,
                        "batch": shape.concurrency,
                        "context_len": shape.context_len,
                        "decode_steps": shape.decode_steps,
                        "model_dim": shape.model_dim,
                        "hidden_dim": shape.hidden_dim,
                        "num_experts": shape.num_experts,
                        "top_k": shape.top_k,
                        "routing_skew": shape.routing_skew,
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
                        "remote_dispatch_ratio_p50": round(float(_percentile(remote_ratio, 50)), 6),
                        "max_abs_err": round(float(max_abs), 8),
                        "max_rel_err": round(float(max_rel), 8),
                    }
                    missing = [col for col in REQUIRED_COLUMNS if col not in row]
                    if missing:
                        raise RuntimeError(f"Missing metrics columns: {missing}")
                    records.append(row)
                    collectives_rows.append(
                        {
                            "timestamp": _timestamp_utc(),
                            "phase": "decode",
                            "setup": setup,
                            "batch": shape.concurrency,
                            "context_len": shape.context_len,
                            "decode_steps": shape.decode_steps,
                            "model_dim": shape.model_dim,
                            "hidden_dim": shape.hidden_dim,
                            "num_experts": shape.num_experts,
                            "top_k": shape.top_k,
                            "routing_skew": shape.routing_skew,
                            "ops": _summarize_collective_samples(collective_samples),
                        }
                    )
                except Exception as exc:
                    if not config.continue_on_runtime_error:
                        raise
                    if ctx.is_primary:
                        print(
                            "[moe-study] warning: setup failed and will be skipped: "
                            f"setup={setup} concurrency={shape.concurrency} context_len={shape.context_len} "
                            f"decode_steps={shape.decode_steps} model_dim={shape.model_dim} hidden_dim={shape.hidden_dim} "
                            f"experts={shape.num_experts} top_k={shape.top_k} error={type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        if config.record_runtime_failures:
                            runtime_failures.append(
                                {
                                    "timestamp": _timestamp_utc(),
                                    "phase": "decode",
                                    "setup": setup,
                                    "batch": int(shape.concurrency),
                                    "context_len": int(shape.context_len),
                                    "decode_steps": int(shape.decode_steps),
                                    "model_dim": int(shape.model_dim),
                                    "hidden_dim": int(shape.hidden_dim),
                                    "num_experts": int(shape.num_experts),
                                    "top_k": int(shape.top_k),
                                    "routing_skew": float(shape.routing_skew),
                                    "error_type": type(exc).__name__,
                                    "error_message": str(exc),
                                }
                            )
                    if ctx.enabled:
                        abort_distributed_run = True
                finally:
                    if ctx.enabled and not abort_distributed_run:
                        try:
                            distributed_barrier(ctx)
                        except Exception as barrier_exc:
                            abort_distributed_run = True
                            if ctx.is_primary and config.record_runtime_failures:
                                runtime_failures.append(
                                    {
                                        "timestamp": _timestamp_utc(),
                                        "phase": "decode",
                                        "setup": setup,
                                        "batch": int(shape.concurrency),
                                        "context_len": int(shape.context_len),
                                        "decode_steps": int(shape.decode_steps),
                                        "model_dim": int(shape.model_dim),
                                        "hidden_dim": int(shape.hidden_dim),
                                        "num_experts": int(shape.num_experts),
                                        "top_k": int(shape.top_k),
                                        "routing_skew": float(shape.routing_skew),
                                        "error_type": type(barrier_exc).__name__,
                                        "error_message": f"Post-setup barrier failed: {barrier_exc}",
                                    }
                                )
                            if ctx.is_primary:
                                print(
                                    "[moe-study] warning: distributed barrier failed; aborting remaining shapes/setups",
                                    flush=True,
                                )
                if abort_distributed_run and ctx.is_primary:
                    print(
                        "[moe-study] stopping early due to distributed runtime failure; preserving partial results",
                        flush=True,
                    )

        if ctx.is_primary:
            _write_metrics(run_dir, records)
            _write_collectives_summary(run_dir, collectives_rows)
            decode_slo = _build_decode_slo_summary(records, config.decode_slo_ms)
            _write_decode_slo_summary(run_dir, decode_slo)
            capacity_frontier = _build_capacity_frontier(records, config.capacity_slo_ms)
            _write_capacity_frontier(run_dir, capacity_frontier)
            if config.record_runtime_failures and runtime_failures:
                _write_runtime_failures(run_dir, runtime_failures)

            repo_root = Path(__file__).resolve().parents[3]
            manifest = build_run_manifest(
                repo_root=repo_root,
                benchmark_config_path=config_path.resolve(),
                variant="moe_service_study",
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
                    "capacity_slo_ms": float(config.capacity_slo_ms),
                    "continue_on_runtime_error": bool(config.continue_on_runtime_error),
                    "record_runtime_failures": bool(config.record_runtime_failures),
                    "runtime_failure_count": int(len(runtime_failures)),
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
