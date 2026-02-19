from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from io_aware_attention.bench.artifacts import (
    REQUIRED_METRIC_COLUMNS,
    build_run_manifest,
    create_run_dir,
    write_manifest,
    write_metrics,
)
from io_aware_attention.bench.roofline import (
    arithmetic_intensity,
    estimate_attention_bytes,
    estimate_attention_flops,
)
from io_aware_attention.config import BenchmarkConfig
from io_aware_attention.kernels.factory import get_kernel
from io_aware_attention.runtime.trainium import mark_step_if_needed, resolve_device, sync_if_needed


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype name: {name}")


def _dtype_bytes(dtype: torch.dtype) -> int:
    if dtype == torch.float32:
        return 4
    if dtype == torch.bfloat16:
        return 2
    raise ValueError(f"Unsupported dtype for byte estimate: {dtype}")


def _percentile_ms(latencies_seconds: list[float], percentile: float) -> float:
    if not latencies_seconds:
        return 0.0
    return float(np.percentile(np.array(latencies_seconds) * 1000.0, percentile))


def _max_relative_error(reference: torch.Tensor, output: torch.Tensor) -> float:
    denom = reference.abs().clamp_min(1e-9)
    return float(((reference - output).abs() / denom).max().item())


def run_benchmark(
    config: BenchmarkConfig,
    config_path: Path,
    output_dir: Path,
    variant_override: str | None = None,
    device_override: str | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    variant = variant_override or config.variant
    device_name = device_override or config.device
    dtype = _dtype_from_name(config.dtype)

    device = resolve_device(device_name)
    kernel = get_kernel(variant)  # target kernel
    reference_kernel = get_kernel("naive")
    run_dir = create_run_dir(output_dir)
    records: list[dict[str, Any]] = []

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():  # pragma: no cover - not used for main path
        torch.cuda.manual_seed_all(config.seed)

    for shape in config.shapes:
        base_q = torch.randn((shape.batch, shape.heads, shape.seq_len, shape.head_dim), dtype=torch.float32)
        base_k = torch.randn((shape.batch, shape.heads, shape.seq_len, shape.head_dim), dtype=torch.float32)
        base_v = torch.randn((shape.batch, shape.heads, shape.seq_len, shape.head_dim), dtype=torch.float32)

        q = base_q.to(dtype=dtype, device=device)
        k = base_k.to(dtype=dtype, device=device)
        v = base_v.to(dtype=dtype, device=device)

        for _ in range(config.warmup_iters):
            _ = kernel(q, k, v, None, config.causal)
            mark_step_if_needed(device)
        sync_if_needed(device)

        latencies: list[float] = []
        output = None
        for _ in range(config.measure_iters):
            start = time.perf_counter()
            output = kernel(q, k, v, None, config.causal)
            mark_step_if_needed(device)
            sync_if_needed(device)
            end = time.perf_counter()
            latencies.append(end - start)

        assert output is not None
        reference = reference_kernel(q, k, v, None, config.causal)
        sync_if_needed(device)

        output_cpu = output.detach().to("cpu", dtype=torch.float32)
        reference_cpu = reference.detach().to("cpu", dtype=torch.float32)
        max_abs_err = float((reference_cpu - output_cpu).abs().max().item())
        max_rel_err = _max_relative_error(reference_cpu, output_cpu)

        p50_ms = _percentile_ms(latencies, 50)
        p90_ms = _percentile_ms(latencies, 90)
        throughput = (shape.batch * shape.seq_len) / (p50_ms / 1000.0) if p50_ms > 0 else 0.0
        estimated_flops = estimate_attention_flops(shape)
        materialize_logits = variant == "naive"
        estimated_bytes = estimate_attention_bytes(
            shape, dtype_bytes=_dtype_bytes(dtype), materialize_logits=materialize_logits
        )

        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "variant": variant,
            "device": str(device_name),
            "dtype": config.dtype,
            "batch": shape.batch,
            "heads": shape.heads,
            "seq_len": shape.seq_len,
            "head_dim": shape.head_dim,
            "latency_ms_p50": round(p50_ms, 6),
            "latency_ms_p90": round(p90_ms, 6),
            "throughput_tokens_per_s": round(float(throughput), 6),
            "estimated_flops": round(float(estimated_flops), 2),
            "estimated_bytes": round(float(estimated_bytes), 2),
            "arithmetic_intensity": round(float(arithmetic_intensity(estimated_flops, estimated_bytes)), 6),
            "max_abs_err": round(float(max_abs_err), 8),
            "max_rel_err": round(float(max_rel_err), 8),
        }
        missing_cols = [col for col in REQUIRED_METRIC_COLUMNS if col not in record]
        if missing_cols:
            raise RuntimeError(f"Metric record missing required columns: {missing_cols}")
        records.append(record)

    write_metrics(run_dir, records)

    repo_root = Path(__file__).resolve().parents[3]
    manifest = build_run_manifest(
        repo_root=repo_root,
        benchmark_config_path=config_path.resolve(),
        variant=variant,
        seed=config.seed,
    )
    write_manifest(run_dir, manifest)

    return run_dir, records

