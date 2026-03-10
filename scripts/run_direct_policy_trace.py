#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from io_aware_attention.bench.artifacts import (  # noqa: E402
    build_run_manifest,
    create_run_dir,
    write_manifest,
)
from io_aware_attention.experiments import kernel_study as ks  # noqa: E402
from io_aware_attention.experiments.phase_study import (  # noqa: E402
    _decode_step_single,
    _dtype_from_name,
    _gather_batch_tensor,
    _make_weights,
    _percentile_ms,
    _prefill_step_request_sharded,
    _prefill_step_single,
    _randn,
    _split_qkv,
    _timestamp_utc,
    _validate_correctness,
)
from io_aware_attention.runtime.trainium import (  # noqa: E402
    distributed_barrier,
    finalize_distributed_context,
    gather_rank_strings,
    get_visible_core_mask,
    init_distributed_context,
    parse_visible_cores,
    resolve_device,
    sync_if_needed,
)

PolicyName = Literal["single->single", "single->request", "request->request"]
SetupName = Literal["single_die", "dual_die_request_sharded"]

POLICY_ORDER: tuple[PolicyName, ...] = (
    "single->single",
    "single->request",
    "request->request",
)
POLICY_SETUPS: dict[PolicyName, tuple[SetupName, SetupName]] = {
    "single->single": ("single_die", "single_die"),
    "single->request": ("single_die", "dual_die_request_sharded"),
    "request->request": ("dual_die_request_sharded", "dual_die_request_sharded"),
}
POLICY_COLORS = {
    "single->single": "#1f77b4",
    "single->request": "#2ca02c",
    "request->request": "#1b7f3a",
}
POLICY_LABELS = {
    "single->single": "single→single",
    "single->request": "single→request",
    "request->request": "request→request",
}


@dataclass(frozen=True)
class DirectTraceConfig:
    device: str
    dtype: str
    seed: int
    warmup_iters: int
    measure_iters: int
    distributed: bool
    dual_world_size: int
    policies: list[PolicyName]
    request_slo_ms: float
    batch: int
    context_len: int
    output_tokens: int
    model_dim: int
    num_heads: int
    mlp_ratio: int
    enforce_correctness: bool
    correctness_abs_tol: float
    correctness_rel_tol: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DirectTraceConfig":
        trace = raw.get("trace", {})
        policies = [str(item) for item in raw.get("policies", list(POLICY_ORDER))]
        cfg = cls(
            device=str(raw.get("device", "trainium")),
            dtype=str(raw.get("dtype", "fp32")),
            seed=int(raw.get("seed", 0)),
            warmup_iters=int(raw.get("warmup_iters", 1)),
            measure_iters=int(raw.get("measure_iters", 5)),
            distributed=bool(raw.get("distributed", True)),
            dual_world_size=int(raw.get("dual_world_size", 2)),
            policies=policies,  # type: ignore[arg-type]
            request_slo_ms=float(raw["request_slo_ms"]),
            batch=int(trace["batch"]),
            context_len=int(trace["context_len"]),
            output_tokens=int(trace["output_tokens"]),
            model_dim=int(trace["model_dim"]),
            num_heads=int(trace["num_heads"]),
            mlp_ratio=int(trace.get("mlp_ratio", 4)),
            enforce_correctness=bool(raw.get("enforce_correctness", True)),
            correctness_abs_tol=float(raw.get("correctness_abs_tol", 0.02)),
            correctness_rel_tol=float(raw.get("correctness_rel_tol", 0.025)),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.dtype not in {"bf16", "fp32"}:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        if self.batch < 1 or self.context_len < 1 or self.output_tokens < 1:
            raise ValueError("batch, context_len, and output_tokens must be > 0")
        if self.model_dim < 2 or self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if self.model_dim % 2 != 0:
            raise ValueError("model_dim must be divisible by 2")
        if self.warmup_iters < 0 or self.measure_iters < 1:
            raise ValueError("warmup_iters must be >= 0 and measure_iters must be >= 1")
        if self.request_slo_ms <= 0:
            raise ValueError("request_slo_ms must be > 0")
        if self.dual_world_size != 2:
            raise ValueError("This trace currently supports dual_world_size=2 only")
        for policy in self.policies:
            if policy not in POLICY_ORDER:
                raise ValueError(f"Unsupported policy: {policy}")
        if any("request" in policy for policy in self.policies):
            if not self.distributed:
                raise ValueError("Request-sharded policies require distributed=true")
            if self.batch % self.dual_world_size != 0:
                raise ValueError(
                    f"batch must be divisible by dual_world_size for request-sharded policies; got {self.batch}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a directly measured end-to-end policy trace for single->single, "
            "single->request, and request->request."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--device",
        choices=["cpu", "trainium"],
        default=None,
        help="Override execution device.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="Directory where run artifacts are created.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable real 2-rank execution with XLA collectives (launch with torchrun --nproc_per_node=2).",
    )
    return parser.parse_args()


def _load_config(path: Path) -> DirectTraceConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return DirectTraceConfig.from_dict(raw)


def _build_kv_cache(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    *,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_norm = ks._rmsnorm_single(x, weights["gamma"])
    qkv = ks._qkv_single(x_norm, weights["w_qkv"])
    _, k, v = _split_qkv(qkv, num_heads=num_heads)
    return k, v


def _policy_runner(
    *,
    policy: PolicyName,
    config: DirectTraceConfig,
    device: Any,
    ctx: Any,
    dtype: torch.dtype,
    weights: dict[str, torch.Tensor],
    prompt_tokens_global: torch.Tensor,
    decode_tokens_global: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    prefill_setup, decode_setup = POLICY_SETUPS[policy]
    local_batch = config.batch // ctx.world_size if ctx.enabled else config.batch
    batch_start = ctx.rank * local_batch if ctx.enabled else 0
    batch_end = batch_start + local_batch

    full_k_cache: torch.Tensor | None = None
    full_v_cache: torch.Tensor | None = None
    local_k_cache: torch.Tensor | None = None
    local_v_cache: torch.Tensor | None = None

    sync_if_needed(device)
    prefill_start = time.perf_counter()
    if prefill_setup == "single_die":
        _, _, _ = _prefill_step_single(prompt_tokens_global, weights, num_heads=config.num_heads)
        full_k_cache, full_v_cache = _build_kv_cache(prompt_tokens_global, weights, num_heads=config.num_heads)
        if decode_setup == "dual_die_request_sharded":
            local_k_cache = full_k_cache[batch_start:batch_end].clone()
            local_v_cache = full_v_cache[batch_start:batch_end].clone()
    else:
        prompt_tokens_local = prompt_tokens_global[batch_start:batch_end]
        _, _, _ = _prefill_step_request_sharded(prompt_tokens_local, weights, num_heads=config.num_heads)
        local_k_cache, local_v_cache = _build_kv_cache(prompt_tokens_local, weights, num_heads=config.num_heads)
    sync_if_needed(device)
    prefill_s = time.perf_counter() - prefill_start

    sync_if_needed(device)
    decode_start = time.perf_counter()
    if decode_setup == "single_die":
        assert full_k_cache is not None
        assert full_v_cache is not None
        k_cache = full_k_cache.clone()
        v_cache = full_v_cache.clone()
        out = torch.zeros((config.batch, 1, config.model_dim), dtype=dtype, device=device)
        for step in range(config.output_tokens):
            out, k_cache, v_cache, _, _ = _decode_step_single(
                decode_tokens_global[step],
                k_cache,
                v_cache,
                weights,
                num_heads=config.num_heads,
                max_cache_len=config.context_len,
            )
        out_full = out
    else:
        assert local_k_cache is not None
        assert local_v_cache is not None
        decode_tokens_local = decode_tokens_global[:, batch_start:batch_end]
        k_cache = local_k_cache.clone()
        v_cache = local_v_cache.clone()
        out = torch.zeros((local_batch, 1, config.model_dim), dtype=dtype, device=device)
        for step in range(config.output_tokens):
            out, k_cache, v_cache, _, _ = _decode_step_single(
                decode_tokens_local[step],
                k_cache,
                v_cache,
                weights,
                num_heads=config.num_heads,
                max_cache_len=config.context_len,
            )
        out_full = _gather_batch_tensor(out, ctx=ctx, device=device)
    sync_if_needed(device)
    decode_s = time.perf_counter() - decode_start
    return out_full, prefill_s, decode_s


def _build_reference(
    *,
    config: DirectTraceConfig,
    device: Any,
    dtype: torch.dtype,
    weights: dict[str, torch.Tensor],
    prompt_tokens_global: torch.Tensor,
    decode_tokens_global: torch.Tensor,
) -> torch.Tensor:
    _ = _prefill_step_single(prompt_tokens_global, weights, num_heads=config.num_heads)
    k_cache, v_cache = _build_kv_cache(prompt_tokens_global, weights, num_heads=config.num_heads)
    out = torch.zeros((config.batch, 1, config.model_dim), dtype=dtype, device=device)
    for step in range(config.output_tokens):
        out, k_cache, v_cache, _, _ = _decode_step_single(
            decode_tokens_global[step],
            k_cache,
            v_cache,
            weights,
            num_heads=config.num_heads,
            max_cache_len=config.context_len,
        )
    sync_if_needed(device)
    return out


def _plot_summary(
    rows: list[dict[str, Any]],
    *,
    config: DirectTraceConfig,
    out_path: Path,
) -> None:
    labels = [POLICY_LABELS.get(str(row["policy"]), str(row["policy"])) for row in rows]
    prefill_ms = np.array([float(row["prefill_ms_p50"]) for row in rows], dtype=float)
    decode_ms = np.array([float(row["decode_ms_p50"]) for row in rows], dtype=float)
    total_ms = np.array([float(row["latency_ms_p50"]) for row in rows], dtype=float)
    req_per_s = np.array([float(row["requests_per_s_p50"]) for row in rows], dtype=float)
    on_time = np.array([float(row["on_time_ratio"]) * 100.0 for row in rows], dtype=float)
    colors = [POLICY_COLORS.get(str(row["policy"]), "#666666") for row in rows]
    x = np.arange(len(rows), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)

    axes[0].bar(x, prefill_ms, color="#9ecae1", label="prefill")
    axes[0].bar(x, decode_ms, bottom=prefill_ms, color=colors, label="decode")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylabel("Measured request latency (ms)")
    axes[0].set_title("Direct End-to-End Latency Split")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()
    left_pad = max(10.0, float(total_ms.max(initial=0.0)) * 0.03)
    for idx, total in enumerate(total_ms):
        axes[0].text(
            float(x[idx]),
            float(total) + left_pad,
            f"{total:.0f} ms",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    axes[0].set_ylim(0.0, float(total_ms.max(initial=0.0)) + left_pad * 4.0)

    axes[1].bar(x, req_per_s, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_ylabel("Measured requests/s (p50-derived)")
    axes[1].set_title("Direct End-to-End Throughput")
    axes[1].axhline(0.0, color="#333333", linewidth=0.8)
    axes[1].grid(axis="y", alpha=0.25)
    right_pad = max(0.02, float(req_per_s.max(initial=0.0)) * 0.03)
    for idx, value in enumerate(req_per_s):
        axes[1].text(
            float(x[idx]),
            float(value) + right_pad,
            f"{value:.2f} req/s\n{on_time[idx]:.1f}% on-time",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1].set_ylim(0.0, float(req_per_s.max(initial=0.0)) + right_pad * 5.0)

    fig.suptitle(
        (
            "Directly Measured End-to-End Policy Trace "
            f"(batch={config.batch}, context={config.context_len}, output_tokens={config.output_tokens})"
        ),
        y=1.04,
    )
    fig.text(
        0.5,
        -0.01,
        f"On-time fraction uses request_slo_ms={config.request_slo_ms:.1f}.",
        ha="center",
        va="top",
        fontsize=9,
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    config = _load_config(args.config)
    device_name = str(args.device or config.device)
    distributed_requested = bool(args.distributed or config.distributed)

    ctx = init_distributed_context(
        device_name=device_name,
        enable_distributed=distributed_requested,
        expected_world_size=(config.dual_world_size if distributed_requested else None),
    )
    run_root = Path(args.output_dir) / args.config.stem
    run_dir = create_run_dir(run_root) if ctx.is_primary else run_root / "_distributed_non_primary"

    try:
        device = resolve_device(device_name)
        dtype = _dtype_from_name(config.dtype)

        if any("request" in policy for policy in config.policies) and not ctx.enabled:
            raise RuntimeError("Request-sharded policies require torchrun + --distributed.")
        if distributed_requested and not ctx.enabled:
            raise RuntimeError("Distributed execution was requested but distributed context is not enabled.")

        if ctx.enabled:
            distributed_barrier(ctx)

        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        weights = _make_weights(
            model_dim=config.model_dim,
            mlp_ratio=config.mlp_ratio,
            dtype=dtype,
            device=device,
        )
        prompt_tokens_global = _randn(
            (config.batch, config.context_len, config.model_dim),
            dtype=dtype,
            device=device,
        )
        decode_tokens_global = _randn(
            (config.output_tokens, config.batch, 1, config.model_dim),
            dtype=dtype,
            device=device,
        )

        reference_out = (
            _build_reference(
                config=config,
                device=device,
                dtype=dtype,
                weights=weights,
                prompt_tokens_global=prompt_tokens_global,
                decode_tokens_global=decode_tokens_global,
            )
            if ctx.is_primary
            else None
        )
        eps = 1e-3 if config.dtype == "bf16" else 1e-6

        local_mask = get_visible_core_mask()
        rank_masks_raw = [str(local_mask["raw"])]
        if ctx.enabled:
            rank_masks_raw = gather_rank_strings(local_value=str(local_mask["raw"]), ctx=ctx, device=device)
        rank_core_masks = [
            {
                "rank": int(rank),
                "visible_cores_raw": str(raw),
                "visible_cores": parse_visible_cores(None if raw == "all" else str(raw)),
                "chip_id": "unknown",
            }
            for rank, raw in enumerate(rank_masks_raw)
        ]

        summary_rows: list[dict[str, Any]] = []
        raw_samples: dict[str, dict[str, list[float]]] = {}

        for policy in config.policies:
            local_skip = bool(policy == "single->single" and ctx.enabled and not ctx.is_primary)
            if ctx.is_primary:
                prefill_setup, decode_setup = POLICY_SETUPS[policy]
                print(
                    "[direct-trace] "
                    f"policy={policy} prefill={prefill_setup} decode={decode_setup} "
                    f"batch={config.batch} context={config.context_len} output_tokens={config.output_tokens}",
                    flush=True,
                )

            for _ in range(config.warmup_iters):
                if not local_skip:
                    _policy_runner(
                        policy=policy,
                        config=config,
                        device=device,
                        ctx=ctx,
                        dtype=dtype,
                        weights=weights,
                        prompt_tokens_global=prompt_tokens_global,
                        decode_tokens_global=decode_tokens_global,
                    )
                if ctx.enabled:
                    distributed_barrier(ctx)

            total_s: list[float] = []
            prefill_s: list[float] = []
            decode_s: list[float] = []
            on_time_flags: list[float] = []
            max_abs_samples: list[float] = []
            max_rel_samples: list[float] = []

            for _ in range(config.measure_iters):
                if not local_skip:
                    out_full, prefill_elapsed, decode_elapsed = _policy_runner(
                        policy=policy,
                        config=config,
                        device=device,
                        ctx=ctx,
                        dtype=dtype,
                        weights=weights,
                        prompt_tokens_global=prompt_tokens_global,
                        decode_tokens_global=decode_tokens_global,
                    )
                    if ctx.is_primary:
                        assert reference_out is not None
                        max_abs_err, max_rel_err = _validate_correctness(
                            setup="single_die" if policy == "single->single" else "dual_die_request_sharded",
                            reference=reference_out,
                            candidate=out_full,
                            enforce=config.enforce_correctness,
                            abs_tol=config.correctness_abs_tol,
                            rel_tol=config.correctness_rel_tol,
                            eps=eps,
                        )
                        total_elapsed = prefill_elapsed + decode_elapsed
                        total_s.append(float(total_elapsed))
                        prefill_s.append(float(prefill_elapsed))
                        decode_s.append(float(decode_elapsed))
                        on_time_flags.append(1.0 if total_elapsed * 1000.0 <= config.request_slo_ms else 0.0)
                        max_abs_samples.append(float(max_abs_err))
                        max_rel_samples.append(float(max_rel_err))
                if ctx.enabled:
                    distributed_barrier(ctx)

            if not ctx.is_primary:
                continue

            p50_ms = _percentile_ms(total_s, 50)
            p90_ms = _percentile_ms(total_s, 90)
            p50_prefill_ms = _percentile_ms(prefill_s, 50)
            p50_decode_ms = _percentile_ms(decode_s, 50)
            requests_per_s_p50 = config.batch / (p50_ms / 1000.0) if p50_ms > 0 else 0.0
            tokens_per_s_p50 = (config.batch * config.output_tokens) / (p50_ms / 1000.0) if p50_ms > 0 else 0.0

            prefill_setup, decode_setup = POLICY_SETUPS[policy]
            row = {
                "timestamp": _timestamp_utc(),
                "policy": policy,
                "prefill_setup": prefill_setup,
                "decode_setup": decode_setup,
                "device": device_name,
                "dtype": config.dtype,
                "batch": int(config.batch),
                "context_len": int(config.context_len),
                "output_tokens": int(config.output_tokens),
                "request_slo_ms": round(float(config.request_slo_ms), 6),
                "latency_ms_p50": round(float(p50_ms), 6),
                "latency_ms_p90": round(float(p90_ms), 6),
                "prefill_ms_p50": round(float(p50_prefill_ms), 6),
                "decode_ms_p50": round(float(p50_decode_ms), 6),
                "requests_per_s_p50": round(float(requests_per_s_p50), 6),
                "tokens_per_s_p50": round(float(tokens_per_s_p50), 6),
                "on_time_ratio": round(float(np.mean(on_time_flags) if on_time_flags else 0.0), 6),
                "max_abs_err": round(float(max(max_abs_samples) if max_abs_samples else 0.0), 8),
                "max_rel_err": round(float(max(max_rel_samples) if max_rel_samples else 0.0), 8),
            }
            summary_rows.append(row)
            raw_samples[policy] = {
                "total_ms": [float(value) * 1000.0 for value in total_s],
                "prefill_ms": [float(value) * 1000.0 for value in prefill_s],
                "decode_ms": [float(value) * 1000.0 for value in decode_s],
                "on_time": [float(value) for value in on_time_flags],
                "max_abs_err": [float(value) for value in max_abs_samples],
                "max_rel_err": [float(value) for value in max_rel_samples],
            }

        if not ctx.is_primary:
            return 0

        fields = [
            "timestamp",
            "policy",
            "prefill_setup",
            "decode_setup",
            "device",
            "dtype",
            "batch",
            "context_len",
            "output_tokens",
            "request_slo_ms",
            "latency_ms_p50",
            "latency_ms_p90",
            "prefill_ms_p50",
            "decode_ms_p50",
            "requests_per_s_p50",
            "tokens_per_s_p50",
            "on_time_ratio",
            "max_abs_err",
            "max_rel_err",
        ]

        csv_path = run_dir / "direct_policy_trace_summary.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

        md_lines = [
            "# Direct Policy Trace Summary",
            "",
            f"- batch: {config.batch}",
            f"- context_len: {config.context_len}",
            f"- output_tokens: {config.output_tokens}",
            f"- request_slo_ms: {config.request_slo_ms:.2f}",
            "",
            "| Policy | p50 latency (ms) | p90 latency (ms) | Prefill p50 (ms) | Decode p50 (ms) | Requests/s | Tokens/s | On-time % |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in summary_rows:
            md_lines.append(
                f"| {row['policy']} | {float(row['latency_ms_p50']):.2f} | {float(row['latency_ms_p90']):.2f} | "
                f"{float(row['prefill_ms_p50']):.2f} | {float(row['decode_ms_p50']):.2f} | "
                f"{float(row['requests_per_s_p50']):.2f} | {float(row['tokens_per_s_p50']):.2f} | "
                f"{float(row['on_time_ratio']) * 100.0:.2f} |"
            )
        md_path = run_dir / "direct_policy_trace_summary.md"
        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

        raw_path = run_dir / "direct_policy_trace_samples.json"
        raw_path.write_text(json.dumps(raw_samples, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        plot_path = run_dir / "direct_policy_trace.png"
        _plot_summary(summary_rows, config=config, out_path=plot_path)

        manifest = build_run_manifest(
            repo_root=REPO_ROOT,
            benchmark_config_path=args.config,
            variant="direct_policy_trace",
            seed=config.seed,
            distributed_enabled=ctx.enabled,
            distributed_world_size=(ctx.world_size if ctx.enabled else 1),
            distributed_rank=(ctx.rank if ctx.enabled else 0),
            rank_core_masks=rank_core_masks,
        )
        write_manifest(run_dir, manifest)

        print(csv_path)
        print(md_path)
        print(plot_path)
        print(raw_path)
        return 0
    finally:
        finalize_distributed_context(ctx)


if __name__ == "__main__":
    raise SystemExit(main())
