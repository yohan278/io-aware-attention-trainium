from __future__ import annotations

import importlib.metadata
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import torch


def _load_xla_model():
    # Workaround for some Neuron runtime builds where importing torch_xla first
    # can fail with `KeyError: 'neuronxcc.nki.compiler'` unless the submodule is
    # materialized in sys.modules first.
    try:
        import neuronxcc.nki.compiler  # type: ignore # noqa: F401
    except Exception:
        pass
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised on Trainium host
        raise RuntimeError(
            "Unable to import torch_xla. Install trainium dependencies from requirements/trainium.txt."
        ) from exc
    return xm


def is_trainium_available() -> tuple[bool, str | None]:
    try:
        xm = _load_xla_model()
        _ = xm.xla_device()
        return True, None
    except Exception as exc:
        return False, str(exc)


def resolve_device(device_name: str) -> Any:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name != "trainium":
        raise ValueError(f"Unsupported device '{device_name}'")
    xm = _load_xla_model()
    return xm.xla_device()


def mark_step_if_needed(device: Any) -> None:
    if "xla" not in str(device):
        return
    xm = _load_xla_model()
    xm.mark_step()


def sync_if_needed(device: Any) -> None:
    if "xla" not in str(device):
        return
    xm = _load_xla_model()
    xm.mark_step()
    wait_device_ops = getattr(xm, "wait_device_ops", None)
    if callable(wait_device_ops):
        wait_device_ops()


def get_instance_type(timeout_seconds: float = 0.2) -> str:
    token_url = "http://169.254.169.254/latest/api/token"
    metadata_url = "http://169.254.169.254/latest/meta-data/instance-type"
    headers = {"X-aws-ec2-metadata-token-ttl-seconds": "60"}
    token_request = urllib.request.Request(token_url, method="PUT", headers=headers)

    try:
        with urllib.request.urlopen(token_request, timeout=timeout_seconds) as token_response:
            token = token_response.read().decode("utf-8")
        metadata_request = urllib.request.Request(
            metadata_url, headers={"X-aws-ec2-metadata-token": token}
        )
        with urllib.request.urlopen(metadata_request, timeout=timeout_seconds) as response:
            return response.read().decode("utf-8")
    except (OSError, urllib.error.URLError):
        return os.getenv("INSTANCE_TYPE", "unknown")


def get_torch_neuronx_version() -> str:
    try:
        return importlib.metadata.version("torch-neuronx")
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def parse_visible_cores(raw: str | None) -> list[int]:
    """Parse NEURON_RT_VISIBLE_CORES syntax like '0-3,8,10-11'."""
    if raw is None:
        return []
    cores: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_raw, end_raw = token.split("-", maxsplit=1)
            start = int(start_raw.strip())
            end = int(end_raw.strip())
            if end < start:
                start, end = end, start
            cores.extend(range(start, end + 1))
        else:
            cores.append(int(token))
    return sorted(set(cores))


def get_visible_core_mask() -> dict[str, Any]:
    raw = os.getenv("NEURON_RT_VISIBLE_CORES")
    return {
        "raw": raw or "all",
        "parsed": parse_visible_cores(raw),
    }


def _encode_string_tensor(value: str, *, max_len: int, device: Any) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = value.encode("utf-8", errors="ignore")[:max_len]
    length = torch.tensor([len(encoded)], dtype=torch.int32, device=device)
    payload = torch.zeros((max_len,), dtype=torch.int32, device=device)
    if encoded:
        payload[: len(encoded)] = torch.tensor(list(encoded), dtype=torch.int32, device=device)
    return length, payload


def _decode_string_tensor(length: torch.Tensor, payload: torch.Tensor) -> str:
    size = int(max(0, length.item()))
    values = payload[:size].detach().to("cpu", dtype=torch.int32).tolist()
    return bytes(int(x) for x in values).decode("utf-8", errors="ignore")


def gather_rank_strings(
    *,
    local_value: str,
    ctx: "DistributedContext",
    device: Any,
    max_len: int = 256,
) -> list[str]:
    """Gather short strings across ranks without using all_gather_object."""
    if not ctx.enabled:
        return [local_value]
    import torch.distributed as dist

    length, payload = _encode_string_tensor(local_value, max_len=max_len, device=device)
    lengths = [torch.zeros_like(length) for _ in range(ctx.world_size)]
    payloads = [torch.zeros_like(payload) for _ in range(ctx.world_size)]
    dist.all_gather(lengths, length)
    dist.all_gather(payloads, payload)
    mark_step_if_needed(device)
    sync_if_needed(device)
    return [_decode_string_tensor(length_item, payload_item) for length_item, payload_item in zip(lengths, payloads)]


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    backend: str = "none"

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


def init_distributed_context(
    *,
    device_name: str,
    enable_distributed: bool,
    expected_world_size: int | None = None,
) -> DistributedContext:
    rank = _env_int("RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    local_rank = _env_int("LOCAL_RANK", rank)

    if not enable_distributed:
        return DistributedContext(enabled=False, rank=rank, world_size=world_size, local_rank=local_rank)
    if world_size < 2:
        raise RuntimeError(
            "Distributed mode requested but WORLD_SIZE < 2. Launch with torchrun --nproc_per_node=2."
        )

    try:
        import torch.distributed as dist
    except Exception as exc:  # pragma: no cover - depends on runtime install
        raise RuntimeError("torch.distributed is unavailable in this environment.") from exc

    backend = "gloo"
    init_method = "env://"
    if device_name == "trainium":
        try:
            import torch_xla.distributed.xla_backend  # type: ignore # noqa: F401
        except Exception as exc:  # pragma: no cover - exercised on Trainium host
            raise RuntimeError(
                "Failed to import torch_xla.distributed.xla_backend for XLA collectives."
            ) from exc
        backend = "xla"
        init_method = "xla://"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method=init_method)

    resolved_rank = int(dist.get_rank())
    resolved_world_size = int(dist.get_world_size())
    if expected_world_size is not None and resolved_world_size != expected_world_size:
        raise RuntimeError(
            f"Expected distributed world_size={expected_world_size}, got {resolved_world_size}."
        )
    return DistributedContext(
        enabled=True,
        rank=resolved_rank,
        world_size=resolved_world_size,
        local_rank=local_rank,
        backend=backend,
    )


def distributed_barrier(ctx: DistributedContext) -> None:
    if not ctx.enabled:
        return
    import torch.distributed as dist

    dist.barrier()


def finalize_distributed_context(ctx: DistributedContext) -> None:
    if not ctx.enabled:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()
