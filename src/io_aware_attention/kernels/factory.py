from __future__ import annotations

from typing import Callable

import torch

from io_aware_attention.config import KernelVariant
from io_aware_attention.kernels.naive_sdpa import forward as naive_forward
from io_aware_attention.kernels.tiled_online_dist_merge import (
    forward_pipelined as tiled_online_dist_merge_pipelined_forward,
)
from io_aware_attention.kernels.tiled_online_dist_merge import (
    forward_sync as tiled_online_dist_merge_sync_forward,
)
from io_aware_attention.kernels.tiled_online_softmax import forward as tiled_online_forward

KernelFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float | None, bool], torch.Tensor]


def get_kernel(variant: KernelVariant) -> KernelFn:
    registry: dict[str, KernelFn] = {
        "naive": naive_forward,
        "tiled_online": tiled_online_forward,
        # Placeholder for future scheduling specialization.
        "tiled_online_dbuffer": tiled_online_forward,
        "tiled_online_dist_merge_sync": tiled_online_dist_merge_sync_forward,
        "tiled_online_dist_merge_pipelined": tiled_online_dist_merge_pipelined_forward,
    }
    try:
        return registry[variant]
    except KeyError as exc:
        supported = ", ".join(sorted(registry))
        raise ValueError(f"Unknown variant '{variant}'. Supported variants: {supported}") from exc
