from __future__ import annotations

import math

import torch


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Reference SDPA implementation that materializes attention logits."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Expected q, k, v tensors with shape [B, H, S, D]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"Expected q, k, v to have the same shape, got {q.shape}, {k.shape}, {v.shape}")

    d_model = q.size(-1)
    scale_value = scale if scale is not None else 1.0 / math.sqrt(d_model)
    logits = torch.matmul(q, k.transpose(-1, -2)) * scale_value

    if causal:
        seq_len = q.size(-2)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
        logits = logits.masked_fill(mask, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    return torch.matmul(probs, v)

