from __future__ import annotations

import math

import torch


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    causal: bool = False,
    tile_n: int = 128,
) -> torch.Tensor:
    """Streaming attention with online softmax state (forward-only)."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Expected q, k, v tensors with shape [B, H, S, D]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"Expected q, k, v to have the same shape, got {q.shape}, {k.shape}, {v.shape}")
    if tile_n < 1:
        raise ValueError("tile_n must be >= 1")

    batch, heads, seq_len, d_model = q.shape
    scale_value = scale if scale is not None else 1.0 / math.sqrt(d_model)

    q_f32 = q.float()
    k_f32 = k.float()
    v_f32 = v.float()

    m_i = torch.full((batch, heads, seq_len), float("-inf"), device=q.device, dtype=torch.float32)
    l_i = torch.zeros((batch, heads, seq_len), device=q.device, dtype=torch.float32)
    acc = torch.zeros((batch, heads, seq_len, d_model), device=q.device, dtype=torch.float32)

    for start in range(0, seq_len, tile_n):
        end = min(start + tile_n, seq_len)
        k_tile = k_f32[:, :, start:end, :]
        v_tile = v_f32[:, :, start:end, :]
        logits = torch.matmul(q_f32, k_tile.transpose(-1, -2)) * scale_value

        if causal:
            q_pos = torch.arange(seq_len, device=q.device).view(1, 1, seq_len, 1)
            k_pos = torch.arange(start, end, device=q.device).view(1, 1, 1, end - start)
            logits = logits.masked_fill(k_pos > q_pos, float("-inf"))

        m_ij = torch.max(logits, dim=-1).values
        m_new = torch.maximum(m_i, m_ij)
        alpha = torch.exp(m_i - m_new)
        p = torch.exp(logits - m_new.unsqueeze(-1))

        l_i = l_i * alpha + p.sum(dim=-1)
        acc = acc * alpha.unsqueeze(-1) + torch.matmul(p, v_tile)
        m_i = m_new

    out = acc / l_i.unsqueeze(-1).clamp_min(1e-9)
    return out.to(q.dtype)

