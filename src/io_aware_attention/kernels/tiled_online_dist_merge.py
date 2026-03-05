from __future__ import annotations

import math
from typing import Callable

import torch

ReduceFn = Callable[[torch.Tensor, str], None]


def _default_reduce(tensor: torch.Tensor, op_name: str) -> None:
    _ = op_name
    return


def _mask_causal_logits_(
    logits: torch.Tensor,
    *,
    q_global_start: int,
    k_global_start: int,
) -> None:
    q_len = logits.size(-2)
    k_len = logits.size(-1)
    q_pos = torch.arange(q_global_start, q_global_start + q_len, device=logits.device).view(1, 1, q_len, 1)
    k_pos = torch.arange(k_global_start, k_global_start + k_len, device=logits.device).view(1, 1, 1, k_len)
    logits.masked_fill_(k_pos > q_pos, float("-inf"))


def _forward_tiled(
    q: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    *,
    global_k_offset: int,
    scale: float,
    causal: bool,
    tile_q: int,
    tile_k: int,
    reduce_fn: ReduceFn,
    pipelined: bool,
) -> torch.Tensor:
    _ = pipelined  # Explicit schedule knob; current implementation is sync-safe on XLA.
    batch, heads, seq_len, head_dim = q.shape
    local_seq = k_local.size(-2)
    q_f32 = q.float()
    k_f32 = k_local.float()
    v_f32 = v_local.float()

    out = torch.zeros((batch, heads, seq_len, head_dim), device=q.device, dtype=torch.float32)
    neg_inf = torch.tensor(float("-inf"), device=q.device, dtype=torch.float32)

    for q_start in range(0, seq_len, tile_q):
        q_end = min(q_start + tile_q, seq_len)
        q_tile = q_f32[:, :, q_start:q_end, :]

        m_prev = torch.full((batch, heads, q_end - q_start), neg_inf.item(), device=q.device, dtype=torch.float32)
        l_prev = torch.zeros((batch, heads, q_end - q_start), device=q.device, dtype=torch.float32)
        o_prev = torch.zeros((batch, heads, q_end - q_start, head_dim), device=q.device, dtype=torch.float32)

        for local_start in range(0, local_seq, tile_k):
            local_end = min(local_start + tile_k, local_seq)
            k_tile = k_f32[:, :, local_start:local_end, :]
            v_tile = v_f32[:, :, local_start:local_end, :]

            logits = torch.matmul(q_tile, k_tile.transpose(-1, -2)) * scale
            if causal:
                _mask_causal_logits_(
                    logits,
                    q_global_start=q_start,
                    k_global_start=global_k_offset + local_start,
                )

            m_tile_local = torch.max(logits, dim=-1).values
            m_tile_global = m_tile_local.clone()
            reduce_fn(m_tile_global, "max")

            stable = logits - m_tile_global.unsqueeze(-1)
            stable = torch.where(torch.isfinite(stable), stable, torch.full_like(stable, -1e9))
            p_local = torch.exp(stable)

            l_local = p_local.sum(dim=-1)
            l_global = l_local.clone()
            reduce_fn(l_global, "sum")

            o_local = torch.matmul(p_local, v_tile)
            o_global = o_local.clone()
            reduce_fn(o_global, "sum")

            m_new = torch.maximum(m_prev, m_tile_global)
            alpha = torch.exp(m_prev - m_new)
            beta = torch.exp(m_tile_global - m_new)
            l_prev = l_prev * alpha + l_global * beta
            o_prev = o_prev * alpha.unsqueeze(-1) + o_global * beta.unsqueeze(-1)
            m_prev = m_new

        out[:, :, q_start:q_end, :] = o_prev / l_prev.unsqueeze(-1).clamp_min(1e-9)

    return out.to(dtype=q.dtype)


def forward_sync(
    q: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    scale: float | None = None,
    causal: bool = False,
    *,
    tile_q: int = 64,
    tile_k: int = 128,
    global_k_offset: int = 0,
    reduce_fn: ReduceFn | None = None,
) -> torch.Tensor:
    if q.ndim != 4 or k_local.ndim != 4 or v_local.ndim != 4:
        raise ValueError("Expected q/k/v with shape [B, H, S, D]")
    if q.shape[:2] != k_local.shape[:2] or q.shape[:2] != v_local.shape[:2]:
        raise ValueError("q/k/v must match on [B, H]")
    if k_local.shape != v_local.shape:
        raise ValueError("k_local and v_local must share shape")
    if q.shape[-1] != k_local.shape[-1]:
        raise ValueError("head_dim mismatch between q and k_local/v_local")
    if tile_q < 1 or tile_k < 1:
        raise ValueError("tile_q and tile_k must be >= 1")

    scale_value = float(scale) if scale is not None else 1.0 / math.sqrt(float(q.size(-1)))
    reduce = reduce_fn or _default_reduce
    return _forward_tiled(
        q,
        k_local,
        v_local,
        global_k_offset=global_k_offset,
        scale=scale_value,
        causal=causal,
        tile_q=tile_q,
        tile_k=tile_k,
        reduce_fn=reduce,
        pipelined=False,
    )


def forward_pipelined(
    q: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    scale: float | None = None,
    causal: bool = False,
    *,
    tile_q: int = 64,
    tile_k: int = 128,
    global_k_offset: int = 0,
    reduce_fn: ReduceFn | None = None,
) -> torch.Tensor:
    scale_value = float(scale) if scale is not None else 1.0 / math.sqrt(float(q.size(-1)))
    reduce = reduce_fn or _default_reduce
    return _forward_tiled(
        q,
        k_local,
        v_local,
        global_k_offset=global_k_offset,
        scale=scale_value,
        causal=causal,
        tile_q=tile_q,
        tile_k=tile_k,
        reduce_fn=reduce,
        pipelined=True,
    )
