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


def _safe_exp_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(lhs) & torch.isfinite(rhs)
    delta = torch.where(valid, lhs - rhs, torch.zeros_like(lhs))
    return torch.where(valid, torch.exp(delta), torch.zeros_like(lhs))


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
    reduce_group_k: int,
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

        group_span = max(1, reduce_group_k) * tile_k
        for group_start in range(0, local_seq, group_span):
            group_end = min(group_start + group_span, local_seq)

            # Merge local tiles first, then perform one global max/sum reduce per group.
            m_group_local = torch.full(
                (batch, heads, q_end - q_start),
                neg_inf.item(),
                device=q.device,
                dtype=torch.float32,
            )
            l_group_local = torch.zeros((batch, heads, q_end - q_start), device=q.device, dtype=torch.float32)
            o_group_local = torch.zeros(
                (batch, heads, q_end - q_start, head_dim),
                device=q.device,
                dtype=torch.float32,
            )

            for local_start in range(group_start, group_end, tile_k):
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
                stable = logits - m_tile_local.unsqueeze(-1)
                stable = torch.where(torch.isfinite(stable), stable, torch.full_like(stable, -1e9))
                p_local = torch.exp(stable)

                l_tile_local = p_local.sum(dim=-1)
                o_tile_local = torch.matmul(p_local, v_tile)

                m_local_new = torch.maximum(m_group_local, m_tile_local)
                alpha_local = _safe_exp_diff(m_group_local, m_local_new)
                beta_local = _safe_exp_diff(m_tile_local, m_local_new)
                l_group_local = l_group_local * alpha_local + l_tile_local * beta_local
                o_group_local = (
                    o_group_local * alpha_local.unsqueeze(-1) + o_tile_local * beta_local.unsqueeze(-1)
                )
                m_group_local = m_local_new

            m_group_global = m_group_local.clone()
            reduce_fn(m_group_global, "max")

            rescale = _safe_exp_diff(m_group_local, m_group_global)
            l_group_adj = l_group_local * rescale
            o_group_adj = o_group_local * rescale.unsqueeze(-1)

            l_group_global = l_group_adj.clone()
            reduce_fn(l_group_global, "sum")

            o_group_global = o_group_adj.clone()
            reduce_fn(o_group_global, "sum")

            m_new = torch.maximum(m_prev, m_group_global)
            alpha = _safe_exp_diff(m_prev, m_new)
            beta = _safe_exp_diff(m_group_global, m_new)
            l_prev = l_prev * alpha + l_group_global * beta
            o_prev = o_prev * alpha.unsqueeze(-1) + o_group_global * beta.unsqueeze(-1)
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
    reduce_group_k: int = 1,
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
    if tile_q < 1 or tile_k < 1 or reduce_group_k < 1:
        raise ValueError("tile_q, tile_k, and reduce_group_k must be >= 1")

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
        reduce_group_k=reduce_group_k,
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
    reduce_group_k: int = 1,
    global_k_offset: int = 0,
    reduce_fn: ReduceFn | None = None,
) -> torch.Tensor:
    if tile_q < 1 or tile_k < 1 or reduce_group_k < 1:
        raise ValueError("tile_q, tile_k, and reduce_group_k must be >= 1")
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
        reduce_group_k=reduce_group_k,
        reduce_fn=reduce,
        pipelined=True,
    )
