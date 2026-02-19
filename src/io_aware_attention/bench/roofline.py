from __future__ import annotations

from io_aware_attention.config import AttentionShape


def estimate_attention_flops(shape: AttentionShape) -> float:
    """Approximate FLOPs for forward SDPA."""
    b, h, s, d = shape.batch, shape.heads, shape.seq_len, shape.head_dim
    qk = 2.0 * b * h * s * s * d
    av = 2.0 * b * h * s * s * d
    softmax = 5.0 * b * h * s * s
    return qk + av + softmax


def estimate_attention_bytes(
    shape: AttentionShape,
    dtype_bytes: int,
    materialize_logits: bool,
) -> float:
    """Approximate off-chip bytes moved for forward SDPA."""
    b, h, s, d = shape.batch, shape.heads, shape.seq_len, shape.head_dim
    qkv_read = 3.0 * b * h * s * d * dtype_bytes
    out_write = 1.0 * b * h * s * d * dtype_bytes
    logits = 2.0 * b * h * s * s * dtype_bytes if materialize_logits else 0.0
    return qkv_read + out_write + logits


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    if bytes_moved <= 0:
        return 0.0
    return flops / bytes_moved

