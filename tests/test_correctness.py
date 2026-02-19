from __future__ import annotations

import torch

from io_aware_attention.kernels.naive_sdpa import forward as naive_forward
from io_aware_attention.kernels.tiled_online_softmax import forward as tiled_forward


def test_tiled_online_matches_naive_forward_fp32() -> None:
    torch.manual_seed(7)
    q = torch.randn((1, 2, 16, 32), dtype=torch.float32)
    k = torch.randn((1, 2, 16, 32), dtype=torch.float32)
    v = torch.randn((1, 2, 16, 32), dtype=torch.float32)

    ref = naive_forward(q, k, v, causal=False)
    out = tiled_forward(q, k, v, causal=False, tile_n=8)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


def test_tiled_online_matches_naive_forward_bf16() -> None:
    torch.manual_seed(11)
    q = torch.randn((1, 2, 16, 32), dtype=torch.float32).to(torch.bfloat16)
    k = torch.randn((1, 2, 16, 32), dtype=torch.float32).to(torch.bfloat16)
    v = torch.randn((1, 2, 16, 32), dtype=torch.float32).to(torch.bfloat16)

    ref = naive_forward(q, k, v, causal=True).float()
    out = tiled_forward(q, k, v, causal=True, tile_n=8).float()
    assert torch.allclose(out, ref, atol=5e-2, rtol=5e-2)

