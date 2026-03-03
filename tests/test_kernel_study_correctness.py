from __future__ import annotations

import torch

from io_aware_attention.experiments.kernel_study import (
    ALL_KERNELS,
    KernelShape,
    run_kernel_once_for_testing,
)


def test_dual_die_modes_match_single_die() -> None:
    shape = KernelShape(batch=1, seq_len=8, model_dim=16, num_heads=4, mlp_ratio=4)

    for kernel in ALL_KERNELS:
        ref = run_kernel_once_for_testing(kernel, "single_die", shape, dtype_name="fp32", seed=13)
        out_naive = run_kernel_once_for_testing(kernel, "dual_die_naive", shape, dtype_name="fp32", seed=13)
        out_opt = run_kernel_once_for_testing(
            kernel,
            "dual_die_optimized",
            shape,
            dtype_name="fp32",
            seed=13,
        )

        assert torch.allclose(out_naive, ref, atol=1e-4, rtol=1e-4), kernel
        assert torch.allclose(out_opt, ref, atol=1e-4, rtol=1e-4), kernel
