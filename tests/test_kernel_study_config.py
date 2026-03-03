from __future__ import annotations

from pathlib import Path

from io_aware_attention.experiments.kernel_study import load_kernel_study_config


def test_load_kernel_study_smoke_config() -> None:
    cfg = load_kernel_study_config(Path("configs/experiments/smoke.yaml"))
    assert cfg.device == "cpu"
    assert cfg.dtype == "fp32"
    assert cfg.kernels == ["qkv_proj", "attention", "mlp", "rmsnorm", "out_proj"]
    assert cfg.setups == ["single_die", "dual_die_naive", "dual_die_optimized"]
    assert len(cfg.shapes) == 1
