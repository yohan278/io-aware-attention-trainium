from __future__ import annotations

from pathlib import Path

from io_aware_attention.experiments.kernel_study import load_kernel_study_config


def test_load_kernel_study_smoke_config() -> None:
    cfg = load_kernel_study_config(Path("configs/experiments/smoke.yaml"))
    assert cfg.device == "cpu"
    assert cfg.dtype == "fp32"
    assert cfg.kernels == ["qkv_proj", "attention", "mlp", "rmsnorm", "out_proj"]
    assert cfg.setups == ["single_die", "dual_die_naive", "dual_die_optimized"]
    assert cfg.distributed is False
    assert cfg.dual_world_size == 2
    assert cfg.enable_fabric_calibration is False
    assert cfg.attention_tile_q == 64
    assert cfg.attention_tile_k == 128
    assert cfg.attention_reduce_group_k == 1
    assert cfg.enforce_correctness is True
    assert len(cfg.shapes) == 1
