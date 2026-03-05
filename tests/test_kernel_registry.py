from __future__ import annotations

import pytest

from io_aware_attention.kernels.factory import get_kernel


def test_known_kernels_are_registered() -> None:
    assert callable(get_kernel("naive"))
    assert callable(get_kernel("tiled_online"))
    assert callable(get_kernel("tiled_online_dbuffer"))
    assert callable(get_kernel("tiled_online_dist_merge_sync"))
    assert callable(get_kernel("tiled_online_dist_merge_pipelined"))


def test_unknown_kernel_variant_raises() -> None:
    with pytest.raises(ValueError):
        get_kernel("invalid")  # type: ignore[arg-type]
