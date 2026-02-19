#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from io_aware_attention.bench.runner import run_benchmark  # noqa: E402
from io_aware_attention.config import load_benchmark_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Trainium benchmark with profiling-friendly defaults.")
    parser.add_argument("--config", type=Path, required=True, help="Path to benchmark YAML config.")
    parser.add_argument(
        "--variant",
        choices=["naive", "tiled_online", "tiled_online_dbuffer"],
        default="tiled_online",
        help="Kernel variant to run for profiling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="Directory where profiling run artifacts will be created.",
    )
    parser.add_argument(
        "--set-neuron-profile-env",
        action="store_true",
        help="Set NEURON_PROFILE=1 for this process before running.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.set_neuron_profile_env:
        os.environ["NEURON_PROFILE"] = "1"

    config = load_benchmark_config(args.config)
    run_dir, _ = run_benchmark(
        config=config,
        config_path=args.config,
        output_dir=args.output_dir,
        variant_override=args.variant,
        device_override="trainium",
    )
    print(f"Trainium profiling run complete: {run_dir}")
    print("Tip: run neuron tooling on this run to inspect stalls and overlap behavior.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
