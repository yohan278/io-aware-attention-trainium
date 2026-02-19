#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from io_aware_attention.bench.runner import run_benchmark  # noqa: E402
from io_aware_attention.config import load_benchmark_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SDPA benchmarks on CPU or Trainium.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to benchmark config YAML.",
    )
    parser.add_argument(
        "--variant",
        choices=["naive", "tiled_online", "tiled_online_dbuffer"],
        default=None,
        help="Override kernel variant.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "trainium"],
        default=None,
        help="Override execution device.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="Directory where run artifacts will be created.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_benchmark_config(args.config)
    run_dir, records = run_benchmark(
        config=config,
        config_path=args.config,
        output_dir=args.output_dir,
        variant_override=args.variant,
        device_override=args.device,
    )
    print(f"Run complete: {run_dir}")
    print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
