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

from io_aware_attention.experiments.moe_service_study import (  # noqa: E402
    ALL_SETUPS,
    load_moe_study_config,
    run_moe_service_study,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MoE service-scale validation study on Trainium: router dispatch + "
            "expert MLP with single-die and dual-die setups."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to MoE study config YAML.")
    parser.add_argument(
        "--device",
        choices=["cpu", "trainium"],
        default=None,
        help="Override execution device.",
    )
    parser.add_argument(
        "--setups",
        nargs="+",
        choices=list(ALL_SETUPS),
        default=None,
        help="Subset of setups to run. Defaults to setups from config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="Directory where run artifacts are created.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable 2-rank execution (launch with torchrun --nproc_per_node=2).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_moe_study_config(args.config)
    run_dir, records = run_moe_service_study(
        config=config,
        config_path=args.config,
        output_dir=args.output_dir,
        device_override=args.device,
        setups_override=args.setups,
        distributed_override=args.distributed,
    )
    if records:
        print(f"MoE service study run complete: {run_dir}")
        print(f"Record count: {len(records)}")
        print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
