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

from io_aware_attention.experiments.phase_study import (  # noqa: E402
    ALL_SETUPS,
    load_phase_study_config,
    run_phase_study,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run phase-aware Trn2 study: prefill and decode for single-die, "
            "dual tensor-parallel, and dual request-sharded setups."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to phase study config YAML.")
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
        help="Enable real 2-rank execution with XLA collectives (launch with torchrun --nproc_per_node=2).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_phase_study_config(args.config)
    run_dir, records = run_phase_study(
        config=config,
        config_path=args.config,
        output_dir=args.output_dir,
        device_override=args.device,
        setups_override=args.setups,
        distributed_override=args.distributed,
    )
    if records:
        print(f"Phase study run complete: {run_dir}")
        print(f"Record count: {len(records)}")
        print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
