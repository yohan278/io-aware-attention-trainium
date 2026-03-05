#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from io_aware_attention.runtime.trainium import get_instance_type, is_trainium_available  # noqa: E402


def main() -> int:
    print("== Trainium Environment Validation ==")
    errors: list[str] = []
    warnings: list[str] = []
    expected_region = os.getenv("EXPECTED_AWS_REGION")
    expected_instance = os.getenv("EXPECTED_INSTANCE_TYPE")

    if sys.version_info < (3, 10):
        errors.append(
            f"Python 3.10+ is required, found {sys.version_info.major}.{sys.version_info.minor}."
        )
    else:
        print(f"[ok] Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    try:
        import torch  # type: ignore

        print(f"[ok] torch: {torch.__version__}")
    except Exception as exc:
        errors.append(f"Unable to import torch: {exc}")
        torch = None

    available, reason = is_trainium_available()
    if available:
        print("[ok] Trainium runtime (torch_xla) is available.")
    else:
        errors.append(
            "Trainium runtime unavailable. Install requirements/trainium.txt or use a Neuron-enabled AMI."
        )
        if reason:
            warnings.append(f"Runtime detail: {reason}")

    instance_type = get_instance_type()
    print(f"[info] Instance type: {instance_type}")
    if instance_type != "unknown" and not instance_type.startswith("trn2"):
        warnings.append(
            f"Expected trn2 for this project, found '{instance_type}'. Runs may not match your benchmark target."
        )
    elif expected_instance and instance_type != "unknown" and instance_type != expected_instance:
        warnings.append(
            f"EXPECTED_INSTANCE_TYPE is '{expected_instance}', found '{instance_type}'."
        )

    aws_region = os.getenv("AWS_REGION")
    if aws_region is None:
        warnings.append("AWS_REGION is not set.")
    elif expected_region and aws_region != expected_region:
        warnings.append(f"AWS_REGION is '{aws_region}', expected '{expected_region}'.")
    elif expected_region:
        print(f"[ok] AWS_REGION={aws_region}")
    else:
        print(f"[info] AWS_REGION={aws_region}")

    visible_cores = os.getenv("NEURON_RT_VISIBLE_CORES")
    if visible_cores is None:
        warnings.append(
            "NEURON_RT_VISIBLE_CORES is not set. Default core selection will be used."
        )
    else:
        print(f"[ok] NEURON_RT_VISIBLE_CORES={visible_cores}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("\nValidation completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
