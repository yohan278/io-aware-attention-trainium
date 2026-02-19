from __future__ import annotations

import importlib.metadata
import os
import urllib.error
import urllib.request
from typing import Any

import torch


def _load_xla_model():
    try:
        import torch_xla.core.xla_model as xm  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised on Trainium host
        raise RuntimeError(
            "Unable to import torch_xla. Install trainium dependencies from requirements/trainium.txt."
        ) from exc
    return xm


def is_trainium_available() -> tuple[bool, str | None]:
    try:
        xm = _load_xla_model()
        _ = xm.xla_device()
        return True, None
    except Exception as exc:
        return False, str(exc)


def resolve_device(device_name: str) -> Any:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name != "trainium":
        raise ValueError(f"Unsupported device '{device_name}'")
    xm = _load_xla_model()
    return xm.xla_device()


def mark_step_if_needed(device: Any) -> None:
    if "xla" not in str(device):
        return
    xm = _load_xla_model()
    xm.mark_step()


def sync_if_needed(device: Any) -> None:
    if "xla" not in str(device):
        return
    xm = _load_xla_model()
    xm.mark_step()
    wait_device_ops = getattr(xm, "wait_device_ops", None)
    if callable(wait_device_ops):
        wait_device_ops()


def get_instance_type(timeout_seconds: float = 0.2) -> str:
    token_url = "http://169.254.169.254/latest/api/token"
    metadata_url = "http://169.254.169.254/latest/meta-data/instance-type"
    headers = {"X-aws-ec2-metadata-token-ttl-seconds": "60"}
    token_request = urllib.request.Request(token_url, method="PUT", headers=headers)

    try:
        with urllib.request.urlopen(token_request, timeout=timeout_seconds) as token_response:
            token = token_response.read().decode("utf-8")
        metadata_request = urllib.request.Request(
            metadata_url, headers={"X-aws-ec2-metadata-token": token}
        )
        with urllib.request.urlopen(metadata_request, timeout=timeout_seconds) as response:
            return response.read().decode("utf-8")
    except (OSError, urllib.error.URLError):
        return os.getenv("INSTANCE_TYPE", "unknown")


def get_torch_neuronx_version() -> str:
    try:
        return importlib.metadata.version("torch-neuronx")
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"

