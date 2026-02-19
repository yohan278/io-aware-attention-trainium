#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import urlparse


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    return bucket, prefix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload benchmark run artifacts to S3.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to a run directory under results/.")
    parser.add_argument(
        "--s3-uri",
        default=os.getenv("S3_URI"),
        help="Destination S3 prefix, e.g. s3://bucket/path. Falls back to S3_URI env var.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")
    if not args.s3_uri:
        raise SystemExit("Missing --s3-uri and S3_URI environment variable.")

    try:
        import boto3  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "boto3 is not installed. Install requirements/base.txt before syncing to S3."
        ) from exc

    bucket, prefix = _parse_s3_uri(args.s3_uri)
    client = boto3.client("s3")
    uploaded = 0

    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(run_dir).as_posix()
        key = f"{prefix.rstrip('/')}/{run_dir.name}/{rel}" if prefix else f"{run_dir.name}/{rel}"
        try:
            client.upload_file(str(path), bucket, key)
            uploaded += 1
            print(f"Uploaded: {path} -> s3://{bucket}/{key}")
        except Exception as exc:
            # Local artifacts remain untouched regardless of upload failures.
            print(f"Upload failed for {path}: {exc}")

    print(f"Sync complete. Uploaded {uploaded} files from {run_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
