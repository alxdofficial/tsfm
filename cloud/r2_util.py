"""Shared Cloudflare R2 (S3-compatible) helpers for the cloud-burst harness.

Reads credentials from the environment (source cloud/.env first: `set -a; . cloud/.env; set +a`).
All object keys are namespaced under the R2_PREFIX (default "halo") inside R2_BUCKET.

Env: R2_BUCKET, R2_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, R2_PREFIX (opt).
"""

import os
import sys
from pathlib import Path


def prefix() -> str:
    return os.environ.get("R2_PREFIX", "halo").strip("/")


def key(*parts: str) -> str:
    return "/".join([prefix(), *[p.strip("/") for p in parts]])


def client():
    """boto3 S3 client for R2, or exit with a clear message if creds/deps are missing."""
    missing = [v for v in ("R2_BUCKET", "R2_ENDPOINT", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
               if not os.environ.get(v)]
    if missing:
        sys.exit(f"R2 env not set: {missing}. Run: set -a; . cloud/.env; set +a")
    try:
        import boto3
    except ImportError:
        sys.exit("boto3 not installed (pip install boto3).")
    return boto3.client(
        "s3", endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def bucket() -> str:
    return os.environ["R2_BUCKET"]


def head(s3, k: str):
    """Return object metadata dict, or None if the key does not exist."""
    try:
        return s3.head_object(Bucket=bucket(), Key=k)
    except Exception:
        return None


def upload(s3, local: Path, k: str, quiet: bool = False):
    size = Path(local).stat().st_size
    if not quiet:
        print(f"  -> s3://{bucket()}/{k}  ({size/1e6:.1f} MB)")
    s3.upload_file(str(local), bucket(), k)
    return size


def put_text(s3, k: str, text: str):
    s3.put_object(Bucket=bucket(), Key=k, Body=text.encode())


def list_prefix(s3, sub: str = ""):
    """List (key, size) under the halo/<sub> prefix."""
    p = key(sub) if sub else prefix() + "/"
    out, token = [], None
    while True:
        kw = dict(Bucket=bucket(), Prefix=p)
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        out += [(o["Key"], o["Size"]) for o in r.get("Contents", [])]
        if not r.get("IsTruncated"):
            break
        token = r["NextContinuationToken"]
    return out
