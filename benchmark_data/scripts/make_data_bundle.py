#!/usr/bin/env python3
"""Package the PROCESSED training/eval artifacts into one content-hashed bundle.

The cloud-burst harness (docs/v2/CLOUD_BURST_HARNESS.md) never reprocesses data on
a pod -- some raw sources are gone (unimib_shar's acc_labels.npy) and the raw
data/*/sessions tree is huge/ephemeral. Instead we tar the small *processed*
tensors here once, sha256 the archive, and (optionally) push it to Cloudflare R2.
A pod pulls `data/bundle-<hash>.tar.gz`, extracts at the repo root, and trains.

Runnable NOW with no cloud credentials: builds the tarball + manifest locally and
prints the exact upload command. With R2 env vars + `--upload`, it also uploads.

Usage:
    python benchmark_data/scripts/make_data_bundle.py                 # build + hash + manifest
    python benchmark_data/scripts/make_data_bundle.py --upload        # + push to R2 (needs env)

R2 env (S3-compatible):
    R2_BUCKET=halo
    R2_ENDPOINT=https://<accountid>.r2.cloudflarestorage.com
    AWS_ACCESS_KEY_ID=<r2 access key id>
    AWS_SECRET_ACCESS_KEY=<r2 secret>
"""

import argparse
import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "benchmark_data" / "bundles"

# Paths included in the bundle, relative to PROJECT_ROOT. Dirs are added recursively.
# Keep this SELECTIVE: only what training/eval reads, not the embedding caches that
# bloat benchmark_data/processed to ~3 GB.
BUNDLE_PATHS = [
    "benchmark_data/processed/limubert",          # data_20_120 + label_20_120 + mapping.json (+ global_label_mapping.json)
    "benchmark_data/processed/ssl_wearables",      # data_30_180 (gravity-present 30Hz windows)
    "benchmark_data/dataset_config.json",          # train/zero-shot lists + per-dataset activities
    "benchmark_data/eval_v2/labels",               # pre-registered per-dataset vocabularies
    "test_output/baseline_evaluation",             # cached 94-way ConSE heads (optional; lets eval-only pods skip refit)
]

# Files matching these suffixes are excluded even under an included dir.
EXCLUDE_SUFFIXES = (".tmp", ".lock")


def _iter_files(rel: str):
    p = PROJECT_ROOT / rel
    if not p.exists():
        print(f"  [skip] {rel} (absent)")
        return
    if p.is_file():
        yield p
        return
    for f in sorted(p.rglob("*")):
        if f.is_file() and not f.name.endswith(EXCLUDE_SUFFIXES):
            yield f


def build_bundle():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    staging = OUT_DIR / "bundle.tar.gz.staging"

    files, total = [], 0
    print("Collecting processed artifacts:")
    for rel in BUNDLE_PATHS:
        n0, sz0 = len(files), total
        for f in _iter_files(rel):
            files.append(f); total += f.stat().st_size
        if len(files) > n0:
            print(f"  [add]  {rel:44} {len(files)-n0:5d} files  {(total-sz0)/1e6:8.1f} MB")
    if not files:
        sys.exit("No files collected -- nothing to bundle.")

    # Deterministic tar (sorted, fixed mtime/uid/gid) so identical data -> identical hash.
    def _reset(ti: tarfile.TarInfo):
        ti.mtime = 0; ti.uid = ti.gid = 0; ti.uname = ti.gname = ""
        return ti

    print(f"\nWriting {staging} ({total/1e6:.1f} MB raw, gzip)...")
    with tarfile.open(staging, "w:gz", compresslevel=6) as tar:
        for f in files:
            tar.add(f, arcname=str(f.relative_to(PROJECT_ROOT)), filter=_reset)

    print("Hashing archive (sha256)...")
    h = hashlib.sha256()
    with open(staging, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    short = digest[:12]

    bundle = OUT_DIR / f"bundle-{short}.tar.gz"
    staging.replace(bundle)
    comp = bundle.stat().st_size

    manifest = {
        "bundle": bundle.name,
        "sha256": digest,
        "raw_bytes": total,
        "compressed_bytes": comp,
        "n_files": len(files),
        "included_paths": BUNDLE_PATHS,
        "extract_at": "<repo-root>",
        "note": "Extract with `tar xzf bundle-<hash>.tar.gz` at the HALO repo root.",
    }
    (OUT_DIR / f"bundle-{short}.manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT_DIR / "bundle-latest.txt").write_text(bundle.name + "\n")

    print(f"\nBundle: {bundle}")
    print(f"  {len(files)} files | raw {total/1e6:.1f} MB -> gz {comp/1e6:.1f} MB "
          f"({comp/total*100:.0f}%) | sha256 {short}")
    return bundle, manifest


def upload_r2(bundle: Path, manifest: dict):
    bucket = os.environ.get("R2_BUCKET")
    endpoint = os.environ.get("R2_ENDPOINT")
    prefix = os.environ.get("R2_PREFIX", "halo").strip("/")   # namespace under halo/ in the shared bucket
    key = f"{prefix}/data/{bundle.name}"
    cmd = (f"aws s3 cp {bundle} s3://{bucket}/{key} "
           f"--endpoint-url {endpoint}")
    if not (bucket and endpoint and os.environ.get("AWS_ACCESS_KEY_ID")):
        print("\n[--upload requested but R2 env not set] Run one of:")
        print(f"  {cmd}")
        print(f"  rclone copy {bundle} r2:{bucket}/data/    # if rclone 'r2' remote configured")
        return
    try:
        import boto3  # noqa
    except ImportError:
        print("\nboto3 not installed; upload manually:")
        print(f"  {cmd}")
        return
    import boto3
    s3 = boto3.client("s3", endpoint_url=endpoint,
                      aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                      aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"])
    print(f"\nUploading -> s3://{bucket}/{key} ...")
    s3.upload_file(str(bundle), bucket, key)
    s3.put_object(Bucket=bucket, Key=f"{prefix}/data/bundle-latest.txt",
                  Body=(bundle.name + "\n").encode())
    print(f"Uploaded bundle + {prefix}/data/bundle-latest.txt pointer.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upload", action="store_true", help="push the bundle to R2 (needs R2_* env)")
    args = ap.parse_args()
    bundle, manifest = build_bundle()
    if args.upload:
        upload_r2(bundle, manifest)
    else:
        print("\n(no --upload) to push to R2 later:")
        print(f"  R2_BUCKET=… R2_ENDPOINT=… AWS_ACCESS_KEY_ID=… AWS_SECRET_ACCESS_KEY=… \\")
        print(f"  python benchmark_data/scripts/make_data_bundle.py --upload")


if __name__ == "__main__":
    main()
