#!/usr/bin/env python3
"""Pod-side recipe applier — reads cloud/recipes.json for one job and:

  --install               pip-install the job's extra deps + pull its R2 backbone (crosshar/limubert).
                          (ssl-wearables via torch.hub, UniMTS/NormWear/TinyLlama via HF download on
                          first use — no action needed here.)
  --print-cmd             print the job's train_cmd (bash expands env like $MODEL_SIZE).
  --sync-results <DEST>   upload the job's result files (recipe 'results' globs) to the R2 DEST key prefix.

Called by cloud/bootstrap.sh. Runs on the pod, where cloud/.env-style env is already exported and
boto3 is installed. Source cloud/.env when running locally.
"""

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cloud"))
import r2_util as R2  # noqa: E402

RECIPES = json.load(open(ROOT / "cloud" / "recipes.json"))

# baseline backbones mirrored to R2 -> the local path each eval script expects.
BACKBONE_R2_TO_LOCAL = {
    "crosshar": ("ckpt/baselines/crosshar/model_masked_6_1.pt",
                 "auxiliary_repos/CrossHAR/saved/pretrain_base_combined_train_20_120/model_masked_6_1.pt"),
    "limubert": ("ckpt/baselines/limubert/pretrained_combined.pt",
                 "auxiliary_repos/LIMU-BERT-Public/saved/pretrain_base_combined_train_20_120/pretrained_combined.pt"),
}


def job_spec(job: str) -> dict:
    j = RECIPES["jobs"].get(job)
    if j is None:
        sys.exit(f"unknown job '{job}' (known: {sorted(RECIPES['jobs'])})")
    return j


def do_install(job: str, spec: dict):
    pip = spec.get("pip", [])
    if pip:
        print(f"[recipe] pip install: {pip}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pip])
    if job in BACKBONE_R2_TO_LOCAL:
        r2_key_suffix, local = BACKBONE_R2_TO_LOCAL[job]
        dest = ROOT / local
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3 = R2.client()
        k = R2.key(*r2_key_suffix.split("/"))
        print(f"[recipe] pull backbone s3://{R2.bucket()}/{k} -> {local}")
        s3.download_file(R2.bucket(), k, str(dest))


def do_sync_results(job: str, spec: dict, dest_prefix: str):
    s3 = R2.client()
    for pat in spec.get("results", []):
        pat = pat.replace("~", str(Path.home()))
        matches = glob.glob(str(ROOT / pat)) if not pat.startswith("/") else glob.glob(pat)
        if not matches:
            print(f"[recipe] (no match for results glob: {pat})")
        for m in matches:
            name = Path(m).name
            k = f"{dest_prefix.rstrip('/')}/{name}"
            print(f"[recipe] sync {m} -> s3://{R2.bucket()}/{k}")
            s3.upload_file(m, R2.bucket(), k)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True)
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--print-cmd", action="store_true")
    ap.add_argument("--sync-results", metavar="DEST_KEY_PREFIX")
    args = ap.parse_args()
    spec = job_spec(args.job)

    if args.print_cmd:
        print(spec["train_cmd"])
        return
    if args.install:
        do_install(args.job, spec)
    if args.sync_results:
        do_sync_results(args.job, spec, args.sync_results)


if __name__ == "__main__":
    main()
