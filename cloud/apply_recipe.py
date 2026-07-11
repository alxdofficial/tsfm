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

# baseline weights mirrored to R2 -> the local path(s) each eval script expects. One job may need
# several files (NormWear: backbone + MSiTF). All are pulled from OUR R2, so a pod never depends on
# HuggingFace / GitHub releases. (crosshar/limubert are self-pretrained local artifacts; unimts/
# normwear are official checkpoints, sha256-verified vs upstream — see halo-baseline-weight-provenance.)
BACKBONE_R2_TO_LOCAL = {
    "crosshar": [("ckpt/baselines/crosshar/model_masked_6_1.pt",
                  "auxiliary_repos/CrossHAR/saved/pretrain_base_combined_train_20_120/model_masked_6_1.pt")],
    "limubert": [("ckpt/baselines/limubert/pretrained_combined.pt",
                  "auxiliary_repos/LIMU-BERT-Public/saved/pretrain_base_combined_train_20_120/pretrained_combined.pt")],
    "unimts": [("ckpt/baselines/unimts/UniMTS.pth",
                "auxiliary_repos/UniMTS/checkpoint/UniMTS.pth")],
    "normwear": [("ckpt/baselines/normwear/normwear_pretrain_ckpt.pth",
                  "auxiliary_repos/NormWear/checkpoints/normwear_pretrain_ckpt.pth"),
                 ("ckpt/baselines/normwear/normwear_msitf_zeroshot_last_checkpoint-5.pth",
                  "auxiliary_repos/NormWear/checkpoints/normwear_msitf_zeroshot_last_checkpoint-5.pth")],
}

# torch.hub cache tarballs mirrored to R2 -> extracted into the local hub dir so the model loads
# via source="local" with NO GitHub dependency (kills the unpinned-main drift + firewalled-pod
# risk the provenance audit flagged). The mirrored tarball is the authentic pinned tag (harnet5
# mtl_5_best.mdl sha256 74ffaefb...). (R2 key suffix, cache-dir name under torch.hub.get_dir()).
HUB_CACHE_R2 = {
    "ssl_wearables": ("ckpt/baselines/ssl_wearables/hub_v1.0.0.tar.gz",
                      "OxWearables_ssl-wearables_v1.0.0"),
}


def job_spec(job: str) -> dict:
    j = RECIPES["jobs"].get(job)
    if j is None:
        sys.exit(f"unknown job '{job}' (known: {sorted(RECIPES['jobs'])})")
    return j


def provision_source(job: str, spec: dict):
    """Clone the baseline's pinned upstream source into auxiliary_repos/<name> (idempotent).

    The evaluators import from a vendored source tree (limubert `from models import ...`,
    unimts `from contrastive import ...`, normwear `from NormWear...`) that a clean pod does
    NOT have — auxiliary_repos/ is gitignored, so the shallow HALO clone never carries it.
    We clone the recipe's `upstream_repo` at its pinned `repo_commit`; the target dir is the
    repo basename, which is exactly what each evaluate_*.py inserts on sys.path. Must run
    BEFORE the backbone pull (which writes into <name>/saved/), or git clone would refuse a
    non-empty dir. Skips baselines with no git upstream (deepconvlstm; ssl_wearables uses
    torch.hub) and an already-present checkout (local dev + idempotent re-runs).
    """
    url = spec.get("upstream_repo", "")
    commit = spec.get("repo_commit", "")
    if not url.startswith("http"):
        return  # deepconvlstm (no repo) or ssl_wearables (torch.hub) — nothing to clone
    if not commit or commit.startswith("PIN@"):
        sys.exit(f"[recipe] {job}: repo_commit is unset/placeholder ({commit!r}); "
                 "pin a real SHA in recipes.json before provisioning a pod.")
    name = url.rstrip("/").split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    dest = ROOT / "auxiliary_repos" / name
    if dest.exists() and any(dest.iterdir()):
        # Already provisioned (pod re-run) or present in local dev (may lack .git if
        # copied rather than cloned). Non-empty -> assume good; never clobber, never
        # crash cloning into a non-empty dir.
        print(f"[recipe] source present: auxiliary_repos/{name} (non-empty; skip clone)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[recipe] clone {url} @ {commit[:10]} -> auxiliary_repos/{name}")
    subprocess.check_call(["git", "clone", "--quiet", url, str(dest)])
    subprocess.check_call(["git", "-C", str(dest), "checkout", "--quiet", commit])


def _safe_extractall(tar, dest: Path):
    """Extract guarding against tar-slip (members that escape dest via .., absolute paths, or links).

    Even though we produce and host the tarball, a corrupted/tampered R2 object must not be able to
    write outside the hub dir. Prefer the stdlib 'data' filter (Python 3.12+); fall back to a manual
    member check on older interpreters.
    """
    dest = dest.resolve()
    try:
        tar.extractall(dest, filter="data")  # 3.12+: rejects traversal, absolute paths, escaping links
        return
    except TypeError:
        pass  # Python < 3.12 has no filter kwarg -> validate members ourselves
    for m in tar.getmembers():
        if m.islnk() or m.issym():
            raise RuntimeError(f"refusing link member {m.name!r} in tar")
        target = (dest / m.name).resolve()
        if target != dest and dest not in target.parents:
            raise RuntimeError(f"tar member {m.name!r} escapes {dest}")
    tar.extractall(dest)


def do_install(job: str, spec: dict):
    pip = spec.get("pip", [])
    if pip:
        print(f"[recipe] pip install: {pip}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pip])
    provision_source(job, spec)   # clone vendored upstream BEFORE weights land in its saved/ dir
    weights = BACKBONE_R2_TO_LOCAL.get(job, [])
    if weights:
        s3 = R2.client()
        for r2_key_suffix, local in weights:
            dest = ROOT / local
            if dest.exists():
                print(f"[recipe] weight present: {local} (skip)")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            k = R2.key(*r2_key_suffix.split("/"))
            print(f"[recipe] pull weight s3://{R2.bucket()}/{k} -> {local}")
            s3.download_file(R2.bucket(), k, str(dest))
    if job in HUB_CACHE_R2:
        import tarfile
        import torch  # present on the pod; only needed to resolve the hub dir
        key_suffix, cache_name = HUB_CACHE_R2[job]
        hub = Path(torch.hub.get_dir())
        dest_dir = hub / cache_name
        if dest_dir.exists() and any(dest_dir.iterdir()):
            print(f"[recipe] hub cache present: {cache_name} (skip)")
        else:
            hub.mkdir(parents=True, exist_ok=True)
            s3 = R2.client()
            k = R2.key(*key_suffix.split("/"))
            tmp = hub / "_hubcache.tar.gz"
            print(f"[recipe] pull hub cache s3://{R2.bucket()}/{k} -> {cache_name}")
            s3.download_file(R2.bucket(), k, str(tmp))
            with tarfile.open(tmp) as t:
                _safe_extractall(t, hub)
            tmp.unlink()


def do_sync_results(job: str, spec: dict, dest_prefix: str):
    s3 = R2.client()
    for pat in spec.get("results", []):
        pat = pat.replace("~", str(Path.home()))
        # recursive=True so '**' in a recipe's results glob (e.g. the halo job's
        # training_output/semantic_alignment/**/best.pt) actually recurses; without it Python
        # treats '**' as a single '*' and the nested checkpoint never matches / never uploads (#91c).
        matches = (glob.glob(str(ROOT / pat), recursive=True) if not pat.startswith("/")
                   else glob.glob(pat, recursive=True))
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
