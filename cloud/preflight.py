#!/usr/bin/env python3
"""Pre-spend checklist for the cloud-burst harness. Verifies every input a pod needs is in place
BEFORE any instance is created (a failed job on a live pod still costs money).

Checks: R2 creds + bucket reachable; processed data bundle + backbones + requirements uploaded;
recipes parse; vast.ai key present; the public repo is reachable at the pinned SHA.

  set -a; . cloud/.env; set +a
  python cloud/preflight.py [--job crosshar] [--sha <git-sha>]

Exit 0 = ready; nonzero = at least one hard blocker.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cloud"))
import r2_util as R2  # noqa: E402

OK, WARN, BAD = "  \033[32m✓\033[0m", "  \033[33m!\033[0m", "  \033[31m✗\033[0m"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", help="also check this job's backbone is uploaded")
    ap.add_argument("--sha", help="git SHA the pod will check out (default: current HEAD)")
    args = ap.parse_args()
    blockers = 0

    # 1) R2 creds + bucket
    try:
        s3 = R2.client()
        s3.head_bucket(Bucket=R2.bucket())
        print(f"{OK} R2 bucket '{R2.bucket()}' reachable")
    except SystemExit as e:
        print(f"{BAD} R2 creds: {e}"); return 2
    except Exception as e:
        print(f"{BAD} R2 bucket unreachable: {e}"); return 2

    keys = {k: sz for k, sz in R2.list_prefix(s3)}

    def need(subkey, label):
        nonlocal blockers
        full = R2.key(*subkey.split("/"))
        if full in keys:
            print(f"{OK} {label} ({keys[full]/1e6:.1f} MB)")
        else:
            print(f"{BAD} {label} MISSING at s3://{R2.bucket()}/{full}"); blockers += 1

    # 2) data bundle + pointer
    ptr = R2.key("data", "bundle-latest.txt")
    if ptr in keys:
        bundle = s3.get_object(Bucket=R2.bucket(), Key=ptr)["Body"].read().decode().strip()
        need(f"data/{bundle}", f"data bundle {bundle}")
        # FRESHNESS (not just presence): the bundle ships the PROCESSED grids + 86-way label map +
        # canonical GT; pods never reprocess. A stale bundle silently trains/evals baselines on
        # pre-fix data (e.g. before a units/vocab correction) — an invalid comparison that a
        # presence-only check misses. Compare R2's pointer to the last LOCAL make_data_bundle build.
        local_ptr = ROOT / "benchmark_data" / "bundles" / "bundle-latest.txt"
        if local_ptr.exists():
            local_bundle = local_ptr.read_text().strip()
            if local_bundle == bundle:
                print(f"{OK} bundle is FRESH (R2 pointer == last local build {bundle})")
            else:
                print(f"{BAD} bundle STALE: R2 points at {bundle} but your last local build is "
                      f"{local_bundle}. Re-run `make_data_bundle.py --upload` so pods get current "
                      f"grids/labels."); blockers += 1
        else:
            print(f"{WARN} cannot verify bundle freshness (no local benchmark_data/bundles/"
                  f"bundle-latest.txt to compare; run make_data_bundle.py to stamp one)")
    else:
        print(f"{BAD} data/bundle-latest.txt MISSING (run make_data_bundle.py --upload)"); blockers += 1

    # 3) requirements + backbones
    need("meta/requirements-core.txt", "requirements-core.txt")
    for bb in (["crosshar", "limubert"] if not args.job else
               ([args.job] if args.job in ("crosshar", "limubert") else [])):
        fn = "model_masked_6_1.pt" if bb == "crosshar" else "pretrained_combined.pt"
        need(f"ckpt/baselines/{bb}/{fn}", f"backbone: {bb}")

    # 4) recipes parse
    try:
        rc = json.load(open(ROOT / "cloud" / "recipes.json"))
        print(f"{OK} recipes.json parses ({len(rc['jobs'])} jobs)")
        if args.job and args.job not in rc["jobs"]:
            print(f"{BAD} job '{args.job}' not in recipes.json"); blockers += 1
    except Exception as e:
        print(f"{BAD} recipes.json: {e}"); blockers += 1

    # 5) vast.ai key (WARN only — not needed for scaffolding/mock runs)
    if os.environ.get("VAST_API_KEY"):
        print(f"{OK} VAST_API_KEY present")
    else:
        print(f"{WARN} VAST_API_KEY not set (fine for --provider mock; required for live vast pods)")

    # 6) repo + the pinned SHA reachable on the remote (HARD blocker). A pod whose
    # `git fetch --depth 1 origin $REPO_SHA` (bootstrap.sh) can't resolve the SHA is guaranteed to
    # fail, so an unpushed/unreachable SHA must block READY, not merely WARN (#91d — was a WARN that
    # let doomed pods spend money). We prove the SHA is a ref tip on the remote via `git ls-remote`.
    url = rc.get("defaults", {}).get("repo", "").replace("git@github.com:", "https://github.com/")
    sha = (args.sha or subprocess.getoutput("git rev-parse HEAD")).strip()
    if not url.startswith("http"):
        print(f"{BAD} recipes.defaults.repo is not an http(s) URL ({url}); cannot verify SHA remotely")
        blockers += 1
    else:
        rc_ls = subprocess.run(["git", "ls-remote", url], capture_output=True, text=True)
        if rc_ls.returncode != 0:
            print(f"{BAD} repo not reachable ({url}); fix recipes.defaults.repo")
            blockers += 1
        elif any(line.split("\t", 1)[0] == sha for line in rc_ls.stdout.splitlines()):
            print(f"{OK} repo reachable; SHA {sha[:12]} is a ref tip on the remote: {url}")
        else:
            print(f"{BAD} SHA {sha[:12]} is NOT a ref tip on {url} — push your branch first "
                  f"(the pod's `git fetch --depth 1 origin {sha[:12]}` will fail)")
            blockers += 1

    print()
    if blockers:
        print(f"NOT READY — {blockers} blocker(s). Fix the ✗ items above.")
        return 1
    print("READY — all pod inputs are in place.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
