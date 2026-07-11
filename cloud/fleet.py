#!/usr/bin/env python3
"""Cloud-burst fleet orchestrator: spin up one GPU pod per job, run it, pull results, DESTROY it.

Design goals (see docs/v2/CLOUD_BURST_HARNESS.md):
- Stateless pods: a pod curls cloud/bootstrap.sh from the public repo, which clones @SHA, pulls the
  R2 data bundle + backbone, trains in tmux, and writes a DONE/FAILED sentinel + results to R2.
- Teardown guaranteed THREE ways (a leaked pod is real money):
    1. try/finally + SIGINT/SIGTERM + atexit destroy every instance this process created;
    2. an on-pod watchdog (in bootstrap.sh) hard-powers-off after MAX_HOURS if we die;
    3. a reconciliation sweep at exit destroys anything tagged with this run-id still alive.
- Providers are pluggable: VastProvider (live, `vastai` CLI) and MockProvider (local dry-run, NO spend).

Usage:
    set -a; . cloud/.env; set +a
    python cloud/fleet.py --provider mock --jobs crosshar limubert          # local dry-run, no spend
    python cloud/fleet.py --provider vast --jobs crosshar --gpu RTX_4090 --max-dph 0.5   # live

Scaffolding status: MockProvider is fully exercised; VastProvider CLI syntax should be confirmed
against the installed `vastai` version before the first live run.
"""

import argparse
import atexit
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cloud"))
import r2_util as R2  # noqa: E402

RECIPES = json.load(open(ROOT / "cloud" / "recipes.json"))
DEFAULTS = RECIPES.get("defaults", {})
REPO_HTTPS = DEFAULTS.get("repo", "").replace("git@github.com:", "https://github.com/").replace(".git", "")
POLL_SECS = 30


def log(msg):
    print(f"[fleet {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pod_env(job, run_id, sha):
    """Env injected into the pod (secrets included — passed at create time, never baked)."""
    return {
        "R2_BUCKET": os.environ["R2_BUCKET"], "R2_ENDPOINT": os.environ["R2_ENDPOINT"],
        "R2_PREFIX": os.environ.get("R2_PREFIX", "halo"),
        "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
        "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
        "HALO_JOB": job, "HALO_RUN_ID": run_id,
        "REPO_URL": REPO_HTTPS + ".git", "REPO_SHA": sha,
        "MAX_HOURS": str(RECIPES["jobs"].get(job, {}).get("max_hours", DEFAULTS.get("watchdog_max_hours", 8))),
    }


def onstart_cmd(sha):
    """One-liner run on the pod at boot: fetch bootstrap.sh from the public repo @SHA and run it.
    Installs curl first if the base image lacks it (python:3.11-slim does) so it works on a tiny
    image — which we use to AVOID the slow/throttled Docker Hub pull of a giant pytorch image."""
    raw = f"https://raw.githubusercontent.com/{REPO_HTTPS.split('github.com/')[-1]}/{sha}/cloud/bootstrap.sh"
    return ("bash -lc 'command -v curl >/dev/null || (apt-get update -qq && apt-get install -y -qq curl); "
            f"curl -fsSL {raw} -o /tmp/bootstrap.sh && bash /tmp/bootstrap.sh'")


# ============================ providers ============================

class Provider:
    def find_offer(self, gpu, max_dph, min_gb): ...
    def create(self, offer, job, env, image): ...
    def is_done(self, run_id, job): ...       # -> "DONE" | "FAILED" | None
    def destroy(self, instance_id): ...
    def list_run(self, run_id): ...           # -> [instance_id] still alive for this run


class MockProvider(Provider):
    """Simulates the full lifecycle locally — NO cloud spend. Fakes the R2 sentinel after a few polls
    so the control flow + teardown paths are all exercised."""
    def __init__(self): self._polls, self._alive = {}, set()
    def find_offer(self, gpu, max_dph, min_gb):
        return {"id": "offer-mock", "dph_total": 0.29, "gpu_name": gpu}
    def create(self, offer, job, env, image):
        iid = f"mock-{job}-{env['HALO_RUN_ID'][-6:]}"
        self._alive.add(iid)
        log(f"[mock] create instance {iid} on {offer['gpu_name']} @ ${offer['dph_total']}/hr")
        log(f"[mock]   onstart: {onstart_cmd(env['REPO_SHA'])[:90]}…")
        log(f"[mock]   env keys: {sorted(env)}")
        return iid
    def is_done(self, run_id, job):
        self._polls[job] = self._polls.get(job, 0) + 1
        return "DONE" if self._polls[job] >= 2 else None   # 'completes' after 2 polls
    def destroy(self, iid):
        self._alive.discard(iid); log(f"[mock] destroy {iid}")
    def list_run(self, run_id):
        return [i for i in self._alive if run_id[-6:] in i]


class VastProvider(Provider):
    """Live vast.ai via the `vastai` CLI. Confirm subcommand syntax against your installed version."""
    def __init__(self):
        if not os.environ.get("VAST_API_KEY"):
            sys.exit("VAST_API_KEY not set (source cloud/.env).")
        # vastai reads VAST_API_KEY from the environment automatically — no `set api-key` needed.

    def _vast(self, *args, raw=True):
        cmd = ["vastai", *args] + (["--raw"] if raw else [])
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(f"vastai {args} failed: {out.stderr.strip()}")
        return json.loads(out.stdout) if raw and out.stdout.strip() else out.stdout

    def find_offer(self, gpu, max_dph, min_gb):
        # Bias toward VERIFIED hosts with fast download (image/bundle pull) and enough disk — the
        # cheapest bottom-tier hosts can sit in 'loading' for 10+ min pulling the image. Order by a
        # price/bandwidth blend so we don't just grab the slowest cheap box.
        q = (f"gpu_name={gpu} num_gpus=1 rentable=true verified=true "
             f"dph_total<{max_dph} inet_down>500 disk_space>50")
        offers = self._vast("search", "offers", q, "-o", "dph_total")
        if not offers:  # relax the bandwidth/verified filters if nothing matches
            offers = self._vast("search", "offers",
                                 f"gpu_name={gpu} num_gpus=1 rentable=true dph_total<{max_dph} disk_space>50",
                                 "-o", "dph_total")
        if not offers:
            raise RuntimeError(f"no vast offer for {gpu} < ${max_dph}/hr")
        return offers[0]

    def create(self, offer, job, env, image):
        envstr = " ".join(f"-e {k}={v}" for k, v in env.items())
        r = self._vast("create", "instance", str(offer["id"]), "--image", image,
                        "--disk", "60", "--onstart-cmd", onstart_cmd(env["REPO_SHA"]),
                        "--env", envstr, "--label", f"halo-{env['HALO_RUN_ID']}-{job}")
        return str(r.get("new_contract") or r.get("id"))

    def is_done(self, run_id, job):
        s3 = R2.client()
        base = R2.key("runs", run_id, job)
        if R2.head(s3, f"{base}/DONE"):
            return "DONE"
        if R2.head(s3, f"{base}/FAILED"):
            return "FAILED"
        return None

    def destroy(self, iid):
        try:
            # -y: non-interactive (an unconfirmed prompt would hang and leak the pod).
            self._vast("destroy", "instance", str(iid), "-y", raw=False); log(f"destroyed instance {iid}")
        except Exception as e:
            log(f"!! destroy {iid} FAILED — retry manually: vastai destroy instance {iid} -y  ({e})")

    def list_run(self, run_id):
        # `show instances` (deprecated) emits no JSON with --raw; instances-v1 does.
        data = self._vast("show", "instances-v1")
        return [str(i["id"]) for i in data.get("instances", [])
                if f"halo-{run_id}-" in str(i.get("label", ""))]


# ============================ orchestration ============================

class Fleet:
    def __init__(self, provider, run_id, image, gpu, max_dph):
        self.p, self.run_id, self.image, self.gpu, self.max_dph = provider, run_id, image, gpu, max_dph
        self._created = set()
        atexit.register(self._teardown_all)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: (self._teardown_all(), sys.exit(130)))

    def _teardown_all(self):
        for iid in list(self._created):
            self.p.destroy(iid); self._created.discard(iid)
        # reconciliation: destroy anything tagged with this run that we lost track of
        for iid in self.p.list_run(self.run_id):
            log(f"reconcile: destroying stray {iid}"); self.p.destroy(iid)

    def run_job(self, job, sha):
        spec = RECIPES["jobs"].get(job) or sys.exit(f"unknown job {job}")
        iid = None
        try:
            offer = self.p.find_offer(self.gpu, self.max_dph, spec.get("gpu_min_gb", 12))
            iid = self.p.create(offer, job, pod_env(job, self.run_id, sha), self.image)
            self._created.add(iid)
            dph = float(offer.get("dph_total", 0) or 0)
            log(f"{job}: instance {iid} up (${dph:.3f}/hr); polling for completion")
            # Hard deadline: never poll forever. If no sentinel by max_hours+15min, give up and
            # let `finally` destroy the pod (the on-pod watchdog is the last backstop).
            deadline = int(RECIPES["jobs"].get(job, {}).get("max_hours", 8)) * 3600 + 900
            t0 = time.time()
            while True:
                st = self.p.is_done(self.run_id, job)
                if st:
                    log(f"{job}: {st} after {int(time.time()-t0)}s (~${dph*(time.time()-t0)/3600:.2f}) "
                        f"-> results at s3://{os.environ['R2_BUCKET']}/{R2.key('runs', self.run_id, job)}/")
                    return {"job": job, "status": st}
                if time.time() - t0 > deadline:
                    log(f"{job}: DEADLINE {deadline}s with no sentinel -> destroying pod (check train.log on R2)")
                    return {"job": job, "status": "TIMEOUT"}
                time.sleep(POLL_SECS)
        finally:
            if iid:
                self.p.destroy(iid); self._created.discard(iid)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["vast", "mock"], default="mock")
    ap.add_argument("--jobs", nargs="+", required=True)
    ap.add_argument("--run-id", default=None, help="default: derived from git SHA")
    ap.add_argument("--sha", default=None, help="git SHA pods check out (default: current HEAD)")
    # Tiny base (~50MB, pulls in seconds even when Docker Hub throttles); bootstrap pip-installs
    # torch from the fast PyTorch CDN (cu128 wheels bundle CUDA; vast host provides the driver).
    ap.add_argument("--image", default="python:3.11-slim")
    ap.add_argument("--gpu", default="RTX_4090")
    ap.add_argument("--max-dph", type=float, default=0.5, help="max $/hr per pod")
    args = ap.parse_args()

    sha = args.sha or subprocess.getoutput("git rev-parse HEAD").strip()
    run_id = args.run_id or f"r{sha[:8]}"
    provider = MockProvider() if args.provider == "mock" else VastProvider()
    log(f"provider={args.provider} run_id={run_id} sha={sha[:12]} jobs={args.jobs} "
        f"gpu={args.gpu} max=${args.max_dph}/hr")

    fleet = Fleet(provider, run_id, args.image, args.gpu, args.max_dph)
    results = []
    try:
        for job in args.jobs:          # sequential here; parallelism = launch multiple fleet procs
            results.append(fleet.run_job(job, sha))
    finally:
        fleet._teardown_all()
    log(f"done: {results}")


if __name__ == "__main__":
    main()
