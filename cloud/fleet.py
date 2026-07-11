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
    def find_offer(self, gpu, max_dph, min_gb, exclude=()): ...
    def create(self, offer, job, env, image): ...
    def status(self, instance_id): ...        # -> 'loading' | 'running' | None(gone)
    def is_done(self, run_id, job): ...       # -> "DONE" | "FAILED" | None
    def destroy(self, instance_id): ...
    def list_run(self, run_id): ...           # -> [instance_id] still alive for this run


class MockProvider(Provider):
    """Simulates the full lifecycle locally — NO cloud spend. Fakes the R2 sentinel after a few polls
    so the control flow + teardown paths are all exercised."""
    def __init__(self): self._polls, self._alive, self._n = {}, set(), 0
    def find_offer(self, gpu, max_dph, min_gb, exclude=()):
        self._n += 1
        return {"id": f"offer-mock-{self._n}", "dph_total": 0.29, "gpu_name": gpu}
    def create(self, offer, job, env, image):
        iid = f"mock-{job}-{env['HALO_RUN_ID'][-6:]}-{self._n}"
        self._alive.add(iid)
        log(f"[mock] create instance {iid} on {offer['gpu_name']} @ ${offer['dph_total']}/hr")
        return iid
    def status(self, iid):
        return "running" if iid in self._alive else None   # boots instantly in the mock
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

    def find_offer(self, gpu, max_dph, min_gb, exclude=(), ram_gb=32):
        # Bias toward VERIFIED, fast-download hosts (the standard pytorch image is usually cached on
        # these, so boot is near-instant). `exclude` skips offers a previous stuck attempt used, so a
        # retry lands on a DIFFERENT host.
        # Resource FLOORS (previously accepted min_gb but never used it -> a host could be picked with
        # too little VRAM/RAM, #74): min_gb = GPU VRAM (GB -> vast gpu_ram is MB); ram_gb = host RAM
        # (GB) — CrossHAR/LiMU-BERT materialize large sequence-embedding arrays and need ~32 GiB
        # (readiness §5). These floors are kept even in the relaxed fallback (never relax resources).
        floors = f"gpu_ram>={int(min_gb) * 1000} cpu_ram>={int(ram_gb)}"
        q = (f"gpu_name={gpu} num_gpus=1 rentable=true verified=true "
             f"dph_total<{max_dph} inet_down>500 disk_space>50 {floors}")
        offers = self._vast("search", "offers", q, "-o", "dph_total")
        if not offers:  # relax the bandwidth/verified filters if nothing matches — but NOT the resource floors
            offers = self._vast("search", "offers",
                                 f"gpu_name={gpu} num_gpus=1 rentable=true dph_total<{max_dph} disk_space>50 {floors}",
                                 "-o", "dph_total")
        offers = [o for o in offers if str(o["id"]) not in exclude]
        if not offers:
            raise RuntimeError(f"no fresh vast offer for {gpu} < ${max_dph}/hr (excluded {len(exclude)})")
        return offers[0]

    def create(self, offer, job, env, image):
        envstr = " ".join(f"-e {k}={v}" for k, v in env.items())
        r = self._vast("create", "instance", str(offer["id"]), "--image", image,
                        "--disk", "60", "--onstart-cmd", onstart_cmd(env["REPO_SHA"]),
                        "--env", envstr, "--label", f"halo-{env['HALO_RUN_ID']}-{job}")
        return str(r.get("new_contract") or r.get("id"))

    def status(self, iid):
        # actual_status: 'loading' (provisioning/pull) -> 'running' (onstart executing) ; None = gone.
        data = self._vast("show", "instances-v1")
        for i in data.get("instances", []):
            if str(i["id"]) == str(iid):
                return i.get("actual_status")
        return None

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
    def __init__(self, provider, run_id, image, gpu, max_dph, boot_deadline=300, max_attempts=5):
        self.p, self.run_id, self.image, self.gpu, self.max_dph = provider, run_id, image, gpu, max_dph
        self.boot_deadline, self.max_attempts = boot_deadline, max_attempts
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

    def _boot_pod(self, job, sha, tried):
        """Create a pod and wait for it to reach 'running'. If it's stuck in 'loading' past
        boot_deadline (a slow/throttled/uncached host — the Docker-Hub-pull lottery), destroy it and
        signal a retry. Returns (iid, dph) on success, or (None, None) to try a fresh host."""
        spec = RECIPES["jobs"].get(job, {})
        offer = self.p.find_offer(self.gpu, self.max_dph, spec.get("gpu_min_gb", 12), exclude=tried)
        tried.add(str(offer["id"]))
        iid = self.p.create(offer, job, pod_env(job, self.run_id, sha), self.image)
        self._created.add(iid)
        dph = float(offer.get("dph_total", 0) or 0)
        log(f"{job}: instance {iid} (${dph:.3f}/hr) — waiting to boot (<= {self.boot_deadline}s)…")
        t0 = time.time()
        while time.time() - t0 <= self.boot_deadline:
            time.sleep(20)                       # let the instance register before the first poll
            st = self.p.status(iid)
            # None = not yet listed / still provisioning (a new instance takes ~20-40s to appear);
            # treat it as 'still booting', NOT vanished. Only 'running' is success; a truly-dead
            # instance just times out below and gets destroyed (a 404 destroy is harmless).
            if st in ("running", "online"):
                log(f"{job}: booted in {int(time.time()-t0)}s"); return iid, dph
        log(f"{job}: not 'running' after {self.boot_deadline}s (last status={st!r}) — "
            f"slow/throttled/uncached host, destroy + retry on a new host")
        self.p.destroy(iid); self._created.discard(iid)
        return None, None

    def run_job(self, job, sha):
        if job not in RECIPES["jobs"]:
            sys.exit(f"unknown job {job}")
        tried = set()
        for attempt in range(1, self.max_attempts + 1):
            log(f"{job}: attempt {attempt}/{self.max_attempts}")
            iid = dph = None
            try:
                iid, dph = self._boot_pod(job, sha, tried)
                if iid is None:
                    continue  # stuck/vanished host — try a fresh offer
                # booted: bootstrap is running; poll for the R2 sentinel.
                deadline = int(RECIPES["jobs"].get(job, {}).get("max_hours", 8)) * 3600 + 900
                t0 = time.time()
                while True:
                    st = self.p.is_done(self.run_id, job)
                    if st:
                        log(f"{job}: {st} after {int(time.time()-t0)}s run "
                            f"(~${dph*(time.time()-t0)/3600:.2f}) -> "
                            f"s3://{os.environ['R2_BUCKET']}/{R2.key('runs', self.run_id, job)}/")
                        return {"job": job, "status": st, "attempts": attempt}
                    if time.time() - t0 > deadline:
                        log(f"{job}: DEADLINE {deadline}s, no sentinel -> destroy (check train.log on R2)")
                        return {"job": job, "status": "TIMEOUT", "attempts": attempt}
                    time.sleep(POLL_SECS)
            finally:
                if iid:
                    self.p.destroy(iid); self._created.discard(iid)
        log(f"{job}: gave up after {self.max_attempts} hosts all stuck booting")
        return {"job": job, "status": "NO_GOOD_HOST", "attempts": self.max_attempts}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["vast", "mock"], default="mock")
    ap.add_argument("--jobs", nargs="+", required=True)
    ap.add_argument("--run-id", default=None, help="default: derived from git SHA")
    ap.add_argument("--sha", default=None, help="git SHA pods check out (default: current HEAD)")
    # Standard PyTorch image: it's the MOST widely-cached image on vast hosts, so most hosts already
    # have it -> near-instant boot with torch preinstalled. Slow/throttled hosts that would have to
    # pull it are handled by fleet's auto-retry (--boot-deadline / --max-attempts), not by us picking
    # a niche/tiny image. bootstrap.sh skips reinstalling torch when the base already has a working one.
    ap.add_argument("--image", default="pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime")
    ap.add_argument("--gpu", default="RTX_4090")
    ap.add_argument("--max-dph", type=float, default=0.5, help="max $/hr per pod")
    ap.add_argument("--boot-deadline", type=int, default=300, help="secs to reach 'running' before retrying a new host")
    ap.add_argument("--max-attempts", type=int, default=5, help="how many hosts to try before giving up")
    args = ap.parse_args()

    sha = args.sha or subprocess.getoutput("git rev-parse HEAD").strip()
    run_id = args.run_id or f"r{sha[:8]}"
    provider = MockProvider() if args.provider == "mock" else VastProvider()
    log(f"provider={args.provider} run_id={run_id} sha={sha[:12]} jobs={args.jobs} "
        f"gpu={args.gpu} max=${args.max_dph}/hr")

    fleet = Fleet(provider, run_id, args.image, args.gpu, args.max_dph,
                  boot_deadline=args.boot_deadline, max_attempts=args.max_attempts)
    results = []
    try:
        for job in args.jobs:          # sequential here; parallelism = launch multiple fleet procs
            results.append(fleet.run_job(job, sha))
    finally:
        fleet._teardown_all()
    log(f"done: {results}")


if __name__ == "__main__":
    main()
