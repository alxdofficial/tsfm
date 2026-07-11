# vast.ai + R2 Cloud-Burst — Operational Runbook

Practical runbook for spinning up remote GPU pods on vast.ai to train HALO baselines (and later
HALO itself), with Cloudflare R2 as the artifact store. Written for future-me: the exact commands,
the control pattern, and every CLI gotcha solved along the way. Architecture lives in
[`CLOUD_BURST_HARNESS.md`](CLOUD_BURST_HARNESS.md); this is the "how to actually drive it" doc.

## TL;DR — the loop

```bash
cd code
set -a; . cloud/.env; set +a          # load R2 + VAST_API_KEY (gitignored)
cloud/halo preflight --job crosshar    # verify every pod input is on R2 (no spend)
cloud/halo up crosshar --vast          # LIVE: create pod -> train -> sync results -> DESTROY
cloud/halo status <run-id>             # results synced back to R2
cloud/halo nuke                        # PANIC BUTTON: destroy every halo-* instance
```
Pods are **stateless**: they curl `cloud/bootstrap.sh` from the public repo, which clones @SHA,
pulls the R2 data bundle + backbone, trains in tmux, and writes a `DONE`/`FAILED` sentinel +
results back to R2. `fleet.py` polls the sentinel and **destroys** the pod on completion.

## One-time setup

1. **vastai CLI**: `pip install vastai` (using 1.3.0). Auth is via the **`VAST_API_KEY` env var**
   — it is read automatically and *overrides* any `vastai set api-key`. Just keep it in `cloud/.env`.
2. **cloud/.env** (gitignored, chmod 600): R2 creds (`R2_BUCKET`, `R2_ENDPOINT`,
   `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) + `VAST_API_KEY`. Source with `set -a; . cloud/.env; set +a`.
3. **Repo is PUBLIC** (`github.com/alxdofficial/tsfm`) → pods `git clone` with no deploy key. The
   pod fetches `bootstrap.sh` from `raw.githubusercontent.com/alxdofficial/tsfm/<SHA>/cloud/bootstrap.sh`,
   so **whatever SHA the pod checks out must already be pushed to the public remote.**
4. **Upload the pod inputs once** (and after any data/backbone change):
   ```bash
   python benchmark_data/scripts/make_data_bundle.py --upload   # processed tensors -> R2
   cloud/halo push                                              # backbones + requirements -> R2
   ```

## What lives in R2 (`gen-purp-bucket/halo/`)

```
data/bundle-<hash>.tar.gz     # processed baseline tensors (limubert+ssl+configs+labels), 594MB
data/bundle-latest.txt        # pointer
ckpt/baselines/<name>/*.pt    # local-only pretrained backbones (crosshar, limubert; ~260KB each)
meta/requirements-core.txt    # frozen deps (torch/vision/audio/clip installed separately by bootstrap)
runs/<run-id>/<job>/          # results.json + ckpt + train.log + DONE/FAILED, synced back by the pod
```
Secrets never live in R2 — they're injected into the pod as env vars at create time.

## The control pattern: tmux + ssh into the remote

Run the controller **inside a local tmux session** so a dropped SSH/network doesn't kill the fleet
(and orphan a paid pod). Training on the pod ALSO runs in its own tmux (survives the controller
disconnecting).

```bash
tmux new -s halo-ctl                    # local control session (survives disconnect)
set -a; . cloud/.env; set +a
python cloud/fleet.py --provider vast --jobs crosshar \
    --gpu RTX_5090 --max-dph 0.5 \
    --image pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
# detach with Ctrl-b d; reattach with `tmux attach -t halo-ctl`
```

To **debug a live pod directly** (watch the bootstrap/train unfold):
```bash
vastai show instances-v1 --raw | jq '.instances[] | {id,label,ssh_host,ssh_port,status}'
vastai ssh-url <instance-id>            # -> ssh://root@<host>:<port>
ssh -p <port> root@<host>               # then on the pod:
  tmux attach -t halo                   #   the training session bootstrap.sh started
  tail -f ~/train.log                   #   live training log
  cat ~/EXIT_CODE ~/DONE ~/FAILED 2>/dev/null
```

## Billing safety (three independent teardowns)

1. `fleet.py` `try/finally` + SIGINT/SIGTERM + `atexit` → destroy every instance it created.
2. **Reconciliation sweep** at exit: `show instances-v1` filtered by the `halo-<run-id>-*` label →
   destroy strays it lost track of (catches "created but the id wasn't captured").
3. **On-pod watchdog** in `bootstrap.sh`: hard `poweroff` after `MAX_HOURS`. ⚠️ See findings — this
   may not work inside a vast *container*; the controller destroy is the real teardown.

Plus a **poll deadline** (`max_hours + 15min`) so `fleet.py` never hangs forever on a silent pod,
and `cloud/halo nuke` as a manual panic button. **Always** end a session by confirming
`vastai show instances-v1` is empty.

## Findings & bugs solved (CLI gotchas — read before editing fleet.py)

- **`vastai show instances --raw` returns EMPTY** (the command is deprecated). Use
  **`vastai show instances-v1 --raw`** → JSON `{"instances":[{id,label,ssh_host,...}]}`. The old one
  silently breaks the reconciliation teardown — a real billing risk.
- **`vastai destroy instance <id>` needs `-y`** — without it, it waits on an interactive
  confirmation prompt and *hangs*, which can leak a paid pod. Always `destroy instance <id> -y`.
- **`VAST_API_KEY` env overrides `vastai set api-key`** — so just export it; don't bother calling
  `set api-key` in code (it also prints a warning to stderr).
- **aws-cli against R2 needs a region**: `export AWS_DEFAULT_REGION=auto` or it errors
  "You must specify a region." (set in bootstrap.sh).
- **5090 = Blackwell (sm_120)** → needs CUDA **12.8+** and torch built for it. Base image
  `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` exists (Docker Hub 200) and works; bootstrap then
  reinstalls `torch==2.9.0+cu128` (has sm_120) for parity with local. Offers ~**$0.34/hr**, ~50 available.
- **`create instance` output**: `{"success":true,"new_contract":<id>}` — the instance id is
  `new_contract` (fleet reads `new_contract or id`). `--env` wants a single quoted string of
  `-e K=V` pairs; `--onstart-cmd` takes the script *contents* (we curl+run bootstrap.sh).
- **requirements-core.txt is a full `pip freeze` (157 pins)** — fragile if the pod's Python ≠ 3.11.
  The base pytorch image is 3.11, so it matches; if a future image differs, loosen the pins.

## <a name="live-run"></a>First live run — results (filled after the pod test)

_TO BE COMPLETED after the first real 5090 `crosshar` pod: boot time, deps-install time, whether
bootstrap ran clean, sentinel + results sync worked, actual cost, and any live bugs found._

## Cost notes

- 5090 ≈ $0.34/hr; a crosshar head-fit + eval is a few minutes → **cents** per job.
- Prefer interruptible offers for cheap sweeps; checkpoint to R2 every N steps so a preempted pod
  resumes on a fresh one (relevant for long HALO runs, not the light baseline jobs).
- The 41 GB HALO session-store bundle is **not** uploaded yet (deferred) — needed only to burst
  HALO training itself, not baselines.
