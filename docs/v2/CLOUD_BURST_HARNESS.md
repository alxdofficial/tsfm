# Cloud-Burst Training Harness (vast.ai + Cloudflare R2)

Status: **design + data-bundle stage** (no cloud credentials wired yet).
Target: run HALO and every baseline as independent GPU jobs, one pod each, in
parallel, with **guaranteed teardown** so we never leak billing.

## Goal

Today every model trains serially on the one local RTX 4090. We have ~7 training
jobs (HALO + crosshar + limubert + ssl_wearables + deepconvlstm + unimts +
normwear) that share **no state**. Run them as N pods in parallel and wall-clock
collapses from "sum of all jobs" to "the single slowest job" (HALO).

```
                    ┌──────────────── control host (local / cheap VM) ────────────────┐
                    │  fleet.py : for each job -> create pod -> bootstrap -> train     │
                    │            -> poll sentinel -> pull results -> DESTROY (trap)    │
                    └───────────────┬───────────────┬───────────────┬────────────────┘
                        vastai create/ssh/destroy    │               │
                    ┌───────────────┴──┐  ┌───────────┴─────┐  ┌──────┴───────────┐
                    │ pod: HALO         │  │ pod: crosshar   │  │ pod: ssl_wearab. │  ...
                    │  clone repo@SHA   │  │  recipe deps    │  │  recipe deps     │
                    │  pull data bundle │  │  pull bundle    │  │  pull bundle     │
                    │  tmux train       │  │  tmux train     │  │  tmux train      │
                    │  push results ────┼──┼── push results ─┼──┼── push results ──┼──┐
                    └───────────────────┘  └─────────────────┘  └──────────────────┘  │
                                                                                        ▼
                                              ┌──────────── Cloudflare R2 (s3://halo) ──────────┐
                                              │ data/bundle-<hash>.tar.zst  (processed tensors) │
                                              │ ckpt/baselines/<name>/...   (local-only backbones)│
                                              │ runs/<run-id>/<job>/{results.json,ckpt,train.log}│
                                              └──────────────────────────────────────────────────┘
```

## Why these choices

- **vast.ai**: cheapest, deep supply, mature `vastai` CLI (`search offers`,
  `create instance`, `ssh-url`, `destroy instance`). Interruptible tier is very
  cheap for the short baseline jobs. runpod can slot in later behind the same
  `Provider` interface.
- **Cloudflare R2** (S3-compatible): credential-based auth (access-key/secret,
  **no headless OAuth** like Drive), **free egress** (pods pull the bundle for
  free), works with `aws s3 --endpoint-url`, `boto3(endpoint_url=…)`, or `rclone`.
- **Process once locally, ship a bundle — never reprocess on the pod.** Forced by
  reality: some raw sources are gone (`unimib_shar`'s `acc_labels.npy`) and the
  raw `data/*/sessions` tree is huge/ephemeral. The pod only ever sees the small
  *processed* tensors. See `make_data_bundle.py`.

## 1. Data bundle (`benchmark_data/scripts/make_data_bundle.py`)

Collects the processed artifacts training/eval actually read, tars+zstd them,
content-hashes the archive, and (if R2 creds are present) uploads to
`s3://<bucket>/data/bundle-<hash>.tar.zst` + a `bundle-latest.txt` pointer.

Included:
- `benchmark_data/processed/limubert/**` (data_20_120 + label_20_120 + mapping)
- `benchmark_data/processed/ssl_wearables/**` (data_30_180)
- `benchmark_data/dataset_config.json`
- `benchmark_data/processed/limubert/global_label_mapping.json` (94-way)
- `benchmark_data/eval_v2/labels/**` (pre-registered vocabularies)
- `test_output/baseline_evaluation/*.pt` (cached ConSE heads — optional; pods can
  refit, but shipping them lets an eval-only pod skip the fit)

Backbone checkpoints that exist ONLY locally (CrossHAR/LiMU-BERT `combined_train`
pretrained encoders) are uploaded separately to `ckpt/baselines/<name>/` by
`fetch_baselines`/a one-time push, because they can't be re-downloaded.

Run now (no creds needed — builds the tarball + manifest locally and prints the
upload command):
```
python benchmark_data/scripts/make_data_bundle.py            # build + hash + manifest
R2_BUCKET=halo R2_ENDPOINT=https://<acct>.r2.cloudflarestorage.com \
AWS_ACCESS_KEY_ID=… AWS_SECRET_ACCESS_KEY=… \
python benchmark_data/scripts/make_data_bundle.py --upload   # + push to R2
```

## 2. Per-baseline recipes (`cloud/recipes.json`)

One record per job declares everything a pod needs — heterogeneous envs are the
real complexity (each baseline has different deps/weights):

| job | encoder source | extra pip | weights | train entry |
|---|---|---|---|---|
| halo | our repo | — | — | `training_scripts/.../semantic_alignment_train.py` |
| crosshar | vendored @SHA | — | `ckpt/baselines/crosshar` (R2) | `refit_conse_heads.py --baselines crosshar` |
| limubert | vendored @SHA | — | `ckpt/baselines/limubert` (R2) | `refit_conse_heads.py --baselines limubert` |
| ssl_wearables | torch.hub | — | hub auto | `evaluate_ssl_wearables.py` (TSFM_ALLOW_SSL_HEADFIT=1) |
| deepconvlstm | our repo | — | none (from scratch) | `run_fewshot_v2.py` |
| unimts | UniMTS@SHA | clip, torchvision | `UniMTS.pth` + CLIP (HF) | `evaluate_unimts.py` |
| normwear | NormWear@SHA | torchaudio | backbone + TinyLlama (HF) | `evaluate_normwear.py` |

Each record: `{repo, commit, pip[], fetch[], train_cmd, sentinel, gpu_min_gb, max_hours}`.
The pod bootstrap reads its record and self-assembles.

## 3. Orchestration + billing safety (`cloud/fleet.py`) — built later

Per-job lifecycle: `search offers → create → wait-ssh → bootstrap → tmux train →
poll sentinel → pull results → destroy`.

**Teardown is guaranteed three independent ways** (a leaked pod is real money):
1. `try/finally` (and SIGINT/SIGTERM trap) in `fleet.py` destroys the instance on
   every exit path.
2. **On-pod watchdog**: an `at`/systemd-timer that hard-stops the box after
   `max_hours` even if the controller dies or loses network.
3. **Reconciliation sweep**: at fleet exit, `vastai show instances` → destroy any
   instance tagged with this `run-id` that's still alive. Idempotent.

Plus: per-job \$/hr × elapsed logged live; interruptible jobs checkpoint to R2
every N steps so a preempted pod resumes on a fresh one.

## 4. Remote control via tmux

Training runs inside a named tmux session on the pod so it survives SSH drops.
The agent drives via `ssh <pod> 'tmux send-keys …'` and tails `~/train.log`. A
sentinel file (`DONE`/`FAILED` + exit code) is the machine-readable "job over"
signal the controller polls — no need to babysit a live stream.

## Secrets

R2 keys + HF token + vast API key live in the control host's env / a `.env` that
is **gitignored**, injected into pods at `create` time as env vars. Never baked
into an image, never committed.

## Build order

1. **[now]** `make_data_bundle.py` + `cloud/recipes.json` + this doc.
2. Push HALO repo to a private GitHub remote; one-time upload of local-only
   backbones to R2 `ckpt/baselines/`.
3. `cloud/bootstrap.sh` (pod-side: clone@SHA, pull bundle, apply recipe, tmux train).
4. `cloud/fleet.py` (control-side lifecycle + the 3-way teardown) against `vastai`.
5. Dry-run on ONE cheap pod with the smallest job (deepconvlstm) end-to-end before
   fanning out.
```
```
