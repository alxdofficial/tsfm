#!/usr/bin/env bash
#
# runpod_experiment.sh — one-shot, arg-driven RunPod runner for the HALO rebuttal ablations.
# =============================================================================================
# RUNS ON THE POD (SSH in and run it, or set it as the pod's startup/"docker command"). It:
#   1. SETUP (idempotent, all from Google Drive — no git auth needed):
#        - code: download this experiment's `git archive` tarball from Drive + extract
#        - data: download the training-data tarball from Drive (gdown) into the RAM disk
#        - deps: pip install (skips torch) + pre-cache SBERT
#   2. RESOLVE  --exp/--variant  ->  the TSFM_* env vars (see resolve_env() below)
#   3. TRAIN to the PERSISTENT volume (/workspace) so checkpoints survive termination
#   4. RETRIEVE artifacts durably (best.pt + hyperparameters.json + plots/metrics.json + log)
#        -> /workspace/artifacts/<exp>/<variant>_seed<N>/   (persistent network volume; never lost)
#        -> pushed to Google Drive via rclone IF a remote is configured (best-effort)
#   5. AUTO-TERMINATE the pod (stops billing) once artifacts are safe — unless --no-terminate,
#        and NEVER on a failed run (failed pods stay alive for SSH debugging).
#
# Usage (on the pod):
#   bash runpod_experiment.sh --exp P6 --variant headline --seed 42
#   bash runpod_experiment.sh --exp P1 --variant semantic
#   bash runpod_experiment.sh --exp P7 --variant temporal_only --dry-run
#
# Variants:
#   P1: none | hard_neg | semantic
#   P6: headline | no_soft_targets | no_queue           (run each at --seed 42/43/44)
#   P7: temporal_only | channel_indep | no_channel_text | cnn_multi | spectral_half | hard_targets |
#       tau_0p3 | soft_weight_0p5 | sbert_mpnet
#
# Code comes via `git clone/pull` from TSFM_REPO_URL (default: public origin, code-only branches).
# Data comes via gdown from DATA_GDRIVE_ID. Required/optional env:
#   RUNPOD_POD_ID         present automatically in every RunPod container (for auto-terminate)
#   DATA_GDRIVE_ID        Google-Drive file id of the training-data tarball (default = 1a6QROP9...)
# Optional env:
#   TSFM_REPO_URL         git remote to clone code from (default: https://github.com/alxdofficial/tsfm.git)
#   RUNPOD_API_KEY        used only as a fallback for termination if runpodctl is unavailable
#   RCLONE_REMOTE         name of a configured rclone remote for Drive artifact upload (default: gdrive)
#   RCLONE_DEST           rclone dest path (default: tsfm_rebuttal/artifacts)
#   TSFM_EPOCHS           override epochs (e.g. 1 for a smoke run; default 100)
# =============================================================================================
set -uo pipefail

# ----------------------------- defaults -----------------------------
EXP=""; VARIANT=""; SEED=42; DRY_RUN=0; NO_TERMINATE=0; SETUP_ONLY=0
WORKDIR="${TSFM_WORKDIR:-/workspace/tsfm}"
REPO_URL="${TSFM_REPO_URL:-https://github.com/alxdofficial/tsfm.git}"  # public origin (code-only branches)
DATA_MOUNT="${TSFM_DATA_MOUNT:-/dev/shm/tsfm_data}"
ARTIFACT_ROOT="${TSFM_ARTIFACT_ROOT:-/workspace/artifacts}"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"
RCLONE_DEST="${RCLONE_DEST:-tsfm_rebuttal/artifacts}"
DATA_GDRIVE_ID="${DATA_GDRIVE_ID:-1a6QROP9qZZetOek_NxbgIWFNYDVY8d0H}"  # legacy (Drive quota-locks under concurrency)
TSFM_HF_DATA_REPO="${TSFM_HF_DATA_REPO:-alxd219p1/tsfm-har-bench}"      # data now lives on HF (CDN, concurrency-safe)

# ----------------------------- arg parsing -----------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp)          EXP="$2"; shift 2;;
    --variant)      VARIANT="$2"; shift 2;;
    --seed)         SEED="$2"; shift 2;;
    --dry-run)      DRY_RUN=1; shift;;
    --setup-only)   SETUP_ONLY=1; shift;;   # do setup + GPU check, then exit (no training) — for the smoke test
    --no-terminate) NO_TERMINATE=1; shift;;
    -h|--help)      sed -n '2,40p' "$0"; exit 0;;
    *) echo "unknown arg: $1 (see --help)"; exit 2;;
  esac
done
[[ -z "$EXP" || -z "$VARIANT" ]] && { echo "ERROR: --exp <P1|P6|P7> and --variant <name> are required (see --help)"; exit 2; }

RUN_TAG="${EXP}_${VARIANT}_seed${SEED}"
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ----------------------------- exp -> PUBLIC code-only branch -----------------------------
# The pod clones the code-only pod/* branch (no paper-rebuttal/) from the public origin. The full
# experiment branches (rebuttal/experiment/*) stay local; scripts/sync_pod_branches.sh regenerates
# these pod/* mirrors. TSFM_BRANCH overrides (e.g. for a one-off test branch).
case "$EXP" in
  P1) BRANCH="pod/queue-ablation";;
  P6) BRANCH="pod/multi-seed";;
  P7) BRANCH="pod/fine-grained-ablations";;
  *)  echo "ERROR: bad --exp '$EXP' (P1|P6|P7)"; exit 2;;
esac
BRANCH="${TSFM_BRANCH:-$BRANCH}"

# ----------------------------- variant -> TSFM_* env -----------------------------
# Exports the right vars. Fails loudly on an unknown variant so we never pay for a no-op run.
declare -a RUN_ENV
resolve_env() {
  RUN_ENV=( "TSFM_MODEL_SIZE=small_deep" "TSFM_SEED=${SEED}" "ABLATION_NAME=${RUN_TAG}" "TSFM_VISUALIZE=0" )  # viz=0: UMAP deadlocks + is useless for ablations (val-acc metric is kept)
  case "${EXP}:${VARIANT}" in
    # ---- P1 queue ablation: GradCache MUST be off so the queue is actually used ----
    P1:none)            RUN_ENV+=( "TSFM_GRAD_CACHE=0" "TSFM_QUEUE_MODE=none" );;
    P1:hard_neg)        RUN_ENV+=( "TSFM_GRAD_CACHE=0" "TSFM_QUEUE_MODE=hard_neg" "TSFM_MEMORY_BANK_SIZE=512" );;
    P1:semantic)        RUN_ENV+=( "TSFM_GRAD_CACHE=0" "TSFM_QUEUE_MODE=semantic" "TSFM_MEMORY_BANK_SIZE=512" );;
    # ---- P6 multi-seed: same configs at 3 seeds (vary --seed) ----
    P6:headline)        : ;;
    P6:no_soft_targets) RUN_ENV+=( "ABLATION_SOFT_TARGETS=0" );;
    P6:no_queue)        RUN_ENV+=( "TSFM_GRAD_CACHE=0" "TSFM_QUEUE_MODE=none" );;
    # ---- P7 fine-grained ablations ----
    P7:temporal_only)   RUN_ENV+=( 'TSFM_CONFIG_OVERRIDES={"feature_extractor_type":"cnn"}' );;
    P7:channel_indep)   RUN_ENV+=( 'TSFM_CONFIG_OVERRIDES={"use_cross_channel":false}' );;  # encoder cross-channel ATTENTION off (NOT ChannelTextFusion)
    P7:no_channel_text) RUN_ENV+=( "ABLATION_CHANNEL_TEXT_FUSION=0" );;                      # ablate the NOVEL text-as-channel-identity mechanism (justifies the contribution)
    P7:cnn_multi)       RUN_ENV+=( 'TSFM_CONFIG_OVERRIDES={"cnn_kernel_sizes":[3,5,7]}' );;
    P7:spectral_half)   RUN_ENV+=( 'TSFM_CONFIG_OVERRIDES={"spectral_ratio":0.5}' );;
    P7:hard_targets)    RUN_ENV+=( "ABLATION_SOFT_TARGETS=0" );;
    P7:tau_0p3)         RUN_ENV+=( "TSFM_SOFT_TARGET_TEMP=0.3" );;
    P7:soft_weight_0p5) RUN_ENV+=( "TSFM_SOFT_TARGET_WEIGHT=0.5" );;
    P7:sbert_mpnet)     RUN_ENV+=( 'TSFM_CONFIG_OVERRIDES={"contrastive_text_model":"all-mpnet-base-v2","contrastive_text_dim":768,"semantic_dim":768}' );;
    *) echo "ERROR: unknown variant '${VARIANT}' for ${EXP}"; exit 2;;
  esac
}
resolve_env

# ----------------------------- dry run -----------------------------
CMD_PREVIEW="env ${RUN_ENV[*]} PYTHONUNBUFFERED=1 TSFM_NO_COMPILE=1 python training_scripts/human_activity_recognition/semantic_alignment_train.py"
if [[ "$DRY_RUN" == 1 ]]; then
  log "DRY RUN — would do:"
  echo "  branch:   $BRANCH   (git clone $REPO_URL)$([[ $SETUP_ONLY == 1 ]] && echo '  [SETUP-ONLY]')"
  echo "  run tag:  $RUN_TAG"
  echo "  command:  cd $WORKDIR && $CMD_PREVIEW"
  echo "  artifacts -> $ARTIFACT_ROOT/$EXP/${VARIANT}_seed${SEED}/  (+ rclone $RCLONE_REMOTE:$RCLONE_DEST)"
  echo "  terminate: $([[ $NO_TERMINATE == 1 ]] && echo no || echo 'yes (runpodctl remove pod '"${RUNPOD_POD_ID:-?}"')')"
  exit 0
fi

# ----------------------------- 1. setup (code + data + deps) -----------------------------
log "=== $RUN_TAG : setup ==="
fatal() { log "FATAL: $*"; exit 1; }   # every critical setup step is checked explicitly (no silent continue)

# code: git clone/pull the experiment branch from the public origin (code-only branches — nothing
# confidential). git is the source of truth for which branch/commit runs (no stale-code marker
# needed). `reset --hard origin/$BRANCH` supports the local-edit -> push -> pod `git pull` loop.
if [[ -d "$WORKDIR/.git" ]]; then
  git -C "$WORKDIR" fetch --quiet origin "$BRANCH" || fatal "git fetch $BRANCH failed"
  git -C "$WORKDIR" checkout --quiet -B "$BRANCH" "origin/$BRANCH" || fatal "git checkout $BRANCH failed"
  git -C "$WORKDIR" reset --hard --quiet "origin/$BRANCH" || fatal "git reset failed"
else
  git clone --quiet --branch "$BRANCH" --single-branch "$REPO_URL" "$WORKDIR" \
    || fatal "git clone failed ($BRANCH from $REPO_URL) — is the branch pushed to origin?"
fi
cd "$WORKDIR"
log "code: $BRANCH @ $(git -C "$WORKDIR" rev-parse --short HEAD)"

# deps (idempotent; never replace the pre-installed CUDA torch). zip is needed for artifact packaging.
if ! python -c "import sentence_transformers, umap" 2>/dev/null; then
  log "installing deps ..."
  # PEP 668 (Ubuntu 24.04 / py3.12): system Python is "externally managed" and refuses pip without
  # this. Newer pip respects the env var; older pip (Ubuntu 22.04 images) ignores it — so it's safe.
  export PIP_BREAK_SYSTEM_PACKAGES=1
  pip install --upgrade pip -q || fatal "pip upgrade failed"
  pip install -q numpy scipy pandas pyarrow matplotlib plotly scikit-learn umap-learn \
                 tqdm joblib sentence-transformers transformers pydantic requests gdown || fatal "pip deps failed"
  python - <<'PY' 2>/dev/null || true
from sentence_transformers import SentenceTransformer
SentenceTransformer('all-MiniLM-L6-v2'); print("SBERT cached")
PY
fi
command -v zip >/dev/null || { apt-get update -qq && apt-get install -y -qq zip unzip; } 2>/dev/null \
  || log "WARN: could not install zip — will rclone the artifact DIRECTORY instead."

# GPU MUST be live before a multi-hour train. A CPU fallback (semantic_alignment_train falls back
# silently) or a pip-clobbered CUDA torch would burn pod money for days — abort instead.
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || fatal "CUDA not available (no GPU, or pip replaced the CUDA torch). Aborting BEFORE a paid train."
log "GPU OK: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)"

# data (RAM disk). Validate the tarball + use a completion sentinel so a PARTIAL extract is never reused.
DATA_DONE="$DATA_MOUNT/.extract_complete"
if [[ ! -f "$DATA_DONE" ]]; then
  log "downloading training data from HF dataset ($TSFM_HF_DATA_REPO) — CDN-backed, concurrency-safe ..."
  rm -rf "$DATA_MOUNT"; mkdir -p "$DATA_MOUNT"
  pip install -q huggingface_hub >/dev/null 2>&1 || true
  # HF Hub is CDN-backed + built for massive concurrent downloads (unlike Drive, which quota-blocks
  # concurrent pulls of one file). Public repo -> no token needed. Light retry only for network blips.
  _dl_ok=0; DATA_TARBALL=""
  for _att in 1 2 3; do
    _p=$(python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='$TSFM_HF_DATA_REPO', filename='data.tar.gz', repo_type='dataset'))" 2>/dev/null)
    if [[ -n "$_p" && -f "$_p" ]] && tar tzf "$_p" >/dev/null 2>&1; then DATA_TARBALL="$_p"; _dl_ok=1; break; fi
    log "HF data download attempt $_att failed — retry in $((_att*20))s ..."; sleep $((_att * 20))
  done
  [[ "$_dl_ok" == 1 ]] || fatal "HF data download failed after 3 attempts"
  # tarball holds the dataset dirs at top level (no wrapper) -> no --strip-components
  tar xzf "$DATA_TARBALL" --no-same-owner -C "$DATA_MOUNT" || fatal "data extract failed"
  touch "$DATA_DONE"
fi
rm -rf "$WORKDIR/data" 2>/dev/null; ln -sf "$DATA_MOUNT" "$WORKDIR/data"
N_DS=$(ls -d "$WORKDIR"/data/*/sessions 2>/dev/null | wc -l)
[[ "$N_DS" -ge 10 ]] || fatal "only $N_DS datasets present (<10) — data incomplete, aborting before a confounded run."
mkdir -p /workspace/training_output; rm -rf "$WORKDIR/training_output" 2>/dev/null
ln -sf /workspace/training_output "$WORKDIR/training_output"
log "setup complete ($N_DS datasets, GPU live)"

if [[ "$SETUP_ONLY" == 1 ]]; then
  log "--setup-only: code + data + deps + GPU all verified, NOT training. Exiting cleanly."
  exit 0
fi

# ----------------------------- 2-3. train -----------------------------
log "=== $RUN_TAG : training ($CMD_PREVIEW) ==="
set +e
env "${RUN_ENV[@]}" PYTHONUNBUFFERED=1 TSFM_NO_COMPILE=1 \
    python training_scripts/human_activity_recognition/semantic_alignment_train.py \
    2>&1 | tee "/workspace/${RUN_TAG}.log"
TRAIN_RC=${PIPESTATUS[0]}
log "training exited rc=$TRAIN_RC"
# NOTE: deliberately NOT under `set -e` here — a failed glob/cp must not abort before we secure artifacts.

# ----------------------------- 4. retrieve artifacts (ALWAYS, even on failure; non-fatal) -----------------------------
DEST="$ARTIFACT_ROOT/$EXP/${VARIANT}_seed${SEED}"
mkdir -p "$DEST"
RUN_DIR=$(ls -dt "$WORKDIR"/training_output/semantic_alignment/*_ablation_${RUN_TAG} 2>/dev/null | head -1 || true)
if [[ -n "$RUN_DIR" ]]; then
  log "collecting artifacts from $RUN_DIR -> $DEST"
  cp -f "$RUN_DIR/best.pt" "$DEST/" 2>/dev/null \
    || cp -f "$(ls -t "$RUN_DIR"/epoch_*.pt 2>/dev/null | head -1)" "$DEST/best.pt" 2>/dev/null || true
  cp -f "$RUN_DIR/hyperparameters.json" "$DEST/" 2>/dev/null || true
  cp -f "$RUN_DIR/plots/metrics.json" "$DEST/metrics.json" 2>/dev/null || true   # lives in plots/, NOT run root
else
  log "WARNING: no run dir matched *_ablation_${RUN_TAG}"
fi
cp -f "/workspace/${RUN_TAG}.log" "$DEST/train.log" 2>/dev/null || true

# verify the ESSENTIAL artifacts landed — this gates auto-terminate (don't kill a pod that lost its run).
ARTIFACTS_OK=0
if [[ -f "$DEST/best.pt" && -f "$DEST/hyperparameters.json" && -f "$DEST/metrics.json" ]]; then
  ARTIFACTS_OK=1; log "artifacts verified on /workspace: $DEST"
else
  log "WARNING: essential artifacts MISSING in $DEST (need best.pt + hyperparameters.json + metrics.json)."
fi

# package (zip if available; else fall back to a directory rclone so we never depend on zip existing)
HAVE_ZIP=0
if command -v zip >/dev/null; then
  ( cd "$ARTIFACT_ROOT/$EXP" && zip -qr "/workspace/${RUN_TAG}.zip" "${VARIANT}_seed${SEED}" ) && HAVE_ZIP=1 || log "WARN: zip failed"
fi

# push to Drive (best-effort). DRIVE_OK gates auto-terminate unless ALLOW_LOCAL_ONLY_TERMINATE=1.
DRIVE_OK=0
if command -v rclone >/dev/null && rclone listremotes 2>/dev/null | grep -q "^${RCLONE_REMOTE}:"; then
  if [[ "$HAVE_ZIP" == 1 ]]; then
    rclone copy "/workspace/${RUN_TAG}.zip" "${RCLONE_REMOTE}:${RCLONE_DEST}/" && DRIVE_OK=1 || log "WARN: rclone zip push failed"
  else
    rclone copy "$DEST" "${RCLONE_REMOTE}:${RCLONE_DEST}/${EXP}/${VARIANT}_seed${SEED}/" && DRIVE_OK=1 || log "WARN: rclone dir push failed"
  fi
  [[ "$DRIVE_OK" == 1 ]] && log "artifacts pushed to ${RCLONE_REMOTE}:${RCLONE_DEST}/"
else
  log "rclone remote '${RCLONE_REMOTE}' not configured — artifacts on /workspace only (retrieve via scripts/runpod_fetch.sh / runpodctl send)."
fi

# ----------------------------- 5. auto-terminate (stop billing) — only when artifacts are SAFE -----------------------------
if [[ "$TRAIN_RC" -ne 0 ]]; then
  log "training FAILED (rc=$TRAIN_RC) — leaving pod ALIVE for SSH debugging. Not terminating."; exit "$TRAIN_RC"
fi
if [[ "$NO_TERMINATE" == 1 ]]; then log "--no-terminate set — pod left running."; exit 0; fi
if [[ "$ARTIFACTS_OK" != 1 ]]; then
  log "ARTIFACTS NOT VERIFIED — leaving pod ALIVE so the run isn't lost. Inspect $DEST, then terminate manually."; exit 0
fi
if [[ "$DRIVE_OK" != 1 && "${ALLOW_LOCAL_ONLY_TERMINATE:-0}" != 1 ]]; then
  log "Artifacts are on /workspace but NOT confirmed on Drive — leaving pod ALIVE (a released volume would lose them)."
  log "  Retrieve now ('runpodctl send /workspace/${RUN_TAG}.zip'), or set ALLOW_LOCAL_ONLY_TERMINATE=1 to terminate anyway."
  exit 0
fi
log "SUCCESS + artifacts safe — terminating pod ${RUNPOD_POD_ID:-?} to stop billing."
if command -v runpodctl >/dev/null && [[ -n "${RUNPOD_POD_ID:-}" ]]; then
  runpodctl remove pod "$RUNPOD_POD_ID" && exit 0
fi
if [[ -n "${RUNPOD_POD_ID:-}" && -n "${RUNPOD_API_KEY:-}" ]]; then
  curl -s -X DELETE "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" \
       -H "Authorization: Bearer ${RUNPOD_API_KEY}" >/dev/null && exit 0
fi
log "WARNING: could not auto-terminate (no runpodctl / RUNPOD_POD_ID / RUNPOD_API_KEY). TERMINATE MANUALLY."
exit 0
exit 0
