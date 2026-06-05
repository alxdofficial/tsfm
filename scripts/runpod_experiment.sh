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
#   P7: temporal_only | channel_indep | cnn_multi | spectral_half | hard_targets |
#       tau_0p3 | soft_weight_0p5 | sbert_mpnet
#
# Required env (export before running, or bake into the pod template):
#   RUNPOD_POD_ID         present automatically in every RunPod container
#   CODE_GDRIVE_ID_<EXP>  Google-Drive file id of the code archive for that experiment's branch,
#                         e.g. CODE_GDRIVE_ID_P1, CODE_GDRIVE_ID_P6, CODE_GDRIVE_ID_P7
#                         (created on the dev box by scripts/make_code_bundle.sh, then uploaded)
#   DATA_GDRIVE_ID        Google-Drive file id of the training-data tarball
#                         (default = the existing 1a6QROP9... data tarball)
# Optional env:
#   RUNPOD_API_KEY        used only as a fallback for termination if runpodctl is unavailable
#   RCLONE_REMOTE         name of a configured rclone remote for Drive artifact upload (default: gdrive)
#   RCLONE_DEST           rclone dest path (default: tsfm_rebuttal/artifacts)
#   TSFM_EPOCHS           override epochs (e.g. 1 for a smoke run; default 100)
# =============================================================================================
set -uo pipefail

# ----------------------------- defaults -----------------------------
EXP=""; VARIANT=""; SEED=42; DRY_RUN=0; NO_TERMINATE=0
WORKDIR="${TSFM_WORKDIR:-/workspace/tsfm}"
DATA_MOUNT="${TSFM_DATA_MOUNT:-/dev/shm/tsfm_data}"
ARTIFACT_ROOT="${TSFM_ARTIFACT_ROOT:-/workspace/artifacts}"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"
RCLONE_DEST="${RCLONE_DEST:-tsfm_rebuttal/artifacts}"
DATA_GDRIVE_ID="${DATA_GDRIVE_ID:-1a6QROP9qZZetOek_NxbgIWFNYDVY8d0H}"

# ----------------------------- arg parsing -----------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp)          EXP="$2"; shift 2;;
    --variant)      VARIANT="$2"; shift 2;;
    --seed)         SEED="$2"; shift 2;;
    --dry-run)      DRY_RUN=1; shift;;
    --no-terminate) NO_TERMINATE=1; shift;;
    -h|--help)      sed -n '2,40p' "$0"; exit 0;;
    *) echo "unknown arg: $1 (see --help)"; exit 2;;
  esac
done
[[ -z "$EXP" || -z "$VARIANT" ]] && { echo "ERROR: --exp <P1|P6|P7> and --variant <name> are required (see --help)"; exit 2; }

RUN_TAG="${EXP}_${VARIANT}_seed${SEED}"
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ----------------------------- exp -> branch + code id -----------------------------
case "$EXP" in
  P1) BRANCH="rebuttal/experiment/queue-ablation";         CODE_ID="${CODE_GDRIVE_ID_P1:-}";;
  P6) BRANCH="rebuttal/experiment/multi-seed";             CODE_ID="${CODE_GDRIVE_ID_P6:-}";;
  P7) BRANCH="rebuttal/experiment/fine-grained-ablations"; CODE_ID="${CODE_GDRIVE_ID_P7:-}";;
  *)  echo "ERROR: bad --exp '$EXP' (P1|P6|P7)"; exit 2;;
esac

# ----------------------------- variant -> TSFM_* env -----------------------------
# Exports the right vars. Fails loudly on an unknown variant so we never pay for a no-op run.
declare -a RUN_ENV
resolve_env() {
  RUN_ENV=( "TSFM_MODEL_SIZE=small_deep" "TSFM_SEED=${SEED}" "ABLATION_NAME=${RUN_TAG}" )
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
    P7:channel_indep)   RUN_ENV+=( 'TSFM_CONFIG_OVERRIDES={"use_cross_channel":false}' );;
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
  echo "  branch:   $BRANCH   (code id: ${CODE_ID:-<unset CODE_GDRIVE_ID_${EXP}>})"
  echo "  run tag:  $RUN_TAG"
  echo "  command:  cd $WORKDIR && $CMD_PREVIEW"
  echo "  artifacts -> $ARTIFACT_ROOT/$EXP/${VARIANT}_seed${SEED}/  (+ rclone $RCLONE_REMOTE:$RCLONE_DEST)"
  echo "  terminate: $([[ $NO_TERMINATE == 1 ]] && echo no || echo 'yes (runpodctl remove pod '"${RUNPOD_POD_ID:-?}"')')"
  exit 0
fi

# ----------------------------- 1. setup (code + data + deps) -----------------------------
log "=== $RUN_TAG : setup ==="
# code: prefer the Drive archive (no git auth); fall back to an existing checkout.
if [[ ! -d "$WORKDIR/training_scripts" ]]; then
  if [[ -n "$CODE_ID" ]]; then
    log "downloading code archive ($BRANCH) from Drive id $CODE_ID ..."
    pip install -q gdown 2>/dev/null || true
    gdown "$CODE_ID" -O /tmp/tsfm_code.tar.gz || { log "FATAL: code download failed"; exit 1; }
    mkdir -p "$WORKDIR"; tar xzf /tmp/tsfm_code.tar.gz -C "$WORKDIR"; rm -f /tmp/tsfm_code.tar.gz
  else
    log "FATAL: no code at $WORKDIR and CODE_GDRIVE_ID_${EXP} unset."
    log "  Create it on the dev box: scripts/make_code_bundle.sh, upload to Drive, export CODE_GDRIVE_ID_${EXP}."
    exit 1
  fi
fi
cd "$WORKDIR"

# deps (idempotent; never touch the pre-installed torch)
if ! python -c "import sentence_transformers, umap" 2>/dev/null; then
  log "installing deps (preserving system torch) ..."
  pip install --upgrade pip -q
  pip install -q numpy scipy pandas pyarrow matplotlib plotly scikit-learn umap-learn \
                 tqdm joblib sentence-transformers transformers pydantic requests gdown
  python - <<'PY' 2>/dev/null || true
from sentence_transformers import SentenceTransformer
SentenceTransformer('all-MiniLM-L6-v2'); print("SBERT cached")
PY
fi

# data (RAM disk; guarded so a re-run is instant)
if [[ ! -d "$DATA_MOUNT/uci_har/sessions" ]]; then
  log "downloading + extracting training data from Drive id $DATA_GDRIVE_ID ..."
  mkdir -p "$DATA_MOUNT"
  [[ -f /tmp/tsfm_data.tar.gz ]] || gdown "$DATA_GDRIVE_ID" -O /tmp/tsfm_data.tar.gz
  tar xzf /tmp/tsfm_data.tar.gz --no-same-owner -C "$DATA_MOUNT" --strip-components=1
  rm -f /tmp/tsfm_data.tar.gz
fi
rm -rf "$WORKDIR/data" 2>/dev/null; ln -sf "$DATA_MOUNT" "$WORKDIR/data"
# checkpoints land on the persistent volume
mkdir -p /workspace/training_output; rm -rf "$WORKDIR/training_output" 2>/dev/null
ln -sf /workspace/training_output "$WORKDIR/training_output"
log "setup complete ($(ls -d "$WORKDIR"/data/*/sessions 2>/dev/null | wc -l) datasets present)"

# ----------------------------- 2-3. train -----------------------------
log "=== $RUN_TAG : training ($CMD_PREVIEW) ==="
set +e
env "${RUN_ENV[@]}" PYTHONUNBUFFERED=1 TSFM_NO_COMPILE=1 \
    python training_scripts/human_activity_recognition/semantic_alignment_train.py \
    2>&1 | tee "/workspace/${RUN_TAG}.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
log "training exited rc=$TRAIN_RC"

# ----------------------------- 4. retrieve artifacts (ALWAYS, even on failure) -----------------------------
DEST="$ARTIFACT_ROOT/$EXP/${VARIANT}_seed${SEED}"
mkdir -p "$DEST"
RUN_DIR=$(ls -dt "$WORKDIR"/training_output/semantic_alignment/*_ablation_${RUN_TAG} 2>/dev/null | head -1)
if [[ -n "$RUN_DIR" ]]; then
  log "collecting artifacts from $RUN_DIR -> $DEST"
  cp -f "$RUN_DIR/best.pt" "$DEST/" 2>/dev/null || cp -f "$(ls -t "$RUN_DIR"/epoch_*.pt 2>/dev/null | head -1)" "$DEST/" 2>/dev/null || true
  cp -f "$RUN_DIR/hyperparameters.json" "$DEST/" 2>/dev/null || true
  cp -f "$RUN_DIR/plots/metrics.json" "$DEST/metrics.json" 2>/dev/null || true   # NOTE: lives in plots/, not run root
else
  log "WARNING: no run dir matched *_ablation_${RUN_TAG} — copying log only"
fi
cp -f "/workspace/${RUN_TAG}.log" "$DEST/train.log" 2>/dev/null || true
( cd "$ARTIFACT_ROOT/$EXP" && zip -qr "/workspace/${RUN_TAG}.zip" "${VARIANT}_seed${SEED}" ) 2>/dev/null || true
log "artifacts persisted on /workspace (survives termination): $DEST"

# best-effort push to Drive (non-fatal). Requires `rclone config` with a remote named $RCLONE_REMOTE.
if command -v rclone >/dev/null && rclone listremotes 2>/dev/null | grep -q "^${RCLONE_REMOTE}:"; then
  log "pushing $RUN_TAG.zip to ${RCLONE_REMOTE}:${RCLONE_DEST}/ ..."
  rclone copy "/workspace/${RUN_TAG}.zip" "${RCLONE_REMOTE}:${RCLONE_DEST}/" 2>&1 | tail -2 || log "rclone push failed (non-fatal)"
else
  log "rclone remote '${RCLONE_REMOTE}' not configured — artifacts are on /workspace only."
  log "  retrieve with: scripts/runpod_fetch.sh  (or 'runpodctl send /workspace/${RUN_TAG}.zip' from this pod)."
fi

# ----------------------------- 5. auto-terminate (stop billing) -----------------------------
if [[ "$TRAIN_RC" -ne 0 ]]; then
  log "training FAILED (rc=$TRAIN_RC) — leaving the pod ALIVE for SSH debugging. Not terminating."
  exit "$TRAIN_RC"
fi
if [[ "$NO_TERMINATE" == 1 ]]; then
  log "--no-terminate set — done; pod left running."
  exit 0
fi
log "SUCCESS — terminating pod ${RUNPOD_POD_ID:-?} to stop billing (artifacts are safe on /workspace + Drive)."
if command -v runpodctl >/dev/null && [[ -n "${RUNPOD_POD_ID:-}" ]]; then
  runpodctl remove pod "$RUNPOD_POD_ID" && exit 0
fi
# fallback to the REST API (needs RUNPOD_API_KEY)
if [[ -n "${RUNPOD_POD_ID:-}" && -n "${RUNPOD_API_KEY:-}" ]]; then
  curl -s -X DELETE "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" \
       -H "Authorization: Bearer ${RUNPOD_API_KEY}" >/dev/null && exit 0
fi
log "WARNING: could not auto-terminate (no runpodctl / RUNPOD_POD_ID / RUNPOD_API_KEY). TERMINATE MANUALLY."
exit 0
