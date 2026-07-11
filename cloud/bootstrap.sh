#!/usr/bin/env bash
# HALO cloud-burst POD bootstrap. Runs ON the pod (vast --onstart, or piped over ssh by fleet.py).
# Self-contained: arms a self-destruct watchdog FIRST, then clones the public repo, installs deps
# (pip-at-boot), pulls the R2 data bundle + job backbone, runs the job in tmux, and syncs results
# back to R2. Idempotent-ish; safe to re-run.
#
# Required env (injected at pod-create time; NEVER baked into an image):
#   R2_BUCKET R2_ENDPOINT R2_PREFIX AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
#   HALO_JOB HALO_RUN_ID REPO_URL REPO_SHA
# Optional: MAX_HOURS (default 8), HALO_WORK (default ~/halo)
set -euo pipefail

: "${HALO_JOB:?set HALO_JOB}"; : "${HALO_RUN_ID:?set HALO_RUN_ID}"
: "${REPO_URL:?set REPO_URL}"; : "${REPO_SHA:?set REPO_SHA}"
: "${R2_BUCKET:?}"; : "${R2_ENDPOINT:?}"; : "${AWS_ACCESS_KEY_ID:?}"; : "${AWS_SECRET_ACCESS_KEY:?}"
export R2_PREFIX="${R2_PREFIX:-halo}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-auto}"   # R2 needs a region set or aws-cli errors
MAX_HOURS="${MAX_HOURS:-8}"
WORK="${HALO_WORK:-$HOME/halo}"
DEST="s3://$R2_BUCKET/$R2_PREFIX/runs/$HALO_RUN_ID/$HALO_JOB"

log(){ echo "[bootstrap $(date -u +%H:%M:%S)] $*"; }
r2(){ aws s3 --endpoint-url "$R2_ENDPOINT" "$@"; }

# 0) SAFETY FIRST — self-destruct watchdog before doing anything that can hang.
log "arming watchdog: hard poweroff after ${MAX_HOURS}h (billing backstop)"
nohup bash -c "sleep $((MAX_HOURS*3600)); echo WATCHDOG-MAXHOURS; (sudo poweroff -f || poweroff -f || shutdown -h now)" \
  >/tmp/halo_watchdog.log 2>&1 & disown || true

# 1) system deps — python:3.11-slim is bare, so install what torch/git/aws need.
export DEBIAN_FRONTEND=noninteractive
apt-get update -y -qq
apt-get install -y -qq --no-install-recommends git tmux curl ca-certificates libgomp1
command -v aws  >/dev/null || pip install -q awscli
pip install -q boto3 >/dev/null 2>&1 || true

# 2) clone the PUBLIC repo at the pinned SHA
log "clone $REPO_URL @ ${REPO_SHA:0:12}"
rm -rf "$WORK"; git clone --quiet "$REPO_URL" "$WORK"
cd "$WORK"; git checkout --quiet "$REPO_SHA"

# 3) python env — torch stack from the cu128 index, then the frozen core reqs, then per-job deps.
log "install torch stack (cu128) + core requirements"
pip install -q --index-url https://download.pytorch.org/whl/cu128 torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
r2 cp "s3://$R2_BUCKET/$R2_PREFIX/meta/requirements-core.txt" /tmp/req-core.txt
pip install -q -r /tmp/req-core.txt
# CLIP (git) only matters for unimts; harmless elsewhere but keep it job-gated via the recipe.
log "apply recipe deps + backbone for job=$HALO_JOB"
python cloud/apply_recipe.py --job "$HALO_JOB" --install

# 4) pull the processed data bundle and extract at repo root
BUNDLE=$(r2 cp "s3://$R2_BUCKET/$R2_PREFIX/data/bundle-latest.txt" - | tr -d '[:space:]')
log "pull + extract data bundle: $BUNDLE"
r2 cp "s3://$R2_BUCKET/$R2_PREFIX/data/$BUNDLE" "/tmp/$BUNDLE"
tar xzf "/tmp/$BUNDLE" -C "$WORK"

# 5) run the job in a detached tmux session; capture exit code + log
CMD="$(python cloud/apply_recipe.py --job "$HALO_JOB" --print-cmd)"
log "launch training in tmux: $CMD"
rm -f "$HOME/EXIT_CODE" "$HOME/DONE" "$HOME/FAILED"
tmux new-session -d -s halo "bash -lc 'cd $WORK && ($CMD) > $HOME/train.log 2>&1; echo \$? > $HOME/EXIT_CODE'"
log "waiting for job to finish (poll EXIT_CODE)…"
while [ ! -f "$HOME/EXIT_CODE" ]; do sleep 15; done
CODE="$(cat "$HOME/EXIT_CODE")"
if [ "$CODE" = "0" ]; then touch "$HOME/DONE"; else touch "$HOME/FAILED"; fi
log "job finished exit=$CODE"

# 6) sync results back to R2 (log + recipe result globs + sentinel)
log "sync results -> $DEST"
r2 cp "$HOME/train.log" "$DEST/train.log" || true
python cloud/apply_recipe.py --job "$HALO_JOB" --sync-results "$R2_PREFIX/runs/$HALO_RUN_ID/$HALO_JOB" || true
if [ -f "$HOME/DONE" ]; then r2 cp "$HOME/DONE" "$DEST/DONE"; else r2 cp "$HOME/FAILED" "$DEST/FAILED"; fi
log "bootstrap complete (exit=$CODE). fleet.py will destroy; watchdog is the backstop."
exit "$CODE"
