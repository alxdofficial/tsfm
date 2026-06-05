#!/usr/bin/env bash
#
# runpod_fetch.sh — pull RunPod ablation artifacts into paper-rebuttal/ (RUN ON THE DEV BOX).
# =============================================================================================
# Downloads the artifact zips the campaign produced and unpacks them into
#   paper-rebuttal/experiments/runpod_artifacts/<EXP>/<variant>_seed<N>/
# so checkpoints + metrics + logs are human-findable, co-located by experiment.
#
# Primary path: an rclone remote (default 'gdrive') the pods pushed to.
# Fallback:     runpodctl send/receive (the pod prints a code; run `runpodctl receive <code>` here).
# =============================================================================================
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
DEST="paper-rebuttal/experiments/runpod_artifacts"
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"
RCLONE_SRC="${RCLONE_SRC:-tsfm_rebuttal/artifacts}"
mkdir -p "$DEST/zips"

if command -v rclone >/dev/null && rclone listremotes 2>/dev/null | grep -q "^${RCLONE_REMOTE}:"; then
  echo "Pulling artifact zips from ${RCLONE_REMOTE}:${RCLONE_SRC}/ ..."
  rclone copy "${RCLONE_REMOTE}:${RCLONE_SRC}/" "$DEST/zips/" --include "*.zip" -P
  shopt -s nullglob
  for z in "$DEST"/zips/*.zip; do
    name=$(basename "$z" .zip)          # e.g. P1_semantic_seed42
    exp="${name%%_*}"                    # P1 / P6 / P7
    mkdir -p "$DEST/$exp"
    unzip -oq "$z" -d "$DEST/$exp/" || { echo "  WARN: failed to unpack $name (corrupt zip?) — skipping"; continue; }
    echo "  unpacked $name -> $DEST/$exp/"
  done
  echo "Done. Artifacts under: $DEST/"
else
  echo "rclone remote '${RCLONE_REMOTE}' not configured."
  echo "Fallback: on the pod run   runpodctl send /workspace/<RUN_TAG>.zip"
  echo "          then here run     runpodctl receive <code>"
  echo "          and unzip into    $DEST/<EXP>/"
fi
