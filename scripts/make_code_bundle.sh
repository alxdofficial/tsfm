#!/usr/bin/env bash
#
# make_code_bundle.sh — create per-experiment code archives for RunPod (RUN ON THE DEV BOX).
# =============================================================================================
# The rebuttal branches are CONFIDENTIAL and must NOT be pushed to the public GitHub remote, so
# pods can't `git clone` them. Instead we ship each experiment branch's working tree (no .git
# history, ~a few MB) as a tarball via Google Drive. Upload the tarballs (private) and pass their
# Drive file ids to the pod as CODE_GDRIVE_ID_P1 / _P6 / _P7.
#
# Usage:
#   bash scripts/make_code_bundle.sh                 # writes /tmp/tsfm_code_bundles/*.tar.gz
#   RCLONE_UPLOAD=1 bash scripts/make_code_bundle.sh # also uploads via an rclone 'gdrive' remote
# =============================================================================================
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT="${1:-/tmp/tsfm_code_bundles}"; mkdir -p "$OUT"

declare -A BRANCHES=(
  [P1]=rebuttal/experiment/queue-ablation
  [P6]=rebuttal/experiment/multi-seed
  [P7]=rebuttal/experiment/fine-grained-ablations
)
echo "Creating code archives (working tree only, no history):"
for exp in P1 P6 P7; do
  br="${BRANCHES[$exp]}"
  out="$OUT/tsfm_code_${exp}.tar.gz"
  git archive --format=tar.gz -o "$out" "$br"
  echo "  $exp  $br  ->  $out  ($(du -h "$out" | cut -f1))"
done

echo
echo "Next: upload these to Google Drive (private) and, on each pod, export the file ids:"
echo "  export CODE_GDRIVE_ID_P1=<id> CODE_GDRIVE_ID_P6=<id> CODE_GDRIVE_ID_P7=<id>"
if [[ "${RCLONE_UPLOAD:-0}" == 1 ]] && command -v rclone >/dev/null && rclone listremotes | grep -q '^gdrive:'; then
  echo "Uploading to gdrive:tsfm_rebuttal/code/ ..."
  for exp in P1 P6 P7; do rclone copy "$OUT/tsfm_code_${exp}.tar.gz" "gdrive:tsfm_rebuttal/code/" -P; done
  echo "Uploaded — grab the per-file share ids from Drive (right-click → share)."
fi
