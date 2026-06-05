#!/usr/bin/env bash
# =============================================================================================
# sync_pod_branches.sh — publish CODE-ONLY mirrors of the experiment branches for the pod to clone.
#
# Why: the full experiment branches (rebuttal/experiment/*) carry paper-rebuttal/ (the reviews +
# strategy — CONFIDENTIAL). The pod only needs code. This regenerates pod/<name> = <experiment
# branch> MINUS paper-rebuttal/, and force-pushes it to the PUBLIC origin. The pod's runner clones
# pod/<name>. Re-run this after any local code change to ship it (the local-edit -> push -> pod-pull
# loop): fix on the experiment branch, run this, then `git pull` on the pod.
#
# Safety: aborts the push if ANYTHING under paper-rebuttal/ (or any obvious confidential file) is
# still tracked on the mirror.
# Usage: bash scripts/sync_pod_branches.sh            (all experiments)
#        bash scripts/sync_pod_branches.sh P1         (one)
# =============================================================================================
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

declare -A SRC=(
  [P1]=rebuttal/experiment/queue-ablation
  [P6]=rebuttal/experiment/multi-seed
  [P7]=rebuttal/experiment/fine-grained-ablations
)
declare -A DST=( [P1]=pod/queue-ablation [P6]=pod/multi-seed [P7]=pod/fine-grained-ablations )

WANT=("$@"); [[ ${#WANT[@]} -eq 0 ]] && WANT=(P1 P6 P7)
START=$(git rev-parse --abbrev-ref HEAD)
[[ -z "$(git status --porcelain)" ]] || { echo "ABORT: working tree dirty — commit/stash first."; exit 1; }

for exp in "${WANT[@]}"; do
  src="${SRC[$exp]:-}"; dst="${DST[$exp]:-}"
  [[ -n "$src" && -n "$dst" ]] || { echo "skip unknown exp '$exp'"; continue; }
  echo "=== $exp : $src -> $dst (code-only) ==="
  git checkout -B "$dst" "$src" >/dev/null 2>&1
  if git ls-files --error-unmatch paper-rebuttal >/dev/null 2>&1; then
    git rm -r -q --cached paper-rebuttal
    git commit -q -m "code-only mirror of $src (strip confidential paper-rebuttal/) for public pod clone"
  fi
  # hard safety gate: never push if anything confidential remains tracked
  if git ls-files | grep -qiE "paper-rebuttal|reviews\.md|rebuttal_plan|SUPERVISOR"; then
    echo "  ABORT: confidential file still tracked on $dst — NOT pushing."; git checkout "$START" >/dev/null 2>&1; exit 1
  fi
  git push -f origin "$dst"
  echo "  pushed $dst @ $(git rev-parse --short HEAD)"
done

git checkout "$START" >/dev/null 2>&1
echo "done. Pod clones: pod/<name> from $(git remote get-url origin)"
