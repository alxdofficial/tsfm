#!/bin/bash
# Run ablation studies for TSFM.
#
# Usage:
#   ./scripts/run_ablations.sh <ablation_name>    # Run single ablation
#   ./scripts/run_ablations.sh all                 # Run all 5 + baseline sequentially
#   ./scripts/run_ablations.sh parallel            # Run all 5 + baseline in parallel (tmux)
#
# Ablation names:
#   baseline             All components enabled (control)
#   no_channel_fusion    Disable channel-text fusion (gated cross-attention)
#   no_label_bank        Use frozen mean-pooled SBERT instead of learnable label bank
#   no_soft_targets      Disable soft targets + batch-mean similarity normalization
#   no_signal_aug        Disable jitter + scale augmentation
#   no_text_aug          Disable label synonyms/templates + Hz/window suffix
#
# Environment overrides (apply to all runs):
#   TSFM_BATCH_SIZE=32   Per-GPU micro-batch size
#   TSFM_ACCUM_STEPS=16  Gradient accumulation steps
#   TSFM_LR=8e-5         Learning rate
#   TSFM_GRAD_CACHE=1    Use GradCache (default: on)

set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_SCRIPT="training_scripts/human_activity_recognition/semantic_alignment_train.py"

# Ablation configurations: name -> env vars
declare -A ABLATION_ENVS
ABLATION_ENVS[baseline]=""
ABLATION_ENVS[no_channel_fusion]="ABLATION_CHANNEL_TEXT_FUSION=0"
ABLATION_ENVS[no_label_bank]="ABLATION_LEARNABLE_LABEL_BANK=0"
ABLATION_ENVS[no_soft_targets]="ABLATION_SOFT_TARGETS=0"
ABLATION_ENVS[no_signal_aug]="ABLATION_SIGNAL_AUG=0"
ABLATION_ENVS[no_text_aug]="ABLATION_TEXT_AUG=0"

run_ablation() {
    local name="$1"
    local envs="${ABLATION_ENVS[$name]}"
    echo "=== Starting ablation: $name ==="
    echo "  Env: ABLATION_NAME=$name $envs"
    env ABLATION_NAME="$name" $envs python "$TRAIN_SCRIPT"
}

# Collect TSFM_* env vars to forward to tmux sub-sessions
_forward_envs() {
    env | grep '^TSFM_' | tr '\n' ' '
}

run_in_tmux() {
    local name="$1"
    local envs="${ABLATION_ENVS[$name]}"
    local session="ablation_${name}"
    local tsfm_envs="$(_forward_envs)"

    # Kill existing session if any
    tmux kill-session -t "$session" 2>/dev/null || true

    tmux new-session -d -s "$session" \
        "cd $(pwd) && $tsfm_envs ABLATION_NAME=$name $envs python $TRAIN_SCRIPT 2>&1 | tee training_output/ablation_${name}.log; echo 'DONE: $name'; read"
    echo "  Started tmux session: $session"
}

case "${1:-}" in
    baseline|no_channel_fusion|no_label_bank|no_soft_targets|no_signal_aug|no_text_aug)
        run_ablation "$1"
        ;;
    all)
        for name in baseline no_channel_fusion no_label_bank no_soft_targets no_signal_aug no_text_aug; do
            run_ablation "$name"
        done
        ;;
    parallel)
        echo "=== Launching all ablations in parallel tmux sessions ==="
        for name in baseline no_channel_fusion no_label_bank no_soft_targets no_signal_aug no_text_aug; do
            run_in_tmux "$name"
        done
        echo ""
        echo "All sessions launched. Monitor with:"
        echo "  tmux ls                          # List sessions"
        echo "  tmux attach -t ablation_baseline # Attach to a session"
        echo "  tail -f training_output/ablation_*.log  # Follow logs"
        ;;
    status)
        echo "=== Ablation run status ==="
        for name in baseline no_channel_fusion no_label_bank no_soft_targets no_signal_aug no_text_aug; do
            session="ablation_${name}"
            if tmux has-session -t "$session" 2>/dev/null; then
                echo "  $name: RUNNING (tmux: $session)"
            else
                echo "  $name: not running"
            fi
        done
        ;;
    *)
        echo "Usage: $0 <ablation_name|all|parallel|status>"
        echo ""
        echo "Ablations: baseline, no_channel_fusion, no_label_bank, no_soft_targets, no_signal_aug, no_text_aug"
        echo "Commands: all (sequential), parallel (tmux), status (check tmux sessions)"
        exit 1
        ;;
esac
