#!/bin/bash
# Auto-evaluation pipeline: waits for TSFM training to complete, then runs all evaluations.
# Designed to run in a tmux session.
#
# Usage: bash scripts/auto_eval_after_training.sh
#
# What it does:
#   1. Monitors training directory for best.pt checkpoint (polls every 60s)
#   2. Waits for training process to exit (no more epoch files being written)
#   3. Updates TSFM evaluator checkpoint path to the latest run
#   4. Runs all 5 baseline evaluations sequentially (4-metric: ZS-open, ZS-close, 1% FT, 10% FT)
#   5. Generates combined results table
#   6. Copies results into docs/baselines/RESULTS.md

set -e
cd /home/alex/code/tsfm
source .venv/bin/activate

TRAINING_DIR="training_output/semantic_alignment/20260217_113136"
EVAL_SCRIPT="val_scripts/human_activity_recognition/evaluate_tsfm.py"
RESULTS_MD="docs/baselines/RESULTS.md"
RESULTS_TABLE="test_output/baseline_evaluation/results_table.md"

echo "============================================"
echo "  Auto-Evaluation Pipeline"
echo "  Started: $(date)"
echo "  Monitoring: $TRAINING_DIR"
echo "============================================"

# ── Step 1: Wait for training to complete ──
echo ""
echo ">>> Waiting for training to complete..."
echo "    (Checking every 60 seconds for training process to finish)"

while true; do
    # Check if training is still running (look for the python training process)
    if pgrep -f "semantic_alignment_train.py" > /dev/null 2>&1; then
        # Training still running - show latest epoch
        latest_epoch=$(ls -1 "$TRAINING_DIR"/epoch_*.pt 2>/dev/null | sort -t_ -k2 -n | tail -1)
        if [ -n "$latest_epoch" ]; then
            epoch_num=$(basename "$latest_epoch" | sed 's/epoch_//;s/\.pt//')
            echo "    [$(date +%H:%M:%S)] Training in progress - latest checkpoint: epoch $epoch_num/100"
        else
            echo "    [$(date +%H:%M:%S)] Training in progress - no checkpoints yet"
        fi
        sleep 60
    else
        echo ""
        echo ">>> Training process exited at $(date)"
        # Give filesystem a moment to flush
        sleep 5
        break
    fi
done

# Verify we have a best.pt checkpoint
if [ ! -f "$TRAINING_DIR/best.pt" ]; then
    echo "ERROR: No best.pt found in $TRAINING_DIR"
    echo "Training may have failed. Check logs."
    exit 1
fi

# Show final epoch count
latest_epoch=$(ls -1 "$TRAINING_DIR"/epoch_*.pt 2>/dev/null | sort -t_ -k2 -n | tail -1)
epoch_num=$(basename "$latest_epoch" | sed 's/epoch_//;s/\.pt//')
echo ">>> Training completed with $epoch_num epochs."
echo ">>> Best checkpoint: $TRAINING_DIR/best.pt"

# ── Step 2: Update TSFM checkpoint path ──
echo ""
echo ">>> Updating TSFM evaluator checkpoint path..."
sed -i "s|CHECKPOINT_PATH = \"training_output/semantic_alignment/[^\"]*\"|CHECKPOINT_PATH = \"$TRAINING_DIR/best.pt\"|" "$EVAL_SCRIPT"
echo "    Updated $EVAL_SCRIPT"

# ── Step 3: Run all evaluations ──
echo ""
echo "============================================"
echo "  Running All Evaluations"
echo "  $(date)"
echo "============================================"

# Delete any stale TSFM cache to force fresh evaluation
rm -f test_output/baseline_evaluation/tsfm_*.pt 2>/dev/null
rm -f test_output/baseline_evaluation/tsfm_*.json 2>/dev/null
# Also delete stale evaluation result JSONs (fine-tuning will produce different results)
rm -f test_output/baseline_evaluation/*_evaluation.json 2>/dev/null

echo ""
echo ">>> [1/6] TSFM evaluation..."
python val_scripts/human_activity_recognition/evaluate_tsfm.py 2>&1 | tee /tmp/eval_tsfm.log
echo ">>> TSFM done at $(date)."

echo ""
echo ">>> [2/6] LiMU-BERT evaluation..."
python val_scripts/human_activity_recognition/evaluate_limubert.py 2>&1 | tee /tmp/eval_limubert.log
echo ">>> LiMU-BERT done at $(date)."

echo ""
echo ">>> [3/6] MOMENT evaluation..."
python val_scripts/human_activity_recognition/evaluate_moment.py 2>&1 | tee /tmp/eval_moment.log
echo ">>> MOMENT done at $(date)."

echo ""
echo ">>> [4/6] CrossHAR evaluation..."
python val_scripts/human_activity_recognition/evaluate_crosshar.py 2>&1 | tee /tmp/eval_crosshar.log
echo ">>> CrossHAR done at $(date)."

echo ""
echo ">>> [5/6] LanHAR evaluation (training + eval - longest)..."
python val_scripts/human_activity_recognition/evaluate_lanhar.py 2>&1 | tee /tmp/eval_lanhar.log
echo ">>> LanHAR done at $(date)."

echo ""
echo ">>> [6/6] Generating combined results table..."
python scripts/generate_results_table.py

# ── Step 4: Update RESULTS.md ──
echo ""
echo ">>> Updating docs/baselines/RESULTS.md with fresh results..."
if [ -f "$RESULTS_TABLE" ]; then
    cp "$RESULTS_TABLE" /tmp/results_table_backup.md
    echo "    Generated table saved to $RESULTS_TABLE"
    echo ""
    echo "============================================"
    echo "  RESULTS PREVIEW"
    echo "============================================"
    cat "$RESULTS_TABLE"
fi

echo ""
echo "============================================"
echo "  All evaluations complete!"
echo "  Finished: $(date)"
echo "  Results: test_output/baseline_evaluation/"
echo "  Table: $RESULTS_TABLE"
echo "============================================"
echo ""
echo "NOTE: Review results and run 'python scripts/generate_results_table.py' to regenerate if needed."
echo "Then update docs/baselines/RESULTS.md with the final numbers."
