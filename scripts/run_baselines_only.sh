#!/bin/bash
# Run the 4 baseline evaluations (skip TSFM - no checkpoint available)
# Usage: bash scripts/run_baselines_only.sh

set -e
cd /home/alex/code/tsfm
source .venv/bin/activate

echo "============================================"
echo "  Running 4 Baseline Evaluations"
echo "  (TSFM skipped - no checkpoint)"
echo "  $(date)"
echo "============================================"

# 1. LiMU-BERT evaluation
echo ""
echo ">>> [1/4] LiMU-BERT evaluation..."
python val_scripts/human_activity_recognition/evaluate_limubert.py
echo ">>> LiMU-BERT done."

# 2. MOMENT evaluation
echo ""
echo ">>> [2/4] MOMENT evaluation..."
python val_scripts/human_activity_recognition/evaluate_moment.py
echo ">>> MOMENT done."

# 3. CrossHAR evaluation
echo ""
echo ">>> [3/4] CrossHAR evaluation..."
python val_scripts/human_activity_recognition/evaluate_crosshar.py
echo ">>> CrossHAR done."

# 4. LanHAR evaluation (longest - trains from scratch)
echo ""
echo ">>> [4/4] LanHAR evaluation (training + eval)..."
python val_scripts/human_activity_recognition/evaluate_lanhar.py
echo ">>> LanHAR done."

# Generate combined results table
echo ""
echo ">>> Generating combined results table..."
python scripts/generate_results_table.py

echo ""
echo "============================================"
echo "  All baseline evaluations complete!"
echo "  $(date)"
echo "============================================"
