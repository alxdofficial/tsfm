#!/bin/bash
# Run all evaluations sequentially and generate results table
# Usage: bash scripts/run_all_evaluations.sh

set -e
cd "$(dirname "$0")/.."

echo "============================================"
echo "  Running All Evaluations"
echo "  $(date)"
echo "============================================"

# 1. TSFM evaluation
echo ""
echo ">>> [1/6] TSFM evaluation..."
python val_scripts/human_activity_recognition/evaluate_tsfm.py
echo ">>> TSFM done."

# 2. LiMU-BERT evaluation
echo ""
echo ">>> [2/6] LiMU-BERT evaluation..."
python val_scripts/human_activity_recognition/evaluate_limubert.py
echo ">>> LiMU-BERT done."

# 3. MOMENT evaluation
echo ""
echo ">>> [3/6] MOMENT evaluation..."
python val_scripts/human_activity_recognition/evaluate_moment.py
echo ">>> MOMENT done."

# 4. CrossHAR evaluation
echo ""
echo ">>> [4/6] CrossHAR evaluation..."
python val_scripts/human_activity_recognition/evaluate_crosshar.py
echo ">>> CrossHAR done."

# 5. LanHAR evaluation (longest - trains from scratch)
echo ""
echo ">>> [5/6] LanHAR evaluation (training + eval)..."
python val_scripts/human_activity_recognition/evaluate_lanhar.py
echo ">>> LanHAR done."

# 6. LLaSA evaluation (optional - requires ~16GB VRAM for 7B model)
echo ""
echo ">>> [6/6] LLaSA evaluation..."
python val_scripts/human_activity_recognition/evaluate_llasa.py
echo ">>> LLaSA done."

# Generate combined results table
echo ""
echo ">>> Generating combined results table..."
python scripts/generate_results_table.py

echo ""
echo "============================================"
echo "  All evaluations complete!"
echo "  $(date)"
echo "============================================"
