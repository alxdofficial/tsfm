#!/bin/bash
set -e
cd /home/alex/code/tsfm

echo "============================================"
echo "Running all baseline evaluations"
echo "Start: $(date)"
echo "============================================"

echo ""
echo ">>> 1/5: TSFM (ours) <<<"
python val_scripts/human_activity_recognition/evaluate_tsfm.py 2>&1
echo "TSFM done: $(date)"

echo ""
echo ">>> 2/5: LiMU-BERT <<<"
python val_scripts/human_activity_recognition/evaluate_limubert.py 2>&1
echo "LiMU-BERT done: $(date)"

echo ""
echo ">>> 3/5: MOMENT <<<"
python val_scripts/human_activity_recognition/evaluate_moment.py 2>&1
echo "MOMENT done: $(date)"

echo ""
echo ">>> 4/5: CrossHAR <<<"
python val_scripts/human_activity_recognition/evaluate_crosshar.py 2>&1
echo "CrossHAR done: $(date)"

echo ""
echo ">>> 5/5: LanHAR <<<"
python val_scripts/human_activity_recognition/evaluate_lanhar.py 2>&1
echo "LanHAR done: $(date)"

echo ""
echo "============================================"
echo "All evaluations complete: $(date)"
echo "============================================"
