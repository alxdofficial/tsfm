#!/bin/bash
set -e
cd /home/alex/code/tsfm

echo "============================================"
echo "Running remaining baseline evaluations"
echo "Start: $(date)"
echo "============================================"

echo ""
echo ">>> 1/4: LiMU-BERT <<<"
python val_scripts/human_activity_recognition/evaluate_limubert.py 2>&1
echo "LiMU-BERT done: $(date)"

echo ""
echo ">>> 2/4: MOMENT <<<"
python val_scripts/human_activity_recognition/evaluate_moment.py 2>&1
echo "MOMENT done: $(date)"

echo ""
echo ">>> 3/4: CrossHAR <<<"
python val_scripts/human_activity_recognition/evaluate_crosshar.py 2>&1
echo "CrossHAR done: $(date)"

echo ""
echo ">>> 4/4: LanHAR <<<"
python val_scripts/human_activity_recognition/evaluate_lanhar.py 2>&1
echo "LanHAR done: $(date)"

echo ""
echo "============================================"
echo "All evaluations complete: $(date)"
echo "============================================"
