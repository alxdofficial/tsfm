#!/bin/bash
set -e
echo "============================================"
echo "Re-evaluating Opportunity (fixed data)"
echo "Start: $(date)"
echo "============================================"

echo ""
echo ">>> 1/2: LiMU-BERT (all datasets, fixes Opportunity) <<<"
python val_scripts/human_activity_recognition/evaluate_limubert.py 2>&1

echo ""
echo ">>> 2/2: TSFM (all datasets, fixes Opportunity) <<<"
python val_scripts/human_activity_recognition/evaluate_tsfm.py 2>&1

echo ""
echo "============================================"
echo "Opportunity re-evaluation complete!"
echo "End: $(date)"
echo "============================================"
