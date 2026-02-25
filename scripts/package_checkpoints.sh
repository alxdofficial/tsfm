#!/bin/bash
# Package all model checkpoints and pretrained weights into a single zip for sharing.
#
# Usage: bash scripts/package_checkpoints.sh [output_name]
#   output_name: Optional zip filename (default: tsfm_checkpoints.zip)
#
# This creates a zip containing:
#   1. TSFM trained checkpoint (best.pt + hyperparams)
#   2. LiMU-BERT pretrained combined checkpoint
#   3. CrossHAR pretrained combined checkpoint
#   4. MOMENT zero-shot SVM cache (optional, saves ~30min refit)
#
# MOMENT and LLaSA model weights are NOT included (auto-downloaded from HuggingFace).
# LanHAR trains from scratch — no checkpoint needed.

set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

OUTPUT_NAME="${1:-tsfm_checkpoints.zip}"
STAGING_DIR=$(mktemp -d)

echo "============================================"
echo "  Packaging Checkpoints"
echo "  $(date)"
echo "============================================"

# -----------------------------------------------
# 1. TSFM (our model)
# -----------------------------------------------
TSFM_DIR="training_output/semantic_alignment"
TSFM_RUN=$(ls -t "$TSFM_DIR" 2>/dev/null | head -1)

if [ -n "$TSFM_RUN" ] && [ -f "$TSFM_DIR/$TSFM_RUN/best.pt" ]; then
    echo ""
    echo ">>> TSFM checkpoint: $TSFM_DIR/$TSFM_RUN/best.pt"
    mkdir -p "$STAGING_DIR/tsfm/$TSFM_RUN"
    cp "$TSFM_DIR/$TSFM_RUN/best.pt" "$STAGING_DIR/tsfm/$TSFM_RUN/"
    # Also copy hyperparams if present
    if [ -f "$TSFM_DIR/$TSFM_RUN/hyperparams.json" ]; then
        cp "$TSFM_DIR/$TSFM_RUN/hyperparams.json" "$STAGING_DIR/tsfm/$TSFM_RUN/"
    fi
    echo "    Copied."
else
    echo ""
    echo ">>> TSFM: No checkpoint found in $TSFM_DIR (skipped)"
fi

# -----------------------------------------------
# 2. LiMU-BERT combined pretrained
# -----------------------------------------------
LIMUBERT_CKPT="auxiliary_repos/LIMU-BERT-Public/saved/pretrain_base_combined_train_20_120/pretrained_combined.pt"

if [ -f "$LIMUBERT_CKPT" ]; then
    echo ""
    echo ">>> LiMU-BERT checkpoint: $LIMUBERT_CKPT"
    mkdir -p "$STAGING_DIR/limubert/pretrain_base_combined_train_20_120"
    cp "$LIMUBERT_CKPT" "$STAGING_DIR/limubert/pretrain_base_combined_train_20_120/"
    echo "    Copied."
else
    echo ""
    echo ">>> LiMU-BERT: No checkpoint at $LIMUBERT_CKPT (skipped)"
fi

# -----------------------------------------------
# 3. CrossHAR combined pretrained
# -----------------------------------------------
CROSSHAR_CKPT="auxiliary_repos/CrossHAR/saved/pretrain_base_combined_train_20_120/model_masked_6_1.pt"

if [ -f "$CROSSHAR_CKPT" ]; then
    echo ""
    echo ">>> CrossHAR checkpoint: $CROSSHAR_CKPT"
    mkdir -p "$STAGING_DIR/crosshar/pretrain_base_combined_train_20_120"
    cp "$CROSSHAR_CKPT" "$STAGING_DIR/crosshar/pretrain_base_combined_train_20_120/"
    echo "    Copied."
else
    echo ""
    echo ">>> CrossHAR: No checkpoint at $CROSSHAR_CKPT (skipped)"
fi

# -----------------------------------------------
# 4. MOMENT zero-shot SVM cache (optional)
# -----------------------------------------------
MOMENT_SVM="test_output/baseline_evaluation/moment_zs_svm.pkl"

if [ -f "$MOMENT_SVM" ]; then
    echo ""
    echo ">>> MOMENT SVM cache: $MOMENT_SVM"
    mkdir -p "$STAGING_DIR/moment"
    cp "$MOMENT_SVM" "$STAGING_DIR/moment/"
    echo "    Copied (optional — saves ~30min SVM refit)."
else
    echo ""
    echo ">>> MOMENT SVM: No cache at $MOMENT_SVM (skipped — will be regenerated)"
fi

# -----------------------------------------------
# 5. Write manifest
# -----------------------------------------------
cat > "$STAGING_DIR/CHECKPOINT_MANIFEST.md" << 'MANIFEST_EOF'
# Checkpoint Manifest

This zip contains pretrained model checkpoints for reproducing TSFM evaluation results.

## Contents

### tsfm/{run_id}/best.pt
TSFM trained checkpoint (our model). Contains the full model state including:
- IMU encoder weights
- Semantic alignment head weights
- LearnableLabelBank (text embeddings)
- Training hyperparameters path

**Restore location**: `training_output/semantic_alignment/{run_id}/best.pt`

### limubert/pretrain_base_combined_train_20_120/pretrained_combined.pt
LiMU-BERT encoder pretrained on all 10 training datasets combined.
This is a BERT-style masked reconstruction checkpoint, not a classifier.

**Restore location**: `auxiliary_repos/LIMU-BERT-Public/saved/pretrain_base_combined_train_20_120/pretrained_combined.pt`

### crosshar/pretrain_base_combined_train_20_120/model_masked_6_1.pt
CrossHAR encoder pretrained on all 10 training datasets combined.
Hierarchical self-supervised pretraining (masked + contrastive).

**Restore location**: `auxiliary_repos/CrossHAR/saved/pretrain_base_combined_train_20_120/model_masked_6_1.pt`

### moment/moment_zs_svm.pkl (optional)
Cached SVM-RBF classifier for MOMENT zero-shot evaluation. Fitted on embeddings
from the 10 training datasets. Saves ~30min of SVM GridSearchCV fitting.

**Restore location**: `test_output/baseline_evaluation/moment_zs_svm.pkl`

## Models NOT Included

- **MOMENT weights**: Auto-downloaded from HuggingFace (`AutonLab/MOMENT-1-large`)
- **LanHAR**: Trains from scratch during evaluation (downloads SciBERT automatically)
- **LLaSA**: Auto-downloaded from HuggingFace (`BASH-Lab/LLaSA-7B`)

## Restore Script

```bash
# From project root:
unzip tsfm_checkpoints.zip -d checkpoints_tmp

# TSFM
cp -r checkpoints_tmp/tsfm/* training_output/semantic_alignment/

# LiMU-BERT
mkdir -p auxiliary_repos/LIMU-BERT-Public/saved
cp -r checkpoints_tmp/limubert/* auxiliary_repos/LIMU-BERT-Public/saved/

# CrossHAR
mkdir -p auxiliary_repos/CrossHAR/saved
cp -r checkpoints_tmp/crosshar/* auxiliary_repos/CrossHAR/saved/

# MOMENT SVM (optional)
mkdir -p test_output/baseline_evaluation
cp checkpoints_tmp/moment/moment_zs_svm.pkl test_output/baseline_evaluation/

rm -rf checkpoints_tmp
```
MANIFEST_EOF

echo ""

# -----------------------------------------------
# 6. Create zip
# -----------------------------------------------
echo ">>> Creating $OUTPUT_NAME..."
cd "$STAGING_DIR"
zip -r "$PROJECT_ROOT/$OUTPUT_NAME" . -x '*.DS_Store'
cd "$PROJECT_ROOT"

# Clean up
rm -rf "$STAGING_DIR"

# Print summary
echo ""
echo "============================================"
echo "  Checkpoint package created: $OUTPUT_NAME"
echo "  $(date)"
echo "============================================"
echo ""
echo "Contents:"
unzip -l "$OUTPUT_NAME" | tail -n +4 | head -n -2
echo ""
echo "Size: $(du -h "$OUTPUT_NAME" | cut -f1)"
echo ""
echo "Share this zip with the researcher along with the repo."
echo "See CHECKPOINT_MANIFEST.md inside the zip for restore instructions."
