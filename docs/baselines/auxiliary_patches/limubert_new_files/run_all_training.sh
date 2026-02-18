#!/bin/bash
# LiMU-BERT multi-dataset training pipeline for baseline comparison
#
# Workflow:
# 1. Pretrain on combined training datasets (100 epochs)
# 2. Generate embeddings for each test dataset using pretrained model
# 3. Train classifiers on each test dataset (100 epochs)
#
# Output logged to training_log.txt
cd /home/alex/code/tsfm/auxiliary_repos/LIMU-BERT-Public

LOG="training_log.txt"
export PYTHONUNBUFFERED=1

VERSION="20_120"
MODEL_VERSION="v1"  # base_v1: 6-dim input
PRETRAIN_NAME="pretrained_combined"

TEST_DATASETS=(motionsense realworld mobiact vtt_coniot)

exec > >(tee -a "$LOG") 2>&1

echo "=============================================="
echo "LiMU-BERT Multi-Dataset Training Pipeline"
echo "Pretrain: combined_train (10 datasets, 111589 windows)"
echo "Test datasets: ${TEST_DATASETS[*]}"
echo "Epochs: 100 (pretrain + classifier)"
echo "Started: $(date)"
echo "=============================================="

# ===== Phase 1: Pretrain on combined training data =====
echo ""
echo "===== PHASE 1: PRETRAINING ON COMBINED DATA ====="
PRETRAIN_DIR="saved/pretrain_base_combined_train_${VERSION}"
PRETRAIN_CKPT="${PRETRAIN_DIR}/${PRETRAIN_NAME}.pt"

if [ -f "$PRETRAIN_CKPT" ]; then
    echo "Pretrained model already exists at $PRETRAIN_CKPT, skipping."
else
    echo "Start: $(date)"
    python -u pretrain.py $MODEL_VERSION combined_train $VERSION \
        -s $PRETRAIN_NAME \
        -t config/pretrain_100ep.json
    if [ $? -ne 0 ]; then
        echo "ERROR: Pretraining failed!"
        exit 1
    fi
    echo "Finished pretraining: $(date)"
fi
echo ""

# ===== Phase 2: Generate embeddings for test datasets =====
echo "===== PHASE 2: EMBEDDING GENERATION ====="

# The pretrained model is at saved/pretrain_base_combined_train_20_120/pretrained_combined.pt
# But embedding.py looks for it at saved/pretrain_base_{dataset}_20_120/pretrained_combined.pt
# So we symlink the pretrained model into each test dataset's pretrain directory.
for ds in "${TEST_DATASETS[@]}"; do
    DS_PRETRAIN_DIR="saved/pretrain_base_${ds}_${VERSION}"
    mkdir -p "$DS_PRETRAIN_DIR"
    if [ ! -f "$DS_PRETRAIN_DIR/${PRETRAIN_NAME}.pt" ]; then
        ln -sf "$(pwd)/${PRETRAIN_CKPT}" "$DS_PRETRAIN_DIR/${PRETRAIN_NAME}.pt"
        echo "Symlinked pretrained model to $DS_PRETRAIN_DIR/"
    fi
done

for ds in "${TEST_DATASETS[@]}"; do
    EMBED="embed/embed_${PRETRAIN_NAME}_${ds}_${VERSION}.npy"
    if [ -f "$EMBED" ]; then
        echo "--- Skipping embedding $ds (file exists) ---"
        continue
    fi
    echo ""
    echo "--- Generating embeddings for $ds ---"
    python -u embedding.py $MODEL_VERSION $ds $VERSION \
        -f $PRETRAIN_NAME \
        -t config/pretrain_100ep.json
    if [ $? -ne 0 ]; then
        echo "ERROR: Embedding failed for $ds"
        continue
    fi
    echo "Finished embeddings $ds: $(date)"
done
echo ""

# ===== Phase 3: Train classifiers on test datasets =====
echo "===== PHASE 3: CLASSIFIER TRAINING ====="
for ds in "${TEST_DATASETS[@]}"; do
    SAVE_DIR="saved/classifier_base_gru_${ds}_${VERSION}"
    CKPT="${SAVE_DIR}/classifier_${ds}.pt"
    if [ -f "$CKPT" ]; then
        echo "--- Skipping classifier $ds (checkpoint exists) ---"
        continue
    fi
    echo ""
    echo "--- Training classifier on $ds (activity, label_index=0) ---"
    echo "Start: $(date)"
    python -u classifier.py v2 $ds $VERSION \
        -f $PRETRAIN_NAME \
        -l 0 \
        -s classifier_${ds} \
        -t config/train_100ep.json
    if [ $? -ne 0 ]; then
        echo "ERROR: Classifier failed for $ds"
        continue
    fi
    echo "Finished classifier $ds: $(date)"
    echo ""
done

echo ""
echo "=============================================="
echo "All training complete: $(date)"
echo "=============================================="
