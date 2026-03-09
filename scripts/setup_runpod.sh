#!/bin/bash
# One-shot RunPod setup: clone repo, download data, install deps, launch training.
#
# Usage (on RunPod):
#   curl -sL https://raw.githubusercontent.com/alxdofficial/tsfm/master/scripts/setup_runpod.sh | bash
#
# Or if you've already cloned:
#   bash scripts/setup_runpod.sh
#
# Prerequisites: RunPod PyTorch template (has torch pre-installed)

set -e

GDRIVE_FILE_ID="1a6QROP9qZZetOek_NxbgIWFNYDVY8d0H"
REPO_URL="https://github.com/alxdofficial/tsfm.git"
WORKDIR="/workspace/tsfm"

echo "============================================"
echo "  TSFM RunPod Setup"
echo "  $(date)"
echo "============================================"

# -----------------------------------------------
# 1. Clone repo (skip if already present)
# -----------------------------------------------
if [ -d "$WORKDIR/.git" ]; then
    echo ">>> Repo already cloned at $WORKDIR, pulling latest..."
    cd "$WORKDIR"
    git pull origin master
else
    echo ">>> Cloning repo..."
    git clone "$REPO_URL" "$WORKDIR"
    cd "$WORKDIR"
fi

# -----------------------------------------------
# 2. Create venv and install deps
# -----------------------------------------------
if [ ! -d "$WORKDIR/.venv" ]; then
    echo ">>> Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

echo ">>> Installing dependencies..."
pip install --upgrade pip -q
# Install torch first (RunPod usually has it, but ensure CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 -q 2>/dev/null || true
pip install -r requirements.txt -q
# gdown for Google Drive downloads
pip install gdown -q

# -----------------------------------------------
# 3. Download training data from Google Drive
# -----------------------------------------------
DATA_TAR="$WORKDIR/tsfm_data_processed.tar.gz"

if [ -d "$WORKDIR/data/uci_har/sessions" ]; then
    echo ">>> Training data already extracted, skipping download."
else
    if [ ! -f "$DATA_TAR" ]; then
        echo ">>> Downloading training data from Google Drive..."
        gdown "$GDRIVE_FILE_ID" -O "$DATA_TAR"
    fi
    echo ">>> Extracting training data..."
    tar xzf "$DATA_TAR" -C "$WORKDIR"
    echo ">>> Data extracted. Removing archive to save space..."
    rm -f "$DATA_TAR"
fi

# Verify data
NUM_DATASETS=$(ls -d "$WORKDIR"/data/*/sessions 2>/dev/null | wc -l)
echo ">>> Found $NUM_DATASETS dataset(s) with session data."

# -----------------------------------------------
# 4. GPU info
# -----------------------------------------------
echo ""
echo ">>> GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# -----------------------------------------------
# 5. Summary
# -----------------------------------------------
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "To start training:"
echo "  cd $WORKDIR"
echo "  source .venv/bin/activate"
echo "  TSFM_GRAD_CHECKPOINT=1 python training_scripts/human_activity_recognition/semantic_alignment_train.py"
echo ""
echo "Training config is in the script header. Key env vars:"
echo "  TSFM_GRAD_CHECKPOINT=1  — enable gradient checkpointing (saves ~60% activation VRAM)"
echo "  TSFM_NO_COMPILE=1       — skip torch.compile (if PyTorch version issues)"
echo ""
