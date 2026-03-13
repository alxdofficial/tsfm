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
#
# Lessons learned:
#   - /workspace is network MFS — painfully slow for small files (parquet sessions)
#   - /tmp overlay is better but still slow for 100K+ small files
#   - tmpfs (RAM) is fastest — use it for training data if enough RAM
#   - pip install requirements.txt can overwrite system torch with CUDA build — skip torch
#   - tar without --no-same-owner spams ownership warnings in containers

set -e

GDRIVE_FILE_ID="1a6QROP9qZZetOek_NxbgIWFNYDVY8d0H"
REPO_URL="https://github.com/alxdofficial/tsfm.git"
WORKDIR="/workspace/tsfm"
# Use RAM disk for training data (fastest for small files)
# Falls back to /tmp if not enough RAM
DATA_MOUNT="/dev/shm/tsfm_data"

echo "============================================"
echo "  TSFM RunPod Setup"
echo "  $(date)"
echo "============================================"

# -----------------------------------------------
# 1. Clone repo to /workspace (persistent across restarts)
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
# 2. Install deps — DO NOT touch torch (system has ROCm/CUDA build)
# -----------------------------------------------
echo ">>> Installing dependencies (preserving system torch)..."
pip install --upgrade pip -q

# Install everything EXCEPT torch/torchvision (already commented out in requirements.txt,
# but double-protect against accidental uncommenting)
pip install -r requirements.txt --no-deps -q 2>/dev/null || true
# Install actual deps individually (skip torch)
pip install -q \
    numpy scipy pandas pyarrow matplotlib plotly \
    scikit-learn umap-learn tqdm joblib \
    sentence-transformers transformers \
    pydantic requests gdown

echo ">>> Torch version: $(python3 -c 'import torch; print(f"{torch.__version__} (HIP={torch.version.hip}, CUDA={torch.version.cuda})")')"

# -----------------------------------------------
# 3. Pre-cache sentence-transformers model (avoid download during training)
# -----------------------------------------------
echo ">>> Pre-caching sentence-transformers model..."
python3 -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('all-MiniLM-L6-v2')
print('  all-MiniLM-L6-v2 cached')
" 2>/dev/null

# -----------------------------------------------
# 4. Download and extract training data to RAM disk
# -----------------------------------------------
# /dev/shm is a tmpfs (RAM-backed) — instant I/O for small parquet files.
# Training data is ~12GB extracted. Most RunPod pods have 128GB+ RAM.

if [ -d "$DATA_MOUNT" ] && [ -d "$DATA_MOUNT/uci_har/sessions" ]; then
    echo ">>> Training data already in RAM at $DATA_MOUNT"
else
    # Check available RAM
    AVAILABLE_SHM_MB=$(df -m /dev/shm | awk 'NR==2 {print $4}')
    REQUIRED_MB=15000  # ~12GB data + headroom

    if [ "$AVAILABLE_SHM_MB" -gt "$REQUIRED_MB" ] 2>/dev/null; then
        EXTRACT_DIR="$DATA_MOUNT"
        echo ">>> Using RAM disk (/dev/shm) — ${AVAILABLE_SHM_MB}MB available"
    else
        EXTRACT_DIR="/tmp/tsfm_data"
        echo ">>> RAM disk too small (${AVAILABLE_SHM_MB}MB), falling back to /tmp"
    fi

    mkdir -p "$EXTRACT_DIR"

    if [ ! -f "/tmp/tsfm_data.tar.gz" ]; then
        echo ">>> Downloading training data from Google Drive..."
        gdown "$GDRIVE_FILE_ID" -O /tmp/tsfm_data.tar.gz
    fi

    echo ">>> Extracting training data to $EXTRACT_DIR ..."
    tar xzf /tmp/tsfm_data.tar.gz --no-same-owner -C "$EXTRACT_DIR" --strip-components=1
    rm -f /tmp/tsfm_data.tar.gz
    echo ">>> Extraction complete."

    DATA_MOUNT="$EXTRACT_DIR"
fi

# Symlink data dir into repo
rm -rf "$WORKDIR/data" 2>/dev/null
ln -sf "$DATA_MOUNT" "$WORKDIR/data"

# Also symlink training output to /workspace (persistent) so checkpoints survive restarts
mkdir -p /workspace/training_output
rm -rf "$WORKDIR/training_output" 2>/dev/null
ln -sf /workspace/training_output "$WORKDIR/training_output"

# Verify data
NUM_DATASETS=$(ls -d "$WORKDIR"/data/*/sessions 2>/dev/null | wc -l)
echo ">>> Found $NUM_DATASETS dataset(s) with session data."

# -----------------------------------------------
# 5. GPU info
# -----------------------------------------------
echo ""
echo ">>> GPU info:"
if command -v rocm-smi &>/dev/null; then
    rocm-smi --showproductname 2>/dev/null | grep "Card series" || true
    python3 -c "import torch; print(f'  PyTorch {torch.__version__}, Device: {torch.cuda.get_device_name(0)}')"
else
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
echo ""

# -----------------------------------------------
# 6. Summary
# -----------------------------------------------
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "To start training:"
echo "  cd $WORKDIR"
echo "  PYTHONUNBUFFERED=1 TSFM_NO_COMPILE=1 TSFM_BATCH_SIZE=64 \\"
echo "    python training_scripts/human_activity_recognition/semantic_alignment_train.py"
echo ""
echo "Key env vars:"
echo "  TSFM_BATCH_SIZE=64       — micro-batch size (MI300X can handle 64+)"
echo "  TSFM_GRAD_CHECKPOINT=1   — gradient checkpointing (if VRAM tight)"
echo "  TSFM_NO_COMPILE=1        — skip torch.compile (ROCm compat)"
echo "  TSFM_LR=8e-5             — learning rate (default)"
echo "  TSFM_GRAD_CACHE=1        — GradCache for full-window negatives (default on)"
echo "  TSFM_MEMORY_BANK_SIZE=1024 — memory bank queue size (default)"
echo ""
echo "Data: $DATA_MOUNT ($(df -h "$DATA_MOUNT" | awk 'NR==2{print $1}'))"
echo "Checkpoints: /workspace/training_output/ (persistent)"
echo ""
