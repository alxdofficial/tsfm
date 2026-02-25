#!/bin/bash
# Package the full working directory for sharing with another researcher.
#
# Creates a self-contained zip that includes:
#   - All source code and documentation
#   - Preprocessed benchmark evaluation data (20Hz + native-rate .npy files)
#   - TSFM trained checkpoint (best.pt only, not intermediate epochs)
#   - Baseline repos with pretrained checkpoints (LiMU-BERT, CrossHAR, LanHAR, LLaSA)
#   - Cached evaluation results and MOMENT SVM
#
# Excludes (can be regenerated):
#   - data/ (36GB raw dataset downloads + session data)
#   - benchmark_data/raw/ (8GB intermediate CSVs)
#   - training_output epoch checkpoints (13GB, only best.pt kept)
#   - auxiliary_repos embed/ and dataset/ dirs (7.7GB, regenerated during eval)
#   - .git/ history
#   - .venv/ virtual environment
#
# Usage: bash scripts/package_workdir.sh [output_name]
#   output_name: Optional zip filename (default: tsfm_workdir.zip)
#
# Expected output size: ~5-6GB

set -e
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

OUTPUT_NAME="${1:-tsfm_workdir.zip}"

echo "============================================"
echo "  Packaging Working Directory"
echo "  $(date)"
echo "============================================"
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Output: $OUTPUT_NAME"
echo ""

# Build exclude list
EXCLUDES=(
    # Version control and environment
    ".git/*"
    ".venv/*"
    "__pycache__/*"
    "*.pyc"
    ".serena/*"

    # Raw data (36GB — researcher re-downloads or we provide separately)
    "data/*"

    # Intermediate benchmark CSVs (8GB — regenerated from data/)
    "benchmark_data/raw/*"

    # TSFM epoch checkpoints (13GB — only best.pt is needed)
    "training_output/semantic_alignment/*/epoch_*.pt"
    # Stage-1 pretraining output (not needed for eval or stage-2 training)
    "training_output/imu_pretraining/*"

    # LiMU-BERT pre-extracted embeddings (7.2GB — regenerated during eval)
    "auxiliary_repos/LIMU-BERT-Public/embed/*"
    # LiMU-BERT original dataset copies (525MB — we use benchmark_data/ instead)
    "auxiliary_repos/LIMU-BERT-Public/dataset/*"
    # Reference papers
    "auxiliary_repos/papers/*"
    "auxiliary_repos/papers_txt/*"

    # Previously generated checkpoint zip
    "tsfm_checkpoints.zip"

    # Debug images
    "data/*/debug_*.png"

    # OS files
    ".DS_Store"
    "*.swp"
    "*.swo"
)

# Build zip exclude arguments
EXCLUDE_ARGS=()
for pattern in "${EXCLUDES[@]}"; do
    EXCLUDE_ARGS+=(-x "$pattern")
done

echo "Creating $OUTPUT_NAME (this may take a few minutes)..."
echo ""

zip -r "$OUTPUT_NAME" . \
    "${EXCLUDE_ARGS[@]}" \
    -x ".claude/*" \
    2>&1 | tail -1

echo ""
echo "============================================"
echo "  Package complete: $OUTPUT_NAME"
echo "  $(date)"
echo "============================================"
echo ""
echo "Size: $(du -h "$OUTPUT_NAME" | cut -f1)"
echo ""

# Print what's included
echo "=== Contents Summary ==="
echo ""
echo "Source code & docs:"
unzip -l "$OUTPUT_NAME" | grep -c '\.py$' | xargs -I{} echo "  {} Python files"
unzip -l "$OUTPUT_NAME" | grep -c '\.md$' | xargs -I{} echo "  {} Markdown docs"
unzip -l "$OUTPUT_NAME" | grep -c '\.sh$' | xargs -I{} echo "  {} Shell scripts"
echo ""
echo "Checkpoints:"
unzip -l "$OUTPUT_NAME" | grep '\.pt$' || echo "  (none)"
echo ""
echo "Evaluation data:"
unzip -l "$OUTPUT_NAME" | grep '\.npy$' | wc -l | xargs -I{} echo "  {} .npy data files"
echo ""

cat << 'USAGE_EOF'
=== For the receiving researcher ===

1. Unzip:
   unzip tsfm_workdir.zip -d tsfm && cd tsfm

2. Create environment:
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt

3. Run evaluations (no data download needed — preprocessed .npy files included):
   bash scripts/run_all_evaluations.sh

4. To retrain TSFM from scratch (requires raw data):
   python datascripts/setup_all_ts_datasets.py   # download + convert datasets
   python training_scripts/human_activity_recognition/semantic_alignment_train.py

See README.md for full documentation.
USAGE_EOF
