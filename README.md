# TSFM: Language-Aligned IMU Foundation Model for Human Activity Recognition

A foundation model that aligns IMU sensor embeddings with natural language descriptions, enabling zero-shot activity recognition on unseen datasets. Trained end-to-end on 10 diverse HAR datasets and evaluated against 5 baselines on 7 held-out test datasets.

---

## Overview

TSFM uses a **CLIP-style** training approach: a dual-branch transformer encoder processes variable-length, variable-channel IMU data into fixed-dimension embeddings, which are aligned with text activity descriptions via contrastive learning. At inference time, activity recognition is performed via cosine similarity between sensor embeddings and text label embeddings — no classifier training needed.

### Architecture

```
Raw IMU Data (variable length, 6-48 channels)
         |
    Patch Tokenization (variable-size patches, interpolated to 64 timesteps)
         |
    Per-Channel Encoding (each channel processed independently)
         |
    Dual-Branch Transformer Encoder (4 layers)
    [Temporal Self-Attention] + [Cross-Channel Self-Attention]
         |
    Semantic Alignment Head
    [Channel Fusion (cross-attention)] -> [Temporal Pooling (cross-attention)]
         |
    384-dim L2-normalized embedding
         |
    Cosine Similarity with LearnableLabelBank text embeddings
         |
    Zero-shot activity prediction
```

### Key Design Choices

- **Channel-independent encoding**: Each sensor channel is processed independently through shared temporal attention, then fused via cross-channel attention. This handles 6-48 channels without retraining.
- **Learnable label bank**: Text embeddings are initialized from SentenceBERT (all-MiniLM-L6-v2) then refined via learnable attention pooling during training.
- **Soft targets**: Contrastive loss uses pairwise text similarity to weight targets, preventing synonym labels (e.g., "walking" and "strolling") from being treated as negatives.
- **Group-balanced sampling**: Training samples are weighted by inverse semantic group frequency with capped oversampling (max 20x) to handle class imbalance across datasets.

---

## Setup

```bash
# Clone the repository
git clone https://github.com/alxdofficial/tsfm.git
cd tsfm

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies (requires CUDA-compatible PyTorch — adjust for your GPU)
pip install -r requirements.txt

# Download and convert all 20 datasets to standardized session format
python datascripts/setup_all_ts_datasets.py

# Generate benchmark evaluation data
python benchmark_data/scripts/export_raw.py
python benchmark_data/scripts/preprocess_limubert.py
python benchmark_data/scripts/preprocess_tsfm_eval.py
```

**Note**: Some datasets require manual download — see `datascripts/setup_all_ts_datasets.py`
for URLs and instructions per dataset. The script will skip datasets whose raw data is not
yet downloaded and tell you where to get them.

---

## Training

Training runs end-to-end, starting from the self-supervised pretrained encoder checkpoint when available:

```bash
python training_scripts/human_activity_recognition/semantic_alignment_train.py
```

**Configuration**: All hyperparameters are constants at the top of the training script. Key settings:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Effective batch size | 512 | 32 micro-batch x 16 accumulation steps |
| Learning rate | 1e-4 | With 3-epoch warmup + cosine decay |
| Epochs | 100 | ~2 min/epoch on RTX 4090 |
| Encoder | 384-dim, 8 heads, 4 layers | ~9.5M parameters |
| Training datasets | 10 | See table below |
| Temperature | 0.07 | CLIP default |
| Memory bank | 256 queue size | MoCo-style additional negatives |

**Data root**: By default, looks for `data/` in the project root. Override with:
```bash
export TSFM_DATA_ROOT=/path/to/your/data
```

**Training outputs** are saved to `training_output/semantic_alignment/{timestamp}/` with checkpoints every 5 epochs, loss plots, and embedding visualizations.

---

## Datasets

### Training (10 datasets, 87 global activity labels)

| Dataset | Channels | Rate | Activities | Description |
|---------|:---:|:---:|:---:|-------------|
| UCI HAR | 9 | 50 Hz | 6 | Smartphone IMU |
| HHAR | 6 | 50 Hz | 6 | Heterogeneous devices |
| MHEALTH | 21 | 50 Hz | 12 | Multi-sensor body |
| PAMAP2 | 48 | 9 Hz | 12 | Physical activity monitoring |
| WISDM | 12 | 20 Hz | 18 | Phone + watch |
| UniMiB SHAR | 3 | 50 Hz | 17 | ADL + falls |
| DSADS | 9 | 25 Hz | 19 | Daily + sports |
| HAPT | 6 | 50 Hz | 12 | Postural transitions |
| KU-HAR | 6 | 100 Hz | 17 | 89 subjects |
| RecGym | 6 | 20 Hz | 11 | Gym exercises |

### Zero-Shot Test (7 datasets, never seen during training)

**Main test datasets** (85-100% label coverage — used for primary results):

| Dataset | Activities | Hz | Difficulty | Group Coverage |
|---------|:---:|:---:|-----------|:---:|
| MotionSense | 6 | 50 | Easy (basic locomotion) | 100% |
| RealWorld | 8 | 50 | Medium (multi-placement) | 100% |
| MobiAct | 13 | 50 | Hard (falls, vehicle entry) | 85% |
| Shoaib | 7 | 50 | Medium (multi-placement smartphone) | 100% |
| Opportunity | 4 | 30 | Medium (raw XSens body IMU) | 100% |
| HARTH | 12 | 50 | Hard (back-only accelerometer, distribution shift) | 100% |

**Severe out-of-domain** (reported separately — 50% of activities have no training equivalent):

| Dataset | Activities | Hz | Difficulty | Group Coverage |
|---------|:---:|:---:|-----------|:---:|
| VTT-ConIoT | 16 | 50 | Severe (industrial/construction) | 50% |

Baseline models evaluate on standardized `(N, 120, 6)` windows at 20Hz. TSFM evaluates on native-rate data — see [Evaluation Protocol](docs/baselines/EVALUATION_PROTOCOL.md) for sampling rate policy.

---

## Baseline Evaluation

We compare TSFM against 5 baselines using a unified 4-metric evaluation framework:

| Baseline | Type | Zero-Shot Method | Embedding Dim |
|----------|------|------------------|:---:|
| **TSFM (ours)** | Text-aligned | Cosine similarity | 384 |
| **LanHAR** | Text-aligned | Cosine similarity (SciBERT) | 768 |
| **LiMU-BERT** | Encoder-only | GRU classifier | 72 |
| **MOMENT** | General time-series | SVM-RBF classifier | 6144 |
| **CrossHAR** | Encoder-only | Transformer classifier | 72 |
| **LLaSA** | LLM-based | Generative text parsing | 7B params |

### 4-Metric Evaluation

1. **Zero-Shot Open-Set**: Classify against all 87 training labels
2. **Zero-Shot Closed-Set**: Classify against test dataset labels only
3. **1% Supervised**: End-to-end fine-tuning on 1% labeled test data
4. **10% Supervised**: End-to-end fine-tuning on 10% labeled test data

### Running Evaluations

```bash
# All baselines sequentially
bash scripts/run_all_evaluations.sh

# Or individually
python val_scripts/human_activity_recognition/evaluate_tsfm.py
python val_scripts/human_activity_recognition/evaluate_limubert.py
python val_scripts/human_activity_recognition/evaluate_moment.py
python val_scripts/human_activity_recognition/evaluate_crosshar.py
python val_scripts/human_activity_recognition/evaluate_lanhar.py
python val_scripts/human_activity_recognition/evaluate_llasa.py   # optional, ~16GB VRAM

# Generate combined comparison table
python scripts/generate_results_table.py
```

**TSFM checkpoint**: The evaluation script auto-discovers the latest checkpoint in
`training_output/semantic_alignment/`. Override with `TSFM_CHECKPOINT` env var.

Results are saved to `test_output/baseline_evaluation/{model}_evaluation.json`.

For baseline setup (cloning repos, checkpoints, data preparation), see
[docs/baselines/BASELINES_SETUP.md](docs/baselines/BASELINES_SETUP.md).

---

## Documentation

| Document | Description |
|----------|-------------|
| **[docs/README.md](docs/README.md)** | Documentation index + single-source-of-truth map |
| **[docs/baselines/RESULTS.md](docs/baselines/RESULTS.md)** | Current evaluation results and fairness analysis |
| **[docs/baselines/EVALUATION_PROTOCOL.md](docs/baselines/EVALUATION_PROTOCOL.md)** | Evaluation framework, fairness justifications, per-dataset label coverage |
| **[docs/baselines/BASELINE_IMPLEMENTATION_NOTES.md](docs/baselines/BASELINE_IMPLEMENTATION_NOTES.md)** | Per-baseline implementation details and design decisions |
| **[docs/baselines/BASELINES_SETUP.md](docs/baselines/BASELINES_SETUP.md)** | How to set up and reproduce baseline evaluations |
| **[model/README.md](model/README.md)** | Model architecture API |
| **[training_scripts/human_activity_recognition/README.md](training_scripts/human_activity_recognition/README.md)** | Training pipeline details |
| **[benchmark_data/README.md](benchmark_data/README.md)** | Benchmark data format and preprocessing |
| **[datascripts/README.md](datascripts/README.md)** | Dataset download and conversion pipeline |
| **[DATA_FORMAT.md](DATA_FORMAT.md)** | Standardized session parquet format spec |

---

## Repository Structure

```
tsfm/
├── model/                          # Model implementations
│   ├── encoder.py                  # Dual-branch transformer encoder
│   ├── semantic_alignment.py       # Semantic alignment head
│   ├── token_text_encoder.py       # LearnableLabelBank, text encoding
│   ├── preprocessing.py            # Patch tokenization, interpolation
│   └── positional_encoding.py      # Sinusoidal + semantic position embeddings
│
├── training_scripts/human_activity_recognition/
│   ├── semantic_alignment_train.py # Main training script (end-to-end)
│   ├── semantic_loss.py            # CLIP-style contrastive loss
│   └── memory_bank.py             # MoCo-style embedding queue
│
├── val_scripts/human_activity_recognition/
│   ├── evaluate_tsfm.py           # TSFM evaluation
│   ├── evaluate_limubert.py       # LiMU-BERT baseline
│   ├── evaluate_moment.py         # MOMENT baseline
│   ├── evaluate_crosshar.py       # CrossHAR baseline
│   ├── evaluate_lanhar.py         # LanHAR baseline
│   ├── evaluate_llasa.py          # LLaSA baseline
│   ├── grouped_zero_shot.py       # Shared zero-shot utilities
│   ├── model_loading.py           # TSFM model/label bank loading
│   ├── evaluation_metrics.py      # Group-aware accuracy, similarity
│   └── plot_utils.py              # Training visualization
│
├── datasets/imu_pretraining_dataset/
│   ├── multi_dataset_loader.py    # Multi-dataset PyTorch dataloader
│   ├── label_groups.py            # 87 labels -> 34 semantic groups
│   └── augmentations.py           # Physical augmentations
│
├── datascripts/                    # Dataset download + conversion (20 datasets)
├── benchmark_data/                 # Standardized evaluation data + preprocessing scripts
├── docs/baselines/                 # Evaluation protocol, results, fairness analysis
├── scripts/                        # Utility scripts (runner, results table)
├── data/                           # Raw + processed training data (gitignored)
└── training_output/                # Checkpoints, plots, logs (gitignored)
```

---

## Testing

```bash
pytest tests/ -v
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TSFM_DATA_ROOT` | `{project_root}/data` | Path to dataset directory |
| `TSFM_CHECKPOINT` | Auto-discovers latest in `training_output/` | Path to trained TSFM checkpoint |
