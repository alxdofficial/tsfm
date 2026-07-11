# TSFM: Language-Aligned IMU Foundation Model for Human Activity Recognition

A foundation model that aligns IMU sensor embeddings with natural language descriptions, enabling zero-shot activity recognition on unseen datasets. Trained end-to-end on 11 diverse HAR datasets and evaluated under protocol v2 on 6 held-out test datasets.

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
    Cosine Similarity with SentenceBERT text prototypes
         |
    Zero-shot activity prediction
```

### Key Design Choices

- **Channel-independent encoding**: Each sensor channel is processed independently through shared temporal attention, then fused via cross-channel attention. This handles 6-48 channels without retraining.
- **Text label prototypes**: Text embeddings come from SentenceBERT (all-MiniLM-L6-v2); frozen mean pooling is the current default, with the learnable label bank kept as an ablation/variant.
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

# Download and convert all 18 datasets to standardized session format
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
| Training datasets | 11 | See table below |
| Temperature | 0.07 | CLIP default |
| Memory bank | Off by default | Optional MoCo-style additional negatives |

**Data root**: By default, looks for `data/` in the project root. Override with:
```bash
export TSFM_DATA_ROOT=/path/to/your/data
```

**Training outputs** are saved to `training_output/semantic_alignment/{timestamp}/` with checkpoints every 5 epochs, loss plots, and embedding visualizations.

---

## Datasets

### Training (11 datasets, 94 HALO train labels)

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
| Capture24 | 3 | 100 Hz | 9 | Free-living wrist accelerometer |

### Zero-Shot Test (6 active datasets, never seen during training)

**Active v2 test datasets**:

| Dataset | Activities | Hz | Difficulty | Group Coverage |
|---------|:---:|:---:|-----------|:---:|
| MotionSense | 6 | 50 | Easy (basic locomotion) | 100% |
| RealWorld | 8 | 50 | Medium (multi-placement) | 100% |
| MobiAct | 13 | 50 | Hard (falls, vehicle entry) | 85% |
| Shoaib | 7 | 50 | Medium (multi-placement smartphone) | 100% |
| HARTH | 10 | 50 | Hard (back+thigh accelerometer, distribution shift) | 100% |
| InclusiveHAR | 6 | 50 | Medium (waist-pouch phone, ability diversity) | 100% |

Opportunity is retained as an appendix dataset; VTT-ConIoT is retired from the primary benchmark. Baseline models evaluate on standardized `(N, 120, 6)` windows at 20Hz. HALO evaluates on native-rate data, with a 20Hz neutral-text parity row — see [Evaluation Protocol v2](docs/baselines/EVALUATION_PROTOCOL_V2.md).

---

## Baseline Evaluation

Protocol v2 uses ZS-XD: zero-shot classification against each target dataset's own frozen label strings, with macro-F1 as the primary metric. Closed-vocabulary baselines are bridged with ConSE.

| Baseline | Type | Zero-Shot Method | Embedding Dim |
|----------|------|------------------|:---:|
| **HALO (ours)** | Text-aligned | Cosine similarity to target label strings | 384 |
| **LiMU-BERT** | Encoder-only | ConSE from cached GRU classifier softmax | 72 |
| **CrossHAR** | Encoder-only | ConSE from cached Transformer classifier softmax | 72 |
| **UniMTS** | Text-aligned | Planned adapter | released |
| **ssl-wearables** | Encoder-only | Planned ConSE adapter | released |

### Running Evaluations

```bash
# HALO native ZS-XD
TSFM_CHECKPOINT=training_output/semantic_alignment/small_deep_v2_4b3fdd6/best.pt \
python val_scripts/human_activity_recognition/evaluate_tsfm_v2.py --zs-only

# HALO 20Hz neutral-text parity row
python val_scripts/human_activity_recognition/evaluate_tsfm_v2.py \
  --zs-only --channel-text neutral --eval-rate 20

# Baselines
python val_scripts/human_activity_recognition/run_baselines_v2.py --baselines crosshar limubert

# Generate combined comparison table
python val_scripts/human_activity_recognition/assemble_v2_table.py
```

**TSFM checkpoint**: The evaluation script auto-discovers the latest checkpoint in
`training_output/semantic_alignment/`. Override with `TSFM_CHECKPOINT` env var.

Results are saved to `test_output/eval_v2/*.json`.

For baseline setup (cloning repos, checkpoints, data preparation), see
[docs/baselines/BASELINES_SETUP.md](docs/baselines/BASELINES_SETUP.md).

---

## Documentation

| Document | Description |
|----------|-------------|
| **[docs/README.md](docs/README.md)** | Documentation index + single-source-of-truth map |
| **[docs/baselines/RESULTS_V2.md](docs/baselines/RESULTS_V2.md)** | Current evaluation results and fairness analysis |
| **[docs/baselines/EVALUATION_PROTOCOL_V2.md](docs/baselines/EVALUATION_PROTOCOL_V2.md)** | Evaluation framework, fairness justifications, per-dataset label coverage |
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
│   ├── evaluate_tsfm_v2.py        # HALO protocol-v2 evaluation
│   ├── run_baselines_v2.py        # Generic protocol-v2 baseline driver
│   ├── baselines/                 # Baseline adapters
│   ├── grouped_zero_shot.py       # Shared zero-shot utilities
│   ├── model_loading.py           # TSFM model/label bank loading
│   ├── evaluation_metrics.py      # Group-aware accuracy, similarity
│   └── plot_utils.py              # Training visualization
│
├── datasets/imu_pretraining_dataset/
│   ├── multi_dataset_loader.py    # Multi-dataset PyTorch dataloader
│   ├── label_groups.py            # semantic label groups for sampling/legacy utilities
│   └── augmentations.py           # Physical augmentations
│
├── datascripts/                    # Dataset download + conversion (18 datasets)
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
