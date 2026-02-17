# TSFM: Language-Aligned IMU Foundation Model for Human Activity Recognition

A foundation model that aligns IMU sensor embeddings with natural language descriptions, enabling zero-shot activity recognition on unseen datasets. Trained end-to-end on 10 diverse HAR datasets and evaluated against 4 baselines on 4 held-out test datasets.

---

## Overview

TSFM uses a **CLIP-style** training approach: a dual-branch transformer encoder processes variable-length, variable-channel IMU data into fixed-dimension embeddings, which are aligned with text activity descriptions via contrastive learning. At inference time, activity recognition is performed via cosine similarity between sensor embeddings and text label embeddings — no classifier training needed.

### Architecture

```
Raw IMU Data (variable length, 6-48 channels)
         |
    Patch Tokenization (variable-size patches, interpolated to 64 timesteps)
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

## Training

Training runs end-to-end from randomly initialized encoder weights:

```bash
python training_scripts/human_activity_recognition/semantic_alignment_train.py
```

**Configuration**: All hyperparameters are constants at the top of the training script. Key settings:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Effective batch size | 512 | 32 micro-batch x 16 accumulation steps |
| Learning rate | 1e-4 | With 3-epoch warmup + cosine decay |
| Epochs | 100 | ~2 min/epoch on RTX 4090 |
| Encoder | 384-dim, 8 heads, 4 layers | 21M parameters |
| Training datasets | 10 | See table below |
| Temperature | 0.07 | CLIP default |
| Memory bank | 256 queue size | MoCo-style additional negatives |

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

### Zero-Shot Test (4 datasets, never seen during training)

| Dataset | Activities | Difficulty | Group Coverage |
|---------|:---:|-----------|:---:|
| MotionSense | 6 | Easy (basic locomotion) | 100% |
| RealWorld | 8 | Medium (multi-placement) | 100% |
| MobiAct | 13 | Hard (falls, vehicle entry) | 85% |
| VTT-ConIoT | 16 | Hard (industrial/construction) | 50% |

All data is standardized to `(N, 120, 6)` windows at 20Hz with 6 IMU channels (acc_xyz + gyro_xyz) for evaluation.

---

## Baseline Evaluation

We compare TSFM against 4 baselines using a unified 4-metric evaluation framework:

| Baseline | Type | Zero-Shot Method | Embedding Dim |
|----------|------|------------------|:---:|
| **TSFM (ours)** | Text-aligned | Cosine similarity | 384 |
| **LanHAR** | Text-aligned | Cosine similarity (SciBERT) | 768 |
| **LiMU-BERT** | Encoder-only | GRU classifier | 72 |
| **MOMENT** | Encoder-only | SVM-RBF classifier | 6144 |
| **CrossHAR** | Encoder-only | Transformer_ft classifier | 72 |

### 4-Metric Evaluation

1. **Zero-Shot Open-Set**: Classify against all 87 training labels
2. **Zero-Shot Closed-Set**: Classify against test dataset labels only
3. **1% Supervised**: End-to-end fine-tuning on 1% labeled test data
4. **10% Supervised**: End-to-end fine-tuning on 10% labeled test data

### Running Evaluations

```bash
# Individual baselines
python val_scripts/human_activity_recognition/evaluate_tsfm.py
python val_scripts/human_activity_recognition/evaluate_limubert.py
python val_scripts/human_activity_recognition/evaluate_moment.py
python val_scripts/human_activity_recognition/evaluate_crosshar.py
python val_scripts/human_activity_recognition/evaluate_lanhar.py

# Generate combined comparison table
python scripts/generate_results_table.py
```

Results are saved to `test_output/baseline_evaluation/{model}_evaluation.json`.

---

## Documentation

| Document | Description |
|----------|-------------|
| **[docs/baselines/EVALUATION_PROTOCOL.md](docs/baselines/EVALUATION_PROTOCOL.md)** | Evaluation framework, fairness justifications, per-dataset label coverage |
| **[docs/baselines/BASELINE_IMPLEMENTATION_NOTES.md](docs/baselines/BASELINE_IMPLEMENTATION_NOTES.md)** | Per-baseline implementation details and design decisions |
| **[docs/baselines/RESULTS.md](docs/baselines/RESULTS.md)** | Current evaluation results table |
| **[model/README.md](model/README.md)** | Model architecture API |
| **[datasets/imu_pretraining_dataset/README.md](datasets/imu_pretraining_dataset/README.md)** | Dataset format and loading |
| **[training_scripts/human_activity_recognition/README.md](training_scripts/human_activity_recognition/README.md)** | Training pipeline details |
| **[benchmark_data/README.md](benchmark_data/README.md)** | Benchmark data format |

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
├── benchmark_data/                 # Standardized evaluation data
├── docs/baselines/                 # Evaluation protocol & results
├── scripts/                        # Utility scripts
├── tests/                          # Pytest regression suite
├── data/                           # Raw + processed training data
└── training_output/                # Checkpoints, plots, logs
```

---

## Testing

```bash
pytest tests/ -v
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision
pip install -r requirements.txt

# Download and convert all datasets
python datascripts/setup_all_datasets.py
```
