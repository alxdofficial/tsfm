# IMU Activity Recognition Encoder - Pretraining

**Self-supervised pretraining of a dual-branch transformer encoder for IMU-based human activity recognition across multiple datasets.**

---

## 🎯 What is This?

This project implements a **pretrained encoder** for IMU (Inertial Measurement Unit) time series data that can be fine-tuned for various activity recognition tasks. The encoder uses a **two-stage training pipeline**:

### Stage 1: Self-Supervised Pretraining
1. **Masked Autoencoding (MAE)** - Reconstructs randomly masked sensor patches
2. **Contrastive Learning** - Aligns augmented views of the same data
3. **Dual-Branch Transformer** - Captures both temporal dynamics and cross-sensor relationships

### Stage 2: Semantic Alignment
4. **Text-IMU Alignment** - Aligns IMU embeddings with activity text descriptions
5. **Prototype Learning** - Learns activity prototypes with memory bank
6. **Zero-shot Classification** - Enables classification without fine-tuning

### Key Features

- ✅ **Cross-channel attention**: Models relationships between different sensors (accelerometer ↔ gyroscope)
- ✅ **Variable channel support**: Handles 6-52 channels with automatic padding/masking
- ✅ **Multi-dataset pretraining**: Trains on 10 datasets (UCI HAR, HHAR, MHEALTH, PAMAP2, WISDM, UniMiB, DSADS, HAPT, KU-HAR, RecGym)
- ✅ **Semantic alignment**: Align IMU embeddings with natural language descriptions
- ✅ **Multi-prototype learning**: K=3 prototypes per activity class capture intra-class variation
- ✅ **SO(3) rotation augmentation**: Random 3D rotations for sensor orientation invariance
- ✅ **Structured masking**: Span masking + channel dropout for robust Stage 1 pretraining
- ✅ **Temperature-based sampling**: `p_i ~ n_i^0.5` balancing across datasets
- ✅ **Physically-plausible augmentations**: Jitter, time warp, magnitude scaling, channel shuffling, rotation
- ✅ **Mixed precision training**: FP16 with torch.compile for ~4x speedup
- ✅ **Per-dataset tracking**: Monitor learning progress per dataset

---

## 🏗️ Architecture Overview

```
Raw IMU Data (variable length, 6-52 channels)
         ↓
    Preprocessing (interpolate to 64-sample patches)
         ↓
    Patch Tokenization (2-second windows → patches)
         ↓
┌────────────────────────────────────────────────┐
│  Dual-Branch Transformer Encoder               │
│                                                 │
│  For each of 4 transformer blocks:            │
│  1. Temporal Self-Attention                    │
│     └─ Attention over time (patch dim)        │
│  2. Cross-Channel Self-Attention               │
│     └─ Attention over sensors (channel dim)   │
│  3. Feed-Forward Network                       │
│                                                 │
│  Output: (batch, patches, channels, d_model)  │
└────────────────────────────────────────────────┘
         ↓                    ↓
   Projection Head    Reconstruction Head
   (for contrastive)  (for MAE)
         ↓                    ↓
   InfoNCE Loss        MSE Loss
         ↘                  ↙
        Combined Loss → Backprop
```

---

## 📂 Repository Structure

```
tsfm/
├── README.md                              # This file
├── DATA_FORMAT.md                         # Data format specification
├── requirements.txt                       # Python dependencies
│
├── data/                                  # Datasets (after processing)
│   ├── uci_har/                          # UCI HAR dataset
│   ├── hhar/                             # HHAR dataset
│   ├── mhealth/                          # MHEALTH dataset
│   ├── pamap2/                           # PAMAP2 dataset
│   ├── wisdm/                            # WISDM dataset
│   ├── unimib_shar/                      # UniMiB SHAR dataset
│   └── motionsense/                      # MotionSense dataset
│
├── datascripts/                          # Dataset download & conversion
│   ├── README.md                         # Dataset documentation
│   ├── setup_all_datasets.py            # Master pipeline
│   └── {dataset}/
│       ├── download.py                   # Download raw data
│       └── convert.py                    # Convert to standard format
│
├── datasets/                             # PyTorch dataset classes
│   └── imu_pretraining_dataset/
│       ├── README.md                     # Dataset usage docs
│       ├── multi_dataset_loader.py       # Multi-dataset dataloader
│       └── augmentations.py              # Physical augmentations
│
├── tools/                                # Model implementations
│   └── models/
│       └── imu_activity_recognition_encoder/
│           ├── README.md                 # Model documentation
│           ├── encoder.py                # Main encoder
│           ├── transformer.py            # Dual-branch transformer
│           ├── semantic_alignment.py     # Semantic alignment head
│           ├── token_text_encoder.py     # Text encoding utilities
│           ├── preprocessing.py          # Data preprocessing
│           ├── positional_encoding.py    # Position embeddings
│           └── config.py                 # Model configurations
│
├── training_scripts/                     # Training scripts
│   └── human_activity_recognition/
│       ├── README.md                     # Training documentation
│       ├── pretrain.py                   # Stage 1: MAE + Contrastive pretraining
│       ├── semantic_alignment_train.py   # Stage 2: Text-IMU alignment
│       ├── losses.py                     # MAE + Contrastive losses
│       ├── semantic_loss.py              # Semantic alignment losses
│       └── memory_bank.py                # Prototype memory bank
│
├── val_scripts/                          # Validation and evaluation
│   └── human_activity_recognition/
│       ├── model_loading.py              # Shared model/label bank loading
│       ├── eval_config.py                # Shared eval config (patch sizes, datasets)
│       ├── evaluate_tsfm.py              # TSFM model evaluation
│       ├── compare_models.py             # Model comparison utilities
│       ├── benchmark_baselines.py        # Baseline model benchmarks
│       ├── evaluation_metrics.py         # Accuracy and metrics
│       ├── plot_utils.py                 # Training visualization
│       └── visualization_3d.py           # Embedding visualization
│
└── tests/                                # Regression test suite (pytest)
    ├── test_model_loading.py             # Model construction & loading
    ├── test_encoder_forward.py           # Encoder forward pass & masks
    ├── test_similarity_computation.py    # Similarity & metrics
    ├── test_losses.py                    # MAE, contrastive, InfoNCE losses
    ├── test_augmentations.py             # SO(3) rotation & augmentations
    ├── test_memory_bank.py               # Memory bank operations
    ├── test_label_groups.py              # Label group mapping
    └── test_data_loading.py              # Dataset & collation
```

### Baseline Metric Protocol (Updated 2026-02-16)

The baseline evaluation scripts under `val_scripts/human_activity_recognition/` now use:

- Fixed-class macro F1 for closed-set metrics (class list is explicit, even when some classes are absent in a split).
- Ambiguity-safe closed-set label mapping: exact label matches are preferred; group-based mapping is only used when it maps to a single target label.
- Strict 1% supervision for MOMENT-style evaluation (no train+val label-budget inflation).
- Full-dataset benchmark loading in `benchmark_baselines.py` (no default session truncation, no random 70/15/15 slicing in benchmark mode).

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision

# Install other dependencies
pip install -r requirements.txt
```

**Dependencies** (see `requirements.txt`):
- numpy, pandas, matplotlib, pyarrow
- scikit-learn, umap-learn (for evaluation/visualization)
- pydantic (configuration validation)
- google-genai (for text embeddings in Stage 2)

### 2. Download & Process Datasets

```bash
# Download and convert all datasets (~20 minutes, ~2GB)
python datascripts/setup_all_datasets.py

# Or process individually
python datascripts/setup_all_datasets.py uci_har
python datascripts/setup_all_datasets.py hhar
python datascripts/setup_all_datasets.py mhealth
python datascripts/setup_all_datasets.py pamap2
python datascripts/setup_all_datasets.py wisdm
python datascripts/setup_all_datasets.py unimib_shar
python datascripts/setup_all_datasets.py motionsense
```

This downloads raw data, converts to standardized format, and splits into train/val/test.

### 3. Run Pretraining

```bash
cd training_scripts/human_activity_recognition

# Stage 1: MAE + Contrastive pretraining
python pretrain.py

# Or resume from checkpoint
python pretrain.py --resume path/to/checkpoint.pt

# Stage 2: Semantic alignment (after Stage 1)
python semantic_alignment_train.py
```

Training outputs:
```
training_output/imu_pretraining/20250110_143052/
├── config.yaml                 # Saved configuration
├── plots/                      # PNG loss curves
│   ├── overall_loss.png
│   ├── loss_components.png
│   ├── per_dataset_losses.png
│   ├── learning_rate.png
│   ├── dataset_*_detail.png
│   └── metrics.json
├── latest.pt                   # Latest checkpoint
├── best.pt                     # Best validation loss
└── checkpoint_epoch_10.pt      # Periodic checkpoints
```

### 4. Monitor Training

Training automatically generates PNG plots every 10 epochs and at the end of training:

```bash
# Plots are saved to:
training_output/imu_pretraining/{timestamp}/plots/
```

Generated plots:
- `overall_loss.png` - Train/val loss curves
- `loss_components.png` - MAE loss vs Contrastive loss
- `per_dataset_losses.png` - Per-dataset loss curves (UCI HAR, MHEALTH, PAMAP2, WISDM)
- `learning_rate.png` - Learning rate schedule
- `dataset_*_detail.png` - Detailed per-dataset metrics
- `metrics.json` - Raw metrics data

---

## 📊 Datasets

| Dataset | Channels | Rate | Activities | Description |
|---------|----------|------|------------|-------------|
| **UCI HAR** | 6 (acc+gyro) | 50 Hz | 6 | Smartphone IMU activities |
| **HHAR** | 6 (acc+gyro) | 50-200 Hz | 6 | Heterogeneous HAR (multiple devices) |
| **MHEALTH** | 23 (3 IMUs+ECG) | 50 Hz | 12 | Multi-sensor body activities |
| **PAMAP2** | 40 (3 IMUs+HR) | 100 Hz | 18 | Physical activity monitoring |
| **WISDM** | 6 (phone acc+gyro) | 20 Hz | 18 | Smartphone activities |
| **UniMiB SHAR** | 3 (acc) | 50 Hz | 17 | ADL and falls detection |
| **MotionSense** | 12 (acc+gyro+attitude) | 50 Hz | 6 | iPhone motion data |

All datasets are converted to a standardized format with:
- Variable-length time series
- Consistent channel naming
- Activity labels (not used during pretraining)
- Train/val/test splits

---

## ⚙️ Configuration

All hyperparameters are **hard-coded** in `pretrain.py` for easy modification:

```python
# Data
DATASETS = ['uci_har', 'hhar', 'mhealth', 'pamap2', 'wisdm', 'unimib_shar']
PATCH_SIZE_SEC = 2.0  # 2-second patches (varies per dataset)

# Model
D_MODEL = 384
NUM_HEADS = 8
NUM_TEMPORAL_LAYERS = 4
USE_CROSS_CHANNEL = True  # Enable cross-channel attention

# Training
EPOCHS = 100
BATCH_SIZE = 20
LEARNING_RATE = 1e-4
WARMUP_EPOCHS = 10

# Loss
MAE_WEIGHT = 1.0
CONTRASTIVE_WEIGHT = 1.0
TEMPERATURE = 0.2
MASK_RATIO = 0.5  # 50% masking
```

To change hyperparameters, edit the constants at the top of `main()` in `pretrain.py`.

---

## 🧪 Model Details

### Encoder Architecture

- **Input:** Patches of shape (batch, num_patches, 64, num_channels)
- **CNN Feature Extraction:** 1D convolution [kernel=5] → d_model=384
- **Positional Embeddings:** Sinusoidal temporal + SentenceBERT channel semantic
- **Transformer Blocks (×4):**
  - Temporal self-attention (across patches)
  - Cross-channel self-attention (across sensors)
  - Feed-forward network (384 → 1536 → 384)
- **Output:** (batch, num_patches, num_channels, d_model)

### Pretraining Objectives

1. **Masked Autoencoding (MAE):**
   - Randomly mask 50% of patches
   - Reconstruct masked patches from encoded representations
   - Normalized MSE loss (per-patch normalization)

2. **Contrastive Learning (InfoNCE):**
   - Apply augmentations (jitter, scale, time warp, channel shuffle)
   - Maximize agreement between original and augmented views
   - Patch-level contrastive loss across batch

### Augmentations

- **Weak:** jitter, scale, time_shift (preserve semantics)
- **Strong:** time_warp, magnitude_warp, resample (more aggressive)
- **Novel:** channel_shuffle (robustness to channel ordering)
- **SO(3) rotation:** Random 3D rotation applied to sensor triads (acc_x/y/z, gyro_x/y/z). Same rotation matrix for all triads at the same body location. Handles sensor orientation variance across placements.

**See [`datasets/imu_pretraining_dataset/README.md`](datasets/imu_pretraining_dataset/README.md) for augmentation details.**

---

## 📈 Training Tips

### Expected Performance

- Initial loss: ~2-3
- After convergence: ~0.5-1.0
- Training time: 8-12 hours on single GPU (V100/A100)

### Per-Dataset Metrics

Monitor per-dataset losses to identify:
- Which datasets are harder to learn
- Dataset imbalance issues
- Overfitting on specific datasets

### Memory Usage

- With mixed precision (FP16): ~8-12 GB GPU memory
- Batch size 32: ~10 GB
- Reduce batch size if OOM errors occur

### Debugging

If training diverges:
1. Check data loading (run tests in `datasets/imu_pretraining_dataset/`)
2. Verify augmentations aren't too aggressive
3. Reduce learning rate or increase warmup
4. Check for NaN gradients in TensorBoard

---

## 🎓 Fine-Tuning (Future Work)

After pretraining, the encoder can be fine-tuned for:

1. **Activity Classification:** Add linear head, fine-tune on labeled data
2. **Activity Detection:** Add segmentation head for temporal localization
3. **Anomaly Detection:** Train classifier on normal data only
4. **Transfer Learning:** Fine-tune on new datasets/activities

The pretrained weights are in `best.pt` under `model_state_dict['encoder']`.

---

## 🔗 Key Documents

- **[tools/models/imu_activity_recognition_encoder/README.md](tools/models/imu_activity_recognition_encoder/README.md)** - Model API
- **[datasets/imu_pretraining_dataset/README.md](datasets/imu_pretraining_dataset/README.md)** - Dataset details
- **[training_scripts/human_activity_recognition/README.md](training_scripts/human_activity_recognition/README.md)** - Training details
- **[DATA_FORMAT.md](DATA_FORMAT.md)** - Standardized data format specification

---

## 🐛 Testing

Run the regression test suite (111 tests):

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_losses.py -v
pytest tests/test_encoder_forward.py -v
```

All tests should pass before training or after any refactoring.

---

## 📝 Recent Changes

- ✅ **4 accuracy improvements**: SO(3) rotation augmentation, multi-prototype learning (K=3), structured masking (span + channel dropout), temperature-based sampling (alpha=0.5)
- ✅ **4x training speedup**: Batch fusion, channel bucketing, caching, torch.compile + 6 bug fixes
- ✅ **8 bug fixes**: MAE normalization, memory bank boundary, scheduler resume, channel encoding fallback, and more
- ✅ Added Stage 2 semantic alignment training pipeline
- ✅ Implemented text-IMU alignment with learnable label bank
- ✅ Added memory bank for prototype learning
- ✅ Added group-balanced sampling and patch size augmentation
- ✅ Expanded to 10 training datasets (UCI HAR, HHAR, MHEALTH, PAMAP2, WISDM, UniMiB, DSADS, HAPT, KU-HAR, RecGym)
- ✅ 4 zero-shot test datasets excluded from training (MotionSense, RealWorld, MobiAct, VTT-ConIoT)
- ✅ Added embedding visualization tools (3D, 4D video)
- ✅ Implemented evaluation metrics and model comparison utilities

---

## 🤝 Contributing

When working on this codebase:

1. **Test before committing:** Run all tests to ensure nothing broke
2. **Document changes:** Update relevant README files
3. **Follow conventions:** Use the existing code style
4. **Check for bugs:** Run the bug review before major changes

---

## 📜 License

[Add your license here]

---

## 🏷️ Branch Info

**Branch:** `master`
**Purpose:** IMU activity recognition encoder pretraining + semantic alignment
**Status:** Active development - two-stage training pipeline complete
