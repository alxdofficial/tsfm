# IMU Activity Recognition Encoder - Pretraining

**Self-supervised pretraining of a dual-branch transformer encoder for IMU-based human activity recognition across multiple datasets.**

---

## 🎯 What is This?

This project implements a **pretrained encoder** for IMU (Inertial Measurement Unit) time series data that can be fine-tuned for various activity recognition tasks. The encoder learns robust representations through:

1. **Masked Autoencoding (MAE)** - Reconstructs randomly masked sensor patches
2. **Contrastive Learning** - Aligns augmented views of the same data
3. **Dual-Branch Transformer** - Captures both temporal dynamics and cross-sensor relationships

### Key Features

- ✅ **Cross-channel attention**: Models relationships between different sensors (accelerometer ↔ gyroscope)
- ✅ **Variable channel support**: Handles 6-40 channels with automatic padding/masking
- ✅ **Multi-dataset pretraining**: Trains on UCI HAR, MHEALTH, PAMAP2, WISDM simultaneously
- ✅ **Physically-plausible augmentations**: Jitter, time warp, magnitude scaling, channel shuffling
- ✅ **Mixed precision training**: FP16 for ~50% memory reduction
- ✅ **Per-dataset tracking**: Monitor learning progress per dataset

---

## 🏗️ Architecture Overview

```
Raw IMU Data (variable length, 6-40 channels)
         ↓
    Preprocessing (interpolate to 96-sample patches)
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

**See [`PRETRAINING_FLOW.md`](PRETRAINING_FLOW.md) for detailed data flow diagrams.**

---

## 📂 Repository Structure

```
tsfm/
├── README.md                              # This file
├── PRETRAINING_FLOW.md                    # Complete architecture diagrams
├── HARDCODED_CONFIG_SUMMARY.md            # Configuration documentation
│
├── data/                                  # Datasets (after processing)
│   ├── uci_har/                          # UCI HAR dataset
│   ├── mhealth/                          # MHEALTH dataset
│   ├── pamap2/                           # PAMAP2 dataset
│   └── wisdm/                            # WISDM dataset
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
│           ├── config.py                 # Model configurations
│           └── tests/                    # Unit tests
│
└── training_scripts/                     # Training scripts
    └── imu_tool_pretraining/
        ├── README.md                     # Training documentation
        ├── pretrain.py                   # Main training script (hard-coded config)
        ├── losses.py                     # MAE + Contrastive losses
        └── config.yaml                   # Reference config (not used)
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install torch torchvision
pip install numpy scipy pandas pyarrow
pip install matplotlib  # For plotting
pip install tqdm  # For progress bars
```

### 2. Download & Process Datasets

```bash
# Download and convert all datasets (~20 minutes, ~2GB)
python datascripts/setup_all_datasets.py

# Or process individually
python datascripts/setup_all_datasets.py uci_har
python datascripts/setup_all_datasets.py mhealth
python datascripts/setup_all_datasets.py pamap2
python datascripts/setup_all_datasets.py wisdm
```

This downloads raw data, converts to standardized format, and splits into train/val/test.

### 3. Run Pretraining

```bash
cd training_scripts/imu_tool_pretraining

# Start pretraining (100 epochs, ~8-12 hours on GPU)
python pretrain.py

# Or resume from checkpoint
python pretrain.py --resume path/to/checkpoint.pt
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

| Dataset | Train | Val | Test | Channels | Rate | Activities |
|---------|-------|-----|------|----------|------|------------|
| **UCI HAR** | 7,352 | 1,470 | 1,477 | 6 (acc+gyro) | 50 Hz | 6 activities |
| **MHEALTH** | ~80 | ~20 | ~20 | 23 (3 IMUs+ECG) | 50 Hz | 12 activities |
| **PAMAP2** | ~140 | ~35 | ~25 | 40 (3 IMUs+HR) | 100 Hz | 18 activities |
| **WISDM** | ~630 | ~135 | ~135 | 6 (phone acc+gyro) | 20 Hz | 18 activities |

**Total:** ~8,200 train samples, ~1,660 val samples, ~1,660 test samples

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
DATASETS = ['uci_har', 'mhealth', 'pamap2', 'wisdm']
PATCH_SIZE_SEC = 2.0  # 2-second patches

# Model
D_MODEL = 128
NUM_HEADS = 8
NUM_TEMPORAL_LAYERS = 4
USE_CROSS_CHANNEL = True  # Enable cross-channel attention

# Training
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WARMUP_EPOCHS = 10

# Loss
MAE_WEIGHT = 1.0
CONTRASTIVE_WEIGHT = 1.0
TEMPERATURE = 0.2
MASK_RATIO = 0.5  # 50% masking
```

**See [`HARDCODED_CONFIG_SUMMARY.md`](HARDCODED_CONFIG_SUMMARY.md) for details.**

To change hyperparameters, edit the constants at the top of `main()` in `pretrain.py` (lines 481-520).

---

## 🧪 Model Details

### Encoder Architecture

- **Input:** Patches of shape (batch, num_patches, 96, num_channels)
- **CNN Feature Extraction:** Multi-scale convolutions [3,5,7] kernels → d_model=128
- **Positional Embeddings:** Patch position + channel position
- **Transformer Blocks (×4):**
  - Temporal self-attention (across patches)
  - Cross-channel self-attention (across sensors)
  - Feed-forward network (128 → 512 → 128)
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

- **[PRETRAINING_FLOW.md](PRETRAINING_FLOW.md)** - Complete architecture & data flow diagrams
- **[HARDCODED_CONFIG_SUMMARY.md](HARDCODED_CONFIG_SUMMARY.md)** - Configuration guide
- **[tools/models/imu_activity_recognition_encoder/README.md](tools/models/imu_activity_recognition_encoder/README.md)** - Model API
- **[datasets/imu_pretraining_dataset/README.md](datasets/imu_pretraining_dataset/README.md)** - Dataset details
- **[training_scripts/imu_tool_pretraining/README.md](training_scripts/imu_tool_pretraining/README.md)** - Training details

---

## 🐛 Testing

Run unit tests:

```bash
# Test encoder
python tools/models/imu_activity_recognition_encoder/tests/test_encoder.py

# Test transformer (including cross-channel attention)
python tools/models/imu_activity_recognition_encoder/tests/test_transformer.py

# Test losses
python training_scripts/imu_tool_pretraining/losses.py

# Test augmentations
python datasets/imu_pretraining_dataset/augmentations.py
```

All tests should pass before training.

---

## 📝 Recent Changes

- ✅ Implemented dual-branch transformer with cross-channel attention
- ✅ Fixed critical bugs in per-dataset tracking and normalization
- ✅ Added mixed precision training (AMP)
- ✅ Hard-coded all hyperparameters in main() for easier modification
- ✅ Added channel shuffling augmentation
- ✅ Fixed learning rate scheduler (per-batch stepping)

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

**Branch:** `tool-use-om2` (renamed from tool-use-om)
**Purpose:** IMU activity recognition encoder pretraining
**Status:** Active development - pretraining infrastructure complete, ready for training
