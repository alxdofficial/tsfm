# Human Activity Recognition - Training Pipeline

Two-stage training pipeline for the IMU Activity Recognition Encoder:
1. **Stage 1: Self-Supervised Pretraining** (MAE + Contrastive)
2. **Stage 2: Semantic Alignment** (Text-IMU alignment with prototype learning)

## Quick Start

```bash
cd training_scripts/human_activity_recognition

# Stage 1: MAE + Contrastive pretraining
python pretrain.py

# Resume from checkpoint
python pretrain.py --resume path/to/checkpoint.pt

# Stage 2: Semantic alignment (after Stage 1)
python semantic_alignment_train.py
```

## File Structure

```
training_scripts/human_activity_recognition/
├── pretrain.py                  # Stage 1: MAE + Contrastive pretraining
├── semantic_alignment_train.py  # Stage 2: Text-IMU semantic alignment
├── losses.py                    # Stage 1 losses (MAE + contrastive)
├── semantic_loss.py             # Stage 2 losses (InfoNCE, single/multi-prototype)
├── memory_bank.py               # MoCo-style momentum memory bank
├── PROTOTYPES_README.md         # Prototype soft targets documentation
└── README.md                    # This file
```

## Stage 1: Self-Supervised Pretraining (`pretrain.py`)

### Objectives

1. **Masked Autoencoding (MAE)**
   - Structured masking: random (40%), span masking (40%), channel dropout (20%)
   - Span masking: contiguous spans of length 2-4 until ~30% ratio
   - Channel dropout: drops 30% of channels, keeps minimum 1
   - Normalized MSE loss on masked patches

2. **Contrastive Learning (Patch-level InfoNCE)**
   - Positive pairs: same patch from augmented views
   - Negative pairs: same patch position, different samples
   - Temperature: 0.2

3. **Combined Loss**
   ```
   total_loss = mae_weight * mae_loss + contrastive_weight * contrastive_loss
   ```

### Augmentations

- Weak: jitter, scale, time_shift
- Strong: time_warp, magnitude_warp
- SO(3) rotation: random 3D rotation of sensor triads (orientation invariance)
- Channel shuffle: robustness to channel ordering

### Configuration

All hyperparameters are hard-coded constants in `pretrain.py` (top of `main()`):

```python
DATASETS = ['uci_har', 'hhar', 'mhealth', 'pamap2', 'wisdm', 'unimib_shar',
            'dsads', 'hapt', 'ku_har', 'recgym']
D_MODEL = 384
BATCH_SIZE = 20
LEARNING_RATE = 1e-4
EPOCHS = 100
MASK_RATIO = 0.5
```

## Stage 2: Semantic Alignment (`semantic_alignment_train.py`)

### Objectives

- **Text-IMU Alignment**: Aligns IMU embeddings with activity text descriptions using InfoNCE loss
- **Prototype Learning**: K=3 learnable prototypes per activity class (LearnableLabelBank)
- **Memory Bank**: MoCo-style FIFO queue for additional negatives
- **Channel Text Fusion**: Cross-attention between sensor tokens and channel description tokens

### Key Components

- **SemanticAlignmentModel**: Wraps encoder + SemanticAlignmentHead + ChannelTextFusion
- **LearnableLabelBank**: Learnable attention pooling for multi-prototype label encoding
- **InfoNCELoss**: Contrastive loss with single/multi-prototype support, soft targets
- **MomentumMemoryBank**: FIFO queue storing recent embeddings as additional negatives

### Configuration

Hard-coded constants in `semantic_alignment_train.py`:

```python
D_MODEL = 384
NUM_PROTOTYPES = 3
MEMORY_BANK_SIZE = 256
LEARNING_RATE = 1e-4
FREEZE_ENCODER = False  # Encoder unfrozen for discriminative learning
```

## Training Outputs

```
training_output/imu_pretraining/{timestamp}/
├── config.yaml                 # Saved configuration
├── plots/                      # PNG loss curves
│   ├── overall_loss.png
│   ├── loss_components.png
│   ├── per_dataset_losses.png
│   └── metrics.json
├── latest.pt                   # Latest checkpoint
├── best.pt                     # Best validation loss
└── checkpoint_epoch_*.pt       # Periodic checkpoints
```

## Loading Pretrained Models

```python
from val_scripts.human_activity_recognition.model_loading import load_model, load_label_bank

model, checkpoint, hyperparams_path = load_model('path/to/best.pt', device)
label_bank = load_label_bank(checkpoint, device, hyperparams_path)
```

## Testing

```bash
# Run regression test suite (111 tests)
pytest tests/ -v

# Run specific test files
pytest tests/test_losses.py -v
pytest tests/test_memory_bank.py -v
```

## Expected Performance

- **Stage 1**: Initial loss ~3-4, converged ~0.5-1.0. Training: ~8-12 hours on single GPU.
- **Stage 2**: Converges in ~50-100 epochs. Zero-shot accuracy on unseen datasets: 40-60%.
- **GPU memory**: ~8-12 GB with mixed precision (FP16) at batch size 32.

## References

- **TS-TCC**: Time-Series Temporal and Contextual Contrasting
- **PRIMUS**: Pretraining IMU Encoders (NeurIPS 2024)
- **Ti-MAE**: Time Series Masked Autoencoders (2023)
- **MoCo**: Momentum Contrast for Unsupervised Visual Representation Learning
