# IMU Tool Pretraining

Pretraining scripts for the IMU Activity Recognition Encoder using dual objectives:
1. **Masked Autoencoding (MAE)** - 50% random masking
2. **Contrastive Learning** - Patch-level contrastive with augmentations

## Overview

This pretraining approach combines:
- **Self-supervised learning** through masked patch reconstruction
- **Contrastive learning** using augmented positive pairs
- **Multi-dataset training** from UCI HAR, MHEALTH, PAMAP2, and WISDM
- **Variable channel support** (6-40 channels)

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision tensorboard pyyaml tqdm pandas numpy scipy
```

### 2. Prepare Data

Ensure datasets are in the standard format under `data/`:
```
data/
├── uci_har/
├── mhealth/
├── pamap2/
└── wisdm/
```

### 3. Run Pretraining

```bash
cd training_scripts/imu_tool_pretraining
python pretrain.py --config config.yaml
```

### 4. Resume from Checkpoint

```bash
python pretrain.py --config config.yaml --resume training_output/imu_pretraining/20250110_123456/best.pt
```

## Configuration

Edit `config.yaml` to customize training:

```yaml
# Key parameters
training:
  epochs: 100
  batch_size: 32
  lr: 1.0e-4

  # Loss weights
  mae_weight: 1.0
  contrastive_weight: 0.5
  temperature: 0.2

  # Masking
  mask_ratio: 0.5  # 50% masking

encoder:
  d_model: 128
  num_temporal_layers: 4
  num_heads: 8
```

## Architecture

### Pretraining Model

```
IMU Encoder (1.2M params)
├── Preprocessing (patching, interpolation, normalization)
├── Feature Extraction (multi-scale CNN)
├── Positional Encoding (temporal + channel semantic)
└── Transformer (temporal attention)

Pretraining Heads:
├── Projection Head (512 → 256) for contrastive
└── Reconstruction Head (d_model → 96) for MAE
```

### Loss Functions

1. **Masked Reconstruction Loss**
   - MSE between predicted and target patches
   - Only on masked, valid positions
   - Per-patch normalization of targets
   - **Structured masking**: Random (40%), span masking (40%), channel dropout (20%)
     - Span masking: contiguous spans of length 2-4 until ~30% ratio
     - Channel dropout: drops 30% of channels, keeps minimum 1

2. **Patch Contrastive Loss**
   - InfoNCE / NT-Xent loss
   - Positive pairs: same patch from augmented views
   - Negative pairs: same patch position, different samples
   - Temperature: 0.2

3. **Combined Loss**
   ```
   total_loss = λ_mae * mae_loss + λ_contrast * contrastive_loss
   λ_mae = 1.0, λ_contrast = 0.5
   ```

## Training Pipeline

### 1. Data Loading
- Random dataset selection per batch
- Random channel sampling (6-40 channels)
- Variable length handling with padding

### 2. Preprocessing
- Patch creation (2-second windows)
- Interpolation to 96 timesteps
- Per-patch normalization

### 3. Augmentation
- Weak: jitter, scale, time_shift
- Strong: time_warp, magnitude_warp
- SO(3) rotation: random 3D rotation of sensor triads (orientation invariance)
- Create positive pairs for contrastive learning

### 4. Forward Pass
- Encode original patches
- Encode augmented patches
- Generate reconstructions
- Project features for contrastive

### 5. Loss Computation
- MAE loss on masked patches
- Contrastive loss on patch features
- Combined weighted loss

### 6. Optimization
- AdamW optimizer
- Learning rate warmup (10 epochs)
- Cosine annealing schedule

## Monitoring

### TensorBoard

```bash
tensorboard --logdir training_output/imu_pretraining
```

**Metrics logged:**
- Train/val loss (total, MAE, contrastive)
- Learning rate
- Masking ratio
- Number of patches per batch

### Checkpoints

Saved to `training_output/imu_pretraining/{timestamp}/`:
- `latest.pt` - Latest checkpoint
- `best.pt` - Best validation loss
- `checkpoint_epoch_N.pt` - Periodic checkpoints

### Checkpoint Contents

```python
{
    'epoch': int,
    'model_state_dict': dict,
    'optimizer_state_dict': dict,
    'train_metrics': dict,
    'val_metrics': dict,
    'config': dict
}
```

## Expected Performance

### Training Time
- **GPU (RTX 3090)**: ~1-2 hours per epoch
- **Batch size 32**: ~310 batches per epoch
- **Total training (100 epochs)**: ~100-200 hours

### Memory Usage
- **GPU**: ~8-10 GB for batch_size=32
- **CPU RAM**: ~4-6 GB

### Loss Values (Typical)
- **Initial loss**: ~3.0-4.0
- **Converged MAE loss**: ~0.5-1.0
- **Converged contrastive loss**: ~0.3-0.7
- **Total loss after 100 epochs**: ~1.0-1.5

## Usage After Pretraining

### Load Pretrained Encoder

```python
import torch
from tools.models.imu_activity_recognition_encoder import IMUActivityRecognitionEncoder

# Load checkpoint
checkpoint = torch.load('best.pt')

# Create encoder
encoder = IMUActivityRecognitionEncoder(d_model=128)

# Load weights (encoder only, not projection/reconstruction heads)
encoder_state = {
    k.replace('encoder.', ''): v
    for k, v in checkpoint['model_state_dict'].items()
    if k.startswith('encoder.')
}
encoder.load_state_dict(encoder_state)

# Use for downstream tasks
encoder.to(device)
encoder.train(False)  # Set to evaluation mode
```

### Fine-tuning

Add a task-specific head and fine-tune on labeled data:

```python
class ActivityClassifier(nn.Module):
    def __init__(self, encoder, num_classes=6):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.d_model, num_classes)

    def forward(self, data):
        # Encode
        features = self.encoder.encode_from_raw(data, ...)

        # Pool over patches and channels
        pooled = features.mean(dim=(0, 1))  # Global average pooling

        # Classify
        logits = self.classifier(pooled)
        return logits
```

## File Structure

```
training_scripts/imu_tool_pretraining/
├── __init__.py
├── pretrain.py        # Main training script
├── losses.py          # MAE + contrastive losses
├── config.yaml        # Training configuration
└── README.md          # This file
```

## Key Features

✅ **Dual-objective pretraining** (MAE + contrastive)
✅ **Multi-dataset training** (4 datasets, 14K+ sessions)
✅ **Variable channel support** (6-40 channels)
✅ **Physically plausible augmentations**
✅ **Automatic padding and masking**
✅ **TensorBoard logging**
✅ **Checkpoint management**
✅ **Learning rate warmup**
✅ **Gradient clipping** (optional)
✅ **Mixed precision training** (optional)

## Hyperparameter Recommendations

Based on research (TS-TCC, PRIMUS, Ti-MAE):

| Parameter | Recommended | Range |
|-----------|-------------|-------|
| Mask ratio | 0.5 (50%) | 0.4-0.6 |
| Temperature | 0.2 | 0.1-0.5 |
| MAE weight | 1.0 | 0.5-2.0 |
| Contrastive weight | 0.5 | 0.3-1.0 |
| Learning rate | 1e-4 | 1e-5 to 1e-3 |
| Batch size | 32 | 16-64 |
| Warmup epochs | 10 | 5-20 |

## Troubleshooting

### Out of Memory
- Reduce `batch_size` to 16 or 8
- Reduce `num_workers` to 2
- Use gradient accumulation

### Slow Training
- Increase `num_workers` to 8
- Use GPU if available
- Reduce `d_model` to 64 (small config)

### Loss Not Decreasing
- Check data loading (verify shapes)
- Reduce learning rate
- Increase warmup epochs
- Check for NaN values

### Contrastive Loss High
- Adjust temperature (try 0.3-0.5)
- Check augmentation strength
- Ensure positive pairs are created correctly

## Citation

If you use this pretraining approach, please cite:

```
@software{imu_pretraining,
  title = {IMU Activity Recognition Encoder Pretraining},
  year = {2025},
  description = {Dual-objective pretraining with MAE and contrastive learning}
}
```

## References

- **TS-TCC**: Time-Series Temporal and Contextual Contrasting
- **PRIMUS**: Pretraining IMU Encoders (NeurIPS 2024)
- **Ti-MAE**: Time Series Masked Autoencoders (2023)
- **TFMAE**: Temporal-Frequency MAE (2024)
