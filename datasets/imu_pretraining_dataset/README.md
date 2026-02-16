# IMU Pretraining Dataset

Multi-dataset loader for pretraining the IMU Activity Recognition Encoder.

## Overview

This dataset loader combines multiple IMU activity recognition datasets with:
- **Random dataset selection** per batch
- **Random channel subset selection** (6-52 channels)
- **Automatic train/val/test splitting** (70/15/15)
- **Padding and masking** for variable-length sequences
- **Compatible with all sampling rates** (20-200 Hz)

## Datasets Included

1. **UCI HAR** - 50 Hz, 9 channels
2. **MHEALTH** - 50 Hz, 23 channels
3. **PAMAP2** - 100 Hz, 40 channels
4. **WISDM** - 20 Hz, 6 channels

**Note:** ActionSense is excluded from pretraining as requested.

## Features

### Variable Channel Sampling
- Randomly selects N channels per sample
- Min/max channels configurable per dataset
- Preserves channel semantic information

### Automatic Preprocessing
- Loads from standardized parquet format
- Handles variable sampling rates
- Creates attention masks for padding
- Provides channel descriptions for semantic encoding

### Train/Val/Test Splits
- 70% training
- 15% validation
- 15% test
- Reproducible with seed parameter

## Usage

### Basic Usage

```python
from datasets.imu_pretraining_dataset import IMUPretrainingDataset

# Create dataset
dataset = IMUPretrainingDataset(
    data_root="/path/to/data",
    datasets=['uci_har', 'mhealth', 'pamap2', 'wisdm'],
    split='train',
    patch_size_sec=2.0,
    min_channels=6,
    max_channels=40,
    seed=42
)

# Get a sample
sample = dataset[0]
print(f"Data shape: {sample['data'].shape}")
print(f"Dataset: {sample['metadata']['dataset']}")
print(f"Channels: {sample['metadata']['num_channels']}")
```

### With DataLoader

```python
from datasets.imu_pretraining_dataset import create_dataloaders

# Create train/val/test loaders
train_loader, val_loader, test_loader = create_dataloaders(
    data_root="/path/to/data",
    datasets=['uci_har', 'mhealth', 'pamap2', 'wisdm'],
    batch_size=32,
    num_workers=4,
    patch_size_sec=2.0,
    seed=42
)

# Iterate
for batch in train_loader:
    data = batch['data']  # (batch, timesteps, channels)
    attention_mask = batch['attention_mask']  # (batch, timesteps)
    channel_mask = batch['channel_mask']  # (batch, channels)
    metadata = batch['metadata']  # List of dicts
```

## Sample Structure

Each sample contains:

```python
{
    'data': torch.Tensor,  # (timesteps, num_channels)
    'attention_mask': torch.BoolTensor,  # (timesteps,) True=valid
    'metadata': {
        'dataset': str,  # 'uci_har', 'mhealth', etc.
        'session_id': str,  # Unique session identifier
        'label': list,  # Activity labels
        'channels': list,  # Selected channel names
        'channel_descriptions': list,  # Channel descriptions
        'sampling_rate_hz': float,  # Sampling rate
        'patch_size_sec': float,  # Patch duration
        'num_channels': int  # Number of channels
    }
}
```

## Batch Structure

Batches are automatically padded and masked:

```python
{
    'data': torch.Tensor,  # (batch, max_timesteps, max_channels)
    'attention_mask': torch.BoolTensor,  # (batch, max_timesteps)
    'channel_mask': torch.BoolTensor,  # (batch, max_channels)
    'metadata': list  # List of metadata dicts
}
```

## Augmentations

The package includes physically plausible augmentations for IMU data:

### Weak Augmentations
- **Jitter**: Add Gaussian noise (σ=0.05)
- **Scale**: Multiply by random factor (0.9-1.1)
- **Time Shift**: Small temporal shifts (±5%)

### Strong Augmentations
- **Time Warp**: Cubic spline warping with 4 knots
- **Magnitude Warp**: Variable scaling via spline
- **Resample**: Slight resampling (95%-105%)

### SO(3) Rotation Augmentation
- **Rotation 3D**: Random proper rotation (det=+1) applied to sensor triads
  - Groups channels into triads via `group_channels_by_sensor()` (e.g., acc_x/y/z → "acc")
  - Same rotation matrix for all triads at the same body location
  - Skips non-triad groups (quaternion channels, single-axis channels)
  - Generated via QR decomposition of random Gaussian matrix
  - Applied with probability 0.8 (controlled by `aug_prob`)
  - Handles sensor orientation variance across device placements

### Usage

```python
from datasets.imu_pretraining_dataset import IMUAugmentation

# Create augmentation
aug = IMUAugmentation(
    aug_types=['jitter', 'scale', 'time_warp'],
    aug_prob=0.8
)

# Apply to data
augmented_data = aug.apply(data, attention_mask)

# Create positive pair for contrastive learning
positive_pair = aug.create_positive_pair(data, attention_mask)
```

## Dataset Statistics

After loading, you'll see:

```
Loaded 9917 sessions for train split from 4 datasets
Loaded 2125 sessions for val split from 4 datasets
Loaded 2126 sessions for test split from 4 datasets
```

Total: **14,168 sessions** across all datasets

## Channel Selection Strategy

- UCI HAR: 6-9 channels
- MHEALTH: 6-23 channels
- PAMAP2: 6-52 channels
- WISDM: 3-6 channels

Channels are randomly sampled from available channels each time.

## Dependencies

- `torch >= 2.0.0`
- `pandas >= 1.5.0`
- `numpy >= 1.20.0`
- `scipy >= 1.7.0`

## File Structure

```
datasets/imu_pretraining_dataset/
├── __init__.py
├── multi_dataset_loader.py    # Main dataset class
├── augmentations.py            # Augmentation implementations
└── README.md                   # This file
```

## Notes

- All augmentations preserve physical plausibility
- Padding is automatically handled with attention masks
- Channel descriptions enable semantic encoding
- Supports any sampling rate through automatic resampling
