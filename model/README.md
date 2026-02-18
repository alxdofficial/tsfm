# IMU Activity Recognition Encoder

A patch-based transformer encoder for IMU sensor data with fixed 64-timestep processing.

## Overview

This encoder processes raw IMU sensor data through multiple stages to produce rich semantic representations suitable for activity recognition, motion capture, and other time-series classification tasks.

### Key Features

- **Variable input support**: Works with 6-52 channels and any sampling rate (20-200 Hz)
- **Fixed patch size**: All patches interpolated to 64 timesteps for consistent architecture
- **Channel-independent processing**: Scales efficiently to many channels
- **1D CNN feature extraction**: Channel-independent Conv1d (kernel=5) for temporal features
- **Temporal attention**: Models dependencies across time patches
- **Channel semantic encoding**: Optional encoding of channel meanings using Sentence-BERT

### Architecture

```
Raw IMU Data (timesteps × channels)
    ↓
1. Preprocessing
   - Patching: Split into fixed-duration windows
   - Interpolation: Resize to 64 timesteps
   - Normalization: Z-score per patch, per channel
    ↓
2. Feature Extraction
   - Channel-independent 1D CNN (kernel: 5)
   - Channel-independent processing
   - Output: (patches × channels × d_model)
    ↓
3. Positional Encoding
   - Temporal: Sinusoidal encoding of patch position
   - Channel semantic: Sentence-BERT encoding of channel descriptions
    ↓
4. Transformer
   - Channel-independent temporal attention
   - Multiple layers with residual connections
   - Output: (patches × channels × d_model)
    ↓
Encoded Representations
```

## Installation

### Dependencies

Required:
```bash
pip install torch scipy numpy
```

Optional (for channel semantic encoding):
```bash
pip install sentence-transformers
```

## Usage

### Basic Usage

```python
import torch
from encoder import IMUActivityRecognitionEncoder

# Create encoder with default configuration
encoder = IMUActivityRecognitionEncoder(d_model=384)

# Encode raw sensor data
data = torch.randn(1000, 9)  # 10 seconds at 100 Hz, 9 channels
features, metadata = encoder.encode_from_raw(
    data,
    sampling_rate_hz=100.0,
    patch_size_sec=2.0
)

print(f"Encoded shape: {features.shape}")  # (5, 9, 384)
# 5 patches, 9 channels, 384 features per patch-channel
```

### Using Different Model Sizes

```python
from config import get_config
from encoder import IMUActivityRecognitionEncoder

# Small model (fast, less accurate)
small_config = get_config("small")
encoder_small = IMUActivityRecognitionEncoder(**small_config)

# Default model (balanced)
default_config = get_config("default")
encoder_default = IMUActivityRecognitionEncoder(**default_config)

# Large model (slow, more accurate)
large_config = get_config("large")
encoder_large = IMUActivityRecognitionEncoder(**large_config)
```

### Using Channel Descriptions

```python
# Define channel meanings
channel_descriptions = [
    "accelerometer x-axis body",
    "accelerometer y-axis body",
    "accelerometer z-axis body",
    "gyroscope x-axis body",
    "gyroscope y-axis body",
    "gyroscope z-axis body",
    "accelerometer x-axis total",
    "accelerometer y-axis total",
    "accelerometer z-axis total"
]

# Encode with channel semantic information
features, metadata = encoder.encode_from_raw(
    data,
    sampling_rate_hz=100.0,
    patch_size_sec=2.0,
    channel_descriptions=channel_descriptions
)
```

### Batched Processing

```python
# Process multiple samples at once
batched_data = torch.randn(32, 1000, 9)  # 32 samples
features, metadata = encoder.encode_from_raw(
    batched_data,
    sampling_rate_hz=100.0,
    patch_size_sec=2.0
)

print(f"Encoded shape: {features.shape}")  # (32, 5, 9, 384)
```

### Overlapping Patches

```python
# Use overlapping patches for smoother representations
features, metadata = encoder.encode_from_raw(
    data,
    sampling_rate_hz=100.0,
    patch_size_sec=2.0,
    stride_sec=1.0  # 50% overlap
)

print(f"Number of patches: {features.shape[0]}")  # More patches due to overlap
```

### Custom Configuration

```python
encoder = IMUActivityRecognitionEncoder(
    # Model architecture
    d_model=384,
    num_heads=8,
    num_temporal_layers=4,
    dim_feedforward=1536,
    dropout=0.1,

    # CNN parameters
    cnn_channels=[32, 64],
    cnn_kernel_sizes=[5],

    # Preprocessing
    target_patch_size=64,
    normalization_method='zscore',
    interpolation_method='linear',

    # Positional encoding
    temporal_init_scale=0.1,
    channel_init_scale=0.1,
    use_channel_encoding=True
)
```

## Datasets Tested

The encoder has been tested on the following datasets:

| Dataset | Sampling Rate | Channels | Activities |
|---------|--------------|----------|------------|
| UCI HAR | 50 Hz | 9 | 6 activities |
| ActionSense | 200 Hz | 30 | 9 activities |
| MHEALTH | 50 Hz | 23 | 12 activities |
| PAMAP2 | 100 Hz | 40 | 12 activities |

## Module Structure

```
imu_activity_recognition_encoder/
├── __init__.py                 # Package initialization
├── encoder.py                  # Main encoder class
├── transformer.py              # Dual-branch transformer (temporal + cross-channel)
├── feature_extractor.py        # Multi-scale 1D CNN
├── positional_encoding.py      # Temporal + channel semantic encoding
├── preprocessing.py            # Patching, interpolation, normalization
├── semantic_alignment.py       # Semantic alignment head + projection + label bank
├── token_text_encoder.py       # Token-level text encoding (SentenceTransformer)
├── config.py                   # Default configurations
└── README.md                   # This file
```

## Configuration Parameters

### Model Architecture
- `d_model` (int): Feature dimension throughout model (default: 384)
- `num_heads` (int): Number of attention heads (default: 8)
- `num_temporal_layers` (int): Number of transformer layers (default: 4)
- `dim_feedforward` (int): Hidden dimension in FFN (default: 1536)
- `dropout` (float): Dropout probability (default: 0.1)

### CNN Parameters
- `cnn_channels` (List[int]): Channel progression (default: [32, 64])
- `cnn_kernel_sizes` (List[int]): Convolution kernel sizes (default: [5])

### Preprocessing
- `target_patch_size` (int): Fixed size after interpolation (default: 64)
- `normalization_method` (str): 'zscore', 'minmax', or 'none' (default: 'zscore')
- `interpolation_method` (str): 'linear', 'cubic', or 'nearest' (default: 'linear')

### Positional Encoding
- `temporal_init_scale` (float): Initial scale for temporal encoding (default: 0.1)
- `channel_init_scale` (float): Initial scale for channel encoding (default: 0.1)
- `use_channel_encoding` (bool): Enable channel semantic encoding (default: True)
- `sentence_bert_model` (str): Sentence-BERT model name (default: 'all-MiniLM-L6-v2')

## Testing

```bash
# Run the project regression test suite (111 tests)
pytest tests/ -v

# Run encoder-specific tests
pytest tests/test_encoder_forward.py -v
pytest tests/test_model_loading.py -v
```

## Implemented Extensions

The following features are fully implemented:

1. **Cross-channel attention**: Dual-branch transformer with temporal + cross-channel attention
2. **Masked autoencoding (MAE)**: Structured masking (random + span + channel dropout) for self-supervised pretraining
3. **Semantic alignment**: Text-IMU alignment with learnable multi-prototype label bank
4. **Channel text fusion**: Cross-attention between sensor tokens and channel description tokens

See [`training_scripts/human_activity_recognition/README.md`](../training_scripts/human_activity_recognition/README.md) for training details.

## Citation

If you use this encoder in your research, please cite:

```
@software{imu_activity_recognition_encoder,
  title = {IMU Activity Recognition Encoder},
  year = {2025},
  author = {Your Name},
  description = {A patch-based transformer encoder for IMU sensor data}
}
```

## License

[Your License Here]

## Contact

[Your Contact Information Here]
