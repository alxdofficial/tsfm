"""
IMU Activity Recognition Encoder

A patch-based encoder for IMU sensor data with fixed 96-timestep processing.

Key features:
- Variable input channels (9-40 channels)
- Fixed patch size of 96 timesteps after interpolation
- Per-patch, per-channel normalization
- Multi-scale 1D CNN feature extraction
- Temporal and cross-channel attention
- Masking support for pretraining

Architecture flow:
1. Patching: Split time series into patches
2. Interpolation: Resize all patches to 96 timesteps
3. Normalization: Z-score per patch, per channel
4. Feature Extraction: Fixed 1D CNN for 96 timesteps
5. Positional Encoding: Temporal + channel semantic
6. Transformer: Temporal + cross-channel attention
"""

from .encoder import IMUActivityRecognitionEncoder

__all__ = ['IMUActivityRecognitionEncoder']
