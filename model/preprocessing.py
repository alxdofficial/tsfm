"""
Preprocessing module for IMU Activity Recognition Encoder

Handles patching, interpolation, and normalization of IMU sensor data.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def create_patches(
    data: torch.Tensor,
    sampling_rate_hz: float,
    patch_size_sec: float,
    stride_sec: Optional[float] = None
) -> torch.Tensor:
    """
    Split time series data into fixed-duration patches.

    Args:
        data: Input tensor of shape (num_timesteps, num_channels)
        sampling_rate_hz: Sampling rate of the data in Hz
        patch_size_sec: Duration of each patch in seconds
        stride_sec: Stride between patches in seconds. If None, uses patch_size_sec (non-overlapping)

    Returns:
        Patches tensor of shape (num_patches, patch_timesteps, num_channels)
        where patch_timesteps = int(sampling_rate_hz * patch_size_sec)

    Example:
        >>> data = torch.randn(1000, 9)  # 1000 timesteps, 9 channels
        >>> patches = create_patches(data, sampling_rate_hz=50.0, patch_size_sec=2.0)
        >>> patches.shape  # (10, 100, 9) - 10 patches of 100 timesteps each
    """
    if stride_sec is None:
        stride_sec = patch_size_sec

    if not isinstance(data, torch.Tensor):
        data = torch.as_tensor(data, dtype=torch.float32)
    elif data.device.type != 'cpu':
        data = data.cpu()

    num_timesteps, num_channels = data.shape

    patch_timesteps = int(sampling_rate_hz * patch_size_sec)
    stride_timesteps = int(sampling_rate_hz * stride_sec)

    if patch_timesteps > num_timesteps:
        raise ValueError(
            f"Patch size ({patch_timesteps} timesteps) is larger than data length ({num_timesteps} timesteps). "
            f"Reduce patch_size_sec or provide more data."
        )

    # Use unfold for vectorized patching: (timesteps, channels) -> (num_patches, patch_timesteps, channels)
    # unfold operates on dim=0 (time), giving (num_patches, channels, patch_timesteps)
    # then transpose to (num_patches, patch_timesteps, channels)
    patches = data.t().unsqueeze(0)  # (1, channels, timesteps)
    patches = patches.unfold(2, patch_timesteps, stride_timesteps)  # (1, channels, num_patches, patch_timesteps)
    patches = patches.squeeze(0)  # (channels, num_patches, patch_timesteps)
    patches = patches.permute(1, 2, 0)  # (num_patches, patch_timesteps, channels)

    return patches.contiguous()


def zero_pad_patches(
    patches: torch.Tensor,
    target_size: int
) -> torch.Tensor:
    """
    Zero-pad native-rate patches to a fixed DFT length (for the PHz-Filterbank
    tokenizer). Unlike interpolate_patches, this does NOT resample — the first N
    samples are the real signal and the tail is zeros, so the tokenizer's rDFT
    (n=target_size) sees the native content with no aliasing.

    Args:
        patches: (num_patches, N, num_channels) native-rate patches
        target_size: DFT size S to pad to (must be >= N)

    Returns:
        (num_patches, target_size, num_channels) with real samples in [0, N)
    """
    num_patches, n, num_channels = patches.shape
    if n == target_size:
        return patches
    if n > target_size:
        raise ValueError(
            f"Native patch length ({n}) exceeds DFT size ({target_size}); "
            f"raise dft_size so that sampling_rate * patch_size_sec <= dft_size."
        )
    out = patches.new_zeros(num_patches, target_size, num_channels)
    out[:, :n, :] = patches
    return out


def preprocess_imu_data(
    data: torch.Tensor,
    sampling_rate_hz: float,
    patch_size_sec: float,
    stride_sec: Optional[float] = None,
    pad_to_size: int = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Preprocessing pipeline for the V2 filterbank tokenizer.

    Create native-rate patches -> zero-pad each to the DFT size S (pad_to_size).
    NO interpolation and NO per-patch z-score — the PHz-Filterbank tokenizer does its
    own DC removal + amplitude preservation. Metadata carries patch_len_samples (true
    N). The legacy interpolate-to-fixed-size + z-score path was removed with the CNN /
    spectral-temporal extractors.

    Args:
        data: Input tensor of shape (num_timesteps, num_channels)
        sampling_rate_hz: Sampling rate in Hz
        patch_size_sec: Duration of each patch in seconds
        stride_sec: Stride between patches in seconds (default: patch_size_sec)
        pad_to_size: DFT size S to zero-pad native patches to (required).

    Returns:
        Tuple of (patches (num_patches, S, num_channels), metadata) with
        metadata['patch_len_samples'] = true N.
    """
    if pad_to_size is None:
        raise ValueError("preprocess_imu_data requires pad_to_size (the filterbank DFT size S)")

    patches = create_patches(data, sampling_rate_hz, patch_size_sec, stride_sec)
    original_patch_size = patches.shape[1]
    patches = zero_pad_patches(patches, target_size=pad_to_size)
    metadata = {
        'means': None,
        'stds': None,
        'original_patch_size': original_patch_size,
        'patch_len_samples': original_patch_size,   # true N for the tokenizer
        'dft_size': pad_to_size,
        'sampling_rate_hz': sampling_rate_hz,
        'patch_size_sec': patch_size_sec,
        'num_channels': data.shape[1],
    }
    return patches, metadata
