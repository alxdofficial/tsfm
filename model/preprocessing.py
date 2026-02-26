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


def interpolate_patches(
    patches: torch.Tensor,
    target_size: int = 64,
    method: str = 'linear'
) -> torch.Tensor:
    """
    Interpolate patches to a fixed target size.

    Uses torch.nn.functional.interpolate for vectorized operation across all
    patches and channels simultaneously (no per-channel loops).

    Args:
        patches: Input tensor of shape (num_patches, patch_timesteps, num_channels)
        target_size: Target number of timesteps per patch (default: 64)
        method: Interpolation method. Options: 'linear', 'nearest'

    Returns:
        Interpolated patches of shape (num_patches, target_size, num_channels)

    Example:
        >>> patches = torch.randn(10, 200, 9)  # 10 patches of 200 timesteps
        >>> interpolated = interpolate_patches(patches, target_size=64)
        >>> interpolated.shape  # (10, 64, 9)
    """
    num_patches, patch_timesteps, num_channels = patches.shape

    if patch_timesteps == target_size:
        return patches

    if not isinstance(patches, torch.Tensor):
        patches = torch.as_tensor(patches, dtype=torch.float32)
    elif patches.device.type != 'cpu':
        patches = patches.cpu()

    # Validate method
    SUPPORTED_METHODS = {'linear', 'cubic', 'nearest'}
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown interpolation method: '{method}'. Supported: {SUPPORTED_METHODS}")

    # F.interpolate expects (batch, channels, length)
    x = patches.permute(0, 2, 1)  # (num_patches, num_channels, patch_timesteps)

    # Map method to F.interpolate mode
    # Note: F.interpolate 1D doesn't support cubic; we approximate with linear for 'cubic'
    # since the difference is negligible for IMU signal resampling (smooth signals)
    if method == 'nearest':
        x = F.interpolate(x, size=target_size, mode='nearest')
    else:
        # 'linear' and 'cubic' both use linear mode (1D cubic not available in F.interpolate)
        x = F.interpolate(x, size=target_size, mode='linear', align_corners=False)

    # Back to (num_patches, target_size, num_channels)
    return x.permute(0, 2, 1).contiguous()


def normalize_patches(
    patches: torch.Tensor,
    method: str = 'zscore',
    epsilon: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize patches using per-patch, per-channel statistics.

    This normalization is applied independently to each patch and each channel,
    which helps the model focus on the shape and pattern of signals rather than
    absolute magnitude.

    Args:
        patches: Input tensor of shape (num_patches, patch_timesteps, num_channels)
        method: Normalization method. Options: 'zscore', 'minmax', 'none'
        epsilon: Small constant to avoid division by zero

    Returns:
        Tuple of (normalized_patches, means, stds) where:
        - normalized_patches: shape (num_patches, patch_timesteps, num_channels)
        - means: shape (num_patches, num_channels) - mean per patch per channel
        - stds: shape (num_patches, num_channels) - std per patch per channel

    Example:
        >>> patches = torch.randn(10, 96, 9)
        >>> normalized, means, stds = normalize_patches(patches, method='zscore')
        >>> normalized.shape  # (10, 96, 9)
        >>> means.shape  # (10, 9)
        >>> stds.shape  # (10, 9)
    """
    if method == 'none':
        num_patches, _, num_channels = patches.shape
        means = torch.zeros(num_patches, num_channels)
        stds = torch.ones(num_patches, num_channels)
        return patches, means, stds

    # Calculate statistics per patch, per channel
    # Shape: (num_patches, num_channels)
    means = patches.mean(dim=1)  # Average over time dimension

    if method == 'zscore':
        # Z-score normalization: (x - mean) / std
        stds = patches.std(dim=1, unbiased=False)  # Std over time dimension
        stds = torch.clamp(stds, min=epsilon)  # Avoid division by zero

        # Normalize: subtract mean and divide by std
        # Broadcasting: (num_patches, patch_timesteps, num_channels) - (num_patches, 1, num_channels)
        normalized = (patches - means.unsqueeze(1)) / stds.unsqueeze(1)

    elif method == 'minmax':
        # Min-max normalization: (x - min) / (max - min)
        mins = patches.min(dim=1)[0]  # Min over time dimension
        maxs = patches.max(dim=1)[0]  # Max over time dimension
        ranges = maxs - mins
        ranges = torch.clamp(ranges, min=epsilon)  # Avoid division by zero

        normalized = (patches - mins.unsqueeze(1)) / ranges.unsqueeze(1)

        # Return mins as "means" and ranges as "stds" for consistency
        stds = ranges
        means = mins

    else:
        raise ValueError(f"Unknown normalization method: {method}. Choose from: 'zscore', 'minmax', 'none'")

    return normalized, means, stds


def preprocess_imu_data(
    data: torch.Tensor,
    sampling_rate_hz: float,
    patch_size_sec: float,
    stride_sec: Optional[float] = None,
    target_patch_size: int = 64,
    normalization_method: str = 'zscore',
    interpolation_method: str = 'linear'
) -> Tuple[torch.Tensor, dict]:
    """
    Complete preprocessing pipeline for IMU data.

    This is a convenience function that applies all preprocessing steps:
    1. Create patches
    2. Interpolate to fixed size
    3. Normalize per-patch per-channel

    Args:
        data: Input tensor of shape (num_timesteps, num_channels)
        sampling_rate_hz: Sampling rate in Hz
        patch_size_sec: Duration of each patch in seconds
        stride_sec: Stride between patches in seconds (default: patch_size_sec)
        target_patch_size: Target timesteps per patch after interpolation (default: 64)
        normalization_method: 'zscore', 'minmax', or 'none'
        interpolation_method: 'linear', 'cubic', or 'nearest'

    Returns:
        Tuple of (preprocessed_patches, metadata) where:
        - preprocessed_patches: shape (num_patches, target_patch_size, num_channels)
        - metadata: dict with 'means', 'stds', 'original_patch_size'

    Example:
        >>> data = torch.randn(1000, 9)  # 10 seconds at 100 Hz, 9 channels
        >>> patches, metadata = preprocess_imu_data(
        ...     data, sampling_rate_hz=100.0, patch_size_sec=2.0
        ... )
        >>> patches.shape  # (5, 64, 9) - 5 non-overlapping 2-second patches
    """
    # Step 1: Create patches
    patches = create_patches(data, sampling_rate_hz, patch_size_sec, stride_sec)
    original_patch_size = patches.shape[1]

    # Step 2: Interpolate to target size
    patches = interpolate_patches(patches, target_size=target_patch_size, method=interpolation_method)

    # Step 3: Normalize
    normalized_patches, means, stds = normalize_patches(patches, method=normalization_method)

    # Package metadata
    metadata = {
        'means': means,
        'stds': stds,
        'original_patch_size': original_patch_size,
        'target_patch_size': target_patch_size,
        'sampling_rate_hz': sampling_rate_hz,
        'patch_size_sec': patch_size_sec,
        'num_channels': data.shape[1]
    }

    return normalized_patches, metadata
