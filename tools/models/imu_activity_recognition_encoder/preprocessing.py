"""
Preprocessing module for IMU Activity Recognition Encoder

Handles patching, interpolation, and normalization of IMU sensor data.
"""

import torch
import numpy as np
from scipy import interpolate
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

    # Convert to numpy for easier indexing
    if isinstance(data, torch.Tensor):
        data_np = data.cpu().numpy()  # Move to CPU first if on GPU
    else:
        data_np = np.array(data)

    num_timesteps, num_channels = data_np.shape

    # Calculate patch size in timesteps
    patch_timesteps = int(sampling_rate_hz * patch_size_sec)
    stride_timesteps = int(sampling_rate_hz * stride_sec)

    if patch_timesteps > num_timesteps:
        raise ValueError(
            f"Patch size ({patch_timesteps} timesteps) is larger than data length ({num_timesteps} timesteps). "
            f"Reduce patch_size_sec or provide more data."
        )

    # Calculate number of patches
    num_patches = (num_timesteps - patch_timesteps) // stride_timesteps + 1

    # Create patches
    patches = []
    for i in range(num_patches):
        start_idx = i * stride_timesteps
        end_idx = start_idx + patch_timesteps
        patch = data_np[start_idx:end_idx, :]
        patches.append(patch)

    patches = np.stack(patches, axis=0)  # (num_patches, patch_timesteps, num_channels)

    return torch.from_numpy(patches).float()


def interpolate_patches(
    patches: torch.Tensor,
    target_size: int = 64,
    method: str = 'linear'
) -> torch.Tensor:
    """
    Interpolate patches to a fixed target size.

    This is a key operation that normalizes all patches to the same temporal resolution,
    regardless of original sampling rate or patch duration.

    Args:
        patches: Input tensor of shape (num_patches, patch_timesteps, num_channels)
        target_size: Target number of timesteps per patch (default: 96)
        method: Interpolation method. Options: 'linear', 'cubic', 'nearest'

    Returns:
        Interpolated patches of shape (num_patches, target_size, num_channels)

    Example:
        >>> patches = torch.randn(10, 200, 9)  # 10 patches of 200 timesteps
        >>> interpolated = interpolate_patches(patches, target_size=96)
        >>> interpolated.shape  # (10, 96, 9)
    """
    num_patches, patch_timesteps, num_channels = patches.shape

    if patch_timesteps == target_size:
        # Already at target size, no interpolation needed
        return patches

    # Convert to numpy for scipy interpolation
    if isinstance(patches, torch.Tensor):
        patches_np = patches.cpu().numpy()  # Move to CPU first if on GPU
    else:
        patches_np = np.array(patches)

    # Create interpolation grids
    original_grid = np.linspace(0, 1, patch_timesteps)
    target_grid = np.linspace(0, 1, target_size)

    # Interpolate each patch and channel
    interpolated_patches = np.zeros((num_patches, target_size, num_channels))

    for patch_idx in range(num_patches):
        for channel_idx in range(num_channels):
            signal = patches_np[patch_idx, :, channel_idx]

            if method == 'linear':
                f = interpolate.interp1d(original_grid, signal, kind='linear')
            elif method == 'cubic':
                # Use cubic spline, fallback to linear if not enough points
                if patch_timesteps >= 4:
                    f = interpolate.interp1d(original_grid, signal, kind='cubic')
                else:
                    f = interpolate.interp1d(original_grid, signal, kind='linear')
            elif method == 'nearest':
                f = interpolate.interp1d(original_grid, signal, kind='nearest')
            else:
                raise ValueError(f"Unknown interpolation method: {method}")

            interpolated_patches[patch_idx, :, channel_idx] = f(target_grid)

    return torch.from_numpy(interpolated_patches).float()


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
    target_patch_size: int = 96,
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
        target_patch_size: Target timesteps per patch after interpolation (default: 96)
        normalization_method: 'zscore', 'minmax', or 'none'
        interpolation_method: 'linear', 'cubic', or 'nearest'

    Returns:
        Tuple of (preprocessed_patches, metadata) where:
        - preprocessed_patches: shape (num_patches, 96, num_channels)
        - metadata: dict with 'means', 'stds', 'original_patch_size'

    Example:
        >>> data = torch.randn(1000, 9)  # 10 seconds at 100 Hz, 9 channels
        >>> patches, metadata = preprocess_imu_data(
        ...     data, sampling_rate_hz=100.0, patch_size_sec=2.0
        ... )
        >>> patches.shape  # (5, 96, 9) - 5 non-overlapping 2-second patches
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
