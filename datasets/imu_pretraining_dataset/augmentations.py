"""
Augmentation strategies for IMU time series data.

Implements physically plausible augmentations for IMU sensor data following
research best practices from TS-TCC, PPDA, and recent literature (2023-2024).

Augmentations are divided into:
- Weak: jitter, scale, time_shift (preserve semantic meaning)
- Strong: time_warp, magnitude_warp, resample (more aggressive)
"""

import torch
import numpy as np
from scipy import interpolate
from typing import Tuple, Optional, List


class IMUAugmentation:
    """
    Augmentation module for IMU time series data.

    Applies combinations of weak and strong augmentations while preserving
    physical plausibility and semantic meaning.
    """

    def __init__(
        self,
        aug_types: List[str] = ['jitter', 'scale', 'time_warp'],
        aug_prob: float = 0.8,
        seed: Optional[int] = None,
        channel_names: Optional[List[str]] = None
    ):
        """
        Args:
            aug_types: List of augmentation types to apply
            aug_prob: Probability of applying each augmentation
            seed: Random seed for reproducibility
            channel_names: Optional channel names (needed for rotation_3d to identify triads)
        """
        self.aug_types = aug_types
        self.aug_prob = aug_prob
        self.channel_names = channel_names

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def apply(
        self,
        data: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        channel_names: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Apply augmentations to data.

        Args:
            data: Input tensor of shape (batch, timesteps, channels)
            attention_mask: Boolean mask where True = valid, False = padding
            channel_names: Optional channel names (overrides self.channel_names for this call)

        Returns:
            Augmented data of same shape
        """
        augmented = data.clone()
        ch_names = channel_names if channel_names is not None else self.channel_names

        for aug_type in self.aug_types:
            if np.random.rand() < self.aug_prob:
                if aug_type == 'jitter':
                    augmented = self.jitter(augmented, attention_mask)
                elif aug_type == 'scale':
                    augmented = self.scale(augmented, attention_mask)
                elif aug_type == 'time_shift':
                    augmented = self.time_shift(augmented, attention_mask)
                elif aug_type == 'time_warp':
                    augmented = self.time_warp(augmented, attention_mask)
                elif aug_type == 'magnitude_warp':
                    augmented = self.magnitude_warp(augmented, attention_mask)
                elif aug_type == 'resample':
                    augmented = self.resample(augmented, attention_mask)
                elif aug_type == 'channel_shuffle':
                    augmented = self.channel_shuffle(augmented, attention_mask)
                elif aug_type == 'rotation_3d':
                    augmented = self.rotation_3d(augmented, attention_mask, ch_names)

        return augmented

    def create_positive_pair(
        self,
        data: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Create augmented positive pair for contrastive learning.

        Args:
            data: Input tensor of shape (batch, timesteps, channels)
            attention_mask: Boolean mask for valid positions

        Returns:
            Augmented version of data
        """
        return self.apply(data, attention_mask)

    # ========== Weak Augmentations ==========

    def jitter(
        self,
        data: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sigma: float = 0.05
    ) -> torch.Tensor:
        """
        Add Gaussian noise (jittering) to simulate sensor noise.

        Applied per-channel independently.

        Args:
            data: Shape (batch, timesteps, channels)
            attention_mask: Valid position mask
            sigma: Standard deviation of Gaussian noise

        Returns:
            Jittered data
        """
        noise = torch.randn_like(data) * sigma

        # Only apply noise to valid (non-padded) positions
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)  # (batch, timesteps, 1)
            noise = noise * mask

        return data + noise

    def scale(
        self,
        data: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        scale_range: Tuple[float, float] = (0.9, 1.1)
    ) -> torch.Tensor:
        """
        Scale signal by random factor to simulate varying motion intensity.

        Applied per-channel independently.

        Args:
            data: Shape (batch, timesteps, channels)
            attention_mask: Valid position mask
            scale_range: (min_scale, max_scale)

        Returns:
            Scaled data
        """
        batch_size, timesteps, num_channels = data.shape

        # Sample scale factor per sample, per channel
        scale_factors = torch.FloatTensor(batch_size, 1, num_channels).uniform_(
            scale_range[0], scale_range[1]
        ).to(data.device)

        return data * scale_factors

    def time_shift(
        self,
        data: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_shift_ratio: float = 0.05
    ) -> torch.Tensor:
        """
        Shift signal in time (phase shift).

        Applied consistently across all channels to maintain temporal alignment.

        Args:
            data: Shape (batch, timesteps, channels)
            attention_mask: Valid position mask
            max_shift_ratio: Maximum shift as ratio of sequence length

        Returns:
            Time-shifted data
        """
        batch_size, timesteps, num_channels = data.shape

        shifted = []
        for i in range(batch_size):
            # Determine valid length
            if attention_mask is not None:
                valid_len = attention_mask[i].sum().item()
            else:
                valid_len = timesteps

            # Random shift amount
            max_shift = int(valid_len * max_shift_ratio)
            shift = np.random.randint(-max_shift, max_shift + 1)

            # Shift all channels together
            if shift > 0:
                # Shift right
                shifted_sample = torch.cat([
                    data[i, :shift].mean(dim=0, keepdim=True).repeat(shift, 1),
                    data[i, :-shift]
                ], dim=0)
            elif shift < 0:
                # Shift left
                shifted_sample = torch.cat([
                    data[i, -shift:],
                    data[i, shift:].mean(dim=0, keepdim=True).repeat(-shift, 1)
                ], dim=0)
            else:
                shifted_sample = data[i]

            shifted.append(shifted_sample)

        return torch.stack(shifted, dim=0)

    def channel_shuffle(
        self,
        data: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Randomly shuffle the order of channels.

        Makes the model robust to channel ordering and encourages learning
        channel-independent features that can be combined in any order.

        Applied per-sample independently (each sample gets different shuffle).

        Args:
            data: Shape (batch, timesteps, channels)
            attention_mask: Valid position mask (unused, but kept for consistency)

        Returns:
            Channel-shuffled data
        """
        batch_size, timesteps, num_channels = data.shape

        shuffled = []
        for i in range(batch_size):
            # Generate random permutation of channel indices
            perm = torch.randperm(num_channels)

            # Apply permutation to channels
            shuffled_sample = data[i, :, perm]  # (timesteps, channels)
            shuffled.append(shuffled_sample)

        return torch.stack(shuffled, dim=0)

    # ========== SO(3) Rotation Augmentation ==========

    def rotation_3d(
        self,
        data: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        channel_names: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Apply random SO(3) rotation to 3-axis sensor triads.

        Simulates different sensor orientations (phone/watch placement variance).
        All triads at the same body location share the same rotation matrix
        (they're on the same physical sensor).

        Only applies to groups with exactly 3 channels (x/y/z triads).
        Groups with other sizes (e.g., quaternion _1/_2/_3/_4, single-axis) are skipped.

        Args:
            data: Shape (batch, timesteps, channels)
            attention_mask: Valid position mask (unused, rotation preserves padding)
            channel_names: List of channel names to identify triads

        Returns:
            Rotated data of same shape
        """
        if channel_names is None:
            return data

        from datasets.imu_pretraining_dataset.multi_dataset_loader import group_channels_by_sensor

        # Group channels into sensor triads
        groups = group_channels_by_sensor(channel_names)

        # Build channel name to index mapping
        ch_to_idx = {name: i for i, name in enumerate(channel_names)}

        # Extract body location from group name (e.g., "chest_acc" → "chest", "acc" → "")
        def get_location(group_name: str) -> str:
            # If group name contains a sensor type suffix, the location is the prefix
            sensor_types = ['acc', 'gyro', 'mag', 'ori']
            for st in sensor_types:
                if group_name.endswith(st):
                    prefix = group_name[:-len(st)].rstrip('_')
                    return prefix
                if group_name.startswith(st):
                    return ''
            return group_name

        # Group triads by location — same location shares same rotation
        location_triads = {}  # location -> list of (group_name, [channel_indices])
        for group_name, channels in groups.items():
            if len(channels) != 3:
                continue  # Skip non-triad groups
            indices = [ch_to_idx[ch] for ch in channels]
            location = get_location(group_name)
            if location not in location_triads:
                location_triads[location] = []
            location_triads[location].append(indices)

        if not location_triads:
            return data

        rotated = data.clone()

        # Generate one random SO(3) matrix per location (shared across batch for consistency)
        for location, triad_list in location_triads.items():
            # Generate random SO(3) via QR decomposition of random Gaussian matrix
            random_matrix = torch.randn(3, 3, device=data.device, dtype=data.dtype)
            Q, R = torch.linalg.qr(random_matrix)
            # Ensure proper rotation (det=+1, not reflection)
            Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
            if torch.det(Q) < 0:
                Q[:, 0] = -Q[:, 0]

            # Apply same R to all triads at this location
            for indices in triad_list:
                # Extract triad: (batch, timesteps, 3)
                triad_data = data[:, :, indices]
                # Rotate: R @ [x,y,z]^T for each timestep
                # einsum 'ij,btj->bti': R is (3,3), triad is (batch, timesteps, 3)
                rotated_triad = torch.einsum('ij,btj->bti', Q, triad_data)
                rotated[:, :, indices] = rotated_triad

        return rotated

    # ========== Strong Augmentations ==========

    def time_warp(
        self,
        data: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        n_knots: int = 4,
        warp_strength: float = 0.2
    ) -> torch.Tensor:
        """
        Apply time warping using cubic spline interpolation.

        Stretches/compresses time slices to simulate speed variations.
        Applied consistently across channels.

        Args:
            data: Shape (batch, timesteps, channels)
            attention_mask: Valid position mask
            n_knots: Number of knots for spline
            warp_strength: Strength of warping (0.2 = ±20%)

        Returns:
            Time-warped data
        """
        batch_size, timesteps, num_channels = data.shape

        warped = []
        for i in range(batch_size):
            # Determine valid length
            if attention_mask is not None:
                valid_len = int(attention_mask[i].sum().item())
            else:
                valid_len = timesteps

            if valid_len < 10:  # Skip if too short
                warped.append(data[i])
                continue

            # Create random time warp
            # Original time grid
            orig_time = np.linspace(0, 1, valid_len)

            # Create warped time grid with random knots
            knot_positions = np.linspace(0, 1, n_knots)
            knot_values = knot_positions + np.random.randn(n_knots) * warp_strength
            knot_values = np.clip(knot_values, 0, 1)
            knot_values[0] = 0  # Fix endpoints
            knot_values[-1] = 1
            knot_values = np.sort(knot_values)  # Ensure monotonic

            # Interpolate warp
            warp_func = interpolate.interp1d(
                knot_positions, knot_values,
                kind='cubic', fill_value='extrapolate'
            )
            warped_time = warp_func(orig_time)
            warped_time = np.clip(warped_time, 0, 1)

            # Apply warp to all channels
            warped_sample = []
            for c in range(num_channels):
                signal = data[i, :valid_len, c].detach().cpu().numpy()
                interp_func = interpolate.interp1d(
                    orig_time, signal,
                    kind='linear', fill_value='extrapolate'
                )
                warped_signal = interp_func(warped_time)
                warped_sample.append(warped_signal)

            warped_sample = np.stack(warped_sample, axis=-1)  # (valid_len, channels)
            warped_sample = torch.from_numpy(warped_sample).float().to(data.device)

            # Pad back to original length if needed
            if valid_len < timesteps:
                padding = torch.zeros(timesteps - valid_len, num_channels).to(data.device)
                warped_sample = torch.cat([warped_sample, padding], dim=0)

            warped.append(warped_sample)

        return torch.stack(warped, dim=0)

    def magnitude_warp(
        self,
        data: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        n_knots: int = 4,
        warp_strength: float = 0.3
    ) -> torch.Tensor:
        """
        Apply magnitude warping using cubic spline.

        Applies variable scaling to different time points.
        Applied per-channel independently.

        Args:
            data: Shape (batch, timesteps, channels)
            attention_mask: Valid position mask
            n_knots: Number of knots for spline
            warp_strength: Strength of magnitude variation

        Returns:
            Magnitude-warped data
        """
        batch_size, timesteps, num_channels = data.shape

        warped = []
        for i in range(batch_size):
            # Determine valid length
            if attention_mask is not None:
                valid_len = int(attention_mask[i].sum().item())
            else:
                valid_len = timesteps

            if valid_len < 10:
                warped.append(data[i])
                continue

            # Create magnitude warp curve (per channel)
            warped_channels = []
            for c in range(num_channels):
                # Create random magnitude curve
                time_points = np.linspace(0, 1, n_knots)
                magnitude_factors = 1.0 + np.random.randn(n_knots) * warp_strength
                magnitude_factors = np.clip(magnitude_factors, 0.5, 1.5)

                # Interpolate to full length
                mag_func = interpolate.interp1d(
                    time_points, magnitude_factors,
                    kind='cubic', fill_value='extrapolate'
                )
                time_grid = np.linspace(0, 1, valid_len)
                magnitude_curve = mag_func(time_grid)
                magnitude_curve = torch.from_numpy(magnitude_curve).float().to(data.device)

                # Apply magnitude warp
                warped_channel = data[i, :valid_len, c] * magnitude_curve

                # Pad if needed
                if valid_len < timesteps:
                    padding = torch.zeros(timesteps - valid_len).to(data.device)
                    warped_channel = torch.cat([warped_channel, padding], dim=0)

                warped_channels.append(warped_channel)

            warped_sample = torch.stack(warped_channels, dim=-1)
            warped.append(warped_sample)

        return torch.stack(warped, dim=0)

    def resample(
        self,
        data: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        resample_range: Tuple[float, float] = (0.95, 1.05)
    ) -> torch.Tensor:
        """
        Resample signal to simulate slight sampling rate variations.

        Applied consistently across channels.

        Args:
            data: Shape (batch, timesteps, channels)
            attention_mask: Valid position mask
            resample_range: (min_factor, max_factor) for resampling

        Returns:
            Resampled data
        """
        batch_size, timesteps, num_channels = data.shape

        resampled = []
        for i in range(batch_size):
            # Determine valid length
            if attention_mask is not None:
                valid_len = int(attention_mask[i].sum().item())
            else:
                valid_len = timesteps

            if valid_len < 10:
                resampled.append(data[i])
                continue

            # Random resample factor
            resample_factor = np.random.uniform(resample_range[0], resample_range[1])
            new_len = int(valid_len * resample_factor)
            new_len = max(10, min(new_len, valid_len * 2))  # Sanity check

            # Resample all channels
            resampled_channels = []
            for c in range(num_channels):
                signal = data[i, :valid_len, c].detach().cpu().numpy()

                # Interpolate
                orig_time = np.linspace(0, 1, valid_len)
                new_time = np.linspace(0, 1, new_len)
                interp_func = interpolate.interp1d(
                    orig_time, signal,
                    kind='linear', fill_value='extrapolate'
                )
                resampled_signal = interp_func(new_time)

                # Crop or pad to original length
                if new_len > valid_len:
                    resampled_signal = resampled_signal[:valid_len]
                elif new_len < valid_len:
                    padding = np.repeat(resampled_signal[-1], valid_len - new_len)
                    resampled_signal = np.concatenate([resampled_signal, padding])

                resampled_signal = torch.from_numpy(resampled_signal).float().to(data.device)

                # Pad to full timesteps if needed
                if valid_len < timesteps:
                    padding = torch.zeros(timesteps - valid_len).to(data.device)
                    resampled_signal = torch.cat([resampled_signal, padding], dim=0)

                resampled_channels.append(resampled_signal)

            resampled_sample = torch.stack(resampled_channels, dim=-1)
            resampled.append(resampled_sample)

        return torch.stack(resampled, dim=0)


def get_weak_augmentation():
    """Get weak augmentation pipeline for contrastive learning."""
    return IMUAugmentation(
        aug_types=['jitter', 'scale', 'time_shift'],
        aug_prob=0.8
    )


def get_strong_augmentation():
    """Get strong augmentation pipeline for contrastive learning."""
    return IMUAugmentation(
        aug_types=['time_warp', 'magnitude_warp', 'resample'],
        aug_prob=0.6
    )


def get_mixed_augmentation():
    """Get mixed weak+strong augmentation pipeline."""
    return IMUAugmentation(
        aug_types=['jitter', 'scale', 'time_warp', 'magnitude_warp', 'rotation_3d'],
        aug_prob=0.7
    )
