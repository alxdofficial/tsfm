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


# =============================================================================
# Unified, configurable augmentation system (V2)
# =============================================================================
# Every augmentation is switched on/off and tuned from a single
# AugmentationConfig, so it is obvious at a glance which augmentations are
# active. Physics/metadata-changing augmentations (gravity, rate, channel
# dropout) also update the per-sample channel description / sampling rate so the
# model's channel-text conditioning stays consistent with the augmented signal
# (the loader appends the "sampled at NHz" suffix from sample.sampling_rate).

import random as _random
from dataclasses import dataclass, field
from dataclasses import fields as _dc_fields
from fractions import Fraction
from scipy import signal as _sps


# ---- Per-augmentation config specs (each has `enabled` + `p` + its params) ----
@dataclass
class JitterCfg:
    """Additive Gaussian sensor noise."""
    enabled: bool = True
    p: float = 0.5
    sigma: float = 0.05


@dataclass
class ScaleCfg:
    """Per-channel amplitude scaling (gain/calibration variance)."""
    enabled: bool = True
    p: float = 0.5
    low: float = 0.9
    high: float = 1.1


@dataclass
class TimeShiftCfg:
    """Whole-window temporal (phase) shift."""
    enabled: bool = False
    p: float = 0.5
    max_ratio: float = 0.05


@dataclass
class TimeWarpCfg:
    """Non-linear time warp (cadence variation)."""
    enabled: bool = False
    p: float = 0.3
    n_knots: int = 4
    strength: float = 0.2


@dataclass
class MagnitudeWarpCfg:
    """Smooth per-channel amplitude modulation over time."""
    enabled: bool = False
    p: float = 0.3
    n_knots: int = 4
    strength: float = 0.3


@dataclass
class GravityCfg:
    """P1 — add/remove gravity. Subtracts a low-pass gravity estimate to
    manufacture the iOS `userAcceleration` (gravity-removed) representation the
    training corpus otherwise lacks; annotates the acc channel text accordingly."""
    enabled: bool = False
    p: float = 0.5
    cutoff_hz: float = 0.4   # gravity is quasi-DC; human motion energy is > ~0.5 Hz
    order: int = 2


@dataclass
class YawRotationCfg:
    """P2 — rotate sensor triads about the (estimated) gravity axis only. Randomizes
    heading (arbitrary for a pocketed phone) while preserving 'which way is down',
    so the posture cue survives (unlike full SO(3), which is why plain rotation
    was disabled)."""
    enabled: bool = False
    p: float = 0.5
    max_deg: float = 180.0


@dataclass
class RateCfg:
    """P3 — anti-aliased resample to a random sampling rate (teaches rate-invariance).
    Updates sample.sampling_rate so the Hz channel-text token co-varies."""
    enabled: bool = False
    p: float = 0.5
    min_hz: float = 15.0
    max_hz: float = 100.0
    min_samples: int = 32   # skip if the resampled window would be shorter than this


@dataclass
class ChannelDropoutCfg:
    """P4 — drop a whole sensor group (e.g. gyro) so the model is robust to
    deployments that expose only an accelerometer. Updates channel list + text.

    Note: reduces a sample's channel count, so within a ChannelBucketBatchSampler
    bucket batches become channel-heterogeneous. This is correct (collate pads to
    the batch max and the per-sample channel_mask isolates padding) but slightly
    reduces the sampler's padding-efficiency; keep `p` modest for that reason."""
    enabled: bool = False
    p: float = 0.3
    groups: tuple = ("gyro",)   # channel-name substrings eligible for dropping


@dataclass
class LabelTextCfg:
    """Label paraphrase: dataset-specific synonym swap + template wrapping (augment_label).
    Effective augmentation rate == `p` (the augmenter's outer gate; augment_label is called
    with rate=1.0 once selected)."""
    enabled: bool = False
    p: float = 0.8
    use_synonyms: bool = True
    use_templates: bool = True


@dataclass
class ChannelTextPhraseCfg:
    """Paraphrase each channel description: swap ONLY sensor-family / axis surface forms and
    wrap in a template. Placement, units, and gravity state are left verbatim, so the
    load-bearing semantics are provably preserved (checked by the measurement harness)."""
    enabled: bool = False
    p: float = 0.5   # fraction of samples whose channel descriptions get paraphrased


@dataclass
class ChannelTextDropoutCfg:
    """Neutralize a random subset of channel descriptions (KEEP the signal) so the model is
    robust to unknown/missing placement metadata. Never neutralizes more than `max_frac`."""
    enabled: bool = False
    p: float = 0.15          # fraction of samples that get any channel-text neutralized
    max_frac: float = 0.5    # never neutralize more than this fraction of a sample's channels
    neutral: str = "an inertial sensor channel"


# Conservative, meaning-preserving substitutions for channel-description paraphrase. Only
# sensor-family + axis SURFACE FORMS are swapped; placement/units/gravity are never touched.
_CH_SYNONYMS = [
    (r"\baccelerometer\b", ["accelerometer", "acceleration sensor", "accelerometer sensor"]),
    (r"\bacceleration\b", ["acceleration", "linear acceleration"]),
    (r"\bgyroscope\b", ["gyroscope", "gyro", "angular rate sensor"]),
    (r"\bangular velocity\b", ["angular velocity", "angular rate", "rotational velocity"]),
    (r"\bmagnetometer\b", ["magnetometer", "magnetic field sensor"]),
    (r"\bmagnetic field\b", ["magnetic field", "magnetic flux"]),
    (r"\bx-axis\b", ["x-axis", "x axis"]),
    (r"\by-axis\b", ["y-axis", "y axis"]),
    (r"\bz-axis\b", ["z-axis", "z axis"]),
    (r"\bmounted\b", ["mounted", "worn", "placed"]),
]
_CH_TEMPLATES = ["{}", "channel: {}", "sensor channel — {}", "signal from {}", "this channel measures {}"]


def _paraphrase_channel(desc: str) -> str:
    """Surface-form paraphrase of one channel description (sensor/axis synonyms + template).
    re.escape not needed — replacements are plain words; placement/units are never matched."""
    import re
    out = desc
    for pat, options in _CH_SYNONYMS:
        if re.search(pat, out, flags=re.I):
            out = re.sub(pat, _random.choice(options), out, flags=re.I)
    return _random.choice(_CH_TEMPLATES).format(out)


@dataclass
class AugmentationConfig:
    """Single source of truth for which augmentations run and how strong they are.

    Defaults reproduce the legacy behaviour (jitter + scale only). Presets:
      - AugmentationConfig()            -> legacy (jitter + scale)
      - AugmentationConfig.default_v2() -> P1-P4 curriculum ON (+ jitter + scale)
      - AugmentationConfig.none()       -> everything off
    Print `cfg.summary()` to see the ON/OFF table.
    """
    jitter: JitterCfg = field(default_factory=JitterCfg)
    scale: ScaleCfg = field(default_factory=ScaleCfg)
    time_shift: TimeShiftCfg = field(default_factory=TimeShiftCfg)
    time_warp: TimeWarpCfg = field(default_factory=TimeWarpCfg)
    magnitude_warp: MagnitudeWarpCfg = field(default_factory=MagnitudeWarpCfg)
    gravity: GravityCfg = field(default_factory=GravityCfg)
    yaw_rotation: YawRotationCfg = field(default_factory=YawRotationCfg)
    rate: RateCfg = field(default_factory=RateCfg)
    channel_dropout: ChannelDropoutCfg = field(default_factory=ChannelDropoutCfg)
    # Text augmentations (unified here so ALL augmentation lives in one config).
    channel_text_phrase: ChannelTextPhraseCfg = field(default_factory=ChannelTextPhraseCfg)
    channel_text_dropout: ChannelTextDropoutCfg = field(default_factory=ChannelTextDropoutCfg)
    label_text: LabelTextCfg = field(default_factory=LabelTextCfg)

    # Application order: metadata/physics-changing first, then value-space, then TEXT last
    # (so channel-text augs see the final, physics-mutated channel set/descriptions).
    # yaw_rotation runs BEFORE gravity (it needs gravity present in the acc to estimate the
    # rotation axis); rate runs after gravity/rotation.
    ORDER = ("channel_dropout", "yaw_rotation", "gravity", "rate",
             "time_warp", "time_shift", "magnitude_warp", "scale", "jitter",
             "channel_text_phrase", "channel_text_dropout", "label_text")

    @classmethod
    def default_v2(cls) -> "AugmentationConfig":
        cfg = cls()
        cfg.gravity.enabled = True
        cfg.yaw_rotation.enabled = True
        cfg.rate.enabled = True
        cfg.channel_dropout.enabled = True
        cfg.channel_text_phrase.enabled = True
        cfg.channel_text_dropout.enabled = True
        cfg.label_text.enabled = True
        return cfg

    @classmethod
    def legacy(cls) -> "AugmentationConfig":
        """Only jitter + scale (pre-V2 effective behaviour)."""
        return cls()

    @classmethod
    def none(cls) -> "AugmentationConfig":
        cfg = cls()
        for name in cls.ORDER:
            getattr(cfg, name).enabled = False
        return cfg

    def summary(self) -> str:
        lines = ["Augmentation config (ON/OFF + params):"]
        for name in self.ORDER:
            spec = getattr(self, name)
            params = ", ".join(
                f"{f.name}={getattr(spec, f.name)}"
                for f in _dc_fields(spec) if f.name not in ("enabled", "p")
            )
            flag = "ON " if spec.enabled else "off"
            lines.append(f"  [{flag}] {name:16s} p={spec.p:<4} {params}")
        return "\n".join(lines)


@dataclass
class IMUSample:
    """Per-sample carrier threaded through the augmenter. Physics augmentations mutate
    sampling_rate / channel_names / channel_descriptions and the TEXT augmentations mutate
    channel_descriptions / label_text, so the loader reads back a fully-augmented sample."""
    data: "torch.Tensor"              # (T, C)
    channel_names: List[str]
    sampling_rate: float
    channel_descriptions: List[str]   # base per-channel text (no Hz/window suffix)
    label: str = ""                   # raw activity label (input to label-text augmentation)
    dataset_name: str = ""            # for dataset-specific label synonyms
    label_text: str = ""              # augmented label text (output; defaults to raw label)

    def __post_init__(self):
        if not self.label_text:
            self.label_text = self.label


def _gravity_present(triad: "np.ndarray", descs=None) -> bool:
    """True if an accelerometer triad still contains the gravity DC component.

    Prefers the DOCUMENTED gravity state (channel-description text is authoritative);
    only falls back to a hardened signal heuristic when the text is silent. The old
    pure DC/RMS ratio misfired on normalized data (e.g. recgym acc centered ~0.5, where
    dc/rms~1) and on low-motion gravity-removed data (e.g. kuhar static postures), so it
    now also requires the DC vector to be axis-concentrated (real gravity points ~down =
    one dominant axis; a uniform per-axis offset spreads across axes and is NOT gravity).
    """
    if descs:
        j = " ".join(str(d).lower() for d in descs)
        if any(k in j for k in ("gravity removed", "gravity-removed", "user acceleration",
                                "useracceleration", "linear acceleration")):
            return False
        if any(k in j for k in ("includes gravity", "including gravity", "with gravity",
                                "gravity included")):
            return True
    a = triad if isinstance(triad, np.ndarray) else triad.detach().cpu().numpy()
    a = a.astype(np.float64)
    m = a.mean(axis=0)
    dc = float(np.linalg.norm(m))
    rms = float(np.sqrt((a ** 2).sum(axis=1).mean())) + 1e-8
    axis_conc = float(np.max(np.abs(m))) / (dc + 1e-8)   # ->1 if one axis dominates
    return dc > 0.5 * rms and axis_conc > 0.6


def _mark_gravity_removed(desc: str) -> str:
    """Rewrite a channel description to state gravity was removed, stripping any
    contradictory 'includes gravity' clause first (avoids 'includes gravity
    (gravity removed)')."""
    import re
    d = re.sub(r"\([^)]*includes gravity[^)]*\)", "", desc, flags=re.I)
    d = re.sub(r"\bincludes gravity\b", "", d, flags=re.I)
    d = re.sub(r"\s{2,}", " ", d).strip().rstrip(",").strip()
    if "gravity removed" not in d.lower():
        d = f"{d} (gravity removed)"
    return d


def _rodrigues(axis: "torch.Tensor", theta: float) -> "torch.Tensor":
    """Rotation matrix (3x3, float32) about a unit `axis` (torch, len 3) by `theta`."""
    ax = axis.detach().cpu().numpy().astype(np.float64)
    n = np.linalg.norm(ax)
    if n < 1e-8:
        return torch.eye(3, dtype=torch.float32)
    x, y, z = ax / n
    c, sn = np.cos(theta), np.sin(theta)
    Cc = 1.0 - c
    R = np.array([
        [c + x * x * Cc,     x * y * Cc - z * sn, x * z * Cc + y * sn],
        [y * x * Cc + z * sn, c + y * y * Cc,     y * z * Cc - x * sn],
        [z * x * Cc - y * sn, z * y * Cc + x * sn, c + z * z * Cc],
    ], dtype=np.float32)
    return torch.from_numpy(R)


class IMUAugmenter:
    """Applies the enabled augmentations (in AugmentationConfig.ORDER) to an
    IMUSample. Operates per sample on (T, C) tensors — padding is added later in
    collate, so no attention mask is needed here."""

    def __init__(self, config: "AugmentationConfig"):
        self.cfg = config

    def __call__(self, sample: "IMUSample") -> "IMUSample":
        for name in AugmentationConfig.ORDER:
            spec = getattr(self.cfg, name)
            if not spec.enabled or _random.random() >= spec.p:
                continue
            sample = getattr(self, "_" + name)(sample, spec)
        return sample

    # ---------- triad helper ----------
    @staticmethod
    def _triads(channel_names):
        """Return {location: [(indices3, group_name), ...]} for x/y/z triads only."""
        from datasets.imu_pretraining_dataset.multi_dataset_loader import (
            group_channels_by_sensor,
        )
        groups = group_channels_by_sensor(channel_names)
        ch_to_idx = {n: i for i, n in enumerate(channel_names)}
        sensor_types = ("acc", "gyro", "mag", "ori")

        def location(g):
            for st in sensor_types:
                if g.endswith(st):
                    return g[: -len(st)].rstrip("_")
                if g.startswith(st):
                    return ""
            return g

        out = {}
        for g, chans in groups.items():
            if len(chans) != 3:
                continue
            out.setdefault(location(g), []).append(([ch_to_idx[c] for c in chans], g))
        return out

    # ---------- value-space (ported) ----------
    def _jitter(self, s, spec):
        s.data = s.data + torch.randn_like(s.data) * spec.sigma
        return s

    def _scale(self, s, spec):
        C = s.data.shape[1]
        factors = torch.empty(1, C, device=s.data.device).uniform_(spec.low, spec.high)
        s.data = s.data * factors
        return s

    def _time_shift(self, s, spec):
        T = s.data.shape[0]
        max_shift = max(1, int(T * spec.max_ratio))
        shift = int(np.random.randint(-max_shift, max_shift + 1))
        if shift > 0:
            fill = s.data[:shift].mean(0, keepdim=True).repeat(shift, 1)
            s.data = torch.cat([fill, s.data[:-shift]], 0)
        elif shift < 0:
            fill = s.data[shift:].mean(0, keepdim=True).repeat(-shift, 1)
            s.data = torch.cat([s.data[-shift:], fill], 0)
        return s

    def _time_warp(self, s, spec):
        T, C = s.data.shape
        if T < 10:
            return s
        orig = np.linspace(0, 1, T)
        knots = np.linspace(0, 1, spec.n_knots)
        vals = np.clip(knots + np.random.randn(spec.n_knots) * spec.strength, 0, 1)
        vals[0], vals[-1] = 0, 1
        vals = np.sort(vals)
        warped_t = np.clip(
            interpolate.interp1d(knots, vals, kind="cubic", fill_value="extrapolate")(orig),
            0, 1,
        )
        x = s.data.detach().cpu().numpy()
        out = np.stack(
            [interpolate.interp1d(orig, x[:, c], kind="linear", fill_value="extrapolate")(warped_t)
             for c in range(C)],
            axis=-1,
        )
        s.data = torch.from_numpy(np.ascontiguousarray(out)).float().to(s.data.device)
        return s

    def _magnitude_warp(self, s, spec):
        T, C = s.data.shape
        if T < 10:
            return s
        grid = np.linspace(0, 1, T)
        knots = np.linspace(0, 1, spec.n_knots)
        x = s.data.detach().cpu().numpy().copy()
        for c in range(C):
            facs = np.clip(1.0 + np.random.randn(spec.n_knots) * spec.strength, 0.5, 1.5)
            curve = interpolate.interp1d(knots, facs, kind="cubic", fill_value="extrapolate")(grid)
            x[:, c] = x[:, c] * curve
        s.data = torch.from_numpy(x).float().to(s.data.device)
        return s

    # ---------- P1: gravity add/remove ----------
    def _gravity(self, s, spec):
        sr = float(s.sampling_rate)
        wn = spec.cutoff_hz / (sr / 2.0)
        if not (0.0 < wn < 1.0):     # cutoff above Nyquist (very low rate) -> skip
            return s
        T = s.data.shape[0]
        if T <= 3 * (spec.order + 1):   # filtfilt needs enough samples
            return s
        b, a = _sps.butter(spec.order, wn, btype="low")
        x = s.data.detach().cpu().numpy().astype(np.float64)
        desc = list(s.channel_descriptions)
        changed = False
        for _loc, triads in self._triads(s.channel_names).items():
            for idxs, gname in triads:
                if "acc" not in gname:       # only accelerometer carries gravity
                    continue
                if not _gravity_present(x[:, idxs], [desc[j] for j in idxs]):  # already gravity-removed -> skip
                    continue
                for j in idxs:
                    grav = _sps.filtfilt(b, a, x[:, j])
                    x[:, j] = x[:, j] - grav
                    desc[j] = _mark_gravity_removed(desc[j])
                changed = True
        if changed:
            s.data = torch.from_numpy(x).float().to(s.data.device)
            s.channel_descriptions = desc
        return s

    # ---------- P2: yaw-only (gravity-preserving) rotation ----------
    def _yaw_rotation(self, s, spec):
        triloc = self._triads(s.channel_names)
        if not triloc:
            return s
        x = s.data

        def acc_axis_for(triads):
            for idxs, gname in triads:
                if "acc" in gname:
                    tri = x[:, idxs]
                    # Only a gravity-bearing acc gives a meaningful 'down' axis;
                    # gravity-removed acc (mean~0) would yield an arbitrary axis.
                    if not _gravity_present(tri.detach().cpu().numpy(),
                                            [s.channel_descriptions[k] for k in idxs]):
                        continue
                    g = tri.mean(0)
                    if g.norm() > 1e-6:
                        return g / g.norm()
            return None

        # global fallback axis (first acc triad anywhere)
        global_axis = None
        for _loc, triads in triloc.items():
            global_axis = acc_axis_for(triads)
            if global_axis is not None:
                break

        theta = float(np.deg2rad(np.random.uniform(-spec.max_deg, spec.max_deg)))
        for _loc, triads in triloc.items():
            axis = acc_axis_for(triads)
            if axis is None:
                axis = global_axis
            if axis is None:
                continue
            R = _rodrigues(axis, theta).to(x.dtype).to(x.device)
            for idxs, _gname in triads:
                x[:, idxs] = torch.einsum("ij,tj->ti", R, x[:, idxs])
        s.data = x
        return s

    # ---------- P3: anti-aliased rate resample ----------
    def _rate(self, s, spec):
        old = float(s.sampling_rate)
        new = float(np.random.uniform(spec.min_hz, spec.max_hz))
        if old <= 0 or abs(new - old) < 1e-3:
            return s
        frac = Fraction(new / old).limit_denominator(50)
        up, down = frac.numerator, frac.denominator
        if up < 1 or down < 1:
            return s
        T = s.data.shape[0]
        if int(round(T * up / down)) < spec.min_samples:
            return s
        x = s.data.detach().cpu().numpy()
        y = _sps.resample_poly(x, up, down, axis=0)     # polyphase, anti-aliased
        s.data = torch.from_numpy(np.ascontiguousarray(y)).float().to(s.data.device)
        s.sampling_rate = old * up / down               # actual achieved rate
        return s

    # ---------- P4: channel / sensor-group dropout ----------
    def _channel_dropout(self, s, spec):
        from datasets.imu_pretraining_dataset.multi_dataset_loader import (
            group_channels_by_sensor,
        )
        names = s.channel_names
        drop = {i for i, n in enumerate(names) if any(g in n for g in spec.groups)}
        keep = [i for i in range(len(names)) if i not in drop]
        if not drop or len(keep) < 3:
            return s
        kept_names = [names[i] for i in keep]
        # require at least one full x/y/z triad to survive the drop
        if not any(len(v) == 3 for v in group_channels_by_sensor(kept_names).values()):
            return s
        s.data = s.data[:, keep]
        s.channel_names = kept_names
        s.channel_descriptions = [s.channel_descriptions[i] for i in keep]
        return s

    # ---------- text: channel-description phrase paraphrase ----------
    def _channel_text_phrase(self, s, spec):
        # Paraphrase each channel description independently (surface form only; placement /
        # units / gravity are preserved by construction — see _paraphrase_channel).
        s.channel_descriptions = [_paraphrase_channel(d) for d in s.channel_descriptions]
        return s

    # ---------- text: channel-description dropout (neutralize, keep signal) ----------
    def _channel_text_dropout(self, s, spec):
        n = len(s.channel_descriptions)
        if n <= 1:
            return s
        max_drop = max(1, int(spec.max_frac * n))
        k = _random.randint(1, max_drop)
        idxs = _random.sample(range(n), min(k, n))
        desc = list(s.channel_descriptions)
        for i in idxs:
            desc[i] = spec.neutral
        s.channel_descriptions = desc
        return s

    # ---------- text: label paraphrase (dataset-specific synonyms + templates) ----------
    def _label_text(self, s, spec):
        from datasets.imu_pretraining_dataset.label_augmentation import augment_label
        # Outer `p` already decided we augment; call augment_label unconditionally (rate=1.0).
        s.label_text = augment_label(
            s.label, s.dataset_name, augmentation_rate=1.0,
            use_synonyms=spec.use_synonyms, use_templates=spec.use_templates,
        )
        return s






