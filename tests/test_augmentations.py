"""Test 4: Augmentation pipeline correctness.

Verifies rotation_3d preserves norms, respects triads, and apply() respects aug_prob.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.imu_pretraining_dataset.augmentations import IMUAugmentation


class TestRotation3D:
    """Test SO(3) rotation augmentation."""

    @pytest.fixture
    def triad_channel_names(self):
        return ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

    @pytest.fixture
    def augmenter(self, triad_channel_names):
        return IMUAugmentation(
            aug_types=['rotation_3d'],
            aug_prob=1.0,
            channel_names=triad_channel_names,
        )

    def test_preserves_norms(self, augmenter, triad_channel_names):
        """Rotation should preserve the L2 norm of each 3D vector."""
        B, T, C = 4, 100, 6
        data = torch.randn(B, T, C)

        rotated = augmenter.apply(data, channel_names=triad_channel_names)

        # Check norm preservation for acc triad (channels 0-2)
        orig_norms = data[:, :, :3].norm(dim=-1)
        rot_norms = rotated[:, :, :3].norm(dim=-1)
        assert torch.allclose(orig_norms, rot_norms, atol=1e-5), \
            f"Rotation changed acc norms: max diff = {(orig_norms - rot_norms).abs().max()}"

        # Check norm preservation for gyro triad (channels 3-5)
        orig_norms_g = data[:, :, 3:6].norm(dim=-1)
        rot_norms_g = rotated[:, :, 3:6].norm(dim=-1)
        assert torch.allclose(orig_norms_g, rot_norms_g, atol=1e-5), \
            f"Rotation changed gyro norms"

    def test_rotation_is_proper(self, augmenter, triad_channel_names):
        """Rotation matrix should have determinant +1 (proper rotation, not reflection)."""
        B, T, C = 1, 3, 6
        # Use 3 linearly independent vectors to recover the rotation matrix
        data = torch.eye(3).unsqueeze(0)  # (1, 3, 3)
        # Pad with zeros for gyro channels
        data_padded = torch.zeros(1, 3, 6)
        data_padded[:, :, :3] = data

        rotated = augmenter.apply(data_padded, channel_names=triad_channel_names)
        # The rotation matrix is the rotated identity
        R = rotated[0, :, :3]  # (3, 3) = rotation matrix
        det = torch.det(R)
        assert abs(det.item() - 1.0) < 1e-4, f"Rotation det should be +1, got {det.item()}"

    def test_same_rotation_for_same_location_triads(self, triad_channel_names):
        """Triads at the same body location should share the same rotation."""
        # Both acc and gyro are at the same location (no prefix = default location)
        augmenter = IMUAugmentation(
            aug_types=['rotation_3d'],
            aug_prob=1.0,
            channel_names=triad_channel_names,
        )
        B, T, C = 1, 10, 6
        data = torch.randn(B, T, C)
        rotated = augmenter.apply(data, channel_names=triad_channel_names)

        # Extract the rotation matrices from both triads
        # Use first timestep to derive rotation: rotated = R @ original
        orig_acc = data[0, 0, :3]
        rot_acc = rotated[0, 0, :3]
        orig_gyro = data[0, 0, 3:6]
        rot_gyro = rotated[0, 0, 3:6]

        # If same rotation R applied: rot_acc = R @ orig_acc, rot_gyro = R @ orig_gyro
        # Verify by checking that the angle between acc and gyro vectors is preserved
        orig_cos = torch.dot(orig_acc, orig_gyro) / (orig_acc.norm() * orig_gyro.norm() + 1e-8)
        rot_cos = torch.dot(rot_acc, rot_gyro) / (rot_acc.norm() * rot_gyro.norm() + 1e-8)
        assert abs(orig_cos.item() - rot_cos.item()) < 1e-4, \
            "Same rotation should preserve angles between triads at same location"

    def test_skips_non_triad_groups(self):
        """Groups with != 3 channels (e.g., quaternion _1/_2/_3/_4) should be skipped."""
        # Quaternion group has 4 channels -> not a triad -> should be untouched
        channel_names = ['ori_1', 'ori_2', 'ori_3', 'ori_4']
        augmenter = IMUAugmentation(
            aug_types=['rotation_3d'],
            aug_prob=1.0,
            channel_names=channel_names,
        )
        B, T, C = 2, 50, 4
        data = torch.randn(B, T, C)
        rotated = augmenter.apply(data, channel_names=channel_names)

        assert torch.allclose(data, rotated), \
            "Non-triad groups (4 channels) should be unchanged by rotation"

    def test_no_channel_names_returns_unchanged(self):
        """Without channel names, rotation_3d should return data unchanged."""
        augmenter = IMUAugmentation(
            aug_types=['rotation_3d'],
            aug_prob=1.0,
            channel_names=None,
        )
        data = torch.randn(2, 50, 6)
        rotated = augmenter.apply(data, channel_names=None)
        assert torch.allclose(data, rotated)


class TestApplyAugProb:
    """Test that aug_prob=0 means no augmentation."""

    def test_prob_zero_returns_identical(self):
        augmenter = IMUAugmentation(
            aug_types=['jitter', 'scale', 'time_shift'],
            aug_prob=0.0,
        )
        data = torch.randn(4, 100, 9)
        result = augmenter.apply(data)
        assert torch.allclose(data, result), \
            "With aug_prob=0, output should match input"

    def test_prob_one_changes_input(self):
        augmenter = IMUAugmentation(
            aug_types=['jitter'],
            aug_prob=1.0,
        )
        data = torch.randn(4, 100, 9)
        result = augmenter.apply(data)
        # Jitter adds noise, so output should differ
        assert not torch.allclose(data, result), \
            "With aug_prob=1 and jitter, output should differ from input"

    def test_output_shape_preserved(self):
        augmenter = IMUAugmentation(
            aug_types=['jitter', 'scale', 'time_warp'],
            aug_prob=0.8,
        )
        B, T, C = 4, 200, 9
        data = torch.randn(B, T, C)
        result = augmenter.apply(data)
        assert result.shape == data.shape, f"Shape changed: {data.shape} -> {result.shape}"
