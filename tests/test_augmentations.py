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

from datasets.imu_pretraining_dataset.augmentations import (
    IMUAugmentation,
    IMUAugmenter,
    IMUSample,
    AugmentationConfig,
)


def _make_sample(T=300, sr=50.0):
    """Synthetic acc(+gravity on z)+gyro sample for the V2 augmenter tests."""
    t = np.linspace(0, T / sr, T)
    acc = np.stack([0.3 * np.sin(2 * np.pi * 2 * t),
                    0.2 * np.cos(2 * np.pi * 2 * t),
                    1.0 + 0.3 * np.sin(2 * np.pi * 3 * t)], 1)   # z carries ~1g
    gyro = np.stack([0.1 * np.sin(2 * np.pi * 2 * t),
                     0.1 * np.cos(2 * np.pi * 2 * t),
                     0.05 * np.sin(2 * np.pi * t)], 1)
    data = torch.from_numpy(np.concatenate([acc, gyro], 1)).float()
    names = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    return IMUSample(data=data, channel_names=list(names), sampling_rate=sr,
                     channel_descriptions=[f"{n} raw includes gravity" for n in names])


def _only(name):
    cfg = AugmentationConfig.none()
    getattr(cfg, name).enabled = True
    getattr(cfg, name).p = 1.0
    return IMUAugmenter(cfg)


class TestAugmentationConfig:
    def test_legacy_default_is_jitter_scale_only(self):
        cfg = AugmentationConfig()
        on = {n for n in AugmentationConfig.ORDER if getattr(cfg, n).enabled}
        assert on == {"jitter", "scale"}

    def test_default_v2_enables_p1_p4(self):
        cfg = AugmentationConfig.default_v2()
        for n in ("gravity", "yaw_rotation", "rate", "channel_dropout"):
            assert getattr(cfg, n).enabled, n

    def test_none_disables_all(self):
        cfg = AugmentationConfig.none()
        assert not any(getattr(cfg, n).enabled for n in AugmentationConfig.ORDER)

    def test_summary_lists_every_aug(self):
        s = AugmentationConfig.default_v2().summary()
        for n in AugmentationConfig.ORDER:
            assert n in s


class TestNewAugmentations:
    def test_gravity_removes_dc_and_updates_text(self):
        s = _only("gravity")(_make_sample())
        assert abs(s.data[:, 2].mean().item()) < 0.3            # gravity gone from acc_z
        assert "gravity removed" in s.channel_descriptions[2]    # acc text updated
        assert "gravity removed" not in s.channel_descriptions[5]  # gyro untouched
        assert not torch.isnan(s.data).any()

    def test_yaw_preserves_norm_and_gravity_direction(self):
        base = _make_sample()
        s = _only("yaw_rotation")(_make_sample())
        assert torch.allclose(base.data[:, :3].norm(dim=1),
                              s.data[:, :3].norm(dim=1), atol=1e-4)
        g0 = base.data[:, :3].mean(0); g0 = g0 / g0.norm()
        g1 = s.data[:, :3].mean(0); g1 = g1 / g1.norm()
        assert torch.dot(g0, g1).item() > 0.99

    def test_rate_changes_rate_without_nan(self):
        s = _only("rate")(_make_sample())
        assert s.sampling_rate != 50.0
        assert s.data.shape[1] == 6 and not torch.isnan(s.data).any()

    def test_channel_dropout_removes_gyro(self):
        s = _only("channel_dropout")(_make_sample())
        assert s.data.shape[1] == 3
        assert all("gyro" not in n for n in s.channel_names)
        assert len(s.channel_descriptions) == 3

    def test_full_pipeline_no_nan(self):
        aug = IMUAugmenter(AugmentationConfig.default_v2())
        for _ in range(50):
            s = aug(_make_sample())
            assert not torch.isnan(s.data).any() and not torch.isinf(s.data).any()
            assert s.data.shape[1] in (3, 6)

    def test_gravity_skips_already_removed_acc(self):
        """Gravity aug must NOT double-high-pass acc that is already gravity-removed
        (e.g. uci_har body_acc), and must not tag it with a contradictory clause."""
        from datasets.imu_pretraining_dataset.augmentations import _gravity_present
        t = np.linspace(0, 6, 300)
        removed = np.stack([0.1 * np.sin(2 * np.pi * 2 * t)] * 3, 1)  # DC ~ 0
        s = IMUSample(
            data=torch.from_numpy(removed).float(),
            channel_names=["acc_x", "acc_y", "acc_z"],
            sampling_rate=50.0,
            channel_descriptions=["body acc x (gravity removed)"] * 3,
        )
        assert not _gravity_present(removed)
        out = _only("gravity")(s)
        assert torch.allclose(out.data, s.data)                       # signal untouched
        assert out.channel_descriptions[0].lower().count("gravity removed") == 1

    def test_mark_gravity_removed_strips_conflicting_clause(self):
        from datasets.imu_pretraining_dataset.augmentations import _mark_gravity_removed
        d = _mark_gravity_removed("Total acceleration X-axis (raw, includes gravity)")
        assert "includes gravity" not in d.lower()
        assert "gravity removed" in d.lower()


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
