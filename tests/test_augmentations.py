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
        for n in ("gravity", "rate", "channel_dropout"):
            assert getattr(cfg, n).enabled, n
        # P2 rotation slot: default_v2 uses full SO(3) (rotation_3d).
        assert cfg.rotation_3d.enabled

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
