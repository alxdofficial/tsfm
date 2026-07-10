"""
Measurement harness for the unified augmentation pipeline: every augmentation is run in
ISOLATION and its claimed effect is measured, so each is verified bug-free. Text
augmentations are additionally sampled many times and checked for semantic preservation.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.imu_pretraining_dataset.augmentations import (  # noqa: E402
    AugmentationConfig, IMUAugmenter, IMUSample, _paraphrase_channel, _gravity_present,
)

CH_NAMES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
CH_DESCS = [
    "accelerometer x-axis, left wrist", "accelerometer y-axis, left wrist",
    "accelerometer z-axis, left wrist", "gyroscope x-axis, left wrist",
    "gyroscope y-axis, left wrist", "gyroscope z-axis, left wrist",
]


def make_sample(T=200, sr=100.0, gravity=True, label="walking", dataset="uci_har"):
    t = np.arange(T) / sr
    ax = 0.3 * np.sin(2 * np.pi * 2 * t)
    ay = 0.2 * np.sin(2 * np.pi * 3 * t)
    az = (9.81 if gravity else 0.0) + 0.4 * np.sin(2 * np.pi * 2 * t)
    gx = 0.5 * np.sin(2 * np.pi * 1.5 * t)
    gy = 0.4 * np.cos(2 * np.pi * 1.5 * t)
    gz = 0.3 * np.sin(2 * np.pi * 1.0 * t)
    data = torch.tensor(np.stack([ax, ay, az, gx, gy, gz], axis=1), dtype=torch.float32)
    return IMUSample(data=data, channel_names=list(CH_NAMES), sampling_rate=sr,
                     channel_descriptions=list(CH_DESCS), label=label, dataset_name=dataset)


def only(aug_name, **overrides):
    """An augmenter with a single augmentation enabled at p=1.0 (isolation)."""
    cfg = AugmentationConfig.none()
    spec = getattr(cfg, aug_name)
    spec.enabled, spec.p = True, 1.0
    for k, v in overrides.items():
        setattr(spec, k, v)
    return IMUAugmenter(cfg)


# ----------------------------------------------------------------- signal / physics
def test_jitter_adds_bounded_noise():
    s = make_sample()
    out = only("jitter", sigma=0.05)(make_sample())
    d = (out.data - s.data)
    assert torch.isfinite(out.data).all()
    assert 0.03 < d.std().item() < 0.08          # ~sigma
    assert out.data.shape == s.data.shape


def test_scale_is_per_channel_constant():
    s = make_sample()                                 # deterministic -> same as augmenter input
    out = only("scale", low=0.5, high=1.5)(make_sample())
    # each channel scaled by a single constant: least-squares scale reconstructs the output
    # (robust to the sinusoid's zero-crossings, which a naive ratio blows up on).
    for c in range(s.data.shape[1]):
        i, o = s.data[:, c], out.data[:, c]
        scale = (o * i).sum() / (i * i).sum().clamp(min=1e-8)
        assert torch.allclose(o, i * scale, atol=1e-4), f"channel {c} is not a constant scale"
    assert torch.isfinite(out.data).all()


def test_gravity_removal_drops_dc_on_acc_only():
    s = make_sample(gravity=True)
    assert _gravity_present(s.data[:, :3].numpy())        # acc has gravity to start
    out = only("gravity")(make_sample(gravity=True))
    assert abs(out.data[:, 2].mean().item()) < 0.5        # acc_z DC removed (~9.81 -> ~0)
    assert not _gravity_present(out.data[:, :3].numpy())
    assert torch.allclose(out.data[:, 3:], s.data[:, 3:], atol=1e-4)  # gyro untouched
    assert any("gravity removed" in d.lower() for d in out.channel_descriptions[:3])


def test_yaw_rotation_preserves_gravity_and_norm():
    s = make_sample(gravity=True)
    out = only("yaw_rotation")(make_sample(gravity=True))
    # per-timestep acc-triad norm is preserved by a rotation
    n_in = s.data[:, :3].norm(dim=1)
    n_out = out.data[:, :3].norm(dim=1)
    assert torch.allclose(n_in, n_out, atol=1e-3)
    # "which way is down" (mean gravity magnitude) preserved
    assert abs(s.data[:, :3].mean(0).norm() - out.data[:, :3].mean(0).norm()) < 1e-2
    assert torch.isfinite(out.data).all()


def test_rate_changes_sampling_rate_no_nan():
    changed = 0
    for _ in range(20):
        s = make_sample(sr=100.0)
        out = only("rate", min_hz=20.0, max_hz=80.0)(make_sample(sr=100.0))
        assert torch.isfinite(out.data).all()
        if abs(out.sampling_rate - 100.0) > 1e-3:
            changed += 1
            assert 15 <= out.sampling_rate <= 100  # within configured-ish range
    assert changed >= 15                            # rate actually changes most of the time


def test_channel_dropout_drops_group_keeps_triad():
    out = only("channel_dropout", groups=("gyro",))(make_sample())
    assert out.data.shape[1] == 3                    # gyro triad dropped
    assert all("gyro" not in n for n in out.channel_names)
    assert len(out.channel_descriptions) == 3
    assert torch.isfinite(out.data).all()


# --------------------------------------------------------------------------- text
def _content_preserved(orig, para):
    """Placement keywords + axis letters must survive the paraphrase verbatim."""
    lo, lp = orig.lower(), para.lower()
    for kw in ["wrist", "waist", "pocket", "chest", "ankle", "torso", "arm", "hip", "thigh", "head"]:
        if kw in lo and kw not in lp:
            return False, f"placement '{kw}' lost: {para!r}"
    for ax in ["x", "y", "z"]:
        if re.search(rf"\b{ax}[- ]axis\b", orig, re.I) and not re.search(rf"\b{ax}[- ]?axis\b", para, re.I):
            return False, f"axis '{ax}' lost: {para!r}"
    if not para.strip():
        return False, "empty paraphrase"
    return True, ""


def test_channel_phrase_preserves_semantics_over_many_samples():
    descs = CH_DESCS + [
        "Acceleration x-axis (±16g scale) from wrist-mounted IMU",
        "Angular velocity y-axis from chest-mounted IMU",
        "Magnetic field z-axis from ankle-mounted IMU (gravity removed)",
        "gyroscope z-axis, front trouser pocket",
    ]
    changed = 0
    for d in descs:
        for _ in range(50):                          # sample many
            p = _paraphrase_channel(d)
            ok, why = _content_preserved(d, p)
            assert ok, why
            if p != d:
                changed += 1
    assert changed > 0                               # paraphrase actually varies the text


def test_channel_text_dropout_neutralizes_subset_keeps_signal():
    for _ in range(30):
        s = make_sample()
        out = only("channel_text_dropout", p=1.0, max_frac=0.5,
                   neutral="an inertial sensor channel")(make_sample())
        neutralized = [d == "an inertial sensor channel" for d in out.channel_descriptions]
        n_neu = sum(neutralized)
        assert 1 <= n_neu <= int(0.5 * len(CH_DESCS))    # some, never more than max_frac
        assert not all(neutralized)                       # never all
        assert torch.allclose(out.data, s.data)           # SIGNAL untouched
        assert out.data.shape[1] == len(out.channel_descriptions)


def test_label_text_is_nonempty_paraphrase():
    seen = set()
    for _ in range(50):
        out = only("label_text")(make_sample(label="walking", dataset="uci_har"))
        assert out.label_text and isinstance(out.label_text, str)
        seen.add(out.label_text)
    assert len(seen) > 3                              # produces varied paraphrases


def test_label_text_defaults_to_raw_without_aug():
    # IMUSample with no augmenter run -> label_text == raw label
    s = IMUSample(data=torch.zeros(10, 3), channel_names=["acc_x", "acc_y", "acc_z"],
                  sampling_rate=50.0, channel_descriptions=["a", "b", "c"], label="running")
    assert s.label_text == "running"


# ---------------------------------------------------------------- integration / stress
def test_default_v2_enables_text_augs():
    cfg = AugmentationConfig.default_v2()
    assert cfg.label_text.enabled and cfg.channel_text_phrase.enabled and cfg.channel_text_dropout.enabled
    # legacy preset leaves text augs OFF
    assert not AugmentationConfig.legacy().label_text.enabled


def test_full_pipeline_no_nan_and_consistent_shapes():
    aug = IMUAugmenter(AugmentationConfig.default_v2())
    for i in range(100):
        out = aug(make_sample(sr=float([20, 50, 100][i % 3])))
        assert torch.isfinite(out.data).all()
        C = out.data.shape[1]
        assert C == len(out.channel_names) == len(out.channel_descriptions)   # stays consistent
        assert out.label_text and out.sampling_rate > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
