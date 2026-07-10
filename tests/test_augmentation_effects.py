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
    channel_std = s.data.std(dim=0, unbiased=False).clamp_min(1e-6)
    rel_noise = d.std(dim=0, unbiased=False) / channel_std
    assert torch.isfinite(out.data).all()
    assert 0.03 < rel_noise.mean().item() < 0.08  # ~sigma relative to each channel
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


def test_gravity_detection_rejects_normalized_recgym_text():
    triad = np.full((200, 3), 0.5, dtype=np.float32)
    descs = ["RecGym min-max normalized Accelerometer X-axis (wrist)",
             "RecGym min-max normalized Accelerometer Y-axis (wrist)",
             "RecGym min-max normalized Accelerometer Z-axis (wrist)"]
    assert not _gravity_present(triad, descs)


def test_rotation_3d_preserves_norm_and_rotates_jointly():
    """Full SO(3): acc + gyro triads rotate by ONE shared R per location, norm-preserving."""
    from datasets.imu_pretraining_dataset.augmentations import _random_so3
    s = make_sample(gravity=True)
    out = only("rotation_3d")(make_sample(gravity=True))
    # per-timestep triad norms preserved (rotation is orthogonal) for BOTH acc and gyro
    assert torch.allclose(s.data[:, :3].norm(dim=1), out.data[:, :3].norm(dim=1), atol=1e-3)
    assert torch.allclose(s.data[:, 3:].norm(dim=1), out.data[:, 3:].norm(dim=1), atol=1e-3)
    # the data actually changed (rotation applied)
    assert not torch.allclose(s.data, out.data, atol=1e-3)
    # acc and gyro share the SAME rotation: recover R from acc (Procrustes), apply to gyro
    A_ = s.data[:, :3].numpy(); B = out.data[:, :3].numpy()
    U, _, Vt = np.linalg.svd(A_.T @ B)
    R_acc = torch.tensor((U @ Vt).T, dtype=torch.float32)
    gyro_pred = torch.einsum("ij,tj->ti", R_acc, s.data[:, 3:])
    assert torch.allclose(gyro_pred, out.data[:, 3:], atol=1e-2)
    assert torch.isfinite(out.data).all()


def test_random_so3_is_proper_rotation():
    """_random_so3 returns proper rotations (orthogonal, det=+1) — no reflections."""
    from datasets.imu_pretraining_dataset.augmentations import _random_so3
    for _ in range(200):
        R = _random_so3()
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-4)
        assert abs(torch.det(R).item() - 1.0) < 1e-4


def test_rotation_3d_skips_gravity_removed():
    """require_gravity gate: gravity-removed / normalized acc is NOT rotated (avoids a
    meaningless SO(2/3) mixing of a signal with no physical 'down')."""
    # gravity=False -> acc mean ~0, no gravity DC -> _gravity_present False -> skipped
    s = make_sample(gravity=False)
    out = only("rotation_3d")(make_sample(gravity=False))
    assert torch.allclose(s.data, out.data, atol=1e-6)


def test_default_v2_uses_full_rotation():
    """default_v2 enables full SO(3) rotation (rotation_3d)."""
    cfg = AugmentationConfig.default_v2()
    assert cfg.rotation_3d.enabled


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


def test_channel_phrase_does_not_invent_linear_acceleration():
    desc = "accelerometer x-axis, left wrist, includes gravity"
    for _ in range(50):
        p = _paraphrase_channel(desc).lower()
        assert "linear acceleration" not in p


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
