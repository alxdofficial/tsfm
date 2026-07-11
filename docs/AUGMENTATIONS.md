# HALO Augmentations — Catalog

Every augmentation available in the HALO training pipeline, its parameters, and which
preset enables it. Source of truth: `datasets/imu_pretraining_dataset/augmentations.py`
(`AugmentationConfig` + per-augmentation `*Cfg` dataclasses). Patch-size (temporal)
augmentation lives in the training script, not this module — see §4.

Presets (`AugmentationConfig` classmethods):
- `default_v2()` — the current training preset (P1–P4 physics curriculum + signal + text).
- `legacy()` — jitter + scale only (v1 behaviour).
- `none()` — everything off (used in tests / to isolate a single augmentation).

## 1. Signal augmentations (sensor-value level)

| Aug | Cfg | Default | in `default_v2` | Params | What it does |
|---|---|:--:|:--:|---|---|
| Jitter | `JitterCfg` | ON | ON | p=0.5, sigma=0.05 | Additive Gaussian noise per sample. |
| Scale | `ScaleCfg` | ON | ON | p=0.5, low=0.9, high=1.1 | Per-channel constant amplitude scaling. |
| Time shift | `TimeShiftCfg` | off | off | p=0.5, max_ratio=0.05 | Circular temporal shift. |
| Time warp | `TimeWarpCfg` | off | off | p=0.3, n_knots=4, strength=0.2 | Smooth nonlinear time-axis warp. |
| Magnitude warp | `MagnitudeWarpCfg` | off | off | p=0.3, n_knots=4, strength=0.3 | Smooth nonlinear amplitude warp. |

## 2. Physics augmentations (P1–P4 — heterogeneity curriculum, ON in `default_v2`)

| Aug | Cfg | Params | What it does + guardrail |
|---|---|---|---|
| **P1 Gravity** | `GravityCfg` | p=0.5, cutoff_hz=0.4, order=2 | Perturbs the quasi-DC gravity/orientation component (low-pass split at 0.4 Hz; human motion is >~0.5 Hz). Simulates mounting-orientation drift. |
| **P2 SO(3) rotation** | `Rotation3dCfg` | p=0.5, require_gravity=True | Haar-uniform 3-D rotation applied JOINTLY to acc+gyro triads (same R). `require_gravity=True` skips locations whose acc is gravity-removed/normalized (else the rotation is meaningless). |
| **P3 Rate resample** | `RateCfg` | p=0.5, min_hz=15, max_hz=100, min_samples=32 | Resamples the window to a different sampling rate (rate-invariance). Skips if the resampled window would be shorter than `min_samples`. |
| **P4 Channel dropout** | `ChannelDropoutCfg` | p=0.3, groups=("gyro",) | Drops whole channel groups (default: gyro) so the model tolerates variable channel availability across devices. |

## 3. Text augmentations (label + channel-description, ON in `default_v2`)

| Aug | Cfg | Params | What it does |
|---|---|---|---|
| Label text | `LabelTextCfg` | p=0.8, use_synonyms, use_templates | Rewrites the activity label with synonyms + sentence templates (text-side contrastive diversity). |
| Channel-desc paraphrase | `ChannelTextPhraseCfg` | (enabled in v2) | Paraphrases the per-channel natural-language descriptions. |
| Channel-desc dropout | `ChannelTextDropoutCfg` | (enabled in v2) | Randomly drops channel descriptions so the model isn't dependent on full metadata. |

## 4. Temporal / patch-size augmentation (training script, not this module)

Patch-size augmentation is configured in
`training_scripts/human_activity_recognition/semantic_alignment_train.py`
(`PATCH_SIZE_RANGE_PER_DATASET`): per step a patch size is sampled per dataset (3–4 discrete
options), then fixed at eval. Ranges are chosen so `max_patch < min_session` (≥1 valid patch
per session). See `docs/v2/data_quantity_report.md` §4 for the per-dataset ranges.

## 5. Notes

- Augmentations multiply *views*, not information — a fresh stochastic realization per window per
  epoch. Report real recording hours as the headline, augmented views as a secondary "views" figure
  (never as equivalent hours). See `data_quantity_report.md` §5/§8.
- `require_gravity` on P2 (and P1's usefulness) depends on physically-meaningful accel — the
  gravity-removed (kuhar) and normalized (recgym) datasets are correctly skipped/ineffective; see
  [`DATASOURCES.md`](DATASOURCES.md) §4.
- Channel-text augmentations must stay consistent with the actual channel selection after
  channel-dropout (P4) — enforced in the loader.
