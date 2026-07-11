# HALO Baselines — Overview

Concise catalog of every model HALO is compared against: how it works, why it's worth comparing,
its heterogeneity handling, open-set (zero-shot label) compatibility, size, and adapter tier.

**Readiness:** all six adapters exist, but none is approved for a final paper run.
See [`../v2/BASELINE_TRAINING_READINESS.md`](../v2/BASELINE_TRAINING_READINESS.md)
for the current go/no-go matrix.

- **Tier** = how the v2 harness scores it (`val_scripts/human_activity_recognition/run_baselines_v2.py`):
  - `conse` — closed-vocabulary classifier over the 94 global labels, bridged to each dataset's
    own vocabulary via ConSE (Norouzi 2014). Needs the bridge to do open-set.
  - `cosine` — text-aligned; window embedding scored by cosine similarity to label-text embeddings.
    Natively open-set.
  - `l1` — text-aligned but scored by Manhattan (L1) distance / argmin (bespoke; NormWear).
  - `fewshot` — supervised, trained from scratch on the target dataset's OWN labels
    (subject-disjoint few-shot; run via `run_fewshot_v2.py`). Not zero-shot.
- Param counts are the **total scored model** (`sum(numel)` over encoder **+** the head trained on
  our corpus), verified by loading each model — not encoder-only. The encoder/head split is given
  where it matters. This avoids overstating the HALO capacity gap: it ranges from ~6× (vs
  ssl-wearables) to ~360× (vs LiMU-BERT), not a uniform "~400×".
- Detailed, cited implementation contracts and caveats live in
  [`BASELINE_IMPLEMENTATION_NOTES.md`](BASELINE_IMPLEMENTATION_NOTES.md).

## Baselines

| Baseline | Tier | ~Params | Input | Open-set | One-line |
|---|---|--:|---|:--:|---|
| **CrossHAR** | conse | ~0.53 M (63K enc + 469K head) | 20 Hz, 6-ch acc+gyro | via ConSE | Masked-Transformer cross-dataset IMU SSL + fine-tuned classifier. |
| **LiMU-BERT** | conse | ~0.073 M (63K enc + 10K GRU head) | 20 Hz, 6-ch acc+gyro | via ConSE | Tiny BERT IMU encoder + GRU head; canonical lightweight SSL. |
| **ssl-wearables (harnet5)** | conse | ~4.54 M (4.23M trunk + 311K head) | 30 Hz, 3-ch acc, **gravity** | via ConSE | UK-Biobank ResNet accelerometer SSL; deployable wrist model. |
| **DeepConvLSTM** | fewshot | ~0.46 M | 20 Hz, 120 x 6 | no (supervised) | Classic 4-conv + 2-LSTM supervised HAR reference. |
| **UniMTS** | cosine | ~68.6 M (incl. CLIP text) | 20 Hz, 3-ch acc, SMPL joint | yes | ST-GCN skeleton encoder aligned to CLIP text; motion-TS zero-shot competitor. |
| **NormWear** | l1 | ~194 M + ~1.1 B text | intended: 65 Hz, 6 s, real channels | yes (L1) | Physiological-signal foundation model (PPG/ECG/EEG/GSR/IMU) + TinyLlama text. |
| **HALO (ours)** | cosine | ~26 M (small_deep) | rate-invariant, any-ch | ✓ | Language-aligned physical-filterbank per-patch encoder. |

## Why each is worth comparing

- **CrossHAR / LiMU-BERT** — the two dominant *IMU-native* self-supervised baselines; both are
  closed-vocabulary, so they isolate "how much does language-alignment (open-set) buy over a
  closed classifier + ConSE bridge?" Also the smallest models (fairness: HALO must beat tiny SSL).
- **ssl-wearables (harnet5)** — a *deployed, large-scale* accelerometer SSL model (UK Biobank).
  Tests HALO against an industrial wrist-only baseline. **Heterogeneity gotcha:** requires
  gravity-present g-unit accel, so it trains on only **8/11** datasets (excludes gravity-removed
  kuhar, normalized recgym, source-lost unimib_shar).
- **DeepConvLSTM** — the *supervised upper-reference* per dataset: what a from-scratch model gets
  with labels. Not open-set; frames the gap zero-shot must close.
- **UniMTS** — the *closest competitor*: also text-aligned and open-set, and explicitly designed
  for cross-position/orientation generalization (rotation-invariant aug, SMPL-joint graph). The
  key head-to-head for the open-set HAR claim. **Heterogeneity:** single IMU is placed at one SMPL
  joint (others zero-filled); accel-only.
- **NormWear** — a *cross-modal* foundation model (physiological signals) with text alignment.
  Tests whether a broad wearable model transfers to HAR. **Gotcha:** its Ricker-CWT scales and
  published preprocessing are tied to 65 Hz. Feeding the current 20 Hz tensors directly is an
  invalid adapter path, not a faithful limitation. Final input must be explicitly resampled and
  preprocessed at 65 Hz. Scored by L1 distance, not cosine.

## Heterogeneity & open-set summary

| Baseline | Rate handling | Channel handling | Gravity requirement | Open-set mechanism |
|---|---|---|---|---|
| CrossHAR | fixed 20 Hz (resampled) | fixed 6-ch | none (per-window norm) | ConSE bridge |
| LiMU-BERT | fixed 20 Hz | fixed 6-ch | none (÷9.8 norm) | ConSE bridge |
| ssl-wearables | fixed 30 Hz | acc-only 3-ch | **required** (g w/ gravity) | ConSE bridge |
| DeepConvLSTM | fixed 20 Hz | fixed 6-ch | none | n/a (supervised) |
| UniMTS | resample to 20 Hz, pad 200 | acc-only, joint-placed | dataset-specific m/s^2 contract | CLIP text cosine |
| NormWear | fixed 65 Hz | channel-independent, real channels only | detrend + smooth + amplitude norm | TinyLlama text, L1 |
| **HALO** | **rate-invariant** (physical-Hz filterbank) | **variable** (channel-independent + text) | signed DC/gravity feature | SBERT text cosine |

## Where configured
- Adapters: `val_scripts/human_activity_recognition/baselines/<name>.py` (+ `@register`).
- Recipes / internals: `val_scripts/human_activity_recognition/evaluate_<name>.py`.
- Global label vocab (ConSE tier): `benchmark_data/scripts/build_global_label_mapping.py` → 94 labels.
- Vendored repos + weights: `auxiliary_repos/<Name>/` (gitignored); see `cloud/recipes.json`.
- Results: `test_output/eval_v2/baseline_v2_<name>.json`; summary in `RESULTS_V2.md`.
