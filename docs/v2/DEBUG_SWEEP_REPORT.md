# HALO Debug Sweep Report

**Date:** 2026-07-03
**Scope:** Comprehensive repo audit covering data preprocessing, dataset setup, training harness, model implementation, baselines, hyperparameters, augmentations, loss, checkpointing, and evaluation fairness.
**Method:** Static analysis (the shell environment is currently unresponsive to `pytest`/`python -m` invocations, so confirmations are based on careful code reading and cross-referencing of converters, configs, label files, and output JSONs).

---

## TL;DR — Issues by Severity

| # | Sev | Component | Issue |
|---|-----|-----------|-------|
| 1 | **CRITICAL** | eval labels | `benchmark_data/eval_v2/labels/mobiact.json` still lists `fall_backward_knees`; converter + `dataset_config.json` use `fall_forward_knees`. Ground truth for FKL falls is semantically wrong during eval. |
| 2 | **HIGH** | data | PAMAP2 `ori_1..ori_4` channels are emitted and described as valid in manifests despite the PAMAP2 paper (Reiss thesis Appendix B, Table B.2) stating they were "turned off/invalid". They are fed to the model as real sensor data. |
| 3 | **HIGH** | eval driver | `run_baselines_v2.py` hardcodes a 6-dataset list missing `opportunity`, so the default baseline run skips Opportunity even though `opportunity.json` exists and results appear in `RESULTS_V2.md`. |
| 4 | **HIGH** | eval label gen | `generate_eval_v2_labels.py` hardcodes `EVALUATED_DATASETS` missing `opportunity`, so even regenerating labels won't produce/update `opportunity.json` from metadata. |
| 5 | **MEDIUM** | baseline data | LiMU-BERT/CrossHAR adapters only have acc channels for `realworld` (no gyro). The LIMU-BERT pretraining corpus never included realworld gyro, so the model is silently evaluated on a channel subset it never trained on. (Footnoted in CROSSCHECK but never resolved.) |
| 6 | **MEDIUM** | few-shot | Few-shot early-stopping monitors validation **accuracy**, but the headline metric is **macro-F1**. On imbalanced datasets this can select a checkpoint that is sub-optimal for the reported number. |
| 7 | **MEDIUM** | doc/code drift | `README.md` states 4 encoder layers and 10 training datasets; the actual `small_deep` config has 8 temporal layers and the training `DATASETS` list has 11 (incl. `capture24`). README still references deprecated `evaluate_tsfm.py` (v1). |
| 8 | **LOW** | output JSON | `evaluate_tsfm_v2.py` writes `"primary_metric": "f1_macro over GT-present classes"` but the actual macro-F1 is computed over GT∪predicted classes (per protocol). The comment is stale; the metric is correct. |
| 9 | **LOW** | baseline ckpt path | `evaluate_limubert.py` loads `pretrain_base_recgym_20_120/pretrained_combined.pt`; `BASELINES_SETUP.md` calls it `pretrain_base_combined_train_20_120`. They are byte-identical (per CROSSCHECK2), so functionally fine, but the path name is misleading. |
| 10 | **INFO** | papers folders | New `papers/` folders for baselines and datasets exist and are well-organized; no broken references found. |

---

## 1. Data Preprocessing & Dataset Setup

### 1.1 Converters (good, with one residual)

The cross-check fixes in `CROSSCHECK.md` were all verified in code:

- **PAMAP2** (`datascripts/pamap2/convert.py`): column interleave fix is present — orientation channels are correctly placed within each IMU block. **But** see issue #2: the orientation channels themselves are still emitted as valid.
- **HARTH** (`datascripts/harth/convert.py`): activity code mapping (1–8, 13, 14, 130, 140) and the merge of inactive-cycling variants into `cycling` are correct. `eval_v2/labels/harth.json` was correctly regenerated to 10 classes. Unit test `tests/test_eval_v2.py` includes a HARTH-bug regression test.
- **MobiAct** (`datascripts/mobiact/convert.py`): FKL → `fall_forward_knees` is correct in the converter and in `benchmark_data/dataset_config.json`. **But** see issue #1: the eval label file is stale.
- **Capture24** (`datascripts/capture24/convert.py`): newly added training dataset; conversion is clean and consistent with the `dataset_config.json` placement (`dominant_wrist`).
- **LiMU-BERT units** (`benchmark_data/scripts/preprocess_limubert.py`): milli-g→m/s² for Opportunity and g→m/s² for HARTH are correctly applied.

### 1.2 Channel descriptions & placements

`benchmark_data/dataset_config.json` is the central source of truth and is internally consistent with the converters for: mobiact (`trouser_pocket`), hhar (`waist`), dsads (`torso`), kuhar (`waist`), recgym (`wrist`), inclusivehar (`waist`). Channel-text generation in `eval_common.py:get_dataset_metadata` consumes this correctly.

### 1.3 Issue #2 — PAMAP2 invalid orientation channels (HIGH)

`datascripts/pamap2/convert.py` still emits `ori_1..ori_4`, and `data/pamap2/manifest.json` describes them as legitimate quaternion components. `multi_dataset_loader.py` includes `'ori'` in `IMU_PATTERNS`, so these channels flow into the model as if they were real sensor data. The PAMAP2 paper explicitly states these were "turned off" during collection. This was flagged HIGH in `CROSSCHECK.md` and is **not** resolved. Recommendation: drop `ori_*` channels at conversion time (or mark them as a separate non-IMU modality that the loader filters out).

---

## 2. Dataset Classes & Multi-Dataset Loader

`datasets/imu_pretraining_dataset/multi_dataset_loader.py` is well-structured:

- **Channel grouping** (`group_channels_by_sensor`) correctly handles body-location prefixes (`chest_acc_*`), triads, quaternions, and single channels. Covered by `tests/test_data_loading.py`.
- **ChannelBucketBatchSampler** groups by channel count for efficient batching — sound design for heterogeneous datasets.
- **Caching** (`session caching` via hash of config) avoids reprocessing; cache invalidation is config-keyed.
- **`__getitem__`** correctly applies physics-changing augmentations (gravity add/remove, yaw rotation, rate resampling) and propagates the updated channel text / sampling rate into the `IMUSample` metadata, which the collate fn then exposes to the model's channel-text encoder. This matches the paper.
- **Label augmentation** (synonym rewriting via `label_groups`) is wired correctly.

No issues found in the loader itself. The PAMAP2 `ori` issue above is the only data-side concern that propagates through here.

---

## 3. HALO Model Implementation

### 3.1 Encoder (`model/encoder.py`, `model/transformer.py`)

- `IMUActivityRecognitionEncoder`: preprocessing (per-channel standardization), feature extraction (conv patch embed), sinusoidal positional encoding, and transformer blocks all match the README/paper.
- `DualBranchTransformer` with `CrossChannelSelfAttention` is the dual-branch temporal+cross-channel attention described in the paper. Masking for padded channels and patches is handled correctly (`channel_mask`, `patch_padding_mask`).
- `use_cross_channel` flag defaults to `False` for backward compatibility but is enabled in the `small_deep` config used by the headline checkpoint.

### 3.2 Semantic Alignment Head (`model/semantic_alignment.py`)

- `CrossChannelFusion` (multi-query cross-attention, channel→patch) and `TemporalAttention`/`MultiQueryPooling` (patch→single embedding) are consistent with the design doc.
- `per_patch_prediction` flag (used by `small_deep`) switches between pooled and per-patch outputs — this is the dense-prediction variant hinted at in `design_objective.md`.
- `ProjectionHead` and the learnable `LabelBank` (SentenceBERT-init + attention pooling) are correctly implemented.
- Soft-target text embeddings (`frozen_text_embeddings`) flow into the loss correctly.

### 3.3 Config (`model/config.py`)

Five tiers (Tiny/Small/Small-Deep/Medium/Large). The headline model is `small_deep`: d_model=192, 8 temporal layers, cross-channel attn on, per-patch prediction on, label bank dim 1024. Internally consistent.

**Minor:** README says "4 encoder layers" — stale; `small_deep` uses 8.

---

## 4. Training Harness, Loss, Objective, Augmentations

### 4.1 `semantic_alignment_train.py`

- Hyperparameters (temperature 0.07, soft-target lambda, memory bank 8192, grad-caching, channel-bucket sampler, group-balanced sampling with `MAX_OVERSAMPLE_RATIO` and `SAMPLING_TEMPERATURE`) are consistent with the protocol doc.
- `DATASETS` list has **11** entries (incl. `capture24`); README says 10. Update README.
- Checkpoint save (`torch.save` of state_dict + config + label_bank) and the legacy-prefix stripping (`_orig_mod.`, `module.`) on load are correct and robust to DDP/compile wrappers. `model_loading.py` additionally converts legacy gate weights. Covered by `tests/test_model_loading.py`.

### 4.2 Loss (`semantic_loss.py`)

- `InfoNCELoss` supports both hard and soft targets. Soft targets are built from `frozen_text_embeddings @ frozen_text_embeddings.T` and capped/normalized correctly so synonyms are not treated as hard negatives.
- Memory bank negatives are appended to the logits along the right dim and masked properly. Covered by `tests/test_similarity_computation.py` and `tests/test_memory_bank.py`.

### 4.3 Augmentations (`augmentations.py`)

- `AugmentationConfig` dataclass exposes jitter, scale, time-shift, time-warp, magnitude-warp, gravity add/remove, yaw-only rotation, anti-aliased rate resampling, and channel dropout — all physically plausible.
- Critical detail: physics-changing augs (gravity/yaw/rate) **update the channel text and sampling rate** on the `IMUSample`, so the model sees consistent (data, description) pairs. This is the correct design and is exercised in `tests/test_augmentations.py`.
- Application order is deterministic and reasonable (value-space augs first, then physics augs, then channel dropout).

No issues found.

---

## 5. Baselines

### 5.1 Adapter structure (`baselines/base.py`, `limubert.py`, `crosshar.py`)

- Clean two-tier design: `ConSEAdapter` (closed-vocab → semantic embeddings via ConSE) and `CosineAdapter` (open-vocab, direct cosine).
- Shared `load_gt` and `score` utilities ensure baselines and HALO are scored by the identical `eval_v2.classification_metrics` path — this is the right fairness guarantee.
- LiMU-BERT and CrossHAR adapters both rely on `data_20_120.npy` / `label_20_120.npy` artifacts produced by `preprocess_limubert.py`; the unit-conversion fixes there (milli-g for opportunity, g for harth) are present.

### 5.2 Issue #5 — realworld gyro (MEDIUM)

`benchmark_data/raw/realworld/metadata.json` lists only `acc_x/y/z` (no gyro). The LIMU-BERT pretraining corpus (recgym+… combined) did not include realworld gyro, so evaluating LiMU-BERT/CrossHAR on realworld with acc-only is the only fair option — but this is a silent channel-count mismatch with HALO, which sees gyro when available. The protocol footnotes this but it is worth surfacing in the report as a known asymmetry. Recommendation: explicitly log the channel set used per (model, dataset) in the output JSON.

### 5.3 Issue #9 — checkpoint path name (LOW)

`evaluate_limubert.py:138` loads `pretrain_base_recgym_20_120/pretrained_combined.pt`; docs say `pretrain_base_combined_train_20_120`. Byte-identical per CROSSCHECK2, so no functional impact, but rename for clarity.

### 5.4 Dropped baselines

`RESULTS_V2.md` documents dropping MOMENT, LLaSA, and LanHAR from the main table for redundancy/deployability reasons (not weakness). The adapter code for these may still exist but is not invoked by `run_baselines_v2.py`. This is a defensible editorial choice and is transparently documented.

---

## 6. Evaluation & Metrics

### 6.1 Protocol v2 (`eval_v2.py`)

- `window_ground_truth`: majority vote over raw native codes with **no offset arithmetic** — the v1 HARTH bug is gone. Regression-tested.
- `subject_disjoint_split`: train/val/test subjects are disjoint (few-shot). Correct.
- `balanced_subsample_indices`: water-filling subject-balanced subsampling for FS-1%/10%. Correct and deterministic given a seed.
- `soft_pool_patch_scores` + `segment_predictions`: soft-logit pooling across patches in a segment, then argmax. Reasonable.
- `classification_metrics`: macro-F1 over **GT∪predicted** classes (correct, matches protocol); balanced accuracy over GT-only. Subject-stratified bootstrap CI. Sound.
- `conse_embeddings` / `conse_predict`: ConSE bridge for closed-vocab baselines correctly composes top-k softmax-weighted label embeddings. Fairness-equivalent to open-vocab cosine for HALO.

### 6.2 Issue #6 — few-shot early-stopping on accuracy (MEDIUM)

`evaluate_tsfm_v2.py:246-250` monitors `val_acc` for early stopping, while the headline metric is `f1_macro`. On imbalanced few-shot splits, the accuracy-best checkpoint can differ from the F1-best checkpoint. Recommendation: monitor `f1_macro` on the val split instead, or report both and select on F1.

### 6.3 Issue #8 — stale comment in output JSON (LOW)

`evaluate_tsfm_v2.py:321` writes `"primary_metric": "f1_macro over GT-present classes"` but the implemented metric is over GT∪predicted. Update the string to `"f1_macro over GT∪predicted classes"`.

### 6.4 Fairness summary

The v2 protocol is, on the whole, **fair** across model types:
- Identical ground-truth derivation, subject-disjoint splits, and metrics for all models.
- ConSE bridge gives closed-vocab baselines a semantic-embedding pathway equivalent to HALO's open-vocab cosine.
- The only asymmetries are unavoidable (realworld gyro absence for LIMU-BERT-family) and documented.
- The one **unfair** aspect is issue #1: MobiAct FKL ground truth is semantically wrong, which harms any model that actually predicts the (correct) forward fall — but this is a bug, not a design asymmetry.

---

## 7. Issue #1 — MobiAct FKL Label Inconsistency (CRITICAL) — Detail

This is the single most important finding.

**Pipeline trace:**
1. `datascripts/mobiact/convert.py` maps raw `FKL` → `"fall_forward_knees"` ✓
2. `benchmark_data/dataset_config.json` lists `"fall_forward_knees"` ✓
3. `code/data/mobiact/labels.json` contains `"fall_forward_knees"` ✓
4. `benchmark_data/scripts/preprocess_tsfm_eval.py` builds `activity_to_idx` from `metadata["activities"]` (sourced from `dataset_config.json`), so `label_native.npy` encodes FKL windows with the index whose label is `"fall_forward_knees"` ✓
5. `benchmark_data/eval_v2/labels/mobiact.json` (frozen eval config) maps that same index → `"fall_backward_knees"` ✗ **STALE**
6. `eval_v2.py` loads `mobiact.json`'s `idx_to_label` to decode ground truth → FKL windows get the **wrong** label string.
7. The ConSE semantic pair in `mobiact.json` also links this index to `"falling_backward"` — wrong direction.

**Evidence in outputs:**
- `test_output/eval_v2/tsfm_v2_crosscheck_refresh.json` shows `fall_backward_knees: 0.0` in per-class F1 for mobiact — the model is being scored against a label string that no prediction matches (because the model's label bank contains `fall_forward_knees`-style descriptions, not `fall_backward_knees`).

**Root cause:** Either `generate_eval_v2_labels.py` was not re-run for mobiact after the converter fix, or `tsfm_eval/mobiact/metadata.json` (the source for generation) was not regenerated from the fixed `dataset_config.json`.

**Fix:**
1. Regenerate `tsfm_eval/mobiact/metadata.json` from the (correct) `data/mobiact/` + `dataset_config.json`.
2. Add `"opportunity"` to `generate_eval_v2_labels.py:EVALUATED_DATASETS`.
3. Re-run `generate_eval_v2_labels.py` to regenerate all `eval_v2/labels/*.json`.
4. Re-run `evaluate_tsfm_v2.py` and `run_baselines_v2.py` for mobiact (and opportunity for baselines).
5. Update `RESULTS_V2.md` with corrected mobiact numbers.

This directly contradicts the "Resolved" claim for mobiact in `CROSSCHECK.md`. The converter was fixed; the eval artifact was not.

---

## 8. Issue #3/#4 — Opportunity Missing from Baseline Driver & Label Generator

- `run_baselines_v2.py` `DATASETS = [...]` is hardcoded to 6 datasets, missing `opportunity`. So `python run_baselines_v2.py` skips Opportunity baselines by default, even though `baseline_v2_limubert.json` shows Opportunity was evaluated (presumably via a manual override). This is a regression in the default execution path.
- `generate_eval_v2_labels.py:EVALUATED_DATASETS` is similarly hardcoded to 6, missing `opportunity`. So regenerating labels won't touch `opportunity.json` (which currently exists, presumably hand-created or from an older script version).

**Fix:** Add `"opportunity"` to both lists, or better, derive the list dynamically from `benchmark_data/eval_v2/labels/*.json` (matching how `evaluate_tsfm_v2.py` discovers datasets).

---

## 9. Recommendations (Ordered)

1. **Regenerate mobiact eval labels** and re-run mobiact evaluation (CRITICAL).
2. **Drop PAMAP2 `ori_*` channels** at conversion, or filter them in the loader (HIGH).
3. **Add `opportunity`** to `run_baselines_v2.py` and `generate_eval_v2_labels.py` dataset lists, or make them dynamic (HIGH).
4. **Switch few-shot early-stopping** to monitor `f1_macro` instead of `val_acc` (MEDIUM).
5. **Log channel set per (model, dataset)** in output JSON to make the realworld-gyro asymmetry explicit (MEDIUM).
6. **Update `README.md`** to reflect 11 training datasets, 8 temporal layers in `small_deep`, and the v2 evaluation scripts (MEDIUM).
7. **Fix the stale `primary_metric` string** in `evaluate_tsfm_v2.py` output JSON (LOW).
8. **Rename LIMU-BERT checkpoint path** or update `BASELINES_SETUP.md` to match (LOW).

---

## 10. What's Solid

To end on the constructive side, the following are well-implemented and need no changes:

- Dual-branch transformer + semantic alignment head architecture.
- Soft-target InfoNCE with memory bank and grad-caching.
- Physics-aware augmentations that update channel text/rate consistently.
- Multi-dataset loader with channel-bucket batching and session caching.
- Checkpoint save/load with legacy-prefix and legacy-gate conversion.
- v2 evaluation core: subject-disjoint splits, water-filling balanced subsampling, macro-F1 over GT∪predicted, subject-stratified bootstrap CIs.
- ConSE bridge for closed-vocabulary baselines (genuinely fair comparison mechanism).
- Unit test coverage for eval core, augmentations, data loading, model loading, label groups, memory bank, similarity, and encoder forward.
- New `papers/` folders for baselines and datasets are well-organized with no broken references.

---

*Report generated from static analysis due to an unresponsive shell environment. Once `pytest`/`python -m` execution is restored, recommended verification steps: run `pytest tests/ -q`, regenerate mobiact eval labels, and re-run the mobiact evaluation to confirm the FKL fix.*
