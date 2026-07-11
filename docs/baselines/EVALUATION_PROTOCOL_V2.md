# Evaluation Protocol v2

**Status:** active protocol, fairness and baseline disclosures refreshed
2026-07-11. Supersedes `EVALUATION_PROTOCOL.md` (v1) for all new numbers.
Design rationale: `docs/v2/design_evaluation.md`. Implementation:
`val_scripts/human_activity_recognition/eval_v2.py` (+ unit tests in
`tests/test_eval_v2.py`), orchestrated by `evaluate_tsfm_v2.py` and
`run_baselines_v2.py`. Per-model input contracts and deviations are canonical in
[`BASELINE_IMPLEMENTATION_NOTES.md`](BASELINE_IMPLEMENTATION_NOTES.md).

## Why v1 was replaced

| v1 problem | v2 fix |
|---|---|
| "Open-set" scored 87 training strings through a hand-authored synonym ontology (`label_groups.py`) whose group boundaries encoded held-out test labels | Zero-shot is scored against the **target dataset's own pre-registered label strings** — no ontology anywhere in the scoring path |
| Random window splits for few-shot -> subject leakage | **Subject-disjoint** train/val/test via the subject index in `label_native.npy[:, 0, 1]` |
| Accuracy headline despite heavy class imbalance (acc 42 vs macro-F1 21) | **Macro-F1 primary** (+ balanced accuracy), consistent with unified zero-shot protocols that emphasize class-balanced evaluation and prohibit test-class tuning [1] |
| `get_window_labels` min-subtraction caused the HARTH label-offset bug | GT maps raw codes through the dataset's own `activity_to_idx` — **no offset arithmetic exists** (regression-tested) |
| Native-rate + rich channel text confounded with architecture | Explicit **parity row** (anti-aliased 20 Hz + neutral text for all models) + capability-Δ rows |
| Hard per-patch majority vote (fragmentation, first-index tie bias) | **Soft voting** (per-patch softmax scores summed across patches; τ = training temperature 0.07); hard vote kept as diagnostic |

## The protocol

### Pre-registered test set (decided 2026-07-02)

One flat tier of **6 test datasets**: motionsense, realworld, mobiact, shoaib,
**harth**, **inclusivehar**. There is no "severe-OOD" category. HARTH (back-mounted
accelerometer, genuine sensor/placement shift) is a regular test dataset,
scored by the same rules as every other. InclusiveHAR is the active
ability-diverse phone test set. **VTT-ConIoT is dropped from the
benchmark**: ~50% of its construction-domain labels have no training
equivalent, so zero-shot scores there measured label coverage rather than
recognition capability. Opportunity is retained as appendix-only because its
4-subject structure gives degenerate CIs. The evaluated set is defined by
`zero_shot_datasets` in `benchmark_data/dataset_config.json` and consumed by
`benchmark_data/scripts/generate_eval_v2_labels.py`.

### Settings

1. **ZS-XD (primary).** Zero-shot cross-dataset classification. For target dataset D
   with pre-registered vocabulary `L_D` (`benchmark_data/eval_v2/labels/{D}.json`):
   `prediction = argmax over c ∈ L_D of cos(f(x), g(c))`. Exact string match.
   No target data is ever used for training, so no split is needed.
2. **FS-1% / FS-10%.** End-to-end fine-tuning on 1%/10% of the training-subject
   windows (class-balanced), with **subject-disjoint** 80/10/10 subject splits
   (at least 3 subjects required, assertion-enforced). Final paper numbers require
   at least five registered seeds or rotated subject-group folds; a single split
   is an exploratory result, especially when it leaves one test subject.
3. **Streaming metrics** (frame-F1, boundary tolerance) join in milestone M3 under
   the same scoring rule.

### Metrics

- **Primary: macro-F1** averaged over `GT-classes ∪ predicted-classes` (sklearn's
  default). This charges false positives a model routes into a candidate class
  that has zero test windows (e.g. HALO predicting HARTH's `cycling_sit`), while
  not injecting automatic F1=0 for never-relevant classes (which averaging over
  the full L_D would do). GT-only averaging would let those FPs escape
  unpenalized.
- **Balanced accuracy** = macro recall over **GT classes only** (recall is
  undefined for a class with no true samples).
- Secondary: plain accuracy, weighted F1, per-class F1.
- **Cross-model comparability caveat:** because the macro-F1 denominator is
  `GT ∪ (that model's own predictions)`, two models on the same dataset can be
  averaged over different-sized class sets — a model that scatters false positives
  into extra zero-window candidate classes is averaged over a larger denominator.
  Read the per-class / per-dataset cells alongside the macro-F1 headline; do not
  over-interpret small aggregate gaps.
- **Uncertainty: subject-stratified bootstrap** (B=1000, seed 3431): resample
  subjects with replacement, not windows — windows within a subject are
  correlated. The scoring class set is FROZEN once on the full sample and reused
  for every replicate (re-deriving it per replicate silently changes the
  estimand and de-brackets the interval). With < 2 subjects the CI is reported
  as NaN with `ci_degenerate: true` — never a fake zero-width 95% interval.
- **Result-schema caveat:** some existing HALO JSON metadata says "GT-present
  classes," but the implementation correctly uses `GT union predicted` as
  described above. Final artifacts and paper prose must use the implementation's
  definition verbatim.

### Scoring closed-vocabulary baselines (CrossHAR, LiMU-BERT, SSL-Wearables)

Adopted from the literature (see `docs/v2/design_evaluation.md`, addendum rev. 2):

1. **Capability-scoped tables.** The main ZS-XD table contains models that can
   classify against an arbitrary label list. Closed-vocab baselines compete
   handicap-free in FS-1%/10%.
2. **ConSE bridge** (Norouzi et al., 2014 [2]) for dagger-marked ZS rows: the classifier's
   full softmax over its training vocabulary forms a convex combination of
   frozen-SBERT label embeddings (top-T=10), scored against `L_D`
   (`eval_v2.conse_predict`). Same frozen encoder (all-MiniLM-L6-v2, mean-pool)
   for every bridged model.
3. **Common-classes table** (appendix): activities where the baseline's training
   vocabulary and `L_D` match 1:1 — exact-string matches are computed
   automatically; semantic pairs must be promoted from
   `proposed_semantic_pairs` (PENDING_REVIEW) in the label configs by a human
   before this table is reported.
4. **Reachability** — reported per (baseline, dataset) two ways: `reachability_lb`
   (the fraction of `L_D` to which some single training label maps nearest — a
   T=1 **lower bound**, since top-T convex combinations can also land elsewhere)
   and `predicted_classes` (the target classes the bridge actually hit on this
   data). Classes outside the reachable set are effectively structural zeros: a
   capability statement, not a scoring artifact.
5. **Frozen-head identity.** Every ConSE row must identify the backbone variant,
   source vocabulary, source datasets, frozen/trainable modules, and head
   architecture. The current SSL row is specifically `harnet5 frozen-head ConSE`,
   not the paper's stronger ten-second full-fine-tune protocol [4].
6. **Source-only calibration.** ConSE mixture weights depend on softmax scale.
   Fit one scalar temperature per head on held-out source subjects using NLL,
   save it with the head, and apply it before top-T selection. Temperature
   scaling is a standard one-parameter post-hoc calibration method [3]. No target
   labels or target metric may select temperature or top-T.
7. **Vocabulary support.** Report how many of the 86 source classes have positive
   training examples for that head. A logit with no positive examples is not
   evidence that the model learned that source activity.

### Heterogeneity policy — which axes we canonicalize vs preserve

HALO's central claim is robustness to *heterogeneous* IMU data. Every axis of
variation we flatten before the models see the data is an axis we forfeit
claiming. So the data contract canonicalizes exactly **one** axis and preserves
the rest; each preserved axis is then handled by every model through its own
documented native path (see Fairness rows below). This section justifies the
split so it is a principled decision, not an implementation accident.

**Fairness test (applied per axis).** Canonicalize an axis only if it passes all
three; otherwise preserve it.

- **(A) Universality** — does every model, deployed for real, already know or
  require this? (i.e. it is device metadata every pipeline already has)
- **(B) Claimed capability** — is "handling variation on this axis" part of what
  HALO claims to do better than baselines?
- **(C) No fabrication** — can the axis be normalized *without* inventing content
  the sensor never observed?

**Canonicalize iff (A) ∧ ¬(B) ∧ (C).** Preserve if it is part of the claim (B);
and never fabricate to normalize (¬C forbids it regardless of A/B).

| Axis | Decision | (A) universal | (B) our claim | (C) no fabrication | Rationale |
|---|---|:--:|:--:|:--:|---|
| **Unit convention** (g vs m/s²) | **Canonicalize → g** | ✓ | ✗ | ✓ | Degenerate axis: two conventions related by the known constant 9.8. "Unit-invariance" is not a capability worth claiming. Additionally *required* for HALO's signed DC/gravity feature to be physically consistent (a DC of 1.0 means gravity only in g). Lossless global rescale, neutral to baselines (they per-window normalize or expect a fixed convention). |
| **Sampling rate** (≈20–100 Hz) | **Preserve (native)** | ✓ | **✓** | — | Rate-invariance (physical-Hz filterbank) is a core contribution; a fixed-rate corpus would erase the capability. Each model ingests natively: HALO physical-Hz; fixed-rate baselines anti-aliased-resample to their own trained rate. |
| **Channel count / modality** (acc-only, +gyro, +mag) | **Preserve (native)** | — | **✓** | partial | Channel-independence is a contribution, so the corpus keeps real channels per dataset. How a *rigid* baseline copes is graded by architecture (see "Channel handling for fixed-width baselines" below): a channel-independent model that pools across channels (NormWear) must get only its **real** channels; a fixed-width conv model (CrossHAR, LiMU-BERT, DeepConvLSTM) may zero-fill an absent sensor **only as its own documented handling**, disclosed as an eval-time input adaptation. A fabricated channel is never scored as if it were observed. |
| **Placement** (wrist/waist/pocket/…) | **Preserve** | — | **✓** | — | Placement is signal content, not a convention; kept as-is and described in channel text. Cross-placement generalization is part of the claim. |
| **Gravity presence** (total vs linear accel) | **Preserve; exclude where required** | — | partial | **✗** | A physical *content* difference, not a convention — removed gravity cannot be re-injected. Gravity-removed data (kuhar) is excluded from gravity-dependent models (SSL-Wearables), never fabricated back, and disclosed. |
| **Window / session length** | **Preserve (native)** | — | **✓** | — | Each model uses its own windowing; only the **canonical GT label per evaluated window** is shared (see Ground-truth handling). Variable context is part of the claim. |

**Rule of thumb.** The safe-to-canonicalize set is deliberately narrow — only an
axis that is simultaneously trivially/universally known, *not* part of the claim,
and normalizable without fabrication. Units are the only IMU axis that qualifies
(and are additionally *required* by the DC feature). Everything the paper claims
to handle stays heterogeneous, adapted each model's native way; anything that
would require fabricating unobserved content is never normalized away.

**Current implementation gaps (must close before the data freeze).**

- *Units are not yet canonicalized.* Physical units currently appear verbatim in
  HALO's channel text (e.g. PAMAP2 `"…values in m/s^2 including gravity"`,
  capture24 `"…in g"`, from each `data/<ds>/manifest.json`) **and** the signal
  scale is inconsistent — LiMU-BERT divides near-g UCI/HAPT/UniMiB by 9.8
  (`BASELINE_TRAINING_READINESS.md` §1.2). Fix: declare the corpus convention (g),
  convert m/s²→g at admission, drop the unit token from channel text, and have each
  baseline adapter apply its documented *from-g* transform.
- *Channels are zero-padded across the board.* The shared baseline tensor
  `benchmark_data/processed/limubert/<ds>/data_20_120.npy` is `(N,120,6)` with
  datasets that lack a real gyro (realworld, harth) **zero-padded to 6**. This is
  harmful for the channel-independent NormWear (a fabricated channel enters its
  cross-channel pool as a real observation) and merely undisclosed for the
  fixed-width conv models. Fix: feed NormWear its real channels only; for the
  fixed-width models, keep zero-fill only if it matches their documented handling
  and disclose it as an input adaptation.

These are tracked in `BASELINE_TRAINING_READINESS.md` (§1.2) and must be resolved
before the data contract is frozen and hashed.

**Channel handling for fixed-width baselines.** Channel-padding is an *eval-time
input adaptation* on frozen backbones, not a training choice — we do not retrain
the baseline encoders, so their expected channel set was fixed by their original
authors. What we control is how our benchmark data is fed in. The graded rule:

| Model | Channel handling | Rate | Fabrication risk |
|---|---|---|---|
| **HALO (ours)** | real channels per dataset, **no pad** (channel-independent + channel text) | native, no resample | none |
| CrossHAR / LiMU-BERT / DeepConvLSTM | fixed 6-ch conv; zero-fill absent sensor **iff documented**, disclosed | resample to their rate (native) | mild (conv absorbs a defined zero) |
| SSL-Wearables | 3 real acc channels, **no pad** | resample to 30 Hz (native) | none |
| NormWear | **must** receive only real channels (channel-independent pool) | 65 Hz native path | **severe if padded** — fix required |

Rate coercion (resampling each model to its own trained rate) is native and fair
and needs no change. Channel coercion is the axis that requires care, and only
NormWear currently needs an active fix; the fixed-width rows need disclosure.

### Fairness rows

Fairness does not mean forcing every architecture to consume byte-identical
tensors or assigning every model the same epoch count. It means freezing the
underlying examples and information budget, preserving each model's documented
input contract, and disclosing every remaining difference.

1. **Same underlying windows and canonical GT.** All models score the same target
   window indices, subjects, and one authoritative window label. Model-specific
   rate copies may not independently majority-vote a different label.
2. **Model-native transforms.** Given the preserved-heterogeneity axes
   (Heterogeneity policy above), rate conversion, gravity convention,
   normalization, real-channel selection, and temporal cropping are frozen per
   model before the run and recorded in the result artifact. Units are
   canonicalized to g at corpus admission; each model then applies its documented
   *from-g* transform. Phantom-channel padding is disallowed for channel-independent
   models that pool across channels (it injects a fabricated observation), and is
   permitted for fixed-width architectures only as that model's own documented
   handling of an absent sensor, disclosed as an input adaptation (see "Channel
   handling for fixed-width baselines"). Applying an incompatible common transform
   is not parity.
3. **Input-parity row.** HALO uses anti-aliased 20 Hz input plus neutral channel
   text; fixed-rate baselines keep their documented 20 Hz preprocessing. This
   controls nominal rate and channel-description information, not exact waveform,
   context length, parameter count, pretraining corpus, or optimization budget.
4. **Capability rows.** HALO native-rate/rich-text and any baseline native
   multi-placement or longer-context result are separate capability rows. For
   example, a five-placement UniMTS run cannot replace its one-core-stream parity
   row.
5. **Comparison categories.** Results tables visibly separate corpus-matched
   backbones (HALO, CrossHAR, LiMU-BERT), externally pretrained models
   (SSL-Wearables, UniMTS, NormWear), and supervised target-trained floors
   (DeepConvLSTM). The categories answer different scientific questions.
6. **Pretraining scale.** Report source datasets, effective windows/hours, and
   external corpora. SSL-Wearables uses roughly 700,000 person-days of
   UK-Biobank wrist acceleration [4], whereas HALO is configured for roughly 399
   hours. Dataset names alone are not a data-budget disclosure.
7. **Capacity.** Report adapted/trainable, sensor-side deployed, and total
   parameters including text towers. Input parity is not capacity parity. The
   measured counts and definitions are in `BASELINE_IMPLEMENTATION_NOTES.md`.
8. **Source balance.** The 86-label source head corpus (10 train datasets; recgym
   dropped 2026-07-11) is long-tailed and
   dataset-imbalanced. Use a pre-registered shared source sampler/loss for the
   fairness-matched run; retain a paper-faithful unweighted run under a separate
   name when the policies differ.
9. **Optimization budget.** Report optimizer steps, examples, effective seconds,
   selected epoch, and wall time. Equal epochs are not equal compute: LiMU-BERT
   creates six classifier examples from each parent window while CrossHAR creates
   one.
10. **Model selection and uncertainty.** Use held-out source subjects for
    zero-shot heads and the same validation metric for supervised target runs.
    Report all registered seeds/folds; do not select a seed or checkpoint using a
    target test score.

**Capability-delta rows:** native-rate, rich-channel-text, longer-context, and
multi-placement deltas are reported explicitly and separately from the parity
row.

### Pre-registered constants (do not tune post-hoc)

| Constant | Value |
|---|---|
| ConSE top-T | 10 |
| Bridge text encoder | all-MiniLM-L6-v2, mean-pool, labels de-underscored |
| ConSE temperature | One scalar per head, fitted by source-validation NLL only |
| Soft-pool temperature τ | 0.07 (training temperature) |
| Bootstrap | B=1000, subject-stratified, seed 3431 |
| Split seed | 3431 |
| Final few-shot uncertainty | At least 5 registered seeds or rotated subject folds |
| Supervised selection metric | Validation macro-F1 for every compared model |
| DeepConvLSTM input | 20 Hz, 120 timesteps, 6 channels; no dataset-specific tuning |
| Label strings | frozen in `benchmark_data/eval_v2/labels/*.json`; no rephrasing |

## Ground-truth handling (HARTH-bug hardening)

`label_native.npy` codes ARE `metadata.json:activity_to_idx` values (written by
`preprocess_tsfm_eval.py`). v2 majority-votes the raw codes and maps them through
`idx_to_label` — there is deliberately no offset arithmetic. HARTH's codes
{2..8, 11} (4 classes have zero windows) map correctly by construction;
`tests/test_eval_v2.py::TestWindowGroundTruth` is the regression test.
(The v1 evaluator's `get_window_labels` was also patched with the `+ t` restore.)

The rate-specific copies are not yet a perfect single source of truth: the
2026-07-11 audit found three RealWorld and one MobiAct windows whose majority
label differs between `label_native.npy` and `label_20_120.npy`, despite identical
window and subject indices (window/subject order aligns 100%: RealWorld
27138/27138, MobiAct 4345/4345).

**Root cause (established 2026-07-11).** Two independent GT pipelines vote
differently at ~50/50 activity-transition windows. `preprocess_tsfm_eval.py`
majority-votes over the full native window (300 samples @50 Hz); `preprocess_limubert.py`
first **nearest-neighbor decimates** the labels to 20 Hz
(`orig_indices = np.linspace(0, n_orig-1, n_target).astype(int)`) and then
majority-votes over 120 samples. On a window split near 50/50 across a boundary,
decimation shifts which class holds the majority, and `np.bincount(...).argmax()`
breaks any resulting tie toward the lowest index — so the two views disagree. It
is a boundary/tie artifact, not a semantic labeling error.

**Fixed 2026-07-11.** Every scorer now consumes the one canonical native-rate
label. HALO already scored on `label_native.npy`; `baselines/base.py::load_gt`
was switched from `label_20_120.npy` to that same native file, with a hard guard
that the native window grid is 1:1 with the 20 Hz model-output grid (same count,
`keep_idx`, and subjects) — a mismatch raises rather than silently mis-scoring.
The 4 transition windows now resolve to the undecimated native majority (e.g.
realworld win5857 → `lying`, mobiact win909 → `car_step_out`); `keep_idx` and
subjects were verified identical across all six test sets, and `tests/test_eval_v2.py`
(29) passes. Model-specific ground truth is thus eliminated.

## Result Provenance

A final result is complete only when its JSON records:

- repository SHA and dirty/clean state;
- data-bundle, label-config, model-checkpoint, and upstream-source hashes;
- comparison category and exact model/weight variant;
- model-native input contract and source inclusion/exclusion list;
- effective source windows/hours and all three parameter counts;
- loss, sampler, optimizer, batch size, optimizer steps, epoch selection,
  calibration temperature, split subjects, and seeds/folds;
- software/GPU versions, wall time, and an explicit success/failure status for
  every requested dataset.

Incremental or stale JSON is never accepted merely because a process exited zero.
The result must contain every pre-registered dataset or be labeled partial and
excluded from the aggregate comparison.

## Running

```bash
# Main ZS re-score (native rate, rich channel text); checkpoint must be the
# finalized current-filterbank run, not the historical spectral checkpoint.
TSFM_CHECKPOINT="$HALO_CHECKPOINT" \
  python val_scripts/human_activity_recognition/evaluate_tsfm_v2.py --zs-only

# Parity row
python val_scripts/human_activity_recognition/evaluate_tsfm_v2.py \
  --zs-only --channel-text neutral --eval-rate 20

# Full (adds subject-disjoint FS-1%/10% fine-tuning; hours)
python val_scripts/human_activity_recognition/evaluate_tsfm_v2.py
```

Outputs: `test_output/eval_v2/tsfm_v2_{channel_text}_{eval_rate}.json`.
The historical default `small_deep_v2_4b3fdd6` has a spectral-temporal/200-epoch
configuration and predates the current filterbank/100-epoch V2 code. It must not
be combined with newly trained baseline rows as if it were the current HALO run.

## Comparability warning

v2 numbers are **not comparable** to v1 numbers: the candidate sets, scoring rule,
metrics, and splits all changed. The v2 baseline table (this branch) is the
reference point for all V2-redesign experiments.

## References

1. Xian, Schiele, and Akata. ["Zero-Shot Learning - the Good, the Bad and the Ugly."](https://openaccess.thecvf.com/content_cvpr_2017/html/Xian_Zero-Shot_Learning_-_CVPR_2017_paper.html) CVPR, 2017.
2. Norouzi et al. ["Zero-Shot Learning by Convex Combination of Semantic Embeddings."](https://arxiv.org/abs/1312.5650) ICLR, 2014.
3. Guo et al. ["On Calibration of Modern Neural Networks."](https://proceedings.mlr.press/v70/guo17a.html) ICML, 2017.
4. Yuan et al. ["Self-supervised learning for human activity recognition using 700,000 person-days of wearable data."](https://www.nature.com/articles/s41746-024-01062-3) npj Digital Medicine 7:91, 2024. DOI: 10.1038/s41746-024-01062-3.
