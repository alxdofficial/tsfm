# Evaluation Protocol v2

**Status:** active on the V2 branch (2026-07-02). Supersedes `EVALUATION_PROTOCOL.md`
(v1) for all new numbers. Design rationale: `docs/v2/design_evaluation.md`.
Implementation: `val_scripts/human_activity_recognition/eval_v2.py` (+ unit tests in
`tests/test_eval_v2.py`), orchestrated by `evaluate_tsfm_v2.py`.

## Why v1 was replaced

| v1 problem | v2 fix |
|---|---|
| "Open-set" scored 87 training strings through a hand-authored synonym ontology (`label_groups.py`) whose group boundaries encoded held-out test labels | Zero-shot is scored against the **target dataset's own pre-registered label strings** — no ontology anywhere in the scoring path |
| Random window splits for few-shot → subject leakage (LOSO literature: 10–15 pp inflation) | **Subject-disjoint** train/val/test via the subject index in `label_native.npy[:, 0, 1]` |
| Accuracy headline despite heavy class imbalance (acc 42 vs macro-F1 21) | **Macro-F1 primary** (+ balanced accuracy), per the ZSL evaluation canon (Xian et al., TPAMI 2018) |
| `get_window_labels` min-subtraction caused the HARTH label-offset bug | GT maps raw codes through the dataset's own `activity_to_idx` — **no offset arithmetic exists** (regression-tested) |
| Native-rate + rich channel text confounded with architecture | Explicit **parity row** (anti-aliased 20 Hz + neutral text for all models) + capability-Δ rows |
| Hard per-patch majority vote (fragmentation, first-index tie bias) | **Soft logit-pooling** of per-patch scores (τ = training temperature 0.07); vote kept as diagnostic |

## The protocol

### Settings

1. **ZS-XD (primary).** Zero-shot cross-dataset classification. For target dataset D
   with pre-registered vocabulary `L_D` (`benchmark_data/eval_v2/labels/{D}.json`):
   `prediction = argmax over c ∈ L_D of cos(f(x), g(c))`. Exact string match.
   No target data is ever used for training, so no split is needed.
2. **FS-1% / FS-10%.** End-to-end fine-tuning on 1%/10% of the training-subject
   windows (class-balanced), with **subject-disjoint** 80/10/10 subject splits
   (LOSO-style; ≥3 subjects required, assertion-enforced).
3. **Streaming metrics** (frame-F1, boundary tolerance) join in milestone M3 under
   the same scoring rule.

### Metrics

- **Primary: macro-F1** over classes present in ground truth (classes with zero
  test windows — e.g. HARTH's cycling variants — are excluded, not free zeros).
- Secondary: balanced accuracy, plain accuracy, weighted F1, per-class F1.
- **Uncertainty: subject-stratified bootstrap** (B=1000, seed 3431): resample
  subjects with replacement, not windows — windows within a subject are correlated.

### Scoring closed-vocabulary baselines (LiMU-BERT, CrossHAR, MOMENT)

Adopted from the literature (see `docs/v2/design_evaluation.md`, addendum rev. 2):

1. **Capability-scoped tables.** The main ZS-XD table contains models that can
   classify against an arbitrary label list. Closed-vocab baselines compete
   handicap-free in FS-1%/10%.
2. **ConSE bridge** (Norouzi et al., 2014) for †-marked ZS rows: the classifier's
   full softmax over its training vocabulary forms a convex combination of
   frozen-SBERT label embeddings (top-T=10), scored against `L_D`
   (`eval_v2.conse_predict`). Same frozen encoder (all-MiniLM-L6-v2, mean-pool)
   for every bridged model. MOMENT's SVM uses `probability=True` (Platt).
3. **Common-classes table** (appendix): activities where the baseline's training
   vocabulary and `L_D` match 1:1 — exact-string matches are computed
   automatically; semantic pairs must be promoted from
   `proposed_semantic_pairs` (PENDING_REVIEW) in the label configs by a human
   before this table is reported.
4. **Reachability** — fraction of `L_D` reachable by the bridge — is reported per
   (baseline, dataset). Unreachable classes are structural zeros: a capability
   statement, not a scoring artifact.

### Fairness rows

- **Parity row:** every model receives the identical anti-aliased 20 Hz signal
  (`scipy.signal.resample_poly`) + neutral channel text
  (`evaluate_tsfm_v2.py --channel-text neutral --eval-rate 20`).
- **Capability-Δ rows:** native-rate and rich-channel-text deltas reported
  explicitly and separately.

### Pre-registered constants (do not tune post-hoc)

| Constant | Value |
|---|---|
| ConSE top-T | 10 |
| Bridge text encoder | all-MiniLM-L6-v2, mean-pool, labels de-underscored |
| Soft-pool temperature τ | 0.07 (training temperature) |
| Bootstrap | B=1000, subject-stratified, seed 3431 |
| Split seed | 3431 |
| Label strings | frozen in `benchmark_data/eval_v2/labels/*.json`; no rephrasing |

## Ground-truth handling (HARTH-bug hardening)

`label_native.npy` codes ARE `metadata.json:activity_to_idx` values (written by
`preprocess_tsfm_eval.py`). v2 majority-votes the raw codes and maps them through
`idx_to_label` — there is deliberately no offset arithmetic. HARTH's codes
{2..8, 11} (4 classes have zero windows) map correctly by construction;
`tests/test_eval_v2.py::TestWindowGroundTruth` is the regression test.
(The v1 evaluator's `get_window_labels` was also patched with the `+ t` restore.)

## Running

```bash
# Main ZS re-score (native rate, rich channel text)
TSFM_CHECKPOINT=training_output/semantic_alignment/small_deep_v2_4b3fdd6/best.pt \
  python val_scripts/human_activity_recognition/evaluate_tsfm_v2.py --zs-only

# Parity row
python val_scripts/human_activity_recognition/evaluate_tsfm_v2.py \
  --zs-only --channel-text neutral --eval-rate 20

# Full (adds subject-disjoint FS-1%/10% fine-tuning; hours)
python val_scripts/human_activity_recognition/evaluate_tsfm_v2.py
```

Outputs: `test_output/eval_v2/tsfm_v2_{channel_text}_{eval_rate}.json`.
The v2 default checkpoint is the headline `small_deep_v2_4b3fdd6` (v1's default
pointed at a superseded model).

## Comparability warning

v2 numbers are **not comparable** to v1 numbers: the candidate sets, scoring rule,
metrics, and splits all changed. The v2 baseline table (this branch) is the
reference point for all V2-redesign experiments.
