# HALO V2 — Evaluation Design Spec

*Component design for [REDESIGN_PLAN.md](REDESIGN_PLAN.md). Where this and the integrated plan disagree, the plan's §1 interface reconciliation wins.*

---

# HALO Consolidated Evaluation Protocol (v2) — Design Spec

Everything below is grounded in the four files I read plus a verification pass on the preprocessed data. The single most important discovery: **subject IDs are already present in the eval tensors** — `benchmark_data/processed/tsfm_eval/{ds}/label_native.npy` has shape `(N, T, 2)` with **channel 0 = activity index, channel 1 = subject index** (unique counts match `metadata.json`'s `subject_to_idx`). `evaluate_tsfm.py:get_window_labels` reads only `label_index=0` and throws channel 1 away. So subject-disjoint evaluation is a **pure code change, no re-preprocessing**. That fact makes the fixes below cheap.

---

## 0. Diagnosis — why the current protocol invites nitpicking

Concretely, from the files:

1. **Open-set over 87 strings is not a real task, and it is leaky.** `evaluate_zero_shot_open_set` (`evaluate_tsfm.py:499`) argmaxes cosine sim over all 87 training strings, then collapses prediction and ground truth through `get_label_to_group_mapping()` before scoring. The headline "ZS-Open 42.0%" (`RESULTS.md:413`) is therefore *not* an accuracy over 87 classes — it is an accuracy over ~25 hand-drawn groups. A reviewer will ask "why 87? why these groups?" and there is no principled answer.

2. **The synonym ontology encodes held-out test labels — this is test-label leakage into the metric definition.** `label_groups.py` comments are explicit:
   - L19–20: `nordic_walking, walking_straight, walking_winding are VTT-ConIoT (zero-shot)` folded into `walking`.
   - L78: `rope_jumping/rope_skipping are RecGym (zero-shot)` folded into `jumping`.
   - L96 `vehicle_entry: [car_step_in, car_step_out]` — these are **MobiAct** test strings.
   - L127–131 `carrying/climbing/kneeling/painting` groups exist *only* to catch **VTT-ConIoT** test strings.
   The group boundaries were drawn *with knowledge of the test labels*. Any metric that routes scoring through this ontology is contaminated by design, independent of whether test *data* was seen.

3. **The ontology is also fragile.** `get_label_to_group_mapping` (L238) takes "the first assignment" for labels in multiple groups — dict-insertion-order dependent (L253 comment admits it for `stairs`). A reordering silently changes scores.

4. **Splits are not subject-disjoint.** `prepare_train_test_split` (`evaluate_tsfm.py:386`) shuffles *windows* and slices 80/10/10. Windows from one subject/session land in both the few-shot support and the query set → within-subject leakage. Literature shows this inflates HAR numbers by 10–15 points [LOSO-Gholamiangonabadi](https://consensus.app/papers/details/f677839713b55ef087ff10478c05aec1/) (99.85→85.1), [LOSO-Rehman](https://consensus.app/papers/details/fcdfa1dfc9985f95ba51f666f8800f90/) (89→76).

5. **Accuracy is the headline; macro-F1 (the honest number) is buried and ~2× lower.** ZS-Open 42.0 acc vs 21.0 macro-F1 (`RESULTS.md:413`) is a class-imbalance tell. For imbalanced ZS-HAR the correct primary is macro-F1 [Modality-Gap](https://consensus.app/papers/details/5e6c92a6f65a5f029618ecf827a0f54c/).

6. **Four metrics × two scoring conventions × per-model asymmetry** (text-aligned = exact match, classifier-based = group match; `RESULTS.md:189–192`) means no two cells in the results table are computed the same way. This is the core nitpick surface.

---

## 1. The consolidated metric set (smallest that fully characterizes the model)

Replace 4 metrics × messy scoring with **3 metrics, one scoring rule (exact match, macro-F1), one split rule (subject-disjoint)**.

### Metric A — **ZS-XD** : Zero-Shot Cross-Dataset closed-set (PRIMARY)

The standard CLIP-style zero-shot transfer [CLIP](https://arxiv.org/abs/2103.00020), instantiated exactly as the top HAR competitor does it [UniMTS](https://consensus.app/papers/details/f924af83748257a8b57407578dfb8e84/): the candidate label set is **the target dataset's own published label strings**, verbatim.

For a window `x_i` in target dataset `D` with its own label vocabulary `L_D = {t_1,…,t_C}` (the strings the dataset ships):

```
ŷ_i = argmax_{c ∈ L_D}  cos( f(x_i), g(t_c) )
```

- `f` = HALO IMU encoder, `g` = frozen text encoder (same one used in training).
- **No 87-way set. No synonym groups. Exact match against L_D.**
- **Primary number = macro-F1**; secondary = balanced accuracy + weighted-F1.

Justification for macro-F1 as primary and for scoring against the *short native label* (not a rich description): rich text collapses SBERT prototype separability and imbalanced accuracy overstates performance — [Modality-Gap](https://consensus.app/papers/details/5e6c92a6f65a5f029618ecf827a0f54c/) recommends macro-F1 as the primary ZS-HAR metric and shows minimal prototypes separate better.

Because every target dataset is entirely held out of pretraining, ZS-XD is *inherently* subject-disjoint from training. The only splitting question is few-shot (Metric B).

### Metric B — **FS-1% / FS-10%** : Few-shot deployability (SECONDARY, the "usefulness" metric)

Fine-tune / probe on `k ∈ {1%, 10%}` labeled support from `D`, evaluate on the disjoint query set. Same cosine-sim head as A (no separate classifier for HALO). **Primary = macro-F1**, subject-disjoint support/query, multi-seed. This is the number a practitioner cares about ("I collected a little of my own data") and directly serves design-goal #1 (real-world usefulness).

### Metric C — **T→S Recall@1** : retrieval diagnostic (APPENDIX-ONLY, keep it)

One retrieval number is *warranted* because it characterizes embedding geometry independent of the argmax decision rule, and the infra already exists (`evaluation_metrics.py:compute_semantic_recall`). Report **class-balanced text→sensor Recall@1** over the exact target label corpus (`use_groups=False`), macro-averaged over classes. Report once, as a diagnostic, not a headline. Do **not** report the recall variant that uses `use_groups=True`.

### Decision on "open-set": **drop it as a headline; replace with one pre-registered, leakage-free open-vocabulary robustness probe (appendix).**

The 87-way open-set answers a question no deployment has ("pick among a private training vocabulary"). If you want to keep an open-vocabulary stress test, define it with **zero dependence on any test label**:

> **OV-Distractor@k**: prediction set = `L_D ∪ R`, where `R` is a **frozen, pre-registered** set of K distractor activity strings drawn from a public activity taxonomy (e.g., the [Google/AudioSet-style or Wikipedia activity list]) that is committed to the repo *before* looking at any test dataset and never edited. Score = macro-F1 over the true `L_D` classes only; picking a distractor counts as an error. K is fixed (e.g., 50).

This measures "does the model resist plausible wrong strings" without groups and without leakage. It is optional. **Recommendation: ship the paper on A + B, put C and OV-Distractor in the appendix.**

**Net: the synonym ontology (`label_groups.py`) is deleted from the evaluation path entirely.** It may remain in the *training* code for class-balancing if training uses it, but it must not be importable by any scorer.

---

## 2. Killing the synonym problem (why target-closed-set makes it vanish)

The synonym ontology exists to bridge one vocabulary to another: training says `jogging`, MobiAct says `jogging`, RealWorld says `running`. Groups were the patch so a `jogging`-emitting classifier isn't punished on a `running` dataset.

With **target-dataset-closed-set**, that bridge is unnecessary by construction:
- You never compare a *training* string to a *test* string. You embed **the target dataset's own strings** (`g(t_c)` for `t_c ∈ L_D`) and predict among them.
- If RealWorld ships `running`, you embed `"running"` and compete among RealWorld's 8 strings. If MobiAct ships `jogging`, you embed `"jogging"`. The model's job is to point at the right *native* string. There is no cross-vocabulary mismatch to reconcile.
- MobiAct's `car_step_in`, `fall_forward`, etc. (currently needing the `vehicle_entry` / `falling` groups, `label_groups.py:69,96`) become plain members of MobiAct's 13-string candidate set. No special-casing.

**Residual synonym handling = none.** Every dataset defines its own labels; scoring is exact match against those exact strings. The only pre-registered text asset that remains is a trivial, auditable **verbatim label list per dataset** (already `DATASET_CONFIG["datasets"][ds]["activities"]`, consumed by `get_dataset_labels`). That list is the dataset's, not ours — no discretion, no leakage.

If a reviewer insists a dataset's own label string is a poor English rendering (e.g., `stairs_down`), the pre-registered rule is: **use the dataset's published string, optionally passed through a fixed, dataset-independent normalizer** (`_` → space, lowercase) committed before evaluation. No per-label editorializing.

---

## 3. Fixing validity

### 3a. Subject-disjoint splits everywhere (LOSO-style)

Use channel 1 of `label_native.npy` as the group key. Two regimes:

- **ZS-XD (Metric A):** already train-disjoint at the dataset level; evaluate on the **full** target set (all subjects), no split needed. Report a **per-subject bootstrap 95% CI** (below).
- **Few-shot (Metric B):** replace window-shuffle with **grouped-by-subject** splitting. For datasets with ≥8 subjects use **Leave-One-Subject-Out (LOSO)**; for small-subject datasets (Opportunity = 4, Shoaib = 10) use **k-fold GroupKFold by subject** (k = n_subjects capped at 5). Support = k% sampled *within training-fold subjects only*; query = held-out subject(s). Never let a subject appear in both.

Cite the leakage magnitude so reviewers see it's principled: [LOSO-Gholamiangonabadi](https://consensus.app/papers/details/f677839713b55ef087ff10478c05aec1/), [LOSO-Rehman](https://consensus.app/papers/details/fcdfa1dfc9985f95ba51f666f8800f90/).

### 3b. Multi-seed error bars

Every reported cell = **mean ± 95% CI**.
- Few-shot: over LOSO folds (or GroupKFold folds) × ≥3 support-sampling seeds.
- Zero-shot (no split randomness): **subject-stratified bootstrap** — resample subjects with replacement B=1000 times, recompute macro-F1, report the 2.5/97.5 percentiles. This gives an honest CI for a single-pass metric.

### 3c. Exact-match, no grouping, is the honest default

macro-F1 over `L_D` with `zero_division=0`:

```
P_c = TP_c/(TP_c+FP_c),  R_c = TP_c/(TP_c+FN_c)
F1_c = 2 P_c R_c / (P_c + R_c)   (:= 0 if P_c+R_c = 0)
macro-F1 = (1/C) Σ_{c=1..C} F1_c
```

Report **balanced accuracy** = `(1/C) Σ_c R_c` alongside. This directly answers the "accuracy 42 vs macro-F1 21" objection: balanced accuracy is an accuracy-family number that is *already* robust to imbalance, so the gap between it and macro-F1 shrinks and is interpretable rather than embarrassing.

---

## 4. Fairness — one clean rule for rate + channel-text

**The confound today:** HALO evaluates at native rate (30–50 Hz) with rich per-channel manifest text; every baseline gets 20 Hz and no metadata (`EVALUATION_PROTOCOL.md:66–99`, `RESULTS.md:159–173`). That is two advantages baked into the headline.

**One rule — "Same signal in, capabilities isolated":**

1. **Headline parity row.** All models — HALO included — receive the **same physical signal**, resampled once to a single common evaluation rate per dataset using a **polyphase anti-aliased resampler** (not naive interpolation; the current `F.interpolate` is itself the aliasing culprit flagged in the brief — see [BlurPool](https://consensus.app/papers/details/74cb2454204e5c6191b34d32f37f05b4/), [AaSP](https://consensus.app/papers/details/519c2b54e3ee5ed99ebb9d68b231a4c7/)). For channel text, all models get the **same neutral channel string** (or none). This row is the like-for-like comparison a reviewer cannot attack.

2. **Capability delta rows (appendix), explicitly labeled.** HALO with (a) native rate and (b) rich channel-text are reported as **+Δ rows on top of the parity row**, never folded into the headline mean. This preserves the (legitimate) architectural story — native-rate handling and channel-text conditioning [GOAT](https://consensus.app/papers/details/ed52d465eb1d5e1f8a35fb611d3ba628/) are real capabilities — while making clear the *core comparison* is confound-free.

This converts "TSFM got extra inputs" from a fatal objection into a measured ablation. It also aligns with design-goal #2: once the new tokenizer is genuinely rate-invariant (principled anti-aliased filterbank, [Learnable-Frontends](https://consensus.app/papers/details/59964a2584355a5ba01c23cb2b451dbc/)), the native-rate Δ should be ~0, which is itself a selling point you can now *demonstrate* with this table.

---

## 5. Migration

### 5a. Exact changes to `evaluate_tsfm.py`

**Delete (open-set + groups path):**
- Import of `LABEL_GROUPS, get_label_to_group_mapping` (L46–49).
- `evaluate_zero_shot_open_set` (L499–548) entirely.
- `open_set=True` branch of `evaluate_zero_shot_majority_vote` (L629–686) and its call site (L1001–1012).
- All `zero_shot_open_set` / `_mv` entries in the results dict and printout (L906, L1001–1012).

**Keep + promote to primary:**
- `evaluate_zero_shot_closed_set` (L551) — this is already ZS-XD (target strings, exact match). Change reported headline field from `accuracy` to `f1_macro`; add `balanced_accuracy`.

**Add subject-disjoint few-shot (the core validity fix):**
- In `main` (L964), capture the subject vector:
  ```python
  window_subjects = raw_labels[:, 0, 1].astype(int)   # channel 1 = subject
  ```
- Replace `prepare_train_test_split` (L386) with a `GroupShuffleSplit`/`LeaveOneGroupOut`/`GroupKFold` on `window_subjects`. `balanced_subsample` (L369) then draws k% *within the training-fold subjects only*.
- Wrap `evaluate_supervised_finetune` (L695) in a fold loop and a seed loop; aggregate mean ± 95% CI.

**Add multi-seed / bootstrap CI:**
- Zero-shot: subject-stratified bootstrap over `window_subjects` around the macro-F1.
- Few-shot: aggregate over folds × seeds.

**Add anti-aliased resampling + parity mode:**
- A `--eval-rate` flag (default = native for capability rows; a fixed common rate for the parity row) using `scipy.signal.resample_poly` instead of the model's internal `F.interpolate`.
- A `--channel-text {native,neutral,none}` flag for the fairness ablation.

Result: the script computes **A (ZS-XD macro-F1 + balanced acc + CI), B (FS-1%/10% macro-F1, subject-disjoint, multi-seed), C (T→S R@1 diagnostic)** — no group import remains.

### 5b. Changes to `EVALUATION_PROTOCOL.md`

- Delete §"Metric 1: Zero-Shot Open-Set", the closed-set *mask* machinery (L132–139), "group-based scoring" fairness section (L162–167), and the "Label Group Mapping Coverage" table (L215–231) — all are group-dependent.
- Rewrite the metric framework to **A/B/C** above, with the formulas from §3c verbatim.
- Add a **Pre-registration** box: "Candidate label set for every dataset = its published label strings, verbatim, committed at `dataset_config.json`. No test label influences any group, threshold, or normalizer. Normalizer is fixed and dataset-independent."
- Add a **Subject-disjoint** box citing the two LOSO papers, and the note that subject indices are already in `label_native.npy[:, :, 1]`.
- Replace the "Sampling Rate Policy" per-model table with the single fairness rule in §4 (parity row + capability Δ rows).

### 5c. Tables the paper should show

- **Table 1 (headline):** ZS-XD **macro-F1** per main dataset + mean, all models at parity rate/neutral text, mean ± 95% CI.
- **Table 2:** FS-1% / FS-10% **macro-F1**, subject-disjoint (LOSO/GroupKFold), multi-seed CI.
- **Table 3 (appendix):** HALO capability Δ — native rate, +channel-text — as increments over the Table 1 parity row.
- **Table 4 (appendix, optional):** T→S Recall@1 diagnostic; OV-Distractor@k.

### 5d. Before/after on one dataset — **MobiAct**

**Before (current):**
- ZS-Open **42.2% acc** = argmax over 87 training strings, then collapsed through ~25 hand groups (`falling`, `vehicle_entry`, …) that were authored knowing MobiAct ships `car_step_in`/`fall_*` — leaky, and it's really an accuracy over groups, not 87 classes.
- ZS-Closed **50.0% acc** (exact) vs classifier baselines scored by **group match** on a 34/87 mask — non-comparable cells.
- Honest macro-F1 (35.2 / 37.1) buried; window-shuffle split (subjects in both support and query).
- Four numbers, two scoring rules, one ontology dependency.

**After (v2):**
- **One primary number:** ZS-XD macro-F1 over MobiAct's own 13 strings `{car_step_in, car_step_out, fall_backward_knees, …, walking}`, exact match, `argmax_c cos(f(x), g(t_c))`, computed identically for HALO and every baseline. Subject-stratified bootstrap CI over MobiAct's 24 subjects. Balanced accuracy reported beside it.
- **FS-1%/10% macro-F1**, LOSO over the 24 subjects, 3 seeds, mean ± CI.
- **No 87-set, no group table, no mask-expansion table, no synonym file import.** Falls and vehicle-entry are just MobiAct labels the model must point at.
- Fairness: Table 1 uses parity-rate input; native-rate/channel-text upside shown separately in Table 3.

A reviewer reading the new MobiAct row sees exactly one, standard, exact-match, class-balanced, subject-disjoint number with a confidence interval — nothing to argue about re: open-vs-closed or synonyms.

---

## Data digest for integration

Per-dataset facts verified from `benchmark_data/processed/tsfm_eval/*/{label_native.npy,metadata.json}`:

| Dataset | Role | Windows | Classes | **Subjects** | Native Hz | has_gyro | Few-shot regime |
|---|---|---:|---:|---:|---:|:--:|---|
| MotionSense | main test | 12,080 | 6 | 24 | 50 | yes | LOSO |
| RealWorld | main test | 27,138 | 8 | 15 | 50 | no | LOSO |
| MobiAct | main test | 4,345 | 13 | 24 | 50 | yes | LOSO |
| Shoaib | main test | 5,537 | 7 | 10 | 50 | yes | GroupKFold-5 |
| Opportunity | main test | 6,453 | 4 | 4 | 30 | yes | GroupKFold-4 (caveat: few subjects) |
| HARTH | main test | 47,330 | 12 | 22 | 50 | no | LOSO |
| InclusiveHAR | main test | 3,370 | 6 | 20 | 50 | yes | LOSO |

Integration keys:
- **Subject vector:** `label_native.npy[:, 0, 1]` (int). Activity: `[:, 0, 0]`. Group by column 1 for all splits.
- **Native label list:** `DATASET_CONFIG["datasets"][ds]["activities"]` via `get_dataset_labels()` (`evaluate_tsfm.py:360`) — this is the entire candidate set; already sorted, verbatim, no groups.
- **Files to stop importing anywhere in the eval path:** `datasets/imu_pretraining_dataset/label_groups.py`, and the `use_groups=True`/`open_set=True` code paths in `evaluation_metrics.py` and `grouped_zero_shot.py`.
- **Resampler for parity:** `scipy.signal.resample_poly` (anti-aliased) — replaces both the baseline 20 Hz pipeline and HALO's internal `F.interpolate` for the fairness row.
- **Primary metric field everywhere:** `f1_macro` (already computed at `evaluate_tsfm.py:544, 590, 683, 861`), plus add `sklearn.metrics.balanced_accuracy_score`.

---

### References
- [CLIP](https://arxiv.org/abs/2103.00020) — zero-shot transfer via cosine sim to target-label text; the template for ZS-XD.
- [UniMTS](https://consensus.app/papers/details/f924af83748257a8b57407578dfb8e84/) — top HAR competitor; per-dataset own-label zero-shot, establishing target-closed-set as the field standard.
- [Modality-Gap](https://consensus.app/papers/details/5e6c92a6f65a5f029618ecf827a0f54c/) — macro-F1 as primary for imbalanced ZS-HAR; minimal prototypes separate better than rich text.
- [LOSO-Gholamiangonabadi](https://consensus.app/papers/details/f677839713b55ef087ff10478c05aec1/), [LOSO-Rehman](https://consensus.app/papers/details/fcdfa1dfc9985f95ba51f666f8800f90/) — magnitude of subject-leakage inflation.
- [BlurPool](https://consensus.app/papers/details/74cb2454204e5c6191b34d32f37f05b4/), [AaSP](https://consensus.app/papers/details/519c2b54e3ee5ed99ebb9d68b231a4c7/), [Learnable-Frontends](https://consensus.app/papers/details/59964a2584355a5ba01c23cb2b451dbc/) — anti-aliased resampling for the fairness/parity row.
- [GOAT](https://consensus.app/papers/details/ed52d465eb1d5e1f8a35fb611d3ba628/) — device-position text conditioning; justifies keeping channel-text as a measured capability Δ.

---

## Addendum (2026-07-02): scoring closed-vocabulary baselines under ZS-XD

**Problem.** ZS-XD scores against the target dataset's own strings `L_D`. Text-aligned models (HALO, LanHAR, LLaSA-with-candidate-prompt, UniMTS/GOAT) consume `L_D` directly. Closed-vocabulary baselines (LiMU-BERT+GRU, CrossHAR+Transformer, MOMENT+SVM) can only emit labels from their training vocabulary `L_train` and have no mechanism to produce unseen test strings — the deleted synonym ontology existed to patch exactly this.

**Policy (rev. 2, 2026-07-02 — literature-grounded; supersedes the earlier argmax bridge, which loses the classifier's uncertainty and would disadvantage closed-vocab baselines).**

The literature establishes three practices for exactly this situation; v2 adopts all three, so every closed-vocab baseline gets at least one stage with zero handicap:

1. **Capability-scoped tables (dominant practice).** The main ZS-XD table contains models capable of classifying against an arbitrary label list — CLIP-benchmark precedent: supervised/closed-vocab models appear as linear-probe/fine-tune rows, never forced into the zero-shot column. The ZSL evaluation canon ([Xian et al., TPAMI](https://consensus.app/papers/details/f74ed3e8efd65aeeaa9de3cf3c934e85/)) also mandates per-class averaged accuracy — consistent with macro-F1 primary. Closed-vocab baselines compete at full strength, handicap-free, in the FS-1%/10% columns.
2. **ConSE bridge for †-marked ZS rows** ([Norouzi et al.](https://consensus.app/papers/details/302f733750005d3093477e9d259e9c05/) — the established, no-retraining conversion of an N-way classifier into a zero-shot predictor): take the baseline's full softmax `p(ℓ|x)` over `L_train`, form the convex combination `v(x) = Σ_{ℓ∈top-T} p(ℓ|x)·E(ℓ) / Σ_{ℓ∈top-T} p(ℓ|x)`, predict `argmax_{c∈L_D} cos(v(x), E(c))`, with `E` = the same frozen SBERT for all models, `T=10`. Strictly generalizes an argmax bridge (T=1) and preserves the classifier's uncertainty — fixing the information-loss disadvantage. Identical strings still behave as exact match; ConSE was strong enough to beat DeViSE on ImageNet ZSL, so this is a *credible* baseline row, not a strawman.
3. **Common-classes table (established cross-dataset HAR practice — label harmonization/intersection, e.g. [Cook et al. survey](https://consensus.app/papers/details/366efc0212ff589ca07e2a666f838cfa/)):** an appendix table restricted to activities where a baseline's training vocabulary and `L_D` correspond 1:1. No bridging, no language machinery — closed-vocab baselines compete purely on signal discrimination. Their best-case stage.
4. **Coverage is computed, not hand-authored:** class `c ∈ L_D` is *reachable* for a baseline iff some ConSE output can land on it. Report reachable-class fraction per (baseline, dataset). Unreachable classes are structural zeros; per-class macro-F1 exposes them — a true capability statement, not a scoring artifact.
5. **Residual, non-patchable asymmetry — stated openly in the protocol:** classifier label semantics were never optimized against SBERT geometry, while HALO's were. That asymmetry *is* the capability under test; the common-classes table (no language) and probe columns (no zero-shot) are the controls that quantify it rather than hide it.
6. **Pre-registered label strings:** `L_D` = each dataset's own documented label names, frozen in a config before any evaluation; no rephrasing. Datasets shipping codes use their documented human-readable names.
7. **Transparency analysis (applies to HALO too):** report each test class's max text-similarity to any training label and correlate with per-class F1 — makes the source of zero-shot transfer (near-synonym vs genuinely novel) explicit instead of contestable.
