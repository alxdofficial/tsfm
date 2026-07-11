# HALO dataset / baseline faithfulness cross-check — agent prompt

Dispatch ONE agent per target. Fill `{{NAME}}` and `{{TYPE}}` (dataset | baseline).
Train datasets: uci_har, hhar, pamap2, wisdm, dsads, kuhar, unimib_shar, hapt, mhealth, recgym, capture24.
Test datasets: motionsense, realworld, mobiact, shoaib, harth, inclusivehar (opportunity=appendix).
Baselines: crosshar, limubert (active); unimts, ssl_wearables (planned).

---

You are auditing ONE `{{TYPE}}` named `{{NAME}}` in the HALO/TSFM repo for faithfulness to its
SOURCE PAPER. This is READ-ONLY: produce a report, change no code. Repo root:
`/home/alex/code/HALO/code`. Use `.venv/bin/python` for JSON/parquet (Counter / schema — never
dump large files). Cite a paper page or a `file:line` for every claim.

## Step 1 — establish ground truth from the paper
1. Find the source: `references/{{TYPE}}s/{{NAME}}/` holds a paper PDF (or `webpage.html`) + `citation.json`.
   Read the PDF directly; use `references/README.md` as the index. If the paper is closed-access /
   absent and the fact isn't in `webpage.html`, mark the affected checks **UNVERIFIABLE** — do NOT guess.
2. Leverage prior passes (re-verify + extend, don't redo from scratch): `docs/v2/CROSSCHECK.md`,
   `docs/v2/CROSSCHECK2.md`, `docs/v2/data_diversity_audit.md`. Note which prior findings still hold
   vs are now fixed.

## Discipline (prior passes hit all of these)
- Classify every finding: **WRONG** (factual error) · **ALLOWED-VARIABILITY** (rate/units the model is
  meant to generalize over — not an error) · **INCOMPLETE** (missing but not false) · **UNVERIFIABLE**.
- Papers can be **buggy/self-contradictory** (e.g. a table whose row-names oppose its tokens). Flag that;
  do NOT "correct" a value to something the paper doesn't actually support.
- Report DISCREPANCIES (with the correct value + citation + `file:line`) AND explicit VERIFIED-CORRECT
  confirmations (so verified items aren't re-audited).
- Prior real errors to look for the class of: wrong placement, garbage interleaved columns, orientation-
  off channels fed as valid, wrong units, fake resampling, vague/renamed labels, dropped laterality,
  mislabeled fall direction, wrong institution.

## If `{{TYPE}}` == dataset — check
Files: `data/{{NAME}}/manifest.json`, `data/{{NAME}}/labels.json`, one
`data/{{NAME}}/sessions/*/data.parquet` (schema only), `datascripts/{{NAME}}/convert.py`,
`benchmark_data/dataset_config.json` (the `{{NAME}}` entry: rate/placement/split),
`benchmark_data/eval_v2/labels/{{NAME}}.json` (if a test set).

1. **Labels** — unique labels in `labels.json`: names, spelling, count, and SEMANTICS faithful to the
   paper's activity list? (vague renames, dropped laterality, merged/omitted/fabricated classes,
   under/over-count in the manifest description). Also check `eval_v2/labels/{{NAME}}.json`.
2. **Data format / syntax** — manifest well-formed (each channel has name + description +
   `sampling_rate_hz`); `labels.json` is `session_id -> [label]`; the parquet carries `timestamp_sec` +
   exactly the manifest's channels; sampling_rate agrees across manifest / config / converter; no
   NaN/placeholder column silently fed as real signal.
3. **Channel descriptions** — per channel: sensor type (accel/gyro/mag), axis (x/y/z), PLACEMENT
   (wrist/waist/trouser-pocket/chest/upper-arm/ankle/torso/…), units, gravity state
   (raw-includes-gravity vs gravity-removed / body-acc), sampling rate — all faithful to the paper?
4. **Dataset description** — manifest top-level description + `dataset_config` placement correct and
   NOT stale vs the converter.
5. **Augmentation validity for THIS dataset** — given its sensors/placement/units/rate, are the applied
   augmentations physically valid? gravity add/remove only on a gravity-bearing accelerometer (skip
   already-removed, e.g. UCI `body_acc`); yaw-rotation axis estimable (needs gravity in acc); rate-
   resample range sane vs native rate; channel-dropout leaves a valid x/y/z triad; the dataset-specific
   LABEL synonyms/templates and any CHANNEL-text paraphrase PRESERVE meaning (no wrong synonym, no
   invented laterality/direction). Files: `datasets/imu_pretraining_dataset/augmentations.py`,
   `label_augmentation.py`, `multi_dataset_loader.py`, `datascripts/shared/windowing.py`.

## If `{{TYPE}}` == baseline — check
Files: `val_scripts/human_activity_recognition/baselines/{{NAME}}.py` (the adapter),
`run_baselines_v2.py`, `docs/baselines/BASELINE_IMPLEMENTATION_NOTES.md`,
`docs/baselines/EVALUATION_PROTOCOL_V2.md`, any vendored original in `auxiliary_repos/`.

1. **Architecture faithfulness** — does the adapter reproduce the paper's model (encoder type,
   depth/width, tokenization, pretraining objective), either by calling the vendored original repo or a
   re-implementation? Flag deviations (wrong depth, missing component, different input length/rate).
2. **Preprocessing faithfulness** — input sampling rate, window length, normalization, channel set match
   the paper AND are applied consistently to HALO's data (resample to the baseline's native rate, right
   channels, right units).
3. **Weights** — released paper weights or a faithful re-train? Correct checkpoint?
4. **Eval-bridge faithfulness** — closed-vocab baselines apply the ConSE bridge (Norouzi 2014) correctly
   (top-T, pooling/temperature); open-vocab/cosine baselines score by cosine; parity rows per
   `EVALUATION_PROTOCOL_V2.md`; subject-disjoint (no leakage); SAME label strings as HALO.
5. **Hyperparameters** — any that depart from the paper without justification.
6. **Labels/format the baseline consumes** — same as the dataset label/format checks.

## Output
- **Findings table**: `# | severity (blocker/high/med/low) | check | discrepancy (WRONG value ->
  correct value + paper citation) | file:line | confidence (HIGH/MED/UNVERIFIABLE)`.
- **Concrete fixes** (file:line + exact correct value + citation), ranked by severity.
- **Verified-correct** list (what is confirmed faithful).
- **Unverifiable** items + the exact source that would resolve each.
