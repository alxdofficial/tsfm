# HALO — full dataset faithfulness cross-check (single orchestrator)

You are the ORCHESTRATOR of a faithfulness audit of ALL HALO/TSFM datasets against their SOURCE
PAPERS. READ-ONLY: produce reports, change no code. Repo root: `/home/alex/code/HALO/code`. Use
`.venv/bin/python` for JSON/parquet (Counter / schema only — never dump large files). Cite a paper
page or a `file:line` for every claim.

## Datasets to audit (17)
- Train (11): uci_har, hhar, pamap2, wisdm, dsads, kuhar, unimib_shar, hapt, mhealth, recgym, capture24
- Test (6): motionsense, realworld, mobiact, shoaib, harth, inclusivehar
- (opportunity = appendix; audit it too if cheap.)

## How to run it
You MAY spawn subagents to parallelize — e.g. one subagent per dataset — IF you have an agent/task-
spawning tool. If so: give each subagent the **DATASET AUDIT SPEC** below VERBATIM with its `{{NAME}}`
filled in, have it return the structured report the spec defines, and collect all reports. If you have
no spawning tool (or prefer), audit the datasets yourself one at a time under the same spec. Either
way, YOU own the final consolidated report — do not skip synthesis.

---

## DATASET AUDIT SPEC (per dataset `{{NAME}}`)

Establish ground truth from the paper FIRST:
1. `references/datasets/{{NAME}}/` holds a paper PDF (or `webpage.html`) + `citation.json`. Read the PDF
   directly; `references/README.md` is the index. If the paper is closed-access/absent and the fact
   isn't in `webpage.html`, mark the affected checks **UNVERIFIABLE** — do NOT guess.
2. Leverage prior passes (re-verify + extend, don't redo): `docs/v2/CROSSCHECK.md`, `CROSSCHECK2.md`,
   `data_diversity_audit.md`. Note which prior findings still hold vs are now fixed.

Discipline (prior passes hit all of these):
- Classify each finding: **WRONG** (factual error) · **ALLOWED-VARIABILITY** (rate/units the model is
  meant to generalize over — not an error) · **INCOMPLETE** (missing but not false) · **UNVERIFIABLE**.
- Papers can be buggy/self-contradictory (a table whose row-names oppose its tokens) — flag that; do
  NOT "correct" to a value the paper doesn't support.
- Report DISCREPANCIES (correct value + citation + `file:line`) AND explicit VERIFIED-CORRECT items.
- Class of prior real errors: wrong placement, garbage interleaved columns, orientation-off channels
  fed as valid, wrong units, fake resampling, vague/renamed labels, dropped laterality, mislabeled fall
  direction, wrong institution.

Files for `{{NAME}}`: `data/{{NAME}}/manifest.json`, `data/{{NAME}}/labels.json`, one
`data/{{NAME}}/sessions/*/data.parquet` (schema only), `datascripts/{{NAME}}/convert.py`,
`benchmark_data/dataset_config.json` (the `{{NAME}}` entry), `benchmark_data/eval_v2/labels/{{NAME}}.json`
(if a test set).

Checks:
1. **Labels** — names, spelling, count, SEMANTICS faithful to the paper's activity list? (vague renames,
   dropped laterality, merged/omitted/fabricated classes, manifest over/under-count). Also
   `eval_v2/labels/{{NAME}}.json`.
2. **Data format / syntax** — manifest well-formed (each channel: name + description +
   `sampling_rate_hz`); `labels.json` is `session_id -> [label]`; parquet carries `timestamp_sec` +
   exactly the manifest's channels; sampling_rate agrees across manifest/config/converter; no
   NaN/placeholder column fed as real signal.
3. **Channel descriptions** — per channel: sensor type, axis (x/y/z), PLACEMENT
   (wrist/waist/trouser-pocket/chest/upper-arm/ankle/torso), units, gravity state (raw-includes-gravity
   vs gravity-removed / body-acc), rate — all faithful to the paper?
4. **Dataset description** — manifest top-level description + `dataset_config` placement correct and NOT
   stale vs the converter.
5. **Augmentation validity for THIS dataset** — given its sensors/placement/units/rate: gravity
   add/remove only on a gravity-bearing accelerometer (skip already-removed, e.g. UCI `body_acc`);
   yaw-rotation axis estimable (needs gravity in acc); rate-resample range sane vs native rate;
   channel-dropout leaves a valid x/y/z triad; the dataset-specific LABEL synonyms/templates and any
   CHANNEL-text paraphrase PRESERVE meaning (no wrong synonym, no invented laterality/direction). Files:
   `datasets/imu_pretraining_dataset/augmentations.py`, `label_augmentation.py`, `multi_dataset_loader.py`,
   `datascripts/shared/windowing.py`.

Per-dataset OUTPUT (return this):
- Findings table: `severity (blocker/high/med/low) | check | discrepancy (WRONG -> correct + citation) |
  file:line | confidence (HIGH/MED/UNVERIFIABLE)`.
- Concrete fixes (file:line + exact correct value + citation).
- Verified-correct list. Unverifiable items + the source that would resolve each.

---

## CONSOLIDATION (you, the orchestrator)
After every dataset is audited, WRITE `docs/v2/dataset_crosscheck_report.md` and return a ~30-line exec
summary containing:
1. A MASTER findings table across ALL datasets: `dataset | severity | check | discrepancy | file:line |
   confidence`.
2. CROSS-DATASET PATTERNS: recurring error classes (placement mislabels, stale manifests, dropped
   laterality, missing units/gravity, augmentation-invalidity) and which datasets are fully
   VERIFIED-CORRECT.
3. GLOBAL RANKED FIX LIST (all HIGH/MED first) — each with `file:line` + exact correct value + citation
   — plus the UNVERIFIABLE items + the source that would resolve each.
Keep every number and citation; do not drop findings in synthesis.
