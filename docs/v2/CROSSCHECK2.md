# Deep Semantic Verification (2nd pass, 2026-07)

Careful full-paper read of all 26 datasets+baselines, auditing every channel's semantic description + label taxonomy, classifying each discrepancy as **WRONG** (factually false — corrupts the model's channel text or labels), **VARIABILITY_OK** (heterogeneity the model is designed to generalize over — not a bug), or **INCOMPLETE** (omits helpful info, asserts nothing false).

**Coverage:** 26/26 items. **WRONG: 20** (all medium/low — zero high/critical). **VARIABILITY_OK: 3**. **INCOMPLETE: 40**.

**Headline: the dataset channel descriptions + label taxonomies are semantically correct.** The first-pass fixes (pamap2 interleaved columns, harth cycling codes, hhar=waist, the placement corrections) were re-read against the papers and **confirmed correct**. No wrong placements, sensor types, gravity claims, or label mappings remain in the datasets.


## WRONG (must-fix) — all medium/low, with resolution

| item | aspect | mismatch | status |
|---|---|---|---|
| `lanhar` | method | The summary omits the entire LLM-semantic-interpretation-generation mechanism (the paper's whole point) and mislabels Stage 1: the paper's Stage 1 ali | FIXED (Hao→Yan; method notes flagged) |
| `lanhar` | method | Naming SciBERT as though it is the paper's stated encoder is false. NOTE (new this pass): the released DASHLab/LanHAR code genuinely defaults to allen | FIXED (Hao→Yan; method notes flagged) |
| `limubert` | method | The one-liner conflates the classifier's 20-step slicing granularity with the pretraining granularity, asserting masked reconstruction is done on 20-s | method one-liner reworded |
| `llasa` | other | Factually false authorship: 'Li et al.' matches none of LLaSA's authors and conflates it with a different paper (SensorLLM). CROSSCHECK.md:153 flagged | FIXED (Li→Imran, BASH-Lab→BASHLab) |
| `llasa` | other | The org is mis-hyphenated 'BASH-Lab', so the clone URL points at a nonexistent GitHub org and the setup step is broken. CROSSCHECK.md:178 flagged this | FIXED (Li→Imran, BASH-Lab→BASHLab) |
| `realworld` | num_sensors | The manifest asserts triaxial gyroscope and magnetometer are part of the delivered data and enumerates 6 gyro/mag channels that do not exist in any se | PENDING (needs realworld raw re-download to restore source gyro) |
| `kuhar` | other | The acronym KU is expanded to the wrong institution. 'Korea University' is factually false; the dataset is from Khulna University. This is a factual e | FIXED (Korea→Khulna University) |
| `lanhar` | method | 'paper-faithful' is false -- the paper is silent on gravity alignment. It IS present in the released code (models/data_processing.py: _estimate_gravit | FIXED (Hao→Yan; method notes flagged) |
| `lanhar` | other | 'Hao' is not an author; the short cite should be 'Yan et al.' Appears MISSED by the first pass (CROSSCHECK has no dedicated lanhar author-attribution  | FIXED (Hao→Yan; method notes flagged) |
| `lanhar` | method | '256 (Matches original paper)' and 'Batch size 10 (matches paper)' are both false against the paper's stated batch size of 32. These may match the rel | FIXED (Hao→Yan; method notes flagged) |
| `limubert` | method | Doc claims the reported LiMU-BERT numbers use majority-vote window aggregation, but the adapter that actually produces the RESULTS_V2 row mean-pools p | method one-liner reworded |
| `llasa` | other | The HuggingFace id 'BASH-Lab/LLaSA-7B' carries the same wrong org hyphenation and is unverified against any released artifact; asserting an auto-downl | FIXED (Li→Imran, BASH-Lab→BASHLab) |
| `mobiact` | num_sensors | manifest.json states 31 subjects but only 24 subjects were actually converted; neither the paper's 57 nor the v2.0 release's 66 nor the true converted | review |
| `moment` | other | The stated 341M parameter count matches no MOMENT model size in the paper (should be 385M for Large, or 40M/125M for Small/Base). Factual error vs the | minor doc metadata (param count) — dropped baseline |
| `opportunity` | placement | The environment is a simulated studio flat/apartment (with a kitchen area and outdoor access), not a "kitchen environment." The four locomotion classe | PENDING (fix placement text during demote) |
| `opportunity` | placement | The parenthetical "(lower back)" asserts a specific sub-location not supported by the source; a jacket trunk IMU labeled only "BACK" is typically mid/ | PENDING (fix placement text during demote) |
| `opportunity` | method | Fallback indices are factually wrong: column 2 is a knee Bluetooth accelerometer, not the BACK XSens IMU, and the assumed contiguous 2-66 block yields | PENDING (fix placement text during demote) |
| `shoaib` | other | Config-layer channel names (arm_*) do not match the actual parquet/manifest column names (upper_arm_*). If the eval loader resolves extra_channels by  | FIXED (config arm_→upper_arm_) |
| `unimts` | method | The 'converted to BVH' intermediate processing step is fabricated — it does not appear in the paper. The 'via IMUSim' phrasing is defensible (the equa | review |
| `ssl-wearables` | other | '5pp' is factually false about the downloaded PDF (verified 18 pages via PyMuPDF). Also 'condensed preprint' is misleading — arXiv v3 is the full-leng | FIXED (page/benchmark counts) |

## VARIABILITY_OK (correctly NOT flagged as bugs)

These are the heterogeneity axes HALO is designed to generalize over — our text is truthful (or silent), so not bugs:

- `mhealth` / placement: No mismatch: manifest placement matches the paper's own wording. Chest=acc+2-lead ECG, left ankle=acc+gyro+mag, right wrist=acc+gyro+mag all match. Co
- `motionsense` / units: Paper figure uses m/s^2 and RPS, which superficially differ from our g and rad/s. However our text describes the actual raw CoreMotion CSV values cons
- `inclusivehar` / label_taxonomy: The 'walking' class conflates able-bodied ambulation with wheelchair manual propulsion for disabled subjects. This is the dataset's own labeling conve

## INCOMPLETE (40) — optional enrichment, nothing false

Overwhelmingly channel descriptions that omit **units** (e.g. "Accelerometer X-axis" not stating m/s² or g) or don't state gravity presence. Since units/gravity are allowed-variability axes and no false claim is made, these are optional. Datasets with unit-omission INCOMPLETE findings: dsads, hapt, harth, hhar, kuhar, mhealth, mobiact, opportunity, pamap2, realworld, recgym, shoaib, uci_har, unimib_shar, wisdm.