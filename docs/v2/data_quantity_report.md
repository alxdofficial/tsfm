# HALO v2 — Training Data Quantity Report

_Last measured: 2026-07-09. Numbers are computed directly from the on-disk
`data/<ds>/sessions/*/data.parquet` files (`duration = num_rows / native_rate`),
not from config metadata. Regenerate with the snippet at the bottom._

## 1. Corpus at a glance

- **11 training datasets** (held-out test datasets are excluded from every number here).
- **Recording available on disk:** ~**1,090 h** across ~596 K raw sessions.
- **Recording actually trained on:** ~**180 h** (the training loader caps each dataset at
  `MAX_SESSIONS_PER_DATASET = 10000`).
- **Base training windows:** **85,367** (≈ one window per session) → **~460 K patches / epoch**.
- **Unique-information bound:** ~180 h · ~85 K windows · **~250 subjects** · 11 datasets.
  Augmentation multiplies *views*, not information (see §5).

## 2. Per-dataset table

| Dataset | Native Hz | Sessions (disk) | Trained (≤10k) | Avail. h | Trained h | Patch (default) |
|---|--:|--:|--:|--:|--:|--:|
| uci_har     | 50  | 10,299  | 10,000 | 7.3   | 7.1  | 1.00 s |
| hhar        | 50  | 318,543 | 10,000 | 226.5 | 7.1  | 1.00 s |
| pamap2      | 100 | 4,522   | 4,522  | 14.9  | 14.9 | 2.00 s |
| wisdm       | 20  | 164,623 | 10,000 | 572.4 | 34.8 | 1.50 s |
| dsads       | 25  | 9,120   | 9,120  | 12.7  | 12.7 | 2.00 s |
| kuhar       | 100 | 17,374  | 10,000 | 40.2  | 23.1 | 1.50 s |
| unimib_shar | 50  | 11,771  | 10,000 | 9.9   | 8.4  | 1.00 s |
| hapt        | 50  | 2,546   | 2,546  | 4.5   | 4.5  | 1.25 s |
| mhealth     | 50  | 2,029   | 2,029  | 4.8   | 4.8  | 1.50 s |
| recgym      | 20  | 7,150   | 7,150  | 28.3  | 28.3 | 1.50 s |
| capture24   | 100 | 48,518  | 10,000 | 168.1 | 34.6 | 1.50 s |
| **TOTAL**   |     | **~596 K** | **85,367** | **~1,090** | **~180** | |

## 3. Session-length distribution (seconds)

| Dataset | min | median | max | mean | shape |
|---|--:|--:|--:|--:|---|
| uci_har     | 2.56 | 2.56  | 2.56  | 2.56  | fixed frame |
| hhar        | 2.56 | 2.56  | 2.56  | 2.56  | fixed frame |
| unimib_shar | 3.02 | 3.02  | 3.02  | 3.02  | fixed frame |
| dsads       | 5.00 | 5.00  | 5.00  | 5.00  | fixed frame |
| hapt        | 1.48 | 5.32  | 19.44 | 6.30  | contiguous bout |
| mhealth     | 2.00 | 8.16  | 19.78 | 8.56  | contiguous bout |
| kuhar       | 2.00 | 6.94  | 59.87 | 8.33  | contiguous bout |
| pamap2      | 2.00 | 10.02 | 59.78 | 11.88 | contiguous bout |
| capture24   | 2.00 | 11.16 | 29.99 | 12.44 | contiguous bout |
| recgym      | 2.00 | 12.90 | 44.95 | 14.25 | contiguous bout |
| wisdm       | 2.00 | 10.35 | 59.95 | 12.75 | contiguous bout |

_min/max for hhar & wisdm are from an 8,000-session stride sample; the other 9 are exact._

**Two corpus shapes.** uci_har, hhar, unimib_shar, dsads are **pre-windowed fixed frames**
(min = median = max). uci_har/hhar are additionally the classic **50%-overlap** UCI frames, so
their recording hours overcount true unique wall-clock by ~2×. The other 7 are **contiguous
activity bouts** (min ~2 s → max 30–60 s). The ~2 s floor is load-bearing: patch sizes were
chosen so `max_patch < min_session`, guaranteeing ≥1 valid patch per session.

## 4. Patch durations — what we picked and why

**Default patch size** (used at evaluation, fixed, no per-dataset tuning): see §2.
Rationale: fixed-frame sets use **1.0 s** (→ ~2–3 patches per 2.56–3 s frame); short-bout sets
use **1.25–1.5 s**; long-bout sets (pamap2, dsads) use **2.0 s**. Every default satisfies
`patch < min_session`.

**Patch-size augmentation** (training only; sampled per step, then fixed 1.0 s at eval):

| Datasets | Sampled patch sizes (s) |
|---|---|
| uci_har, hhar, unimib_shar, hapt | {0.75, 1.0, 1.25} |
| mhealth, wisdm, kuhar, recgym, capture24 | {1.0, 1.25, 1.5, 1.75} |
| pamap2 | {1.0, 1.5, 2.0} |
| dsads | {1.5, 2.0, 2.5} |

**Patches per window.** A window = one session, tokenized into `ceil(session / patch)` patches,
capped at `MAX_PATCHES_PER_SAMPLE = 48`. In practice: fixed frames ≈ 3 patches; contiguous bouts
have a **median of ~5–9** and a **30–60 s tail reaching ~20–40** (so long sessions are barely
truncated). A window is therefore **not** a fixed size — the encoder sees **1–48 patches**
depending on the source.

Per-dataset base windows / patches-per-epoch (at the 10k cap):

| Dataset | Windows | Patches/epoch |
|---|--:|--:|
| uci_har | 10,000 | 30,000 |
| hhar | 10,000 | 30,000 |
| pamap2 | 4,522 | 27,132 |
| wisdm | 10,000 | 80,000 |
| dsads | 9,120 | 27,360 |
| kuhar | 10,000 | 60,000 |
| unimib_shar | 10,000 | 30,000 |
| hapt | 2,546 | 12,730 |
| mhealth | 2,029 | 12,174 |
| recgym | 7,150 | 71,500 |
| capture24 | 10,000 | 80,000 |
| **TOTAL** | **85,367** | **460,896** |

## 5. Effective training exposure (with augmentation)

Training runs `EPOCHS = 100`. Each epoch re-augments every window, so:

- **Window-views over training:** 85,367 × 100 = **~8.54 M**
- **Patch-views over training:** 460,896 × 100 = **~46.1 M**

Each view is a fresh stochastic realization from the augmentation manifold (below). This
multiplies the *number of distinct views* the model sees; it does **not** add new unique
information (no new subjects/activities/recordings). The honest statement:
**~180 h of real recording (~85 K windows), seen as ~8.5 M augmented window-views over 100 epochs.**

**Augmentations enabled (`AugmentationConfig.default_v2`):**

| Group | Augmentations |
|---|---|
| Signal | jitter, scale |
| Physics | gravity (P1), yaw rotation (P2), rate resample ±5% (P3), channel dropout (P4) |
| Text | label synonym+template (p≈0.8), channel-description paraphrase, channel-description dropout |
| Temporal | patch-size augmentation (3–4 options/dataset, §4) |

Signal / gravity / yaw / ±5%-rate are **continuous** (effectively unbounded distinct realizations
per window); patch-size is 3–4 discrete; text is ~tens of combinations. So the reachable
augmented versions of a single window is effectively continuous — the model samples 100 of them.

## 6. Caveats & things to reconcile

1. **Two different subsampling caps exist.** HALO training uses
   `MAX_SESSIONS_PER_DATASET = 10000` (→ ~180 h). `benchmark_data/dataset_config.json` uses a
   different cap (hhar/wisdm 15,000, capture24 20,000 → ~255 h) for the baseline pipeline.
   **Decide which the v2 run should use** — it directly sets how much capture24/wisdm/hhar the
   model actually trains on.
2. **Overlap overcount.** uci_har & hapt (and the UCI-family fixed frames) overlap ~50%, so their
   "hours" overstate unique wall-clock ~2×.
3. **capture24 sessions are pre-segmented bouts** (median 11 s, max 30 s), not the raw ~24 h
   free-living streams — relevant when citing "free-living scale."
4. **On-disk vs config session counts diverge for hhar** (disk 318,543 vs
   `dataset_config.json` 146,557; pamap2 disk 4,522 vs 4,377). Worth investigating for a possible
   double-count / stale config before the retrain.

## 7. Where these are configured

- `training_scripts/human_activity_recognition/semantic_alignment_train.py`:
  `PATCH_SIZE_PER_DATASET`, `PATCH_SIZE_RANGE_PER_DATASET`, `MAX_PATCHES_PER_SAMPLE = 48`,
  `MAX_SESSIONS_PER_DATASET = 10000`, `EPOCHS = 100`, aug preset (`default_v2`).
- `datasets/imu_pretraining_dataset/augmentations.py`: `AugmentationConfig.default_v2`.
- `benchmark_data/dataset_config.json`: the alternate subsampling caps (baseline pipeline).

## 8. Training target (~400 h) — IMPLEMENTED

**Target: ~400 h of *original* trainable recording**, reached from the current 11 datasets by
raising the per-dataset session caps — no new datasets required. Implemented as a per-dataset
HOURS budget in `semantic_alignment_train.py` (`TRAIN_HOURS_PER_DATASET` → `MAX_SESSIONS_PER_DATASET`,
a per-dataset dict the loader now understands).

Scripted data is retained deliberately — nothing is wrong with it; the goal is simply more hours.
All small/medium sets and **all** free-living capture24 are used in full; the two large scripted
reservoirs (hhar, wisdm) are capped. capture24 lands at ~42% of the mix just by using all of it —
lifting the free-living share from ~15% → ~42% without dropping any scripted data.

| Dataset | Cap (sessions) | Used | Hours |
|---|---|--:|--:|
| capture24 (free-living) | all | 48,518 | 167.7 |
| wisdm | 24,000 | 24,000 | 85.0 |
| kuhar | all | 17,374 | 40.2 |
| recgym | all | 7,150 | 28.3 |
| hhar | 33,750 | 33,750 | 24.0 |
| pamap2 | all | 4,522 | 14.9 |
| dsads | all | 9,120 | 12.7 |
| unimib_shar | all | 11,771 | 9.9 |
| uci_har | all | 10,299 | 7.3 |
| mhealth | all | 2,029 | 4.8 |
| hapt | all | 2,546 | 4.5 |
| **TOTAL** | | **~171,079** | **~399 h** |

To adjust: edit `TRAIN_HOURS_PER_DATASET` (None = all sessions). `TSFM_SESSION_CAP=<int>` overrides
with a flat global cap for smoke tests.

_Scripted vs free-living: scripted = subjects perform a prescribed activity list on cue in a
controlled setting (clean labels, unrealistic); free-living = continuous recording during normal
daily life, labeled post-hoc (messy labels, realistic — what deployment faces). Our 10 lab sets
are scripted; capture24 (and the NHANES/ExtraSensory/Nymeria/PAAWS adds) are free-living._

**Still to consider at retrain time:**
- ~171k windows is ~2× the old 85k. Consider **~45–50 epochs instead of 100** — keeps total
  augmented exposure (~8 M window-views) roughly constant while the model sees ~2× more
  *distinct* data (better generalization, ~flat compute).
- Reconcile with the baseline pipeline's separate caps in `dataset_config.json` (§6).
- Optional further free-living adds (NHANES / ExtraSensory) would push free-living past 50% —
  deferred, not required for the 400 h target.

**On the augmentation multiplier:** treat augmentation as robustness/coverage, NOT as an hours
multiplier. Report the **real ~400 h** as the headline; augmentation multiplies *views*, not
unique information (a 1.0 s vs 1.25 s crop of the same window shares ~80% of its content). If an
"effective" figure is wanted, phrase it as *views* ("~400 h presented as ~N M augmented
window-views"), never as equivalent hours.

## 9. How to regenerate

```python
import glob, os, statistics as st
import pyarrow.parquet as pq
RATES = {'uci_har':50,'hhar':50,'pamap2':100,'wisdm':20,'dsads':25,'kuhar':100,
         'unimib_shar':50,'hapt':50,'mhealth':50,'recgym':20,'capture24':100}
for ds, rate in RATES.items():
    dirs = sorted(glob.glob(f"data/{ds}/sessions/*/"))
    samp = dirs[::max(1, len(dirs)//8000)] if len(dirs) > 50000 else dirs
    durs = sorted(pq.ParquetFile(os.path.join(d,'data.parquet')).metadata.num_rows / rate
                  for d in samp)
    print(ds, len(dirs), round(durs[0],2), round(st.median(durs),2),
          round(durs[-1],2), round(sum(durs)/len(durs),2))
```
