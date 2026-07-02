# Results under Evaluation Protocol v2

**Checkpoint:** `small_deep_v2_4b3fdd6/best.pt` (headline model, epoch 186) —
re-scored under protocol v2 (`EVALUATION_PROTOCOL_V2.md`) on 2026-07-02.
Source JSONs: `test_output/eval_v2/tsfm_v2_native_native.json` (native) and
`tsfm_v2_neutral_20.json` (parity). **Not comparable to v1 numbers** (different
candidate sets, scoring, metrics).

**Benchmark composition (decided 2026-07-02):** one flat test set of **6
datasets** — motionsense, realworld, mobiact, shoaib, opportunity, **harth**.
There is no "severe-OOD" tier anymore. VTT-ConIoT was dropped from the
benchmark: with ~50% of its construction-domain labels having no training
equivalent, every model's zero-shot score there measured label coverage, not
recognition capability. (Its last scores under this checkpoint, for the
record: ZS-XD F1 1.2 native / 2.5 parity.)

## ZS-XD — zero-shot vs each dataset's own vocabulary (macro-F1 primary)

| Dataset | **F1 [95% CI]** | bAcc | Acc | vote F1 | meanpool F1 | parity F1† |
|---|---:|---:|---:|---:|---:|---:|
| motionsense | **49.3** [46.5, 52.0] | 52.5 | 64.1 | 49.2 | 49.3 | 44.7 |
| realworld | **37.9** [34.4, 41.1] | 44.0 | 48.0 | 37.8 | 37.9 | 20.0 |
| mobiact | **12.8** [10.6, 14.4] | 17.7 | 50.0 | 13.0 | 12.8 | 17.3 |
| shoaib | **51.5** [47.9, 54.6] | 54.5 | 54.2 | 51.7 | 51.5 | 43.5 |
| opportunity | **34.2** [31.9, 36.1] | 39.6 | 49.3 | 34.7 | 34.2 | 31.8 |
| harth | **19.8** [17.1, 22.3] | 27.0 | 30.4 | 19.8 | 19.8 | 20.5 |
| **average (6)** | **34.3** | | | | | **29.6** |

†parity = anti-aliased 20 Hz resample + neutral channel text (fairness row).
CIs: subject-stratified bootstrap, B=1000.

## FS — subject-disjoint few-shot (running; appended as results land)

| Dataset | FS-1% F1 | FS-1% Acc | FS-10% F1 | FS-10% Acc |
|---|---:|---:|---:|---:|
| harth | 36.0 | 41.6 | — | — |
| *(others)* | *running* | | | |

Early signal: harth FS-1% accuracy 41.6 vs the leaky v1 number 62.0 — the
~20-point drop is the subject leakage being removed, not a regression.

## Reading the table honestly

- **Continuity check:** the former "main-5" average macro-F1 (37.2) under v2 ≈
  the old closed-set macro-F1 (37.1) — the protocol swap is not quietly moving
  numbers. The headline average is now the flat 6-dataset mean: **34.3**.
- **MobiAct is exposed, by design:** accuracy 50.0 but macro-F1 12.8 — the
  falls/vehicle classes are near-unrecognized and were previously hidden
  behind majority-class accuracy. This is precisely why macro-F1 is primary.
- **HARTH is now just a test dataset** — back-mounted accelerometer, real
  distribution shift, F1 19.8. Reported in the same table as everything else,
  with the same rules.
- **Pooling choice doesn't drive results:** soft / vote / meanpool agree within
  ~0.5 F1 everywhere — the pre-registered soft pooling is not a cherry-pick.
- **Capability Δ (native + rich text vs parity): +4.7 F1 on the 6-dataset
  average** (34.3 vs 29.6), but heterogeneous: realworld −17.9 under parity
  (accel-only dataset) while harth/mobiact are flat-to-slightly-better under
  parity. The per-axis 2×2 decomposition is future work (M4 ablations).

## Status

- [x] ZS-XD native + parity rows (6-dataset benchmark)
- [ ] FS-1%/10% subject-disjoint (running; harth FS-1% landed)
- [ ] Baseline rows (ConSE bridge + common-classes) — needs baseline adapters
      to emit softmax distributions; next milestone
- [ ] Semantic common-class pairs in `benchmark_data/eval_v2/labels/*.json`
      are PENDING_REVIEW (human sign-off required before the common-classes
      table is reported)
