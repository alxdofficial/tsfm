# Results under Evaluation Protocol v2

**Checkpoint:** `small_deep_v2_4b3fdd6/best.pt` (headline model, epoch 186) —
re-scored under protocol v2 (`EVALUATION_PROTOCOL_V2.md`) on 2026-07-02.
Source JSONs: `test_output/eval_v2/tsfm_v2_native_native.json` (native) and
`tsfm_v2_neutral_20.json` (parity). **Not comparable to v1 numbers** (different
candidate sets, scoring, metrics).

## ZS-XD — zero-shot vs each dataset's own vocabulary (macro-F1 primary)

| Dataset | **F1 [95% CI]** | bAcc | Acc | vote F1 | meanpool F1 | parity F1† |
|---|---:|---:|---:|---:|---:|---:|
| motionsense | **49.3** [46.5, 52.0] | 52.5 | 64.1 | 49.2 | 49.3 | 44.7 |
| realworld | **37.9** [34.4, 41.1] | 44.0 | 48.0 | 37.8 | 37.9 | 20.0 |
| mobiact | **12.8** [10.6, 14.4] | 17.7 | 50.0 | 13.0 | 12.8 | 17.3 |
| shoaib | **51.5** [47.9, 54.6] | 54.5 | 54.2 | 51.7 | 51.5 | 43.5 |
| opportunity | **34.2** [31.9, 36.1] | 39.6 | 49.3 | 34.7 | 34.2 | 31.8 |
| **avg main-5** | **37.2** | | | | | 31.5 |
| harth (OOD) | **19.8** [17.1, 22.3] | 27.0 | 30.4 | 19.8 | 19.8 | 20.5 |
| vtt_coniot (OOD) | **1.2** [0.8, 1.6] | 2.3 | 2.3 | 1.2 | 1.1 | 2.5 |

†parity = anti-aliased 20 Hz resample + neutral channel text (fairness row).
CIs: subject-stratified bootstrap, B=1000.

## Reading the table honestly

- **Continuity check:** main-5 avg macro-F1 37.2 under v2 ≈ the old closed-set
  macro-F1 (37.1) — expected, since for the fully label-overlapping main
  datasets v2's candidate set matches v1's closed set. The protocol change is
  not quietly inflating or deflating the model.
- **MobiAct is exposed, by design:** accuracy 50.0 but macro-F1 12.8 — the
  falls/vehicle classes are near-unrecognized and were previously hidden
  behind majority-class accuracy. This is precisely why macro-F1 is primary.
- **Pooling choice doesn't drive results:** soft / vote / meanpool agree within
  ~0.5 F1 everywhere — the pre-registered soft pooling is not a cherry-pick.
- **Capability Δ (native + rich text vs parity): +5.7 F1 on main-5 average**,
  but heterogeneous: realworld −17.9 under parity (accel-only dataset; 50→20 Hz
  + neutral text hurts most) while harth/vtt are flat-to-slightly-better under
  parity. The per-axis 2×2 decomposition is future work (M4 ablations).
- **HARTH 30.4 acc** is consistent with the post-label-fix corrected numbers
  (the fix is committed on this branch; v2's GT path is offset-free by
  construction).
- **FS-1%/10% (subject-disjoint)**: running; will be appended here. Expect
  lower than the leaky v1 few-shot numbers (76.5/85.7 acc) — that drop is the
  leakage being removed, not a regression.

## Status

- [x] ZS-XD native + parity rows (this table)
- [ ] FS-1%/10% subject-disjoint (running)
- [ ] Baseline rows (ConSE bridge + common-classes) — needs baseline adapters
      to emit softmax distributions; next milestone
- [ ] Semantic common-class pairs in `benchmark_data/eval_v2/labels/*.json`
      are PENDING_REVIEW (human sign-off required before the common-classes
      table is reported)
