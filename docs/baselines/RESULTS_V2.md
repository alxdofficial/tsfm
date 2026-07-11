# Results under Evaluation Protocol v2

> ⚠️ **NUMBERS STALE — need re-scoring.** This table was scored 2026-07-02, before the test set
> changed (`opportunity` demoted, `inclusivehar` added) and before the ConSE vocab grew to 94
> labels / 11 train datasets and the baseline set grew to 6. Treat every number here as
> provisional; re-run `run_baselines_v2.py` + `evaluate_tsfm_v2.py` and `assemble_v2_table.py`
> against the current config before reporting. Baseline roster: [`BASELINES_OVERVIEW.md`](BASELINES_OVERVIEW.md).

> **Stale pending regeneration (2026-07-10):** These numbers predate the
> current active benchmark (`motionsense`, `realworld`, `mobiact`, `shoaib`,
> `harth`, `inclusivehar`) and the Capture24 training-set addition. They are
> retained only as historical V2 run output until HALO and baseline rows are
> regenerated from `test_output/eval_v2/*.json`. Do not cite the aggregate
> tables below as current results.

**Checkpoint:** `small_deep_v2_4b3fdd6/best.pt` (headline model, epoch 186) —
re-scored under protocol v2 (`EVALUATION_PROTOCOL_V2.md`) on 2026-07-02.
Source JSONs: `test_output/eval_v2/tsfm_v2_native_native.json` (native) and
`tsfm_v2_neutral_20.json` (parity). **Not comparable to v1 numbers** (different
candidate sets, scoring, metrics).

**Benchmark composition (updated 2026-07-10):** one flat active test set of **6
datasets** — motionsense, realworld, mobiact, shoaib, **harth**, **inclusivehar**.
There is no "severe-OOD" tier anymore. VTT-ConIoT was dropped from the
benchmark: with ~50% of its construction-domain labels having no training
equivalent, every model's zero-shot score there measured label coverage, not
recognition capability. Opportunity is retained as an appendix-only dataset
because its 4-subject structure gives degenerate CIs.

## ZS-XD — HALO, zero-shot vs each dataset's own vocabulary (macro-F1 primary)

*Numbers refreshed 2026-07-02 after the M0 debug-sweep fixes (commit `753925f`):
macro-F1 now averages over GT∪predicted classes, so HARTH's false positives
into its 4 zero-window candidate classes (`cycling_sit/stand`,
`transport_sit/stand`) are charged — HARTH F1 19.8 → 13.2; no other dataset
moved (all their candidates appear in GT). Averages: 34.3 → **33.2**.*

| Dataset | **F1 [95% CI]** | bAcc | Acc | vote F1 | meanpool F1 | parity F1† |
|---|---:|---:|---:|---:|---:|---:|
| motionsense | **49.3** [46.5, 52.0] | 52.5 | 64.1 | 49.2 | 49.3 | 44.7 |
| realworld | **37.9** [34.4, 41.1] | 44.0 | 48.0 | 37.8 | 37.9 | 20.0 |
| mobiact | **12.8** [10.6, 14.4] | 17.7 | 50.0 | 13.0 | 12.8 | 17.3 |
| shoaib | **51.5** [47.9, 54.6] | 54.5 | 54.2 | 51.7 | 51.5 | 43.5 |
| opportunity | **34.2** [31.9, 36.1] | 39.6 | 49.3 | 34.7 | 34.2 | 31.8 |
| harth | **13.2** [11.4, 14.9] | 27.0 | 30.4 | 13.2 | 13.2 | 20.5 |
| **average (6)** | **33.2** | | | | | **29.6** |

†parity = anti-aliased 20 Hz resample + neutral channel text (fairness row).
CIs: subject-stratified bootstrap, B=1000.

## ZS-XD — HALO vs baselines (macro-F1)

> **Baseline heterogeneity flexibility** (how much each model can flex on sampling rate,
> channel count, window/session length, modality, placement, streaming, open-vocab) is
> documented in [`baseline_flexibility.md`](baseline_flexibility.md). Each baseline is run in the
> exact input format it was built for (LiMU-BERT/CrossHAR 20 Hz/6-ch, ssl-wearables 30 Hz/3-ch,
> UniMTS resampled/skeleton, …); HALO runs native. This is the fairness basis for the per-model
> preprocessing.

All models scored under the SAME v2 rule: zero-shot, argmax over each dataset's
own label strings, exact match, macro-F1 over GT∪predicted. Text-aligned models
(HALO, LanHAR) encode `L_D` directly; **†** closed-vocab classifiers are bridged
with ConSE (softmax over the cached baseline label mapping → convex combination of frozen-SBERT
label embeddings, top-T=10 → argmax over `L_D`). See `EVALUATION_PROTOCOL_V2.md`.

| Model | tier | motionsense | realworld | mobiact | shoaib | opportunity | harth | **avg** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **HALO (Small-Deep)** | text-aligned | **49.3** | **37.9** | 12.8 | **51.5** | 34.2 | 13.2 | **33.2** |
| HALO (parity, 20 Hz) | text-aligned | 44.7 | 20.0 | 17.3 | 43.5 | 31.8 | 20.5 | 29.6 |
| CrossHAR † | conse | 30.3 | 27.3 | **13.6** | 22.1 | **48.5** | **20.8** | 27.1 |
| LiMU-BERT † | conse | 43.0 | 20.5 | 8.0 | 27.6 | 8.3 | 2.4 | 18.3 |
| UniMTS | cosine | *planned (released weights)* | | | | | | |
| ssl-wearables † | conse | *planned (released weights)* | | | | | | |

*Baseline set changed in the V2 cleanup: **MOMENT, LanHAR, LLaSA dropped** (slow /
weak / undeployable); **UniMTS + ssl-wearables to be added** as adapters. For the
record, the dropped baselines' last v2 averages were LanHAR 19.3, LiMU-BERT-tier
MOMENT (partial), LLaSA ~near-random.*

**Reading it:**
- **HALO leads the 6-dataset average (33.2)** and wins 3/6 datasets outright
  (motionsense, realworld, shoaib) by wide margins vs the next-best model.
- **CrossHAR is the strongest baseline (27.1 avg)** and actually beats HALO on
  3 datasets — opportunity (48.5 vs 34.2), harth (20.8 vs 13.2), mobiact (13.6
  vs 12.8). HALO does **not** dominate everywhere; the honest picture is a lead
  on average and on the locomotion-rich sets, losses on the placement-shifted /
  fall-heavy sets. This is exactly what macro-F1 + a leakage-free protocol are
  supposed to surface.
- **Even HALO's parity row (29.6)** — stripped of native rate and channel text,
  i.e. the same inputs the baselines get — still beats every baseline's average,
  which is the architecture-only claim.
- CrossHAR/LiMU-BERT/MOMENT reachability was ≥0.83 on every dataset (the ConSE
  bridge could reach almost all candidate classes; mobiact/harth's `car_step`/
  cycling/transport classes are the structural gaps), so the closed-vocab
  numbers are not artifacts of an unreachable label space.
- **Baselines evaluated at their native 20 Hz** with the cached zero-shot
  classifiers from the v1 pipeline (no retraining) — LanHAR text-aligned via
  SciBERT, the others ConSE-bridged.

## FS — HALO subject-disjoint few-shot (refreshed 2026-07-02, post-fix)

| Dataset | FS-1% F1 | FS-1% Acc | FS-10% F1 | FS-10% Acc | test subj | v1 leaky 1% Acc |
|---|---:|---:|---:|---:|:---:|---:|
| motionsense | 77.2 | 76.6 | 85.9 | 86.8 | 3 | 88.6 |
| realworld | 68.1 | 68.2 | 67.6 | 68.9 | 2 | 75.1 |
| mobiact | 28.3 | 70.9 | 27.7 | 65.3 | 3 | 65.3 |
| shoaib | 76.1 | 76.4 | 87.0 | 87.5 | 1 (deg) | 81.6 |
| opportunity | 60.1 | 57.4 | 70.0 | 69.8 | 1 (deg) | 72.0 |
| harth | 30.7 | 39.3 | 61.4 | 64.7 | 2 | 62.0 |
| **avg (6)** | **56.8** | **64.8** | **66.6** | **73.8** | | |

(Numbers moved slightly vs the first FS run because the balanced-subsample
water-filling and macro-F1 union-label fixes changed the FT train set and
scoring; "test subj" is the count behind each CI, with `deg` = degenerate
single-subject CI, now flagged rather than shown as a fake `[x,x]` interval.)

**The leakage removal, quantified:** subject-disjoint FS-1% accuracy averages
~12 points below the v1 random-window splits — squarely in the 10–15 pp range
the LOSO literature predicts (Gholamiangonabadi et al.; Rehman et al.). This
drop is the correction, not a regression; it is the number the paper must
report. ZS-XD rows are unaffected (no split needed — the full cohort is scored).

**Known limitation (flagged, filed as task #7):** an 80/10/10 *subject* split
leaves only 1 test subject on shoaib/opportunity (degenerate CI) and 2 on
realworld/harth. Fix = GroupKFold rotation, mean±std across folds (~5× FT
compute). Also visible by design: mobiact FS-1% acc 70.9 vs F1 28.3 — few
balanced samples can't learn the fall/vehicle classes; accuracy alone hides it.

## Reading the table honestly

- **Continuity check:** the former "main-5" average macro-F1 (37.2) under v2 ≈
  the old closed-set macro-F1 (37.1) — the protocol swap is not quietly moving
  numbers. The headline average is the flat 6-dataset mean **33.2** (post-fix).
- **MobiAct is exposed, by design:** accuracy 50.0 but macro-F1 12.8 — the
  falls/vehicle classes are near-unrecognized and were previously hidden
  behind majority-class accuracy. This is precisely why macro-F1 is primary.
- **HARTH is now just a test dataset** — back-mounted accelerometer, real
  distribution shift, F1 13.2. Reported in the same table as everything else,
  with the same rules; CrossHAR (20.8) beats HALO here.
- **Pooling choice doesn't drive results:** soft / vote / meanpool agree within
  ~0.5 F1 everywhere — the pre-registered soft pooling is not a cherry-pick.
- **Capability Δ (native + rich text vs parity): +3.6 F1 on the 6-dataset
  average** (33.2 vs 29.6), but heterogeneous: realworld −17.9 under parity
  (accel-only dataset) while harth/mobiact are flat-to-slightly-better under
  parity. The per-axis 2×2 decomposition is future work (M4 ablations).

## Status

- [x] ZS-XD native + parity rows (6-dataset benchmark), post debug-sweep fixes
- [x] FS-1%/10% subject-disjoint (complete; CIs flagged where degenerate)
- [x] Baseline ZS-XD rows: LanHAR (cosine), CrossHAR / LiMU-BERT (ConSE)
- [ ] MOMENT ZS-XD row — running on CPU (GPU busy with an unrelated job); will
      be filled in on completion
- [ ] LLaSA ZS-XD row — 7B generative, not yet run (`--include-llasa`)
- [ ] Baseline subject-disjoint FS-1%/10% — larger follow-up (each baseline
      fine-tunes per dataset; see fewshot notes in the adapter recipes)
- [ ] Semantic common-class pairs in `benchmark_data/eval_v2/labels/*.json`
      are PENDING_REVIEW (human sign-off before the common-classes table)
- [ ] GroupKFold rotation for non-degenerate FS CIs (task #7)
