# Pre-Ablation Debug Sweep — Findings (read-only, nothing changed)

Source: 27-agent read-only workflow (`wf_7016151c-73c`) over all model + baseline + shared code, each
high/critical finding adversarially verified. **3 critical + 1 high** confirmed. No code was modified.

> **Principle applied (per Alex):** make *our* model maximally correct; for baselines, do NOT fix genuine
> structural limitations, but DO fix anywhere *our harness* misrepresents them. Several findings are
> exactly the latter (we are unfairly hurting LLaSA and LanHAR) — fixing those is required for fairness.

---

## ⚠️ Implications for results we've ALREADY reported

| our result | affected? | why |
|---|---|---|
| Headline 5-main (EXP-X, EXP-P2, EXP-F, EXP-P5) | **SAFE** | only **HARTH** has stored label `min>0` (codes [2..8,11]); all 5 main + VTT have min=0, so the label bug is a no-op there. Verified-reproduce checks still hold. |
| **EXP-P3 HARTH numbers + per-activity table** | **AFFECTED** | computed via the buggy `get_window_labels` → HARTH GT shifted by −2, so the **2.0% ZS** and the per-activity confusions are mis-scored. **Likely artificially LOW** (correct predictions scored wrong). Needs re-run after the fix. |
| EXP-P3 gravity root-cause (0.975 centroid, alignment inversion, recovery curve, signal stats) | **INTACT** | these are label-independent (embeddings/signal); supervised recovery is invariant under the permutation. The *why* survives; the *HARTH ZS magnitude* and *which activities fail* must be re-validated. |
| HARTH column in the deployed JSON (all models) | **AFFECTED** | same bug; the whole HARTH ZS column (every model) is mis-scored and must be re-run. |
| In-distribution val_acc (~85%) | **INFLATED** | cache-split bug leaks ~73% of val into train on a fresh run. Headline ZS (held-out datasets) is unaffected. |

**Net:** the fair, scoring-robust win story (EXP-X/F/P5, all 5-main) is untouched. The **HARTH severe-OOD
narrative needs re-validation** — fixing the label bug may raise HARTH ZS above 2% and reshuffle the
per-activity story (the gravity mechanism stays).

---

## The findings (consolidated, ranked)

### 🔴 CRITICAL #1 — HARTH label shift corrupts GT for ALL models
`get_window_labels` (`evaluate_tsfm.py:347-357`, dup'd in `evaluate_moment/limubert/crosshar`, consumed by
`harth_analysis.py`) does `t=min(labels); labels-=t`. Stored labels are already canonical 0-based indices
into `sorted(activities)`; the subtract is a no-op **only when index-0 is present**. **HARTH is the only
dataset (of 17) with min>0** (present codes [2,3,4,5,6,7,8,11] → shift −2: lying→cycling_sit, walking→
transport_sit…). Predictions come from an offset-independent space, so GT shifts while predictions don't →
genuine mis-scoring of the **entire HARTH ZS column (open/closed, pool/MV) for every model**, the HARTH
supervised per-class metrics, and the EXP-P3 case study. **Fix:** drop the `t=min` subtraction (index raw
stored code); apply across all 5 eval scripts + the training-side SVM label path (`evaluate_moment.py:638`);
re-run HARTH for every model; regenerate harth figures; assert `max(label) < len(activities)` per dataset.

### 🔴 CRITICAL #2 — Cache-state-dependent splits → train/val leakage
`multi_dataset_loader.py:204-339`. On cache-MISS, `random.shuffle` runs for the 5 datasets exceeding
`MAX_SESSIONS_PER_DATASET=10000`; on cache-HIT it doesn't → global RNG state reaching the split shuffle
(L324) differs → **same config yields different train/val/test depending on whether `data/.cache/` exists**
(~30% of sessions move). **Worse:** train (cache-miss, creates file) and val (then cache-hit) are built as
separate instances → **~73% val/train overlap = leakage on the first run.** **Fix:** re-seed
`random.seed(self.seed)` right before the split shuffle + use a dedicated `random.Random(self.seed)` for the
truncation (or hash-based splits); **delete stale `data/.cache/`**. Verify identical per-split session ids
with/without cache.

### 🔴 CRITICAL #3 (fairness) — LLaSA accelerometer not ÷9.8
`evaluate_llasa.py:159-214`. LLaSA's own pipeline requires `acc/9.8` (its `load_image` divides by 9.8;
`evaluate_limubert.py` does normalize). 6/8 datasets are m/s²-scale → 3/6 channels fed ~9.8× too large into
a LayerNorm encoder → every LLaSA embedding the LLM sees is OOD → **unfairly cripples LLaSA across all
datasets and inflates our relative advantage.** **Fix:** divide acc by 9.8 **per-dataset** (hapt/harth are
already g-scale — don't blindly divide; fix the per-dataset unit inconsistency).

### 🟠 HIGH — Multi-seed ablation impossible (`SEED=42` hardcoded)
`semantic_alignment_train.py:134,220,1849,2006…`. No `TSFM_SEED` env override anywhere; numpy unseeded;
seed not recorded. EXP-P6 "report variance over seeds" cannot vary the seed without source edits — all
"seeds" collapse to identical init/order/augmentation, so reported mean±std would be artificially tiny and
**overstate our model's stability.** **Fix:** `SEED=int(os.environ.get('TSFM_SEED','42'))`; seed
random/np/torch together in `main()`; record in `hyperparameters.json`; add a seed loop to `run_ablations.sh`.

### 🟠 HIGH (fairness) — LanHAR HARTH label shift (un-restored)
`evaluate_lanhar.py:1300-1308` + `generate_lanhar_descriptions.py`. Same `t`-subtraction but never restored
(LLaSA's version adds it back) → HARTH ZS labels permuted → LanHAR unfairly penalized on HARTH ZS (supervised
is self-consistent). **Fix:** return `window_labels + t` (or drop the subtraction); regenerate HARTH
descriptions; re-run LanHAR HARTH ZS.

### 🟡 Ablation-infra blockers (must wire before the campaign)
- **MODEL_SIZE frozen at import, no override** (`train.py:166-208`) → P7 tokenizer/kernel/resample arms would
  silently train the default `small_deep` arch. Add `TSFM_MODEL_SIZE`; resolve config inside `main()`.
- **Queue-mode / resample-mode have NO flag plumbing**; **'semantic queue' is not implementable** —
  `memory_bank.py` stores no label/group info, so hard-queue ≡ semantic-queue (a meaningless no-difference
  row); queue also bypassed under GradCache. → Either extend `MemoryBank` to store group ids, or drop the
  semantic cell. (Blocks EXP-P1.)
- `TSFM_CHECKPOINT` default is the **legacy `small_v1` CNN** checkpoint → any eval that forgets to set it
  silently scores a stale, structurally-different model. Make it mandatory or default to `small_deep_v2`.
- Pin & log `NUM_WORKERS`, `prefetch`, GradCache on/off, `torch.compile` state, `ACTIVE_LABEL_GROUPS` across
  the whole series (each silently shifts effective-LR / sampling / numerics between arms).
- Cached baseline classifiers (`moment_zs_svm.pkl`, `crosshar_zs_transformer.pt`, `limubert_zs_gru.pt`,
  `lanhar_model.pt`) key only on filename → won't invalidate on the label fix. **Delete before re-running.**

### 🟡 Shared-scoring / fairness asymmetries
- Closed-set is scored group-match for classifier baselines but exact for text-aligned (TSFM/LanHAR) →
  on collapse datasets this **helps the classifier baselines** (self-penalizing for us) but is non-apples.
  Make symmetric or aggregate baseline logits to test labels.
- Grouped open-set collapses some datasets to far fewer effective classes (harth 8→? , realdisp 33→7) →
  inflated, not comparable to exact-closed; report effective class count per dataset.

### 🟢 Lower-severity our-model notes (mostly dormant, relevant to specific ablation arms)
stale `96` patch-size literals in `losses.py` docstrings; non-GradCache effective-LR fluctuation; missing
all-masked-channel NaN guard in `CrossChannelFusion` (would bite a channel-dropout arm); `ChannelTextFusion`
writes into pad channels; val loss uses learnable (not frozen) targets; `normalize 'none'` device mismatch;
`ProjectionHead` ignores its dropout arg; per-patch mode leaves dead untrained pooling params + misleading
grad-norm telemetry; positional-encoding SBERT fallback returns identical random vectors per width (silent
if sentence-transformers missing on a node).

### ✅ Confirmed FAIR / not bugs (do not "fix")
CrossHAR & LiMU-BERT encoder/checkpoint reimplementations verified byte-faithful to upstream; RealWorld
gyro-zero is a genuine acc-only dataset property (identical for all models); LanHAR mean-of-prototypes +
simplified gravity alignment are faithful design choices applied uniformly; spectral-branch FFT without 1/N
scaling is a no-op under the following LayerNorm (by design).

---

## Recommended sequence (decisions for Alex)
1. **Fix the 2 shared CRITICAL bugs** (HARTH label shift; cache-split/leakage) + delete `data/.cache/`.
   ✅ **DONE.** Re-ran HARTH for HALO + all 5 baselines. **HARTH ZS rose 2.0%→29.1% (HALO)** — far more
   than "above 2%": this **retracts** the gravity mechanism (it was the label bug, not gravity) and
   reframes severe-OOD as HARTH=graceful-degradation vs VTT=genuine-collapse. See `rebuttal_plan.md`
   "Severe-OOD framing" and `RESULTS.md` HARTH table.
2. **Fix the baseline fairness bugs we owe** (LLaSA ÷9.8 per-dataset; LanHAR HARTH labels) → re-run those.
   Delete cached baseline classifiers first.
3. **Wire the ablation infra** (TSFM_SEED, TSFM_MODEL_SIZE→runtime config, queue/resample/loss flags +
   hyperparameters.json recording; decide if 'semantic queue' is implementable or dropped); pin+log workers/
   GradCache/compile/grouping; make TSFM_CHECKPOINT mandatory.
4. **Then** run P1/P6/P7 on a clean, reproducible, fair baseline.

All fixes are on a fresh branch off `pre-ablation-baseline`; nothing changed yet.

---

## Merge with Alex's independent read-only sweep

The two sweeps overlap heavily (mutual corroboration) and each caught things the other missed.

**Corroborated by BOTH (high confidence):**
- Per-patch standard-path gradient scaling uses session count, not valid-patch count (`train.py:888` vs
  `903/924`); GradCache is more correct but off by default (`:232`). *(Alex sharpens my "effective-LR
  fluctuation" — the scale confound is specifically session-vs-patch in per-patch mode.)*
- Random session-level splits, not subject/group-aware (`multi_dataset_loader.py:321`). *(My sweep found
  the more severe variant on top: cache-state-dependent splits → first-run val/train leakage.)*
- Per-patch mode bypasses the temporal attention/pooling head (`semantic_alignment.py:454` vs `459`) —
  Medium's extra capacity may not help cross-patch temporal structure.
- `MODEL_SIZE` hardcoded `small_deep` (`train.py:166`) — Medium/ablation arms need source edits.
- Memory-bank queued negatives get zero soft-target mass (`semantic_loss.py:164`) — synonyms in the queue
  treated as pure negatives (this is also Reviewer A2's concern + my "semantic-queue not implementable").
- Supervised splits are random window-level, not subject-aware (affects TSFM + baselines alike).

**NEW from Alex's sweep (folded in — mostly latent / infra, verify before relevant arms):**
- A1. Eval patch-size: unseen-in-training monitoring (`train.py:2106`) doesn't pass
  `PATCH_SIZE_PER_DATASET`, and final eval hardcodes 1.0s (`evaluate_tsfm.py:79`) → training, monitoring,
  and final eval are not directly comparable.
- A2. Multi-prototype soft targets still use **learnable** (not frozen) text embeddings
  (`semantic_loss.py:211` vs single-proto frozen path `:344`) — latent (configs use 1 prototype), but
  would bite a multi-prototype arm.
- A3. Memory-bank warmup accounting is session-based while updates are patch-based (`train.py:692` vs
  `:718`) — warmup fill estimate is off.
- A4. Pretraining is broken/stale: `pretrain.py:939` calls `get_encoder_config('default')`, but
  `config.py:289` only accepts named sizes → corroborates that Stage-1 pretraining is non-functional/unused.
- A5. `scripts/auto_eval_after_training.sh:72` seds an old checkpoint assignment that no longer exists
  (`evaluate_tsfm.py:70` uses `TSFM_CHECKPOINT`) → may silently eval the legacy default checkpoint.
- A6. Text-encoder device handling (`token_text_encoder.py:41,62`) keeps the SentenceTransformer outside
  module registration → fine single-GPU, risky for DDP/non-default GPU — **relevant if we use multi-GPU
  RunPod**.
- A7. *(design, not a bug)* Patches interpolated to 64 steps (`preprocessing.py:231`, `config:37`) is an
  information bottleneck that may cap Medium's advantage — a hypothesis worth a targeted ablation, not a fix.

**Caught ONLY by my sweep (the high-severity ones — Alex's sweep did not reach these):**
- 🔴 **HARTH `get_window_labels` min-subtraction label bug** (corrupts HARTH GT for all models + EXP-P3).
- 🔴 **Cache-state-dependent split → first-run val/train leakage** (beyond the subject-awareness point).
- 🔴 **LLaSA accelerometer not ÷9.8** (unfairly cripples LLaSA).
- 🟠 **LanHAR HARTH label shift** (unfairly penalizes LanHAR ZS).
- 🟠 **`SEED=42` not env-pluggable** (multi-seed EXP-P6 impossible without edits).

**Takeaway:** the union — not either sweep alone — is the safe pre-ablation checklist. The three CRITICALs
(HARTH label, leakage, LLaSA ÷9.8) and the seed/LanHAR fairness items came only from my pass; the latent
soft-target/warmup/device/pretrain/auto-eval items came only from Alex's. Both are now in this report.
