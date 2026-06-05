# HALO / TSFM — Codebase ↔ Paper Audit (for rebuttal)

Source: deep codebase-mapping workflow `w4z5340ug` (7 agents) + `docs/baselines/RESULTS.md`,
cross-checked against the submitted PDF. Author-eyes. Inconsistencies are **not** softened —
the point is to find them before a reviewer does.

> **Bottom line:** the *architecture* the paper describes is largely faithful and is in the
> deployed model. The exposure is in (1) the **two-stage pretraining** framing, (2) the
> **open-set scoring** description, (3) the **42.0% averaging convention**, and a handful of
> falsifiable constants (kernels, mask ratio, synonym count, conditioning equation).

---

## 1. Deployed forward path (the `small_deep` config = paper "Small", per-patch mode)

| # | Stage | File / function | Note |
|---|---|---|---|
| 1 | Session load | `datasets/imu_pretraining_dataset/multi_dataset_loader.py:IMUPretrainingDataset.__getitem__` | Pickle cache keyed by `md5(datasets,max_sessions,seed)` — **not** invalidated by channel/patch config changes (stale-cache hazard). |
| 2 | Channel filter/group | same, `IMU_PATTERNS` (loader:364) | keeps `acc/gyro/mag/ori`; all groups, sorted; `shuffle_channels=False`. |
| 3 | Conditioning text | loader:426–552 | `"{dataset_desc} {channel_desc} (sampled at {Hz}Hz, {patch}s window)"`. Placement comes from manifest channel **name/description**, NOT `dataset_config.json:placement`. Terse manifests → literal `"Channel: {ch}"`. |
| 4 | Patching | `model/preprocessing.py:create_patches` | **seconds-based** `int(Hz*patch_sec)`. Eval fixed **1.0s** (`evaluate_tsfm.py:81`); train randomizes per-dataset. Cap `MAX_PATCHES_PER_SAMPLE=48`. |
| 5 | Interp + norm | `preprocessing.py:46–186` | linear interp → **64 timesteps**; per-patch per-channel **z-score**. |
| 6 | Tokenize | `model/feature_extractor.py:SpectralTemporalExtractor (225–368)` | temporal MultiScaleConv1D (`cnn_channels=[32,64]`, **`cnn_kernel_sizes=[5]` single kernel**) + AdaptiveAvgPool → 288-d; spectral `rfft(n=64).abs()`→33 bins→MLP → 96-d; concat → **D=384**. Channel-shared. |
| 7 | PE + tokens | `model/encoder.py` + `model/positional_encoding.py:18–117` | `+ scale·PE`, scale learnable init 0.1; pad_token for padded patches/channels. |
| 8 | Encoder | `model/transformer.py:DualBranchTransformer (8 blocks, 338–472)` | per block: temporal self-attn → cross-channel self-attn → **one shared FFN** (4×D); **post-norm**; SDPA masks. |
| 9 | **Sensor conditioning** | `model/token_text_encoder.py:ChannelTextFusion (377–440)` | **AFTER** encoder. `fused = sensor + sigmoid(W_s·sensor + W_c·e_c + b)·e_c` — **per-element sigmoid gate, GELU MLP**. (The paper's additive `H0+γ·e_c` path = `positional_encoding.py:ChannelSemanticEncoding` is **disabled** for Small via `train.py:1865 use_channel_encoding=False`, but **enabled** in `MEDIUM_CONFIG`.) |
| 10 | Channel fusion | `model/semantic_alignment.py:CrossChannelFusion` | **6 fusion queries** → `U (B,P,384)`. |
| 11 | Per-patch projection | `semantic_alignment.py:454–458` | `per_patch_prediction=True`: `(B,P,384)`→3-layer MLP→L2-norm `z_imu (B,P,384)`. **Temporal-pooling Transformer + multi-query pool are BYPASSED** (dead at inference; only the non-per-patch `small` config uses them). |
| 12 | Text side | `token_text_encoder.py:TokenTextEncoder (frozen) + LearnableLabelBank` | frozen MiniLM tokens (cached, kept out of state_dict) + learnable query pooling → `z_text`. Same instance feeds conditioning + label bank. |
| 13 | Alignment loss | `training_scripts/.../semantic_loss.py:InfoNCELoss` | bidirectional CLIP; soft targets from **frozen mean-pooled** MiniLM text-text sim (z-scored, `τ_s=0.5`, softmax); MoCo FIFO queue (256) both modalities; `logit_scale` init `log(1/0.07)`, clamp [1,50]. `SOFT_TARGET_WEIGHT=1.0` ⇒ in-batch **pure soft**, queue = hard negatives. |
| 14 | ZS inference | `val_scripts/.../evaluate_tsfm.py:evaluate_zero_shot_majority_vote` | per-patch argmax-cosine vote → session label. **Open-set scored via synonym-GROUP map**; closed-set via exact name. |

---

## 2. Paper ↔ code discrepancies, ranked, with rebuttal triage

**ACTIVE = a reviewer can derive it from the *paper* now (must address in rebuttal).**
**LATENT = only visible once code is released (fix before camera-ready / artifact eval).**

| ID | Sev | Triage | Paper says | Code reality | Where |
|----|-----|--------|-----------|--------------|-------|
| **C1** | HIGH | LATENT | Two-stage: Stage-1 heterogeneity-aware **SSL pretraining** produces the encoder used in Stage 2 | **Released ckpt is alignment-only, from scratch.** `PRETRAINED_ENCODER_PATH=None` (train:213); no `training_output/imu_pretraining` artifact; `pretrain.py` builds a *different* arch (CNN, 4 layers). The heterogeneity-aware **architecture** (tokenizer/conditioning) IS in the model; the **SSL pretraining procedure** is not used. | `semantic_alignment_train.py:213`; `pretrain.py:74–82` |
| **C2** | HIGH | ACTIVE | ZS open-set uses **exact string match** for text-aligned (HALO/LanHAR) | Open-set is **synonym-GROUP scored for ALL models incl. HALO** (43 groups, avg 3.5/group). Exact match only for *closed-set* text-aligned. | `grouped_zero_shot.py:112–142`; `evaluate_tsfm.py:526–544` |
| **C4** | HIGH | ACTIVE | Headline ZS open-set **42.0%** | **42.0% = mean over the 5 MAIN datasets only** (MobiAct 42.2 / MotionSense 49.3 / RealWorld 48.0 / Shoaib 49.2 / Opportunity 21.1). All-7 mean (with VTT 1.3, **HARTH 29.1 post label-fix**) ≈ **34.3%** (was ≈30.4 with the buggy HARTH 2.0). | `tsfm_evaluation.json` |
| **C3** | HIGH | LATENT | Conditioning = additive `H'=H0+γ·e_c`, scalar γ=0.1, **ReLU** residual MLP, pre-encoder | Small uses **gated** `ChannelTextFusion` (sigmoid gate, **GELU**, post-encoder); Medium uses the additive path → **inconsistent across sizes**. | `token_text_encoder.py:377–440`; `config.py:204` |
| **C5** | HIGH | LATENT | Tokenizer = **multi-scale** CNN kernels **{3,5,7}** | Every config sets `cnn_kernel_sizes=[5]` (single). {3,5,7} only in unused defaults/tests. | `config.py:36,91,141,193,248` |
| **C7** | MED | ACTIVE-ish | LanHAR evaluated at **50Hz** | LanHAR loads `data_20_120.npy`, `DATA_SAMPLING_RATE=20.0`; "50Hz" is only a comment. Run **off its native rate** while HALO gets native. | `evaluate_lanhar.py:1148,1295` |
| **C8** | MED | ACTIVE | **7** held-out test datasets | `dataset_config.json` lists **10** ZS datasets (+realdisp, daphnet_fog, usc_had). Reported JSON contains exactly the 7; the extra 3 have no published numbers. | `dataset_config.json`; deployed JSON |
| **C6** | MED | LATENT | Soft contrastive = **blended** (soft in-batch + hard queue) | `SOFT_TARGET_WEIGHT=1.0` ⇒ in-batch **pure soft, not blended**; soft targets from **frozen** mean-pooled MiniLM (not learnable). | `semantic_loss.py:158–187`; `train.py:257` |
| **C9** | MED | LATENT | MAE mask ratio **0.5**, random | `MASK_RATIO=0.3`; mixed masking (40% rand / 40% span / 20% chan-drop). Moot if Stage-1 de-emphasized. | `pretrain.py:106`; `losses.py:604–642` |
| **C10** | MED | ACTIVE | avg **2.3** synonyms/activity, **WordNet**-derived | Computed **3.56**; hand-authored dicts, **no WordNet call**. Falsifiable number. | `label_augmentation.py` |
| **C11** | MED | ACTIVE | Table 8 scaling Small-Deep **46.0** | `SCALING.md` says 46.0; deployed JSON says **42.0**. Two official numbers, same model. No JSON backs 46.0. | `docs/baselines/SCALING.md:23,101` |
| **C14** | LOW | LATENT | Temporal pooling (Transformer + multi-query) → z_imu | Exists but **bypassed** in deployed per-patch configs. | `semantic_alignment.py:454–464` |
| **C15** | LOW | LATENT | 70/15/15 **per dataset** | Random **session-level** split over pooled list — **not subject-disjoint**. Inflates the in-dist ~85% val_acc (not the held-out ZS numbers). | `multi_dataset_loader.py:321–339` |
| **C12/13** | LOW | LATENT | Contrastive = 2 aug views, jitter+scale+**time-warp**; λ_mae=λ_con=1.0 | One aug view; default aug = jitter+scale+**time_shift**; EMA dynamic loss balancing. Stage-1; subsumed by C1. | `augmentations.py:562`; `pretrain.py:99–103` |
| **C16** | LOW | ACTIVE | Trainable **35M** | RESULTS 35M, some docs 29.3M, Medium ~66.6M vs paper 63M. Reconcile. | instantiated counts |

**Resolved:** **Medium `d_model = 512`** (`config.py:186`), definitively — paper is correct. The "768"
is Medium's `semantic_dim`/`contrastive_text_dim` (MPNet alignment space), not encoder width. The
`MEMORY.md`/`RESULTS.md` "d=768" notes are stale.

**Note on the "signal-aug closes 9.5pp gap" claim:** this is a **Stage-2 loader** augmentation
ablation (jitter+scale in `multi_dataset_loader.py:471–483`) and **is** backed by an ablation JSON —
it is *not* a Stage-1 claim. So C1 does **not** invalidate the signal-aug result. (The w4z5340ug
stage-1 agent over-attributed it; corrected here.)

---

## 3. Numbers provenance (Tables 3–9)

| Table | Backed? | Path | Note |
|---|---|---|---|
| T3 ZS open/closed (42.0 vs 28.3) | Yes | `test_output/baseline_evaluation/tsfm_evaluation_small_deep_v2.json`, `moment_*.json` | 42.0 = 5-main MV mean. ⚠ `tsfm_evaluation.json` (md5 `17a644ad…`) is the **STALE no_text_aug** ablation — do not cite (6–14pp off). |
| T4 per-dataset | Yes (Small-Deep + baselines) | `*_evaluation.json` | **Medium/Tiny per-dataset have no JSON.** |
| T5 HARTH recovery | Yes | `test_output/harth_analysis/analysis_summary.json` + figs | backed. |
| T6 ablations | Yes (6 JSONs) | ablation JSONs + `docs/ablation_results.md` | **single-seed (3431), no CI.** |
| T8 scaling | **CONFLICT** | `docs/baselines/SCALING.md` | 46.0 vs 42.0; no Medium/Tiny scaling JSON. |
| T9 iPhone latency | **NO artifact** | — | unbacked; keep the raw profiling export. |
| Embedding (76.7% NN, 0.857, 0.377) | **NO** | fig only | `evaluation_metrics.py:133–227` computes structure, not these values. Save to JSON. |

---

## 4. File index (quick lookup)

| Need | File:line |
|---|---|
| 5 size configs | `model/config.py` (Small-Deep 134–, Medium 186–) |
| `cnn_kernel_sizes=[5]` | `model/config.py:36,91,141,193,248` |
| Tokenizer | `model/feature_extractor.py:225–368` |
| Dual-branch block | `model/transformer.py:338–472` |
| Additive conditioning (disabled for Small) | `model/positional_encoding.py:120–336` |
| **Real conditioning (gated)** | `model/token_text_encoder.py:377–440` |
| Per-patch vs pooled head | `model/semantic_alignment.py:454–464` |
| `use_channel_encoding=False` (Small) | `semantic_alignment_train.py:1865` |
| `PRETRAINED_ENCODER_PATH=None` | `semantic_alignment_train.py:213` |
| `SOFT_TARGET_WEIGHT=1.0`; queue warmup | `semantic_alignment_train.py:257`, `:688–737` |
| Soft-target sharpening τ_s=0.5; queue | `semantic_loss.py:158–187,208` |
| MAE/InfoNCE pretrain (vestigial) | `pretrain.py:106,74–82`; `losses.py:604–642` |
| Random session-level split | `multi_dataset_loader.py:321–339` |
| Conditioning template + fallback | `multi_dataset_loader.py:426–552` |
| Synonym maps (hand-authored), rate 0.8 | `label_augmentation.py:506–560` |
| Synonym groups (43) for scoring | `datasets/imu_pretraining_dataset/label_groups.py` |
| Eval 4 settings; patch=1.0s; default ckpt=small_v1 | `evaluate_tsfm.py:71,81,999–1049` |
| Open-set group vs closed-set exact | `grouped_zero_shot.py:112–142` |
| LanHAR at 20Hz | `evaluate_lanhar.py:1148,1295` |
| MOMENT eval | `evaluate_moment.py` |
| **Deployed headline JSON (7 datasets)** | `test_output/baseline_evaluation/tsfm_evaluation_small_deep_v2.json` |
| **STALE — do not cite** | `test_output/baseline_evaluation/tsfm_evaluation.json` |
| Scaling conflict | `docs/baselines/SCALING.md:23,101` |
| Best ckpt metadata | `training_output/semantic_alignment/small_deep_v2_4b3fdd6/hyperparameters.json` |
