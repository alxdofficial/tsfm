# HALO / TSFM — Data Diversity, Metadata-Conditioning & Text-Embedding Audit

**Date:** 2026-07-04
**Scope:** Synthesis of four investigations — (i) augmentation inventory + metadata-conditioning verification, (ii) training-corpus diversity, (iii) frozen-MiniLM text-embedding quality, (iv) the "planned but not done" backlog for MobiCom'27.

**Load-bearing framing (read first).** The headline model is `MODEL_SIZE="small_deep"`, **hardcoded** at `semantic_alignment_train.py:163` (not env-overridable), resolved by `get_config` at `config.py:323`. Its saved config (`training_output/semantic_alignment/small_deep_v2_4b3fdd6/hyperparameters.json`) is `feature_extractor_type="spectral_temporal"`, with **no** `dft_size` and **no** `use_rope`. The physical-Hz filterbank tokenizer + RoPE live only in the sibling config `small_deep_fb` (`config.py:186-210`), which is **not** the headline model and is unreachable without editing line 163. This `small_deep` vs `small_deep_fb` split determines almost every "active?" answer in Section 1. The default augmentation preset is `v2` (`TSFM_AUG_PRESET` default `"v2"`, `semantic_alignment_train.py:357`).

---

## 1. Is the metadata-conditioning actually built + active? (checklist)

All four designed conditioning mechanisms are **genuinely built and wired end-to-end** (none are dead code). But **only mechanism (a) is active in the shipped headline model.** (b), (c), (d) are inert in `small_deep` and switch on only under `small_deep_fb`.

| Claim | Built? | Active in headline (`small_deep`) / default run? | Evidence |
|---|---|---|---|
| **(a) placement/gravity text → ChannelTextFusion cross-attention** | **YES** | **YES — ACTIVE** | `ChannelTextFusion` is genuine cross-attn: `nn.MultiheadAttention` queries attend to text tokens (`token_text_encoder.py:355-360, 419-425`) + gated residual `fused = sensor + gate*channel_embs` (`437-438`). Applied in forward at `semantic_alignment_train.py:548-550` (`use_fusion`←`ABLATION_CHANNEL_TEXT_FUSION` default 1, `322`). Placement/gravity strings reach it via `metadata['channel_descriptions']`→`text_encoder.encode` (`773, 779, 537-550`). The encoder's *own* additive channel-PE is **disabled** (`use_channel_encoding=False`, `2035`; encoder `335-352`), so ChannelTextFusion is the **sole** channel-text path — and it is on. |
| **(b) sampling rate in tokenizer (φ = m·r/S + Nyquist mask oₖ on token)** | **YES** | **NO — NOT ACTIVE** | φ = m·r/S computed per-sample (`feature_extractor.py:624`); Nyquist mask `o` (`630-636`) and masked ê + `o` concatenated into the token (`733-735, 742`). But this is `PhysicalFilterbankTokenizer`, only instantiated when `feature_extractor_type=="physical_filterbank"` (`encoder.py:141-158`). Headline = `spectral_temporal` (`config.py:143`; checkpoint json). `IS_FILTERBANK=False`, DataLoader built `dft_size=None` (`2184`). In the headline path rate reaches the model **only** through the frozen-SBERT text suffix "sampled at NHz" (`multi_dataset_loader.py:555`); the signal is interpolated to 64 samples so the waveform carries **no** rate (comment `550-552`). Frozen SBERT can't do numeracy — weak conditioning. |
| **(c) patch duration → resolution flag resₖ on token** | **YES** | **NO — NOT ACTIVE** | `res` computed from D=N/r (`feature_extractor.py:637-639`), appended when `use_resolution_mask` (`736-737`). Same filterbank-only gate as (b). Headline conveys duration only via the "Xs window" text suffix (`555`). |
| **(d) rate + duration → RoPE physical-time positions** | **YES** | **NO — NOT ACTIVE** | Positions = p·D, D=N/r seconds (`encoder.py:360-373`); RoPE inv_freq over physical periods (`transformer.py:108-113`) applied in temporal attention (`153-154`), threaded encoder→transformer→attn (`379-381, 508-524`). Gated by `use_rope`, **False** for `small_deep` (no key; default `191`); headline uses additive learned temporal PE (`encoder.py:335-336`). `use_rope=True` only in `small_deep_fb` (`config.py:201`). Even if enabled, without `patch_len_samples`/rate the positions silently fall back to integer patch indices (`encoder.py:374-375`), which are only supplied by the filterbank DataLoader. |

**Bottom line.** In the shipped headline model, sampling-rate and patch-duration conditioning is carried **entirely by numeric substrings inside frozen-SBERT channel text** ("sampled at 50Hz, 1.0s window") — a channel the code's own comments (`multi_dataset_loader.py:544-552`) admit SBERT cannot do numeracy on. The physical-Hz rate tokenizer, Nyquist/resolution masks, and physical-time RoPE — the whole "physical units in the token" story — are dormant until someone trains `small_deep_fb`.

**Additional conditioning flags:**
- The headline `hyperparameters.json` records **no augmentation config** — you cannot tell from the checkpoint which aug preset trained it, only that the *current code default* is `v2`. Reproducibility gap.
- `yaw_rotation` changes sensor orientation but nothing in the text/token conditioning encodes orientation, so unlike gravity/rate/duration it is a pure invariance augmentation with **no co-varying metadata signal**.
- Two sibling checkpoints exist for ablations (`..._ablation_no_signal_aug`, `..._ablation_no_text_aug`), confirming the signal/text kill-switches are exercised.

---

## 2. Augmentation inventory + gaps

### 2.1 Signal augmentations (v2 preset — the shipped default)

Per-sample, **train-split only**, applied in `IMUAugmenter.__call__` in this fixed ORDER: channel_dropout → yaw_rotation → gravity → rate → time_warp → time_shift → magnitude_warp → scale → jitter (`augmentations.py:690-691, 792-798`). "Default ON?" is under the **v2** preset.

| Augmentation | What it does | Default ON? (v2) | prob / params | Co-varies channel text? |
|---|---|---|---|---|
| **jitter** | Additive Gaussian sensor noise | **ON** | p=0.5, σ=0.05 (`augmentations.py:578-582, 827-829`) | No |
| **scale** | Per-channel amplitude scaling | **ON** | p=0.5, [0.9,1.1] (`585-591, 831-835`) | No |
| **time_shift** | Whole-window phase shift | off | p=0.5, max_ratio=0.05 (`594-599`) | No |
| **time_warp** | Non-linear cadence warp (cubic spline) | off | p=0.3, n_knots=4, strength=0.2 (`602-608`) | No |
| **magnitude_warp** | Smooth per-channel amplitude modulation | off | p=0.3, n_knots=4, strength=0.3 (`611-617`) | No |
| **gravity [P1]** | Butterworth low-pass removes gravity DC from acc triads → iOS `userAcceleration` (only where gravity still present, `_gravity_present`) | **ON** | p=0.5, cutoff=0.4Hz, order=2 (`620-628, 886-912`) | **YES** — rewrites desc to "(gravity removed)" (`_mark_gravity_removed` `754-764`; applied `907, 911`) |
| **yaw_rotation [P2]** | Rodrigues rotation about the estimated gravity axis only (heading randomization; preserves "down") | **ON** | p=0.5, max_deg=180 (`631-639, 915-952`) | **No** (signal only; no orientation token exists) |
| **rate [P3]** | Anti-aliased `resample_poly` to random Hz; updates `sample.sampling_rate` | **ON** | p=0.5, 15–100 Hz, min_samples=32 (`642-650, 955-971`) | **YES\*** — new rate flows into the "sampled at NHz" text suffix in the **non-filterbank** loader path (`multi_dataset_loader.py:555`). In filterbank mode the suffix is suppressed and rate feeds the tokenizer instead (`544-548`). |
| **channel_dropout [P4]** | Drops a whole sensor group (default gyro) if ≥1 acc triad survives | **ON** | p=0.3, groups=("gyro",) (`653-664, 974-990`) | **YES** — drops names + descriptions (`987-989`); changes the channel *list*, not linguistic text |

**Patch-duration variability:** **ON.** `USE_PATCH_SIZE_AUGMENTATION=True` (`342`) + `PATCH_SIZE_RANGE_PER_DATASET` (`377-394`, per-dataset (min,max,step), e.g. uci_har=(0.75,1.25,0.25)), threaded to the train loader (`2180`). Loader randomizes `actual_patch_size` per sample (`multi_dataset_loader.py:510-518`) and **co-varies the "{X}s window" text suffix** (`555`, non-filterbank path).

**Dead / disabled:** The legacy `IMUAugmentation` class (`augmentations.py:18-556`, incl. `rotation_3d` full SO(3), `channel_shuffle`, `resample`) is **dead** — the loader builds only the V2 `IMUAugmenter` (`multi_dataset_loader.py:207-216`); `USE_ROTATION_AUGMENTATION=False` (`semantic_alignment_train.py:343`).

**Kill-switches:** `ABLATION_SIGNAL_AUG=0` disables jitter+scale only (`366-368`); `TSFM_AUG_PRESET=none|legacy` (`357-364`).

### 2.2 Text augmentation

**(a) LABEL augmentation — `augment_label` (`label_augmentation.py:423-477`): ON.** It does **both** synonym replacement (per-dataset dicts, e.g. `walking→{strolling, striding, ambulating, pacing}`) **and** template/phrase-ification (e.g. `"person {}"`, `"{} activity"`, `"physical activity: {}"`). Not just synonyms. Wired ON via `use_text_augmentation=ABLATION_TEXT_AUG` (default `"1"`→True, `326`, `2187`). **Fraction augmented:** `augmentation_rate = 0.8` for train, `0.0` for val (`multi_dataset_loader.py:473-476`) — so ~80% of train labels enter the synonym/template branch. Effective *change* rate is lower because template lists include the identity `"{}"` and some datasets are identity-only (`realworld/shoaib/opportunity/harth = ["{}"]`), but those are zero-shot **test** sets where rate is 0 anyway. All 10 train datasets have real synonym+template configs.

**(b) CHANNEL-DESCRIPTION text augmentation — GAP: none.** Each channel's base description is a **single fixed manifest string**, e.g. uci_har `body_acc_x` = `"Body acceleration X-axis from accelerometer (gravity removed)"`, pamap2 `hand_acc16_x` = `"Acceleration x-axis (±16g scale) from wrist-mounted IMU"` (verified from `data/*/manifest.json`). It is deterministically wrapped as `"{dataset_desc} {ch_desc} (sampled at NHz, Xs window)"` (`multi_dataset_loader.py:443-457, 553-557`). The **only** variation applied to channel text is non-linguistic: (i) numeric Hz/window suffix, (ii) gravity→"(gravity removed)" flip, (iii) channel_dropout removing entries. There is **no paraphrase / synonym / template variation** of the channel string itself (no `augment_channel*` function exists anywhere — grep confirms). The placement/sensor semantics the model conditions on are a fixed vocabulary with **zero linguistic augmentation**, unlike labels — a real robustness asymmetry.

**Net augmentation gaps:**
1. **Channel-text linguistic augmentation is entirely missing** (asymmetric with label aug). Placement/sensor phrasing is a fixed 1-string-per-channel vocabulary → the model never sees paraphrases of "wrist-mounted IMU" and will be brittle to real-world placement descriptions.
2. **Orientation has an invariance aug (yaw) but no co-varying metadata token** — a missed opportunity if orientation-awareness is ever wanted.
3. **Rate/duration aug co-varies only a numeric text substring** in the headline path — a signal SBERT cannot parse (see §4).

---

## 3. Dataset diversity table + well-roundedness verdict

**Source of truth:** `benchmark_data/dataset_config.json` → `train_datasets` (**11** datasets). Per-dataset numbers counted directly from `data/<name>/manifest.json` + `labels.json`; patch durations from `semantic_alignment_train.py:135` (`PATCH_SIZE_PER_DATASET`) and `:377` (`PATCH_SIZE_RANGE_PER_DATASET`).

> Doc drift: `CLAUDE.md` says "Train (10)" and omits `capture24`, but the config lists **11** and capture24 is on disk (48,518 windows). All 11 audited.

| Dataset | Native IMU rate (Hz) | Placement (device) | #classes | #windows (raw) | Imbalance max/min | Patch dur s (aug range) |
|---|---|---|---|---|---|---|
| uci_har | 50 | waist (phone) | 6 | 10,299 | **1.4** (laying 1944 / w_downstairs 1406) | 1.0 (0.75–1.25) |
| hhar | 50 | waist (phone) | 6 | 175,253 | **11.6** (standing 112,112 / cycling 9,630) | 1.0 (0.75–1.25) |
| pamap2 | 100¹ | hand+chest+ankle (dedicated IMU) | 12 | 4,377 | 7.6 (lying 611 / rope_jumping 80) | 2.0 (1.0–2.0) |
| wisdm | 20 | pocket(phone)+wrist(watch) | 18 | 164,623 | 5.0 (clapping 17,802 / eating_pasta 3,528) | 1.5 (1.0–1.75) |
| dsads | 25 | torso (dedicated IMU) | 19 | 9,120 | **1.0** (all 480) | 2.0 (1.5–2.5) |
| kuhar | 100 | waist (phone) | 17 | 17,374 | **16.8** (standing_up_from_sitting 2,585 / walking_backwards 154) | 1.5 (1.0–1.75) |
| unimib_shar | 50 | pocket (phone, **acc-only**) | 17 | 11,771 | 9.9 (running 1,944 / sitting_down 197) | 1.0 (0.75–1.25) |
| hapt | 50 | waist (phone) | 12 | 2,546 | 7.6 (standing 466 / sit_to_stand 61) | 1.25 (0.75–1.25) |
| mhealth | 50 | chest+ankle+arm (dedicated IMU) | 12 | 2,029 | 2.9 (sitting 197 / jump_front_back 67) | 1.5 (1.0–1.75) |
| recgym | 20 | wrist | 11 | 7,150 | 8.2 (walking 1,245 / rope_skipping 152) | 1.5 (1.0–1.75) |
| capture24 | 100 | wrist (**acc-only**, free-living) | 10 | 48,518 | 10.1 (sitting 7,550 / sports 747) | 1.5 (1.0–1.75) |

¹ PAMAP2 manifest lists a 9 Hz channel — that's only `heart_rate`; all 51 IMU channels are 100 Hz and the loader keeps only acc/gyro/mag/ori (`multi_dataset_loader.py:380`), so effective IMU rate is 100 Hz.

**Raw total: 453,060 windows — but the effective corpus is ~85k and re-shaped by a cap.**

### 3.1 The 10k cap dominates everything (and the config's "subsampling" is dead code)

`semantic_alignment_train.py:160` sets `MAX_SESSIONS_PER_DATASET = 10000`, passed to the loader (`:2181`). The loader applies it with a **one-time seeded `random.shuffle` then truncate** (`multi_dataset_loader.py:329-332`) — **not** the stratified `subsampling` block in `dataset_config.json` (hhar→15k / wisdm→15k / capture24→20k), which is **never read by this training script**. Consequences:

- Effective per-dataset windows: uci_har 10k, hhar 10k, pamap2 4,377, wisdm 10k, dsads 9,120, kuhar 10k, unimib_shar 10k, hapt 2,546, mhealth 2,029, recgym 7,150, capture24 10k → **effective total ≈ 85,222** (not 453k).
- The cap is a *fixed* random subset for the whole run, so **~94% of HHAR and ~79% of CAPTURE-24 are never seen in any epoch.** The single most valuable real-world signal (free-living CAPTURE-24) is throttled to 10k/48k windows.
- Net effect: crude accidental dataset-level rebalancing (mega-sets stop drowning small ones) — good, but an accident, not a designed sampler, and it silently discards most free-living data.

**Effective (post-cap) corpus balance:** locomotion (walk/run/stairs/cycle/jump) **36.4%**, posture (sit/stand/lie/sleep) **32.8%**, everything else **30.7%**. Top effective classes: standing 15.6%, walking 10.3%, sitting 8.6%, running 4.3%, upstairs 3.5%, downstairs 3.3%, lying 2.7%, cycling 2.6% — the top 3 (all posture/locomotion) are ~35% of the corpus.

### 3.2 Diversity assessment

- **Rate diversity — thin, 50 Hz-skewed.** Only **4 distinct native rates**: 50 Hz (5 sets), 100 Hz (3), 20 Hz (2), 25 Hz (1). **No 30 Hz** (a common Android default), no 40/64 Hz. The two 20 Hz sets (wisdm, recgym → 10 Hz Nyquist) can't carry fast-cadence content. Partly absorbed by the resampling tokenizer, so secondary — but the corpus does not stress-test the 20–30 Hz band real phones frequently deliver.
- **Placement — real but waist/pocket-heavy; watch under-represented.** By dominant device across the *effective* corpus: waist-phone ≈ 38% (uci_har+hhar+kuhar+hapt), pocket-phone ≈ 23% (unimib_shar+wisdm), dedicated wrist ≈ 20% (recgym+capture24), dedicated torso/multi-IMU ≈ 18% (dsads+mhealth+pamap2). So **~61% of training windows are a phone at the waist or trouser pocket, only ~20% is dedicated wrist.** Multi-body sets add thin ankle/chest/hand channels. Genuinely missing: **head/upper-arm, and in-hand phone (scrolling/texting/on a call)** — the exact posture a phone is in during much real use.
- **Activity — 94 unique strings, but locomotion+posture is the backbone and the tail is siloed.** Only ~10 activities appear in ≥3 datasets (walk/stand/sit/run/stairs/cycle/lie/jump); everything else lives in a single dataset. Thin for real life:
  - **Transport/vehicle:** essentially one class — capture24 `vehicle` (~0.99% effective) + dsads elevator (480 windows). No driving/bus/train/escalator distinction. A shipped phone HAR model needs transport detection; this corpus barely has it.
  - **Free-living:** capture24 is the **only** naturalistic set, and it's wrist-**acc-only**, coarse 10-class (sitting/standing/walking/sleeping/vehicle/household_chores/mixed_activity/manual_work/sports/bicycling). Every other dataset is a scripted lab protocol. Real deployment is dominated by messy transitions and idle time that are almost absent.
  - **Phone/watch gestures:** wisdm adds typing/writing/clapping/drinking/brushing (watch), but no scrolling, texting, calling, or hand-raise-to-look — the micro-gestures a watch model is asked about.
  - **Household/ADL:** present but siloed (pamap2 ironing/vacuuming, wisdm folding/eating, capture24 household_chores) with little cross-dataset reinforcement.
  - **Falls:** unimib_shar adds 8 fall classes — specialized, short, rare; arguably noise for general HAR.
  - Well-covered: walking, standing, sitting, stairs up/down, running/jogging, cycling, lying — the classic HAR core.
- **Duration — narrow, all short windows.** Base patch durations span **1.0–2.0 s**; augmentation widens the union to **0.75–2.5 s**. **No long-context window (10 s / 30 s).** Poor fit for slow/free-living activities: CAPTURE-24 free-living is chopped into 1.5 s patches, throwing away exactly the long-horizon context that distinguishes "sitting-working" from "in a vehicle" from "household chores."
- **Class balance — standing/walking/sitting dominate; long rare tail.** Within-dataset imbalance ranges from perfectly balanced (dsads 1.0) to severe (kuhar 16.8, hhar 11.6, capture24 10.1, unimib_shar 9.9). Across the corpus **~60 of 94 activities each contribute <1%** of windows — a heavy tail of single-dataset gym/exercise/fall/transition classes with 60–500 windows apiece (pamap2 rope_jumping 80, hapt sit_to_stand 61, mhealth jump_front_back 67).

### 3.3 Verdict: not yet well-rounded for real-world phone/watch HAR

A **solid lab-HAR benchmark** (11 sets, 4 rates, ~85k effective windows, strong walk/stand/sit/stairs/run/cycle coverage across waist/pocket/wrist/torso) — but for a *genuinely useful in-the-wild phone/watch* model it has concrete holes:

1. **Almost no free-living data actually reaches the model** — the one naturalistic set (CAPTURE-24) is wrist-acc-only, coarse-labeled, and **79% discarded** by the 10k cap. 90%+ of training is scripted lab protocols.
2. **Placement is phone-at-waist/pocket-dominant (~61%)**; dedicated wrist is only ~20%; in-hand phone / upper-arm / head are missing.
3. **Transport and phone-interaction gestures are effectively absent** — two categories any shipped HAR feature must handle.
4. **Short windows only (≤2.5 s)** — no long-context modeling for slow/ambient activities.
5. **Rate coverage skips the 30 Hz band** and leans on 50 Hz.
6. **Heavy posture/locomotion prior (~69% effective)** with a 60-class <1% tail — rare real-world classes get almost no gradient.

Highest-leverage additions: (a) more **free-living wrist+pocket** data with fine labels and **long windows**; (b) explicit **transport** classes; (c) **in-hand/phone-interaction** samples; (d) replace the fixed-random 10k truncation with a **stratified, per-epoch resampler** so CAPTURE-24's free-living signal isn't thrown away.

Key file:line: caps `semantic_alignment_train.py:160,2181`; random-truncate `multi_dataset_loader.py:329-332`; all-IMU-channel selection `multi_dataset_loader.py:380-399`; patch durations `semantic_alignment_train.py:135-157,377-394`; aug flag `:342`; dead stratified-subsample config `benchmark_data/dataset_config.json` `subsampling` block.

---

## 4. Text-embedding quality (frozen MiniLM)

**Answer: BOTH — a genuine coarse-activity geometry sitting on a bag-of-words substrate that collapses exactly the fine-grained distinctions HALO's label set depends on (direction, side, order, negation).** All cosines from `SentenceTransformer('all-MiniLM-L6-v2')`, `normalize_embeddings=True`, the frozen encoder at `model/config.py:154`.

**Code path.** Recognition = cosine of IMU embedding to text label embeddings: `logits = emb @ text_embs.T` (`eval_common.py:434`), text from `label_bank.encode(labels_ld)` (`evaluate_tsfm_v2.py:118`). The label bank is a *learnable* attention pool over **frozen** SBERT token embeddings (`token_text_encoder.py:497`); training soft-targets use **frozen mean-pool** `encode_frozen` (`token_text_encoder.py:515`). The learnable pool **cannot manufacture separation that isn't in the shared frozen tokens** — for "stairs up"/"stairs down" the only differing token is up/down, whose SBERT vectors are near-collinear.

**Concrete train/eval inconsistency:** recognition passes labels **with underscores** (`evaluate_tsfm_v2.py:329`; motionsense config literally stores `'walking_downstairs'`) straight to `label_bank.encode` — no de-underscoring. Only the ConSE **baseline** path de-underscores (`eval_v2.py:418`). Empirically the underscored form is only 0.86–0.90 cosine to its de-underscored self and makes the up/down collapse slightly *worse* (0.897 vs 0.868). Minor, but a real mismatch.

**TEST 1 — 87-label pairwise structure.** Off-diagonal cosine **mean 0.268, median 0.247, max 0.970**, only **1.2% of pairs > 0.8**, 7.2% > 0.5. Globally **NOT** a smeared space — the top-level taxonomy is well spread (walking↔running 0.495, sitting↔standing 0.425, walking↔laying 0.281, accelerometer↔gyroscope 0.453). But the top pairs are a **directional/order collapse**, not synonyms:

| cosine | pair | problem |
|---|---|---|
| 0.970 | lie to sit \| sit to lie | opposite transitions, order ignored |
| 0.941 | going down stairs \| going up stairs | **opposite direction** |
| 0.932 | stairs down \| stairs up | **opposite direction** |
| 0.930 | ascending stairs \| descending stairs | **opposite direction** |
| 0.893 | falling left \| falling right | **opposite side** |
| 0.868 | walking upstairs \| walking downstairs | **opposite direction** |

Single-word labels floor at mean **0.312** (vs 0.266 multi-word), min 0.134 — a modest floor, not catastrophic. **Real closed-set vocabs show the danger directly:** motionsense `walking_downstairs`↔`walking_upstairs` are **mutual nearest neighbors at 0.868**; harth & mobiact `stairs_down`↔`stairs_up` **0.932** (mutual NN); mobiact `car_step_in`↔`car_step_out` **0.854**. For these the text prior is nearly degenerate — the InfoNCE soft-target gives almost no gradient to separate the pair, and cosine-argmax must rely entirely on the IMU encoder out-sharpening a ~0.93 text tie.

**TEST 2 — bare label vs phrase templates (does label phrase-augmentation help?).** Phrasing **does move** embeddings (cos(bare,"a person X") ≈ 0.66–0.80). Effect on separating 8 confusable pairs (mean cosine, **lower = better**):

| template | mean confusable cosine |
|---|---|
| bare `walking` | 0.483 |
| `a person walking` | 0.469 |
| `someone is walking` | **0.444** (best, ~8% better) |
| `the activity of walking on flat ground` | **0.801** (catastrophic) |

Short, discriminative-word-dominant templates help **modestly**; **verbose shared boilerplate collapses everything** (sitting↔lying 0.305→0.869) because ~5 shared tokens drown the 1 discriminative token. **Templating is a double-edged sword, not a free win** — it helps only if kept short and content-dominant.

**TEST 3 — bag-of-words / order / negation.** MiniLM **ignores negation and order** on label-length strings: gravity removed | gravity **not** removed 0.923; cycling | **not** cycling 0.876; sit to stand | **stand to sit** 0.964; left wrist | right wrist 0.946; up | down 0.868–0.932. Only true content-word swaps separate (accelerometer | gyroscope 0.453). Any distinction carried by a function/directional token (not/up/down/left/right/forward, or A-to-B order) is smeared to 0.87–0.97.

**TEST 4 — real channel descriptions.** Far more collapsed (off-diag **mean 0.611**, heavy shared boilerplate). Modality survives: acc↔gyro 0.775 (waist), gyro-waist's NN is gyro-torso. Placement partially survives: wrist↔pocket 0.528, chest↔back 0.593. But **side is invisible** (left-upper-arm↔right-upper-arm **0.996**, mutual NN — Opportunity's bilateral IMUs get identical channel text) and **axis is nearly invisible** (acc-X↔acc-Y 0.921, X↔Z 0.851).

**Would a stronger encoder help? — Empirically NO for labels.** all-mpnet-base-v2 (768-d): same 87-vocab off-diag mean **0.268 (MiniLM) vs 0.270 (mpnet)**, max 0.970 vs 0.946, frac>0.8 0.012 vs 0.009 — statistically identical. On antonyms mpnet is a wash-or-worse (up/down stairs 0.868→**0.909**; left/right wrist 0.946→**0.981**). The **only** place mpnet clearly wins is a *full grammatical sentence with negation* ("the person is walking" vs "…not walking" 0.849→**0.573**). **The fix is text form (full sentences), not encoder size.**

**Verdict.** MiniLM is **sufficient for HALO's coarse activity taxonomy** (walk/run/sit/stand/lie/cycle/jump separate cleanly at 0.2–0.5; mean 0.27, not globally smeared). It is **insufficient — and not rescuable by a bigger encoder — for exactly the fine-grained axes HALO's 87-label set is built on** (direction, side, order, negation → 0.87–0.97). For any dataset whose classes differ only in direction/side/order (motionsense/harth/mobiact stairs, mobiact car step in/out), the "zero-shot cosine-to-label-text" story is doing ~no work; the burden is fully on the IMU encoder against a near-degenerate contrastive target.

Probe scripts: `/tmp/claude-1000/-home-alex-code-HALO/27fe69bb-9410-4f99-99c1-c5650fde1607/scratchpad/minilm_probe.py` (tests 1–4), `.../mpnet_cmp.py` (encoder comparison).

---

## 5. Prioritized "planned but not done" backlog for MobiCom'27

**Verified current code state:** `halo/` package, `scripts/export/`, gravity canonicalization (Mahony), `from_pretrained/predict`, and ONNX/CoreML/ExecuTorch code **do not exist** (greps empty). Baseline adapters present are **only** `crosshar.py` + `limubert.py` (no UniMTS/ssl-wearables). The streamable-encoder code (RoPE, `build_temporal_mask`, distillation) and the P1–P4 aug curriculum **are** built. Boundary: M0 eval-v2 + M1 tokenizer + Phase A/B/E encoder code + CAPTURE-24/InclusiveHAR + aug module = **DONE-as-code**; everything below is not.

### The single gating item
- **[MUST] M4 — retrain `small_deep_fb` end-to-end and re-evaluate.** All of M1/PhaseA/B/E is code-built but **never trained** (BUILD_PLAN.md:94). Until this run exists the paper has **zero** new numbers — the honest baseline is still the old-tokenizer checkpoint (ZS-XD avg **33.2 native / 29.6 parity**, RESULTS_V2.md:33). This run also carries: gradcache in-place distillation, trained session-pool, calibrate-then-freeze norm stats over the augmented (r,D) mix, delete old CNN classes at cutover (BUILD_PLAN.md:94–96). **This is what activates conditioning mechanisms (b)(c)(d) from Section 1.**

### (1) Modeling
- **[MUST] Trained session-pool objective.** Offline segment-label path pools with an *untrained* mean-pool; add a trained pooled-session objective under the same masks (BUILD_PLAN.md:69–71). Without it the headline segment number is served through a code path never seen in training.
- **[MUST] Gradcache-path distillation.** Dual-forward offline→online distillation is wired into the *standard* train step only; the gradcache step still needs it. Headline trains under gradcache → critical path, not optional.
- **[MUST for the streaming claim] Gravity-frame canonicalization (Mahony/OS-gravity) + gravity-as-described-channel.** REDESIGN_PLAN M2 (108, 167–168). **Not built** — only the P1 gravity add/remove *augmentation* exists; the canonicalization *preprocessing stage* and explicit gravity channel do not (grep for mahony/canonicaliz empty). **This is the named architectural deficit vs UniMTS** (halo-debug-sweep-findings novelty #1).
- **[NICE] Dense per-frame objective + boundary head + smoothness loss + soft mixture targets.** Explicitly DEFERRED (BUILD_PLAN.md:36–42) — every dataset is single-activity sessions; needs per-timestamp labels or stitched-session aug. Onset/boundary precision capped until this lands (BUILD_PLAN.md:77–80). Required only if the paper claims boundary/segmentation quality.

### (2) Data
- **[MUST-ish] ExtraSensory free-living phone+watch TEST set.** DATA PLAN LOCKED, user-approved. Real deployment condition is currently **unmeasured** (both iPhone test sets are lab-scripted). Needs multi-label→single-activity projection. High reviewer value.
- **[MUST] hhar re-conversion to true 50 Hz.** Converter asserts 50 Hz but does **no** resampling — 200 Hz Nexus-4 windows are temporally compressed and mislabeled, and the model is *told* 50 Hz in channel text (CROSSCHECK.md:72–75). "Re-conversion in progress" (CROSSCHECK.md:219) — verify/finish. It's a **TRAIN** set, so this corrupts training.
- **[MUST] Placement-metadata mislabel fixes + declare units (g vs m/s²).** dsads/kuhar/recgym config-vs-data placement mismatches corrupt channel text; DEBUG_SWEEP_REPORT.md:40 says config is now internally consistent — **verify it matches the data**, and add the still-missing g-vs-m/s² unit declaration.
- **[NICE] Demote `opportunity` to appendix** (4 subjects → degenerate CIs, object/ambient sensors). **[NICE] Frame-level `activity_label` per `timestamp_sec`** in native converters (coupled to dense objective). **[NICE] Virtual-IMU (UniMTS/IMUGPT) augmentation** as the placement/orientation scale lever — "next-submission."
- *(Context: CAPTURE-24 train-add [48,518 windows, 151 subj] and InclusiveHAR test-add [2,049 windows, 20 subj] are DONE; ability-stratified reporting owed — see Eval.)*

### (3) Baselines
- **[MUST] UniMTS adapter (cosine, released weights).** Planned, not built (RESULTS_V2.md:52; only crosshar/limubert exist). **No head-to-head vs the top competitor = primary reject risk.** Highest-priority baseline.
- **[MUST] ssl-wearables adapter (ConSE, released weights).** Planned, not built (RESULTS_V2.md:53) — the deployment-relevant FM reviewers expect.
- **[NICE] Baseline subject-disjoint FS-1%/10% rows** (RESULTS_V2.md:136). **[NICE] MOMENT ZS-XD row** — partially run / GPU-blocked (RESULTS_V2.md:132–133). GOAT has **no released checkpoint**; LLaSA → appendix/drop.

### (4) Evaluation
- **[MUST] GroupKFold rotation for non-degenerate few-shot CIs (task #7).** shoaib/opportunity have **1 test subject** (degenerate CI), realworld/harth have 2 (RESULTS_V2.md:139, 103–107). Current FS CIs are not defensible.
- **[MUST] Ablation grids for the new architecture:** tokenizer arms A/B/C + norm frozen/PCEN/off + log/amplitude on-off; rate-flatness B1 (native/×0.5/×2/37 Hz); placement-text B2 correct/wrong/generic; the capability-Δ 2×2 decomposition (native-rate vs channel-text — currently only the lumped +3.6 F1 is reported, RESULTS_V2.md:122–125); streaming ablations (random-mask-vs-distill drop, K-lookahead latency knee, RoPE-on-long-sessions, attention-sink stability, trained-session-pool vs mean-pool). The redesign's claims rest on these.
- **[NICE] Streaming frame-F1 + boundary-tolerance + rep-count MAE** — blocked on the deferred dense objective + multi-activity data. **[NICE] Gravity/robustness ablations A1/A1b/A1c, corruption C1, accel-only D1** — depend on unbuilt canonicalization. **[NICE] Common-classes appendix table** (`PENDING_REVIEW`). **[NICE] OV-Distractor@k / per-class transparency. [NICE] Ability-stratified ZS reporting** for InclusiveHAR (owed to deliver the inclusivity claim).

### (5) Application / export (M5) — all greenfield, none built
- **[MUST if the paper claims "deployable"] `halo/api.py` `from_pretrained().predict()` + `set_labels()`** with abstain threshold. No `halo/` package exists.
- **[MUST for that claim] One slim canonical checkpoint + model card.** Bless small_deep_v2, **strip the dead ~9.3M temporal-attn+pooling head** (27% of the "35M"; honest active ~25.8M — halo-debug-sweep-findings #3), fold LayerNorms, ship fp16.
- **[MUST for on-device] Freeze/precompute label matrix + channel-text constants** so SBERT drops out — unblocks export and "<10 ms predict."
- **[NICE] `forward_export` pure-tensor path + Core ML / ONNX-web / ExecuTorch + `|Δcos|<1e-3` parity tests. [NICE] `HALOStreamSession`** (ring buffer + KV cache + boundary smoother — substrate verified to 1.2e-6, wrapper not built). **[NICE] Phone-web live demo + "add activity live" + CLI. [NICE] Accel-only gravity-LP fallback + predict-time hygiene.**

### (6) Results-integrity — paper-text musts (≈0 compute)
- **[MUST] Reframe "two-stage MAE SSL" → single-stage.** `PRETRAINED_ENCODER_PATH=None`; the encoder trains from scratch. Presenting it as pretraining is a fabrication risk.
- **[MUST] Correct the native-rate claim.** Paper Table 7 "+11.4 pp" is stale (old 4-layer ckpt); corrected +0.41 pp, rich channel text −3.27 pp isolated; under v2 the lumped capability Δ is +3.6 F1 (RESULTS_V2.md:122).
- **[MUST] Fix unbacked "Small=46.0" (deployed 42.0)** and **drop the leakage-inflated ~85% val** (non-subject-disjoint, ~73% train/val overlap) → honest subject-disjoint numbers.
- **[MUST] Reconcile paper-vs-code architecture text:** gated `ChannelTextFusion`/GELU/post-encoder vs paper's additive/ReLU/pre-encoder; CNN kernels [5] not {3,5,7}; MAE mask 0.3 not 0.5; param count 35.1M vs "25M" vs "29.3M"; open-set scoring group-matched for HALO too.
- **[MUST] Demote LearnableLabelBank to frozen mean-pool for the headline** — net-negative, verified **+5.83 pp** (5-main) for frozen mean-pool on 7/7 datasets (debug-sweep #4). Retrain flag `use_mean_pooling=True` / `ABLATION_LEARNABLE_LABEL_BANK=0` (BUILD_PLAN.md:29–30). *(Directly connected to Section 4: the frozen substrate is what matters, and mean-pool over it beats the learnable pool.)*
- **[MUST] `QUEUE_MODE=semantic`** — the MoCo queue currently zeros soft-target mass → false-negative pushing on synonyms (`semantic_loss.py:171`). Set at M4.
- **[NICE] Purge stale results plumbing** (`generate_results_table.py` reads a removed `"accuracy"` key; stale pre-HARTH-fix results). **[NICE] Resolve orphaned rebuttal experiments** (P1/P6/P7 trained-not-evaluated; open-vocab EXP-P2 "done" with no artifacts).

**Priority reading.** Blocked on **M4 retrain** (nothing new exists without it) → **UniMTS + ssl-wearables head-to-heads** (reject risk) → **gravity canonicalization** (the one true architectural gap vs UniMTS, unbuilt) → **eval-integrity musts** (§4 GroupKFold CIs, §6 paper corrections, ~0 compute). The entire **M5 application/export** and **dense/streaming-metrics** tracks are the "deployable + unified-objective" story — high narrative value but greenfield/data-blocked; nice-to-have unless the paper's central claim is on-device deployment.

Key docs: `docs/v2/{REDESIGN_PLAN,BUILD_PLAN,design_objective,design_application,design_evaluation,design_tokenizer,research_streaming_design,research_conditioning,CROSSCHECK}.md`, `docs/baselines/RESULTS_V2.md`, memory `halo-{redesign-plan,known-discrepancies,debug-sweep-findings,data-baseline-strategy,repo-cleanup-plan}.md`.

---

## RECOMMENDATIONS (highest-value concrete actions)

1. **Run M4 — retrain `small_deep_fb` end-to-end.** It is the single gate: it unlocks the paper's only new numbers *and* flips conditioning mechanisms (b) rate-tokenizer, (c) resolution mask, (d) physical-time RoPE from inert to active (Section 1). Nothing downstream matters without it.
2. **At the M4 flag-flip, also set the free wins:** `use_mean_pooling=True` (frozen mean-pool, **+5.83 pp** verified over the learnable label bank), `QUEUE_MODE=semantic` (stop false-negative pushing on synonyms, `semantic_loss.py:171`), and de-underscore eval labels (self-consistency 0.86→1.0). All are retrain flags, ~0 extra compute.
3. **Fix the label/channel TEXT FORM, not the encoder.** mpnet is empirically a wash on labels (0.268 vs 0.270) — do **not** pay 2× compute. Instead feed short content-dominant sentences at recognition time ("someone is walking"), which cut confusable cosine 0.483→0.444, and **never** verbose boilerplate (0.483→0.801). Accept that direction/side/order/negation classes (stairs up/down, left/right, in/out) are text-degenerate under any SBERT and hand their separation to the IMU head — do not claim zero-shot text recognition for them.
4. **Add channel-text linguistic augmentation.** It is the one augmentation asymmetry: labels get 80% synonym+template aug but channel descriptions get a fixed 1-string vocabulary (no `augment_channel*` exists). Paraphrase placement/sensor strings so the model isn't brittle to real-world channel descriptions — and note side/axis are text-invisible (left/right arm 0.996), so if they must inform the model they need a numeric/positional injection, not text.
5. **Replace the fixed-random 10k truncation with a stratified per-epoch resampler** (`multi_dataset_loader.py:329-332`). The current cap silently discards ~94% of HHAR and ~79% of free-living CAPTURE-24 for the whole run — the single most valuable in-the-wild signal is throttled to 10k/48k windows.
6. **Build the two head-to-head baselines: UniMTS + ssl-wearables adapters.** Both have released weights; their absence is the primary reject risk. Only crosshar/limubert adapters exist today.
7. **Close the eval-integrity gaps that cost ~0 compute:** GroupKFold rotation for the degenerate 1–2-subject few-shot CIs (shoaib/opportunity/realworld/harth), reframe "two-stage MAE SSL"→single-stage, correct the stale "+11.4 pp" native-rate claim, and drop the leakage-inflated ~85% val for honest subject-disjoint numbers.
8. **Broaden the corpus toward real phone/watch use:** add free-living wrist+pocket data with fine labels and **long (10–30 s) windows**, explicit **transport** classes, and **in-hand/phone-interaction** gestures — the corpus is currently ~61% waist/pocket-phone, ~69% posture/locomotion, ≤2.5 s windows, with transport and phone-gesture coverage effectively absent. Fix the hhar 50 Hz re-conversion and placement/unit metadata mislabels while touching the data, since they corrupt training and channel text respectively.
