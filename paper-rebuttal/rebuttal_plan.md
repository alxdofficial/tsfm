# HALO (MobiCom'26 #1698) — Rebuttal Plan & Progress Tracker

**Living doc.** Status keys: `☐ todo` · `◐ in progress` · `☑ done` · `✗ dropped`.
Companion: [`codebase_audit.md`](codebase_audit.md) (paper↔code discrepancies), [`reviews.md`](reviews.md) (parsed reviews).

- **Deadline:** rebuttal response ~**June 10, 2026** (~6 days from June 4).
- **Scores:** A2(WR,know) · B3(WA,nofam) · C2(WR,know) · D2(WR,know) · E3(WA,know). Mean **2.4**.
- **Strategy:** **harden E → champion**, **convert A**, neutralize C/D, keep B. Address every concrete
  experiment/analysis ask. Temper overclaims *proactively* (pre-empt, don't get caught).
- **Constraint:** no code/paper edits until greenlit. This folder (planning) is fair game.

---

## 0. Headline reframings forced by the audit (decide first)

| # | Issue | Decision needed | Default recommendation |
|---|---|---|---|
| **D-1** | **C1:** released model is alignment-only; "two-stage SSL pretraining" not used in the shipped ckpt. User: pretraining only ever helped early-training *stability*, not final accuracy; "skip it, don't over-explain." | How does the **paper** frame Stage 1? | Reframe to: heterogeneity-aware **architecture** (tokenizer + conditioning, which *is* deployed) + end-to-end alignment training; present SSL pretraining as **optional/auxiliary** (stabilizes early training), not load-bearing. Drop "two-stage foundation model" as the spine. **Avoids the "show me the pretrained checkpoint" trap.** |
| **D-2** | **C4:** 42.0% = 5-main-dataset average; all-7 ≈ 30.4%. | State the averaging convention explicitly? | **Yes** — say "42.0% averaged over 5 main datasets; severe-OOD reported separately." Pre-empts D's "shouldn't there be 7?" |
| **D-3** | **C2:** open-set is synonym-group-scored for HALO too (paper says exact-match for text-aligned). | Add exact-match open-set column? | **Yes** (cheap). Defuses E's scoring concern; show ranking is preserved. |
| **D-4** | **C11:** 46.0 (SCALING.md) vs 42.0 (deployed JSON) for the same model. | Which is canonical? | Use **42.0** (JSON-backed). Footnote/kill 46.0 before submission. |
| **D-5** | Rebuttal format/length limit (single combined box seen in portal). | Confirm limit. | User to check HotCRP. |
| **D-6** | **EXP-F found Table 7 (native-rate +11.4pp) is STALE** (old 4-layer ckpt); doesn't replicate on deployed model. | Re-run Table 7 on deployed model, or remove it & reframe conditioning as the fusion mechanism (generic identity suffices)? | **Recommend: replace Table 7** with the deployed 2×2 + the fairness-regime comparison (HALO@20Hz+generic +13.3pp vs MOMENT) — turns a stale claim into a stronger fairness result. |

---

## Findings log (completed experiments)

- **EXP-X (exact-match scoring)** — branch `rebuttal/experiment/exact-match-scoring`. **HALO's open-set
  lead survives and WIDENS under strict exact match.** 5-main: HALO group 41.97 / exact 38.63 (drop
  −3.34, smallest among models); MOMENT group 28.28 / exact 20.68. Lead +13.7→**+17.95pp** under exact;
  **HALO-exact (38.6) > every baseline's GROUP (≤28.3)**. Group leniency does not favor HALO. Validity:
  HALO group reproduces deployed JSON to 0.000pp (all 7 ds); smoke gate passed. → defuses Reviewer E /
  C2. Paper action: add exact-match column, fix scoring text. See experiment `README.md` + `results/`.

---

## Severe-OOD framing (two distinct, EXPECTED failure modes) — for A1 / D2 / E3

Severe-OOD is **not one cause**. Telling reviewers "it's just a labels problem" is wrong (HARTH
disproves it) — the airtight story decomposes it:

- **VTT-ConIoT = label novelty.** 50% coverage; 8/16 construction activities (climbing ladder,
  roll/spray painting, leveling, lifting, pushing cart, carrying, kneeling work) have **no analog**
  in the 87-label vocabulary → an **irreducible zero-shot floor for *any* label-retrieval method**,
  not a HALO defect.
- **HARTH = sensor/modality shift, NOT labels.** **100% label coverage**, yet ~2% ZS, because
  back/thigh **accelerometer-only (no gyro, gravity-laden)** ≠ waist/wrist phone IMUs in training.
  RESULTS.md: "near-zero ZS accuracy is entirely due to distribution shift, not label coverage."
- **Neither is a representation failure** — both recover to 64–78% with 1–10% labels.

**Root cause (EXP-P3 evidence) = GRAVITY.** Datasets we excel on are smartphone CoreMotion streams
(**gravity-removed** user acceleration); HARTH is **raw Axivity acc with gravity** — a static gravity
vector **132× larger than its motion** (dynamic range 0.01 vs MotionSense 0.2), and **gyro = all zeros**.
The encoder still maps HARTH to the *same* region (centroid cosine **0.975**, identical norms); only the
text↔IMU **alignment inverts** (correct sim 0.836 < max-wrong 0.954, margin −0.118, 2.5% top-1) → it
recovers 2%→78% with 0.5–5% labels. VTT covered-half also fails (2.53%) → VTT = label floor **+** signal
OOD (gravity-laden hip acc, odd-unit gyro), not labels alone.

Terminology to make explicit in the paper: the 5 "main" sets are **entire held-out datasets** (never
trained / validated / tested on) that are distributionally *near* → this is genuine **cross-dataset
zero-shot transfer**, a stronger claim than a held-out test split. Severe-OOD = same signal-format
family pushed on **labels** (VTT) or **sensor placement/modality** (HARTH). Both are expected limits
of a deliberately small model on a bounded corpus → motivates tempering to "effective under *moderate*
heterogeneity," not "universal." **EXP-P3 quantifies exactly this split.**

---

## Findings log addendum

- **EXP-P5 (scoring sensitivity)** — branch `rebuttal/experiment/scoring-sensitivity`. HALO is **#1 in
  all four cells** {open,closed}×{exact,group}: open 38.63/41.97, closed 53.11/53.11 vs best baseline
  MOMENT 20.68/28.28, 44.70/44.70. **Ranking invariant** to scoring; largest margin under strictest
  (open-exact **+18pp**); closed exact≈group. Validity reproduces deployed JSON. → E1 fully answered;
  report exact-match as conservative headline.

- **EXP-F (fairness 2×2)** — branch `rebuttal/experiment/fairness-native-rate`. **(1) Fairness defense
  airtight:** HALO in the baseline input regime (20Hz + generic descriptions) scores **41.56** 5-main
  ZS-open, **+13.3pp over MOMENT (28.3)** — native-rate+metadata net **+0.41pp** → win is architectural.
  **(2) NEW PAPER RISK — Table 7 is stale:** its +11.4pp native-rate benefit was computed on an old
  4-layer ckpt (`tsfm_eval_native_rate.log`: `20260217_113136`, layers=4); does NOT replicate on
  deployed model (combined +0.41, metadata **−3.27**). **(3)** channel-text *fusion* essential
  (ablation −20pp) but generic channel identity ≥ rich placement text → temper conditioning framing.
  → **D-6 below.**

- **EXP-P2 (open-vocab eval)** — branch `rebuttal/experiment/open-vocab-eval`. (1) **Distractor
  robustness = clean win**: 6 absent-activity distractors change accuracy ≤0.2pp, selected 0–0.5% →
  safe vocabulary expansion, no hallucination. (2) **Open-vocab demonstrated but phrasing-sensitive**:
  novel strings absent from training retrieve at 98–133% retention for clear paraphrases
  (standing→"standing still and upright"), degrade for poor ones (frozen-text-encoder artifact, ties
  E5). T0 reproduces deployed closed-set JSON on all 5. Confirms E2/D2 with honest caveat.

- **EXP-P3 (severe-OOD analysis)** — branch `rebuttal/experiment/ood-failure-analysis`. Root cause =
  **gravity** (raw gravity-laden acc, 132× motion; no/odd gyro; odd placement), an **alignment** failure
  not a representation one (HARTH↔train centroid cos 0.975; ranking inverts; recovers 2%→78% @0.5–5%).
  HARTH = signal/alignment (100% labels covered, even running/sitting fail); VTT = label floor (novel
  0.0%) + signal OOD (covered 2.53%). Reproduces deployed JSON; smoke passed. See experiment `results/`.

---

## 1. Reviewer response map

### Reviewer A — Weak reject, knowledgeable → **CONVERT**
- ☐ **A1 (temper foundation/open-set + OOD):** proactively scope claims; OOD collapse affects *all* models (confirms distribution shift, not HALO-specific); SFT recovers (HARTH 78.3@10%). → **EXP-P3**.
- ☐ **A2 (FIFO queue vs synonym objective):** A is *correct* per audit — in-batch is pure soft (`SOFT_TARGET_WEIGHT=1.0`), queue entries are hard negatives, similarity **not** computed over the queue. Run the exact ablation A asked for. → **EXP-P1**.

### Reviewer B — Weak accept, no familiarity → **KEEP** (light touch)
- ☐ **B1 (incremental novelty):** novelty reframing (systems capability + heterogeneity×open-set jointly). No B-specific experiment.

### Reviewer C — Weak reject, knowledgeable → **NEUTRALIZE**
- ☐ **C1 (novelty):** reframing.
- ☐ **C2 (foundation overstated; heuristics under-justified):** report corpus scale honestly; justify synonym maps + soft targets; τ_s sensitivity. → **EXP-P7** (τ_s sweep) + framing.
- ☐ **C3 (unfair comparison + OOD):** → **EXP-F** (HALO@20Hz + LanHAR@native) + **EXP-P5** + Table-7 emphasis ("native rate isn't free — RealWorld −9.9pp").

### Reviewer D — Weak reject, knowledgeable → **NEUTRALIZE**
- ☐ **D1 (what's fundamentally new / joint insight):** reframing. (Note: **EXP-P4 dropped** — see §2.)
- ☐ **D2 (recognize no-analog activities? post-processing enough?):** clarify open-vocab taxonomy; post-hoc mapping needs per-dataset engineering + can't handle runtime-unseen labels. → **EXP-P2**.
- ☐ **D3 (synonym focus justified? text-sim ≠ motion-sim):** quantify cross-dataset label conflicts; soft-vs-hard ablation; corr(text-sim, IMU-sim) analysis. → **EXP-P7** + analysis.
- ☐ **D4 (42% too low?):** reframe (hardest 87-way, chance ~1.1%); deployment = closed/few-shot 53–96% + demo 97.5%.
- ☐ **D5 (Figure 2: 4 vs 7 test sets, more diverse; more train/test combos):** regenerate Fig 2 with all 7 (severe-OOD are the outliers). Optional alt-split robustness.
- ☐ **D6 (placement-unknown datasets):** per-dataset rich-vs-generic deltas (have Table-7 numbers); generic-desc already in Table 7.
- ☐ **D7 (param tradeoff, more→better?):** Table 8 — Medium **regresses** (overfits at this corpus scale). Reconcile 46.0/42.0 first (D-4).

### Reviewer E — Weak accept, knowledgeable → **HARDEN → champion** (defines the slate)
- ☐ **E1 (balanced protocol; group-vs-exact scoring):** → **EXP-F + EXP-P5** (incl. exact-match open-set column).
- ☐ **E2 (define + evidence open-vocab):** 4-level taxonomy (ZS-transfer / closed-set / open-set-over-train-labels / true open-vocab); eval w/ unseen+paraphrase+fine-grained+distractor, split by synonym-distance. → **EXP-P2** (headline new result).
- ☐ **E3 (deep OOD failure analysis):** confusion/nearest-label/per-activity/placement-modality-coverage. → **EXP-P3**.
- ☐ **E4 (fine-grained ablations + multi-seed):** → **EXP-P1 + EXP-P6 + EXP-P7**.
- ☐ **E5 (temper + remedies):** limitations paragraph: per-sample metadata, calibration sets, domain-adaptive pretraining, uncertainty rejection.

---

## 2. Experiment slate (tracker)

Type: **EVAL** = inference-only, runs on local RTX 4090. **TRAIN** = Stage-2 retrain → **RunPod** (parallel).
All outputs → `paper-rebuttal/experiments/<id>/`.

| ID | Experiment | Addresses | Type | Where | Key files to change | Status |
|----|-----------|-----------|------|-------|--------------------|--------|
| **EXP-X** | **Exact-match open-set column** for HALO + LanHAR (vs group-match) | C2, E1, **D-3** | EVAL | local | `evaluate_tsfm.py:526–544`, `grouped_zero_shot.py` (`score_exact` vs `score_with_groups`) | ☑ **DONE** |
| **EXP-F** | Fairness/native-rate: **HALO through 20Hz/120/6-ch** pipeline (all 5) + **LanHAR at native 50Hz** | C3, C7, A3, E1 | EVAL | local | `evaluate_tsfm.py` (20Hz data path exists), `evaluate_lanhar.py:1295` (load `data_native.npy`, fix filter fs) | ☑ **HALO 2×2 DONE** · LanHAR-native = RunPod |
| **EXP-P2** | Open-vocab eval: novel + paraphrase + fine-grained + distractor labels, split by synonym-distance | D2, E2 | EVAL | local | `evaluate_tsfm.py` candidate-set construction; `label_augmentation.py` | ☑ **DONE** |
| **EXP-P3** | OOD failure analysis (HARTH **+ VTT**): confusion, nearest-text-label, per-activity, placement/modality/coverage breakdowns | A1, C3, D2, E3 | EVAL | local | generalize `harth_analysis.py` → VTT; add breakdowns | ☑ **DONE** |
| **EXP-P5** | Scoring-protocol sensitivity (exact↔group, both directions, all models) | C2, E1 | EVAL | local | `grouped_zero_shot.py`, `evaluation_metrics.py` | ☑ **DONE** |
| **EXP-P1** | Queue ablation: **none / hard-neg (current) / semantic-aware** queue | A2, E4 | TRAIN ×3 | RunPod | `memory_bank.py` (store label identity/text emb), `semantic_loss.py` (target matrix over queue), add `TSFM_QUEUE_MODE` | ☐ |
| **EXP-P6** | Multi-seed (≥3) for headline + key ablations; report mean±std on +13.7pp | A6, E4 | TRAIN ×N | RunPod | `run_ablations.sh` seed loop; eval | ☐ |
| **EXP-P7** | Fine-grained ablations: spectral-temporal **vs** temporal-only; **soft vs hard** targets; adaptive-pool **vs** fixed-resample; **additive vs gated** conditioning (C3); kernel `[5]` **vs** `[3,5,7]` (C5); τ_s sweep; SBERT swap | C2, D3, E4 | TRAIN ×several | RunPod | `config.py`, `feature_extractor.py`, `semantic_loss.py`, `token_text_encoder.py` | ☐ |
| ~~EXP-P4~~ | ~~No-Stage-1-pretrain ablation~~ | — | — | — | **✗ DROPPED** — user: tried before, marginal; released model is *already* pretrain-free (C1); pretraining only aided early-training stability, not final accuracy. Don't over-explain. | ✗ |

**Quick-win ordering (start here, all EVAL on the 4090):** EXP-X → EXP-F → EXP-P3 → EXP-P2 → EXP-P5.
These pre-empt the scoring/fairness/OOD attacks and need only small *additive* eval-script changes.
**RunPod track (parallel):** EXP-P6 (cheapest, highest value) → EXP-P1 → EXP-P7.

### Fairness / native-rate detail (EXP-F)
- **Lead argument:** Table 7 already shows HALO handicapped *down* to 20Hz+generic still wins; native
  rate is **not a free lunch** (+24.5 MotionSense, +19.7 MobiAct, but **−9.9 RealWorld**). Reframe
  "unfair advantage" → "capability with costs."
- **Baselines are architecturally rate-locked** (LiMU-BERT/CrossHAR fixed 120-pos@20Hz; MOMENT 512-pos
  no Hz) — that limitation *is* HALO's contribution. Don't fake-handicap baselines (input-distribution
  shift would make them look worse → bad faith).
- **Concrete runs:** (a) HALO @ 20Hz/120/6-ch on all 5 main (extend Table 7 from 3→5); (b) fix LanHAR
  to native 50Hz; (c) MOMENT is the only rate-flexible baseline — optional native-rate run.

---

## 3. Paper fixes (HELD until greenlight; tracked)

| # | Fix | Trigger | Status |
|---|-----|---------|--------|
| ☐ | Reframe Stage-1 (optional/auxiliary, not load-bearing two-stage) | D-1, C1 | ☐ |
| ☐ | State 42.0% = 5-main average; report all-7 transparently | D-2, C4 | ☐ |
| ☐ | Correct scoring description (open-set = group for all; exact only closed-set text-aligned) + add exact-match column | D-3, C2 | ☐ |
| ☐ | Kill 46.0 / standardize on 42.0; regen Medium/Tiny scaling JSONs | D-4, C11 | ☐ |
| ☐ | Correct conditioning method text → gated `ChannelTextFusion` (or run additive-vs-gated ablation) | C3 | ☐ |
| ☐ | Correct tokenizer text → kernel-5 + FFT (or retrain with `[3,5,7]`) | C5 | ☐ |
| ☐ | Fix synonym count (3.56, hand-authored not WordNet) | C10 | ☐ |
| ☐ | Regenerate Figure 2 with all 7 test datasets | D5 | ☐ |
| ☐ | Limitations + remedies paragraph | A1, E5 | ☐ |
| ☐ | Reconcile param counts (35M vs 29.3M vs Medium 66.6M) | C16 | ☐ |
| ☐ | Back Table 9 (iPhone) + embedding numbers with saved artifacts | provenance | ☐ |

---

## 4. Execution / infrastructure

**Verification log:**
- ☑ **HALO small_deep_v2 reproduces on master** — `rebuttal/experiment/verify-reproduction`,
  `paper-rebuttal/experiments/verify_reproduction/verify_tsfm_zs.py`. ZS (pooled+MV, open+closed) on
  5 datasets = deployed JSON **exactly** (max |Δ|=0.000pp). Master's refactor did not break loading.
- ☑ **Baselines reproducible by construction** — `evaluate_{moment,limubert,crosshar,lanhar}.py`,
  `grouped_zero_shot.py`, `evaluation_metrics.py` are **byte-identical** master↔eval-branch (refactor
  touched only HALO-side files); cached ZS classifiers present; data frozen. Same code+classifier+data.
- **Note for EXP-X:** baseline JSONs already carry `accuracy_exact` AND `accuracy_group`; only
  HALO/LanHAR open-set emit a single group number → EXP-X just adds the exact column for those two.

- **Local (RTX 4090 24GB):** all EVAL experiments; Small-Deep Stage-2 trains fit here too (slower, serial).
- **RunPod (TRAIN track):** parallelize EXP-P1/P6/P7. Medium needs >48GB. Plan: tmux session driving
  RunPod SSH; `scripts/setup_runpod.sh` + `training_output/runpod_ablations/` already exist in repo.
  **Awaiting:** user's RunPod workflow notes + API key / pod-launch recipe.
- **Checkpoint:** `training_output/semantic_alignment/small_deep_v2_4b3fdd6/best.pt` (paper "Small").
  ⚠ `evaluate_tsfm.py` defaults to `small_v1` — **must set `TSFM_CHECKPOINT`** to the small_deep ckpt.

## 5. Open decisions / waiting on user
- ☐ D-1 (Stage-1 paper framing) — **needs user call.**
- ☐ D-5 (rebuttal length limit) — user checking HotCRP.
- ☐ Greenlight to start touching **eval** code for the quick-win experiments (additive, non-destructive).
- ☐ RunPod workflow + credentials.
