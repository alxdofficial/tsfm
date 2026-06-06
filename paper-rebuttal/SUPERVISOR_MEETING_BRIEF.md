# 📋 HALO #1698 — Rebuttal Brief

*One-pager for the supervisor meeting. Brief by design — each line is enough to explain what it tests, what we found, and why it helps.*

## 1. Where we stand
5 reviewers. **3 weak-rejects (A, C, D — all knowledgeable), 2 weak-accepts (B non-expert, E knowledgeable).** To convert, the rebuttal must satisfy the **experts (A/C/D/E)**.

**The concerns that actually matter:**
1. **⭐ NOVELTY / "what is fundamentally new" — the dominant concern (B, C, *and* D all lead with it).** Components (adaptive pooling, channel-independent encoding, MAE+contrastive, CLIP-style alignment) are individually known; reviewers want the **joint insight** — what emerges from handling sensing heterogeneity **and** open-vocabulary *together* — plus a crisp claim of what's new vs. engineering integration. **A framing/writing fix, not an experiment — and the main accept/reject axis.**
2. **OOD generalization overclaimed** — HARTH/VTT looked like total collapse (A, C, E).
3. **Scoring may be inflated** — open-set uses synonym-**group** match; group-vs-exact differs across model types (E).
4. **Unfair comparison** — HALO uses native sampling rate + rich channel text; baselines 20 Hz/limited (C, E).
5. **"open-set" / "open-vocab" overclaimed + under-evidenced** — ZS open-set predicts only over the 87 *training* labels (E, D).
6. **Memory-queue design doesn't match the synonym objective** (A — and A is *correct*).
7. **Ablations not fine-grained enough + no multi-seed/variance** (E).

**Also explicitly raised — don't forget (mostly framing/secondary):**
- **Placement-unknown datasets** (UCI-HAR = "smartphone"): does not-knowing placement cause wrong predictions? Drop them? Tested with/without? (D) — *a concrete ask we don't currently cover.*
- **42% absolute ZS-open too low for deployment?** (D) — reframe (hardest 87-way, chance ~1.1%; deployment = closed/few-shot 53–96%).
- **"Why not just post-process the label mismatches?"** (D) + **text-similarity ≠ sensor-dynamics** (D).
- **Param tradeoff — more params → better?** (D) — Medium *regresses* (overfits).
- **Figure 2: only 4 of 7 visible / not diverse; try more train-test splits** (D).

## 2. Concern → Experiment → Result → Why it helps
Legend: ✅ done (local eval) · ☐ to-run (RunPod) · ✍️ writing.

| # | Concern (reviewer) | Experiment — what it tests | Status | Result | Why it supports the paper |
|---|---|---|---|---|---|
| 0 | **⭐ Novelty / what's new** (B, C, D) | *Framing, not an experiment* — articulate the joint heterogeneity + open-vocab insight + a crisp "what's new" claim | ✍️ | — | **The dominant accept/reject axis.** No experiment fixes it; the rebuttal must argue the positioning. |
| 1 | **OOD collapse** (A, C, E) | **HARTH label-bug fix** • re-run all 5 models | ✅ | HARTH ZS-open **2.0 → 29.1%** (HALO); it was an **eval bug** (HARTH label offset) | Removes the #1 attack. HARTH = **graceful degradation**, not collapse. Honest fix. |
| 2 | **Group-match inflates lead** (E, C) | **EXP-X — exact-match scoring** | ✅ | HALO exact **38.6** (smallest drop of any model). **HALO-exact > every baseline's *group* (≤28.3)** | Lead **widens** under strict scoring (+13.7 → **+17.95 pp**) |
| 3 | **Ranking is protocol-dependent** (E) | **EXP-P5 — scoring sensitivity** | ✅ | HALO **#1 in all 4 cells** | Ranking is **invariant** to scoring choice |
| 4 | **Unfair comparison** (C, E) | **EXP-F — fairness 2×2** (HALO forced to 20 Hz + generic text) | ✅ (LanHAR-native ☐) | HALO @20 Hz+generic = **41.6**, **+13.3 pp over MOMENT** | Win is **architectural**. ⚠ Table 7 native-rate claim is **stale** → replace |
| 5 | **Open-vocab overclaimed** (E, D) | **EXP-P2 — open-vocab eval** | ✅ | Distractors change acc **≤0.2 pp**; paraphrases retrieve **98–133%** | Concrete evidence for the open-vocab claim |
| 6 | **Queue ≠ synonym objective** (A) | **EXP-P1 — queue ablation** | ☐ RunPod ×3 | — | Runs the *exact* ablation A asked for (A is correct) |
| 7 | **No multi-seed** (E) | **EXP-P6 — ≥3 seeds** | ☐ RunPod | — | mean±std → +13.7 pp lead is **seed-robust** *(do first)* |
| 8 | **Ablations not fine-grained** (C, D, E) | **EXP-P7** — temporal-only · **channel-indep vs channel-specific** · pool · soft/hard · conditioning · kernel · τ_s · SBERT | ☐ RunPod | — | Justifies every flagged choice; E named these |
| 9 | **Param tradeoff** (D) | Scaling check | partial | Medium **regresses** (overfits) | Honest scaling story |
| 10 | **Placement-unknown datasets** (D) | Train/test **with vs without** placement-unknown datasets (UCI-HAR); or justify | ☐ small | — | Answers D's concrete ask — *currently uncovered* |

## 3. Supporting numbers
**Core result — 5 main datasets, zero-shot open-set (group, %)** — HALO leads by ~14 pp:

| HALO | MOMENT | LiMU-BERT | CrossHAR | LanHAR |
|---|---|---|---|---|
| **42.0** | 28.3 | 21.7 | 21.2 | 15.8 |

**Severe-OOD, zero-shot open-set (%) — two distinct failure modes:**

| Dataset | HALO | MOMENT | CrossHAR | LanHAR | LiMU-BERT | Reading |
|---|---|---|---|---|---|---|
| **HARTH** (sensor shift) | 29.1 | 17.5 | 30.3 | 9.3 | 0.2 | graceful degradation (most 9–30%) |
| **VTT-ConIoT** (50% novel labels) | 1.3 | 1.6 | 0.7 | 8.3 | 3.4 | genuine collapse — *for everyone* |

- **HARTH** — old "collapse" was a label bug; corrected, HALO recovers to **76.8% @10% labels**.
- **VTT** — inherent zero-shot floor (no training analog for half the activities) → not a HALO defect.
- **LiMU-BERT** HARTH 0.2 open / 54.2 closed = open-set *calibration* failure (note so it isn't used against us).

## 4. Written (non-code) paper changes
**Framing / claims**
- [ ] **⭐ Lead with a crisp contribution statement** — the *joint* heterogeneity + open-vocab insight + exactly what's new vs. integration. **#1 reviewer concern (B, C, D); main accept/reject axis.**
- [ ] Reframe **Stage-1 pretraining** as optional/auxiliary, not a load-bearing "two-stage foundation model" — deployed ckpt is alignment-only (D-1, C1).
- [ ] Add **limitations + remedies** paragraph: moderate-heterogeneity scope; per-sample metadata, calibration sets, domain-adaptive pretraining, uncertainty rejection (A1, E5).
- [ ] Reframe **severe-OOD** into two modes (HARTH = sensor shift / graceful; VTT = novel-label / collapse-for-all) (A1, E3).
- [ ] Justify why **post-processing / label normalization is insufficient** (runtime-unseen labels; per-dataset engineering; text-sim ≠ motion-sim) (D2, D3).
- [ ] Reframe the **42% absolute** (hardest 87-way, chance ~1.1%; deployment = closed/few-shot 53–96% + 97.5% iPhone demo) (D4).

**Numbers / tables**
- [ ] State **42.0% = 5-main average**; report all-7 (≈34.3%) transparently (D-2, C4).
- [ ] Correct **scoring description**: open-set = group-match for all; exact only closed-set text-aligned. Add **exact-match column** (D-3, C2).
- [ ] Standardize on **42.0** (kill stray 46.0) (D-4, C11).
- [ ] **Replace stale Table 7** with the fairness 2×2 / +13.3 pp result (D-6).
- [ ] Update **HARTH column** in Table 5 with corrected numbers.
- [ ] Reconcile **param counts** (35M vs 29.3M; Medium 66.6M) (C16).

**Method text corrections**
- [ ] Correct **conditioning** text → gated `ChannelTextFusion` (C3).
- [ ] Correct **tokenizer** text → kernel-5 + FFT (C5).
- [ ] Fix **synonym count** (3.56 avg, hand-authored — not WordNet) (C10).

**Figures / provenance**
- [ ] Regenerate **Figure 2** with all 7 test datasets (D5).
- [ ] Back **Table 9 (iPhone demo)** + embedding numbers with saved artifacts.

---
*Status: corrected results + 4 experiments (X, P5, F, P2) done & verified. Remaining: RunPod ablation campaign (P1, P6, P7) + the written changes above. Paper LaTeX edits held until greenlit. (Recovered from Notion 2026-06-06 after an accidental untracked-file deletion; now committed so it persists.)*
