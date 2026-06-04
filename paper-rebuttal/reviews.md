# MobiCom'26 Paper #1698 — Reviews (HALO)

Source: `current submission/#1698 - MobiCom 2026.pdf` (HotCRP export, 2026-06-04).
Authors (de-anon in portal): Z. Ding, L. Zhang, X. Ouyang.
**Rebuttal response deadline: ~6 days (≈ June 10, 2026).**
PC conflicts listed: Mo Li, Qian Zhang, Sung-Ju Lee, Zhenyu Yan.

## Scores

| Review | Overall merit | Reviewer expertise |
|--------|---------------|--------------------|
| #1698A | **2 — Weak reject** | 3 — Knowledgeable |
| #1698B | **3 — Weak accept** | 1 — No familiarity |
| #1698C | **2 — Weak reject** | 3 — Knowledgeable |
| #1698D | **2 — Weak reject** | 3 — Knowledgeable |
| #1698E | **3 — Weak accept** | 3 — Knowledgeable |

Mean overall merit = **2.4**. The three Weak-rejects are all *Knowledgeable* reviewers; one Weak-accept (B) is *No familiarity*. The other Weak-accept (E) is Knowledgeable but lists many required improvements.

---

## Review #1698A — Weak reject (expertise: Knowledgeable)

**Summary.** HALO = heterogeneity-aware, language-aligned IMU model for open-set HAR. Stage-1 pretrain (adaptive tokenization, channel-independent modeling, sensor-description conditioning, MAE, patch-level contrastive); Stage-2 aligns IMU↔text via synonym-aware soft contrastive learning. Strong on main held-out datasets; ablations + smartphone deployment.

**Strengths.**
- Motivation clear/important for mobile sensing (rate, channel, placement, label-vocabulary variation); two-stage framework well aligned to challenges.
- Broad evaluation: multiple datasets, held-out tests, several baselines, ablations, fairness controls, model scaling, smartphone latency.

**Weaknesses.**
- Severe-OOD zero-shot results are weak — nearly collapses on HARTH and VTT-ConIoT → raises concern about true foundation-model generalization.
- **FIFO queue may conflict with the synonym-aware objective**: soft targets are built for in-batch labels, but queued samples appear to be hard negatives → may push semantically similar labels apart.

**Comments / questions.**
1. Severe-OOD: very low ZS on HARTH/VTT-ConIoT suggests good transfer to *related* HAR but not to strong sensor shift or new activity domains. **Temper the "foundation model" and "open-set" claims**; discuss the limitation more clearly.
2. Queue objective: adaptive soft targets reduce in-batch synonym conflict, but queue entries seem treated as hard negatives; if queue holds "walking" & "strolling," reintroduces the conflict the method is solving. **Clarify whether semantic similarity is computed over the queue too; add ablation: no queue vs hard-negative queue vs semantic-aware queue.**

---

## Review #1698B — Weak accept (expertise: No familiarity)

**Summary.** Two-stage HALO; adaptive-pooling tokenization (variable rates), channel-independent extraction (variable suites), NL sensor conditioning (modality/axis/placement); Stage-2 synonym-aware soft contrastive alignment; cosine-similarity retrieval vs arbitrary label banks. Trained 10 / tested 7 datasets; ZS open/closed + low-label FT.

**Strengths.**
- Adaptive-pooling tokenization, channel-independent encoding, sensor-description conditioning each map to a real heterogeneity axis.
- Language-aligned inference (swap label banks at runtime, no new classifier) is a practical open-set capability.
- Useful ablations (channel-text conditioning, text aug, signal aug).
- Practical for mobile: 7.16 ms isolated latency on iPhone 16 Pro + iOS case study.

**Weaknesses.**
- **Incremental novelty**: essentially a combination of known ML approaches applied to heterogeneous sensing (MAE, contrastive, channel-independent TS modeling, text embeddings, CLIP-style alignment all well established). Contribution is primarily the combination.

**What could convince acceptance:** N/A given.

---

## Review #1698C — Weak reject (expertise: Knowledgeable)

**Summary.** Language-aligned IMU foundation model; heterogeneity-aware SSL pretraining + IMU–text contrastive alignment for open-set across configs/rates/vocabularies. Improvements over baselines in zero-/low-shot.

**Strengths.**
- Practical problem (cross-device/cross-dataset HAR generalization).
- Heterogeneity handling reasonably designed (adaptive tokenization, channel-independent encoder).
- Extensive evaluation (datasets, ZS transfer, ablations, deployment).

**Weaknesses.**
- **Novelty more incremental than claimed** — most components are combinations/extensions of existing SSL, contrastive alignment, multimodal representation ideas.
- **"Foundation model" framing overstated** given limited training scale; many design choices heuristic and under-justified (synonym mapping, soft-target construction).
- **Comparison may not be fair**: HALO benefits from richer metadata + native-rate inputs; several baselines run under more constrained settings. Severe-OOD reveals it still fails badly under strong shift.

**Comments.** Solid engineering, thorough empirics, but not convinced it reaches MobiCom novelty/conceptual depth. Many contributions are incremental integrations; novelty/generality sometimes overstated. "Foundation model" terminology misleading given corpus scale + semantic coverage. Fairness: even with the Table-7 control, the comparison doesn't fully isolate the architecture's contribution.

**What could convince acceptance:** No.

---

## Review #1698D — Weak reject (expertise: Knowledgeable)

**Summary.** Heterogeneity-aware, language-aligned IMU foundation model for open-set HAR; tackles sensing heterogeneity + open-vocabulary generalization; SSL IMU pretraining + text alignment + synonym-aware contrastive. Well written, experimentally extensive; concerns on novelty, clarity of contribution, practical significance.

**Strengths.**
- Addresses common rate/channel/placement variation problem.
- Extensive evaluation; real-world deployment + smartphone latency strengthen practical relevance.

**Weaknesses.**
- **Combination of existing ideas; unclear what is fundamentally new** vs integration. Adaptive pooling tokenization, augmentation-based contrastive, language alignment presented as adaptations, not novel methods.
- No clear **deeper modeling insight** from addressing heterogeneity + open-set *jointly* rather than separately.
- Key contributions not clearly delineated (novel vs engineering integration).

**Comments / questions.**
- How can cosine-similarity retrieval recognize activities with **no semantic analog** in training? Severe-OOD collapse seems to confirm the limitation. Simpler label-mismatch (walking vs jogging) may be solvable by post-processing / label normalization rather than a new foundation-model framework.
- Is HAR label variation complex enough to justify the synonym-aware focus? Many inconsistencies may be mapped manually/automatically post-prediction. Textual similarity may not correspond to sensor-dynamics similarity (semantically related but physically different).
- Absolute **ZS open-set ≈ 42% is low** — sufficient for real deployment?
- **Figure 2**: test datasets cluster together; "shouldn't there be 7 of them, and more diverse?" Also evaluate additional train/test combinations.
- Datasets without placement annotations (UCI-HAR "smartphone" only): wouldn't unknown placement cause wrong predictions? Better to exclude? Have you tested with/without such datasets in train/test?
- 35M params, 10× fewer than MOMENT — **is there a tradeoff? Can more params improve accuracy?**

**What could convince acceptance:** Clarify what is fundamentally new; stronger justification for why challenges need architectural (not simpler post-processing) solutions.

---

## Review #1698E — Weak accept (expertise: Knowledgeable)

**Summary.** Heterogeneity-aware, language-aligned IMU foundation model; two stages; outperforms strong baselines on main held-out datasets (esp. ZS open-set) but drops sharply under severe OOD.

**Strengths.**
- Important/practical problem; clearly identifies heterogeneity + inconsistent vocabularies as barriers.
- Well-motivated, technically coherent design; each component tied to a motivation challenge.
- Broad comparisons incl. main + severe-OOD datasets.
- Useful practical analysis beyond accuracy (helps understand usefulness + limits).

**Weaknesses.**
- **Evaluation protocol not fully balanced** — HALO gets native rates + rich descriptions; several baselines get 20 Hz + limited inputs (only partial fairness control).
- **"Open-set"/"open-vocabulary" claims need clearer definition + stronger evidence** — ZS open-set predicts over the 87 *training* labels, so it doesn't demonstrate recognition of genuinely novel activity names at inference.
- **Collapses under severe OOD** (near-zero ZS on HARTH/VTT-ConIoT) → weakens foundation-model claim.
- **Ablations not fine-grained enough** — broad modules only; can't isolate adaptive pooling, spectral-temporal tokenization, soft targets, memory queues, synonym augmentation, SBERT choice.

**Comments.**
- Make comparison protocol more systematically controlled; current "richest input each model can consume" makes it hard to isolate gains from architecture vs native-rate vs richer metadata vs language interface. Add comparison with feasible adaptive baselines using same channel sets + rates. Discuss how group-scoring (classifiers) vs exact-string-match (text-aligned) affects numbers.
- Distinguish carefully: "ZS transfer to unseen datasets" vs "closed-set with test labels" vs "open-set over training labels" vs "open-vocabulary with truly novel labels." Add eval with **unseen test labels, paraphrases, fine-grained labels, distractor labels**; report separately for labels with close training synonyms vs genuinely novel labels.
- Severe-OOD discussion too brief vs how strongly it challenges the framing. Add **failure analysis: confusion matrices, nearest text labels, per-activity accuracy, breakdowns by placement/modality/label coverage.**
- Temper claims (effective under moderate heterogeneity, not extreme shift). Discuss remedies: per-sample sensor metadata, small calibration sets, domain-adaptive pretraining, uncertainty-based rejection.
- Add ablations: adaptive-pooling tokenizer vs fixed resampling; temporal-only vs spectral-temporal; channel-independent vs channel-specific. **Report variance over multiple seeds** (esp. since ablation runs use a shorter schedule).

**What could convince acceptance:**
- More controlled comparison isolating modeling choices from input-setting differences (native rate, richer descriptions, scoring).
- Direct open-vocabulary eval with truly unseen labels, paraphrases, fine-grained, distractors.
- Deeper severe-OOD failure analysis on HARTH/VTT-ConIoT.
