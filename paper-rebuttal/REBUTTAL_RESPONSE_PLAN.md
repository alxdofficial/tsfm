# MobiCom'26 #1698 (HALO) — Rebuttal Response Plan

> Working doc to draft the actual ~500-word response. Captures supervisor direction (meeting 2026-06-06).
> Companion to `reviews.md` (concerns), `novelty.md`, `rebuttal_plan.md` (earlier planning). Response deadline ≈ **June 10**.

---

## HARD CONSTRAINTS (from supervisor)

- **~500 words total. Point form, not prose.** Space is the binding constraint — every line must earn its place.
- **3 sections**, each in point form:
  1. **Positioning / Framing** — our novelty argument; how HALO is different / not just a combination.
  2. **Technicalities & Evaluations** — fairness, metrics, missing evals, protocol. Supply results.
  3. **Design** — rationale for design choices (memory queue, soft vs hard targets, why architectural not post-processing). Mostly argumentation, little new experimentation.
- **Per-concern point-form format:**
  - **Concern** (one line) + **(which reviewers raised it)** in brackets.
  - Underneath: 1–3 supporting lines — argument, and/or a table / result pointer.

## PRIORITIZATION (from supervisor)

- **Prioritize the Weak-REJECT reviewers: A, C, D** (all *Knowledgeable*). These flip the decision.
- **Don't over-invest in Weak-accepts: B (No familiarity), E (Knowledgeable).** MobiCom is selective — weak accept is already a good outcome. (E still gives the most actionable asks; address cheaply where it overlaps A/C/D.)

| Review | Merit | Expertise | Priority |
|---|---|---|---|
| **A** | Weak reject | Knowledgeable | **HIGH** |
| **C** | Weak reject | Knowledgeable | **HIGH** |
| **D** | Weak reject | Knowledgeable | **HIGH** |
| B | Weak accept | No familiarity | low |
| E | Weak accept | Knowledgeable | medium (cheap overlaps) |

## STRATEGY RULES (from supervisor — venue culture)

- **⚠️ CORRECTION (venue-verified, overrides supervisor's "use new results"):** The **MobiCom 2026 CFP FORBIDS new experiments, new data, or new figures in the rebuttal**, and forbids *promising* additional work. (Verbatim, fetched 2026-06-07 from sigmobile.org/mobicom/2026/cfp.html.) Permitted content is narrow: **(a) correct factual errors in the reviews, or (b) directly answer reviewer questions.** Both of our group's own winning rebuttals (#293 MobiCom'24, #162 MobiSys'23) confirm the pattern — they are ~80% pointers to *already-submitted* `Section#/Figure#/Table#`, with essentially no new experiments.
  - **The supervisor's "new results are fine" is almost certainly a MobiSys-ism** — MobiSys explicitly *does* allow new experiments in rebuttal; **MobiCom 2026 is stricter.** → **Flag to supervisor before drafting.**
  - **So: P1/P6/P7 ablation numbers CANNOT be shown as new tables/figures in the 500-word rebuttal.** Use them only as the *substance* of a clarifying answer (e.g. P1 → "the queue stores only hard negatives; in-batch supervision is purely soft — clarifying R-A's question"), never as "we ran a new ablation showing…". The actual tables are **REVISION deliverables**, not rebuttal content.
  - Preference order *within the rebuttal*: (1) point to an existing in-paper result; (2) argue from existing results/logic; (3) where neither closes it, signal the concern is **fixable in a one-shot revision** (don't promise — "this is addressable in a single revision" framing).
- **DO NOT admit the HARTH heart-/activity-label bug.** Venue culture: don't volunteer a bug. **Argue around** the severe-OOD/HARTH point instead (temper the OOD framing; redirect to strong main-dataset evidence). Never reference the label fix.
- Don't "argue in general" on objective points — if a claim is objective and we can't back it with existing results, that's where (and only where) a new result goes.

---

## TEMPLATE FROM OUR GROUP'S WINNING REBUTTALS (#293 MobiCom'24, #162 MobiSys'23 — both by X. Ouyang, our co-author)

Both were **exactly ~500 words**, both **won → went to revision/shepherding → accepted**. Copy this pattern:
- **Opening line (verbatim-ish):** *"We sincerely appreciate the reviewers' constructive feedback. The following are our responses to the major concerns."*
- **Theme-grouped numbered sections** (e.g. "1. Design Considerations", "2. Comparison with related work", "3. Technical Contributions") — group by TOPIC, not by reviewer. Our 3 sections (Positioning / Eval / Design) fit this.
- **Each point: `+ Topic (Review#A/B):`** then `−` sub-bullets with the answer. The `(Review#X/Y)` tag is exactly our format.
- **~80% of every answer is a pointer to EXISTING paper content** — `Section#6.2.3`, `Figure#16`, `Table#2`, `Figure#9(b)`. This is the dominant move. **Map each concern → a specific existing Section/Figure/Table.**
- **Concede-then-redirect tone:** *"We agree that … while …"* then pivot. Never combative.
- **A "Key Novelties" sub-block:** *"We are the **first** to …"*, *"We propose the **first** …"*, *"Such a design is **not straightforward** without our key findings."* + contrast to NAMED prior work (FedHGB, ClusterFL, FedAvg) with the precise delta.
- **GOAL is to reach REVISION, not direct accept.** Both ours did: rebuttal → TPC "move to shepherding" → revision-plan PDF → revised manuscript → accept. So make objections look **fixable in one bounded revision** (CFP: reviewers name ≤3 major changes; revision MAY include new experiments). That is where P1/P6/P7 live.

## ⚠️ LANDMINES (flagged by the deep paper re-read — verify, but treat as real; tie to "don't admit the bug")

The submitted paper contains numbers that DIFFER from our corrected/deployed results. Surfacing the corrected ones in the rebuttal would (a) contradict the submission, (b) count as "new data" (forbidden), (c) expose the HARTH label bug. **Do NOT surface corrected numbers; lean on already-accepted framings:**
- **HARTH severe-OOD:** submitted Table 5 = HALO 2.0%; paper text says "all models collapse → genuine sensor shift," and **A/D/E already accepted that framing**. Do NOT introduce the corrected ~29% recovery. Temper to "effective under moderate heterogeneity."
- **Table 7 native-rate delta is STALE** (printed +11.4pp; deployed ~+0.41pp). LEAD with the safe row instead: HALO @20Hz+generic = 30.6/41.6 **still beats** MOMENT 28.3 → "wins even handicapped to baseline inputs." Don't lean on the native-rate delta.
- **Fig 2 actually plots 4 test datasets, not 7** — Reviewer D is **factually right**. Concede precisely ("Fig 2 shows only the 4 main datasets with channel metadata"); note as a revision fix; do NOT promise a regenerated figure.
- **42% / 8-metric sweep are over the 5 MAIN datasets only** (exclude the 2 severe-OOD) — OWN this convention explicitly to pre-empt the cherry-pick read (answers D's "shouldn't there be 7?").
- **Scoring asymmetry (group-score vs exact-match):** answer E's question WITHOUT contradicting the paper's own claim (it states text-aligned used exact match); don't introduce a conflicting table.

## CHEAP HIGH-VALUE MOVES (near-zero words, multi-reviewer payoff)
- **Adopt Reviewer E's 4-term taxonomy VERBATIM** (zero-shot transfer to unseen datasets / closed-set with test labels / open-set over training labels / open-vocabulary with truly novel labels) → answers E's "define open-set" AND C/D's "overclaimed" in one move.
- **Open Positioning with the CFP's own words:** novelty lives in *techniques, system designs, implementations, AND applications* — anchors HALO as a system/application contribution, not a new ML primitive.
- **Param-scaling is a WIN, not a weakness (D):** Table 8 shows the Medium model overfits/regresses at this corpus scale → 35M is *sufficient*; scale is corpus-bounded, not architecture-bounded (35M beats 341M MOMENT).
- **Quote E's 4 remedies as "limitations we identify"** (per-sample metadata, calibration sets, domain-adaptive pretraining, uncertainty rejection) — goodwill; phrase as limitations, NOT promised work.
- **A is the highest-ROI flip:** A is novelty-silent + warmest; A's 2 asks are narrow (temper claims + the one queue clarification). Convert A cheaply in §3.

## REVISION-RESERVED RESULTS (P1/P6/P7) — NOT presentable as new experiments in the rebuttal

The fixed-code ablations finishing tonight map directly onto reviewer asks — but per the CFP they are **revision deliverables**, usable in the rebuttal ONLY as the substance of a clarifying answer (no new table/number/figure):

| Running ablation | Answers concern | Reviewer |
|---|---|---|
| **P1 queue: none / hard-neg / semantic** | "FIFO queue may conflict with synonym-aware soft targets; add ablation no-queue vs hard-neg vs semantic-aware" | **A** (exact request) |
| **P6 headline ×3 seeds** | "Report variance over multiple seeds (ablations use a shorter schedule)" | E |
| **P7 fine-grained** (temporal-only, channel-indep, no-channel-text, cnn-multi, spectral-half, hard-targets, τ, soft-weight, SBERT-mpnet) | "Ablations not fine-grained enough — can't isolate pooling, spectral-temporal, soft targets, memory queue, synonym aug, SBERT" | E |
| Completed canaries: P6_headline=72.0%, P7_temporal_only=66.7%, P7_no_channel_text=62.4% | no_channel_text 62.4% vs headline 72% = ChannelTextFusion (a novel component) matters | (supports Positioning) |

---

## SECTION 1 — POSITIONING / FRAMING  (≈ 180 words; the make-or-break section, 3 weak-rejects here)

- **"Incremental / just a combination of known ideas" (B, C, D).**  ← top priority
  - Argue: the *contribution is the unified formulation*, not any single block — jointly solving sensing heterogeneity (rate/channel/placement) AND open-vocabulary labels in one IMU↔text model is new; prior work handles these separately.
  - Point to existing paper evidence: the heterogeneity components (adaptive tokenization, channel-independent, sensor-description conditioning) + the language interface are co-designed; ablations show each is load-bearing (cite Table X; reinforce with no_channel_text 62.4% vs 72%).
- **"Foundation model" / "open-set" framing overstated (C, D, E).**
  - Concede terminology precisely, don't retreat on substance: define our claim as *open-vocabulary retrieval over arbitrary label banks at runtime* (demonstrated), distinct from "open-world novelty." Reframe rather than over-claim.
- **No deeper insight from joint heterogeneity+open-set (D).**
  - Argue the joint treatment is the insight: a shared text space is what lets one model absorb heterogeneous channel/rate configs AND swap vocabularies — neither is achievable by the separate-solution baselines.
- **TODO:** pull the 2–3 sharpest "what is fundamentally new" lines from `novelty.md`.

## SECTION 2 — TECHNICALITIES & EVALUATIONS  (≈ 180 words; supply results / point to paper)

- **Comparison not fair — HALO gets native rates + rich metadata; baselines 20 Hz + limited inputs (C, E).**  ← high (C is weak-reject)
  - FIRST point to the existing fairness control (Table 7) reviewers acknowledge but discount; argue what it already isolates.
  - If needed, add the "same channel-set + rate" controlled comparison E asks for (check what's already in paper before running new).
- **Severe-OOD collapse on HARTH / VTT-ConIoT (A, D, E).**  ← **DO NOT mention label bug**
  - Argue around: temper to "effective under moderate heterogeneity"; redirect to strong main held-out results; frame severe-OOD as scoped future work, not a refutation.
- **ZS open-set ≈ 42% "low" (D); open-set predicts over 87 *training* labels, not novel (E).**
  - Clarify the metric definitions (ZS-transfer vs closed-set vs open-set-over-training vs open-vocab); point to where the paper already separates these.
- **35M vs MOMENT 10× — tradeoff? (D)** → point to scaling results already in paper.
- **Variance over seeds (E)** → P6 multi-seed result. **Fig-2 clustering / more train-test combos / placement-annotation datasets (D)** → argue from existing; new only if objective.

## SECTION 3 — DESIGN  (≈ 140 words; argumentation, minimal new experiments)

- **Memory queue conflicts with synonym-aware soft targets (A).**  ← high (A weak-reject, explicit ask)
  - Clarify whether semantic similarity is computed over the queue; **back with P1 ablation (none vs hard-neg vs semantic-aware queue)** — this is the one place a new result is clearly warranted.
- **Why soft targets not hard targets?**
  - Argue: synonyms/paraphrases in a batch make hard targets contradictory (treats "walking"≈"strolling" as negatives); soft targets weight by semantic similarity. Back with P7 hard-targets ablation if it helps.
- **Why architectural, not simpler post-processing / label normalization? (C, D).**  ← high
  - Argue: post-hoc label mapping needs a fixed closed label set + can't fix the *representation* under rate/channel shift; the heterogeneity problem is upstream of labels. Textual≈sensor-dynamics similarity objection (D): concede partial, argue empirical alignment holds.

---

## WORD-BUDGET SKETCH (≈500)
- Intro/thanks: ~20 · Positioning: ~180 · Eval: ~180 · Design: ~120. Cut ruthlessly; tables/numbers count toward the limit.

## NEXT STEPS (draft tomorrow)
1. Pull exact existing-result pointers (Table/Figure numbers) from `_paper_text.txt` + `RESULTS.md` for every "point to paper" item above.
2. Slot tonight's P1/P6/P7 numbers where reserved.
3. Write the ~500-word point-form response; trim to budget.
4. (Optional) mirror to Notion alongside the existing brief.
