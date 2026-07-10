# Representation Collapse / Anisotropy in Text Embeddings, and How Contrastive & Cross-Modal Alignment Mitigate It

**Date:** 2026-07-05
**Scope:** Literature review + HALO-specific recommendations. Motivated by the empirical probe in
`docs/v2/data_diversity_audit.md` §4, which found HALO's frozen SentenceBERT (all-MiniLM-L6-v2)
activity-label embeddings collapse on *fine* axes — stairs up↔down cos 0.87–0.93, left↔right
0.89–0.996, sit-to-stand↔stand-to-sit 0.96, cycling↔not-cycling 0.88 — while *coarse* activities
(walk/run/sit/stand) separate cleanly at 0.2–0.5. Upgrading MiniLM→mpnet-768 was a statistical wash.
The team already applies a "mean-subtraction" step in the contrastive objective.

> **Reading note (precise statement of HALO's current fix).** What HALO calls "mean-subtraction" is
> *standardization of the frozen text–text similarity matrix used to build the InfoNCE soft targets*:
> `semantic_loss.py:159–162` computes `sim = soft_text @ soft_text.T` then
> `sim = (sim − sim.mean()) / sim.std() / soft_target_temperature` before a softmax. This spreads and
> sharpens the *soft-target distribution*. It is **not** mean-centering or whitening of the *anchor
> embeddings* — the vectors actually dotted against the IMU at train time (`semantic_loss.py:141
> logits = imu @ all_text.T`) and at eval (`eval_common.py:434 logits = emb @ text_embs.T`) are the raw
> normalized label embeddings, with **no** centering, whitening, or standardization applied. This
> distinction drives several recommendations in §5.

---

## Executive summary (~20 lines)

1. **Two different problems wear the name "collapse."** (a) *Global anisotropy / cone effect* —
   transformer embeddings occupy a narrow cone so random pairs have high average cosine
   (Ethayarajh, EMNLP 2019). (b) *Local semantic degeneracy* — near-synonyms/antonyms map to nearly
   identical vectors. HALO's probe (mean off-diagonal cos 0.27, not smeared) shows its problem is
   **mostly (b), not (a)**: a bag-of-words encoder that ignores direction/side/order/negation.
2. **The classic anisotropy fixes are linear post-hoc transforms of the frozen space:** mean-centering,
   all-but-the-top (Mu & Viswanath, ICLR 2018), whitening / BERT-flow (Li et al., EMNLP 2020; Su et al.,
   2021), and per-dimension standardization to kill "rogue dimensions" (Timkey & van Schijndel,
   EMNLP 2021). They demonstrably improve isotropy and STS correlation at ~zero cost.
3. **But no linear transform can separate two vectors that are already near-collinear.** Whitening
   beats mean-subtraction for (a); for HALO's (b) axes (up/down, left/right, in/out, order, negation)
   there is a **hard ceiling** — the discriminative token barely moves the vector, so the separation
   must come from either *richer text content* or *the sensor signal*, not from re-scaling the frozen
   space.
4. **Contrastive learning fixes anisotropy on the *trained* side** by optimizing uniformity
   (Wang & Isola, ICML 2020; SimCSE, Gao et al., EMNLP 2021) — but HALO's text tower is **frozen**, so
   uniformity/margin terms can only reshape the **IMU** embeddings, never the text anchors.
5. **Cross-modal specifics:** CLIP-style models show a *modality gap* (Liang et al., NeurIPS 2022) from
   the init cone + contrastive optimization, modulated by the learnable temperature; recent work ties
   it to low uniformity (Fahim et al., 2024) and to a few dominant dimensions (Schrodi et al., 2024).
   **Aligning to a frozen text tower makes the text geometry immutable** — its collapse becomes a hard
   constraint the sensor side cannot fix.
6. **Comparable sensor↔text systems dodge label collapse by not using bare labels:** UniMTS (NeurIPS
   2024), IMU2CLIP (EMNLP 2023 Findings), LanHAR (IMWUT 2025), GOAT (IMWUT 2024) all align to
   **LLM-generated rich activity descriptions / narrations**, not class names. Ghosh (arXiv 2026)
   attacks exactly HALO's setup (frozen text, zero-shot HAR) with separability-optimized prototypes.
7. **HALO cheap wins:** (i) apply **whitening / all-but-the-top / standardization to the anchor label
   matrix** (currently un-isotropized — a free, unexploited lever); (ii) short content-dominant label
   phrases ("someone is walking", not verbose boilerplate — HALO's own probe: 0.483→0.444 vs 0.801);
   (iii) tune/learn the InfoNCE **temperature**; (iv) fix the semantic queue so antonyms are true
   negatives (`QUEUE_MODE=semantic`, already flagged).
8. **HALO research bets:** (v) a **uniformity term** and/or (vi) an **angular-margin / separability
   loss** on the IMU→text logits for confusable pairs; (vii) **LLM description augmentation** injecting
   the *physics* of the distinction ("ascending … against gravity" vs "descending … controlled
   lowering").
9. **Text-degenerate axes to concede:** left/right, X/Y/Z axis, and A→B vs B→A order are
   **mirror-symmetric or function-word distinctions SBERT cannot encode** — carry them in the *signal*
   (and a numeric/positional side-channel for side/axis), and do **not** advertise zero-shot *text*
   recognition for them.

---

## 1. The anisotropy / "cone effect" problem

### 1.1 Contextual embeddings live in a narrow cone (Ethayarajh, EMNLP 2019)
Ethayarajh's *"How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT,
ELMo, and GPT-2 Embeddings"* (EMNLP-IJCNLP 2019) is the canonical diagnosis. Measuring the average
cosine similarity between representations of **randomly sampled** words, he found it is far above zero
in every layer of BERT/ELMo/GPT-2 and *increases with depth* (e.g., in GPT-2's last layer two random
words have cosine ≈ 1.0). Because all vectors point in roughly the same direction, they occupy a
**narrow cone** rather than being spread over the sphere — the space is **anisotropic**. He argued this
is an inherent by-product of contextualization. Consequence for HALO: raw cosine is an inflated,
biased similarity — a "0.9" between two labels is not as meaningful as it looks, and small true
differences are compressed near the cone axis.

### 1.2 Anisotropy hurts sentence semantics; the frequency bias (Li et al. / "BERT-flow", EMNLP 2020)
*"On the Sentence Embeddings from Pre-trained Language Models"* (Li, Zhou, He, Wang, Yang, Li, EMNLP
2020) — the paper usually shorthand'd **BERT-flow** — showed that averaged BERT embeddings induce a
**non-smooth, anisotropic** sentence space that correlates poorly with human similarity, and that the
space is **biased by word frequency**: high-frequency words sit closer to the origin and to each other,
distorting distances. Their fix maps the anisotropic distribution to an isotropic Gaussian via a
learned **normalizing flow**, giving large STS gains.
*(Note: the task brief attributed BERT-flow to "Gao et al." — the correct first author is **Bohan Li**.
"Gao et al. 2021" is SimCSE, §1.5.)*

### 1.3 Whitening does the same thing with a linear map (Su et al., 2021; Huang et al., 2021)
Su, Cao, Liu, Ou, *"Whitening Sentence Representations for Better Semantics and Faster Retrieval"*
(2021) showed that classic **whitening** — subtract the mean, then apply the PCA transform
`W = U Λ^{-1/2}` so the covariance becomes the identity — recovers essentially all of BERT-flow's gain
with **no learned model**, a few lines of linear algebra, and the bonus of **dimensionality reduction**.
Huang et al., *"WhiteningBERT"* (Findings of EMNLP 2021) corroborated that a <10-line whitening step
consistently boosts unsupervised sentence embeddings. Take-away: **isotropizing the frozen space is
cheap and effective — and whitening ≥ mean-centering**, because it also equalizes per-direction
variance, not just recenters.

### 1.4 A few "rogue dimensions" dominate cosine (Timkey & van Schijndel, EMNLP 2021)
*"All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational
Quality"* (EMNLP 2021) sharpened the picture: a **handful of outlier dimensions** with enormous
variance dominate cosine and Euclidean distance, so measured similarity reflects those 1–5 rogue axes
rather than semantics. Their fix is **standardization (z-scoring each dimension)** — or subtracting the
mean and dividing by the std — which reweights all dimensions and *dramatically* improves correlation
with human similarity judgments, often more than removing dimensions outright. This is the closest
classical analog to what HALO does — but HALO standardizes the *similarity matrix* for soft targets,
not the *embedding dimensions* of the anchors (see reading note).
Related earlier post-processing: Mu & Viswanath, *"All-but-the-Top: Simple and Effective Postprocessing
for Word Representations"* (ICLR 2018) — subtract the common mean vector **and** project out the top
~d/100 principal (dominating) directions.

### 1.5 The contrastive view: alignment + uniformity (Wang & Isola, ICML 2020; SimCSE, EMNLP 2021)
Wang & Isola, *"Understanding Contrastive Representation Learning through Alignment and Uniformity on
the Hypersphere"* (ICML 2020) decomposed what a good contrastive space needs into two measurable
properties: **alignment** (positive pairs close) and **uniformity** (features spread evenly over the
unit sphere, formalized via a Gaussian-potential energy). Anisotropy is exactly *low uniformity*.
Gao, Yao, Chen, *"SimCSE"* (EMNLP 2021) connected this to sentence embeddings: their unsupervised
objective (dropout-as-augmentation positives) **flattens the singular-value spectrum** of the
embedding matrix and **improves uniformity while keeping alignment**, directly counteracting the
BERT anisotropy — and they show removing the augmentation causes **representation collapse**. The
supervised variant adds NLI **hard negatives** (contradictions), which is the mechanism most relevant
to separating *near-synonyms*.

### 1.6 Which problem does HALO actually have?
HALO's probe (`data_diversity_audit.md` §4) reports off-diagonal cosine **mean 0.268 / median 0.247**
over 87 labels, with only 1.2% of pairs > 0.8. That is **not** a globally smeared cone — the coarse
taxonomy is well spread. HALO's failure is **local semantic degeneracy**: the top-cosine pairs are
antonyms/near-synonyms (`up/down` 0.87–0.93, `left/right` 0.89–0.996, `sit-to-stand/stand-to-sit`
0.96, `cycling/not cycling` 0.88). MiniLM is behaving as a **bag-of-words** encoder that ignores
direction, side, order, and negation. This matters for the fixes: §1.1–1.4 techniques cure *global
anisotropy* (problem a); HALO mostly has *lexical/semantic degeneracy* (problem b), for which linear
re-isotropization has a **ceiling** (§2.3).

---

## 2. The fixes and their trade-offs

### 2.1 Comparison table

| Technique | What it does | Cures global anisotropy (a)? | Separates near-synonyms (b)? | Acts on which side? | Cost | Primary ref |
|---|---|---|---|---|---|---|
| **Mean-centering** (subtract mean vector) | Removes the shared "cone-axis" component; recenters on origin | Partially (removes 1 dominant direction) | **No** (rigid translation preserves relative angles beyond the mean) | Frozen text (post-hoc) | ~0 | Mu & Viswanath 2018 |
| **All-but-the-top** | Mean-subtract **+** project out top ~d/100 PCs | Yes | Weakly (only if degeneracy lies on a top PC) | Frozen text | ~0 | Mu & Viswanath, ICLR 2018 |
| **Whitening / BERT-flow** | Center + decorrelate + equalize variance → identity covariance (linear; flow = nonlinear) | **Yes (strong)** | Weakly–moderately (rescales suppressed directions up) | Frozen text | ~0 (whiten) / train (flow) | Su et al. 2021; Li et al. EMNLP 2020 |
| **Standardization / z-scoring** | Per-dimension (x−μ)/σ; neutralizes rogue dims | **Yes (strong)** | Weakly–moderately | Frozen text (or similarity) | ~0 | Timkey & van Schijndel, EMNLP 2021 |
| **L2 normalization** | Projects to unit sphere; removes magnitude (frequency) bias | Partially | No | Both | ~0 | standard (CLIP/SBERT) |
| **Temperature τ in InfoNCE** | Scales logits; low τ sharpens & penalizes hard negatives more | n/a (geometry via optimization) | **Indirectly yes** (low τ pushes confusables apart) | Trained side | ~0 | Wang & Isola 2020; CLIP 2021 |
| **Uniformity regularizer** | Explicit sphere-spreading energy term | **Yes (by construction)** | Yes, on the *trained* embeddings | **Trained side only** | low | Wang & Isola 2020; SimCSE 2021 |
| **Hard-negative mining** | Oversample/upweight confusable negatives | n/a | **Yes (direct)** | Trained side | low–med | SimCSE (supervised) 2021 |
| **Angular-margin losses** (SphereFace/CosFace/ArcFace) | Enforce a geodesic/cosine margin between class prototypes on the sphere | n/a | **Yes (strongest for near-duplicates)** | Trained side / prototypes | med | Liu 2017; Wang 2018; Deng 2019 |

### 2.2 Where does mean-subtraction rank?
**Necessary but the weakest of the isotropizers.** Mean-subtraction is a single rigid translation: it
removes the common component (the cone axis / a large part of the frequency bias) and is why it is a
standard first step. But because it only *shifts* the cloud, it cannot change relative angles among the
residuals. **Whitening and per-dimension standardization strictly dominate it** for problem (a): they
additionally decorrelate and **re-inflate the low-variance directions** where the discriminative
signal often hides, and they neutralize rogue dimensions (Timkey & van Schijndel). Empirically in the
sentence-embedding literature, whitening/standardization recover BERT-flow-level STS gains while plain
centering recovers only a fraction. **For separating genuine near-synonyms, none of the linear
post-hoc methods is the right tool** — that job belongs to the *learned* side: contrastive
**hard negatives**, a **uniformity** term, or an **angular margin** (ArcFace/CosFace) that explicitly
forces a gap between confusable prototypes. Face-recognition losses were invented for exactly this
"minimize intra-class, maximize inter-class, near-duplicate identities" regime, and are the reference
solution when you must pry apart embeddings that are almost identical.

### 2.3 The ceiling that matters for HALO
All of mean-centering, whitening, standardization, and all-but-the-top are **linear (or fixed
nonlinear) maps of the frozen embeddings**. A linear map cannot increase the *rank* of the difference
between two vectors: if `up` and `down` differ only in a direction that is intrinsically tiny (because
SBERT's `up`/`down` token vectors are near-collinear and the sentence is otherwise identical), then
after *any* whitening the two anchors remain close **unless that tiny direction happened to be
variance-suppressed** (in which case whitening helps) rather than **genuinely absent** (in which case
nothing helps). HALO's mpnet-768 wash (§4 of the audit: up/down 0.868→0.909, left/right 0.946→0.981)
is the tell that for the antonym axes the discriminative direction is **genuinely absent**, not merely
suppressed — a **degeneracy**, not an anisotropy. Corollary: whitening the label matrix is a free win
for the *coarse* geometry and the moderately-confusable pairs, but the up/down, left/right, in/out,
and A→B/B→A axes are **fundamentally text-degenerate** and must be carried by the signal or by
injecting new discriminative *content* into the text (§5).

---

## 3. Cross-modal (CLIP-style) specifics

### 3.1 The modality gap / cone effect (Liang et al., NeurIPS 2022)
*"Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning"*
(NeurIPS 2022) showed that in CLIP-like two-tower models, image and text embeddings do **not**
intermix — each modality occupies its **own narrow cone**, and the two cones sit "at arm's length"
with a **gap** between them. They trace it to two causes: (1) the **cone effect at initialization** —
a randomly-initialized deep net already maps inputs into a narrow cone, and two independently
initialized encoders produce two *different* cones; (2) **contrastive optimization**, which preserves a
gap whose width is governed by the **temperature**. They further show the gap size affects downstream
zero-shot accuracy and fairness — i.e., it is a **tunable** property, not strictly to be minimized.

Follow-ups refine the mechanism and connect it back to §1–2:
- Fahim et al. (2024), *"It's Not a Modality Gap: Characterizing and Addressing the Contrastive Gap"*
  argue the gap is **inherent to the two-encoder contrastive loss** and is caused by **low uniformity**
  (embeddings occupy only a small patch of the sphere); adding Wang–Isola **alignment + uniformity**
  terms to the CLIP loss closes it and improves downstream tasks — a direct bridge from the sentence-
  embedding cure to the multimodal setting.
- Schrodi et al. (2024), *"Two Effects, One Trigger …"* find that only a **few embedding dimensions
  drive the gap** (echoing Timkey's rogue dimensions) and that the root cause is **information
  imbalance** between the paired modalities.

### 3.2 CLIP's learnable temperature (Radford et al., ICML 2021)
CLIP learns the InfoNCE temperature as a **log-parameterized scalar** (`logit_scale`), initialized to
`log(1/0.07)` and **clamped so the effective scale never exceeds 100** (to stabilize training). The
temperature is the single knob that sets how hard confusable negatives are pushed apart and, per
Liang et al., how wide the modality gap is. For HALO this is directly actionable: HALO uses a
`logit_scale`/`soft_target_temperature` already; treating it as a tuned or learned parameter (rather
than fixed) is a near-free lever on the geometry.

### 3.3 Does a frozen text tower make collapse worse?
**Yes, structurally.** In CLIP both towers are trained, so contrastive optimization can *reshape the
text cone* to improve uniformity and to pull antonyms apart if the images demand it. HALO instead
freezes the text tower (all-MiniLM) — the setup is the mirror of **LiT**, Zhai et al., *"LiT:
Zero-Shot Transfer with Locked-image Text Tuning"* (CVPR 2022), which locks the *image* tower and
trains the text tower and found the locked-side representation is essentially inherited unchanged.
Applied to HALO's locked-**text** design, the implication is stark: **the frozen text geometry —
including every collapse in §1.6 — is immutable.** The InfoNCE gradient can only move the IMU
embeddings; it can never move two near-identical text anchors apart. When `up` and `down` anchors sit
at cosine 0.90, the IMU encoder is asked to hit two nearly-coincident targets, and the soft-target
InfoNCE supplies almost **no separating gradient** (this is exactly the audit's §4 conclusion). So a
frozen text tower does not *create* the collapse, but it **removes the one mechanism (jointly training
the text side) that would otherwise dissolve it**, and converts the text-side degeneracy into a hard
architectural constraint.

---

## 4. What comparable sensor / time-series ↔ text systems do

**The dominant field-wide answer to text-label collapse is: don't align to bare labels — align to
richer text.** Concretely:

- **ImageBind** (Girdhar et al., CVPR 2023) — *"One Embedding Space to Bind Them All"* binds six
  modalities incl. **IMU** to a **frozen OpenCLIP** image+text space via InfoNCE with temperature,
  training only the new-modality encoders + adapters. It aligns IMU to **image/video**, using text
  mostly for emergent zero-shot; it does **not** specifically diagnose or fix activity-**label**
  collapse (it inherits CLIP's text geometry, gap included).
- **IMU2CLIP** (Moon et al., Findings of EMNLP 2023; arXiv 2210.14395, 2022) aligns IMU to CLIP's
  **text and video** using **egocentric video narrations** (free-form natural language), not class
  names — sidestepping bare-label degeneracy by construction.
- **UniMTS** (Zhang et al., NeurIPS 2024) — *"Unified Pre-training for Motion Time Series"* is the most
  direct precedent for HALO. It contrastively aligns a **graph-convolutional** motion encoder to text
  descriptions that are **augmented by an LLM** (rich activity descriptions rather than labels), plus
  physics-based skeleton-to-IMU synthesis and **rotation-invariant** augmentation for placement/
  orientation generalization. Its handling of "text quality" is precisely **LLM description
  enrichment** — the recommended §5 text-side fix — and its rotation-invariance is the same instinct
  as HALO's yaw augmentation / gravity-canonicalization plan.
- **GOAT** (*"A Generalized Cross-Dataset Activity Recognition Framework with Natural Language
  Supervision"*, IMWUT 2024) aligns wearable data to **natural-language attributes of the activity and
  the sensor location**, again richer than a class name, to enable open-vocabulary / cross-dataset
  transfer.
- **LanHAR** (*"Large Language Model-Guided Semantic Alignment for HAR"* / "Language-centered HAR",
  arXiv 2410.00003; IMWUT 2025) uses an LLM to generate **semantic interpretations of both the sensor
  readings and the activity labels**, then aligns them with a text encoder and **two contrastive
  tasks**, with an **iterative re-generation** loop to raise interpretation quality — an explicit
  attack on label ambiguity/heterogeneity across datasets.
- **SensorLM** (Zhang et al., 2025) uses **hierarchical caption/subtitle generation** to align motion
  and language — same "describe, don't label" philosophy.
- **Ghosh (arXiv 2606.10789, 2026)** — *"Closing the Modality Gap in Zero-Shot HAR: Contrastive
  Training and Separability-Optimized Prototypes on IMU Data"* is essentially HALO's problem statement:
  a **frozen (CLIP-based) text encoder**, the observation that **semantically similar activities
  collapse (anisotropy) and reduce discriminative power**, and a toolbox of **contrastive training,
  separability-optimized prototypes, temperature scaling, whitening, uniformity constraints, hard-
  negative mining, and angular-margin losses** — reporting that explicitly optimizing prototype
  **separability** on top of contrastive alignment improves zero-shot HAR. This is the closest
  external validation that HALO's candidate fixes (§5) are the right menu.

**Pattern:** every system that takes fine-grained/open-vocabulary HAR seriously either (i) trains the
text side, or (ii) replaces bare labels with LLM-generated **descriptions/narrations**, and pairs it
with **temperature + uniformity/separability** machinery. None relies on bare-class-name cosine for
fine distinctions — which is exactly the regime HALO is currently exposed on.

---

## 5. Concrete recommendations for HALO (frozen all-MiniLM text tower)

Given the tower is frozen, split the fixes by *which side they act on* — because only the IMU side and
the (post-hoc) text transform are movable.

### 5.1 Cheap wins (≈0 extra compute, mostly retrain-flag or precompute)

1. **Isotropize the ANCHOR label matrix, not just the soft-target similarities.** Today, standardization
   is applied to the *soft-target* similarity matrix (`semantic_loss.py:159–162`) but the anchors used
   for the actual logits (`:141`) and at eval (`eval_common.py:434`) are raw. Precompute a
   **whitening / all-but-the-top / z-scoring transform** of the frozen label matrix (fit once on the
   label set) and apply it to the anchors at both train and eval. This is the Su et al. / Timkey /
   Mu-Viswanath fix, it strictly dominates the current implicit mean-effect, costs one matrix multiply,
   and is currently **unexploited**. Expect gains on the *coarse* and *moderately confusable* geometry;
   do **not** expect it to rescue the degenerate antonym axes (§2.3).
2. **Short, content-dominant label phrases at recognition time.** HALO's own probe shows
   `"someone is walking"` cut confusable cosine 0.483→0.444 while verbose boilerplate
   (`"the activity of walking on flat ground"`) blew it up to 0.801. Use short templates where the
   *discriminative* word dominates; never add shared boilerplate. Also **de-underscore** eval labels
   (`walking_downstairs`→`walking downstairs`) — the audit shows underscores make up/down collapse
   *worse* and cost self-consistency (0.86→1.0). Free.
3. **Tune / learn the InfoNCE temperature (CLIP-style).** A log-parameterized, clamped learnable
   temperature (Radford et al. 2021) directly controls how hard confusable negatives are separated and
   the modality-gap width (Liang et al. 2022). Cheap knob, likely underset today.
4. **Make antonyms true negatives in the queue.** The audit already flags `QUEUE_MODE=semantic`
   (`semantic_loss.py:171`): the current queue zeros soft-target mass and can push synonyms as false
   negatives. Ensuring genuinely-different activities (up vs down) are treated as **hard negatives**
   (SimCSE-supervised logic) is the cheapest lever that actually attacks problem (b). Free at retrain.
5. **Prefer frozen mean-pool over the LearnableLabelBank** (audit: +5.83 pp). A learnable pool over
   frozen tokens **cannot manufacture separation absent in the shared tokens** — consistent with §2.3.
   This is a free correctness fix, not a collapse fix.

### 5.2 Research bets (new loss terms / data pipeline; validate with ablations)

6. **Uniformity term on the IMU embeddings** (Wang & Isola 2020; SimCSE 2021; Fahim et al. 2024). Since
   the text side is frozen, add the sphere-spreading energy on the **sensor** side so the IMU encoder
   *out-sharpens* degenerate text targets and fills the sphere rather than collapsing onto the text
   cone. Directly addresses the contrastive/modality gap. Moderate.
7. **Angular-margin / separability loss on the IMU→text logits** (CosFace, Wang 2018; ArcFace, Deng
   2019; ported to the prototype setting by Ghosh 2026). You cannot add margin to *fixed* text targets,
   but you can impose a **cosine/geodesic margin in the InfoNCE** so the IMU embedding for `up` must
   beat the `down` anchor by a margin — the reference technique for prying apart near-duplicate
   prototypes. This is the single most targeted fix for the confusable pairs that *do* have signal
   support. Medium; needs tuning to avoid the training instabilities SphereFace/ArcFace are known for.
8. **LLM description augmentation (UniMTS / LanHAR pattern) that injects the *physics* of the
   distinction.** Replace/augment bare labels with short descriptions whose **content words** carry the
   axis: e.g., `"walking upstairs: ascending steps, lifting the body against gravity"` vs
   `"walking downstairs: descending steps, controlled lowering"`. This adds *real* discriminative
   tokens the frozen encoder *can* separate — unlike synonym/template augmentation, which only reshapes
   the same tokens. Generate offline once; keep phrases short (§5.1.2 caveat). Highest-leverage
   text-side bet; moderate cost. Note the audit's asymmetry: label text is augmented but **channel
   text has zero linguistic augmentation** — the same description strategy should extend to channel/
   placement text.
9. **(Optional, principled) A trained projection head on top of the frozen text vector.** Distinct from
   the LearnableLabelBank (which pools *tokens*): a small learned linear/MLP map applied to the *pooled*
   frozen embedding, supervised contrastively, *can* rotate near-collinear anchors apart **iff the IMU
   signal provides the supervision to do so**. Given the LearnableLabelBank was net-negative, treat
   this as a hypothesis to ablate, not a default — and expect it to help only on axes where the sensor
   can actually discriminate.

### 5.3 Axes to concede as text-degenerate (carry in the signal, not the label text)
From the audit's TEST 3/TEST 4 and the mpnet wash, these are **not rescuable by any SBERT-side fix**
because they are mirror-symmetric or function-word/order distinctions a bag-of-words encoder cannot
represent:

- **Side: left/right** (left/right wrist 0.946–0.996) — mirror-symmetric; **zero** usable text signal.
- **Axis: X/Y/Z** channel descriptions (acc-X↔acc-Y 0.921) — same.
- **Order / transition: A→B vs B→A** (sit-to-stand↔stand-to-sit 0.96; lie-to-sit↔sit-to-lie 0.97) —
  SBERT ignores token order at label length.
- **Negation:** cycling↔not-cycling 0.88; gravity removed↔not removed 0.92.
- **Direction: up/down** (0.87–0.93) — *partially* recoverable via §5.2.8 physics descriptions
  (ascend/descend, against/with gravity are genuine content words), but not via re-isotropization.

**Recommendation:** for side and axis, inject a **numeric/positional side-channel** (not text); for
direction/order/negation, hand separation to the **IMU head** plus a §5.2.7 margin, use §5.2.8
descriptions where content words exist, and **do not claim zero-shot *text* recognition** for datasets
whose classes differ only along these axes (motionsense/harth/mobiact stairs, mobiact car step in/out).
This matches the audit's own §4 verdict and is defensible in the paper.

### 5.4 One-line ranking for HALO
> **Cheap, do now:** whiten/standardize the anchor label matrix → short content-dominant phrases +
> de-underscore → learnable/tuned temperature → semantic queue (hard negatives) → frozen mean-pool.
> **Research bets, ablate:** IMU-side uniformity → InfoNCE angular margin for confusable pairs →
> LLM physics-description augmentation (labels **and** channels) → optional trained text projection.
> **Concede to the signal:** left/right, X/Y/Z, A→B order, negation.

---

## References (primary sources; venue, year)

**Anisotropy / sentence-embedding geometry**
- Ethayarajh, K. *How Contextual are Contextualized Word Representations? Comparing the Geometry of
  BERT, ELMo, and GPT-2 Embeddings.* EMNLP-IJCNLP 2019.
  https://aclanthology.org/D19-1006/
- Li, B., Zhou, H., He, J., Wang, M., Yang, Y., Li, L. *On the Sentence Embeddings from Pre-trained
  Language Models* (BERT-flow). EMNLP 2020.
  https://aclanthology.org/2020.emnlp-main.733/
- Su, J., Cao, J., Liu, W., Ou, Y. *Whitening Sentence Representations for Better Semantics and Faster
  Retrieval.* arXiv:2103.15316, 2021.
  https://arxiv.org/abs/2103.15316
- Huang, J. et al. *WhiteningBERT: An Easy Unsupervised Sentence Embedding Approach.* Findings of EMNLP
  2021. https://aclanthology.org/2021.findings-emnlp.23/
- Mu, J., Viswanath, P. *All-but-the-Top: Simple and Effective Postprocessing for Word
  Representations.* ICLR 2018. https://openreview.net/forum?id=HkuGJ3kCb
- Timkey, W., van Schijndel, M. *All Bark and No Bite: Rogue Dimensions in Transformer Language Models
  Obscure Representational Quality.* EMNLP 2021. https://aclanthology.org/2021.emnlp-main.372/

**Contrastive-learning theory / sentence contrastive**
- Wang, T., Isola, P. *Understanding Contrastive Representation Learning through Alignment and
  Uniformity on the Hypersphere.* ICML 2020. https://proceedings.mlr.press/v119/wang20k.html
- Gao, T., Yao, X., Chen, D. *SimCSE: Simple Contrastive Learning of Sentence Embeddings.* EMNLP 2021.
  https://aclanthology.org/2021.emnlp-main.552/
- Reimers, N., Gurevych, I. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP
  2019 (all-MiniLM lineage). https://aclanthology.org/D19-1410/

**Cross-modal / modality gap**
- Radford, A. et al. *Learning Transferable Visual Models From Natural Language Supervision* (CLIP;
  learnable log-temperature clamped at 100). ICML 2021. https://proceedings.mlr.press/v139/radford21a.html
- Liang, W. et al. *Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive
  Representation Learning.* NeurIPS 2022. https://arxiv.org/abs/2203.02053
- Fahim, A. et al. *It's Not a Modality Gap: Characterizing and Addressing the Contrastive Gap.*
  arXiv:2405.18570, 2024. https://arxiv.org/abs/2405.18570
- Schrodi, S. et al. *Two Effects, One Trigger: On the Modality Gap, Object Bias, and Information
  Imbalance in Contrastive Vision-Language Learning.* arXiv:2404.07983, 2024. https://arxiv.org/abs/2404.07983
- Zhai, X. et al. *LiT: Zero-Shot Transfer with Locked-image Text Tuning.* CVPR 2022.
  https://arxiv.org/abs/2111.07991

**Angular-margin / prototype-separation losses**
- Liu, W. et al. *SphereFace: Deep Hypersphere Embedding for Face Recognition.* CVPR 2017.
  https://arxiv.org/abs/1704.08063
- Wang, H. et al. *CosFace: Large Margin Cosine Loss for Deep Face Recognition.* CVPR 2018.
  https://arxiv.org/abs/1801.09414
- Deng, J. et al. *ArcFace: Additive Angular Margin Loss for Deep Face Recognition.* CVPR 2019.
  https://arxiv.org/abs/1801.07698

**Sensor / time-series ↔ text alignment**
- Girdhar, R. et al. *ImageBind: One Embedding Space to Bind Them All.* CVPR 2023.
  https://arxiv.org/abs/2305.05665
- Moon, S. et al. *IMU2CLIP: Language-grounded Motion Sensor Translation with Multimodal Contrastive
  Learning.* Findings of EMNLP 2023 (arXiv:2210.14395, 2022). https://aclanthology.org/2023.findings-emnlp.883/
- Zhang, X. et al. *UniMTS: Unified Pre-training for Motion Time Series.* NeurIPS 2024.
  https://arxiv.org/abs/2410.19818
- *GOAT: A Generalized Cross-Dataset Activity Recognition Framework with Natural Language Supervision.*
  IMWUT 2024. https://dl.acm.org/doi/10.1145/3699736
- *Large Language Model-Guided Semantic Alignment for Human Activity Recognition* (LanHAR). arXiv:2410.00003,
  2024; IMWUT/ACM 2025. https://arxiv.org/abs/2410.00003
- Ghosh, A. *Closing the Modality Gap in Zero-Shot HAR: Contrastive Training and Separability-Optimized
  Prototypes on IMU Data.* arXiv:2606.10789, 2026. https://arxiv.org/abs/2606.10789

**HALO internal**
- `docs/v2/data_diversity_audit.md` §4 (empirical text-embedding probe).
- `training_scripts/human_activity_recognition/semantic_loss.py:141,159–162,171` (logits; soft-target
  similarity standardization; queue).
- `val_scripts/human_activity_recognition/eval_common.py:434` (recognition logits = emb @ text_embs.T).
