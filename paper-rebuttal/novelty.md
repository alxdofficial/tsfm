# HALO Novelty — Positioning for MobiCom '26

*Purpose: a defensible, code-grounded answer to the #1 reviewer concern ("components are individually
known"), tuned specifically for the MobiCom venue — not a generic ML-novelty rebuttal.*

## TL;DR — the three genuinely-novel techniques
1. **Text-as-sensor-interface (`ChannelTextFusion`).** Each sensor channel is conditioned on a
   natural-language description of *what and where it is*, via a gated `O(C)` cross-attention
   (`model/token_text_encoder.py`). So a dataset's sensor layout enters the network **as language**,
   not as a fixed input slot — one weight set ingests **3–51 channels** at any placement, with
   *semantic* channel identity.
2. **Synonym-aware soft-target alignment.** Unifying 10 datasets' label vocabularies makes synonyms
   ("walking"/"strolling") collide as in-batch negatives, which one-hot InfoNCE punishes with
   contradictory gradients. We replace the one-hot target with a **frozen-SBERT similarity soft
   target** (`training_scripts/.../semantic_loss.py`) — the supervision that makes cross-vocabulary
   training stable, and the architectural answer to "why not post-process the labels."
3. **A single *deployable* open-vocabulary HAR model** that runs unmodified across heterogeneous
   real-world IMUs with **no per-dataset retraining**, validated **on-device (iPhone, 97.5%)** — the
   systems/deployment capability the venue actually rewards.

## The honest core claim
HALO's contribution is a **system-level capability + co-design, not a new ML primitive.** The
primitives (MAE, channel-independent CNN, contrastive loss, SBERT, memory queue) are **conceded prior
art.** What is new is the **coupling**: natural-language text does *triple duty* — channel-conditioning
input **and** open-set classifier **and** heterogeneity-tolerant supervision — so the *same* language
interface that enables open-vocabulary recognition is what lets one model span incompatible sensor
formats. Neither line of work delivers this jointly.

## ⚠️ The landscape is crowded — be precise, or lose the novelty fight
The research turned up **a lot** of very recent, very close work. We must cite and *differentiate
explicitly*, or reviewers will (rightly) call it incremental:

| Related work (venue) | What they do | How HALO differs |
|---|---|---|
| **SensorLM** (NeurIPS'25) — "Learning the Language of Wearable Sensors" | sensor↔language alignment | likely fixes the sensor format; we make **per-channel text the conditioning signal** for heterogeneity |
| **IMUZero / LLM-guided semantic alignment** (IMWUT'25) | zero-shot IMU HAR via text | open-vocab, but **fixed sensor layout**; we couple open-vocab *with* 3–51-channel heterogeneity |
| **One Model to Fit Them All** (IMWUT'25) | universal cross-dataset IMU HAR w/ LLM-assisted reps | **closest competitor** — must read + differentiate; our claim is text-as-*channel-identity* + open-vocab, not LLM-generated features |
| **NormWear**, **Babel** (SenSys'25), **RelCon** (ICLR'25) | heterogeneous-sensor foundation models (channel-aware attn / modality alignment) | channel handling **without semantic/text channel identity**; not open-vocab |
| **Mobile Foundation Model as Firmware** (MobiCom'24) | on-device FM-as-firmware | systems/deployment framing precedent at *this venue* — align with it |

**Implication:** "open-vocab IMU↔text" is **not** ours alone, and "heterogeneous-sensor foundation
model" is **not** ours alone. The only defensible novelty is the **specific coupling** (text-as-channel-
identity that simultaneously powers open-vocab) **+ the deployable system**. (Note: the 2025 works above
are largely *concurrent* — cite as related, not as prior art we failed to beat, but expect reviewers to
know them.)

## 🎯 MobiCom-specific positioning (this is the strategy)
MobiCom is a **systems venue**, not an ML venue. Its CFP rewards "practical working systems" with
"real-world measurement or deployment," and reproducibility (released code/data). **We will lose a
pure-ML-novelty fight** against SensorLM / One-Model at an ML/ubicomp venue — so **don't fight it.**
Position HALO as a **sensing-systems contribution**:
- **Lead with the capability + deployment**, not the loss function: *"the first deployable open-vocabulary
  HAR system that runs one model across heterogeneous real-world wearables without per-dataset
  retraining,"* anchored by the **on-device iPhone deployment (97.5%)**.
- **Frame the 3 techniques as the systems mechanisms** that make this deployment possible (text-as-channel-
  interface solves the real-world "every wearable has a different sensor layout" problem).
- **Lean on reproducibility** (release code, data, checkpoints) — MobiCom explicitly favors it.
- **Concede ML primitives openly**; claim the **system + capability + co-design + deployment**. That is
  exactly the contribution a systems venue values and a "just a combination" critique cannot cross.

## Rebuttal-ready paragraph
> We agree the individual primitives — masked autoencoding, channel-independent encoding, contrastive
> learning, text embeddings, CLIP-style alignment — are established, and we claim no new ML building
> block. Our contribution is a **new capability and the systems design that delivers it**: a single
> deployed model performing open-vocabulary activity recognition across wearables with incompatible
> channel counts (3–51) and rates (20–100 Hz) **without per-dataset retraining**, validated on-device.
> The non-obvious insight is that heterogeneity and open-vocabulary must be solved *together, through the
> same interface*: in HALO, text is simultaneously the cross-channel conditioning signal
> (`ChannelTextFusion` injects each channel's description, so sensor layout enters as language, not a
> fixed slot), the open-set classifier, and the heterogeneity-tolerant supervision (frozen-SBERT soft
> targets resolve the contradictory synonym gradients that arise when ten vocabularies share a batch).
> This also answers "why not post-processing": label normalization runs after an argmax over a fixed
> label set and cannot admit a runtime-unseen label or condition the encoder on a runtime-unseen channel
> layout — both of which HALO folds into the forward pass. We position this as a sensing-systems
> contribution (deployable capability + real-world validation), and we cite and differentiate from the
> concurrent IMU-language and heterogeneous-sensor lines accordingly.

## Honesty guardrails (don't reopen other attacks)
- No new ML primitive — claim **system-level capability + co-design**.
- **Deployed checkpoint is alignment-only** — reframe Stage-1 SSL as auxiliary/stability (avoids the
  "show me the pretrained checkpoint" trap; EXP-P4 dropped because the released model is pretrain-free).
- Soften **"foundation model"** language (limited corpus); claim **moderate** heterogeneity (3–51 ch /
  20–100 Hz / 10 datasets), not universal.
- **Cite the concurrent 2025 work** (SensorLM, IMUZero, One-Model, Babel, NormWear) — silence reads as
  ignorance and invites the "not novel" reject.
- Honest numbers: 42.0% = 5-main avg (report all-7 ~34.3%); conditioning is **gated** `ChannelTextFusion`
  (not additive); synonyms are **hand-authored** (3.56/label, not WordNet); open-set scoring is
  group-match (add the exact-match column showing the lead widens).

## Open action
- **Read the 3 closest competitors** (One Model to Fit Them All; SensorLM; NormWear) and write a precise
  one-paragraph differentiation each — the related-work section is where this paper lives or dies.
- Reconcile this with `SUPERVISOR_MEETING_BRIEF.md` (currently omits the related-work threat).

*Sources:* [MobiCom'26 CFP](https://www.sigmobile.org/mobicom/2026/) · [MobiCom'25 CFP — "build practical working systems / real-world deployment"](https://www.sigmobile.org/mobicom/2025/cfp.html) · [One Model to Fit Them All (IMWUT'25)](https://dl.acm.org/doi/10.1145/3749509) · [Customizable FMs for HAR (IMWUT'25)](https://dl.acm.org/doi/10.1145/3749479) · [Awesome-IMU-Sensing (landscape)](https://github.com/rh20624/Awesome-IMU-Sensing)
