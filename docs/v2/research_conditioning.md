# Conditioning on Metadata: Best Practices for HALO

Research memo for the HALO/TSFM v2 design discussion. Scope: how neural models
should condition on (a) open-ended semantic text metadata (sensor placement,
device/gravity description) and (b) rigid numeric scalars (sampling rate Hz,
patch/window duration s), plus (c) rate/duration-derived observability masks.

Author: research pass, 2026-07-03. No code was changed. Every non-obvious claim
is cited with paper + venue + year; where a claim could not be verified from a
primary source it is flagged explicitly.

---

## 1. Executive summary

- **Q1 (open-ended semantic text — placement, gravity):** Best practice for a
  *short* descriptive phrase is to encode it with a (frozen) sentence/text
  encoder and inject the *pooled* embedding into the backbone via **FiLM** or as
  a **prepended conditioning token** (cheap, robust); reserve **CLIP-style
  contrastive alignment** for the text you actually classify against (activity
  label names), and reserve **cross-attention (Flamingo/Perceiver-Resampler)**
  for *long / multi-sentence* context you cannot pool. For HALO's per-channel
  placement string, a pooled-SBERT-embedding → FiLM/token path is the right tool.
- **Q2 (rigid numeric scalars — rate, duration):** Best practice is a
  **dedicated numeric embedding**: map the scalar (use log-Hz, log-seconds)
  through a **Fourier-feature / sinusoidal encoding** + small MLP, then inject
  via **FiLM** or **adaLN/adaLN-zero** (and/or as a token). Do **not** leave the
  number inside a free-text string read by a frozen text encoder — subword
  tokenizers have documented poor numeracy (Wallace et al., EMNLP 2019), so
  "50Hz" vs "100Hz" is not recoverable as a magnitude/ratio.
- **One-line verdict on HALO today:** *Partially aligned, and diverges exactly
  where it matters most.* Putting **placement + gravity in free text → frozen
  SBERT** is aligned with best practice (it is essentially what GOAT does).
  Consuming **rate numerically in the new filterbank tokenizer** is aligned and
  good. But **also stuffing sampling rate and window duration into that same
  free-text SBERT string** is the known-bad "numbers-in-text" anti-pattern: the
  frozen encoder cannot give the transformer precise access to those quantities.
  The fix is small — split the string, keep placement/gravity as text, and route
  the two scalars through a Fourier-feature→FiLM/adaLN path.

---

## 2. Q1 — Conditioning on open-ended semantic text metadata

Task: convey a *short* descriptive phrase ("accelerometer on the left wrist",
"front trouser pocket", "gravity removed") into a time-series backbone.

### 2.1 Methods

| Method | How it works | Pros | Cons | Representative users |
|---|---|---|---|---|
| **CLIP-style contrastive alignment** | Two encoders (signal, text); train so paired embeddings have high cosine similarity (InfoNCE). Inference = argmax cosine to candidate texts. | Enables zero-shot; single global objective; the text side can be frozen. | Aligns *one* pooled vector per item; not a natural way to *feed* side-info into the backbone — it aligns to it. Good for the label you classify against, weaker as a channel-attribute injector. | CLIP (Radford et al., ICML 2021); UniMTS (Zhang et al., NeurIPS 2024, activity text only); GOAT (Miao & Chen, IMWUT 2024); IMU2CLIP (Moon et al., EMNLP-Findings 2023); **HALO today**. |
| **Text-as-tokens / prompt concatenation** | Encode the phrase (or its tokens) and prepend/insert as extra tokens in the backbone sequence; attention mixes them into signal tokens. | Simple; lets self-attention decide how much to use; flexible length. | Adds sequence length; a frozen text encoder's token space may not match the signal token space (needs a projector); position/ordering must be handled. | Prompt/prefix tuning (Li & Liang, ACL 2021; Lester et al., EMNLP 2021); most "LLM + sensor token" systems (e.g. LLaSA, Imran et al., 2024). |
| **Cross-attention to a text encoder (Flamingo / Perceiver-Resampler)** | Backbone layers cross-attend to a (possibly resampled to fixed length) text-encoder output; gated so init ≈ identity. | Handles *long / variable-length* context; keeps text encoder frozen; expressive. | Heaviest option (extra attention layers, params, latency); overkill for a 2-4 word placement phrase. | Flamingo (Alayrac et al., NeurIPS 2022); Perceiver-IO (Jaegle et al., ICLR 2022). |
| **FiLM / conditional norm from a pooled text embedding** | Pool the phrase to one vector; an MLP maps it to per-channel affine (γ, β) that modulate backbone features. | Cheap, parameter-light, injects into every layer; ideal for a *fixed-meaning short attribute*; text encoder stays frozen. | Only an affine (limited capacity per layer); pooling discards word order (fine for "left wrist"). | FiLM (Perez et al., AAAI 2018); widely used for "condition a signal net on a short descriptor" (e.g. FiLM-conditioned ASR, speech, PPG). |
| **Prefix / prompt tuning** | Learn a small set of continuous "virtual token" vectors (optionally produced from the metadata) prepended to the sequence; backbone frozen. | Parameter-efficient adaptation; good when metadata is categorical-ish. | Learns *per-configuration* prefixes rather than generalizing over open-ended text; less suited to unseen placement strings. | Li & Liang (ACL 2021); Lester et al. (EMNLP 2021). |

### 2.2 Recommendation for HALO (Q1)

For a **short, fixed-meaning** attribute like placement or gravity state, the
best-practice injector is **pooled-embedding → FiLM (or a single prepended
conditioning token)**, not cross-attention (too heavy) and not contrastive
alignment (that is the right home for the *label* text, not for channel
side-info). This is exactly the niche HALO's `ChannelTextFusion` already
occupies. Concretely: keep encoding the placement/gravity phrase with frozen
SBERT (all-MiniLM-L6-v2, d=384) and fuse the pooled vector per channel. The only
change Q1 needs is **removing the numbers from that phrase** (see Q2). GOAT
(IMWUT 2024) is the closest published precedent: it pretrains with natural-
language supervision over *"textual attributes from activity labels and sensor
locations"* plus a device-position encoding and a cosine-similarity loss — i.e.
it conveys placement as text, validating HALO's text-for-placement choice.
(The exact form of GOAT's separate "device position encoding" — text vs. a
learned lookup — could not be confirmed from an accessible primary source; the
ACM/ResearchGate pages were blocked. Verified only that placement enters via
*textual attributes*, per the survey arXiv:2508.12213.)

---

## 3. Q2 — Conditioning on rigid numeric scalar metadata

Task: give the transformer *precise* access to a continuous quantity — sampling
rate (Hz) and patch/window duration (s) — so it can be **used**, not merely
correlated with.

### 3.1 Methods

| Method | How it works | Pros | Cons | Notes / users |
|---|---|---|---|---|
| **Number-in-free-text → frozen text encoder** | Write "…sampled at 50Hz, 2s window…", encode with BERT/SBERT/T5. | Zero new machinery; human-readable. | **Known-bad for precise use.** Subword tokenizers split numbers arbitrarily; frozen encoders encode magnitude poorly and ratios not at all (Wallace et al., EMNLP 2019). The model cannot recover 50 vs 100 as a 2× relation. | This is **HALO's current handling of rate/duration**. Anti-pattern for Q2. |
| **FiLM from a scalar** | scalar → MLP → per-channel (γ, β) affine modulation of features. | Cheap; injects into every layer; the value directly scales activations. | Raw scalar into an MLP has spectral-bias problems (below) — pair with Fourier features. | FiLM (Perez et al., AAAI 2018). Standard for "condition on a continuous knob" (e.g. STSM-FiLM conditions on a continuous speed factor, 2025; demographic FiLM in PPG BP models, 2026). |
| **Fourier features / random Fourier features** | Map scalar x → [sin(2πB x), cos(2πB x)] for a set of frequencies B, then MLP. | **Directly fixes the core problem:** a plain MLP has an NTK with rapid frequency falloff and cannot learn fine dependence on a raw coordinate; a Fourier mapping turns it into a stationary, tunable-bandwidth kernel so the net can use the value precisely and at multiple scales. | Must choose frequency bandwidth; too high → noise/aliasing. | Tancik et al., NeurIPS 2020 ("Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"); same principle as NeRF positional encoding (Mildenhall et al., ECCV 2020). **Best default for encoding a physical scalar.** |
| **Time2Vec** | scalar τ → [ω₀τ+φ₀ (linear), sin(ωₖτ+φₖ) (K periodic)], ω,φ learned. | Learnable sinusoidal+linear; captures periodic and non-periodic dependence; model-agnostic drop-in. | Learned frequencies can under-cover the range; essentially a learnable-frequency variant of Fourier features. | Kazemi et al., 2019 (arXiv:1907.05321). Good if you expect periodic structure in the scalar. |
| **Sinusoidal scalar encoding** | Apply the Transformer positional-encoding formula to an arbitrary real value (not just integer positions). | No learned params; smooth; multi-scale. | Fixed frequency schedule; less adaptive than random/learned Fourier features. | Vaswani et al., NeurIPS 2017 (repurposed). |
| **Learned embedding / binning / quantization** | Bucket the scalar into bins, learn an embedding per bin (optionally interpolate). | Simple; robust; works when only a few discrete values occur (e.g. {20,30,50,100} Hz). | Discards within-bin precision; no extrapolation beyond seen bins; boundary artifacts. | TimesFM's coarse categorical **frequency** input {0,1,2} = high/med/low (Das et al., ICML 2024) is exactly this — deliberately coarse. |
| **adaLN / adaLN-zero (DiT)** | A shared MLP maps the conditioning vector to per-block LayerNorm scale/shift **and a residual gate**; adaLN-**zero** inits the gate to 0 so each block starts as identity. | Proven best way to inject a *global* scalar into *every* transformer block; benign init (identity) → stable training; DiT found it beats cross-attention and in-context conditioning while being most compute-efficient. | Global (one modulation per block) — good for rate/duration, not for per-token differences. | Peebles & Xie, ICCV 2023 ("Scalable Diffusion Models with Transformers"). **Best injector once you have a good scalar embedding.** |
| **Hypernetworks** | A network generates the backbone's weights (or a low-rank delta) from the scalar. | Maximum expressivity; the scalar can reshape computation, not just rescale it. | Heavy, harder to train, more params; usually unnecessary for a 1-2 scalar knob. | Ha et al., ICLR 2017. Overkill here. |

### 3.2 The numeracy problem (why "put the number in text" fails)

- **Subword tokenization destroys magnitude.** Wallace et al. (EMNLP-IJCNLP
  2019, "Do NLP Models Know Numbers? Probing Numeracy in Embeddings") probe
  token embeddings on list-maximum, decoding, and addition. Character/word-level
  embeddings (ELMo, word2vec/GloVe) capture magnitude reasonably up to ~1,000,
  but **BERT's sub-word units are markedly worse** — precisely the regime a
  SentenceBERT (MiniLM) tokenizer sits in. A frozen sentence encoder therefore
  gives the backbone no reliable handle on "50" vs "100", let alone their ratio.
- **It is a general, surveyed failure mode**, not a one-off: Thawani et al.
  (NAACL 2021, "Representing Numbers in NLP: a Survey and a Vision") catalog why
  standard NLP number handling is inadequate and call for dedicated numeric
  representations.
- **The fix that the field converged on is continuous numeric encoding**, not
  more text. xVal (Golkar et al., 2023, arXiv:2310.02989, "A Continuous
  Numerical Tokenization…") represents a number with a *single* token whose
  embedding is *scaled by the value*, making the map end-to-end continuous in the
  number and improving out-of-distribution generalization — the same spirit as
  Fourier-feature/FiLM scalar conditioning.
- **Why this bites HALO specifically:** the channel description is encoded by a
  **frozen** SBERT, so the model cannot even fine-tune its way around the poor
  numeracy. And the *point* of the new physical-Hz tokenizer is to use rate as a
  *precise* physical quantity; a downstream module that only sees a fuzzy text
  embedding of "50Hz" cannot reason about Nyquist limits, band observability, or
  duration-vs-rate trade-offs with the needed precision.

### 3.3 Recommendation for HALO (Q2)

**Answer to the (a)/(b)/(c) question: (b) — a dedicated numeric embedding.**
Not (a) free text (fails numeracy), and not (c) "both" (the text copy adds a
noisy, redundant, frozen-encoder path with no upside; keep the text for
*semantic* attributes only).

Concretely, add a small **scalar-conditioning module**:

1. Normalize each scalar to a natural scale: **log₂(rate/rate₀)** and
   **log₂(duration/dur₀)** (log makes octave/Nyquist relationships linear).
2. Encode each with **Fourier features** (Tancik et al., NeurIPS 2020) → small
   shared MLP → a conditioning vector `c`.
3. Inject `c` via **adaLN-zero** (Peebles & Xie, ICCV 2023) on the dual-branch
   transformer blocks (identity init → zero migration risk), **or** via FiLM if
   you prefer to keep the existing fusion style. Optionally also expose `c` as
   one prepended conditioning token.

Rate is *also* consumed numerically in the DSP filterbank front-end (good, keep
that). The explicit rate/duration embedding is still worth adding because (i)
duration is not fully encoded by a per-patch filterbank, and (ii) downstream
transformer layers benefit from knowing the physical scale of each token
(e.g. to weight bands near Nyquist). This is complementary, not redundant.

---

## 4. What comparable systems actually do (concrete findings)

### Time-series foundation models — the dominant pattern is "normalize rate away, don't condition"

- **MOIRAI** (Woo et al., "Unified Training of Universal Time Series Forecasting
  Transformers", ICML 2024). Does **not** feed frequency as a learned embedding.
  It handles frequency indirectly via **multi patch-size projection layers**: it
  learns 5 input/output projections for patch sizes {8,16,32,64,128} and uses
  **larger patches for higher-frequency data**; patch size is selected by a
  *manual heuristic mapping from data frequency*. "Any-variate attention"
  flattens multivariate series and uses learned per-variate attention biases —
  it is about *variate identity*, not sampling rate. So MOIRAI's only frequency
  "conditioning" is a hand-set patch-size choice, not a numeric input.
- **Moirai-MoE** (Liu et al., 2024, arXiv:2410.10469; ICLR 2025). Explicitly
  **abandons the frequency heuristic**: it argues *"frequency is not a reliable
  indicator of the underlying patterns"* (different-frequency series can share
  patterns; same-frequency series can differ), replaces the multi-patch scheme
  with a **single projection + sparse Mixture-of-Experts** doing token-level
  specialization, and reports ~17% gains. Direct evidence that *coarse frequency
  conditioning is often too blunt* — relevant caution for HALO.
- **TimesFM** (Das et al., "A decoder-only foundation model for time-series
  forecasting", ICML 2024). The one mainstream TSFM that **does** take a
  frequency input: an optional **categorical indicator {0,1,2}** (high/medium/low
  granularity) that is embedded and added to the input. It is deliberately
  **coarse** (three buckets, default 0), i.e. a *binning* approach, not a precise
  numeric rate. Confirms: even the FM that conditions on frequency does so
  categorically, not as a precise scalar.
- **Chronos** (Ansari et al., 2024, arXiv:2403.07815; TMLR). Scales then
  **quantizes** values into a fixed token vocabulary and trains an LM with
  cross-entropy; it **ignores time/frequency information**, treating the series
  as a plain token sequence. No rate conditioning.
- **MOMENT** (Goswami et al., ICML 2024). Sharpest quote in the literature:
  *"We did not explicitly model the temporal resolution of time series, since
  this information is often unavailable outside of time series forecasting
  datasets."* Forces every series to length 512 (sub-sample/pad), patch size 8,
  channel-independent + RevIN. No rate, no placement.
- **Time-MoE** (Shi et al., 2024, arXiv:2409.16040; ICLR 2025) and **Timer /
  Timer-XL** (Liu et al., ICML 2024 / ICLR 2025). Decoder-only, point-wise or
  patch tokenization, RoPE, multi-resolution heads. **No sampling-frequency
  conditioning input** in either; resolution heterogeneity is absorbed by
  tokenization + positional scheme.

**Takeaway:** across TSFMs, rate is almost always *normalized/tokenized away*;
where it is conditioned on at all (TimesFM), it is a **coarse categorical bin**,
and the most recent work (Moirai-MoE) argues even that is too blunt. None puts
the rate into a text string. HALO's *numeric* use of rate in the tokenizer is
actually **more principled** than the TSFM norm — the weakness is only the
redundant text copy.

### IMU / wearable / motion models — placement via text; rate via resampling

- **UniMTS** (Zhang et al., NeurIPS 2024). Placement → **not conditioned**;
  handled *geometrically* by assigning each real sensor to its nearest joint on a
  22-node SMPL **skeleton graph** (+ random joint masking). Orientation →
  **SO(3) rotation-invariant augmentation** (train to be invariant, so no
  orientation input needed). Rate → **resample everything to a fixed 20 Hz**;
  never used numerically. Window length → absorbed by temporal average pooling.
  Text → CLIP text encoder, contrastive, on **activity descriptions only** (no
  placement, no rate in the text). A pure "make the model invariant / normalize
  it away" stance.
- **LIMU-BERT** (Xu et al., SenSys 2021). Resample → fixed **20 Hz**, fixed 6 s
  (120 samples); channels **fused** by a linear projection; placement is a
  *downstream prediction target*, never an input; no rate/placement conditioning,
  no FiLM.
- **ssl-wearables / Yuan et al.** (npj Digital Medicine 2024). Resample → fixed
  **30 Hz**, fixed 10 s (300 samples); placement fixed to **wrist** (never an
  input); axis/orientation handled by **augmentation** (random axis swaps/
  rotations) rather than conditioning; no metadata tokens.
- **GOAT** (Miao & Chen, IMWUT 2024). The closest analog to HALO. **Natural-
  language supervision** over *"textual attributes from activity labels and
  sensor locations"*, a **device-position encoding**, a Transformer activity
  encoder, and a **cosine-similarity** (CLIP-style) loss. → Placement **is**
  conveyed as text/attributes for cross-dataset generalization. Validates HALO's
  text-for-placement choice. (Whether the device-position encoding is text or a
  learned lookup, and whether rate is used, could not be confirmed from an
  accessible primary source.)
- **IMU2CLIP** (Moon et al., EMNLP-Findings 2023). CLIP-style: **frozen** CLIP
  text/image encoders; IMU encoder = 1D-CNN + GRU; text is **motion narrations**.
  Data is head-mounted egocentric (Ego4D/Aria) at a **fixed window & rate**, so
  placement and rate are constants — **not conditioned on**.
- **ImageBind** (Girdhar et al., CVPR 2023). IMU = fixed-length ~5 s, ~2,000-
  sample, 6-channel (acc+gyro) clips → 1D conv (kernel 8) → Transformer, bound to
  other modalities contrastively. Head-mounted, **fixed rate and placement**;
  **no** rate or placement conditioning. (Exact Hz reported inconsistently across
  secondary sources — the load-bearing fact is that it is *fixed*, not fed in.)
- **LanHAR** (Yan et al., 2024) and **LLaSA** (Imran et al., 2024) are
  text-centric IMU systems (LLM-generated semantic interpretations; sensor-aware
  LLM QA) but do **not** make sampling rate or placement a precise conditioning
  input — metadata is not their focus.

### Audio foundation models — the "resample to a canonical rate" school

- **Whisper** (Radford et al., ICML 2023): **resample all audio to 16 kHz**,
  80-channel log-Mel, fixed 30 s window. **wav2vec 2.0** (Baevski et al.,
  NeurIPS 2020): raw waveform at a fixed **16 kHz**. **AST** (Gong et al.,
  Interspeech 2021) and **PaSST** (Koutini et al., Interspeech 2022): Mel-
  spectrogram → patch tokens, at a fixed front-end rate. Universally, audio
  models **resample to one canonical rate** and never condition on rate; feeding
  a mismatched rate degrades performance because the input distribution shifts.
  This is the polar-opposite philosophy to HALO's *rate-native* filterbank — and
  the reason HALO cannot just borrow the audio recipe: HALO deliberately keeps
  heterogeneous rates, so it *must* make rate available to the model, which is
  exactly what a numeric rate embedding provides.

---

## 5. Concrete recommendation for HALO

Given the architecture (channel-independent patch encoder → dual-branch
transformer; tokens CLIP-aligned to **frozen** SBERT label-text; new physical-Hz
filterbank tokenizer that consumes rate in the DSP front-end), condition each
metadata type as follows:

| Metadata | Type | Recommended conditioning | Where it enters | Migration cost vs today |
|---|---|---|---|---|
| **Sensor placement** ("left wrist", "front pocket") | open-ended text | Frozen SBERT of a **short phrase** → pooled → **FiLM** (or one prepended channel token). Keep contrastive alignment for the *label* text, not for this. | Per-channel, as in `ChannelTextFusion` today. | **Low** — reuse the existing SBERT/`ChannelTextFusion` path; just shorten the phrase to placement-only. |
| **Gravity state** ("gravity removed") | open-ended (near-categorical) text | Same path as placement (append to the placement phrase), *or* a 1-bit/small learned flag if it is truly binary. | Same as placement. | **Low.** |
| **Sampling rate (Hz)** | rigid scalar | (i) Keep numeric use in the **filterbank DSP** (already planned). (ii) Add **log₂-rate → Fourier features → MLP → adaLN-zero/FiLM** conditioning of the transformer. **Remove from the text string.** | DSP front-end + a new small scalar-conditioning module feeding transformer blocks. | **Medium** — one new module (Fourier features + MLP + adaLN/FiLM head, ~10²–10³ K params); delete rate substring from channel text. |
| **Patch/window duration (s)** | rigid scalar | **log₂-duration → Fourier features → MLP →** same adaLN-zero/FiLM head (share the scalar module with rate; concatenate the two scalar embeddings). **Remove from the text string.** | Same scalar-conditioning module. | **Medium** (shared with rate). |
| **Per-band observability / resolution masks** | derived from rate & duration | Apply as **attention/patch masks** on bands that violate Nyquist or lack resolution (this is *input masking*, à la MOMENT's observed-mask / MAE, not conditioning). Optionally FiLM the surviving band tokens with the resolution scalar. | Tokenizer emits the mask; transformer honors it. | **Low–Medium** — derives directly from the tokenizer's Nyquist logic. |

**Net design change:** split today's single free-text channel description into
(1) a *semantic* text phrase (placement + gravity) that stays on the frozen-SBERT
→ fusion path, and (2) a *numeric* pair (rate, duration) that moves to a new
Fourier-feature → adaLN-zero/FiLM scalar path. This keeps everything HALO already
does well, removes the one anti-pattern (numbers in frozen text), and costs a
single small, identity-initialized module — so it is safe to A/B against the
current model.

**Why this is the right call, in one paragraph:** placement is genuinely
open-ended language, and text is the correct, precedent-backed medium for it
(GOAT, IMU2CLIP). Rate and duration are *precise physical quantities* the model
must *use*, and the entire numeric-representation literature (Wallace 2019;
Thawani 2021; xVal 2023) plus the scalar-conditioning literature (FiLM 2018;
Fourier features 2020; adaLN-zero / DiT 2023) says: encode them as continuous
numbers with a Fourier/positional lift and inject via FiLM/adaLN — not as English
words fed to a frozen sentence encoder. HALO's rate-native tokenizer already
embodies this philosophy for the front-end; the recommendation simply extends the
same principled numeric handling to the transformer body and to duration.

---

## 6. Open questions / risks

1. **Is a rate embedding redundant with the filterbank?** The filterbank already
   consumes rate in DSP. The claim that the transformer *also* benefits is
   plausible (duration is not front-end-encoded; layers may need physical scale
   for band weighting) but **should be ablated** (tokenizer-only vs.
   tokenizer + scalar-conditioning). Do not add it on faith.
2. **Coarse vs. precise.** Moirai-MoE's finding that *frequency is an unreliable
   indicator* is a caution: a *precise* rate embedding could overfit dataset-rate
   as a shortcut/leakage signal (rate correlates with dataset identity → subject
   leakage risk noted elsewhere in HALO's audit). Consider randomizing/augmenting
   rate (resample augmentation) so the model uses rate physically, not as a
   dataset fingerprint.
3. **adaLN vs. FiLM in a two-branch, contrastive setup.** adaLN-zero is proven in
   DiT (diffusion), less battle-tested in contrastive time-series encoders;
   FiLM is the more conservative choice and matches HALO's existing fusion style.
   Recommend trying FiLM first, adaLN-zero as an upgrade.
4. **GOAT's device-position mechanism is unverified.** Could not access the
   primary text; if GOAT uses a *learned* position embedding (not text) that
   would be a useful additional precedent for a hybrid placement encoding — worth
   reading the IMWUT'24 PDF directly.
5. **Fourier-feature bandwidth.** Choosing the frequency scale for the scalar
   lift matters (Tancik 2020): too high aliases, too low underfits. Needs a small
   sweep over the log-rate/log-duration ranges HALO actually spans (~ 20–200 Hz;
   ~0.5–10 s).
6. **Observability masks vs. contrastive objective.** Masking Nyquist-violating
   bands changes the token set per sample; ensure the contrastive/pooling head is
   robust to a variable, rate-dependent number of live tokens (variable-length
   pooling, not a fixed flatten).

---

## References (venue + year)

- Alayrac et al. **Flamingo**, NeurIPS 2022.
- Ansari et al. **Chronos: Learning the Language of Time Series**, 2024 (arXiv:2403.07815; TMLR).
- Baevski et al. **wav2vec 2.0**, NeurIPS 2020.
- Das et al. **A decoder-only foundation model for time-series forecasting (TimesFM)**, ICML 2024 (arXiv:2310.10688).
- Girdhar et al. **ImageBind: One Embedding Space to Bind Them All**, CVPR 2023.
- Golkar et al. **xVal: A Continuous (Numerical) Number Encoding for LLMs**, 2023 (arXiv:2310.02989).
- Gong et al. **AST: Audio Spectrogram Transformer**, Interspeech 2021.
- Goswami et al. **MOMENT: A Family of Open Time-series Foundation Models**, ICML 2024.
- Ha et al. **HyperNetworks**, ICLR 2017.
- Imran et al. **LLaSA: A Sensor-Aware LLM for NL Reasoning of Human Activity from IMU**, 2024 (arXiv:2406.14498).
- Jaegle et al. **Perceiver IO**, ICLR 2022.
- Kazemi et al. **Time2Vec: Learning a Vector Representation of Time**, 2019 (arXiv:1907.05321).
- Koutini et al. **PaSST: Efficient Training of Audio Transformers with Patchout**, Interspeech 2022.
- Lester et al. **The Power of Scale for Parameter-Efficient Prompt Tuning**, EMNLP 2021.
- Li & Liang. **Prefix-Tuning**, ACL 2021.
- Liu et al. **Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts**, 2024 (arXiv:2410.10469; ICLR 2025).
- Liu et al. **Timer / Timer-XL**, ICML 2024 / ICLR 2025 (arXiv:2402.02368).
- Miao & Chen. **GOAT: A Generalized Cross-Dataset Activity Recognition Framework with Natural Language Supervision**, IMWUT 2024 (ACM DOI 10.1145/3699736).
- Mildenhall et al. **NeRF**, ECCV 2020.
- Moon et al. **IMU2CLIP**, EMNLP-Findings 2023 (arXiv:2210.14395).
- Peebles & Xie. **Scalable Diffusion Models with Transformers (DiT; adaLN-zero)**, ICCV 2023 (arXiv:2212.09748).
- Perez et al. **FiLM: Visual Reasoning with a General Conditioning Layer**, AAAI 2018 (arXiv:1709.07871).
- Radford et al. **CLIP: Learning Transferable Visual Models from Natural Language Supervision**, ICML 2021.
- Radford et al. **Whisper: Robust Speech Recognition via Large-Scale Weak Supervision**, ICML 2023.
- Reimers & Gurevych. **Sentence-BERT**, EMNLP 2019.
- Shi et al. **Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts**, 2024 (arXiv:2409.16040; ICLR 2025).
- Tancik et al. **Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains**, NeurIPS 2020 (arXiv:2006.10739).
- Thawani et al. **Representing Numbers in NLP: a Survey and a Vision**, NAACL 2021.
- Vaswani et al. **Attention Is All You Need**, NeurIPS 2017.
- Wallace et al. **Do NLP Models Know Numbers? Probing Numeracy in Embeddings**, EMNLP-IJCNLP 2019 (arXiv:1909.07940).
- Woo et al. **Unified Training of Universal Time Series Forecasting Transformers (MOIRAI)**, ICML 2024 (arXiv:2402.02592).
- Xu et al. **LIMU-BERT**, SenSys 2021.
- Yan et al. **LanHAR: LLM-Guided Semantic Alignment for HAR**, 2024 (arXiv:2410.00003).
- Yuan et al. **Self-supervised Learning for HAR Using 700,000 Person-days of Wearable Data (ssl-wearables)**, npj Digital Medicine 2024.
- Zhang et al. **UniMTS: Unified Pre-training for Motion Time Series**, NeurIPS 2024 (arXiv:2410.19818).

*Verification notes: Time-series/IMU/audio mechanisms above were cross-checked
against the local PDFs in `references/baselines/` (UniMTS, LIMU-BERT, MOMENT,
ssl-wearables) and against primary abstracts/paper pages for the rest. Perplexity
MCP was unavailable (quota); Consensus MCP confirmed the FiLM primary source.
Claims that could not be verified from a primary source are flagged inline
(GOAT device-position mechanism; ImageBind exact Hz).*

---

## 7. Decision (2026-07-04)

Discussed and decided. The memo's **Q1 finding is adopted in full** and its **Q2
executive-summary recommendation is deliberately NOT** — because HALO's rate-native
tokenizer already specializes the numeric-conditioning principle correctly, and its own
§6.1/§6.2 caveats (redundancy-with-filterbank; rate-as-dataset-fingerprint/leakage) are
the deciding considerations for *this* architecture.

**Adopted:**
1. **Strip sampling rate + window duration from the channel-description text.** The text
   carries only *semantic* attributes (placement + gravity state), on the existing
   frozen-SBERT → `ChannelTextFusion` path. Removes the numbers-in-frozen-text anti-pattern.
2. **Rate = tokenizer + observability mask only. No rate embedding.** The filterbank
   consumes rate to canonicalize the spectrum and emits `o_k` per token; that is the
   complete, generalization-safe encoding of rate's downstream consequence. A precise
   rate scalar is rejected: redundant for the one legitimate use, and a dataset-identity
   fingerprint that would undercut cross-dataset generalization (the leakage the audit
   already flagged). The mask being *lossy about absolute rate* is the intended invariance.
3. **Duration = resolution flag `res_k` only. No duration embedding** (user decision:
   "masks only"). `res_k` (added in M1) already carries duration's main downstream effect
   (low-band resolution). Revisit *only* if the P3 variable-duration ablation shows a real
   gap; do not add on faith.

**Net:** the conditioning change is subtractive (delete the number substring from the
channel text), not additive — no new module. This lands at the **M4 tokenizer cutover**,
coupled to the retrain: today's CNN model interpolates rate away, so the text string is
its *only* rate signal and must not be stripped until the filterbank carries rate.
