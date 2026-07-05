# HALO Streaming Design — Literature Assessment

**Scope.** This reviews HALO's *streaming regimen* for a single-weight-set model that serves both offline whole-session prediction and online low-latency streaming. Architecture under review: channel-independent per-patch physical-Hz filterbank tokenizer (already streaming-safe, offline==incremental to 1e-5) → dual-branch transformer (temporal attention over patches + cross-channel attention within a patch) → per-patch InfoNCE head aligned to frozen SentenceBERT label text. The three streaming bets: (1) RoPE indexed by physical time in seconds; (2) temporal-attention mask randomly sampled per micro-batch from {full-bidirectional, causal-infinite, causal-window-W}; (3) unchanged per-patch InfoNCE loss.

**Method.** Synthesis of a five-lens literature review plus an adversarial brief, with independent spot-verification of the load-bearing claims (see *Verification status* at end). I weight the adversary at full strength and adopt its conclusions where the evidence holds; I depart from it only to add caveats it under-states.

---

## 1. Executive summary

The **architecture and the paradigm are correct and well-precedented.** "One weight set, offline-vs-online = the temporal-attention mask" is exactly the deployed ASR recipe (WeNet U2/U2++, Google dual-mode and cascaded encoders) and the NLP mixed-mask lineage (UniLM, GLM, prefix-LM). HALO's per-patch tokenizer makes the premise *more* literally true than in ASR: temporal attention is genuinely the only cross-time operator, so the offline and online forwards differ by nothing but the mask — no leaky causal-conv/pool/BatchNorm to fix. Nobody should swap the backbone or split into two models; the design's resistance to both is a genuine strength.

The problem is **the specific recipe HALO wrote down is, point for point, the weakest defensible instantiation of that correct idea.** "Randomly sample one mask per micro-batch, loss unchanged, strictly zero-lookahead causal, no streaming smoothing, under-specified RoPE band" is precisely the configuration the foundational ablations identify as leaving the most accuracy on the table — and it does so exactly in the low-latency online regime HALO cares about for phones/watches. The single highest-value fix (offline→online in-place distillation) is *nearly free precisely because of HALO's per-patch design*, which is what makes shipping the current recipe hard to defend in review.

There is also a **latent train/serve mismatch outside the three named bets:** the loss is per-patch only, but offline session-labeling is proposed via mean-pooling that is never trained. And a **coupled data risk:** training on single-activity windows means the offline "full-mask" mode and the RoPE band are both *extrapolating* when deployed on long multi-activity sessions.

### VERDICT: **RIGHT-WITH-CHANGES.**
Keep the architecture, the paradigm, the windowed-causal attention backbone, and RoPE-by-physical-time. Change the *training recipe* (add joint dual-forward + in-place offline→online distillation — the one must-change), add a bounded-lookahead mask as a latency knob, add causal label smoothing, and calibrate the RoPE frequency band. This is a recipe-level revision, not an architecture redesign.

---

## 2. Per-bet assessment

| # | Bet | Literature verdict | Keep / Change | Why (one line) |
|---|-----|--------------------|---------------|----------------|
| 1 | **Mixed-mask "train-both/eval-both"** (one weight set, sample mask at train, choose at inference) | **Paradigm SOUND, recipe WEAKEST-VARIANT** | **Keep paradigm; CHANGE recipe** | Correct and deployed (U2/U2++, dual-mode ASR), but *pure random per-micro-batch sampling with no coupling* is the exact configuration ablations beat with joint+distillation; skipping distillation forfeits ~20% relative streaming accuracy (Dual-mode ASR, ICLR'21). |
| 2 | **RoPE indexed by continuous physical time (seconds)** | **Right FAMILY, under-specified** | **Keep; MUST-TUNE** | Rotary on real m is mathematically legal (SO(2) continuous) and empirically competitive for irregular time series (RoMAE, RoTHP proves the shift-invariance that makes re-anchoring safe, TimelyGPT/xPos); but "RoPE by time" says nothing about the frequency band, which is load-bearing once positions are seconds — default base=10000 mis-covers HAR scales. |
| 3 | **Windowed-causal bounded-memory streaming + KV-cache** (vs SSM/Mamba backbone) | **SUPPORTS the choice** | **Keep; add attention sink** | HAR context is short/bounded (2.5–3.5 s motion, ~0.5 s pose) and patching already collapses N ~100×, so the SSM O(1)-over-infinite-history win never triggers; and SSM bidirectionality is a *separate backward scan, not a mask*, which would break the one-weight-set thesis. Only gap: pure windowed-causal softmax needs a persistent anchor patch (StreamingLLM). |
| 4 | **Per-patch causal emission as the online HAR paradigm** | **Right FAMILY, 3 sub-bets wrong** | **Keep; ADD lookahead + smoothing; FIX precedent** | Matches online-action-detection SOTA (LSTR, TeSTra, OnlineTAS, Continual Transformers), and low token-rate is an on-device advantage; but (a) strictly-causal is dominated by semi-online / bounded-lookahead, (b) raw per-patch emission over-segments without smoothing, (c) the cited precedent **P2LHAP is bidirectional+offline** — a category error as a *causal* precedent. |
| 5 | **One model vs. two-model / adapter alternatives for serving both modes** | **One-model CORRECT; recipe still weakest** | **Keep one model; CHANGE recipe** | Two models is the naive baseline the field abandoned (2× storage/maintenance/skew); the unified *offline* mode already matches/beats a standalone offline model (U2 beats a standalone non-streaming transformer by 5.6% rel. CER). But the unified *streaming* mode under-performs a specialized one unless distillation/consistency + right-context are added (Dynamic Chunk Conv ICASSP'23; consistency-reg RNNT 2026). |

**Overall confidence: high** on bets 3 and 5 (strong, convergent, directly-verified evidence); **high** on the *direction* of bets 1, 2, 4 with **medium** confidence on the *magnitude* of the gains for IMU/contrastive specifically (all primary evidence is ASR/video/LLM; the HAR transfer is an inference and must be measured — see §5).

---

## 3. Recommended changes (ranked, each tied to evidence)

### CHANGE 1 — MUST: joint dual-forward + in-place offline→online distillation. *(The headline.)*
Replace "sample one mask per micro-batch, loss unchanged" with: **each step, forward the same patches under both a full/offline mask and a causal-window mask, and add a loss that distills the offline per-patch posterior into the causal one** — the offline mode is an on-the-fly *teacher*, the streaming mode the *student*. For HALO's InfoNCE-to-text head this is unusually clean: distill the streaming patch's **softmax over label-text cosine similarities toward the full-context one (KL)**, or **MSE the projected patch embedding**, teacher→student, stop-gradient on the teacher.
- *Evidence:* Dual-mode ASR (Yu, Chiu, Sainath et al., **ICLR 2021**, arXiv:2010.06030) — verified abstract: joint training + in-place KD "significantly improves both emission latency and recognition accuracy of streaming ASR," at essentially zero cost to the offline mode. The ablation attributing ~20% relative streaming WER (test-other 10.6→8.5) is **second-hand from the findings** — treat the *magnitude* as unconfirmed, the *direction* as verified. Corroborated by consistency-regularization RNNT (arXiv:2604.19079, verified to exist) and Dynamic Chunk Convolution (Li et al., ICASSP 2023).
- *Why HALO especially cannot skip it:* ASR had friction (dual-mode conv/pool/BatchNorm all need causal variants); HALO has none — the two forwards differ **only** by the mask, so in-place distillation is almost free. The "mask is the only difference" purity HALO markets as elegance *removes its excuse* for omitting distillation.
- *Cost:* ~1.3–2× training compute (one extra masked forward/step; KV can be shared), **zero** offline-accuracy and **zero** inference cost.
- *My added caveat (not in the brief):* the teacher has future context the boundary-patch student **cannot** match, so an over-weighted distillation term can suppress the student's ability to commit early. Tune the distillation weight/temperature and consider down-weighting near-onset patches. The distillation term should be a **separate** KL/MSE, not folded into the in-batch-negative InfoNCE.

### CHANGE 2 — SHOULD (a latency *knob*, not a mandate): add a bounded-lookahead mask.
Add **causal + K-patch-lookahead** to the sampled mask set (K=0 lowest latency; K=1–3 near-offline accuracy) and expose K as an **inference-time latency dial.** Physical-time RoPE already indexes the future positions cleanly, so this is a bounded-latency chunk, KV-cache-compatible.
- *Evidence:* Time-Shifted Contextual Attention / Dynamic Right Context (arXiv:2502.15158, 2025) — **verified**: future context gives "10 to 13.9% relative WER reduction" on LibriSpeech. Transformer-Transducer (Tripathi et al., 2020) and Cascaded Encoders (Narayanan/Sainath, ICASSP 2021): ~1–2 s right context / 50–100 ms latency recovers *most* offline accuracy. OnlineTAS (Zhong et al., **NeurIPS 2024**): semi-online beats fully-online on all metrics — the closest task-analog.
- *Caveat I add:* this is genuinely a **knob**, not a default. For latency-critical single-event detection (falls, gestures) K patches (~1–2 s) of added latency is unacceptable, so the K=0 mode must remain first-class. Converting a binary full-vs-causal cliff into a graceful dial is the win; forcing lookahead on everyone is not.

### CHANGE 3 — SHOULD (cheap, credibility): causal streaming smoothing + fix the miscited precedent.
(a) Add **causal temporal label smoothing** (à la TLS-RWKV, 2024) or a small causal HMM/CRF over the per-patch label stream — raw per-patch causal emission provably over-segments (OnlineTAS: "severe over-segmentation" online; P2LHAP itself relies on a majority-vote smoothing pass). (b) **Stop citing P2LHAP (arXiv:2403.08214) as the causal precedent** — it is bidirectional+offline and gets its accuracy from smoothing over *surrounding* (future-inclusive) patches; it is evidence for HALO's *offline* mode only. Reposition against the video online-action-detection lineage (LSTR NeurIPS'21, TeSTra ECCV'22, Continual Transformers ICLR'23, OnlineTAS NeurIPS'24) and **report a measured offline→online drop** rather than assume parity.
- *My caveat:* since HALO currently trains on **single-activity windows**, intra-session boundary jitter is not exercised at training time and cannot be fully fixed until the deferred dense per-frame objective lands. Causal smoothing is a deployment-time band-aid; the real fix is coupled with dense supervision (see §5).

### CHANGE 4 — MUST-TUNE (implementation-level): calibrate the RoPE frequency band and re-anchor.
Keep RoPE-by-physical-time, but specify the band the bet leaves open:
- (i) **Tune base / number of frequencies to HAR time-scales** — fast oscillator half-period resolves the finest patch spacing (~0.5–1 s); slow oscillator's period exceeds the longest offline session span in seconds. The LLM default base=10000 was calibrated for integer token indices and mis-covers seconds.
- (ii) **Re-anchor timestamps to a local zero per session/window** — keeps absolute angles small (FP32 precision ceiling) and in-distribution; RoTHP's translation-invariance (Prop. 2) guarantees identical attention, *provided all cached keys use the same anchor*.
- (iii) **Ablate xPos-style decay or a time-ALiBi bias** for the causal-window mode (adds the recency signal vanilla RoPE lacks).
- (iv) **Keep physical time strictly in the rotation** — do NOT also concatenate a Time2Vec feature; RoMAE shows this breaks the relative-position property.
- *Evidence:* RoMAE (arXiv:2505.20535), RoTHP (arXiv:2405.06985), TimelyGPT/xPos (arXiv:2312.00817), TART (2025), CTLPE (arXiv:2409.20092), ALiBi (Press et al., ICLR 2022). The signal-processing bounds (aliasing base>L/2π; DC-stability tightening with depth; FP32 ceiling ~1e7) are from **arXiv:2602.10959 (Liu, 2026) — existence and bound-types verified**, but that paper derives them for *integer LLM positions* (validated on LLaMA/Mistral/DeepSeek); mapping L→"span in seconds" is my extrapolation, reasonable but unproven.
- *Clarification the findings make well:* the "20–200 Hz Nyquist" worry is a **red herring for the temporal PE** — the patch tokenizer already absorbs intra-patch high-frequency content, so temporal-patch attention only spans patch-rate scales. The real constraint is coarse-end phase wraparound (slowest oscillator period must exceed the max span), not the raw IMU sample rate.

### CHANGE 5 — SHOULD (surfaced bug, outside the three named bets): train the session-pooling.
HALO trains a **per-patch-only** InfoNCE loss but proposes to **mean-pool patches to a session label at offline inference** — a pooling operation used at inference but **never trained**. This is a train/serve mismatch for the session-label use case and a cross-patch op that must be verified causal-safe. Fix: add a **pooled-session objective** (or a learned attention-pool) trained *under the same sampled masks*, so the offline session-label path is represented at training time.

### Also fold in from the mask/init literature (low-cost):
- **Explicit sampling schedule, not uniform-over-3:** follow U2 (~50% full + 50% a randomly-drawn window spanning the deployment W range) so the streaming modes get enough gradient and the easy full mask doesn't dominate (Zhang et al., arXiv:2012.05481).
- **Warm-start the causal modes from the full-context checkpoint** — Dynamic Chunk Convolution (ICASSP'23) shows weight-init from the offline model materially shrinks the streaming gap.
- **Add an attention sink** — reserve a persistent anchor patch (session's first patch or a learned sink) in every causal window; pure windowed-causal softmax is known to destabilize without it (StreamingLLM, ICLR'24). Cheap and currently omitted.

---

## 4. Strongest alternative design — stated at full strength, then a reasoned decision

The strongest *architectural* alternative is a **state-space / selective-SSM (Mamba) or linear-attention backbone** replacing windowed-causal attention, for O(1)/step and O(1)-memory streaming over unbounded history. Argued at full strength: Mamba (Gu & Dao, arXiv:2312.00752) is "hardware-aware," gives constant per-step cost and constant state, and the strongest efficient-LM results are recurrence/attention hybrids — a real, serious pitch.

**Decision: REJECT the SSM swap.** Four load-bearing reasons, in descending force:
1. **It breaks HALO's entire thesis.** Bidirectionality in *every* deployed HAR/vision SSM (ViM arXiv:2401.09417; HARMamba arXiv:2403.20183; BabyMamba 2026) is a **separate backward scan with its own parameters — not a mask.** So an SSM cannot express "offline vs online = same weights, different mask." It forces either always-causal inference (forfeiting the offline bidirectional gains ViM shows are worth several points on dense tasks) or a second computational path (destroying the one-weight-set property that is the paper's whole point). This alone is disqualifying.
2. **The advantage never triggers.** HAR context is short/bounded (motion-mode 2.5–3.5 s, pose ~0.5 s; Wang et al., Sensors 2018); a causal window of tens of patches is already effectively O(1). "O(1) over infinite history" solves a problem HALO does not have.
3. **Patching already removed the cost SSMs sell against.** The per-patch tokenizer collapses N ~100× (10-min session ≈ 600 tokens; 600² is trivial). SSMs claim linear scaling exactly in the regime HALO has already exited.
4. **NPU-hostile.** The selective scan is a sequential, data-dependent recurrence that parallelizes poorly and resists the static INT8 quantization phone/watch NPUs require (Quamba; Mamba-PTQ arXiv:2407.12397); attention is matmul — statically quantizable. Plus an IMU-specific strike: lightweight Mamba's low-frequency bias underperforms CNN/Transformer on HAR unless frequency-decoupled (Machar, 2026 — second-hand).

The **one legitimate carry-over** from this lens is the **attention sink** (Change 5 bullet). If unbounded O(1) memory ever becomes truly necessary (bare MCUs without matmul NPUs, or tasks needing minutes-to-hours of continuous context), the better-fitting middle path is **local attention + gated linear recurrence (Griffin, arXiv:2402.19427)** — not pure Mamba — but even that bakes in a fixed decay rather than a swappable mask, partially breaking the mask-swap unification. So: revisit only under those specific future constraints.

The strongest *serving* alternative — **two specialized models** — is also rejected (2× storage/maintenance/train-serve skew; the unified offline mode already matches/beats a standalone offline model, U2 arXiv:2012.05481). Its **valuable half — making the offline model teach the online model — is exactly Change 1.** Reject two models; adopt the distillation.

The strongest *within-paradigm* alternative worth keeping in the back pocket if distillation+lookahead still under-deliver is the **cascaded-encoder pattern** (shared causal base + a small non-causal top stack, single head; Narayanan/Sainath ICASSP'21) — still one model, deployed on-device, +10–27% relative offline. But it weakens the strict "mask is the only difference" purity, so prefer distillation+lookahead first.

---

## 5. Open risks + what to validate in our own ablations

**The core honest gap: all primary evidence is ASR / video-OAD / LLM. IMU-specific unified streaming+offline literature is thin.** The direction transfers; the magnitudes do not. Run these ablations on a rate-mixed, variable-patch-duration, **leave-one-subject-out** split (subject leakage is already a flagged integrity issue in this codebase — do not let it inflate these numbers):

1. **The must-change, measured:** `random-mask-no-distill` vs `joint+in-place-distillation` vs `+lookahead`. Report the offline(full-mask)→online(causal-window) accuracy drop for each. This is the single most important experiment; it directly tests whether Dual-mode ASR's ~20% relative streaming gain survives the switch from token-CTC to per-patch-InfoNCE-to-text. **Unverified for IMU/contrastive — measure it.**
2. **Distillation failure mode:** does hard distillation on **onset/boundary patches** (where the teacher has future context the student cannot) hurt early-commit confidence? Ablate distillation weight and near-boundary down-weighting.
3. **Latency dial:** offline==online agreement and accuracy vs K∈{0,1,2,3} lookahead patches; find the accuracy/latency knee.
4. **RoPE band + a subtle data-coupled extrapolation risk:** training on **single-activity windows** means the offline full-mask mode and the RoPE band are both **extrapolating** at long physical-time spans when deployed on long multi-activity sessions — the training window-W distribution never covers deployment-length Δt. Ablate `{physical-time RoPE, RoPE+xPos-decay, time-ALiBi, integer-index RoPE}` **on long multi-activity test sessions**, not just short single-activity windows, and confirm the band covers deployment span.
5. **Session-pooling train/serve mismatch (Change 5):** compare untrained mean-pool vs a trained pooled-session objective vs learned attention-pool for the offline session-label task.
6. **Attention-sink long-run stability:** stream for hours; confirm no degradation; test with/without a persistent anchor patch. Note a KV-cache operational subtlety: re-anchoring timestamps mid-stream (needed to bound FP32 phase drift over long sessions) requires re-rotating cached keys to the new anchor (one cheap complex-multiply per key) or a cache flush — verify this path.
7. **On-device budget:** real energy/latency/step for the dual-branch softmax transformer vs a quantized causal-TCN / linear-attention baseline; the low patch token-rate is HALO's friend but this must be *shown*, and the Continual-Transformers caveat (stacked causal-attention layers can reintroduce redundancy) means the multi-layer temporal branch must be verified to actually stream cheaply.
8. **Leakage audit (defeats the whole premise if violated):** verify nothing except temporal attention mixes across time in *either* mode — any conv/pool/normalization using cross-patch statistics reintroduces train/infer mismatch. HALO's per-patch tokenizer + per-token LayerNorm should be safe; the offline session-pool is the suspect (Change 5).
9. **Coupled decision:** the "defer dense per-frame supervision" and "mixed-mask" decisions **interact** and must be evaluated together — causal predictions near activity onsets have the least context and benefit most from *both* right-context (Change 2) and an offline teacher (Change 1); deferring dense supervision compounds the streaming weakness rather than being orthogonal to it.

---

## 6. References (paper — venue — year)

**Mixed-mask / unified streaming (Bets 1, 5):**
- Yu, Han, Gulati, Chiu, Li, Sainath, Wu, Pang et al. "Dual-mode ASR: Unify and Improve Streaming ASR with Full-context Modeling." **ICLR 2021.** arXiv:2010.06030. *(Most on-point; distillation ablation numbers second-hand.)*
- Zhang, Wu et al. "Unified Streaming and Non-streaming Two-pass End-to-end Model for Speech Recognition (U2)." **arXiv 2012.05481 / Interspeech**, 2020–21.
- Wu, Zhang, Yang, Peng et al. "U2++: Unified Two-pass Bidirectional End-to-end Model." arXiv:2106.05642, 2021.
- Yao, Wu, Yang, Zhang et al. "WeNet: Production Oriented Streaming and Non-streaming E2E Speech Recognition Toolkit." **Interspeech 2021.** arXiv:2102.01547.
- Li et al. "Dynamic Chunk Convolution for Unified Streaming and Non-Streaming Conformer ASR." **ICASSP 2023.**
- Narayanan, Sainath et al. "Cascaded Encoders for Unifying Streaming and Non-Streaming ASR." **ICASSP 2021.** arXiv:2010.14606.
- Tripathi et al. "Transformer Transducer: One Model Unifying Streaming and Non-streaming Speech Recognition." 2020.
- Andrusenko et al. "Reducing the Offline-Streaming Gap for Unified ASR Transducer with Consistency Regularization." **arXiv:2604.19079, 2026** *(existence verified).*
- "Improving Streaming Speech Recognition With Time-Shifted Contextual Attention and Dynamic Right Context Masking." **arXiv:2502.15158, 2025** *(10–13.9% rel. WER from future context — verified).*
- Kim et al. "Multi-mode Transformer Transducer with Stochastic Future Context." Interspeech 2021. arXiv:2106.09760.
- Elbayad, Besacier, Verbeek. "Efficient Wait-k Models for Simultaneous Machine Translation." Interspeech 2020. arXiv:2005.08595.
- Dong et al. "UniLM: Unified Language Model Pre-training." **NeurIPS 2019.** arXiv:1905.03197.
- Du et al. "GLM: General Language Model Pretraining with Autoregressive Blank Infilling." **ACL 2022.**
- Raffel et al. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)." **JMLR 2020.** arXiv:1910.10683.

**Positional encoding by physical time (Bet 2):**
- Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." **Neurocomputing 2024** (arXiv:2104.09864, 2021).
- Zivanovic et al. "Rotary Masked Autoencoders are Versatile Learners (RoMAE)." arXiv:2505.20535, 2025.
- Gao et al. "RoTHP: Rotary Position Embedding-based Transformer Hawkes Process." arXiv:2405.06985, 2024. *(Proves translation-invariance.)*
- Song et al. "TimelyGPT: Extrapolatable Transformer Pre-training for Long-term Time-Series Forecasting in Healthcare." **ACM-BCB 2024.** arXiv:2312.00817.
- Shi et al. "Time-Aware Rotary Transformer (TART)." **IEEE IoT Journal 2025.**
- Liu, F. "Rotary Positional Embeddings as Phase Modulation: Theoretical Bounds on the RoPE Base for Long-Context Transformers." **arXiv:2602.10959, 2026** *(existence + bound-types verified; derived for integer LLM positions — seconds mapping is our extrapolation).*
- Kim et al. "CTLPE: Continuous-Time Linear Positional Embedding for Irregular Time Series." arXiv:2409.20092, 2024.
- Press, Smith, Lewis. "Train Short, Test Long: Attention with Linear Biases (ALiBi)." **ICLR 2022.** arXiv:2108.12409.
- Kazemi et al. "Time2Vec." arXiv:1907.05321, 2019. / Shukla, Marlin. "Multi-Time Attention Networks (mTAND)." **ICLR 2021.** arXiv:2101.10318.
- Foumani et al. "ConvTran (tAPE + eRPE)." **Data Mining and Knowledge Discovery 2023.**

**Backbone: windowed-causal vs SSM (Bet 3):**
- Gu, Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752, 2023.
- Zhu et al. "Vision Mamba (ViM): Bidirectional State Space Model." **ICML 2024.** arXiv:2401.09417.
- Li et al. "HARMamba: Efficient and Lightweight Wearable Sensor HAR Based on Bidirectional Mamba." **IEEE IoT Journal**, arXiv:2403.20183, 2024.
- Jelassi, Brandfonbrener, Kakade, Malach. "Repeat After Me: Transformers are Better than State Space Models at Copying." **ICML 2024.** arXiv:2402.01032.
- Jiang et al. "Mistral 7B" (sliding-window attention + rolling-buffer KV cache). arXiv:2310.06825, 2023.
- Xiao et al. "Efficient Streaming Language Models with Attention Sinks (StreamingLLM)." **ICLR 2024.** arXiv:2309.17453.
- Sun et al. "Retentive Network (RetNet)." arXiv:2307.08621, 2023. / Yang et al. "Gated Linear Attention." **ICML 2024.** arXiv:2312.06635.
- De et al. "Griffin: Mixing Gated Linear Recurrences with Local Attention." arXiv:2402.19427, 2024.
- Quamba (post-training quant for selective SSMs, OpenReview) / "Mamba-PTQ" arXiv:2407.12397, 2024.
- "Machar: Frequency-Aware Mamba–Convolution Hybrid for Sensor-Based HAR." IEEE IoT Journal, 2026 *(second-hand).* / Mandal, "BabyMamba-HAR." arXiv:2602.09872, 2026 *(second-hand).*

**Per-patch causal HAR paradigm (Bet 4):**
- Li, Nie et al. "P2LHAP: HAR, segmentation and forecast through Patch-to-Label Seq2Seq Transformer." **IEEE IoT Journal 2024.** arXiv:2403.08214. *(Bidirectional+offline — not a causal precedent.)*
- Zhong et al. "OnlineTAS: An Online Baseline for Temporal Action Segmentation." **NeurIPS 2024.** arXiv:2411.01122.
- Hedegaard, Bakhtiarnia, Iosifidis. "Continual Transformers: Redundancy-Free Attention for Online Inference." **ICLR 2023.** arXiv:2201.06268.
- Zhao, Krähenbühl. "Real-time Online Video Detection with Temporal Smoothing Transformers (TeSTra)." **ECCV 2022.**
- Xu et al. "Long Short-Term Transformer for Online Action Detection (LSTR)." **NeurIPS 2021.** arXiv:2107.03377.
- Zhu et al. "TLS-RWKV: Real-time Online Action Detection with Temporal Label Smoothing." **Neural Processing Letters 2024.**
- Farha, Gall. "MS-TCN: Multi-Stage TCN for Action Segmentation." **CVPR 2019.** / Zhang et al. "HAR Based on Motion Sensor Using U-Net." IEEE Access 2019.
- Schäfer, Leser. "TEASER: Early and Accurate Time Series Classification." **DMKD 2020.** / "Early Classification of Time Series: A Survey and Benchmark." arXiv:2406.18332, 2024.
- Wang et al. "Impact of Sliding Window Length in Indoor Human Motion Modes and Pose Pattern Recognition." **Sensors (MDPI) 2018.** / Jaén-Vargas et al. "Effects of sliding-window variation in acceleration-based HAR." PeerJ CS 2022. / Ignatov. "Real-time HAR from accelerometer via CNN." Applied Soft Computing 2017.
- "Foundation Models Defining A New Era in Sensor-based HAR: A Survey and Outlook." 2026. / "Wearable Foundation Models Should Go Beyond Static Encoders." 2026.
- Tokas et al. "Real-Time Deployable Attention-Driven CNN-LSTM for HAR." IEEE Sensors Journal 2025.

---

## Verification status (honest)
- **Directly verified this session:** Dual-mode ASR abstract-level claim (arXiv:2010.06030) — distillation+joint training "significantly improves … accuracy of streaming ASR," SOTA on LibriSpeech; the exact ablation figures (test-other 10.6→8.5, ~20% relative) are **second-hand** from the research findings (PDF ablation table not in the abstract). arXiv:2502.15158 lookahead "10–13.9% relative WER reduction" — **verified verbatim.** arXiv:2602.10959 RoPE base-bounds preprint — **existence and bound-types verified** (Feilong Liu, submitted 2026-02-11; aliasing/DC/precision bounds confirmed), but derived for **integer LLM positions** — the physical-seconds (L in seconds) mapping is our reasonable-but-unproven extrapolation. arXiv:2604.19079 consistency-regularization RNNT — **existence verified.**
- **Second-hand (cited from findings, not independently confirmed):** the 2026 HAR-Mamba preprints (Machar, BabyMamba arXiv:2602.09872) and the precise Dual-mode distillation magnitude.
- **Biggest honest caveat overall:** every primary source is ASR / video-OAD / LLM / general time-series. The IMU/contrastive-HAR transfer is an *inference*; the recommended changes' *direction* is well-supported but their *magnitude for HALO must be measured* (§5), and subject-leakage-clean, long-multi-activity-session evaluation is required for the numbers to mean anything.
