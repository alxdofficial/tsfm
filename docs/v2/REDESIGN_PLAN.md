# HALO V2 Redesign Plan

**Date:** 2026-07-01 | **Branch:** V2 | **Status:** design approved, implementation starting

Goals (from project lead):
1. Make the model actually useful/playable for real-world phone & smartwatch HAR (robustness + dev artifacts).
2. Replace the tokenizer with a mathematically principled, rate-agnostic design — no aliasing/resampling concerns, fewer moving parts.
3. KEEP language-as-channel-semantic-encoding (ChannelTextFusion).
4. One training objective serving both segment classification and live streaming.
5. Consolidate evaluation; eliminate open-set/synonym nitpicking.

Component design specs: [tokenizer](design_tokenizer.md) | [objective](design_objective.md) | [application](design_application.md) | [evaluation](design_evaluation.md)

---

## HALO Next-Submission Redesign — Integrated Technical Plan

## 0. The redesigned HALO in one picture

**End-to-end pipeline (one branch, one head, two output modes):**

```
native IMU stream (any rate, any channel set, phone/watch)
  │  gravity-frame canonicalization (Mahony / OS gravity)  +  yaw-only aug
  ▼
create_patches → native-rate zero-padded rDFT  (NO interpolation, NO resampling)
  ▼
PhysicalFilterbankTokenizer  (fixed constant-Q Gaussian bank in physical Hz)
   log-compress → frozen per-band standardize → +amplitude scalar → +Nyquist mask
   one shared Linear(→384)                                     ⇒ tokens (B,P,C,384)
  ▼
KEPT language channel-encoding: ChannelSemanticEncoding + ChannelTextFusion
   (per-channel placement/rate/sensor text; gravity as its own described channel)
  ▼
DualBranchTransformer:  within-patch cross-channel  ×  temporal self-attn
   temporal mask sampled per micro-batch ∈ {full, causal-∞, causal-W}   (train-both)
   RoPE over PHYSICAL-TIME patch offsets  (rate-invariant, KV-cache-exact)
  ▼
Dense per-patch head (per_patch_prediction=True): (B,P,384) L2-normed embeddings
   + tiny boundary head g(fused)→(b_t,o_t)
  ▼
score = cosine( e_patch , label_bank.encode(candidate_strings) )        ← open-vocab
  ├── SEGMENT: soft-logit-pool patches over a finished recording → one label
  └── STREAM : ring-buffer + KV-cache, emit ŷ_t per patch, boundary-gated smoothing
  ▼
Consolidated eval: ZS-XD macro-F1 (segment) + frame-F1/boundary-tol (stream),
   subject-disjoint, exact-match vs native label strings, no synonym ontology.
```

**One-sentence contribution:** *HALO is a language-aligned IMU foundation model with a principled, rate-invariant, anti-aliased-by-construction tokenizer and a single dense per-patch text-alignment objective that yields both cross-dataset zero-shot segment classification and open-vocabulary on-device live streaming — a combination (rate-agnostic + open-vocab + unified streaming + deployable) that no prior HAR model offers jointly.*

---

## 1. Interface reconciliation (the critical part)

Each seam below is stated as an explicit contract. Conflicts between the four designs are flagged with **⚔** and resolved.

### 1.1 Tokenizer output → encoder input
- **Contract:** `PhysicalFilterbankTokenizer.forward(patches, sampling_rate_hz, patch_len_samples) → (B, P, C, d_model=384)`. Input `patches: (B,P,S,C)` native-rate zero-padded to fixed `S=256`; `sampling_rate_hz: (B,)`; `patch_len_samples: (B,)` true `N` for Hann windowing.
- **⚔ Conflict:** the OBJECTIVE design hedges with a `token_adapter: Linear(d_token→d_model)` "if the tokenizer emits tokens directly," while the TOKENIZER design already ends in a shared `Linear(→d_model)`. **Resolution:** the tokenizer emits `d_model` directly; **no `token_adapter`**. In `model/encoder.py`, delete the `feature_extractor_type` cnn/spectral branch (lines ~124–141) and route patches straight through the tokenizer. The old `feature_extractor` slot is replaced, not adapted.
- **Rate metadata:** `sampling_rate_hz`/`patch_len_samples` are already carried as `sampling_rates`/`patch_sizes` through `forward_from_raw` (`semantic_alignment_train.py`). Contract: **stop interpolating to `TARGET_PATCH_SIZE`; zero-pad to `S` and pass rate through.** Delete `interpolate_patches()` and the interpolation step in `preprocess_imu_data()` (`model/preprocessing.py`).
- **Observability mask:** tokenizer concatenates the Nyquist mask `o_k` into the projection input, so unmeasurable bands (e.g. >9 Hz at 20 Hz) are signalled, not faked. This is consistent with — and now *matches* — the channel text that asserts "sampled at {r}Hz".

### 1.2 Per-patch tokens → dense objective
- **Contract:** encoder returns `(B,P,C,384)`; head with `per_patch_prediction=True` returns `(B,P,384)` (already exists, `model/semantic_alignment.py:454-458`). `flatten_per_patch_embeddings` flattens to `(N_valid,384)`.
- **⚔ Conflict:** today every patch is aligned to the *session* label (`frozen_text.repeat_interleave(patch_counts)`). Dense objective needs a **per-frame** label. **Resolution:** replace the per-session expansion (`semantic_alignment_train.py:951-953`) with `label_bank.encode(flat_frame_labels)` / `encode_frozen(...)`. Straddling patches use a soft mixture target `m_i ∈ Δ^L` fed into the existing `(N,N)` soft-target path (`semantic_loss.py:150-187`) — **no core change to `InfoNCELoss`**. Single-activity patches recover a one-hot `m_i` = today's behavior, so the change is strictly a generalization (de-risks it).
- **Frozen soft targets + MoCo queue are KEPT.** Set `TSFM_QUEUE_MODE=semantic` so a queued "walking" patch is not a hard negative for an in-batch "strolling" patch.

### 1.3 Causal encoder → streaming app runtime
- **Contract:** streaming = drive `TemporalSelfAttention` with a banded causal mask (patch `t` attends `[t−W+1,t]`, `W≈16–32`); the mask is already threaded end-to-end and ANDed into SDPA (`transformer.py:91-99`). Cross-channel attention is within-patch → no temporal leakage → unchanged.
- **⚔ Conflict — positional encoding (two designs, two proposals):** TOKENIZER wants temporal PE indexed by **physical time** (seconds) for rate-invariance; OBJECTIVE wants **RoPE / windowed-relative** for unbounded streams + KV-cache exactness. Current PE is absolute sinusoidal over `max_patches=5000` (`positional_encoding.py:43-64`) — runs off the table on long streams. **Resolution (unifies both):** **RoPE keyed to physical-time patch offsets** `Δτ = (t−j)·stride_sec`. RoPE is relative (exact under KV-cache, unbounded) *and*, keyed to seconds, gives a 5 s activity the same temporal code at 20 Hz and 100 Hz. `ChannelSemanticEncoding` untouched.
- **⚔ Conflict — causal vs bidirectional:** APP requires causal for O(1)/patch streaming; EVAL/segment wants full context for best accuracy. **Resolution: train-both / eval-both** — sample the mask per micro-batch from `{full, causal-∞, causal-W}` ≈ `{0.34,0.33,0.33}`. One weight set is valid offline (full) and online (causal-W). This is the ⚑ decision a human should ratify (§4), but the recommended default.
- **Runtime:** `HALOStreamSession` holds {gravity-filter state, per-layer KV ring buffer (last `W` patch tokens), boundary-smoothing buffer}. Cost/patch = `O(W·d) + O(C²·d)`, independent of stream length. Latency = `patch_sec + (k+0.5)·patch_sec` smoother lag.
- **Streaming equivalence guarantee:** because tokenizer normalization is **frozen per-band** (no cross-patch stats) and PE is relative, the per-patch token stream is bit-identical offline vs incremental — assert to `1e-5`. This is the substrate that makes the unified objective coherent.

### 1.4 Label-bank / channel-text freezing → on-device export
- **Contract:** at deploy, SBERT drops out. Precompute (a) **label matrix** `[L×384]` = `LearnableLabelBank.encode(labels)`; (b) **channel-text embeddings** for enumerated placement×sensor×rate combos, fed as constants into `ChannelTextFusion` (replacing its runtime SBERT pooling at `token_text_encoder.py:431-438`).
- **⚔ Conflict:** `ChannelTextFusion` must stay identical for training (SBERT) and export (constants). **Resolution:** it is already `P`-independent — it pools each channel's text once and broadcasts `(B,1,C,384)`. So training runs SBERT, export swaps in precomputed constants; **shape and math are identical** for `P=1` (stream) and `P=full` (offline). No code fork.
- **Traceability:** add pure-tensor `forward_export(patches, channel_emb_const, label_matrix_const)` with static shapes / dynamic axes for #patches, #channels. The tokenizer being a fixed matmul/FFT (not per-sample Python `F.interpolate`) is what makes the graph traceable. Targets: Core ML `.mlpackage`, ONNX + ONNX-web, ExecuTorch `.pte`; parity `|Δcos|<1e-3`.

### 1.5 New closed-set eval → what the model must output
- **Contract:** for target dataset `D` with native strings `L_D`, model outputs per-patch embeddings; scoring is `argmax_{c∈L_D} cos(f(x), g(t_c))`, **exact match, macro-F1 primary.**
- **⚔ Conflict:** EVAL design v2 argmaxes at the **window** level; OBJECTIVE emits **per-patch**. **Resolution:** segment prediction = **soft logit-pool** the per-patch scores `S_t=E_t T^⊤` → `argmax_k Σ_t softmax(S_t/τ)[k]`. This *is* the dense head pooled; hard per-patch majority vote remains available as a special case. Eval consumes `forward_from_raw(return_per_patch=True)` (already exists, `evaluate_tsfm.py:288`).
- **Two protocols, one scoring rule:** (A) **segment** ZS-XD macro-F1 (pool patches); (B) **streaming** frame-F1 + boundary-tolerance (emit per patch). Same cosine-vs-`L_D` scoring; only the pooling/smoothing differs. This is what lets goals 4 (unified objective) and 5 (consolidated eval) share one metric family.

---

## 2. Component summaries

### 2.1 Tokenizer — Physical-Hz Filterbank (PHz-FB)
- **Chosen default (Arm A):** fixed constant-Q Gaussian filterbank in physical Hz on a native-rate zero-padded rDFT, then log-compression + frozen per-band standardization + amplitude scalar + one shared `Linear(→384)`. Rate-invariant and anti-aliased **by construction** — no resampling exists, so nothing can alias. Justified by [Ghaffari 2024](https://consensus.app/papers/details/59964a2584355a5ba01c23cb2b451dbc/): gains come from normalization+compression, not learned filter shapes.
- **Key spec:** bin→Hz `φ[m]=m·r/S`; `H_k[m]=exp(−½((φ[m]−f_k)/σ_k)²)`, `σ_k=f_k/2Q`, `Q≈4`, `f_k` log-spaced `0.3–15 Hz`, `K=32`, `S=256`. Band energy `E_{k,c}=Σ_m H_k[m]|X_c[m]|²`; `ê=(log(1+E)−μ_k)/s_k`; token `= Linear(concat[ê(K), o(K), a_c(1)]→384)`. Observability `o_k=1[f_k+2σ_k ≤ 0.9·r/2]`.
- **Validated (PoC):** rate-invariance cosine 0.997/0.991/0.995 across 20/50/100 Hz; exact peak-band coincidence at 2 Hz & 5 Hz; current interp-to-64 aliases 22 Hz→20.67 Hz phantom.
- **Params/latency:** Arm A `Linear(65→384)=25,344` trainable (0 filter params); leaner impute-mask variant 12,672. **vs current `SpectralTemporalExtractor` ≈55–100K** → 2–4× fewer, one branch, no dual-LayerNorm, no interpolation. Single batched `rfft(n=S)`, `torch.compile`-friendly, export-traceable.
- **Deleted:** `interpolate_patches()`, interpolation step + `target_patch_size`/`interpolation_method` args (`preprocessing.py`); default per-patch z-score on main path; `MultiScaleConv1D`, `ChannelIndependentCNN`, `FixedPatchCNN`, `SpectralTemporalExtractor` (`feature_extractor.py`); `assert seq_len==target_patch_size` + dual-branch plumbing (`encoder.py:236`, 124–141); CNN/spectral config keys (`config.py`).
- **Added:** `PhysicalFilterbankTokenizer`; config keys `{n_bands:32, f_min:0.3, f_max:15, Q:4, dft_size:256, nyquist_margin:0.9, tokenizer_learnable:false, tokenizer_norm:"frozen", use_amplitude:true}`.
- **Ablation arms:** ⚑ **A (fixed, default)** vs **B (learnable-but-constrained Gabor centers/bandwidths, softplus+Nyquist-clamped)** [Michau 2021](https://consensus.app/papers/details/5e035cb6e5425a97b056e1c803d74770/) vs **C (AaSP band-limited kernels)** [AaSP 2025](https://consensus.app/papers/details/519c2b54e3ee5ed99ebb9d68b231a4c7/). Same I/O, config-selectable. Also: norm frozen/PCEN/off, log-compress on/off, amplitude on/off, patch-size-`D` sweep.

### 2.2 Unified objective — dense per-patch text-aligned sequence labeling
- **Chosen approach:** keep symmetric InfoNCE + frozen-SBERT soft targets + MoCo queue; make the target text **per-frame** instead of per-session. Add smoothness + boundary terms.
- **Loss:** `L = L_NCE(dense) + λ_sm·L_sm + λ_bd·L_bd`, defaults `λ_sm=0.1, λ_bd=0.3`. `L_sm` = gated symmetric-KL between adjacent-patch posteriors (gate=0 at true boundaries) [P2LHAP](https://consensus.app/papers/details/ca93f18a1445527fb380739fba835277/); boundary head `g(fused)→(b_t,o_t)` (BCE + smooth-L1) [Duan 2023](https://consensus.app/papers/details/bbc337dc251c51cdb021b1799f3a6cb1/). Straddling patches → soft mixture target `m_i`.
- **Params/latency:** boundary head = one tiny 2-output MLP beside `projection_head`; negligible. Reuses the ~9.3M-param-saving per-patch path (temporal-attn+pooling head already skipped).
- **Deleted/changed:** per-session label expansion (`semantic_alignment_train.py:951-953`); absolute PE → RoPE (`positional_encoding.py:43-64`). **No change** to `semantic_loss.py` core or `ChannelTextFusion`.
- **Added:** per-frame label lookup; `L_sm`/`L_bd`; per-micro-batch temporal-mask sampler; KV-cache streaming path (`transformer.py:91-99,394-432`); boundary head (`semantic_alignment.py:414`); frame-label derivation in loader.
- **Ablation arms:** dense vs segment-only supervision; ⚑ causal vs bidirectional vs train-both; streaming latency/accuracy sweep (`k`,`W`); boundary term on/off; queue `hard_neg` vs `semantic`.

### 2.3 App / robustness — deployable, playable, robust
- **Gravity-frame canonicalization (rank-3 fix):** Mahony/complementary filter (or OS gravity) rotates accel+gyro so gravity≡+Z, removing 2 tilt DOF of mount nuisance while preserving posture; residual **yaw-only aug** handles the unobservable heading DOF. Re-enables the rotation aug currently hard-disabled at `semantic_alignment_train.py:377`. Expose gravity as an explicit **described channel** → posture handed to the model via ChannelTextFusion. [HAR-DoReMi](https://consensus.app/papers/details/30228e1951c751a18e53494118d60ef6/) + [UniMTS](https://consensus.app/papers/details/f924af83748257a8b57407578dfb8e84/).
- **`HALO.from_pretrained().predict()`** (new `halo/api.py`): open-vocab `labels=[...]`, `placement`, `rate_hz`, optional `timestamps`, `channels`; auto-detect g vs m/s²; abstain threshold (max-cosine<τ→"unknown"); returns `Prediction(label, confidence, per_label, embedding, abstained)`. `set_labels()` for runtime label-swap.
- **One canonical checkpoint:** bless `training_output/semantic_alignment/small_deep_v2_4b3fdd6/best.pt` as `halo-small-deep-v2` (≡ `halo_best.pt` per MEMORY); strip dead temporal-attn+pooling head, fold LayerNorms, ship fp16 + model card (macro-F1 primary, known failure modes).
- **Export:** freeze text side → graph = gravity-canon → PHz-FB matmul → encoder → channel-const fusion → head → label-matrix matmul → softmax. Core ML / ONNX(-web) / ExecuTorch; stateful causal export with KV I/O.
- **Accel-only + corruption robustness:** modality dropout (drop gyro/axes), accel-only gravity low-pass fallback; train-time BLE-style corruption (Bernoulli dropout, burst gaps, sensor noise, clipping, clock jitter); per-patch coverage ratio routed through `patch_mask`; the integrate-don't-resample tokenizer tolerates gappy timestamps natively.
- **Demo:** phone-web `DeviceMotion` page streaming to HALO (or ONNX-web client-side) with live labels + "add activity live" box; `halo predict file.csv --labels ...` CLI.
- **Ablations:** A1 device-frame vs SO(3) vs gravity+yaw; A1b OS-gravity vs Mahony; A1c gravity-channel-as-language; B1 rate flatness (native/×0.5/×2/37 Hz); B2 placement-text correct/wrong/generic; C1 corruption AUC; D1 accel-only gap.
- **Showcase scenarios (pick 3):** (1) add a NEW activity live (headline capability); (2) phone-in-pocket vs watch (placement generalization, head-to-head [UniMTS]); (3) fitness rep-counting (streaming). Drop fall-detection from headline.

### 2.4 Evaluation — consolidated v2
- **3 metrics, one scoring rule, one split rule.** **A — ZS-XD (primary):** `argmax_{c∈L_D} cos(f(x),g(t_c))` over the target dataset's own verbatim strings, **macro-F1** primary + balanced accuracy [Modality-Gap/Ghosh 2026](https://consensus.app/papers/details/5e6c92a6f65a5f029618ecf827a0f54c/), [UniMTS](https://consensus.app/papers/details/f924af83748257a8b57407578dfb8e84/). **B — FS-1%/10% (secondary):** subject-disjoint few-shot, multi-seed. **C — T→S Recall@1 (appendix diagnostic).**
- **Kills the synonym ontology:** target-closed-set means training strings are never compared to test strings → `label_groups.py` deleted from the eval path entirely (no `use_groups=True`/`open_set=True`). Removes the test-label leakage baked into the group boundaries.
- **Subject-disjoint splits — pure code change:** subject index is already at `label_native.npy[:, 0, 1]` (currently discarded). LOSO for ≥8 subjects, GroupKFold(≤5) for small ones. Cite inflation magnitude [Gholamiangonabadi 2020](https://consensus.app/papers/details/f677839713b55ef087ff10478c05aec1/), [Rehman 2024](https://consensus.app/papers/details/fcdfa1dfc9985f95ba51f666f8800f90/).
- **Error bars:** subject-stratified bootstrap (B=1000) for ZS; folds×seeds for FS. **Fairness:** parity row = same physical signal resampled once with `scipy.signal.resample_poly` (anti-aliased) + neutral channel text for all models; native-rate and rich-channel-text reported as explicit **+Δ capability rows** (appendix). Once PHz-FB is genuinely rate-invariant, the native-rate Δ→0 becomes a *selling point*.
- **Streaming protocol (from objective):** frame-F1 + boundary-tolerance + rep-count MAE, same cosine-vs-`L_D` scoring — folds into the same metric family.
- **Deleted:** `evaluate_zero_shot_open_set`, group imports, `open_set`/`use_groups` branches, mask machinery (`evaluate_tsfm.py`, `EVALUATION_PROTOCOL.md`). **Added:** `--eval-rate`, `--channel-text {native,neutral,none}`, GroupKFold/LOSO splitters, bootstrap CI.

---

## 3. What this does to novelty & competitiveness

**The joint capability no competitor has.** UniMTS and GOAT each own *one* axis; HALO now owns their intersection plus deployment:

| Capability | UniMTS | GOAT | HALO (redesigned) |
|---|:--:|:--:|:--:|
| Open-vocab / text-aligned labels | partial | ✓ | ✓ (kept) |
| Per-channel language conditioning (placement/rate/sensor) | ✗ | ✓ | ✓ (kept, goal 3) |
| Principled rate-agnostic tokenizer, **no resampling/aliasing** | ✗ (SO(3)+resample) | ✗ | ✓ **by construction** |
| Unified segment **and** streaming from one objective/head | ✗ | ✗ | ✓ (dense per-patch) |
| On-device export + live demo + abstain | ✗ | ✗ | ✓ |
| Leakage-free, subject-disjoint, single-scoring eval | — | — | ✓ |

- **vs UniMTS:** UniMTS achieves orientation-invariance by *randomizing* SO(3) and resampling — which destroys posture signal and aliases. HALO instead *canonicalizes* to the gravity frame (posture preserved) + yaw-only aug, and never resamples. The rate-invariance is provable (cosine 0.99+, closed-form sinusoid test), not empirical.
- **vs GOAT:** GOAT conditions on channel text but is offline/segment-only. HALO keeps that channel-text path *and* adds a dense streaming objective + real export, with the channel text now *consistent* with an anti-aliased tokenizer (the text no longer asserts a rate the pipeline interpolated away).

**Reviewer critiques neutralized:**
1. "Why 87 strings / who drew these groups?" → gone; exact-match vs native labels.
2. "Synonym ontology encodes held-out test labels (leakage)." → ontology deleted from eval path.
3. "Splits aren't subject-disjoint (inflated 10–15 pts)." → LOSO/GroupKFold using the already-present subject index.
4. "Accuracy 42 vs macro-F1 21 — imbalance hidden." → macro-F1 primary + balanced accuracy.
5. "TSFM got native rate + rich text; baselines didn't." → parity row + explicit Δ capability rows.
6. "FFT bins are rate-dependent / interpolation aliases." → physical-Hz bins + no resampling; demonstrated aliasing fix.
7. "Fixed windows contain multiple activities." → soft mixture targets + boundary head.
8. "It's a benchmark model, not usable." → `predict()`, checkpoint, exported artifacts, live demo.

---

## 4. Phased build plan

Dependencies: **eval fix before any retraining** (so every number is trustworthy); **tokenizer + causal encoder before streaming app**; **label-bank freeze before export**.

**Milestone 0 — Eval v2 (no retraining; unblocks everything).**
- Wire subject vector `label_native.npy[:,0,1]`; GroupKFold/LOSO; macro-F1+balanced-acc primary; bootstrap CI; delete open-set/group paths; add `--eval-rate` (resample_poly) + `--channel-text`. Re-score the *current* checkpoint to establish an honest baseline. Files: `val_scripts/human_activity_recognition/evaluate_tsfm.py`, `EVALUATION_PROTOCOL.md`.
- ⚑ **Decision:** confirm the A/B/C metric set and the parity-vs-capability-Δ table structure.

**Milestone 1 — Tokenizer (PHz-FB) + PE.**
- Implement `PhysicalFilterbankTokenizer` (Arm A); delete interpolation + CNN/spectral zoo; pass rate through; RoPE over physical-time offsets. Unit tests: closed-form 2 Hz sinusoid identical `E_k` at 20/50/100; offline↔incremental token match `1e-5`.
- ⚑ **Decision (flavor):** ship **Arm A fixed** as default; schedule B/C as ablations only. Do not block on B/C.

**Milestone 2 — Gravity canonicalization + robustness aug.**
- Mahony/complementary filter preprocessing; gravity as described channel; re-enable rotation aug as **yaw-only**; modality dropout + corruption aug. Depends on M1 (tokenizer consumes canonicalized, possibly gappy input). Files: `datasets/imu_pretraining_dataset/augmentations.py`, `semantic_alignment_train.py:377`.

**Milestone 3 — Dense objective + causal-capable encoder.**
- Frame-label derivation in converters/loader (native per-timestamp datasets) + stitched-session aug (segment-only datasets); per-frame `encode`; `L_sm`/`L_bd`; boundary head; `QUEUE_MODE=semantic`; per-micro-batch temporal-mask sampler. Depends on M1 (per-patch token stream) + M2 (clean input).
- ⚑ **Decision (encoder):** adopt **train-both / eval-both** (recommended). Fallback = streaming-first causal-windowed with wider-window offline eval.

**Milestone 4 — Retrain `small_deep` end-to-end.**
- One run with M1–M3. Evaluate with Milestone-0 harness: ZS-XD macro-F1 on the 6 test datasets (flat tier, no severe-OOD category — VTT-ConIoT dropped, HARTH promoted to the main set; decided 2026-07-02); rate-flatness; frame-F1/boundary; ablation grid.

**Milestone 5 — Freeze + export + app.**
- Precompute label matrix + channel-text constants; `forward_export`; Core ML / ONNX(-web) / ExecuTorch with `|Δcos|<1e-3` parity; `halo/api.py` `predict()`/`set_labels()`; `HALOStreamSession`; phone-web demo + CLI. Depends on M3 (causal weights) + frozen banks.

**Milestone 6 — Paper tables + scenarios.**
- Tables 1–5; showcase scenarios 1/2/3; model card.

---

## 5. Risks & open questions

1. **Frame-level labels don't exist for all datasets.** Only PAMAP2/Opportunity/DSADS/RealDisp/MHEALTH/HARTH/Daphnet carry per-timestamp annotation; UCI-HAR/HHAR/WISDM/KU-HAR/UniMiB/RecGym are segment-only. *De-risk:* segment-only datasets fall back to segment-label-per-patch (identical to today — strictly no regression) and drive the boundary head via **stitched-session augmentation** (concatenate two single-activity sessions with a known boundary). The dense loss reduces to the current loss when `m_i` is one-hot, so M3 cannot underperform the baseline by construction on those sets.

2. **Causal encoder may lose accuracy vs bidirectional.** *De-risk:* train-both/eval-both keeps one weight set valid under full context; report the offline↔causal-W gap explicitly (ablation 2). If the gap is large, offline segment mode still uses full context — streaming is an *added* mode, not a replacement, so headline segment numbers are protected.

3. **Fixed filterbank may underperform learned features.** *De-risk:* [Ghaffari 2024](https://consensus.app/papers/details/59964a2584355a5ba01c23cb2b451dbc/) predicts filter shape doesn't matter (norm+compression dominate); Arm B (learnable Gabor) and Arm C (AaSP) drop into the same slot via config, so if Arm A trails on macro-F1 we flip a flag rather than re-architect. Frontend-decomposition ablation quantifies this directly.

4. **Gravity estimation quality on-device / accel-only.** Mahony needs gyro; accel-only degrades posture. *De-risk:* A1b shows OS-gravity ≈ Mahony (lossless on-device path); D1 shows modality-dropout training + accel-only gravity-LP recovers most posture classes. Expose the gravity channel so the model is *told* when it's low-quality.

5. **RoPE-over-physical-time interaction with variable stride.** If `stride_sec` varies across datasets, relative offsets differ. *De-risk:* keying RoPE to seconds is the intended fix (rate-invariant), but validate that mixed-stride training doesn't destabilize; fallback = windowed-relative sinusoid keyed to `t−j∈[0,W)`.

6. **Frozen per-band norm stats estimated in one epoch may be poorly calibrated for rare rates.** *De-risk:* estimate `μ_k,s_k` pooled across all rates (band index is rate-independent); PCEN is an ablation swap if device-gain variation bites in deployment.

7. **Export of MHA / stateful KV to Core ML / ExecuTorch is fiddly.** *De-risk:* App design already verified MHA traces; keep the tokenizer a fixed matmul/FFT (traceable) and gate the release on the `|Δcos|<1e-3` parity test — if a target backend fails, ship the others (ONNX-web is enough for the browser demo).

8. **Open question ⚑ (human):** which 3 showcase scenarios and which single blessed label vocabulary ship in the default checkpoint's model card? Recommended: scenarios 1/2/3 above; label bank = union of the 6 test-set native vocabularies plus a documented open-vocab path.

**Key file paths (single source of truth for implementation):** `model/feature_extractor.py`, `model/preprocessing.py`, `model/encoder.py`, `model/positional_encoding.py`, `model/config.py`, `model/semantic_alignment.py`, `model/transformer.py`, `model/token_text_encoder.py` (ChannelTextFusion — **unchanged**), `training_scripts/human_activity_recognition/semantic_alignment_train.py`, `.../semantic_loss.py` (**core unchanged**), `datasets/imu_pretraining_dataset/multi_dataset_loader.py`, `.../augmentations.py`, `datascripts/{ds}/convert.py`, `val_scripts/human_activity_recognition/evaluate_tsfm.py`, `val_scripts/human_activity_recognition/model_loading.py`, new `halo/api.py` + `scripts/export/`, canonical checkpoint `training_output/semantic_alignment/small_deep_v2_4b3fdd6/best.pt`.