# HALO V2 — Tokenizer Design Spec

*Component design for [REDESIGN_PLAN.md](REDESIGN_PLAN.md). Where this and the integrated plan disagree, the plan's §1 interface reconciliation wins.*

---

Numerically confirmed: the physical-Hz filterbank is rate-invariant by construction (raw log-band-energy cosine 0.99+ across 20/50/100 Hz; peak-energy bands coincide exactly at 2 Hz and 5 Hz), while the current interpolate-to-64 path aliases a real 22 Hz tone down to 20.67 Hz. Longer windows tighten low-rate agreement, confirming patch-size is handled cleanly. Here is the buildable spec.

---

# HALO Replacement Tokenizer — Physical-Hz Filterbank (PHz-FB)

## 0. Recommendation up front

**Default = Arm (A): a FIXED constant-Q Gaussian filterbank in physical Hz, applied to a native-rate zero-padded rDFT, followed by log-compression + a frozen per-band standardization + one shared `Linear(→d_model)`.** One branch, ~13–25K params, zero learned filters, rate-invariant and anti-aliased *by construction*. This is directly justified by the learnable-frontend finding that the gains come from **normalization + compression, not learned filter shapes** ([Ghaffari 2024](https://consensus.app/papers/details/59964a2584355a5ba01c23cb2b451dbc/)); a fixed mel/log filterbank is competitive, so we spend no parameters or aliasing risk on learning filters.

**Ablation arm = (B): learnable-but-constrained Gabor centers/bandwidths** (softplus-positive, Nyquist-clamped), the native-domain analogue of anti-aliased/perfect-reconstruction wavelet frontends ([Michau 2021](https://consensus.app/papers/details/5e035cb6e5425a97b056e1c803d74770/), [Wave-U-Net DWT 2021](https://consensus.app/papers/details/a6ecb83c936a5363a48db16cec92f805/)). **Stretch comparison = (C) AaSP** band-limited complex kernels with explicit anti-alias subband analysis ([AaSP 2025](https://consensus.app/papers/details/519c2b54e3ee5ed99ebb9d68b231a4c7/)). Same I/O contract, selectable by config, so all three drop into the same slot.

---

## 1. Math

### 1.1 Setup (no interpolation, native rate)
A patch is `D` seconds of channel-`c` signal at native rate `r` Hz → `N = round(r·D)` real samples `x[0..N-1]`. Batch patches are zero-padded to a fixed DFT length `S` (e.g. 256; `S ≥ max N`).

1. **Window + DC removal:** `x̃[n] = w[n]·(x[n] − x̄)`, `w = Hann(N)`, zeros for `n ≥ N`. Hann controls leakage; DC removal drops the raw-offset bin.
2. **Native-rate zero-padded rDFT:** `X[m] = Σ_n x̃[n] e^{−j2π m n / S}`, `m = 0..M`, `M = S/2`.
   **Key identity:** because the *sample spacing is 1/r*, DFT bin `m` maps to the exact physical frequency
   `φ[m] = m · r / S  Hz`.
   Zero-padding is sinc-interpolation of the spectrum — it never changes the Hz meaning of a bin, it only needs `r` and `S` (both known). This is the entire rate-invariance mechanism.

### 1.2 Fixed physical-Hz filterbank
Center frequencies `f_1..f_K` fixed in Hz, log-spaced over the human-motion band (posture/gravity micro-motion, gait fundamentals 0.5–3 Hz, harmonics 3–8 Hz, tremor/impact transients 8–15 Hz):
`f_k = f_min·(f_max/f_min)^{(k−1)/(K−1)}`, e.g. `f_min=0.3, f_max=15, K=32`.
Constant-Q Gaussian weights (bandwidth scales with center → log resolution):
`H_k[m] = exp( −½ ((φ[m] − f_k)/σ_k)² )`, `σ_k = f_k /(2Q)`, `Q≈4`.
Band energy (physical units), **normalized by window energy** `Σ_n w[n]²` so it is a
power estimate independent of the sample count `N=r·D` (by Parseval `Σ_m|X|² ∝ Σ_n w² ∝ N`;
without this both `E` and the amplitude scalar would scale with `r·D` and silently encode
the sampling rate — caught in the M1 debug sweep):
`E_{k,c} = (Σ_m H_k[m] · |X_c[m]|²) / Σ_n w[n]²`.

Because `H_k` is a fixed function of **physical Hz** and `φ[m]` is computed from native `(r,S)`, a 2 Hz gait signal deposits energy in the same `f_k≈2 Hz` filter at 20/50/100 Hz. *Verified:* raw log-`E` cosine across rates = 0.997/0.991/0.995 (D=1.0–2.5 s); top-2 energy bands coincide exactly at 1.99 Hz and ~5 Hz for all three rates.

### 1.3 Compression + normalization (the load-bearing part)
`ẽ_{k,c} = log(1 + E_{k,c})` → per-band standardization with **frozen** buffers `μ_k, s_k` (estimated once as running mean/var over the first training epoch, then fixed):
`ê_{k,c} = (ẽ_{k,c} − μ_k)/s_k`.
Frozen per-band stats keep normalization **rate-invariant** (does not depend on which bands are observed — that was the only source of residual drift in the PoC). Optionally swap for **PCEN** (per-channel energy normalization) for extra robustness to varying device gain in real-world deployment — recommended as a robustness ablation.
**Amplitude is preserved** (fixes Bottleneck 3 in `FEATURE_EXTRACTOR_ANALYSIS.md`): append one scalar `a_c = log(1 + Σ_k E_{k,c})` (total log-energy) so running vs walking are no longer collapsed by per-patch z-scoring.

### 1.4 Anti-aliasing & Nyquist masking (principled, by construction)
- **No resampling exists** in the pipeline → no operation can alias. The native rDFT exactly represents all content ≤ `r/2` (Nyquist–Shannon); contrast the current path, which linear-interpolates to 64 with no anti-alias filter ([BlurPool/Zhang 2019](https://consensus.app/papers/details/74cb2454204e5c6191b34d32f37f05b4/)) and aliases (PoC: real 22 Hz → phantom 20.67 Hz).
- **Observability (Nyquist) mask** — a band is measurable only if it fits under native Nyquist with margin:
  `o_{k} = 1[ f_k + 2σ_k ≤ β·(r/2) ]`, `β≈0.9`.
  Unobservable bands (e.g. everything > ~9 Hz at 20 Hz) are set to the neutral value 0 and their `o_k=0` is passed to the projection. The model is *told* the band is missing rather than fed an artifact.
- **Low-frequency resolution limit** is handled softly by the Hann window (a `<1`-cycle band naturally leaks to low energy), not a hard mask — this is exactly what makes **patch-size augmentation clean**: same filterbank, the observable set just widens/narrows with `(r, D)`. Longer `D` improves low-band agreement (PoC), which is physically correct.

### 1.5 Token
`token_{p,c} = Linear( concat[ ê_{·,c} (K), o (K), a_c (1) ] → d_model )`, shared across all patches/channels. `o` is the observability mask (per patch, from `r`); giving it to the projection lets the model condition on which bands were seen. Output token dim = `d_model` (=384, MiniLM lock preserved).

---

## 2. Module spec

New class in `model/feature_extractor.py`, replacing the CNN/spectral zoo:

```python
class PhysicalFilterbankTokenizer(nn.Module):
    def __init__(self, d_model=384, n_bands=32, f_min=0.3, f_max=15.0, Q=4.0,
                 dft_size=256, nyquist_margin=0.9, learnable=False,   # learnable=True → Arm B
                 use_amplitude=True, norm='frozen'):                  # 'frozen'|'pcen'|'layernorm'
        # register_buffer('centers', log_spaced(f_min,f_max,n_bands))   # Arm A: fixed
        # if learnable: self.centers = softplus params, clamped ≤ nyquist  # Arm B
        # register_buffer('mu'), register_buffer('sd')  # frozen per-band stats
        in_dim = n_bands + n_bands + (1 if use_amplitude else 0)
        self.proj = nn.Linear(in_dim, d_model)

    def forward(self, patches, sampling_rate_hz, patch_len_samples=None):
        # patches: (B, P, S, C) native-rate, zero-padded to S; sampling_rate_hz: (B,) or scalar
        # patch_len_samples: (B,) true N per sample for Hann windowing (defaults to S)
        # returns: tokens (B, P, C, d_model)   ← identical contract to old extractors
```

Implementation notes: build one `(num_unique_rates, K, M)` filterbank tensor per forward (few distinct rates: 20/25/30/50/64/100), index per sample; single batched `torch.fft.rfft(n=S)`; everything vectorized, `torch.compile`-friendly (fixed `S`). `get_output_dim()` returns `d_model`.

**Param count (d_model=384, K=32):** `Linear(65→384)` = **25,344** (Arm A). Leaner variant (impute mask instead of concatenating it, `Linear(33→384)`) = **12,672**. Arm B adds ~3·K≈96 learnable scalars. Filterbank + DFT + norm buffers are **0 trainable params**. **vs current `SpectralTemporalExtractor` ≈ 55K–100K** (2 conv layers + GroupNorms + `Linear 64→288` + spectral MLP `33→192→96` + two LayerNorms) → **2–4× fewer params, one branch instead of two, no interpolation.**

**Exact deletions:**
- `model/preprocessing.py`: **delete `interpolate_patches()`** entirely; **delete the interpolation step** and the `target_patch_size`/`interpolation_method` args from `preprocess_imu_data()`; **remove the default per-patch `normalize_patches()` z-score** from the main path (amplitude now preserved in the tokenizer; keep the function only if MAE recon still needs `means/stds`). `create_patches()` **stays** (still emits native patches, now un-interpolated, zero-padded to `S`).
- `model/feature_extractor.py`: **delete `MultiScaleConv1D`, `ChannelIndependentCNN`, `FixedPatchCNN`, `SpectralTemporalExtractor`** → replace with `PhysicalFilterbankTokenizer`.
- `model/encoder.py`: **delete the `assert seq_len == self.target_patch_size`** (line 236); **delete the `feature_extractor_type` cnn/spectral_temporal branch** (lines 124–141) → single tokenizer; pass `sampling_rate_hz`/`patch_len` into `forward`; drop `target_patch_size`, `interpolation_method`, `spectral_ratio`, `cnn_channels`, `cnn_kernel_sizes` plumbing.
- `model/config.py`: **remove** `target_patch_size`, `interpolation_method`, `feature_extractor_type`, `spectral_ratio`, `cnn_channels`, `cnn_kernel_sizes` from all tiers; **add** `n_bands`, `f_min`, `f_max`, `Q`, `dft_size`, `tokenizer_learnable`, `tokenizer_norm`.

---

## 3. Channel handling (unchanged semantics)
Filterbank is applied **identically and independently per channel** → channel-independence preserved ([Han TKDE'23](https://consensus.app/papers/details/6739ef37d4ab5af28979718e661fe3dd/), [PatchTST](https://consensus.app/papers/details/7425f108b08556ce919a01fa9d1376ac/)). Output is still `(B, P, C, d_model)`, so:
- **Absent/padded channels:** the existing `channel_mask` and `pad_token`/`mask_token` machinery in `encoder.forward` (lines 245–263) is **untouched** — masking still happens at the token level after the tokenizer.
- **Language channel-encoding is fully preserved:** `IMUPositionalEncoding.ChannelSemanticEncoding` and `ChannelTextFusion` consume `(B,P,C,d_model)` tokens exactly as today — no change. Nice synergy: the channel text already encodes `"…(sampled at {r}Hz, {D}s window)"`, which now *matches* the tokenizer's explicit observability mask — the semantics are consistent instead of the text asserting a rate the tokenizer had interpolated away ([GOAT IMWUT'24](https://consensus.app/papers/details/ed52d465eb1d5e1f8a35fb611d3ba628/)).

---

## 4. Interface for encoder + unified objective
- **Output:** `(B, P, C, d_model)` — drop-in; the transformer, positional encoding, `ChannelTextFusion`, and `SemanticAlignmentHead` (incl. `per_patch_prediction`) are unchanged.
- **Patch count is already variable** and stays variable: `TemporalPositionalEncoding` slices `pe[:num_patches]`, so nothing changes. **Recommended enhancement for the physical story + streaming:** index temporal PE by **physical time** (`patch_start_seconds` = `p·stride_sec`) instead of integer patch index, so a 5 s activity gets the same temporal code at 20 Hz and 100 Hz. Trivial change in `TemporalPositionalEncoding.forward` (pass `positions` in seconds).
- **Streaming-ready by construction:** each patch's token depends only on that patch's native samples (no cross-patch normalization, no global interpolation), so the per-patch token stream is *bit-identical* whether computed offline on a whole recording or incrementally online. This is exactly the substrate a P2LHAP-style patch→label unified objective needs ([P2LHAP 2024](https://consensus.app/papers/details/ca93f18a1445527fb380739fba835277/)); segment classification = pool patch tokens, live recognition = emit per-patch. Feed rate metadata as `sampling_rate_hz` (already plumbed as `sampling_rates`/`patch_sizes` in `forward_from_raw`).

---

## 5. Ablations / validation
1. **Rate-invariance (headline):** train at native rates; take a zero-shot dataset (e.g. MotionSense 50 Hz), *properly* anti-alias-resample to 20/100 Hz, evaluate. Metrics: token cosine (target >0.98; PoC already 0.99+) and **macro-F1 drift ≤ 2 pts** vs current pipeline's larger drift. Include the closed-form sinusoid test (pure 2 Hz at 20/50/100 → identical `E_k`).
2. **Anti-alias benefit:** inject an above-Nyquist tone at a low rate; show current interp path creates a phantom low-freq feature while PHz-FB masks it; measure downstream robustness gain on a high-rate dataset downsampled without an anti-alias filter ([Zhang 2019](https://consensus.app/papers/details/74cb2454204e5c6191b34d32f37f05b4/)).
3. **Fewer-parts, no-accuracy-loss:** retrain `small_deep` with PHz-FB; compare zero-shot **macro-F1** (primary, per [Ghosh 2026](https://consensus.app/papers/details/5e6c92a6f65a5f029618ecf827a0f54c/)) on the 7 unseen datasets. Target: within noise (≤1 pt) or better at 2–4× fewer tokenizer params, interpolation and dual-LayerNorm removed.
4. **Frontend decomposition (reproduce [Ghaffari 2024](https://consensus.app/papers/details/59964a2584355a5ba01c23cb2b451dbc/)):** ablate log-compression on/off, norm frozen/PCEN/off, and Arm A (fixed) vs Arm B (learnable centers) vs Arm C (AaSP). Expect: norm+compression dominate; learned filters ≈ fixed → validates the default.
5. **Patch-size robustness:** vary `D` under patch-size augmentation; verify low-band agreement improves with `D` (PoC trend) and macro-F1 is stable — confirms clean patch-size handling with no interpolation ceiling.
6. **Streaming equivalence check:** assert offline vs incremental per-patch tokens match to `1e-5`.

---

## Data digest for integration
- **Sampling rates in corpus:** 20 (WISDM/RecGym), 25 (DSADS), 30 (Opportunity), 50 (UCI/HHAR/MHEALTH/UniMiB/MobiAct/RealWorld/HAPT/VTT/MotionSense/RealDisp/Shoaib/HARTH), 64 (Daphnet), 100 (PAMAP2/KUHAR/USC-HAD) Hz → `f_max=15` keeps all bands observable at ≥30 Hz; 20 Hz observes ≤~9 Hz (masked above). Set `dft_size S=256` (`≥ max N`: 100 Hz·2.5 s=250).
- **Files to edit:** `model/feature_extractor.py` (replace classes), `model/preprocessing.py` (delete interpolation + default z-score), `model/encoder.py` (delete assert + dual-branch + interpolation plumbing; pass rate), `model/config.py` (swap CNN/spectral keys for filterbank keys), and the tokenizer call sites in `training_scripts/human_activity_recognition/semantic_alignment_train.py` (`_preprocess_raw_batch`/`forward_from_raw` already carry `sampling_rates`/`patch_sizes` — stop interpolating to `TARGET_PATCH_SIZE`, zero-pad to `S`, pass `sampling_rate_hz`).
- **Untouched (preserve):** `token_text_encoder.py` (`ChannelTextFusion`, label banks), `positional_encoding.py` channel-semantic path, `transformer.py`, `semantic_alignment.py`, `semantic_loss.py`.
- **Config keys added:** `{n_bands:32, f_min:0.3, f_max:15.0, Q:4.0, dft_size:256, nyquist_margin:0.9, tokenizer_learnable:false, tokenizer_norm:"frozen", use_amplitude:true}`.
- **PoC (validated numbers):** rate-invariance cosine 0.997/0.991/0.995 across 20/50/100 Hz; exact peak-band coincidence at 2 Hz & 5 Hz; aliasing contrast 22 Hz→20.67 Hz phantom under current interp-to-64.

---

## Implementation status (M1 build + debug sweep)

**Built + verified.** `model/feature_extractor.py::PhysicalFilterbankTokenizer` (Arm A default,
Arm B/`learnable` selectable), wired as a selectable `feature_extractor_type='physical_filterbank'`
through `preprocessing.py` (`zero_pad_patches` + `pad_to_size` mode), `encoder.py` (rate/N threaded
into `forward`/`encode_from_raw`), and `config.py` (`small_deep_fb`). `tests/test_physical_filterbank_tokenizer.py`
= 20 passing tests (rate-invariance, streaming==offline, masks, guardrails, calibration, Arm B). No
regressions in the existing model suite. Current checkpoint/eval path untouched.

**Debug sweep (6 adversarial lenses) — findings resolved:**
1. *(CONFIRMED, fixed)* Band energy `E` and the amplitude scalar scaled with `N=r·D` (unnormalized
   rDFT) → now divided by window energy `Σw²` (§1.2). Regression tests assert amp is rate- and
   duration-invariant for a fixed physical tone.
2. *(fixed)* `encoder.forward` now **requires** `patch_len_samples` in filterbank mode (was silently
   defaulting to `N=S`, corrupting Hann/DC for padded patches) — loud error, symmetric with the rate guard.
3. *(fixed)* Arm B learnable centers use a smooth `sigmoid`→`(f_min,f_max)` map instead of a hard
   `clamp` (which froze the top band from step 0 and could produce `0·inf=NaN`).
4. *(fixed)* Calibration (`accumulate_norm_stats`) now applies the Nyquist mask per band, so high
   bands' frozen stats are not dragged toward 0 by low-rate samples; unseen bands fall back to identity.
5. *(noted for M4)* Tokenizer hyperparameters are not yet threaded through the checkpoint save/load
   chain (`get_config()` added on the tokenizer; see M4 checklist). Inert today (config == defaults).

**Deferred to the M4 retrain cutover (co-designed with an actual training run):**
- Training-loop path: thread `sampling_rate_hz` + `patch_len_samples` through
  `SemanticAlignmentModel.forward`/`_preprocess_raw_batch`, and have the DataLoader emit patches
  zero-padded to `S` (not interpolated to `TARGET_PATCH_SIZE`). Until then the training path fails
  *loudly* under `small_deep_fb` (verified: missing-rate ValueError + `seq_len==dft_size` assert).
- Chain-of-custody for tokenizer hyperparams: persist `n_bands/f_min/f_max/tokenizer_Q/dft_size/
  nyquist_margin/tokenizer_learnable/tokenizer_norm/use_amplitude/use_resolution_mask` in the saved
  `encoder` config block; read + pass them in `model_loading.load_model`; add a reload-equality guard.
- Run `fit_norm_stats` over the **augmented** `(r,D)` distribution during the first training epoch,
  then freeze.
- **Strip sampling rate + window duration from the channel-description text** (multi_dataset_loader.py):
  keep placement + gravity only. Rate/duration are conveyed by the tokenizer + the `o_k`/`res_k` masks;
  the frozen-SBERT text can't do numeracy and a raw rate scalar is a dataset fingerprint (leakage risk).
  Decision recorded in `research_conditioning.md` §7 — masks only, no rate/duration embedding. Must land
  *with* the tokenizer swap, not before (the current CNN model relies on the text as its only rate signal).
- The "exact deletions" above (remove CNN/spectral classes + interpolation) happen at cutover.

**References:** [AaSP 2025](https://consensus.app/papers/details/519c2b54e3ee5ed99ebb9d68b231a4c7/), [BlurPool/Zhang 2019](https://consensus.app/papers/details/74cb2454204e5c6191b34d32f37f05b4/), [Michau 2021](https://consensus.app/papers/details/5e035cb6e5425a97b056e1c803d74770/), [Wave-U-Net DWT 2021](https://consensus.app/papers/details/a6ecb83c936a5363a48db16cec92f805/), [Ghaffari 2024 (learnable frontends)](https://consensus.app/papers/details/59964a2584355a5ba01c23cb2b451dbc/), [Koneripalli 2020 (rate-invariant AE)](https://consensus.app/papers/details/92264aa39b645e5999bff8f27a41169f/), [Han TKDE'23 (CI)](https://consensus.app/papers/details/6739ef37d4ab5af28979718e661fe3dd/), [PatchTST](https://consensus.app/papers/details/7425f108b08556ce919a01fa9d1376ac/), [P2LHAP 2024](https://consensus.app/papers/details/ca93f18a1445527fb380739fba835277/), [GOAT IMWUT'24](https://consensus.app/papers/details/ed52d465eb1d5e1f8a35fb611d3ba628/), [Ghosh 2026 (macro-F1)](https://consensus.app/papers/details/5e6c92a6f65a5f029618ecf827a0f54c/).