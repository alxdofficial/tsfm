# Feature Extractor Bottleneck Analysis

Analysis of whether the conv-based feature extractor (or other components) limits TSFM's performance, with concrete alternatives.

Date: 2026-02-18

---

## Current Pipeline Summary

```
Raw sensor data  (B, T, C)
       │
       ▼
[Preprocessing]  preprocessing.py
  create_patches()        → (P, patch_T, C)     non-overlapping windows
  interpolate_patches()   → (P, 64, C)          resample to fixed 64 steps
  normalize_patches()     → (P, 64, C)          z-score per-patch per-channel
       │
       ▼
[Feature Extraction]  feature_extractor.py
  ChannelIndependentCNN
  Reshape to (B*P*C, 1, 64)                     every (patch, channel) pair independent
  → MultiScaleConv1D: Conv1d(1→32, k=5) + GroupNorm + GELU
  → MultiScaleConv1D: Conv1d(32→64, k=5) + GroupNorm + GELU
  → AdaptiveAvgPool1d(1)                         collapse 64 timesteps → 1 value
  → Linear(64→384) + GELU + Dropout
       │
       ▼
(B, P, C, 384)
       │
       ▼
[Positional Encoding]  positional_encoding.py
  + TemporalPositionalEncoding     sinusoidal PE along patch dim, learnable scale
  + ChannelSemanticEncoding        frozen SentenceBERT + 2-layer MLP projection
       │
       ▼
(B, P, C, 384)  →  Transformer
```

**Key numbers:**
- CNN params: ~36K (shared across all channels and patches)
- Conv receptive field: 9 timesteps (kernel 5, 2 layers)
- d_model locked to 384 (must match MiniLM-L6-v2 output dim)
- Default config: `cnn_channels=[32, 64]`, `cnn_kernel_sizes=[5]`, `num_scales=1`

---

## Bottlenecks Found

### Bottleneck 1: AdaptiveAvgPool1d(1) destroys within-patch temporal structure (HIGH)

The CNN produces `(B*P*C, 64, 64)` — 64 feature maps across 64 timesteps. Global average
pooling collapses this to `(B*P*C, 64)` — a single mean value per feature map. All temporal
ordering within a patch is lost.

The model cannot distinguish "impact at start of patch" from "impact at end of patch." The
CNN acts as a bag-of-local-features extractor — it detects *whether* a pattern occurs, not
*when* within the patch. The Transformer handles cross-patch ordering, but within-patch
dynamics are gone.

This matters for HAR because:
- Gait cycles have asymmetric phases (heel strike vs toe-off) at specific sub-patch positions
- Fall detection relies on the *sequence* of impact → freefall → stillness within one patch
- The 9-timestep conv receptive field captures local motifs, but pooling discards their arrangement

### Bottleneck 2: Interpolation to fixed 64 timesteps is lossy (MEDIUM)

All patches are resampled to exactly 64 timesteps via `F.interpolate(mode='linear')`. For
50Hz/1.0s patches (50 native samples), this upsamples to 64 — mild distortion. For higher
rates or larger patches, information loss is significant. The fixed 64-step budget was chosen
for CNN input-size invariance, but it's a hard ceiling on temporal resolution.

Also: the "cubic" interpolation option in preprocessing.py silently falls back to linear
(F.interpolate 1D has no cubic mode). This is a documentation bug.

### Bottleneck 3: Per-patch z-score normalization removes amplitude information (LOW-MEDIUM)

Each patch is independently z-scored per channel. A patch where the accelerometer reads
15 m/s² (running) looks identical to one at 5 m/s² (gentle walking) if their shapes are
similar. Cross-patch amplitude differences — highly informative for activity discrimination
— are erased. The `stds` are computed but only used for MAE reconstruction, never fed to
the encoder.

---

## What's NOT a Bottleneck

- **Channel-independent processing** — The TKDE 2024 study (Han et al., "The Capacity and
  Robustness Trade-off") confirms CI trades capacity for robustness, and the optional
  cross-channel attention in the Transformer compensates. This is fine.
- **CNN vs linear projection** — The CNN already outperforms PatchTST-style linear projections
  (confirmed by Dartmouth tokenization study, 2024). The CNN is the right class of tokenizer.
- **The 36K parameter count** — The CNN is intentionally lightweight (shared across all
  channels/patches). The Transformer has the capacity. The tokenizer just needs good features.
- **SentenceBERT channel encoding** — The residual-init MLP projection is well-designed.
  The d_model=384 lock to MiniLM-L6-v2 is a constraint but not a performance bottleneck.

---

## Recommended Alternatives

### Option A: Replace AvgPool with Strided Depthwise Conv (LOW effort, HIGH impact)

Instead of `AdaptiveAvgPool1d(1)`, use a small depthwise conv with stride to preserve
temporal structure:

```
Current:  Conv→Conv→AvgPool→Linear(64→384)       = 1 temporal slot per token
Proposed: Conv→Conv→DepthwiseConv(stride=4)→Flatten→Linear(1024→384) = 16 temporal slots
```

A stride-4 depthwise conv on 64-timestep feature maps produces 16 positions × 64 channels
= 1024 features, then a linear projects to 384. Preserves **16 temporal slots** instead of
collapsing to 1.

| Aspect | Details |
|--------|---------|
| Compute | ~1.2x current CNN cost |
| Memory | Negligible increase (linear 1024→384 vs 64→384) |
| Compatibility | Drop-in — output still `(B, P, C, d_model)` |
| Risk | Low. Model can learn to ignore temporal positions if unhelpful. |

### Option B: Add FFT Amplitude Features (LOW effort, MEDIUM impact)

Concatenate FFT amplitude spectrum to the CNN features before projection:

```
Current:  CNN(patch) → AvgPool → [64-dim] → Linear(64→384)
Proposed: CNN(patch) → AvgPool → [64-dim] ⊕ FFT_amp(patch)[32-dim] → Linear(96→384)
```

Compute FFT of each 64-timestep patch, take first 32 amplitude bins (up to Nyquist),
concatenate with CNN features. The linear projection absorbs both.

| Aspect | Details |
|--------|---------|
| Compute | FFT is O(n log n) = negligible. Linear goes 96→384 instead of 64→384. |
| Value | Captures periodic structure (walking ~2Hz, running ~3Hz, gait harmonics). rTsfNet showed FFT amplitude ratio is among the most discriminative features for HAR. |
| Compatibility | Drop-in — just changes projection input dim. |
| Risk | Very low. |

### Option C: Wavelet Tokenization Replacing CNN (MEDIUM effort, HIGH potential)

Replace the CNN entirely with a Discrete Wavelet Transform (DWT):

```
Current:  patch(64 steps) → Conv1→Conv2→AvgPool→Linear(64→384)
Proposed: patch(native steps) → 3-level DWT → [approx ⊕ details] → Linear(64→384)
```

A 3-level Haar DWT on 64 timesteps produces:
- Level 1 detail: 32 coefficients (12.5-25 Hz — high-frequency transients, impacts)
- Level 2 detail: 16 coefficients (6.25-12.5 Hz — hand tremor, gait harmonics)
- Level 3 detail: 8 coefficients (3.125-6.25 Hz — step frequency, running cadence)
- Level 3 approximation: 8 coefficients (0-3.125 Hz — posture, gravity)
- Total: 64 coefficients → Linear → 384

| Aspect | Details |
|--------|---------|
| Compute | DWT is O(n), strictly cheaper than 2-layer CNN |
| Memory | Less than CNN (DWT is parameter-free, only projection is learned) |
| Value | Natural multi-resolution decomposition. Each coefficient band captures a different physical phenomenon. The CNN with kernel=5 has no principled frequency separation. |
| Variable rate | DWT works at native rate without interpolation — just decompose more/fewer levels based on sampling rate. Could **eliminate the interpolation step entirely**. |
| Risk | Moderate. Haar wavelets are simple; learned wavelets (WaveToken-style) would be more powerful but more complex. |

**Key references:**
- WaveToken (ICML 2025): Wavelet-based tokenization outperforms patch-based on 42 datasets
- LMWT (arXiv Apr 2025): Learnable multi-scale Haar wavelet with O(n) complexity

### Option D: Physics-Aware Triad Rotation (LOW effort, MEDIUM impact)

Add a learnable 3×3 rotation matrix for each sensor triad before the channel-independent CNN:

```
Current:  each of 6 channels processed independently
Proposed: acc_xyz → Learnable3DRotation(3×3) → 3 rotated channels → CNN
          gyro_xyz → Learnable3DRotation(3×3) → 3 rotated channels → CNN
```

Inspired by rTsfNet (IMWUT 2024) which achieved 97.76% on UCI HAR using rotation-aware
triaxial processing.

| Aspect | Details |
|--------|---------|
| Compute | One 3×3 matmul per triad per patch — negligible |
| Memory | 18 parameters total (9 per triad) |
| Value | Learns gravity alignment automatically (instead of hardcoded Butterworth filter). Makes per-channel features more consistent across device orientations. |
| Risk | Low. Orthogonal to other improvements. |

---

## Implementation Priority

| Option | Effort | Impact | Risk | Recommendation |
|--------|--------|--------|------|----------------|
| **A: Replace AvgPool** | Low | High | Low | **Do first** — directly fixes the biggest bottleneck |
| **B: Add FFT features** | Low | Medium | Very low | **Do second** — easy win, complementary to A |
| **D: Triad rotation** | Low | Medium | Low | **Do alongside A** — orthogonal improvement |
| **C: Wavelet tokenizer** | Medium | High | Medium | **Explore after A+B** — could replace CNN entirely and solve variable-rate handling |

Options A and B are drop-in changes to `feature_extractor.py` with no architectural side
effects. Option D is orthogonal and can be done in parallel. Option C is a bigger redesign
but potentially transformative — it could also eliminate the interpolation step by working
at native sampling rates directly.

---

## Literature References

- PatchTST (Nie et al., ICLR 2023): Established patch-based tokenization paradigm
- MOMENT (Goswami et al., ICML 2024): Linear patch embedding for time-series foundation model
- Dartmouth Tokenization Study (Asher, 2024): CNN embeddings outperform linear for most datasets
- WaveToken (ICML 2025): Wavelet-based tokenization outperforms patch-based methods
- LMWT (arXiv Apr 2025): Learnable multi-scale Haar wavelet Transformer, O(n) complexity
- TOTEM (TMLR 2024): VQ-VAE for general time series tokenization
- rTsfNet (IMWUT 2024): Physics-aware triaxial processing, 97.76% on UCI HAR
- Han et al. (TKDE 2024): Channel-independent vs dependent — capacity/robustness tradeoff
- iTransformer (ICLR 2024): Variate tokens with cross-channel attention
- BSAT (arXiv Jan 2025): B-spline adaptive tokenizer for variable-rate data
- FreEformer (2025): Frequency-domain variate tokens
- BPE for Time Series (arXiv May 2025): Byte pair encoding for time series compression
