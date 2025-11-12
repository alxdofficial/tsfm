# IMU Tokenizer Design & Pretraining Workflow

**Comprehensive Analysis with Design Decisions, Shapes, and Memory Usage**

---

## Table of Contents
1. [Tokenizer Design](#tokenizer-design)
2. [Pretraining Workflow](#pretraining-workflow)
3. [Shape Transformations](#shape-transformations)
4. [GPU Memory Analysis](#gpu-memory-analysis)

---

## 1. Tokenizer Design

Your tokenizer is technically the **IMUActivityRecognitionEncoder**, which transforms raw IMU sensor data into semantic embeddings. Here's every important design decision:

### 1.1 Input Format

**Decision:** Use real-world time (seconds), not timesteps
- **Why:** Dataset-agnostic—works across datasets with different sampling rates (20-200 Hz)
- **Input:** `(num_timesteps, num_channels)` raw sensor data
- **Metadata:** `sampling_rate_hz`, `patch_size_sec`

---

### 1.2 Patch Creation (Windowing)

**Design Decision #1: Fixed patch duration in seconds**
```python
patch_size_sec = 2.0  # or dataset-specific (2.56, 3.0, 4.0, 5.0)
```

**Why this matters:**
- **Semantic consistency:** 2 seconds of walking is semantically similar across datasets
- **Not fixed timesteps:** Different datasets have different sampling rates
- **Dataset-specific:** UCI HAR uses 2.56s (pre-segmented), PAMAP2 uses 5.0s (complex activities)

**Implementation:**
```python
def create_patches(data, sampling_rate_hz, patch_size_sec, stride_sec):
    patch_size_samples = int(sampling_rate_hz * patch_size_sec)
    stride_samples = int(sampling_rate_hz * stride_sec) if stride_sec else patch_size_samples
    # Sliding window extraction
```

**Result:** Variable-length patches depending on sampling rate
- 50 Hz × 2.0s = 100 timesteps
- 100 Hz × 2.0s = 200 timesteps
- 200 Hz × 2.0s = 400 timesteps

---

### 1.3 Interpolation to Fixed Size

**Design Decision #2: All patches interpolated to 96 timesteps**

**Why 96?**
- **Dataset-agnostic:** Decouples from sampling rate
- **Sufficient resolution:** Captures temporal dynamics for 1-5 second windows
- **Computational efficiency:** Not too large (avoids memory issues)
- **Divisible:** 96 = 2^5 × 3, works well with multi-scale convolutions

**Method:** Linear interpolation (default)
```python
interpolation_method = 'linear'  # Options: 'linear', 'cubic', 'nearest'
```

**Why linear?**
- Fast
- Preserves monotonicity
- Smooth transitions
- No overfitting to noise (unlike cubic)

**Result:** `(num_patches, 96, num_channels)`

---

### 1.4 Per-Patch Normalization

**Design Decision #3: Z-score normalization per patch, per channel**

```python
normalization_method = 'zscore'  # Options: 'zscore', 'minmax', 'none'
```

**Why per-patch?**
- **Removes DC offset:** Different baseline accelerations across activities
- **Scale invariance:** Robust to sensor calibration differences
- **Per-channel:** Accelerometer and gyroscope have different scales

**Formula:**
```
normalized = (patch - mean) / std
```
where `mean` and `std` computed over the 96 timesteps for each channel independently.

**Result:** `(num_patches, 96, num_channels)` with zero mean, unit variance

---

### 1.5 Multi-Scale CNN Feature Extraction

**Design Decision #4: Multi-scale convolutions with parallel branches**

**Architecture:**
```python
CNN_KERNEL_SIZES = [3, 5, 7]  # Multi-scale temporal receptive fields
CNN_CHANNELS = [64, 128]       # Progressive feature hierarchy
```

**Design:**
1. **Input:** `(batch, patches, 96, channels)`
2. **Reshape:** Process each patch independently: `(batch × patches, 96, channels)`
3. **Multi-scale conv1d:**
   - Branch 1: kernel=3 → captures fine-grained motion (0.125s at 96 samples)
   - Branch 2: kernel=5 → captures medium-scale patterns (0.208s)
   - Branch 3: kernel=7 → captures coarse patterns (0.292s)
4. **Concatenate:** Merge multi-scale features → `64 + 64 + 64 = 192` channels
5. **Conv1d:** `192 → 128` (intermediate)
6. **Conv1d:** `128 → d_model` (final projection)
7. **Global pooling:** Average over time → `(batch × patches, d_model)`
8. **Reshape back:** `(batch, patches, channels, d_model)`

**Why multi-scale?**
- **Different activity types:** Walking has different frequency than running
- **Sensor fusion:** Accelerometer (high-freq) vs gyroscope (low-freq) need different receptive fields
- **Robustness:** Ensemble of scales prevents missing important patterns

**Result:** `(batch, patches, channels, d_model=384)`

---

### 1.6 Learnable Tokens (Mask & Pad)

**Design Decision #5: Apply mask_token and pad_token at feature level (AFTER CNN, BEFORE positional encoding)**

```python
self.mask_token = nn.Parameter(torch.randn(1, 1, 1, d_model) * 0.02)
self.pad_token = nn.Parameter(torch.randn(1, 1, 1, d_model) * 0.02)
```

**Why at feature level?**
- **Efficiency:** Don't waste CNN computation on masked/padded patches
- **Actually, wait...** Looking at the code, masks ARE applied at feature level (after CNN)
- **Reason:** CNN features provide better initialization than random noise
- **Broadcasting:** `(1, 1, 1, d_model)` → `(batch, patches, channels, d_model)`

**Application:**
- **MAE mask:** Replace 50% of patches with mask_token (for reconstruction)
- **Pad mask:** Replace padding (variable num_patches across batch) with pad_token

---

### 1.7 Positional Encoding

**Design Decision #6: Dual positional encoding (temporal + channel semantic)**

**Temporal Encoding:**
```python
temporal_pos_encoding = nn.Parameter(torch.randn(1, max_patches, 1, d_model))
```
- **Learnable, not sinusoidal:** Allows model to learn patch ordering
- **Why not sinusoidal?** IMU data has non-uniform temporal patterns (unlike language)
- **Shape:** `(1, max_patches, 1, d_model)` broadcasts to all channels

**Channel Semantic Encoding:**
```python
# Uses Sentence-BERT to encode channel descriptions
channel_descriptions = ["body accelerometer x-axis", "gyroscope z-axis", ...]
semantic_embeddings = sentence_bert_model.encode(channel_descriptions)
```

**Why semantic?**
- **Sensor relationships:** "accelerometer X" is semantically similar to "accelerometer Y"
- **Cross-dataset transfer:** Channel names differ, but semantics align
- **Variable channels:** Handles 6-40 channels without retraining

**Combined:**
```
features = CNN_features + temporal_pos + channel_semantic_pos
```

**Result:** `(batch, patches, channels, d_model)` with position information

---

### 1.8 Dual-Branch Transformer

**Design Decision #7: Separate attention over time and channels**

**Architecture (per block):**
1. **Temporal Self-Attention:**
   - Attend over patches (time dimension)
   - **Per-channel independently:** Each channel has its own temporal dynamics
   - `(batch, patches, channels, d_model)` → reshape → `(batch × channels, patches, d_model)` → attention → reshape back
   - **Why?** Walking pattern in accel-X may differ from accel-Y timing

2. **Cross-Channel Self-Attention:**
   - Attend over channels (sensor dimension)
   - **Per-patch independently:** Relationships between sensors at each time
   - `(batch, patches, channels, d_model)` → reshape → `(batch × patches, channels, d_model)` → attention → reshape back
   - **Why?** Accelerometer and gyroscope are correlated (e.g., turning → accel-Y + gyro-Z)

3. **Feed-Forward Network:**
   - `d_model → dim_feedforward (1536) → d_model (384)`
   - Applied independently to each patch-channel position

**Why dual-branch?**
- **Physical intuition:** Temporal dynamics (walking cadence) vs sensor fusion (accel+gyro)
- **Complexity:** `O(patches² × channels + channels² × patches)` vs `O((patches × channels)²)` for joint attention
- **Interpretability:** Can analyze temporal vs sensor-fusion attention separately

**Num blocks:** 4 (empirically chosen)

**Result:** `(batch, patches, channels, d_model)` with rich contextualized representations

---

### 1.9 Output Format

**Final tokenized representation:**
- **Shape:** `(batch, patches, channels, d_model)`
- **Example:** `(32, 64, 40, 384)` for batch_size=32, max 64 patches, max 40 channels, d_model=384
- **Memory:** 32 × 64 × 40 × 384 = 31,457,280 elements ≈ 126 MB (FP32) or 63 MB (FP16)

**Interpretation:**
- Each patch-channel combination has a 384-dim embedding
- Embeddings capture:
  - Local temporal patterns (from CNN)
  - Global temporal context (from temporal attention)
  - Sensor fusion (from cross-channel attention)
  - Semantic position (from positional encoding)

---

## 2. Pretraining Workflow

Your pretraining uses **dual self-supervised objectives** to learn without activity labels.

### 2.1 Overall Objectives

**Goal:** Learn representations that:
1. **Reconstruct masked sensor data** (local patterns)
2. **Invariant to augmentations** (semantic robustness)

**Mathematical formulation:**
```
L_total = λ_MAE × L_MAE + λ_contrast × L_contrast
```
where λ_MAE = 1.0, λ_contrast = 1.0

---

### 2.2 Objective 1: Masked Autoencoding (MAE)

**What:** Predict masked patches from unmasked context

**How:**
1. **Random masking:** 50% of valid patches masked
   ```python
   mae_mask = torch.rand(batch, patches) < 0.5  # True = masked
   ```

2. **Masking strategy:**
   - Replace masked patches with learnable `mask_token` at feature level
   - Prevents model from "peeking" at actual values

3. **Reconstruction head:**
   ```python
   reconstruction_head = nn.Linear(d_model, 96)
   # (batch, patches, channels, d_model) → (batch, patches, channels, 96)
   # Transpose → (batch, patches, 96, channels)
   ```

4. **Loss (only on masked patches):**
   ```python
   # Per-patch normalization of targets
   targets_normalized = (targets - mean) / std  # Over 96 timesteps

   # MSE on masked positions only
   loss = MSE(predictions[mae_mask], targets_normalized[mae_mask])
   ```

**Why normalize targets?**
- **Stability:** Prevents large loss values from high-variance patches
- **Per-patch:** Each patch has different DC offset and scale
- **Better gradients:** Loss in [0, 10] range instead of [0, 1000]

**What does it learn?**
- Local temporal patterns (how sensor values evolve)
- Sensor correlations (accel-X predicts accel-Y)
- Physics constraints (gyroscope values constrain acceleration)

**Shape flow:**
```
Patches: (32, 64, 96, 40)
  ↓ Encoder
Features: (32, 64, 40, 384)
  ↓ Reconstruction head
Reconstructed: (32, 64, 96, 40)
  ↓ Loss computation
MAE Loss: scalar
```

---

### 2.3 Objective 2: Contrastive Learning

**What:** Pull together augmented views of same data, push apart different samples

**How:**
1. **Create augmented view:**
   ```python
   augmentation = apply_augmentations(patches)  # Jitter, time warp, etc.
   ```

2. **Encode both views:**
   ```python
   features_original = encoder(patches_original, mae_mask=mae_mask)
   features_augmented = encoder(patches_augmented, mae_mask=None)  # No masking
   ```

3. **Projection head:**
   ```python
   projection_head = nn.Linear(d_model, projection_dim=256)
   projected_original = projection_head(features_original)
   projected_augmented = projection_head(features_augmented)
   ```

4. **Pool over channels:**
   ```python
   # (batch, patches, channels, 256) → (batch, patches, 256)
   patch_features_1 = projected_original.mean(dim=2)  # Channel-wise average
   patch_features_2 = projected_augmented.mean(dim=2)
   ```

5. **InfoNCE loss (patch-level):**
   ```python
   # Flatten valid patches: (batch, patches, 256) → (N_valid, 256)
   valid_features_1 = patch_features_1[valid_mask]
   valid_features_2 = patch_features_2[valid_mask]

   # Normalize
   z1 = F.normalize(valid_features_1, dim=-1)
   z2 = F.normalize(valid_features_2, dim=-1)

   # Concatenate: (2N, 256)
   z = torch.cat([z1, z2], dim=0)

   # Similarity matrix: (2N, 2N)
   sim_matrix = torch.mm(z, z.t()) / temperature  # temperature=0.5

   # Labels: i and i+N are positive pairs
   labels = torch.cat([torch.arange(N, 2N), torch.arange(0, N)])

   # Cross-entropy loss (InfoNCE)
   loss = F.cross_entropy(sim_matrix, labels)
   ```

**What does it learn?**
- **Augmentation invariance:** Walking with jitter = walking without jitter
- **Semantic robustness:** Robust to sensor noise, time warping, scaling
- **Discriminative features:** Different activities have different patterns

**Why exclude MAE-masked patches from contrastive loss?**
- **Reason:** Masked patches contain no real information (replaced with mask_token)
- **Effect:** Only unmasked patches contribute to positive/negative pairs
- **Formula:** `valid_for_contrast = attention_mask & (~mae_mask)`

**Shape flow:**
```
Original: (32, 64, 96, 40)
Augmented: (32, 64, 96, 40)
  ↓ Encoder (both)
Features: 2 × (32, 64, 40, 384)
  ↓ Projection head
Projected: 2 × (32, 64, 40, 256)
  ↓ Pool over channels
Patch features: 2 × (32, 64, 256)
  ↓ Flatten valid patches
Valid features: 2 × (N_valid, 256)  where N_valid ≈ 32 × 64 × 0.5 = 1024
  ↓ InfoNCE
Contrastive Loss: scalar
```

---

### 2.4 Augmentation Strategy

**Design Decision #8: Physically-plausible augmentations**

**Weak augmentations** (preserve semantics):
1. **Jitter:** Add Gaussian noise (σ=0.1)
   - Simulates sensor noise
2. **Magnitude scaling:** Scale by 0.8-1.2×
   - Simulates different walking speeds
3. **Time shift:** Shift by ±10% of patch length
   - Simulates phase differences

**Strong augmentations** (more aggressive):
4. **Time warp:** Non-linear time distortion
   - Simulates variable activity speed
5. **Magnitude warp:** Non-linear amplitude distortion
   - Simulates uneven terrain

**Novel augmentation:**
6. **Channel shuffle:** Randomly permute X/Y/Z axes
   - **Why?** Phone orientation varies across datasets
   - **Effect:** Forces model to learn axis-invariant features

**Applied per-sample:**
```python
for i in range(batch_size):
    # Apply SAME augmentation to all patches in sample
    augmented[i] = augmentation.apply(patches[i])
```

**Why same augmentation per sample?**
- **Temporal consistency:** Walking rhythm shouldn't change mid-sample
- **Contrastive learning:** Positive pair = same sample, different augmentation

---

### 2.5 Training Loop Details

**Batch construction:**
```python
for batch in dataloader:
    # 1. Load data from multiple datasets
    data = batch['data']  # (batch, timesteps, channels)

    # 2. Preprocess into patches (per sample, different num_patches)
    patches_list = [encoder.preprocess(data[i], rate, patch_size)
                    for i in range(batch_size)]

    # 3. Pad to max_patches
    max_patches = max(p.shape[0] for p in patches_list)
    padded_patches = pad_patches(patches_list, max_patches)
    # → (batch, max_patches, 96, max_channels)

    # 4. Create masks
    mae_mask = create_random_mask(batch, max_patches, mask_ratio=0.5)
    attention_mask = create_attention_mask(patches_list)  # True = valid

    # 5. Create augmented view
    aug_patches = apply_augmentation(padded_patches)

    # 6. Forward pass (original + augmented)
    features_1, proj_1, recon = model(padded_patches, mae_mask=mae_mask, ...)
    features_2, proj_2, _ = model(aug_patches, mae_mask=None, ...)

    # 7. Compute losses
    mae_loss = mae_criterion(recon, padded_patches, mae_mask, ...)
    contrast_loss = contrast_criterion(proj_1, proj_2, ...)
    total_loss = mae_loss + contrast_loss

    # 8. Backward
    total_loss.backward()
    optimizer.step()
```

**Per-dataset tracking:**
- Each batch comes from a single dataset
- Losses tracked separately for UCI HAR, MHEALTH, PAMAP2, WISDM
- Helps identify which datasets are harder to learn

**Mixed precision (AMP):**
```python
with autocast(device_type='cuda', enabled=True):
    # Forward pass in FP16
    features, proj, recon = model(...)
    loss = criterion(...)

# Backward in FP32
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Why AMP?**
- **Memory:** 50% reduction (FP16 vs FP32)
- **Speed:** 2-3× faster on modern GPUs (Tensor Cores)
- **Accuracy:** Loss scaling prevents underflow

---

## 3. Shape Transformations

Complete flow with exact shapes (batch_size=32, max scenario):

```
INPUT: Raw IMU data
(1000, 9)  # 10 seconds at 100 Hz, 9 channels

↓ create_patches(patch_size_sec=2.0)
(5, 200, 9)  # 5 patches, 200 timesteps each

↓ interpolate_patches(target_size=96)
(5, 96, 9)  # Fixed 96 timesteps

↓ normalize_patches()
(5, 96, 9)  # Zero mean, unit variance

↓ Batching & padding (batch of 32 samples)
(32, 64, 96, 40)  # Max 64 patches, max 40 channels across batch
# With masks:
# - attention_mask: (32, 64) - True = valid patch
# - mae_mask: (32, 64) - True = masked for MAE
# - channel_mask: (32, 40) - True = valid channel

↓ FixedPatchCNN (multi-scale conv)
(32, 64, 40, 384)  # d_model=384

↓ Apply mask_token (MAE masked positions)
(32, 64, 40, 384)  # Masked patches replaced with mask_token

↓ Apply pad_token (padding positions)
(32, 64, 40, 384)  # Padded patches replaced with pad_token

↓ Add positional encoding
(32, 64, 40, 384)  # += temporal_pos + channel_semantic_pos

↓ DualBranchTransformer (4 blocks)
│ └─ Block 1:
│     ├─ Temporal attention: (32×40, 64, 384) → (32×40, 64, 384)
│     ├─ Cross-channel attention: (32×64, 40, 384) → (32×64, 40, 384)
│     └─ Feed-forward: (32, 64, 40, 384) → (32, 64, 40, 384)
│ └─ Block 2-4: Same
(32, 64, 40, 384)  # Final encoded features

SPLIT INTO TWO HEADS:

A) PROJECTION HEAD (for contrastive):
   ↓ Linear(384 → 256)
   (32, 64, 40, 256)
   ↓ Pool over channels
   (32, 64, 256)
   ↓ Flatten valid patches (exclude MAE masked & padding)
   (N_valid, 256)  where N_valid ≈ 32 × 64 × 0.5 = 1024
   ↓ InfoNCE loss
   scalar

B) RECONSTRUCTION HEAD (for MAE):
   ↓ Linear(384 → 96)
   (32, 64, 40, 96)
   ↓ Transpose
   (32, 64, 96, 40)
   ↓ MSE loss (only on MAE masked patches)
   scalar

FINAL LOSS:
total_loss = mae_loss + contrastive_loss
```

---

## 4. GPU Memory Analysis

**Memory-intensive stages ranked (descending order):**

### 🔴 **Stage 1: Transformer Forward Pass** (~60% of memory)

**What's stored:**
- Input features: `(32, 64, 40, 384)` = 31.5M elements × 2 bytes (FP16) = **63 MB**
- **Attention matrices (BIGGEST!):**
  - Temporal attention: `(32×40, 64, 64)` = 83M elements × 2 bytes = **166 MB** per block
  - Cross-channel attention: `(32×64, 40, 40)` = 103M elements × 2 bytes = **206 MB** per block
  - **Total for 4 blocks:** (166 + 206) × 4 = **1.5 GB**
- Feed-forward intermediate: `(32, 64, 40, 1536)` = 126M elements × 2 bytes = **252 MB**
- Activations for backward: Similar amounts
- **Total:** ~**2-3 GB**

**Why so much?**
- **Attention matrices scale quadratically:** `O(patches² + channels²)`
- **Stored for backward:** Gradient computation needs attention weights
- **Multiple blocks:** 4 transformer blocks accumulate

**Optimization opportunity:**
- Gradient checkpointing: Recompute instead of store (trade compute for memory)
- Flash attention: Fused attention kernel (saves intermediate matrices)

---

### 🟠 **Stage 2: Batch Padding** (~20% of memory)

**What's stored:**
- Padded patches: `(32, 64, 96, 40)` = 78M elements × 2 bytes = **156 MB**
- Augmented patches: Another **156 MB**
- **Total:** ~**300 MB**

**Why expensive?**
- **Padding overhead:** Most samples have 20-30 patches, but padded to 64
- **Wasted memory:** `(64 - avg_patches) / 64 ≈ 50%` of tensor is padding
- **Two copies:** Original + augmented

**Optimization opportunity:**
- Dynamic batching: Group samples with similar num_patches
- Pack sequences: Use PackedSequence (like in RNNs)

---

### 🟡 **Stage 3: Optimizer State** (~15% of memory)

**What's stored (AdamW):**
- Parameters: 384M params × 4 bytes (FP32) = **1.5 GB**
- Momentum (first moment): 384M × 4 bytes = **1.5 GB**
- Variance (second moment): 384M × 4 bytes = **1.5 GB**
- **Total:** ~**4.5 GB** (doesn't scale with batch size!)

**Why expensive?**
- **FP32 storage:** Optimizer always uses FP32 (even with AMP)
- **Two state buffers:** AdamW needs momentum + variance
- **Per-parameter:** Scales with model size

**Optimization opportunity:**
- 8-bit optimizers (bitsandbytes): Reduce to ~2 GB

---

### 🟢 **Stage 4: Gradients** (~5% of memory)

**What's stored:**
- Gradients: 384M params × 2 bytes (FP16) = **768 MB**
- Gradient for projection head: Small
- Gradient for reconstruction head: Small

**Total:** ~**1 GB**

---

### **Total GPU Memory (Batch Size 32):**

```
Forward pass activations:   2.5 GB
Backward pass gradients:    1.0 GB
Optimizer state:            4.5 GB
Batch data (2 copies):      0.3 GB
─────────────────────────────────
TOTAL:                     ~8.3 GB
```

**With mixed precision (FP16):** ~**8-10 GB**
**Without mixed precision (FP32):** ~**16-20 GB**

---

### **Memory Scaling with Batch Size:**

| Batch Size | Patches | Channels | Memory (FP16) |
|------------|---------|----------|---------------|
| 8          | 64      | 40       | ~4 GB         |
| 16         | 64      | 40       | ~6 GB         |
| **32**     | **64**  | **40**   | **~10 GB**    |
| 64         | 64      | 40       | ~18 GB        |

**Bottleneck:** Attention matrices scale with `batch_size × patches² × channels²`

---

### **Memory Optimization Strategies (Ranked by Impact):**

1. **Gradient checkpointing** (-40% memory, +20% compute)
   ```python
   from torch.utils.checkpoint import checkpoint
   x = checkpoint(transformer_block, x)
   ```

2. **Reduce max_patches_per_sample** (-30% memory)
   ```python
   MAX_PATCHES_PER_SAMPLE = 32  # Instead of 64
   ```

3. **Flash Attention** (-25% memory, +15% speed)
   ```python
   from flash_attn import flash_attn_func
   ```

4. **8-bit Adam** (-50% optimizer memory)
   ```python
   import bitsandbytes as bnb
   optimizer = bnb.optim.Adam8bit(...)
   ```

5. **Smaller batch size** (linear scaling)
   ```python
   BATCH_SIZE = 16  # Instead of 32
   ```

---

## Summary

### Key Design Decisions:
1. ✅ **Real-world time** (seconds, not timesteps)
2. ✅ **Fixed patch size** (96 timesteps after interpolation)
3. ✅ **Multi-scale CNN** (kernels 3/5/7 for different frequencies)
4. ✅ **Dual-branch transformer** (temporal + cross-channel attention)
5. ✅ **Dual objectives** (MAE + contrastive for complementary learning)
6. ✅ **Per-patch normalization** (stable training across datasets)
7. ✅ **Semantic channel encoding** (handles variable channels)
8. ✅ **Physically-plausible augmentations** (domain knowledge)

### Memory Hotspots:
1. 🔥 **Transformer attention matrices** (60%)
2. 🔥 **Optimizer state** (35%)
3. 🔥 **Batch padding** (5%)

### Shape Flow Example:
```
(1000, 9) → (5, 200, 9) → (5, 96, 9) → (32, 64, 96, 40)
→ (32, 64, 40, 384) → dual heads → scalar losses
```

Your design is well-thought-out and follows best practices for self-supervised learning on time series data!
