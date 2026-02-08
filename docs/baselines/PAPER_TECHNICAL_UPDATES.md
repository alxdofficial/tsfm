# Paper Technical Updates

Technical content changes needed based on current codebase. Focused on implementation details, not placeholder sections.

---

## 1. DATASET UPDATES

### 1.1 Training Datasets (6 → 11)

**Paper says:** 6 datasets (UCI-HAR, HHAR, UniMiB-SHAR, MHEALTH, PAMAP2, WISDM)

**Current code:** 11 datasets
```python
DATASETS = ['uci_har', 'hhar', 'mhealth', 'pamap2', 'wisdm', 'unimib_shar',
            'dsads', 'hapt', 'kuhar', 'vtt_coniot', 'recgym']
```

**New datasets to add to Table 1:**

| Dataset | Hz | #Ch | Patch (s) | Description |
|---------|-----|-----|-----------|-------------|
| DSADS | 25 | 45 | 2.0 | Daily/sports activities, 5 body positions |
| HAPT | 50 | 6 | 1.25 | Smartphone IMU with postural transitions |
| KU-HAR | 100 | 6 | 1.5 | Smartphone IMU, 18 activities, 90 subjects |
| VTT-ConIoT | 52 | 6 | 2.0 | Industrial IoT manufacturing context |
| RecGym | 20 | 6 | 1.5 | Smartwatch gym exercises |

### 1.2 Zero-Shot Datasets (1 → 3)

**Paper says:** MotionSense only

**Current code:** 3 datasets
```python
UNSEEN_DATASETS = ['motionsense', 'realworld', 'mobiact']
```

**Add to Table 1:**

| Dataset | Hz | #Ch | Patch (s) | Description |
|---------|-----|-----|-----------|-------------|
| MotionSense | 50 | 6 | 1.5 | Smartphone IMU, basic activities |
| RealWorld | 50 | 3 | 1.5 | Smartphone/smartwatch, acc-only |
| MobiAct | 50 | 6 | 1.25 | Includes falls and ADLs (novel activities) |

### 1.3 Updated Patch Sizes

**Paper Table 1 has outdated patch sizes. Current values:**

| Dataset | Paper | Current |
|---------|-------|---------|
| MHEALTH | 2.0 | 1.5 |
| WISDM | 2.0 | 1.5 |
| MotionSense | 2.0 | 1.5 |

---

## 2. SOFT TARGETS IMPLEMENTATION (Section 5.9)

**Paper describes:** "global z-score normalization" but lacks detail

**Actual implementation (semantic_loss.py:145-161):**

```python
# Compute text similarity over full dimension (batch + queue)
text_similarity_full = torch.matmul(text_embeddings, all_text.T)  # (batch, batch+queue)

# Adaptive soft targets: normalize similarities to z-scores
# Problem: SentenceBERT gives 0.4-0.9 for ALL human activities → weak discrimination
# Solution: Convert to z-scores so differences are amplified
sim_mean = text_similarity_full.mean()
sim_std = text_similarity_full.std().clamp(min=0.1)
text_similarity_full = (text_similarity_full - sim_mean) / sim_std / soft_target_temperature

# Apply softmax to get probability distribution
soft_targets_full = F.softmax(text_similarity_full, dim=1)

# Blend with hard targets
targets = (1 - soft_target_weight) * hard_targets + soft_target_weight * soft_targets_full
```

**Key insight to add:** SentenceBERT produces similarities in the 0.4-0.9 range for ALL human activities, providing weak discrimination. Z-score normalization amplifies the differences:
- True synonyms (walking/strolling): z ≈ +2 → high weight
- Different activities (walking/sitting): z ≈ -1 → low weight

**Current hyperparameters:**
- `SOFT_TARGET_TEMPERATURE = 0.5` (sharper than paper implies)
- `SOFT_TARGET_WEIGHT = 1.0` (pure soft targets, not blended)

---

## 3. MEMORY BANK IMPLEMENTATION (Section 5.10)

**Paper describes:** "FIFO queues of detached embeddings stored on CPU"

**Additional details from code (memory_bank.py):**

1. **Stores both IMU and text embeddings** (not just one modality)
2. **Queue size:** 512 embeddings (configurable)
3. **Memory bank warmup:** Queue is pre-filled before training starts to avoid volatility from empty/low-quality initial embeddings
4. **Queue negatives are hard negatives:** Text labels are NOT stored with queue items. Re-encoding 512+ labels per batch would be expensive.

**Add to paper:**
> The memory bank caches both IMU and text embeddings, avoiding re-encoding of text labels for queue items. Before training begins, we perform a warmup phase that fills the queue with embeddings from the training data, reducing volatility from early low-quality representations.

---

## 4. LABEL AUGMENTATION (Missing from paper)

**Not mentioned in paper but critical for training:**

The dataset loader applies dataset-specific label augmentation during training (80% probability):

```python
# From label_augmentation.py
UCI_HAR_SYNONYMS = {
    "walking": ["walking", "strolling", "striding", "ambulating", "pacing"],
    "walking_upstairs": ["walking upstairs", "climbing stairs", "ascending stairs"],
    ...
}

UCI_HAR_TEMPLATES = [
    "{}",
    "person {}",
    "person is {}",
    "{} activity",
    ...
]
```

**Add new subsection 5.X: Label Augmentation**
> During training, we augment activity labels with dataset-specific synonyms and natural language templates. Each dataset has a curated set of synonyms (e.g., "walking" ↔ "strolling" ↔ "ambulating") and templates (e.g., "{}" → "person {}" → "{} activity"). Augmentation is applied with 80% probability during training and disabled during validation. This provides rich variation for contrastive learning and is essential for soft targets to work correctly—without augmentation, treating all non-matching labels as hard negatives would be appropriate, but with augmentation, soft targets prevent contradictory gradients when synonymous labels appear in the same batch.

---

## 5. PATCH SIZE AUGMENTATION (Missing from paper)

**Not mentioned in paper:**

```python
USE_PATCH_SIZE_AUGMENTATION = True

PATCH_SIZE_RANGE_PER_DATASET = {
    'uci_har':      (0.75, 1.25, 0.25),  # [0.75, 1.0, 1.25]
    'hhar':         (0.75, 1.25, 0.25),
    'mhealth':      (1.0, 1.75, 0.25),   # [1.0, 1.25, 1.5, 1.75]
    'pamap2':       (1.0, 2.0, 0.5),     # [1.0, 1.5, 2.0]
    ...
}
```

**Add to Section 5.4 (Preprocessing):**
> During training, we apply patch size augmentation by randomly selecting patch durations from a dataset-specific range (e.g., 0.75-1.25s for UCI-HAR). This improves robustness to temporal variations and is disabled during validation/evaluation where fixed patch sizes are used.

---

## 6. CLASS BALANCING (Missing from paper)

**Not mentioned in paper:**

```python
USE_GROUP_BALANCED_SAMPLING = True
MAX_OVERSAMPLE_RATIO = 20.0  # Cap oversampling to prevent rare groups dominating

# Compute weights for group-balanced sampling (with capped oversampling)
sample_weights = train_dataset.compute_group_weights(max_oversample_ratio=MAX_OVERSAMPLE_RATIO)
sampler = WeightedRandomSampler(weights=sample_weights, ...)
```

**Add to Section 6.3 (Training Protocol):**
> We use group-balanced sampling with capped oversampling (maximum 20×) to address class imbalance across datasets. Activity labels are grouped into semantic categories (e.g., "walking", "strolling", "nordic_walking" → walking group), and sampling weights are computed inversely proportional to group frequency. The cap prevents rare activities from dominating training while still providing sufficient coverage.

---

## 7. UPDATED RESULTS

### 7.1 Zero-Shot Accuracy (Table 4)

**Paper:** 37.98% on MotionSense only

**Current (from comparison_results.json):**

| Dataset | Group-Acc (%) | MRR (%) | Patch (s) |
|---------|---------------|---------|-----------|
| MotionSense | 56.93 | 69.64 | 1.5 |
| RealWorld | 45.73 | 63.03 | 1.5 |
| MobiAct | 43.22 | 55.69 | 1.25 |
| **Combined** | **49.41** | — | — |

### 7.2 Dataset Scaling Effect (New table)

| Training Config | Zero-Shot Acc (%) | Δ |
|-----------------|-------------------|---|
| 6 datasets | 29.75 | baseline |
| 11 datasets | 49.41 | +19.66 pp (+66% relative) |

**Per-dataset improvement:**

| Dataset | 6 DS (%) | 11 DS (%) | Δ |
|---------|----------|-----------|---|
| MotionSense | 34.64 | 56.93 | +22.29 |
| RealWorld | 27.40 | 45.73 | +18.33 |
| MobiAct | 25.46 | 43.22 | +17.76 |

### 7.3 Novel vs Expected Activities

Zero-shot evaluation distinguishes between:
- **Expected labels:** Activities with semantic overlap to training (walking, running, sitting, stairs)
- **Novel labels:** Activities NOT in training (falling, vehicle entry - only in MobiAct)

| Category | Accuracy (%) | Sample Count |
|----------|--------------|--------------|
| Expected activities | 51.07 | 3039 |
| Novel activities | 25.37 | 205 |

This demonstrates the model transfers well to new datasets with similar activities but struggles with truly novel activity categories.

---

## 8. MODEL SIZE (Add to Section 5 or 6)

**Not stated in paper. Current values:**

| Component | Parameters |
|-----------|------------|
| IMU Encoder (trainable) | ~23.1M |
| SBERT backbone (frozen) | 22.7M |
| **Total** | **45.8M** |

**Breakdown:**
- CNN tokenizer: ~2M
- Temporal Transformer (6 layers): ~15M
- Cross-channel fusion: ~3M
- Projection head: ~3M

---

## 9. HYPERPARAMETER UPDATES

**Section 6.3 needs these current values:**

| Parameter | Paper | Current |
|-----------|-------|---------|
| Temperature | 0.1 | 0.07 (CLIP default) |
| Soft target temperature | τ_s | 0.5 |
| Soft target weight | — | 1.0 (pure soft) |
| Memory bank size | Q | 512 |
| Gradient accumulation | 32 | 32 (unchanged) |
| Max oversample ratio | — | 20.0 |

---

## 10. LEARNABLE TEMPERATURE (Section 5.9)

**Paper mentions:** "learned temperature scale clamped to a bounded range"

**Actual implementation:**
```python
# Initialized to 1/0.07 ≈ 14.3 (CLIP-style)
self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/temperature))

# Clamped during forward pass
logits = torch.matmul(imu, text.T) * self.logit_scale.exp().clamp(1, 50)
```

The temperature is learned as `exp(logit_scale)` to ensure positivity, clamped to [1, 50].

---

## 11. BIDIRECTIONAL LOSS (Section 5.9)

**Paper mentions symmetric loss but code shows explicit implementation:**

```python
# IMU→Text direction
logits = torch.matmul(imu_embeddings, all_text.T) * scale
loss_imu_to_text = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

# Text→IMU direction
logits_t2i = torch.matmul(text_embeddings, all_imu.T) * scale
loss_text_to_imu = -(targets * F.log_softmax(logits_t2i, dim=1)).sum(dim=1).mean()

loss = (loss_imu_to_text + loss_text_to_imu) / 2.0
```

---

## 12. EVALUATION PROTOCOL CLARIFICATION

### 12.1 Open-Vocabulary Candidate Set

**Paper says:** "60+ unique labels"

**Current:** ~100 unique labels (101 training labels across 32 semantic groups + zero-shot labels)

### 12.2 Retrieval Pool Size

Zero-shot evaluation uses `max_retrieval_pool=100` labels for the combined inventory.

---

## SUMMARY CHECKLIST

- [ ] Update Table 1: Add 5 new training datasets + 2 new zero-shot datasets
- [ ] Update Table 1: Fix patch sizes (MHEALTH, WISDM, MotionSense)
- [ ] Update Table 4: 3 datasets instead of 1, new accuracy numbers
- [ ] Add dataset scaling table (6→11 datasets effect)
- [ ] Add Section 5.X: Label Augmentation
- [ ] Add to Section 5.4: Patch size augmentation
- [ ] Add to Section 6.3: Class balancing with capped oversampling
- [ ] Update Section 5.9: Soft target z-score normalization detail
- [ ] Update Section 5.9: Learnable temperature initialization (CLIP-style)
- [ ] Update Section 5.10: Memory bank warmup, stores both modalities
- [ ] Add model parameter counts (~45.8M total)
- [ ] Update hyperparameters (temperature=0.07, soft_weight=1.0)
- [ ] Update candidate set size (60+ → ~100)
- [ ] Add novel vs expected activity breakdown
