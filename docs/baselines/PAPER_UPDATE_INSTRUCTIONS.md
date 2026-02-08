# Paper Update Instructions

This document contains detailed instructions for updating the TSFM paper draft to reflect the current state of the codebase and experiments.

---

## 1. PLACEHOLDER SECTIONS REQUIRING CONTENT

The following sections are currently placeholders and need to be written:

### 1.1 Abstract
**Current:** "abstract"
**Action:** Write a ~150-word abstract covering:
- Problem: HAR from IMUs with heterogeneous datasets
- Approach: Language-aligned sensor representations with open-vocabulary retrieval
- Key results: 49.41% zero-shot accuracy on 3 unseen datasets, trained on 11 datasets
- Contribution: Domain-specific foundation model for wearable sensing

### 1.2 Introduction (Section 1)
**Current:** Just an outline (7 bullet points)
**Action:** Write full introduction following the outline structure. Key points to include:
- HAR is important for health monitoring, fitness, etc.
- Challenge: heterogeneous devices, sensors, label vocabularies
- Existing approaches: closed-world classifiers, general TSFMs
- Our approach: language-aligned IMU encoder with retrieval-based inference
- Key innovation: soft targets for semantic overlap, memory bank, channel-independent architecture
- Results: 11 datasets for training, 49.41% zero-shot on 3 unseen datasets
- Contributions: (1) multi-dataset training framework, (2) semantic alignment with soft targets, (3) open-vocabulary zero-shot evaluation protocol

### 1.3 Related Work (Section 2)
**Current:** "topic 1, topic 2, topic 3"
**Action:** Write 3 subsections:
1. **Human Activity Recognition from IMUs** - Traditional ML, deep learning, CNN/LSTM/Transformer approaches
2. **Time-Series Foundation Models** - Chronos, TimesFM, MOMENT, Lag-Llama; discuss their limitations for high-frequency multi-channel IMU data
3. **Vision-Language and Sensor-Language Alignment** - CLIP, ImageBind, and prior work on aligning sensor data with text

### 1.4 Discussion (Section 7)
**Current:** "Discuss the limitation and future work."
**Action:** Write discussion covering:
- **Limitations:**
  - No Stage 1 pretraining ablation shown yet
  - Novel activities (falling, vehicle entry) still poorly recognized
  - Computational cost (~45.8M parameters)
  - Limited to IMU modality
- **Future work:**
  - Stage 1 self-supervised pretraining experiments
  - Adding more activity categories (falls, ADLs)
  - Multi-modal extension (IMU + audio, IMU + video)
  - Deployment on edge devices

### 1.5 Conclusion (Section 8)
**Current:** "In this paper, we introduce NAME, a new xx"
**Action:** Write conclusion (~100 words):
- Introduced language-aligned IMU encoder for open-vocabulary HAR
- Trained on 11 heterogeneous datasets with soft targets and memory bank
- Achieved 49.41% zero-shot accuracy on 3 unseen datasets
- Demonstrates viability of domain-specific foundation models for wearable sensing

### 1.6 References
**Current:** Empty
**Action:** Add references for:
- HAR datasets: UCI-HAR, HHAR, PAMAP2, WISDM, MHEALTH, UniMiB-SHAR, DSADS, HAPT, KU-HAR, VTT-ConIoT, RecGym, MotionSense, RealWorld, MobiAct
- TSFMs: Chronos, TimesFM, MOMENT, Lag-Llama
- Contrastive learning: CLIP, SimCLR, MoCo
- Sentence-BERT, InfoNCE loss
- Prior HAR deep learning work

---

## 2. DATASET TABLE UPDATE (Table 1)

**Current table shows:**
| Dataset | Hz | #Ch | Patch (s) |
|---------|-----|-----|-----------|
| UCI-HAR | 50 | 9 | 1.0 |
| HHAR | ~50 | 6 | 1.0 |
| UniMiB-SHAR | 50 | 3 | 1.0 |
| MHEALTH | 50 | 6+ | 2.0 |
| PAMAP2 | 100 | 27 | 2.0 |
| WISDM | 20 | 6 | 2.0 |
| MotionSense (unseen) | 50 | 6 | 2.0 |

**Replace with this updated table (11 training + 3 zero-shot):**

| Dataset | Hz | #Ch | Patch (s) | Role |
|---------|-----|-----|-----------|------|
| UCI-HAR | 50 | 9 | 1.0 | Train |
| HHAR | 50 | 6 | 1.0 | Train |
| UniMiB-SHAR | 50 | 3 | 1.0 | Train |
| MHEALTH | 50 | 23 | 1.5 | Train |
| PAMAP2 | 100 | 40 | 2.0 | Train |
| WISDM | 20 | 6 | 1.5 | Train |
| DSADS | 25 | 45 | 2.0 | Train |
| HAPT | 50 | 6 | 1.25 | Train |
| KU-HAR | 100 | 6 | 1.5 | Train |
| VTT-ConIoT | 52 | 6 | 2.0 | Train |
| RecGym | 20 | 6 | 1.5 | Train |
| MotionSense | 50 | 6 | 1.5 | Zero-shot |
| RealWorld | 50 | 3 | 1.5 | Zero-shot |
| MobiAct | 50 | 6 | 1.25 | Zero-shot |

**Update caption:** "Dataset heterogeneity across 11 training datasets and 3 zero-shot evaluation datasets. We unify time resolution by resampling each patch to 64 timesteps and support variable channel counts (3-45 channels) via channel-independent tokenization with masked fusion and pooling."

---

## 3. UPDATE SECTION 6.2 (Datasets)

**Current text:**
> We train jointly on six public HAR datasets spanning diverse sampling rates, channel counts, and label vocabularies: UCI-HAR, HHAR, UniMiB-SHAR, MHEALTH, PAMAP2, and WISDM. We evaluate zero-shot transfer on MotionSense, which is never used during training.

**Replace with:**
> We train jointly on eleven public HAR datasets spanning diverse sampling rates (20-100 Hz), channel counts (3-45 channels), and label vocabularies: UCI-HAR, HHAR, UniMiB-SHAR, MHEALTH, PAMAP2, WISDM, DSADS, HAPT, KU-HAR, VTT-ConIoT, and RecGym. These datasets collectively cover 101 unique activity labels across 32 semantic groups. We evaluate zero-shot transfer on three held-out datasets: MotionSense, RealWorld, and MobiAct, which are never used during training. The zero-shot datasets include both expected activities (walking, running, sitting, etc.) and novel activities not seen in training (falling, vehicle entry).

---

## 4. RESULTS TABLE UPDATES

### 4.1 Table 2: Overall Validation Performance
**Current:**
| Method | Group-Acc (%) | MRR (%) |
|--------|---------------|---------|
| IMU-to-Text Alignment (ours) | 79.48 | 86.81 |

**Action:** Keep or update based on latest 11-dataset training run. If you have validation metrics from the 11-dataset model, use those. The paper should report metrics from the final model configuration.

### 4.2 Table 3: Per-Dataset Validation Accuracy
**Current shows 6 datasets. Replace with 11 datasets:**

| Dataset | Val Group-Acc (%) |
|---------|-------------------|
| UCI-HAR | [get from latest run] |
| HHAR | [get from latest run] |
| UniMiB-SHAR | [get from latest run] |
| MHEALTH | [get from latest run] |
| PAMAP2 | [get from latest run] |
| WISDM | [get from latest run] |
| DSADS | [get from latest run] |
| HAPT | [get from latest run] |
| KU-HAR | [get from latest run] |
| VTT-ConIoT | [get from latest run] |
| RecGym | [get from latest run] |

**Note:** Run the compare script with training dataset evaluation to get these numbers, or extract from training logs.

### 4.3 Table 4: Zero-Shot Performance
**Current:**
| Setting | Group-Acc (%) | MRR (%) |
|---------|---------------|---------|
| MotionSense (unseen), open-vocab candidates | 37.98 | 49.75 |

**Replace with (from latest comparison_results.json):**

| Dataset | Group-Acc (%) | MRR (%) | Patch (s) |
|---------|---------------|---------|-----------|
| MotionSense | 56.93 | 69.64 | 1.5 |
| RealWorld | 45.73 | 63.03 | 1.5 |
| MobiAct | 43.22 | 55.69 | 1.25 |
| **Combined** | **49.41** | — | — |

**Update caption:** "Zero-shot performance on three unseen datasets under open-vocabulary label retrieval (training-labels ∪ unseen-labels, ~100 unique labels)."

---

## 5. ADD NEW TABLE: Dataset Scaling Ablation

**Add a new table showing the effect of adding more training datasets:**

| Training Datasets | Zero-Shot Acc (%) | Improvement |
|-------------------|-------------------|-------------|
| 6 datasets | 29.75 | baseline |
| 11 datasets | 49.41 | +66% relative |

**Per-dataset breakdown:**

| Dataset | 6 Datasets (%) | 11 Datasets (%) | Δ |
|---------|----------------|-----------------|-----|
| MotionSense | 34.64 | 56.93 | +22.29 |
| RealWorld | 27.40 | 45.73 | +18.33 |
| MobiAct | 25.46 | 43.22 | +17.76 |

---

## 6. MODEL PARAMETERS (Add to Section 5 or 6)

**Add a paragraph or table describing model size:**

> The complete model has approximately 45.8M parameters: 23.1M trainable parameters in the IMU encoder and alignment head, plus 22.7M frozen parameters in the Sentence-BERT backbone (all-MiniLM-L6-v2). The IMU encoder consists of a CNN tokenizer (~2M params), temporal Transformer (6 layers, ~15M params), cross-channel fusion (~3M params), and projection head (~3M params).

---

## 7. UPDATE SECTION 6.4 (Open-Vocabulary Candidate Set)

**Current:**
> This yields a candidate inventory on the order of 60+ unique labels

**Replace with:**
> This yields a candidate inventory of approximately 100 unique labels, combining 101 training labels across 32 semantic groups with labels from the unseen datasets.

---

## 8. UPDATE SECTION 6.7 (Zero-Shot Generalization)

**Current text references only MotionSense. Replace with:**

> Table 4 reports zero-shot results on three unseen datasets (MotionSense, RealWorld, MobiAct) under open-vocabulary retrieval over the combined label inventory of ~100 labels. The model achieves 49.41% combined accuracy across all three datasets without any fine-tuning. Performance varies by dataset: MotionSense (56.93%), which has activities most similar to training, performs best, while MobiAct (43.22%), which includes novel fall detection activities, is more challenging.
>
> We observe that activities with semantic overlap to training (walking, running, sitting, standing, stairs) achieve 51.07% accuracy (expected labels), while truly novel activities like falling and vehicle entry achieve only 25.37% accuracy. This suggests the model successfully transfers to new datasets with similar activities but struggles with activity categories not represented in training.

---

## 9. UPDATE SECTION 6.8 (Embedding Separability)

**Current values are from old model. Update with new values if available from 11-dataset model:**

> On validation, the model achieves mean similarity [X.XX] for matched IMU-text pairs and [X.XX] for different-group pairs (gap [X.XX]), indicating that the learned space clusters semantically aligned IMU and text representations while separating non-matching activities.

---

## 10. ABLATIONS SECTION 6.9

**Current text says "we plan the following ablations"**

**Option A (if ablations are done):** Replace with actual results.

**Option B (if ablations not done):** Remove "we plan" and either:
1. Present as future work in Discussion section, OR
2. Remove entirely

**Recommended ablations to run and include:**
1. Hard targets vs soft targets
2. Queue size ablation (Q=0, Q=512, Q=1024)
3. With/without Stage 1 pretraining
4. Dataset count scaling (already done - see Section 5 above)

---

## 11. ADD: Training Details

**Add to Section 6.3 or create new subsection:**

> **Class balancing:** We use group-balanced sampling with capped oversampling (max 3x) to prevent rare activity groups from dominating training while avoiding excessive repetition of rare samples.
>
> **Patch size augmentation:** During training, patch sizes are randomly varied within dataset-specific ranges (e.g., 0.8-1.2s for UCI-HAR) to improve robustness to temporal variations.
>
> **Label augmentation:** For each batch, we include multiple phrasings of the same activity (e.g., "walking", "walking normally", "slow walking") as soft positive targets to improve semantic generalization.

---

## 12. FIGURE UPDATES

### Figure 1 (System Overview)
**Current:** Text-based diagram
**Action:** Consider creating a proper figure showing:
- Left: IMU session → patches → CNN tokenizer → Temporal Transformer → Fusion → Projection
- Right: Text label → Frozen SBERT → Learned pooling → Projection
- Center: Contrastive loss with memory bank

### Add New Figures:
1. **Zero-shot accuracy vs training datasets** - Line plot showing improvement from 6→11 datasets (file: `accuracy_vs_datasets.png`)
2. **Confusion matrix** - For best model on zero-shot (file: `confusion_matrix_11_datasets_zeroshot.png`)
3. **Per-group accuracy histogram** - Showing which activity groups transfer well (file: `histogram_11_datasets_zeroshot.png`)

---

## 13. MINOR TEXT UPDATES

### Section 4.3
**Change:** "Stage 1 pretrains the IMU tokenizer/encoder..."
**Note:** If Stage 1 pretraining ablation is not done, clarify that current results use Stage 2 only (trained from scratch or from Stage 1).

### Section 5.8
**Change:** "all-MiniLM-L6-v2"
**To:** "all-MiniLM-L6-v2 (22.7M parameters, 384-dimensional embeddings)"

### Throughout
**Change:** References to "6 datasets" → "11 datasets"
**Change:** References to "MotionSense" as only zero-shot → "MotionSense, RealWorld, and MobiAct"

---

## 14. SUMMARY OF KEY NUMBERS TO UPDATE

| Metric | Old Value | New Value |
|--------|-----------|-----------|
| Training datasets | 6 | 11 |
| Zero-shot datasets | 1 (MotionSense) | 3 (MotionSense, RealWorld, MobiAct) |
| Training labels | ~60 | 101 |
| Semantic groups | — | 32 |
| Zero-shot accuracy (combined) | 37.98% | 49.41% |
| Zero-shot MRR | 49.75% | — (varies by dataset) |
| Open-vocab candidate size | 60+ | ~100 |
| Model parameters | not stated | 45.8M (23.1M trainable + 22.7M frozen) |
| Channel range | 3-27 | 3-45 |
| Sampling rate range | 20-100 Hz | 20-100 Hz (unchanged) |

---

## 15. DATASET DESCRIPTIONS (for Section 6.2 or Appendix)

**New datasets to describe:**

| Dataset | Description | Source |
|---------|-------------|--------|
| DSADS | Daily and Sports Activities, 19 activities, 8 subjects, 45 channels from 5 body positions | Bilkent University |
| HAPT | Smartphone accelerometer/gyroscope, 12 activities including postural transitions | Smartlab Genova |
| KU-HAR | Smartphone IMU, 18 activities, 90 subjects | Kyungpook National University |
| VTT-ConIoT | Industrial context IoT, 6 activities in manufacturing setting | VTT Finland |
| RecGym | Gym exercises with smartwatch, 22 exercise types | RecSports |
| RealWorld | Smartphone/smartwatch, 8 activities, 15 subjects, multiple body positions | TU Darmstadt |
| MobiAct | Smartphone IMU, 11 activities including falls and ADLs, 57 subjects | Biomedical Research Foundation |

---

## CHECKLIST

- [ ] Write Abstract
- [ ] Write Introduction
- [ ] Write Related Work (3 subsections)
- [ ] Update Table 1 (datasets) - 11 training + 3 zero-shot
- [ ] Update Table 2 (overall validation) - verify numbers
- [ ] Update Table 3 (per-dataset validation) - add 5 new datasets
- [ ] Update Table 4 (zero-shot) - 3 datasets, new numbers
- [ ] Add dataset scaling table
- [ ] Add model parameters
- [ ] Update Section 6.2 (Datasets description)
- [ ] Update Section 6.4 (candidate set size)
- [ ] Update Section 6.7 (zero-shot results)
- [ ] Update Section 6.8 (embedding separability) - if new numbers available
- [ ] Fix Section 6.9 (ablations) - either run or reframe
- [ ] Write Discussion section
- [ ] Write Conclusion
- [ ] Add References
- [ ] Create/update figures
- [ ] Global find/replace: "6 datasets" → "11 datasets"
- [ ] Global find/replace: "MotionSense" alone → "MotionSense, RealWorld, and MobiAct"
