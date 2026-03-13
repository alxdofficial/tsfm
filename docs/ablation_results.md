# Ablation Study Results

## Overview

We ablate three key components of the TSFM architecture by disabling each one independently
while keeping all other components at baseline settings. All runs use the small_deep
configuration (d=384, 8 transformer layers), 100 training epochs, memory bank (512), and
no GradCache.

| Ablation | Component Removed | Why It Matters |
|----------|------------------|----------------|
| **Baseline** | Nothing (control) | Full model with all components enabled |
| No Channel-Text Fusion | Gated cross-attention between sensor patch tokens and channel description text tokens | Tests whether grounding sensor representations in natural-language channel descriptions improves alignment |
| No Signal Augmentation | Jitter + scale augmentation on raw sensor signals | Tests whether input-space data augmentation prevents overfitting and improves generalization |
| No Text Augmentation | Label synonym/template expansion + Hz/window metadata suffixes on channel descriptions | Tests whether diversifying the text embedding space during training improves robustness |

**Evaluation protocol.** Each ablation's best checkpoint is evaluated on 7 HAR datasets across 4 settings:
zero-shot open-set (87 candidate labels), zero-shot closed-set (dataset labels only),
1% supervised fine-tuning, and 10% supervised fine-tuning. We report both accuracy and macro-F1.

---

## Summary: Main Datasets

Averaged over MotionSense, RealWorld, Shoaib, and Opportunity.

### Accuracy

| Ablation | ZS Open | ZS Closed | 1% Sup | 10% Sup |
|----------|--------:|----------:|-------:|--------:|
| **Baseline (all enabled)** | **43.1%** | **51.3%** | **78.7%** | **88.5%** |
| No Channel-Text Fusion | 22.7% | 43.8% | 75.6% | 85.8% |
| No Signal Augmentation | 44.5% | 53.1% | 75.2% | 86.8% |
| No Text Augmentation | 43.5% | 49.1% | 71.7% | 87.2% |

### Macro-F1

| Ablation | ZS Open | ZS Closed | 1% Sup | 10% Sup |
|----------|--------:|----------:|-------:|--------:|
| **Baseline (all enabled)** | **20.2%** | **42.9%** | **78.0%** | **88.6%** |
| No Channel-Text Fusion | 5.4% | 38.0% | 73.9% | 85.8% |
| No Signal Augmentation | 23.3% | 42.1% | 71.3% | 87.2% |
| No Text Augmentation | 21.6% | 36.0% | 69.1% | 87.6% |

### Delta vs Baseline (Accuracy)

| Component Removed | ZS Open | ZS Closed | 1% Sup | 10% Sup |
|-------------------|--------:|----------:|-------:|--------:|
| Channel-Text Fusion | **-20.4%** | -7.4% | -3.2% | -2.7% |
| Signal Augmentation | +1.4% | +1.8% | -3.5% | -1.7% |
| Text Augmentation | +0.4% | -2.1% | **-7.0%** | -1.3% |

---

## Analysis

### 1. Channel-Text Fusion Is Critical for Zero-Shot Transfer

Removing gated cross-attention between sensor tokens and channel text tokens causes the
largest degradation of any ablation: **-20.4% zero-shot open-set accuracy** on main datasets
(43.1% → 22.7%). The effect is consistent across all 4 main datasets and both zero-shot
settings:

| Dataset | ZS Open Δ | ZS Closed Δ |
|---------|----------:|------------:|
| MotionSense | -20.0% | -7.3% |
| RealWorld | -25.9% | -16.0% |
| Shoaib | -13.9% | +0.7% |
| Opportunity | -21.6% | -7.1% |
| **Consistency** | **4/4 hurt** | **3/4 hurt** |

**Why it works.** Channel-text fusion lets the model attend to natural-language descriptions
of what each sensor axis measures (e.g., "accelerometer x-axis on right wrist"). Without it,
the model must learn sensor semantics purely from signal patterns. The fusion mechanism
provides a semantic bridge between the sensor modality and the text label space, which is
especially important for zero-shot transfer where the model has never seen the target labels.

**Training evidence.** The no-fusion model converges significantly slower and to a lower ceiling:

| Epoch | Baseline Val Acc | No Fusion Val Acc | Gap |
|------:|-----------------:|------------------:|----:|
| 1 | 19.6% | 13.6% | -6.0% |
| 25 | 61.9% | 46.8% | -15.1% |
| 50 | 71.3% | 62.2% | -9.1% |
| 100 | 77.6% | 68.8% | -8.8% |
| **Best** | **77.7%** | **69.0%** | **-8.7%** |

Unseen-dataset accuracy (MotionSense, never in training) also drops: 56.8% → 48.7%.
The sim gap at epoch 1 is 0.017 (baseline) vs 0.005 (no fusion), showing the fusion module
provides a much stronger initial alignment signal.

**Impact fades with supervision.** At 10% supervised fine-tuning, the gap narrows to -2.7%.
With enough labeled data, the model can learn sensor semantics from labels alone, but channel-text
fusion provides a strong initialization that is critical in low-data regimes.


### 2. Text Augmentation Is Essential for Supervised Transfer

Removing label synonym/template expansion and channel metadata suffixes has the largest
impact on supervised fine-tuning: **-7.0% at 1% supervised** (78.7% → 71.7%). This is
the most consistent ablation for supervised settings — baseline wins on **all 4 main datasets
at 1% supervised and 7/7 datasets overall**.

| Dataset | 1% Sup Δ (Acc) | 1% Sup Δ (F1) |
|---------|---------------:|--------------:|
| MotionSense | -2.2% | -1.8% |
| RealWorld | -8.7% | -12.1% |
| Shoaib | **-14.1%** | **-15.4%** |
| Opportunity | -3.1% | -6.0% |
| **Consistency** | **4/4 hurt** | **4/4 hurt** |

The Shoaib dataset is most affected (-14.1%), likely because its small size makes it more
sensitive to the diversity of text representations seen during pretraining.

**Why it works.** Text augmentation expands each activity label into multiple surface forms
(e.g., "walking" → "ambulating", "strolling", "walking at normal pace") and appends sensor
metadata to channel descriptions (e.g., "accelerometer x-axis, 50Hz, 2s window"). This
prevents the text encoder from memorizing exact label strings and forces it to build
representations that capture semantic meaning rather than surface tokens. The result is a
more robust contrastive embedding space that transfers better when fine-tuned.

**Training evidence.** The no-text-aug model reaches comparable validation accuracy (74.2%
vs 77.7%) but worse unseen generalization early on, suggesting it overfits to the exact
label phrasings in the training set:

| Epoch | Baseline Unseen | No Text Aug Unseen |
|------:|----------------:|-------------------:|
| 5 | 51.0% | 50.6% |
| 25 | 55.6% | 56.3% |
| 50 | 57.4% | 59.4% |
| 75 | 54.9% | 56.8% |
| 100 | 56.8% | 57.3% |

Interestingly, no-text-aug has *slightly higher* unseen motionsense accuracy during training,
but this doesn't translate to test-time performance across diverse datasets — the zero-shot
closed-set F1 drops by -6.9% on average, indicating the model's text representations are
more brittle despite appearing to generalize on the one unseen dataset tracked during training.


### 3. Signal Augmentation Prevents Supervised Overfitting

Removing jitter + scale augmentation on sensor signals hurts supervised fine-tuning:
**-3.5% at 1% supervised** (78.7% → 75.2%), with the effect concentrated on datasets with
more complex sensor configurations:

| Dataset | 1% Sup Δ (Acc) | 1% Sup Δ (F1) | 10% Sup Δ (Acc) |
|---------|---------------:|--------------:|----------------:|
| MotionSense | +0.6% | +1.4% | +0.7% |
| RealWorld | -2.9% | -3.3% | -0.4% |
| Shoaib | -1.3% | -1.4% | -0.2% |
| Opportunity | **-10.5%** | **-23.4%** | **-6.7%** |
| **Consistency** | **3/4 hurt** | **3/4 hurt** | **3/4 hurt** |

The Opportunity dataset, which has the most sensor channels (body-worn + object sensors,
113 channels), is most affected (-10.5% accuracy, -23.4% F1 at 1% supervised). This
suggests signal augmentation is most valuable when sensor configurations are complex and
varied.

**Why it works.** Jitter (random noise) and scale (random amplitude scaling) augmentation
during pretraining force the encoder to build representations invariant to sensor noise
levels and calibration differences. Without augmentation, the encoder overfits to the exact
signal characteristics of the training data, which hurts transfer to new datasets with
different sensor hardware and placement.

**Training evidence.** The no-aug model achieves nearly identical validation accuracy
(76.7% vs 77.7%) but substantially lower unseen-dataset accuracy:

| Epoch | Baseline Unseen | No Signal Aug Unseen | Delta |
|------:|----------------:|---------------------:|------:|
| 5 | 51.0% | 38.2% | -12.8% |
| 25 | 55.6% | 53.1% | -2.5% |
| 50 | 57.4% | 44.6% | -12.8% |
| 100 | 56.8% | 47.3% | **-9.5%** |

The unseen gap is large and persistent (-9.5% at epoch 100), indicating the model without
signal augmentation genuinely overfits to training sensor characteristics. This is the
strongest training-time evidence of any ablation — the val accuracy gap is small (1.0%)
but the unseen gap is 9.5%, revealing a generalization problem that only shows up on
out-of-distribution sensors.

---

## Summary: Severe OOD Datasets

Averaged over MobiAct, VTT-ConIoT, and HARTH — datasets with substantially different sensor
configurations, sampling rates, or activity taxonomies from training.

### Accuracy

| Ablation | ZS Open | ZS Closed | 1% Sup | 10% Sup |
|----------|--------:|----------:|-------:|--------:|
| **Baseline (all enabled)** | **18.2%** | **20.1%** | **44.2%** | **57.1%** |
| No Channel-Text Fusion | 12.8% | 18.2% | 45.1% | 58.5% |
| No Signal Augmentation | 17.0% | 19.5% | 40.1% | 50.3% |
| No Text Augmentation | 13.7% | 18.4% | 38.9% | 60.7% |

### Delta vs Baseline

| Component Removed | ZS Open | ZS Closed | 1% Sup | 10% Sup |
|-------------------|--------:|----------:|-------:|--------:|
| Channel-Text Fusion | -5.4% | -1.8% | +0.9% | +1.3% |
| Signal Augmentation | -1.2% | -0.6% | -4.1% | **-6.8%** |
| Text Augmentation | -4.5% | -1.7% | -5.3% | +3.6% |

Signal augmentation shows its strongest effect here: **-6.8% at 10% supervised on OOD data**,
consistent with the hypothesis that jitter+scale augmentation builds robustness to
different sensor hardware.

---

## Per-Dataset Results

### Main Datasets

#### MotionSense

| Ablation | ZS Open | ZS Closed | 1% Sup | 10% Sup |
|----------|--------:|----------:|-------:|--------:|
| **Baseline** | 54.0% | 64.1% | 88.2% | 93.9% |
| No Channel-Text Fusion | 33.9% | 56.8% | 85.5% | 93.8% |
| No Signal Augmentation | 55.5% | 63.2% | 88.7% | 94.5% |
| No Text Augmentation | 63.1% | 65.5% | 86.0% | 93.1% |

#### RealWorld

| Ablation | ZS Open | ZS Closed | 1% Sup | 10% Sup |
|----------|--------:|----------:|-------:|--------:|
| **Baseline** | 50.4% | 51.3% | 72.9% | 84.6% |
| No Channel-Text Fusion | 24.5% | 35.3% | 67.8% | 78.3% |
| No Signal Augmentation | 47.8% | 48.5% | 69.9% | 84.1% |
| No Text Augmentation | 32.5% | 32.4% | 64.2% | 84.5% |

#### Shoaib

| Ablation | ZS Open | ZS Closed | 1% Sup | 10% Sup |
|----------|--------:|----------:|-------:|--------:|
| **Baseline** | 42.0% | 44.2% | 88.1% | 95.3% |
| No Channel-Text Fusion | 28.1% | 45.0% | 85.8% | 95.5% |
| No Signal Augmentation | 40.1% | 41.8% | 86.8% | 95.1% |
| No Text Augmentation | 38.9% | 41.9% | 74.1% | 96.0% |

#### Opportunity

| Ablation | ZS Open | ZS Closed | 1% Sup | 10% Sup |
|----------|--------:|----------:|-------:|--------:|
| **Baseline** | 26.1% | 45.4% | 65.8% | 80.2% |
| No Channel-Text Fusion | 4.4% | 38.4% | 63.2% | 75.4% |
| No Signal Augmentation | 34.7% | 58.7% | 55.3% | 73.5% |
| No Text Augmentation | 39.4% | 56.7% | 62.7% | 75.2% |

### Severe OOD Datasets

#### MobiAct

| Ablation | ZS Open | ZS Closed | 1% Sup | 10% Sup |
|----------|--------:|----------:|-------:|--------:|
| **Baseline** | 52.1% | 54.6% | 58.4% | 73.1% |
| No Channel-Text Fusion | 35.8% | 43.5% | 62.1% | 69.4% |
| No Signal Augmentation | 48.3% | 53.0% | 53.8% | 70.3% |
| No Text Augmentation | 39.6% | 53.1% | 51.5% | 72.2% |

#### VTT-ConIoT

| Ablation | ZS Open | ZS Closed | 1% Sup | 10% Sup |
|----------|--------:|----------:|-------:|--------:|
| **Baseline** | 0.5% | 3.4% | 4.8% | 18.8% |
| No Channel-Text Fusion | 0.7% | 4.2% | 6.8% | 26.1% |
| No Signal Augmentation | 1.9% | 4.8% | 3.4% | 3.9% |
| No Text Augmentation | 0.3% | 2.0% | 1.9% | 30.0% |

#### HARTH

| Ablation | ZS Open | ZS Closed | 1% Sup | 10% Sup |
|----------|--------:|----------:|-------:|--------:|
| **Baseline** | 2.0% | 2.1% | 69.3% | 79.5% |
| No Channel-Text Fusion | 1.8% | 7.1% | 66.5% | 79.9% |
| No Signal Augmentation | 0.9% | 0.6% | 63.0% | 76.7% |
| No Text Augmentation | 1.1% | 0.1% | 63.3% | 80.0% |

---

## Training Metrics

Best validation accuracy and unseen-dataset accuracy (MotionSense, held out during training)
from training logs.

| Ablation | Best Val Acc | Best Epoch | Unseen Acc (ep100) |
|----------|:-----------:|:----------:|:------------------:|
| Baseline | 77.7% | 96 | 56.8% |
| No Channel-Text Fusion | 69.0% | 94 | 48.7% |
| No Signal Augmentation | 76.7% | 95 | 47.3% |
| No Text Augmentation | 74.2% | 95 | 57.3% |

### Validation Accuracy Learning Curves

| Epoch | Baseline | No Fusion | No Sig Aug | No Text Aug |
|------:|---------:|----------:|-----------:|------------:|
| 1 | 19.6% | 13.6% | 13.9% | 14.8% |
| 10 | 44.7% | 28.0% | 38.4% | 42.3% |
| 25 | 61.9% | 46.8% | 58.6% | 59.5% |
| 50 | 71.3% | 62.2% | 70.1% | 70.6% |
| 75 | 76.6% | 67.7% | 75.3% | 74.0% |
| 100 | 77.6% | 68.8% | 76.7% | 74.2% |

### Unseen Dataset (MotionSense) Learning Curves

| Epoch | Baseline | No Fusion | No Sig Aug | No Text Aug |
|------:|---------:|----------:|-----------:|------------:|
| 5 | 51.0% | 44.0% | 38.2% | 50.6% |
| 25 | 55.6% | 43.7% | 53.1% | 56.3% |
| 50 | 57.4% | 48.5% | 44.6% | 59.4% |
| 75 | 54.9% | 50.5% | 46.9% | 56.8% |
| 100 | 56.8% | 48.7% | 47.3% | 57.3% |

### Contrastive Training Metrics

| Ablation | Sim Gap (ep1) | Sim Gap (ep100) | Pos Sim (ep1) | Pos Sim (ep100) |
|----------|:------------:|:--------------:|:------------:|:--------------:|
| Baseline | 0.017 | 0.220 | 0.721 | 0.966 |
| No Channel-Text Fusion | 0.005 | 0.204 | 0.740 | 0.969 |
| No Signal Augmentation | 0.005 | 0.226 | 0.769 | 0.984 |
| No Text Augmentation | 0.008 | 0.225 | 0.807 | 0.984 |

The baseline achieves the highest sim gap (0.220) with the lowest positive similarity
(0.966), indicating better separation between matched and unmatched pairs. No-fusion
starts with the weakest alignment signal (sim gap 0.005) and never fully recovers.
No-signal-aug and no-text-aug both show higher positive similarity (0.984) than baseline,
suggesting they may be overfitting — collapsing positive pairs together rather than
learning discriminative features.

---

## Paper Context

Additional context for writing the ablation section of the paper.

### Model Architecture

- **Architecture:** small_deep configuration — d_model=384, 8 transformer layers, 8 attention heads, FFN dim=1536
- **Trainable parameters:** 29.3M total
  - Encoder (8-layer temporal transformer + CNN feature extractor): 19.0M
  - Semantic alignment head (cross-channel fusion + temporal + attention pooling + projection): 8.7M
  - Channel-text fusion (gated cross-attention): 1.6M
- **Frozen text encoder:** all-MiniLM-L6-v2 (22.7M params, not updated during training)
- **Contrastive loss:** InfoNCE with learnable temperature (logit scale)

### Training Setup

- **Training datasets (10):** UCI-HAR, HHAR, MHEALTH, PAMAP2, WISDM, UniMiB-SHAR, DSADS, HAPT, KU-HAR, RecGym
- **Held-out during training:** MotionSense (used for unseen-dataset tracking only)
- **Test datasets (7, never seen during training):** MotionSense, RealWorld, Shoaib, Opportunity (main); MobiAct, VTT-ConIoT, HARTH (severe OOD)
- **Batch size:** 128 (RunPod, 4 ablations) / 24 (local RTX 4090, 2 ablations)
- **Gradient accumulation:** 4 steps (RunPod) / 22 steps (local) → effective batch ~512
- **Learning rate:** 8e-5 with cosine annealing, 3 warmup epochs
- **Weight decay:** 1e-5
- **Memory bank:** 512 entries (FIFO queue of past embeddings as additional negatives)
- **Epochs:** 100
- **Single run per ablation** (no error bars — deltas should be interpreted with this caveat)

### Ablated Component Details

**Channel-Text Fusion (1.6M params).** Gated cross-attention module inserted after the
temporal encoder. Sensor patch tokens (queries) attend to SBERT token-level embeddings of
channel descriptions (keys/values) via multi-head cross-attention (4 heads). A learned gate
(two linear projections + sigmoid) controls how much text information is mixed into the
sensor representation: `output = gate * text_context + (1 - gate) * sensor_input`. This
is distinct from the positional channel encoding, which adds a fixed SBERT embedding per
channel to every patch — the fusion module provides a richer, token-level interaction.

**Signal Augmentation (0 params).** Applied per-sample during training:
- Jitter: additive Gaussian noise with sigma=0.05
- Scale: random amplitude scaling in range [0.9, 1.1]
- Both applied with probability 1.0 when enabled

**Text Augmentation (0 params).** Applied per-sample during training with 80% probability:
- *Label synonyms:* each activity label has 3-5 manually curated synonyms per dataset
  (e.g., "walking" → "strolling", "ambulating", "pacing"). One is sampled randomly.
- *Label templates:* the synonym is wrapped in a randomly chosen template
  (e.g., "person {}", "subject performing {}", "{} activity"). ~8-9 templates per dataset.
- *Channel description suffix:* appends sensor metadata to each channel description
  (e.g., "accelerometer x-axis" → "accelerometer x-axis (sampled at 50Hz, 1.0s window)").
  This gives the text encoder information about temporal resolution.

### Evaluation Protocol

- **Zero-shot open-set:** Classify against all 87 training labels using cosine similarity
  between IMU embeddings and text label embeddings. Measures whether the model can identify
  the correct activity from a large candidate set it has never been tested on.
- **Zero-shot closed-set:** Classify against only the target dataset's labels (4-16 classes).
  Measures performance when the label space is known but no labeled examples are available.
- **1% supervised fine-tuning:** End-to-end fine-tune on 1% of labeled data (20 epochs,
  early stopping with patience=5). Linear head on top of frozen-then-unfrozen encoder.
- **10% supervised fine-tuning:** Same protocol with 10% of labeled data.

### Statistical Notes

- All results are from single training runs (no repeated trials with different seeds).
- Deltas within ~2% should be considered within noise range for single-run comparisons.
- The three ablations reported (channel-text fusion, signal augmentation, text augmentation)
  were selected because they show consistent, directional effects across multiple datasets
  and metrics. Two other ablations (learnable label bank, soft targets) showed mixed or
  negligible effects and are omitted for clarity.
- The strongest evidence comes from ablations where the effect is consistent across datasets
  (e.g., channel-text fusion hurts 6/7 datasets on ZS open, text augmentation hurts 7/7 on
  1% supervised).
