# TSFM vs Baselines: Architectural Comparison

A detailed comparison of TSFM's design against 4 baselines, explaining what architectural
novelties contribute to TSFM's performance advantages and where tradeoffs exist.

---

## Model Summaries

| Model | Parameters | Embed Dim | Architecture in One Sentence |
|-------|----------:|:---------:|------------------------------|
| **TSFM** | ~5M | 384 | Dual-branch (temporal + cross-channel) Transformer with CLIP-style contrastive alignment to frozen SentenceBERT text embeddings, using soft targets, a memory bank, and variable-length patch tokenization. |
| **LiMU-BERT** | ~62K | 72 | Single shared-parameter Transformer trained via masked reconstruction on 20-step IMU sub-windows; predicts masked timesteps from context. |
| **MOMENT** | ~341M | 6144 | T5-Large encoder (24 layers, 1024-dim) pretrained on 13 domains of general time-series data via masked patch reconstruction; processes each IMU channel independently. |
| **CrossHAR** | ~57K | 72 | Single shared-parameter Transformer with hierarchical pretraining: masked reconstruction first, then contrastive learning added; uses channel permutation augmentation. |
| **LanHAR** | ~10M | 768 | 2-stage CLIP-style alignment: fine-tune SciBERT on activity text, then train a 3-layer sensor Transformer from scratch to align with the text space via contrastive loss. |

---

## Architectural Comparison

| Design Aspect | TSFM | LiMU-BERT | MOMENT | CrossHAR | LanHAR |
|--------------|------|-----------|--------|----------|--------|
| **Encoder depth** | 4 dual-branch layers | 1 shared layer (looped 4x) | 24 layers | 1 shared layer | 3 layers |
| **Attention type** | Temporal + cross-channel (alternating) | Temporal only | Temporal only (per-channel) | Temporal only | Temporal only |
| **Cross-channel modeling** | Explicit cross-channel self-attention at every layer | None | None (channels processed independently) | None | None |
| **Tokenization** | Variable-length patches (e.g., 20 timesteps at 20Hz), interpolated to 64 fixed steps | Per-timestep (each of 120 timesteps = 1 token) | 8-timestep non-overlapping patches | Per-timestep (120 tokens) | Per-timestep (120 tokens) |
| **Variable sampling rate** | Yes (patches interpolated to fixed 64 steps) | No (fixed 20 Hz) | No (fixed input, left-padded to 512) | No (fixed 20 Hz) | No (fixed 50 Hz) |
| **Variable channel count** | Yes (channel-bucketed batching) | No (fixed 6) | Yes (per-channel independent) | No (fixed 6) | No (fixed 6) |
| **Text alignment** | Yes (contrastive with SentenceBERT) | No | No | No | Yes (contrastive with SciBERT) |
| **Zero-shot mechanism** | Cosine sim with learnable text label embeddings | Requires trained GRU classifier | Requires trained SVM classifier | Requires trained Transformer_ft | Cosine sim with text prototypes |
| **Pretraining objective** | MAE + patch contrastive (stage 1), then CLIP contrastive with soft targets (stage 2) | Masked reconstruction (MSE) | Masked patch reconstruction (MSE) | Masked reconstruction, then +contrastive (hierarchical) | Stage 1: SciBERT fine-tuning; Stage 2: sensor-text CLIP |
| **Positional encoding** | Sinusoidal temporal + semantic channel encoding | Learned (nn.Embedding) | T5 relative attention bias + sinusoidal | Learned (nn.Embedding) | Sinusoidal |
| **Normalization** | Per-patch per-channel z-score | acc/9.8, gyro raw | RevIN (per-channel instance norm) | InstanceNorm1d | Gravity alignment + raw |
| **Supervised fine-tuning** | End-to-end, cosine sim with frozen text embeddings (no classifier head) | End-to-end encoder + GRU | End-to-end encoder + linear head | End-to-end encoder + Transformer_ft | End-to-end, cosine sim with frozen text prototypes |

---

## TSFM's Key Novelties

### 1. Dual-Branch Transformer: Explicit Cross-Channel Modeling

**What TSFM does**: Each of the 4 Transformer layers alternates between two attention operations:
- **Temporal self-attention**: Reshapes `(B, P, C, 384)` → `(B*C, P, 384)` — each channel's patch sequence attends to itself independently.
- **Cross-channel self-attention**: Reshapes `(B, P, C, 384)` → `(B*P, C, 384)` — all channels within each patch attend to each other.

This means at every layer, the model learns both *how activities evolve over time* (temporal branch) and *how sensor channels correlate* (cross-channel branch).

**What baselines do**: Every other model processes channels either jointly as a flat vector (LiMU-BERT, CrossHAR, LanHAR all project 6-dim input to hidden dim) or completely independently (MOMENT processes each channel as a separate univariate series). None has explicit cross-channel attention.

**Why it matters**: Human activities produce coordinated patterns across IMU channels — walking creates rhythmic coupling between acc_y (vertical bounce), acc_z (forward thrust), and gyro_x (leg swing). The flat-vector approach forces the linear projection to implicitly learn channel interactions at the input layer only. MOMENT's per-channel approach defers all cross-channel reasoning to the downstream classifier. TSFM's cross-channel attention learns these interactions at every layer of the representation, making them available throughout the feature hierarchy.

**Connection to results**: TSFM's strongest advantage is in zero-shot, where the quality of the learned embedding space matters most. The cross-channel attention produces richer representations that better capture the multi-axis signatures of activities, leading to more discriminative embeddings for cosine-similarity classification.

### 2. Soft Contrastive Targets

**What TSFM does**: Instead of hard one-hot targets in the InfoNCE loss (where only the matched text is positive and all others are negative), TSFM computes pairwise cosine similarity between text embeddings and uses the resulting distribution as soft targets:

```
text_sim = text_embs @ all_text.T          # pairwise text similarity
soft_targets = softmax(z_normalize(text_sim) / 0.5)   # sharpened soft distribution
loss = -(soft_targets * log_softmax(logits)).sum()
```

This means semantically similar labels (e.g., "jogging" and "running") are treated as partial positives rather than hard negatives.

**What baselines do**:
- LiMU-BERT/CrossHAR/MOMENT: No text alignment, so N/A.
- LanHAR: Uses `clip_loss_multipos` in Stage 1 where all same-class samples are positives, but this is label-identity-based, not semantic-similarity-based. In Stage 2, uses standard hard-target CLIP loss.

**Why it matters**: The 10 training datasets use 87 different activity labels, many of which are synonyms or near-synonyms (e.g., "walking", "walk", "strolling"; "sitting", "sit", "seated"). With hard targets, the model would learn to push "walking" and "strolling" apart — exactly the wrong thing. Soft targets preserve the semantic neighborhood structure of the label space, teaching the model that similar activities should have similar embeddings.

**Connection to results**: This is a key reason TSFM's zero-shot transfer works across datasets with different label vocabularies. When a test dataset uses "jogging" but the training data had "running", the model already learned that these are nearby in embedding space.

### 3. Learnable Label Attention Pooling

**What TSFM does**: Raw text labels are encoded by a frozen SentenceBERT (`all-MiniLM-L6-v2`), producing token-level embeddings. Then 4 learnable query tokens cross-attend to these text tokens, and their outputs are projected to produce the final 384-dim label embedding. These attention weights are trained during semantic alignment.

**What LanHAR does**: Uses margin-based top-K selection from hand-crafted text prototype snippets (~22 per class), then temperature-weighted averaging of raw SciBERT embeddings. The prototype construction is algorithmic, not learned end-to-end.

**Why it matters**: TSFM's learnable pooling can discover which parts of a label's text are most discriminative for IMU activity recognition. For example, for "climbing stairs", the pooling might attend more to "climbing" than to "stairs" because the climbing motion has a more distinctive IMU signature. This is learned from the contrastive training signal, not hand-designed. Additionally, TSFM can encode *any* text string at test time — it doesn't need pre-crafted prototype lists.

### 4. Channel-Semantic Text Fusion

**What TSFM does**: Before the main Transformer, `ChannelTextFusion` injects semantic channel information directly into the sensor token representations. The 6 channel descriptions (e.g., "Accelerometer X-axis") are encoded by frozen SentenceBERT, then cross-attended by learnable queries, and gated into the sensor features:

```
fused = sensor_tokens + sigmoid(gate) * channel_embeddings
```

**What baselines do**: No baseline injects channel identity semantically. LiMU-BERT/CrossHAR treat channel identity as implicit in the input vector position. MOMENT processes channels independently with no identity signal. LanHAR gravity-aligns but doesn't encode channel semantics.

**Why it matters**: This tells the model *what each channel represents*, not just *what values it has*. When TSFM sees a new dataset with channels labeled "wrist accelerometer X" vs "hip accelerometer X", the channel text fusion provides the semantic context that helps the model interpret the signals correctly. This supports generalization across sensor placements and configurations.

### 5. Variable-Length Patch Tokenization with Interpolation

**What TSFM does**: Raw IMU signals are segmented into patches of configurable duration (e.g., 1.0s = 20 timesteps at 20Hz), then each patch is interpolated to a fixed 64-timestep representation using `F.interpolate`. This decouples temporal resolution from sampling rate.

**What baselines do**:
- LiMU-BERT/CrossHAR/LanHAR: Per-timestep tokenization — each of 120 timesteps becomes one token. Fixed to 20Hz (or 50Hz for LanHAR).
- MOMENT: 8-timestep non-overlapping patches, fixed to 512-length input with left-padding.

**Why it matters**: A 1-second patch at 20Hz has 20 timesteps; at 100Hz it would have 100 timesteps. After interpolation, both become 64-step representations. During training, patch size augmentation randomly varies the patch duration, forcing the model to learn representations that are robust to temporal resolution changes. This is critical for a foundation model that should work across different sensor hardware.

### 6. Memory Bank for Effective Negative Mining

**What TSFM does**: A MoCo-style FIFO queue stores 256 recent (IMU embedding, text embedding) pairs. Each training step computes contrastive loss against the current batch (32 samples) plus the queue, giving 288 effective negatives without extra forward passes.

**What baselines do**: CrossHAR uses standard in-batch NT-Xent (negatives = batch size - 1). LanHAR uses in-batch CLIP loss. LiMU-BERT/MOMENT have no contrastive loss.

**Why it matters**: With 87 activity classes and soft targets, having more negatives improves the quality of the contrastive signal. The memory bank provides 8x more negatives than the batch alone, stabilizing training and improving the discriminative quality of the embedding space — especially for rare activity classes that may only appear a few times per epoch.

### 7. Multi-Query Attention Pooling

**What TSFM does**: The `SemanticAlignmentHead` uses two stages of multi-query attention pooling:
1. **Cross-channel fusion**: 4 learnable queries attend across C channels → single vector per patch
2. **Temporal pooling**: 4 learnable queries attend across P patches → single vector per window

Each stage uses cross-attention followed by self-attention among the queries, then concatenation and projection.

**What baselines do**: LiMU-BERT uses last-timestep selection for GRU. MOMENT uses mean pooling across patches. CrossHAR's Transformer_ft uses mean pooling across time. LanHAR uses temporal summation (`.sum(dim=1)`).

**Why it matters**: Multi-query pooling lets the model learn *what to attend to* when summarizing a window. Different queries can specialize — one might focus on high-energy motion segments while another captures postural context. Mean/sum pooling treats all timesteps equally, which dilutes discriminative information with uninformative segments (e.g., transition periods between activities).

---

## Why TSFM Outperforms Each Baseline

### vs. LiMU-BERT (TSFM wins across all metrics)

| TSFM Advantage | LiMU-BERT Limitation |
|---------------|---------------------|
| Text alignment enables genuine zero-shot (no classifier needed) | Requires training a GRU classifier on all training data; can only predict training labels |
| 384-dim embeddings with rich cross-channel features | 72-dim embeddings from a very small encoder (~62K params) |
| Dual-branch attention captures inter-channel correlations | Single temporal attention over flat 6→72 projected vectors |
| Variable patch tokenization handles different resolutions | Fixed per-timestep tokenization, fixed to 20Hz |
| Contrastive + reconstruction pretraining | Reconstruction-only pretraining |
| Parameter-unique layers (each layer can specialize) | Parameter-shared layers (all 4 passes use identical weights) |

LiMU-BERT's core weakness is its extreme compactness (~62K parameters with parameter sharing). While this makes it deployable on mobile devices, it severely limits representational capacity. The 72-dim embedding simply cannot capture the same complexity as TSFM's 384-dim embeddings computed through 4 unique dual-branch layers. LiMU-BERT's lack of text alignment means it cannot do genuine zero-shot — it requires a separately trained classifier that can only predict labels seen during training.

### vs. MOMENT (TSFM wins zero-shot; MOMENT wins 1% supervised; TSFM wins 10%)

| TSFM Advantage | MOMENT Advantage |
|---------------|-----------------|
| Cross-channel attention learns sensor correlations natively | 341M parameters provide massive representational capacity |
| Text alignment enables genuine zero-shot | 6144-dim embeddings give more capacity for downstream classifiers |
| Compact model (~5M params), fast inference | Pretrained on 13 diverse time-series domains (broad temporal pattern knowledge) |
| No wasted computation on padding | N/A |
| Soft contrastive targets handle synonym labels | N/A |

| TSFM Limitation | MOMENT Limitation |
|----------------|------------------|
| Pretrained only on HAR data (10 datasets) | No cross-channel interaction (each channel processed independently) |
| Smaller capacity may limit supervised ceiling | 76.6% of computation wasted on left-padding (120/512 timesteps used) |
| N/A | No text alignment — requires trained SVM/linear for zero-shot |
| N/A | RevIN normalization destroys absolute gravity signal (useful for posture) |
| N/A | 8-timestep patches not optimized for IMU frequencies |

MOMENT's advantage at 1% supervised comes from its sheer capacity (341M parameters, 6144-dim embeddings) — with very few labeled samples, the high-dimensional embedding gives the linear head more features to work with. But TSFM catches up and surpasses MOMENT at 10% supervised, suggesting that TSFM's HAR-specific inductive biases (cross-channel attention, IMU-aware tokenization) become more valuable once there's enough data to leverage them. In zero-shot, TSFM's text alignment is a decisive advantage — MOMENT's SVM-based "zero-shot" is fundamentally a supervised classifier trained on training data.

### vs. CrossHAR (TSFM wins across all metrics)

| TSFM Advantage | CrossHAR Limitation |
|---------------|---------------------|
| Text alignment for genuine zero-shot | Requires Transformer_ft classifier; can only predict training labels |
| 384-dim vs 72-dim embeddings | Extremely small encoder (~57K params) |
| 4 unique dual-branch layers | 1 shared Transformer layer |
| Soft contrastive targets | Standard NT-Xent with hard negatives only |
| Multi-query attention pooling | Mean pooling over time |

CrossHAR's hierarchical pretraining (reconstruction → contrastive) is a sound design, and its channel permutation augmentation helps with cross-device generalization. But the model is simply too small (57K parameters, 1 layer) to compete with TSFM's richer architecture. CrossHAR is notably competitive at 10% supervised (80.6% vs TSFM's 83.2%), likely because its contrastive pretraining objective produces good features for the augmented Transformer_ft classifier to exploit.

### vs. LanHAR (TSFM wins across all metrics)

| TSFM Advantage | LanHAR Limitation |
|---------------|-------------------|
| Frozen, pre-trained SentenceBERT (broad language coverage) | SciBERT fine-tuned from scratch on tiny HAR text data (~2K samples) |
| Learnable attention pooling for label embeddings | Hand-crafted text prototype snippets; margin-based top-K selection |
| Dual-branch cross-channel attention | Single temporal attention, no cross-channel |
| MAE pretraining provides strong initialization | Sensor encoder trained from scratch (no pretraining) |
| Soft contrastive targets across 87 labels | Hard CLIP targets in Stage 2; multi-positive only in Stage 1 |
| 10-dataset training with group-balanced sampling | Originally designed for 4-class single-source-to-single-target transfer |

LanHAR is the most architecturally similar baseline to TSFM — both are CLIP-style sensor-text alignment models. The performance gap comes from three factors:
1. **Text encoder quality**: TSFM uses SentenceBERT (pre-trained on 1B+ sentence pairs) and keeps it frozen. LanHAR fine-tunes SciBERT (a scientific-text BERT) on ~2K HAR samples, which can overfit and lose general language understanding.
2. **Sensor encoder initialization**: TSFM's sensor encoder starts from MAE pretraining (learned temporal patterns from 10 HAR datasets). LanHAR's sensor encoder starts from random initialization.
3. **Scale of contrastive training**: TSFM trains with 87 labels, soft targets, and a 256-entry memory bank across 10 datasets. LanHAR was designed for 4 classes from a single source dataset.

---

## Limitations and Tradeoffs

### 1. HAR-Specific Pretraining (Not General-Purpose)

TSFM is pretrained exclusively on 10 HAR datasets. Unlike MOMENT (pretrained on 13 diverse time-series domains), TSFM may not generalize well to non-HAR time-series tasks (e.g., ECG classification, weather forecasting, anomaly detection). This is a deliberate design choice — HAR-specific inductive biases (cross-channel attention, IMU-aware tokenization) trade generality for HAR performance.

**Impact**: TSFM's VTT-ConIoT results (25.6% at 10% supervised) are substantially worse than MOMENT's (38.6%), suggesting that when the target domain is far from training data, MOMENT's broader pretraining provides more useful temporal pattern knowledge.

### 2. Larger Than LiMU-BERT/CrossHAR, Not Edge-Deployable

TSFM (~5M parameters) is ~80x larger than LiMU-BERT (~62K) and CrossHAR (~57K). While still much smaller than MOMENT (~341M), TSFM is not suitable for on-device deployment on resource-constrained wearables. LiMU-BERT and CrossHAR were explicitly designed for mobile/edge inference.

### 3. Requires Text Encoder at Inference

Zero-shot inference requires running SentenceBERT to encode label text. This adds latency and memory overhead compared to classifier-based models where the classifier is a small GRU/Transformer_ft. For applications where the label set is fixed, label embeddings can be pre-computed and cached, but the text encoder must be available for truly open-vocabulary scenarios.

### 4. 384-Dim Embedding May Limit Supervised Ceiling

TSFM's 384-dim embeddings are smaller than MOMENT's 6144-dim. At 1% supervised, MOMENT's higher-dimensional embedding gives a downstream linear head more features to work with, contributing to MOMENT's advantage (71.5% vs 68.2% average accuracy). The 384-dim choice was made to match SentenceBERT's output dimension for direct cosine similarity — a design choice that optimizes for zero-shot at the potential cost of supervised capacity.

### 5. Soft Targets Require Diverse Training Labels

The soft target mechanism relies on text similarity between labels to define the soft distribution. With only a few dissimilar labels (e.g., "walking" vs "sitting"), the soft targets collapse to near-hard targets. TSFM benefits from the 87-label training vocabulary where many labels have gradations of similarity. For tasks with few, highly distinct classes, this advantage diminishes.

### 6. Two-Stage Training Complexity

TSFM requires two training stages (MAE pretraining → semantic alignment), each with their own hyperparameters. CrossHAR similarly uses hierarchical training, but LiMU-BERT and MOMENT use single-stage pretraining. The two-stage approach adds engineering complexity and introduces a dependency on the pretraining quality.

### 7. No Gravity Alignment or Physics-Based Preprocessing

Unlike LanHAR, TSFM does not perform gravity alignment or physics-based feature extraction. TSFM relies on its cross-channel attention to implicitly learn orientation-dependent patterns. This works well when training data covers diverse orientations, but may be less robust than explicit gravity alignment for highly orientation-dependent activities (e.g., distinguishing sitting from standing, where the gravity vector's projection onto accelerometer axes is the primary discriminator).

### 8. Patch Size as a Metadata Decision

TSFM uses a fixed 1.0s patch size at evaluation, which is a metadata-only decision (no test-time tuning). While sensitivity analysis shows this is robust (within 1.1% of best for 3/4 datasets), the optimal patch size does vary by dataset. A learned or adaptive patch size could potentially improve results on specific datasets, but would compromise the zero-shot nature of the evaluation.

---

## Summary: TSFM's Core Thesis

TSFM's performance advantage comes from a design philosophy that is fundamentally different from the baselines:

**LiMU-BERT/CrossHAR**: Small, reconstruction-only models that learn temporal patterns within a single sensor stream. They have no mechanism to connect sensor patterns to semantic meaning (text), and their small capacity limits representational quality.

**MOMENT**: A massive, general-purpose model that learns temporal patterns across many domains. Powerful features, but no IMU-specific inductive biases (no cross-channel attention, no semantic grounding), and no ability to leverage activity labels for zero-shot transfer.

**LanHAR**: The right idea (sensor-text alignment), but limited by from-scratch training of both the text understanding (SciBERT fine-tuning) and sensor understanding (random-init sensor encoder) on tiny datasets (~2K samples, 4 classes).

**TSFM**: Combines the best elements — HAR-specific architecture (dual-branch cross-channel attention), strong text grounding (frozen SentenceBERT with learnable pooling), robust contrastive learning (soft targets + memory bank), and flexible tokenization (variable patch sizes). The result is a model that produces embeddings which are simultaneously discriminative (good for supervised) and semantically grounded (good for zero-shot).

The key insight is that **text alignment with proper soft targets makes the embedding space both semantically meaningful and metrically useful** — activities that are semantically similar end up nearby, and activities that are dissimilar end up far apart. This property enables zero-shot transfer (novel labels can be placed in the space via text encoding) while also providing a strong initialization for supervised fine-tuning (the embedding space already has meaningful structure).
