# Semantic Alignment for Human Activity Recognition from IMU Data

## Overview

This document describes a contrastive learning framework for aligning IMU (Inertial Measurement Unit) sensor embeddings with natural language descriptions in a shared semantic space. The model learns to map raw accelerometer/gyroscope data to text embeddings that describe the activity being performed, enabling zero-shot classification and cross-dataset generalization.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          HIGH-LEVEL ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    IMU Data                                Text Labels                       │
│   (T×C sensor                             ("walking",                        │
│    readings)                              "running", ...)                    │
│        │                                       │                             │
│        ▼                                       ▼                             │
│  ┌──────────────┐                    ┌─────────────────┐                    │
│  │  Preprocess  │                    │  TokenTextEncoder│                   │
│  │  + Patching  │                    │  (frozen SBERT)  │                   │
│  └──────────────┘                    └─────────────────┘                    │
│        │                                       │                             │
│        ▼                                       ▼                             │
│  ┌──────────────────────────┐       ┌─────────────────────┐                │
│  │  IMUActivityRecognition  │       │ LabelAttentionPooling│               │
│  │       Encoder            │       │   (learnable)        │               │
│  │  (CNN + Transformer)     │       └─────────────────────┘                │
│  └──────────────────────────┘                 │                             │
│        │                                       │                             │
│        ▼                                       ▼                             │
│  ┌──────────────────────────┐       ┌────────────────┐                     │
│  │  ChannelTextFusion       │       │ Text Embedding │                     │
│  │  (cross-modal attention) │       │    (384-dim)   │                     │
│  └──────────────────────────┘       └────────────────┘                     │
│        │                                       │                             │
│        ▼                                       │                             │
│  ┌──────────────────────────┐                 │                             │
│  │  SemanticAlignmentHead   │                 │                             │
│  │  (perceiver-style pool)  │                 │                             │
│  └──────────────────────────┘                 │                             │
│        │                                       │                             │
│        ▼                                       ▼                             │
│  ┌────────────────┐                 ┌────────────────┐                     │
│  │ IMU Embedding  │◄───────────────►│ Text Embedding │                     │
│  │   (384-dim)    │   InfoNCE Loss  │    (384-dim)   │                     │
│  └────────────────┘                 └────────────────┘                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. IMU Encoder Architecture

### 1.1 Input Preprocessing

Raw sensor data undergoes several preprocessing steps before entering the model:

**Patching**: Time series data is divided into fixed-duration windows (patches).
- Default patch duration: 2 seconds (dataset-specific: 1-2 sec)
- Patches are interpolated to a fixed 64 timesteps (target_patch_size)
- This enables handling of variable sampling rates (20-100 Hz across datasets)

**Normalization**: Z-score normalization per channel
```
x_normalized = (x - mean) / std
```

**Output shape**: `(batch, num_patches, 64, num_channels)`

### 1.2 FixedPatchCNN (Feature Extraction)

A channel-independent 1D CNN extracts temporal features from each patch.

```
┌─────────────────────────────────────────────────────────────┐
│                    FixedPatchCNN                             │
├─────────────────────────────────────────────────────────────┤
│  Input: (batch, patches, 64, channels)                       │
│                        │                                     │
│                        ▼                                     │
│    ┌───────────────────────────────────────┐                │
│    │  Channel-Independent 1D Conv Layers   │                │
│    │  - Conv1d(1 → 64, kernel=5)           │                │
│    │  - BatchNorm + GELU + Dropout         │                │
│    │  - Conv1d(64 → 128, kernel=5)         │                │
│    │  - BatchNorm + GELU + Dropout         │                │
│    └───────────────────────────────────────┘                │
│                        │                                     │
│                        ▼                                     │
│    ┌───────────────────────────────────────┐                │
│    │  Global Average Pooling               │                │
│    │  (temporal_dim → 1)                   │                │
│    └───────────────────────────────────────┘                │
│                        │                                     │
│                        ▼                                     │
│    ┌───────────────────────────────────────┐                │
│    │  Linear Projection (128 → d_model)    │                │
│    └───────────────────────────────────────┘                │
│                        │                                     │
│  Output: (batch, patches, channels, d_model)                 │
└─────────────────────────────────────────────────────────────┘
```

**Key Design Decision**: Channel-independent processing
- Each sensor channel (acc_x, acc_y, gyro_z, etc.) is processed independently
- Enables variable-channel support (3-51 channels across datasets)
- Cross-channel fusion happens later via attention

### 1.3 Positional Encoding

Positional encodings are added to CNN output:

1. **Temporal Position**: Encodes patch position within the sequence
   - Sinusoidal encoding with learnable scale factor: `(max_patches, d_model)`

2. **Channel Semantics** (optional): Encodes sensor meaning via SentenceBERT
   - Channel descriptions: "accelerometer x-axis", "gyroscope z-axis", etc.
   - Frozen SentenceBERT embeddings → learnable projection → `(1, d_model)`

### 1.4 IMUTransformer (Temporal Modeling)

Standard transformer encoder for temporal attention across patches.

```python
IMUTransformer(
    d_model=384,
    num_temporal_layers=4,
    num_heads=8,
    dim_feedforward=1536,  # 4x d_model
    dropout=0.1
)
```

**Note**: Cross-channel attention is enabled during training (`use_cross_channel=True`). It allows information exchange between channels at each transformer layer.

---

## 2. Text Encoding Pipeline

### 2.1 TokenTextEncoder (Frozen Backbone)

Uses `all-MiniLM-L6-v2` from Sentence-BERT:
- 384 dimensions
- 22M parameters (lightweight)
- Max 256 tokens

```python
class TokenTextEncoder(nn.Module):
    """Frozen text encoder outputting token-level embeddings."""

    def encode(self, texts: List[str], device) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            token_embeddings: (batch, seq_len, 384)
            attention_mask: (batch, seq_len) bool
        """
```

**Key Design Decision**: Token-level outputs, not pooled
- Standard SentenceBERT pools to single vector (loses information)
- We keep token sequences for learned attention pooling
- Frozen backbone (cache-friendly, no training needed)

### 2.2 LabelAttentionPooling (Learnable Component)

Perceiver-style attention pooling to refine label representations.

```
┌─────────────────────────────────────────────────────────────┐
│                 LabelAttentionPooling                        │
├─────────────────────────────────────────────────────────────┤
│  Token embeddings: (batch, seq_len, 384)                     │
│                        │                                     │
│                        ▼                                     │
│    ┌───────────────────────────────────────┐                │
│    │  Learnable Query Tokens (4 queries)   │                │
│    │  queries: (4, 384) Parameter          │                │
│    └───────────────────────────────────────┘                │
│                        │                                     │
│                        ▼                                     │
│    ┌───────────────────────────────────────┐                │
│    │  Cross-Attention                      │                │
│    │  Q: queries, K/V: token embeddings    │                │
│    │  4 heads, dropout=0.1                 │                │
│    └───────────────────────────────────────┘                │
│                        │                                     │
│                        ▼                                     │
│    ┌───────────────────────────────────────┐                │
│    │  Output Projection                    │                │
│    │  (4 × 384) → 384 → 384               │                │
│    │  + residual from query mean           │                │
│    └───────────────────────────────────────┘                │
│                        │                                     │
│                        ▼                                     │
│    ┌───────────────────────────────────────┐                │
│    │  L2 Normalize                         │                │
│    └───────────────────────────────────────┘                │
│                        │                                     │
│  Output: (batch, 384) - refined label embedding              │
└─────────────────────────────────────────────────────────────┘
```

**Key Design Decision**: Multi-query attention vs. mean pooling
- Default SentenceBERT uses mean pooling across tokens
- Our learned attention allows focusing on discriminative tokens
- Ablation: `USE_MEAN_POOLING` flag for comparison

### 2.3 MultiPrototypeLabelPooling (K>1 Prototypes)

When `NUM_PROTOTYPES > 1`, replaces `LabelAttentionPooling` with K independent query sets sharing one cross-attention layer:

```
┌─────────────────────────────────────────────────────────────┐
│            MultiPrototypeLabelPooling (K=3)                  │
├─────────────────────────────────────────────────────────────┤
│  Token embeddings: (batch, seq_len, 384)                     │
│                        │                                     │
│                        ▼                                     │
│    ┌───────────────────────────────────────┐                │
│    │  K Learnable Query Sets               │                │
│    │  queries: (K, num_queries, 384)       │                │
│    │  - K=3 independent query groups       │                │
│    │  - Shared cross-attention (efficient) │                │
│    │  - K independent output projections   │                │
│    └───────────────────────────────────────┘                │
│                        │                                     │
│                        ▼                                     │
│    ┌───────────────────────────────────────┐                │
│    │  Per-Prototype Projection + L2 Norm   │                │
│    └───────────────────────────────────────┘                │
│                        │                                     │
│  Output: (batch, K, 384) - K prototype embeddings            │
└─────────────────────────────────────────────────────────────┘
```

**Why multiple prototypes?**
- "Walking" covers fast/slow walks, different sensor placements, different people
- A single prototype embedding collapses this diversity
- K=3 prototypes let the model represent intra-class variation
- Backward compatible: K=1 falls back to `LabelAttentionPooling`

---

## 3. Cross-Modal Fusion (ChannelTextFusion)

Fuses sensor tokens with their channel descriptions via cross-attention.

```
┌─────────────────────────────────────────────────────────────┐
│                  ChannelTextFusion                           │
├─────────────────────────────────────────────────────────────┤
│  Sensor tokens: (batch, patches, channels, d_model)          │
│  Channel text tokens: (channels, seq_len, d_model)           │
│                                                              │
│  Step 1: Pool each channel's text → (channels, d_model)      │
│    ┌───────────────────────────────────────┐                │
│    │  For each channel:                    │                │
│    │  - queries attend to text tokens      │                │
│    │  - O(C) attention ops, not O(B×P×C)   │                │
│    └───────────────────────────────────────┘                │
│                                                              │
│  Step 2: Broadcast & Gate                                    │
│    ┌───────────────────────────────────────┐                │
│    │  gate = σ(concat(sensor, text))       │                │
│    │  fused = sensor + gate * text         │                │
│    └───────────────────────────────────────┘                │
│                                                              │
│  Output: (batch, patches, channels, d_model)                 │
└─────────────────────────────────────────────────────────────┘
```

**Key Design Decision**: Efficient broadcast fusion
- Problem: Per-patch, per-channel attention is expensive O(B×P×C)
- Solution: Pool text once per channel O(C), broadcast to all patches
- Gated fusion allows sensor tokens to control text incorporation

---

## 4. Semantic Alignment Head

Converts encoder output to a single embedding aligned with text space.

```
┌─────────────────────────────────────────────────────────────┐
│                 SemanticAlignmentHead                        │
├─────────────────────────────────────────────────────────────┤
│  Encoder output: (batch, patches, channels, d_model=384)     │
│                           │                                  │
│                           ▼                                  │
│  ┌────────────────────────────────────────────────┐         │
│  │  CrossChannelFusion (perceiver-style)          │         │
│  │  - 4 learnable queries                         │         │
│  │  - cross-attn to channels                      │         │
│  │  - self-attn among queries                     │         │
│  │  - project: (4 × d_model) → d_model_fused     │         │
│  └────────────────────────────────────────────────┘         │
│                           │                                  │
│  Output: (batch, patches, d_model_fused=384)                 │
│                           │                                  │
│                           ▼                                  │
│  ┌────────────────────────────────────────────────┐         │
│  │  TemporalAttention                             │         │
│  │  - 2 transformer layers                        │         │
│  │  - self-attention across patches               │         │
│  └────────────────────────────────────────────────┘         │
│                           │                                  │
│  Output: (batch, patches, d_model_fused)                     │
│                           │                                  │
│                           ▼                                  │
│  ┌────────────────────────────────────────────────┐         │
│  │  MultiQueryPooling (perceiver-style)           │         │
│  │  - 4 learnable queries                         │         │
│  │  - pools patches → single vector               │         │
│  └────────────────────────────────────────────────┘         │
│                           │                                  │
│  Output: (batch, d_model_fused)                              │
│                           │                                  │
│                           ▼                                  │
│  ┌────────────────────────────────────────────────┐         │
│  │  ProjectionHead                                │         │
│  │  MLP: d_model_fused → 2×d_model_fused → 384   │         │
│  │  + L2 normalization                            │         │
│  └────────────────────────────────────────────────┘         │
│                           │                                  │
│  Output: (batch, 384) - IMU semantic embedding               │
└─────────────────────────────────────────────────────────────┘
```

### 4.1 MultiQueryAttention (Core Building Block)

Used in both CrossChannelFusion and MultiQueryPooling.

```python
class MultiQueryAttention(nn.Module):
    """
    Multi-query attention for sequence-to-vector transformation.

    Reference: Set Transformer (ICML 2019), Perceiver, PMA
    """

    def __init__(
        self,
        d_model: int,
        num_queries: int = 4,      # Number of learnable queries
        num_heads: int = 8,
        use_self_attention: bool = True  # Queries coordinate
    ):
        # Learnable queries (small init prevents collapse)
        self.queries = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
```

**Key Design Decisions**:
1. **Multiple queries**: Each query can attend to different aspects
2. **Self-attention among queries**: Allows coordination (can be disabled for ablation)
3. **Small initialization**: Prevents representation collapse early in training
4. **Residual connection**: `output = projected + mean(queries)` for stability

---

## 5. Training Objectives

### 5.1 InfoNCE Loss with Soft Targets

The core contrastive loss aligns IMU embeddings with text embeddings.

```
┌─────────────────────────────────────────────────────────────┐
│                     InfoNCE Loss                             │
├─────────────────────────────────────────────────────────────┤
│  IMU embeddings: (B, 384)  normalized                        │
│  Text embeddings: (B, 384)  normalized                       │
│                                                              │
│  1. Compute similarity matrix:                               │
│     logits = (imu @ text.T) × exp(logit_scale)              │
│     logit_scale: learnable (initialized to 1/0.07 ≈ 14.3)  │
│                                                              │
│  2. Compute soft targets (if enabled):                       │
│     text_sim = text @ text.T                                 │
│     # Z-score normalization for adaptive calibration         │
│     text_sim = (text_sim - mean) / std / temperature        │
│     soft_targets = softmax(text_sim, dim=1)                  │
│                                                              │
│  3. Blend targets:                                           │
│     targets = (1-w) × hard_targets + w × soft_targets       │
│                                                              │
│  4. Bidirectional loss:                                      │
│     loss_i2t = KL(softmax(logits), targets)                 │
│     loss_t2i = KL(softmax(logits.T), targets)               │
│     loss = (loss_i2t + loss_t2i) / 2                        │
└─────────────────────────────────────────────────────────────┘
```

**Multi-prototype extension** (when K > 1):
- Positive: best prototype per sample via `max_k sim(imu_i, text_i_k)`
- Negatives: all K prototypes of other labels (flattened to B*K keys)
- Soft targets: label similarity via `max_{k,l} sim(proto_i_k, proto_j_l)`
- Text-to-IMU direction: uses winning prototype per sample

**Key Design Decision**: Soft targets for label augmentation
- Problem: Label augmentation creates batches with semantically equivalent labels
  - Example: "walking", "strolling", "person walking" in same batch
  - Hard targets treat these as negatives → contradictory gradients
- Solution: Soft targets weight by semantic similarity
  - Text similarity from SentenceBERT identifies synonyms
  - Z-score normalization amplifies small differences (0.4-0.9 range → -2 to +2)

### 5.2 Learnable Temperature

CLIP-style learnable temperature:

```python
self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))
# Usage: logits = similarity × exp(logit_scale).clamp(1, 50)
```

- Initialized to effective temperature τ = 0.07
- Clamped to [1, 50] range to prevent explosion
- Learned during training for optimal sharpness

### 5.3 Memory Bank (MoCo-style Queue)

Provides additional negatives without recomputation.

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory Bank                               │
├─────────────────────────────────────────────────────────────┤
│  Queue of past embeddings: (queue_size, embedding_dim)       │
│  - IMU queue: cached IMU embeddings                          │
│  - Text queue: cached text embeddings                        │
│                                                              │
│  With batch_size=16 and queue_size=256:                      │
│  - Total negatives per step: 16 + 256 = 272                  │
│  - Effective batch size: 272 (without extra GPU memory)      │
│                                                              │
│  FIFO update: oldest embeddings replaced after each step     │
│  Embeddings stored detached (no gradient through queue)      │
└─────────────────────────────────────────────────────────────┘
```

**Key Design Decision**: Minimal text staleness
- Problem: In MoCo, key encoder momentum causes stale features
- Our solution: Text from frozen SentenceBERT + learnable pooling
- Minor staleness exists (learnable pooling evolves), but with small queue (256 items, ~16 batches lag) this is acceptable (standard MoCo practice)

---

## 6. Data Loading and Label Augmentation

### 6.1 Multi-Dataset Loader

Loads from 10 HAR training datasets with consistent preprocessing:

| Dataset | Activities | Channels | Sampling Rate | Patch Size |
|---------|------------|----------|---------------|------------|
| UCI-HAR | 6 | 9 (acc+gyro+total_acc) | 50 Hz | 1.0 sec |
| HHAR | 6 | 6 (acc+gyro) | 50 Hz | 1.0 sec |
| MHEALTH | 12 | 23 (multi-location) | 50 Hz | 1.5 sec |
| PAMAP2 | 12 | 51 (multi-location) | 100 Hz | 2.0 sec |
| WISDM | 18 | 12 (phone+watch) | 20 Hz | 1.5 sec |
| UniMiB-SHAR | 17 | 3 (acc only) | 50 Hz | 1.0 sec |
| DSADS | 19 | 9 (acc+gyro+mag) | 25 Hz | 2.0 sec |
| HAPT | 12 | 6 (acc+gyro) | 50 Hz | 1.25 sec |
| KU-HAR | 17 | 6 (acc+gyro) | 100 Hz | 1.5 sec |
| RecGym | 11 | 6 (acc+gyro) | 20 Hz | 1.5 sec |

**Zero-shot test datasets** (excluded from training): MotionSense, RealWorld, MobiAct, VTT-ConIoT

**Splits**: 70% train / 15% val / 15% test (by session)

### 6.2 Label Augmentation

Each dataset has custom synonyms and templates for rich text variation.

```python
# Example: UCI-HAR
UCI_HAR_SYNONYMS = {
    "walking": ["walking", "strolling", "striding", "ambulating", "pacing"],
    "walking_upstairs": ["walking upstairs", "climbing stairs", "ascending stairs"],
    "sitting": ["sitting", "seated", "sitting down", "in a seated position"],
    ...
}

UCI_HAR_TEMPLATES = [
    "{}",           # "walking"
    "person {}",    # "person walking"
    "person is {}", # "person is walking"
    "{} activity",  # "walking activity"
    ...
]

def augment_label(label: str, dataset: str) -> str:
    """Apply random synonym + random template."""
    synonym = random.choice(SYNONYMS[dataset][label])
    template = random.choice(TEMPLATES[dataset])
    return template.format(synonym)
```

**Purpose**:
1. Creates diverse text during training (regularization)
2. Teaches model that "walking" ≈ "person strolling" (semantic equivalence)
3. Works with soft targets to handle synonyms correctly

### 6.3 Channel Descriptions

Each sensor channel has a human-readable description:

```python
channel_descriptions = [
    "accelerometer x-axis",
    "accelerometer y-axis",
    "accelerometer z-axis",
    "gyroscope x-axis",
    "gyroscope y-axis",
    "gyroscope z-axis"
]
```

Used by:
1. `IMUPositionalEncoding`: Adds semantic channel embeddings
2. `ChannelTextFusion`: Cross-attention with sensor tokens

---

## 7. Evaluation Metrics

### 7.1 Group-Aware Classification

Handles synonyms in evaluation (different datasets use different label names).

```python
LABEL_GROUPS = {
    'walking': ['walking', 'nordic_walking'],
    'ascending_stairs': ['ascending_stairs', 'climbing_stairs', 'going_up_stairs', 'walking_upstairs'],
    'descending_stairs': ['descending_stairs', 'going_down_stairs', 'walking_downstairs'],
    'running': ['running', 'jogging'],
    'sitting': ['sitting', 'sitting_down'],
    'standing': ['standing', 'standing_up'],
    'lying': ['lying', 'laying', 'reclining'],
    ...
}
```

**Group-aware accuracy**: Prediction correct if it belongs to same group as ground truth.

### 7.2 Zero-Shot Evaluation

Test on unseen dataset (e.g., MotionSense) not used during training:
1. Encode all unique labels from test set with label_bank
2. Compute IMU→Text similarity for each sample
3. Predict label with highest similarity
4. Report group-aware accuracy

---

## 8. Model Configuration Summary

### Default Hyperparameters

```python
# Encoder
D_MODEL = 384                    # Feature dimension (matches SentenceBERT)
NUM_TEMPORAL_LAYERS = 4          # Transformer layers in encoder
NUM_HEADS = 8                    # Attention heads
DIM_FEEDFORWARD = 1536           # FFN hidden dim (4x d_model)
DROPOUT = 0.1                    # Dropout rate throughout

# Semantic Head
D_MODEL_FUSED = 384              # After channel fusion (same as d_model)
NUM_SEMANTIC_TEMPORAL_LAYERS = 2 # Temporal attention in semantic head
NUM_FUSION_QUERIES = 4           # Channel fusion queries (perceiver-style)
NUM_POOL_QUERIES = 4             # Temporal pooling queries (perceiver-style)
USE_FUSION_SELF_ATTENTION = True # Queries coordinate via self-attention
USE_POOL_SELF_ATTENTION = True   # Queries coordinate via self-attention

# Text Encoding
TOKEN_TEXT_NUM_HEADS = 4         # LabelAttentionPooling heads
TOKEN_TEXT_NUM_QUERIES = 4       # Learnable queries for pooling
USE_MEAN_POOLING = False         # Ablation: use mean instead of learned
FREEZE_LABEL_BANK = False        # Ablation: freeze text encoding

# Training
BATCH_SIZE = 8                   # Micro-batch size
ACCUMULATION_STEPS = 64          # Effective batch size = 512
LEARNING_RATE = 1e-4             # AdamW learning rate
WARMUP_EPOCHS = 3                # Linear warmup epochs
TOTAL_EPOCHS = 100               # Total training epochs
WEIGHT_DECAY = 1e-5              # Low weight decay (SimCLR-style)

# Loss
TEMPERATURE = 0.07               # CLIP-style temperature (learnable)
USE_SOFT_TARGETS = True          # Essential for label augmentation
SOFT_TARGET_TEMPERATURE = 0.5    # Sharpens soft targets
SOFT_TARGET_WEIGHT = 1.0         # Pure soft targets
MEMORY_BANK_SIZE = 256           # MoCo-style queue size

# Accuracy improvements (currently disabled for debugging)
NUM_PROTOTYPES = 1               # K prototypes per class (multi-prototype disabled)
# SAMPLING_TEMPERATURE = 0.5     # Square-root balancing (disabled)
# USE_ROTATION_AUGMENTATION = True # SO(3) rotation augmentation (disabled)
```

### Model Parameter Count

| Component | Parameters |
|-----------|------------|
| IMUActivityRecognitionEncoder | ~9.5M |
| SemanticAlignmentHead | ~8.7M |
| LearnableLabelBank | ~1.3M |
| **Total trainable** | **~19.5M** |
| TokenTextEncoder (frozen) | 22M |

---

## 9. Key Innovations Summary

1. **Token-level text encoding**: Keep full token sequences from SentenceBERT, apply learnable attention pooling instead of mean pooling.

2. **Perceiver-style multi-query attention**: Used throughout for channel fusion and temporal pooling. Multiple learnable queries with self-attention enable richer representations.

3. **ChannelTextFusion**: Efficient cross-modal fusion that pools text once per channel, then broadcasts to all sensor tokens with gated incorporation.

4. **Soft targets for label augmentation**: Z-score normalized text similarity prevents contradictory gradients when batch contains semantically equivalent labels.

5. **Memory bank with minimal staleness**: Frozen text backbone ensures queue embeddings have only minor staleness from the learnable pooling layer, enabling more negatives without encoder momentum.

6. **Group-aware evaluation**: Handles label synonyms across datasets for fair cross-dataset comparison.

7. **Structured masking (Stage 1)**: Three masking strategies randomly selected per batch — random patch masking (40%), span masking with contiguous spans of length 2-4 (40%), and channel dropout dropping 30% of channels (20%). Trains the model for real-world IMU failure modes.

### Implemented but Currently Disabled

The following features are implemented in code but disabled in the current training configuration:

8. **SO(3) rotation augmentation** *(disabled)*: Random 3D rotations applied to sensor triads (acc_x/y/z, gyro_x/y/z). Same rotation matrix shared across all triads at the same body location. Generated via QR decomposition of random Gaussian matrices with proper orientation (det=+1).

9. **Multi-prototype learning** *(disabled, NUM_PROTOTYPES=1)*: K independent prototype embeddings per activity class via `MultiPrototypeLabelPooling`. Shared cross-attention with K independent output projections. With K=1, this reduces to standard single-prototype attention pooling.

10. **Temperature-based sampling** *(disabled)*: Per-sample weight `w_i = count(group_i)^(alpha-1)` with alpha=0.5 (square-root balancing). Currently uses group-balanced sampling with capped oversampling instead.

---

## 10. File Structure

```
model/
├── encoder.py              # IMUActivityRecognitionEncoder
├── feature_extractor.py    # FixedPatchCNN, ChannelIndependentCNN
├── positional_encoding.py  # IMUPositionalEncoding
├── transformer.py          # IMUTransformer, DualBranchTransformer
├── semantic_alignment.py   # SemanticAlignmentHead, MultiQueryAttention
└── token_text_encoder.py   # TokenTextEncoder, LabelAttentionPooling

training_scripts/human_activity_recognition/
├── semantic_alignment_train.py  # Main training script
├── semantic_loss.py             # InfoNCELoss, SemanticAlignmentLoss
├── losses.py                    # MaskedReconstructionLoss, PatchContrastiveLoss
└── memory_bank.py               # MoCo-style memory bank

datasets/imu_pretraining_dataset/
├── multi_dataset_loader.py  # IMUPretrainingDataset
├── label_augmentation.py    # Synonyms and templates per dataset
└── label_groups.py          # Semantic label groups for evaluation

val_scripts/human_activity_recognition/
├── evaluation_metrics.py    # Group-aware accuracy, semantic recall
├── evaluate_tsfm.py         # TSFM 4-metric baseline evaluation
├── evaluate_limubert.py     # LiMU-BERT baseline evaluation
├── evaluate_moment.py       # MOMENT baseline evaluation
├── evaluate_crosshar.py     # CrossHAR baseline evaluation
├── evaluate_lanhar.py       # LanHAR baseline evaluation
└── model_loading.py         # Shared model/label bank loading
```
