# Baseline Quick Reference

Brief descriptions of each model, how it works, and key limitations.

---

## TSFM (Ours)

**What it is**: Dual-branch Transformer with CLIP-style contrastive alignment between IMU sensor
patches and text activity labels.

**How it works**: Raw IMU signals are segmented into seconds-based patches, interpolated to fixed
64 steps, then processed by alternating temporal self-attention (how the signal evolves over time)
and cross-channel self-attention (how sensor axes correlate). A semantic alignment head pools patch
embeddings into a single 384-dim vector, trained via contrastive loss to match frozen SentenceBERT
embeddings of activity labels. Uses soft contrastive targets (semantically similar labels are
partial positives), a MoCo-style memory bank, and learnable label attention pooling.

**Zero-shot**: Cosine similarity between sensor embedding and text label embedding. No classifier needed.

**Key capabilities**:
- Handles variable sampling rates natively (seconds-based patches + interpolation)
- Uses per-dataset channel descriptions as semantic input (e.g., "Accelerometer X-axis (waist)")
- Patch size augmentation during training forces resolution-robust representations
- ~19.5M trainable parameters (encoder ~9.5M + semantic head ~8.7M + label bank ~1.3M)

---

## LiMU-BERT

**What it is**: BERT-style masked reconstruction model for IMU data. Predicts masked timesteps
from surrounding context.

**How it works**: The 120-timestep window (6s at 20Hz) is split into six 20-step sub-windows.
Each sub-window is processed by a single shared-parameter Transformer layer (looped 4x). During
pretraining, random timesteps are masked and the model reconstructs them via MSE loss. Produces
72-dim per-timestep embeddings.

**Zero-shot**: Requires training a separate GRU classifier on frozen embeddings from 10 training
datasets. Can only predict labels seen during training.

**Key limitations**:
- **Cannot handle variable sampling rates**: Learned positional embedding `nn.Embedding(120, 72)`
  is trained for exactly 120 positions at 20Hz. Different rates would change what each position
  represents, invalidating learned temporal patterns.
- Very small model (~62K parameters, 72-dim embeddings)
- Reconstruction-only pretraining (no contrastive or text alignment)
- No cross-channel attention
- Struggles severely at 1% supervised (25.6% avg)

---

## MOMENT

**What it is**: General-purpose time-series foundation model from AutonLab (CMU). Pretrained on
13 diverse domains of time-series data (not HAR-specific). Based on T5-Large.

**How it works**: Each of the 6 IMU channels is processed independently as a univariate time series.
Input is left-padded to 512 timesteps, segmented into 8-timestep non-overlapping patches (64 tokens),
and processed by a 24-layer T5 encoder (1024-dim per channel). Per-channel embeddings are mean-pooled
then concatenated to produce a 6144-dim embedding.

**Zero-shot**: Requires training a separate SVM-RBF classifier on frozen embeddings from training data.

**Key limitations**:
- **Cannot handle variable sampling rates**: Rate-agnostic design has no concept of physical time.
  Paper: *"We did not explicitly model temporal resolution."* An 8-timestep patch covers 0.4s at
  20Hz but 0.08s at 100Hz, yet the model cannot distinguish them. There is no mechanism to benefit
  from native rates.
- No cross-channel interaction (channels processed independently)
- 76.6% of computation wasted on left-padding (120/512 timesteps used)
- Very large model (~341M parameters)
- RevIN normalization destroys absolute gravity signal (useful for posture detection)

---

## CrossHAR

**What it is**: Hierarchical self-supervised model combining masked reconstruction and contrastive
learning. Builds on LiMU-BERT's architecture.

**How it works**: Single shared-parameter Transformer layer processes per-timestep tokens (120 tokens
for a 6s window at 20Hz). Pretraining has two stages: (1) masked reconstruction (same as LiMU-BERT),
then (2) contrastive learning added on top using NT-Xent loss with channel permutation augmentation.
Produces 72-dim per-timestep embeddings.

**Zero-shot**: Requires training a separate Transformer_ft classifier on frozen embeddings.

**Key limitations**:
- **Cannot handle variable sampling rates**: Same learned positional embedding constraint as
  LiMU-BERT. Code explicitly checks for `dataset_version='20_120'`.
- Very small model (~57K parameters, 72-dim embeddings)
- No text alignment
- No cross-channel attention (despite channel permutation augmentation)

---

## LanHAR

**What it is**: CLIP-style sensor-text alignment model. Conceptually similar to TSFM but trains
both the text encoder and sensor encoder from scratch during evaluation.

**How it works**: Two-stage training: (1) Fine-tune SciBERT on activity text descriptions using
CLIP + cross-entropy + triplet loss (10 epochs). (2) Train a 3-layer sensor Transformer from
random initialization to align with the fine-tuned text space via CLIP contrastive loss (50 epochs).
Uses gravity alignment preprocessing and optionally per-sample LLM-generated descriptions.

**Zero-shot**: Cosine similarity between sensor embedding and SciBERT-encoded text. Same mechanism
as TSFM but with a weaker text encoder.

**Key limitations**:
- **Trains entirely from scratch during evaluation** (~90 minutes per run). No pretrained checkpoint.
- **Sampling rate**: Paper designed for 50Hz. In our benchmark, uses 20Hz (trains from scratch so
  adapts to any rate, but cannot dynamically handle multiple rates).
- SciBERT fine-tuned on tiny HAR text corpus (~2K samples) loses general language understanding
- Sensor encoder starts from random init (no MAE pretraining like TSFM)
- Originally designed for single-source-to-single-target 4-class transfer, not 87-label cross-dataset

---

## Why Baselines Cannot Use Native Sampling Rates

The ability to handle variable sampling rates is a genuine architectural novelty of TSFM, not
something that can be trivially added to other models:

| Model | Architectural Constraint | Why It Can't Be Fixed Trivially |
|-------|-------------------------|-------------------------------|
| **LiMU-BERT** | `nn.Embedding(120, 72)` positional encoding trained at 20Hz | Changing to a different rate changes what each position means; would need complete retraining. Sinusoidal positional encoding could work but is a fundamental architecture change. |
| **CrossHAR** | Same positional embedding as LiMU-BERT | Same constraint. |
| **MOMENT** | No concept of physical time in the architecture | Adding frequency awareness would require architectural changes to the T5 backbone (e.g., conditioning on sampling rate). The model treats all sequences as abstract number sequences. |
| **LanHAR** | Per-timestep tokenization with fixed 120 tokens | Could theoretically handle different rates since it trains from scratch, but has no mechanism to adapt dynamically (no seconds-based patching, no rate conditioning). |

**TSFM's solution**: Seconds-based patch sizes + `F.interpolate` to fixed 64 steps. A 1.0s patch
has 20 timesteps at 20Hz, 50 at 50Hz, or 100 at 100Hz — all interpolated to the same 64-step
representation. This decouples temporal resolution from sampling rate at the architecture level.
