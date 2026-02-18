# Semantic Alignment Experiments

This document describes the experimental design, goals, and results for the IMU-to-text semantic alignment model.

---

## 1. Experiment Goals

### Primary Objective
Train a model that maps IMU sensor data to a shared embedding space with natural language activity labels, enabling:
1. **Zero-shot classification**: Classify activities by comparing IMU embeddings to text embeddings
2. **Cross-dataset generalization**: Train on multiple datasets, generalize to unseen datasets
3. **Semantic understanding**: Model understands that "walking" ≈ "strolling" ≈ "ambulating"

### Research Questions
1. Does learnable attention pooling for text encoding improve over mean pooling?
2. Does perceiver-style multi-query attention improve sensor fusion and temporal pooling?
3. Can soft targets handle label augmentation effectively?
4. Does the model generalize to unseen datasets (zero-shot)?

---

## 2. Experimental Setup

### 2.1 Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Micro-Batch Size | 32 | Per-step batch |
| Accumulation Steps | 16 | Gradient accumulation |
| **Effective Batch Size** | **512** | 32 × 16 |
| Learning Rate | 1e-4 | With warmup + cosine decay |
| Warmup Epochs | 3 | Linear warmup |
| Total Epochs | 100 | Current training run |
| Optimizer | AdamW | weight_decay=1e-5 |
| Precision | Mixed (fp16) | GradScaler on CUDA |
| Embedding Dim | 384 | Matches SentenceBERT |
| Text Encoder | all-MiniLM-L6-v2 | Frozen, 22M params |

### 2.2 Model Architecture Summary

| Component | Trainable | Parameters | Notes |
|-----------|-----------|------------|-------|
| IMUEncoder (CNN + Transformer) | ✓ | ~9.5M | Channel-independent processing |
| SemanticAlignmentHead | ✓ | ~8.7M | Perceiver-style pooling |
| LearnableLabelBank | ✓ | ~1.3M | 4 learnable queries |
| TokenTextEncoder (SBERT) | ✗ Frozen | 22M | all-MiniLM-L6-v2 |
| Logit Scale (temperature) | ✓ | 1 | CLIP-style learnable |

**Key Design Choices**:
- Channel-independent CNN (handles 3-45 channels)
- Perceiver-style multi-query attention for fusion and pooling
- Frozen text encoder with learnable attention pooling
- 384-dim shared embedding space

### 2.3 Datasets

**Training Datasets** (10 datasets):

| Dataset | Activities | Channels | Sampling Rate | Patch Size |
|---------|------------|----------|---------------|------------|
| UCI-HAR | 6 | 9 | 50 Hz | 1.0 sec |
| HHAR | 6 | 6 | 50 Hz | 1.0 sec |
| MHEALTH | 12 | 23 | 50 Hz | 1.5 sec |
| PAMAP2 | 12 | 51 | 100 Hz | 2.0 sec |
| WISDM | 18 | 12 | 20 Hz | 1.5 sec |
| UniMiB-SHAR | 17 | 3 | 50 Hz | 1.0 sec |
| DSADS | 19 | 9 | 25 Hz | 2.0 sec |
| HAPT | 12 | 6 | 50 Hz | 1.25 sec |
| KU-HAR | 17 | 6 | 100 Hz | 1.5 sec |
| RecGym | 11 | 6 | 20 Hz | 1.5 sec |

**Zero-shot datasets** (excluded from training):
- **MotionSense**: 6 activities, smartphone IMU
- **RealWorld HAR**: 8 activities, accelerometer only
- **MobiAct**: 13 activities including falls (novel categories)
- **VTT-ConIoT**: 16 activities, industrial IoT context

### 2.3 Loss Function

**InfoNCE with Soft Targets**:
- Symmetric contrastive loss (IMU→Text + Text→IMU)
- Learnable temperature: initialized to 1/0.07 ≈ 14.3, clamped to [1, 50]
- Soft targets: text-to-text similarity provides supervision for synonyms
- Z-score normalization amplifies differences (SentenceBERT gives 0.4-0.9 for all activities)
- Soft target weight: 1.0 (pure soft targets)

**Memory Bank (MoCo-style)**:
- Queue size: 256 embeddings
- Both IMU and text embeddings cached
- Provides 32 + 256 = 288 negatives per step
- Text embeddings have minimal staleness (frozen backbone, learnable pooling evolves slowly)

### 2.4 Evaluation Protocol

**Validation Set**: 15% of training data (stratified by session/subject)
- Evaluated every epoch

**Metrics**:
1. **Group-Aware Accuracy**: Top-1 prediction correct if in same semantic group as ground truth
2. **Mean Reciprocal Rank (MRR)**: Average of 1/rank for correct group prediction
3. **Positive/Negative Similarity**: Cosine similarity of matched vs. unmatched pairs
4. **Zero-Shot Accuracy**: Performance on unseen datasets (MotionSense, RealWorld, MobiAct, VTT-ConIoT)

**Label Groups** (handles synonyms):
```
walking: [walking, nordic_walking]
ascending_stairs: [ascending_stairs, climbing_stairs, going_up_stairs, walking_upstairs]
running: [running, jogging]
sitting: [sitting, sitting_down]
lying: [lying, laying, reclining]
... (44 groups total, 137 labels)
```

---

## 3. Results

**Status**: 10-dataset model trained (100 epochs, best checkpoint at epoch 96).
Baseline comparison results for all 5 models are available in `docs/baselines/RESULTS.md`.

---

## 4. Ablation Studies

### 4.1 Learnable Label Bank vs. Frozen (FREEZE_LABEL_BANK)

**Purpose**: Test whether learnable attention pooling for text encoding contributes to performance.

**Configuration**:
- `FREEZE_LABEL_BANK = False` (default): LabelAttentionPooling is trainable
- `FREEZE_LABEL_BANK = True` (ablation): LabelAttentionPooling frozen, uses initial random queries

**What's being tested**:
- Learnable pooling allows focusing on discriminative tokens in label text
- Example: For "person walking upstairs", attention can focus on "upstairs" vs. "walking"

**Status**: ✅ Implemented (ready to run)

### 4.2 Learnable Pooling vs. Mean Pooling (USE_MEAN_POOLING)

**Purpose**: Compare learnable attention pooling against simple mean pooling.

**Configuration**:
- `USE_MEAN_POOLING = False` (default): Use LabelAttentionPooling with learnable queries
- `USE_MEAN_POOLING = True` (ablation): Use mean of token embeddings (standard SentenceBERT)

**What's being tested**:
- Does learning which tokens to attend to help classification?
- Mean pooling is the standard approach in sentence transformers

**Status**: ✅ Implemented
- Checkpoint `20260110_104541` trained with `USE_MEAN_POOLING = True` (60 epochs)
- Results pending comparison with baseline

### 4.3 Ablation Matrix

| Experiment | Label Bank | Pooling | Tests |
|------------|------------|---------|-------|
| Baseline | Trainable | Learned Attention | Full system |
| Ablation A | **Frozen** | Learned Attention | Value of learnable text encoding |
| Ablation B | Trainable | **Mean Pooling** | Value of attention vs. mean |
| Ablation C | **Frozen** | **Mean Pooling** | Pure perceiver attention contribution |

---

## 5. Key Findings

### 5.1 What Works

1. **Perceiver-style attention pooling**: Multi-query attention with self-attention between queries provides stable, rich representations for both sensor fusion and temporal pooling.

2. **Soft targets for label augmentation**: Essential when using synonym-based augmentation. Without soft targets, semantically equivalent labels in the same batch create contradictory gradients.

3. **Memory bank**: Provides 256 additional negatives per step without extra GPU memory. Text embeddings have minimal staleness (frozen backbone with slowly-evolving learned pooling).

4. **Channel-independent processing**: Better cross-dataset generalization than joint channel processing.

### 5.2 Challenges

1. **Zero-shot performance**: Challenging — novel activities (falls, vehicle entry) are poorly recognized.

2. **WISDM dataset**: Hardest training dataset — may be due to low sampling rate (20 Hz) or activity diversity (18 activities).

3. **Label confusion**: Similar activities (sitting vs. standing, walking vs. nordic_walking) are sometimes confused.

---

## 6. Training Curves

Training typically shows:
- **Epochs 1-3**: Warmup phase, learning rate ramps up linearly
- **Epochs 3-20**: Rapid improvement, loss decreases significantly
- **Epochs 20-40**: Steepest improvement in accuracy
- **Epochs 40-60**: Gradual improvement, accuracy plateaus ~79%
- **Epochs 60+**: Minor improvements, risk of overfitting

Key observations:
- Validation loss and accuracy correlate well (no train/val gap)
- Similarity gap (pos - neg) increases steadily, indicating better discrimination
- Temperature (logit_scale) stabilizes around 3.5-4.0

---

## 7. Reproducibility

### 7.1 Checkpoint Structure

Each training run saves to `training_output/semantic_alignment/{timestamp}/`:
```
epoch_XX.pt                 # Model checkpoint
hyperparameters.json        # Full configuration
training_plots.png          # Loss and accuracy curves
final_embedding_space.png   # t-SNE visualization
```

### 7.2 Checkpoint Contents

```python
{
    'epoch': int,
    'model_state_dict': ...,
    'label_bank_state_dict': ...,  # LabelAttentionPooling weights
    'criterion_state_dict': ...,    # Learnable temperature
    'optimizer_state_dict': ...,
    'scheduler_state_dict': ...,
    'memory_bank_state_dict': ...,
    'train_metrics': {...},
    'val_metrics': {...},
    'hyperparameters': {...}
}
```

### 7.3 Evaluation Scripts

Baseline evaluations are in `val_scripts/human_activity_recognition/evaluate_*.py`.
Each script uses `PATCH_SIZE_PER_DATASET` to match training exactly.

---

## 8. Future Experiments

### 8.1 Planned

1. **FREEZE_LABEL_BANK ablation**: Quantify contribution of learnable text encoding
2. **USE_MEAN_POOLING ablation**: Compare attention vs. mean pooling
3. **Longer training**: Train to 100+ epochs to see if accuracy improves further
4. **Different text encoder**: Try larger models (all-mpnet-base-v2)

### 8.2 Potential Improvements

1. **Domain adaptation for zero-shot**: Add unsupervised loss on target domain
2. **Contrastive channel descriptions**: Learn better channel representations
3. **Multi-scale patching**: Variable patch sizes for different activities
4. **Hard negative mining**: Sample difficult negatives from memory bank

---

## 9. Code Pointers

| Task | File | Key Function/Class |
|------|------|-------------------|
| Training | `semantic_alignment_train.py` | `main()`, `train_epoch()`, `validate()` |
| Model | `semantic_alignment.py` | `SemanticAlignmentHead`, `MultiQueryAttention` |
| Text encoding | `token_text_encoder.py` | `LearnableLabelBank`, `LabelAttentionPooling` |
| Loss | `semantic_loss.py` | `SemanticAlignmentLoss` |
| Evaluation | `val_scripts/.../evaluate_tsfm.py` | `evaluate_supervised_finetune()` |
| Data loading | `multi_dataset_loader.py` | `IMUPretrainingDataset` |

---

## 10. Checkpoints Reference

| Checkpoint | Date | Configuration | Notes |
|------------|------|---------------|-------|
| `20260107_102025` | 2026-01-07 | 6-dataset baseline (learnable attention) | Old, superseded |
| `20260216_225955` | 2026-02-16 | 10-dataset model, NUM_PROTOTYPES=1 | Current (training in progress) |
