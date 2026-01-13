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
| Batch Size | 16 × 2 = 32 | Gradient accumulation |
| Learning Rate | 1e-4 | With warmup + cosine decay |
| Warmup Epochs | 15 | Linear warmup |
| Total Epochs | 200 | Early stopping if plateau |
| Optimizer | AdamW | weight_decay=0.01 |
| Precision | Mixed (fp16) | GradScaler on CUDA |

### 2.2 Datasets

**Training Datasets** (6 datasets, ~46K sessions):

| Dataset | Activities | Channels | Sampling Rate | Patch Size | Sessions |
|---------|------------|----------|---------------|------------|----------|
| UCI-HAR | 6 | 9 | 50 Hz | 1.0 sec | ~10K |
| HHAR | 6 | 6 | ~50 Hz | 1.0 sec | ~10K (limited) |
| MHEALTH | 12 | 6 | 50 Hz | 2.0 sec | ~3K |
| PAMAP2 | 12 | 27 | 100 Hz | 2.0 sec | ~4K |
| WISDM | 18 | 6 | 20 Hz | 2.0 sec | ~10K (limited) |
| UniMiB-SHAR | 9 | 3 | 50 Hz | 1.0 sec | ~10K (limited) |

**Unseen Dataset** (zero-shot evaluation):
- **MotionSense**: 6 activities, smartphone sensors, not seen during training

### 2.3 Evaluation Protocol

**Validation Set**: 15% of training data (stratified by session/subject)
- 6,955 samples across 6 datasets
- Evaluated every epoch

**Metrics**:
1. **Group-Aware Accuracy**: Top-1 prediction correct if in same semantic group as ground truth
2. **Mean Reciprocal Rank (MRR)**: Average of 1/rank for correct group prediction
3. **Positive/Negative Similarity**: Cosine similarity of matched vs. unmatched pairs
4. **Zero-Shot Accuracy**: Performance on MotionSense (unseen dataset)

**Label Groups** (handles synonyms):
```
walking: [walking, nordic_walking]
ascending_stairs: [ascending_stairs, climbing_stairs, going_up_stairs, walking_upstairs]
running: [running, jogging]
sitting: [sitting, sitting_down]
lying: [lying, laying, reclining]
... (25 groups total)
```

---

## 3. Baseline Results (Epoch 60)

### 3.1 Validation Performance

| Metric | Value |
|--------|-------|
| **Group-Aware Accuracy** | **79.48%** |
| **MRR** | **86.81%** |
| Positive Similarity | 0.8636 |
| Negative Similarity | 0.5019 |
| Similarity Gap | 0.3617 |
| Loss | 1.0313 |

### 3.2 Per-Dataset Breakdown

| Dataset | Accuracy | Notes |
|---------|----------|-------|
| UCI-HAR | 88.91% | Best performance |
| UniMiB-SHAR | 88.04% | Accelerometer only |
| HHAR | 86.58% | Good generalization |
| MHEALTH | 82.03% | More activities |
| PAMAP2 | 81.95% | Most channels (27) |
| WISDM | 54.22% | Hardest - lowest sampling rate |

### 3.3 Zero-Shot Performance (MotionSense)

| Metric | Value |
|--------|-------|
| Group-Aware Accuracy | 37.98% |
| MRR | 49.75% |
| Positive Similarity | 0.6913 |
| Negative Similarity | 0.5931 |

**Analysis**: Zero-shot performance is moderate. The model struggles with the distribution shift from training data. This is an area for improvement.

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

**Status**: Planned (not yet run)

### 4.2 Learnable Pooling vs. Mean Pooling (USE_MEAN_POOLING)

**Purpose**: Compare learnable attention pooling against simple mean pooling.

**Configuration**:
- `USE_MEAN_POOLING = False` (default): Use LabelAttentionPooling with learnable queries
- `USE_MEAN_POOLING = True` (ablation): Use mean of token embeddings (standard SentenceBERT)

**What's being tested**:
- Does learning which tokens to attend to help classification?
- Mean pooling is the standard approach in sentence transformers

**Status**: Planned (not yet run)

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

3. **Memory bank**: Provides 256 additional negatives per step without extra GPU memory. Text embeddings don't go stale because text encoder is frozen.

4. **Channel-independent processing**: Better cross-dataset generalization than joint channel processing.

### 5.2 Challenges

1. **Zero-shot performance**: 37.98% on MotionSense suggests room for improvement in out-of-distribution generalization.

2. **WISDM dataset**: 54.22% accuracy, significantly lower than other datasets. May be due to low sampling rate (20 Hz) or activity diversity (18 activities).

3. **Label confusion**: Similar activities (sitting vs. standing, walking vs. nordic_walking) are sometimes confused.

---

## 6. Training Curves

Training typically shows:
- **Epochs 1-15**: Warmup phase, loss decreases rapidly
- **Epochs 15-40**: Steepest improvement in accuracy
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

- `compare_models.py`: Full evaluation with metrics and visualizations
- `session_explorer.py`: Interactive 3D embedding visualization
- Both scripts use `PATCH_SIZE_PER_DATASET` to match training exactly

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
| Evaluation | `compare_models.py` | `compute_metrics()` |
| Data loading | `multi_dataset_loader.py` | `IMUPretrainingDataset` |

---

## 10. Changelog

| Date | Change | Result |
|------|--------|--------|
| 2026-01-07 | Initial training run | 79.48% accuracy |
| 2026-01-13 | Fixed evaluation scripts (PATCH_SIZE_PER_DATASET) | Metrics now match training |
| 2026-01-13 | Added FREEZE_LABEL_BANK ablation flag | Ready for ablation |
| 2026-01-13 | Added confusion matrix filtering for unseen datasets | Cleaner visualizations |
