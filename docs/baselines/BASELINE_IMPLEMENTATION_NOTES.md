# Baseline Implementation Notes

Per-baseline design decisions, implementation details, paper-matching considerations,
and how we ensure fairness across the comparison.

---

## Fairness Principles

Our benchmark compares 5 models on the same 4 test datasets using a unified evaluation
framework. To ensure fair comparison:

1. **Same data**: All models evaluate on identical `(N, 120, 6)` windows at 20Hz
2. **Same splits**: Identical random seeds produce identical train/val/test partitions
3. **Same metrics**: All models report accuracy, F1 macro, and F1 weighted
4. **Per-baseline classifiers**: Each model uses its own paper's downstream classifier
   (not a one-size-fits-all), so representation quality is measured through each model's
   intended evaluation path
5. **No data leakage**: Test datasets (MotionSense, RealWorld, MobiAct, VTT-ConIoT) were
   never seen during any model's pre-training
6. **Frozen embeddings**: For supervised/linear-probe metrics, embeddings are extracted once
   from the frozen pretrained model — only the downstream classifier is trained

### Intentional Protocol Adaptations

Each baseline paper uses its own evaluation protocol (different datasets, different splits,
different class sets). We intentionally adapt all baselines to our **common benchmark** rather
than replicating each paper's exact experiment. This is necessary for cross-baseline comparison.
Key differences from original papers:

- **LanHAR paper**: Single-source-to-single-target transfer with 4 classes. We use all training
  data as source and evaluate on our 4 test datasets with their full label sets.
- **CrossHAR paper**: Train on source datasets, test on a specific target dataset. We use the
  pretrained encoder and evaluate within each test dataset via random splits.
- **LiMU-BERT paper**: Separate pretrained models per dataset. We use a single combined
  pretrained model across all datasets.
- **MOMENT paper**: Evaluated on UCR/UEA classification benchmarks. We apply their frozen
  embeddings to our HAR benchmark with SVM/linear classifiers.

These are deliberate design choices for a fair unified comparison, not oversights.

---

## 1. LiMU-BERT

**Paper**: Xu et al., SenSys 2021
**Script**: `val_scripts/human_activity_recognition/evaluate_limubert.py`

### What It Is
Self-supervised BERT-style masked reconstruction pretraining for IMU data.
Produces 72-dim embeddings per timestep. NOT text-aligned (no zero-shot capability).

### Pretrained Model
- Checkpoint: `auxiliary_repos/LIMU-BERT-Public/saved/pretrain_base_recgym_20_120/pretrained_combined.pt`
- Pretrained on: All 10 training datasets combined
- Embeddings: Pre-extracted to `.npy` files for all 14 datasets (including test datasets)
- Embedding shape: `(N, 120, 72)` — per-timestep, 72-dim

### Key Implementation Details

**Split-then-reshape (data leakage prevention)**:
The original LiMU-BERT `partition_and_reshape()` splits full `(N, 120, D)` windows into
train/val/test FIRST, then reshapes each partition into `(M, 20, D)` sub-windows. This prevents
sub-windows from the same original window appearing in different splits. Our implementation
mirrors this with `split_full_windows()` followed by `reshape_and_merge()` on each partition.

**Global label offset**:
`reshape_and_merge()` subtracts the minimum label index to zero-base labels. When called on
individual splits after splitting, a partition might not contain all classes, so the per-partition
minimum would differ. We compute the global minimum BEFORE splitting and pass it as `label_offset`
to ensure consistent label indices across train/val/test.

**Sub-window filtering**:
After reshaping `(N, 120, 72)` into `(N*6, 20, 72)` sub-windows, only sub-windows where ALL
20 timesteps have the same activity label are kept. This discards transition windows.

**GRU classifier (paper's architecture)**:
- 2-layer GRU: input_dim=72 -> hidden=20 -> hidden=10
- Final Linear(10, num_classes)
- 100 epochs (using `train_100ep.json` config from original repo), batch_size=512
- Original paper also provides 700-epoch config

**Embedding format for linear probe**:
Mean-pooled across the 120 time steps: `(N, 120, 72)` -> `(N, 72)`.

### What We Do NOT Replicate
- Original paper evaluates 4 separate per-dataset pretrained models. We use one combined model.
- Original paper uses specific dataset-to-dataset transfer experiments. We evaluate within-dataset.

### Intentional Protocol Differences
- **Batch size**: We use batch_size=512 vs original's 128 for the GRU classifier.
  Applied uniformly across all baselines as a speed optimization for RTX 4090.
- **Balanced subsampling**: Our `balanced_subsample()` draws `max(1, budget//n_classes)`
  per class without capping at the smallest class size. The original's
  `prepare_simple_dataset_balance` caps at `min(min_class_count, budget_per_class)`.
  Our approach is unified across all baselines and avoids discarding data when class
  sizes are imbalanced.
- **Window labels**: We use majority-vote per window; the original filters out windows
  where not all timesteps share the same label. Our approach is unified across all
  baselines — all baselines evaluate on the same set of windows.

---

## 2. MOMENT

**Paper**: Goswami et al., ICML 2024
**Script**: `val_scripts/human_activity_recognition/evaluate_moment.py`

### What It Is
General-purpose time series foundation model pretrained on diverse time series data.
Produces 6144-dim embeddings (6 channels × 1024-dim per channel, concatenated).
NOT text-aligned (no zero-shot capability).

### Pretrained Model
- Downloaded from HuggingFace: `AutonLab/MOMENT-1-large`
- No training by us — used as-is in embedding mode
- We never trained or fine-tuned MOMENT

### Key Implementation Details

**Left-padding (paper-faithful)**:
MOMENT expects 512-timestep inputs. Our 120-timestep windows are LEFT-zero-padded:
```
padded[:, :, -120:] = data  # data on right side (positions 392-511)
input_mask[:, -120:] = 1.0  # mask marks real data
```
This matches MOMENT's official `ClassificationDataset` implementation. Earlier versions of our
code incorrectly used RIGHT-padding — fixed to match paper.

**Per-channel embedding concatenation (paper-faithful)**:
Following MOMENT's multivariate evaluation protocol (`unsupervised_representation_learning_multivariate.py`),
each IMU channel is processed as an independent univariate series `(N, 1, 512)`, producing
per-channel 1024-dim embeddings. These are concatenated into `(N, 6*1024) = (N, 6144)`.
This preserves channel-specific information that would be lost by averaging.

**SVM-RBF classifier (paper's protocol)**:
The MOMENT paper's `fit_svm` function uses:
- SVM with RBF kernel, `gamma="scale"`
- GridSearchCV over C = [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000, 1e4]
- 5-fold cross-validation
- `max_iter=10000000` (convergence guarantee)
- If training set > 10,000 samples, stratified subsample to 10,000

**Linear probe**:
Standard linear classifier on 6144-dim concatenated embeddings, 100 epochs.

### What We Do NOT Replicate
- Original paper evaluates on UCR/UEA classification archive. We apply to HAR.
- Original paper uses their own classification head. We use SVM matching their `fit_svm`.

### Intentional Protocol Differences
- **Batch sizes**: We use larger batch sizes (512 vs 128) for downstream classifiers
  as a speed optimization for RTX 4090. This is applied uniformly across all baselines.
- **No StandardScaler**: The original pipeline applies `StandardScaler` before MOMENT's
  internal RevIN normalization. We skip this since RevIN already handles per-sample
  normalization, and within each dataset sensor scales are consistent.

---

## 3. CrossHAR

**Paper**: Dang et al., IMWUT 2024
**Script**: `val_scripts/human_activity_recognition/evaluate_crosshar.py`

### What It Is
Hierarchical self-supervised pretraining: masked reconstruction + contrastive learning.
Produces 72-dim per-timestep embeddings. NOT text-aligned (no zero-shot capability).

### Pretrained Model
- Checkpoint: `auxiliary_repos/CrossHAR/saved/pretrain_base_combined_train_20_120/model_masked_6_1.pt`
- Pretrained on: All 10 training datasets combined (masked pretraining)
- Architecture: 1-layer Transformer with 4 heads, hidden=72, ff=144

### Key Implementation Details

**InstanceNorm preprocessing**:
CrossHAR's `IMUDataset` applies `InstanceNorm1d` to each sample before feeding to the encoder.
Our extraction pipeline replicates this:
```python
inst_norm = nn.InstanceNorm1d(6)
normed = inst_norm(data.transpose(1,2)).transpose(1,2)  # per-sample normalization
```

**Full sequence embeddings (not pooled)**:
Unlike LiMU-BERT's sub-window approach, CrossHAR's `Transformer_ft` classifier operates on the
full `(120, 72)` sequence. The classifier embeds to 100-dim, adds positional encoding, runs a
1-layer Transformer, then mean-pools and classifies.

**Transformer_ft classifier (paper's architecture)**:
- Linear(72, 100) + PositionalEncoding + TransformerEncoder(1 layer, 4 heads, ff=2048)
- Mean pool over sequence -> Linear(100, num_classes)
- 100 epochs, Adam, lr=1e-3, batch_size=512
- Model selection by best validation loss

**Linear probe**:
Mean-pooled embeddings: `(N, 120, 72)` -> `(N, 72)` -> Linear(72, num_classes).

### What We Do NOT Replicate
- Original paper trains on source datasets, evaluates on a held-out target dataset.
  We evaluate within each test dataset using random splits.
- Original paper uses specific source-target pairs. We use one combined pretrained model.

### Intentional Protocol Differences
- **Batch size**: We use batch_size=512 vs original's 128 for the Transformer_ft classifier.
  Applied uniformly across all baselines as a speed optimization.
- **Window labels**: We use majority-vote per window; the original's `merge_dataset(mode='all')`
  discards windows where not all timesteps share the same label. Our approach is unified
  across all baselines — all baselines evaluate on the same set of windows.

---

## 4. LanHAR

**Paper**: Hao et al., 2024
**Script**: `val_scripts/human_activity_recognition/evaluate_lanhar.py`

### What It Is
CLIP-style sensor-text alignment with 2-stage training. Uses SciBERT for text encoding.
Text-aligned model with zero-shot capability. **Trains from scratch during evaluation.**

### Why It Trains From Scratch
LanHAR's sensor encoder is not a general pretrained model — it's specifically trained to align
with SciBERT text embeddings via contrastive learning. There is no "pretrained LanHAR checkpoint"
analogous to LiMU-BERT or CrossHAR. The training IS the method. SciBERT provides the starting
text encoder weights (from HuggingFace), and everything else is learned.

### Key Implementation Details

**Stage 1: SciBERT fine-tuning (10 epochs)**
- Fine-tunes `allenai/scibert_scivocab_uncased` on the 87 training activity labels
- Losses: Multi-positive CLIP + Cross-entropy + 2x Triplet (matching original paper)
- Batch size: 10 (matches paper — only 87 labels, small dataset)
- lr: 1e-5
- Result: Text encoder adapted to activity-aware embedding space

**Stage 2: Sensor-text CLIP training (50 epochs)**
- Trains TimeSeriesTransformer sensor encoder from random initialization
- CLIP contrastive loss between sensor embeddings and text embeddings
- Optimizes: sensor_encoder + txt_proj + sen_proj + logit_scale
- BERT is FROZEN (only projections and sensor encoder train)
- Batch size: 256 (matches paper)
- lr: 4e-5
- Uses source (training) data ONLY — test data is never seen during training

**Stale label embeddings fix**:
During Stage 2, `txt_proj` is trainable, so text-space label embeddings change each epoch.
Validation retrieval accuracy recomputes label embeddings every epoch via
`compute_label_embeddings()` to avoid using stale anchors.

**Gravity alignment (paper-faithful)**:
Original LanHAR applies gravity alignment preprocessing:
1. Estimate gravity vector via 0.3Hz lowpass Butterworth filter on accelerometer
2. Compute Rodrigues rotation matrix to rotate gravity onto +Z axis
3. Apply rotation to both accelerometer and gyroscope
4. Adapted to our 20Hz sampling rate (original uses 50Hz)

**Per-sample LLM descriptions (optional)**:
Original LanHAR uses GPT-4 to generate per-sample text descriptions from signal processing
features. We replicate this with a local LLM via `generate_lanhar_descriptions.py`:
- Extracts EDA features, gyro features, gait synchronization from each window
- Generates structured 7-category analysis prompt
- Queries Ollama (e.g., Qwen2.5:14B) for pattern summaries
- Used with 70% probability during Stage 2 training (30% falls back to class descriptions)
- **Optional**: LanHAR runs without descriptions (lower quality but functional)

**Label dropout disabled**:
Original LanHAR's `LabelAttentionPooling` has dropout=0.1 in cross-attention. We set this to 0.0
because dropout on the label bank makes contrastive text targets stochastic during training,
adding noise to the alignment objective.

### What We Do NOT Replicate
- Original paper uses single-source-to-single-target transfer with ~4 classes.
  We use all 10 training datasets as source with 87 classes.
- Original paper generates descriptions with GPT-4. We use local LLM (optional).
- Original paper evaluates weighted F1. We report both macro and weighted F1.

### Intentional Protocol Differences
- **Template casing**: Original uses `.upper()` for class names in `wrap_template`
  (e.g., `"Activity=WALK"`). We use lowercase with spaces (e.g., `"Activity=walking"`).
  The original has only 4 short class names; with 87 longer labels, uppercasing
  produces unnatural text that SciBERT handles worse.
- **Label prototype recomputation**: Original recomputes label prototypes every batch
  in Stage 1 (4 classes, fast). We recompute once per epoch (87 classes, 6x slower
  per recomputation). With only 10 Stage 1 epochs, the impact is modest.
- **Validation label embeddings in Stage 2**: Original uses margin-based top-k weighted
  averaging across all text descriptions per class. We use the first text prototype per
  class through `txt_proj`. This simplification affects model selection quality but not
  the final embeddings used for evaluation.
- **Validation split**: Original validates on target domain only (20/80 split). We
  validate on a held-out portion of source data (90/10 split) since we have multiple
  test datasets and evaluate on each independently.
- **No target data in Stage 2**: Original combines source + target domains in Stage 2.
  We use only source (training) data, ensuring the sensor encoder never sees test data
  during training. This matches the constraint on all other baselines and prevents
  LanHAR from having an unfair distributional advantage.

---

## 5. TSFM (Our Model)

**Script**: `val_scripts/human_activity_recognition/evaluate_tsfm.py`

### What It Is
Our text-aligned IMU foundation model. Dual-branch Transformer encoder with semantic alignment
head, trained via contrastive learning with soft targets and memory bank.

### Pretrained Model
- Checkpoint: `training_output/semantic_alignment/{run_id}/best.pt`
- Trained on: 10 HAR datasets (87 activity labels)
- **NOTE**: Training in progress. Results pending completion.

### Key Implementation Details

**Patch size sweep**:
TSFM supports variable patch sizes. Evaluation sweeps [1.0, 1.25, 1.5, 1.75, 2.0] seconds
per test dataset independently. For each dataset, a 20% held-out sweep split (seed=42) is
used to select the best patch size by zero-shot closed-set accuracy, then embeddings are
re-extracted on the full dataset with the chosen patch size. This prevents test-time
hyperparameter tuning from inflating reported metrics.

**384-dim embeddings**:
After the semantic alignment head (channel fusion + temporal pooling), embeddings are 384-dim
and L2-normalized.

**Text embeddings from label bank**:
Zero-shot evaluation uses the trained `LearnableLabelBank` to produce text embeddings for each
activity label. These are L2-normalized for cosine similarity retrieval.

---

## Common Infrastructure

### balanced_subsample()
Used by all baselines for 1%/10% supervised metrics. Draws `max(1, round(count * rate))`
samples per class to ensure every class has at least 1 representative. This can slightly
exceed the strict label-rate budget for rare classes — an intentional design choice to prevent
degenerate classifiers with missing classes.

### Seed Consistency
All baselines use `CLASSIFIER_SEED = 3431` for data splits and classifier initialization.
This ensures identical train/val/test partitions across all models for the same dataset.

### Batch Sizes (Optimized for RTX 4090, 24GB)
| Component | Batch Size | Original | Rationale |
|-----------|-----------|----------|-----------|
| Embedding extraction (MOMENT) | 128 | N/A | Large model, near GPU limit |
| Embedding extraction (CrossHAR) | 512 | N/A | Small model |
| GRU training (LiMU-BERT) | 512 | 128 | Speed optimization, applied uniformly |
| Transformer_ft (CrossHAR) | 512 | 128 | Speed optimization, applied uniformly |
| SVM (MOMENT) | N/A | N/A | CPU, sklearn |
| Stage 2 (LanHAR) | 256 | 256 | Matches original paper |
| Linear classifiers (all) | 512 | N/A | All small |
