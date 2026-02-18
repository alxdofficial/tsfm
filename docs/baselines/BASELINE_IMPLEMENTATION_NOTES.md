# Baseline Implementation Notes

Per-baseline design decisions, implementation details, paper-matching considerations,
and how we ensure fairness across the comparison.

---

## Fairness Principles

Our benchmark compares 5 models on the same 4 test datasets using a unified evaluation
framework. To ensure fair comparison:

1. **Same data**: All models evaluate on the same windows. Baselines that require fixed sampling
   rates (LiMU-BERT, CrossHAR, MOMENT, LanHAR) receive data resampled to 20Hz as `(N, 120, 6)`.
   TSFM receives data at each dataset's **native sampling rate** (see Sampling Rate Policy below)
2. **Same splits**: Identical random seeds produce identical train/val/test partitions
3. **Same metrics**: All models report accuracy, F1 macro, and F1 weighted
4. **Per-baseline fine-tuning**: Each model fine-tunes with its own paper's native classification
   mechanism (not a one-size-fits-all), so transfer quality is measured through each model's
   intended evaluation path
5. **No data leakage**: Test datasets (MotionSense, RealWorld, MobiAct, VTT-ConIoT) were
   never seen during any model's pre-training
6. **End-to-end fine-tuning**: For supervised metrics (1%, 10%), the encoder is fine-tuned
   jointly with the classifier using differential learning rates (encoder: 1e-5, head: 1e-3)

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

### Sampling Rate Policy

**Principle**: Models should train and evaluate at each dataset's native sampling rate whenever
their architecture supports it. Resampling is only applied when a model requires a fixed rate.

| Model | Supports Native Rates? | Rate Used | Why |
|-------|:----------------------:|:---------:|-----|
| **TSFM** | Yes | Native per dataset | Seconds-based patch tokenization + interpolation. Sampling rate and channel descriptions are passed to the model and actively used in every forward pass. |
| **LiMU-BERT** | No | 20 Hz | Learned positional embedding fixed at 120 positions; paper designed for 20 Hz. |
| **CrossHAR** | No | 20 Hz | Inherits LiMU-BERT data format; same positional embedding constraint. |
| **MOMENT** | No (rate-agnostic) | 20 Hz | No concept of sampling rate — processes raw number sequences. Cannot benefit from native rates. |
| **LanHAR** | No | 20 Hz | Paper designed for 50 Hz, but trains from scratch in our pipeline. Uses shared 20 Hz benchmark data. |

---

## 1. LiMU-BERT

**Paper**: Xu et al., SenSys 2021
**Script**: `val_scripts/human_activity_recognition/evaluate_limubert.py`
**One-liner**: BERT-style masked reconstruction on 20-step IMU sub-windows; predicts masked timesteps from context to learn motion representations.

### What It Is
Self-supervised BERT-style masked reconstruction pretraining for IMU data.
Produces 72-dim embeddings per timestep. NOT text-aligned — zero-shot uses a
GRU classifier trained on training data.

**Sampling rate**: Fixed 20 Hz. All datasets resampled via temporal bin-and-mean averaging.
The paper explicitly chose 20 Hz to reduce model complexity. Learned positional embeddings
(`nn.Embedding(120, 72)`) are fixed at 120 positions, so the model cannot accept other rates.

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

**End-to-end fine-tuning** *(our addition — paper freezes encoder)*:
For supervised metrics, the BERT encoder is fine-tuned jointly with a GRU classifier.
Raw data is normalized (acc/9.8), passed through the encoder, reshaped into sub-windows,
and classified. The original paper only trains the GRU on static pre-extracted embeddings
(frozen encoder), but we fine-tune end-to-end for consistency with other baselines.

**Window-level majority voting** *(our addition — paper scores sub-windows directly)*:
The original paper evaluates accuracy/F1 at the sub-window level — each 20-step sub-window
is scored independently, giving ~6x more evaluation samples per dataset. We aggregate
sub-window predictions to window-level via majority vote before scoring, so that LiMU-BERT
is evaluated on the same N windows as all other models. This changes the evaluation unit
but does not change the model's predictions.

### What We Do NOT Replicate
- Original paper evaluates 4 separate per-dataset pretrained models. We use one combined model.
- Original paper uses specific dataset-to-dataset transfer experiments. We evaluate within-dataset.
- Original paper scores at sub-window level. We score at window level (majority vote).

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
**One-liner**: General time-series Transformer pretrained on diverse data via masked reconstruction; processes each IMU channel as an independent univariate series.

### What It Is
General-purpose time series foundation model pretrained on diverse time series data (no HAR).
Produces 6144-dim embeddings (6 channels x 1024-dim per channel, concatenated).
NOT text-aligned — zero-shot uses an SVM-RBF classifier trained on training data.

**Sampling rate**: Rate-agnostic — has no concept of physical time or sampling frequency. The
paper states: *"We did not explicitly model temporal resolution."* All input is treated as raw
number sequences padded to 512 timesteps. Uses 20 Hz benchmark data for consistency.

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

**End-to-end fine-tuning** *(our addition — paper uses SVM only for classification)*:
For supervised metrics, MOMENT's encoder is fine-tuned jointly with a linear classification
head (`Linear(6144, num_classes)`). The MOMENT paper's classification evaluation only uses
SVM-RBF on frozen embeddings — it does not fine-tune the encoder for classification tasks.
However, MOMENT's codebase includes a built-in `ClassificationHead` (linear layer) and
supports end-to-end fine-tuning in tutorials. We use this for consistency with other baselines'
end-to-end protocol. SVM is not differentiable and cannot participate in backpropagation.
The encoder's original weights are restored after each fine-tuning run to isolate evaluations.

### What We Do NOT Replicate
- Original paper evaluates on UCR/UEA classification archive. We apply to HAR.
- Original paper uses only SVM on frozen embeddings for classification. We add end-to-end
  fine-tuning with a linear head for supervised metrics.
- Original pipeline applies `StandardScaler` before MOMENT's internal RevIN normalization.
  We skip this since RevIN already handles per-sample normalization.

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
**One-liner**: Hierarchical self-supervised pretraining combining masked reconstruction and contrastive learning on IMU sequences for cross-dataset transfer.

### What It Is
Hierarchical self-supervised pretraining: masked reconstruction + contrastive learning.
Produces 72-dim per-timestep embeddings. NOT text-aligned — zero-shot uses a Transformer_ft
classifier trained on training data.

**Sampling rate**: Fixed 20 Hz. Inherits LiMU-BERT data format (`data_20_120.npy`). Same
learned positional embedding constraint (`nn.Embedding(120, 72)`). Code only accepts
`dataset_version='20_120'`.

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

**End-to-end fine-tuning** *(our addition — paper freezes encoder)*:
For supervised metrics, the encoder is fine-tuned jointly with a Transformer_ft classifier.
Raw data is InstanceNorm-preprocessed in a differentiable way, passed through the encoder,
then through the Transformer_ft head. **The original CrossHAR paper freezes the encoder and
trains only the Transformer_ft classifier on static pre-extracted embeddings.** We fine-tune
end-to-end for consistency with other baselines. This gives CrossHAR a slight advantage over
its paper's protocol, since the encoder can adapt to the target dataset.

### What We Do NOT Replicate
- Original paper trains on source datasets, evaluates on a held-out target dataset.
  We evaluate within each test dataset using random splits.
- Original paper uses specific source-target pairs. We use one combined pretrained model.
- Original paper freezes the encoder during downstream evaluation. We fine-tune end-to-end.

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
**One-liner**: 2-stage CLIP-style alignment: (1) fine-tune SciBERT on activity text, (2) train a sensor Transformer from scratch to align with the text embedding space.

### What It Is
CLIP-style sensor-text alignment with 2-stage training. Uses SciBERT for text encoding.
Text-aligned model with zero-shot capability. **Trains from scratch during evaluation.**

**Sampling rate**: Paper uses 50 Hz; our benchmark uses 20 Hz. Since LanHAR trains its sensor
encoder from scratch (no pretrained weights), it learns temporal patterns at whatever rate it
receives. Gravity alignment and Butterworth filter preprocessing correctly adapts via the `fs`
parameter. Uses shared 20 Hz benchmark data.

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
- **No target data in Stage 2** *(intentional — paper combines source + target)*: Original
  combines source + target domains in Stage 2, giving the sensor encoder access to the test
  data distribution during training. We use only source (training) data, ensuring the sensor
  encoder never sees test data. This matches the constraint on all other baselines and prevents
  LanHAR from having an unfair distributional advantage, but slightly disadvantages LanHAR
  relative to its paper's reported numbers.
- **Supervised fine-tuning** *(our addition — paper is zero-shot only)*: The original LanHAR
  paper has no supervised fine-tuning protocol. We add 1%/10% supervised evaluation by
  fine-tuning the sensor encoder + sen_proj via cosine similarity with frozen text prototypes.
  This uses LanHAR's native cosine-sim mechanism rather than adding an external classifier.

---

## 5. TSFM (Our Model)

**Script**: `val_scripts/human_activity_recognition/evaluate_tsfm.py`
**One-liner**: Dual-branch Transformer trained with CLIP-style contrastive alignment between variable-length IMU patches and learnable text label embeddings.

### What It Is
Our text-aligned IMU foundation model. Dual-branch Transformer encoder with semantic alignment
head, trained via contrastive learning with soft targets and memory bank.

**Per-dataset metadata** (unique to TSFM — no baseline uses any of these):
- **Sampling rate**: Native per dataset (e.g., 50 Hz for UCI HAR, 100 Hz for PAMAP2, 20 Hz for
  WISDM). Read from dataset manifest and passed to `create_patches()`, which converts seconds-based
  patch size to timesteps dynamically. Each patch is interpolated to a fixed 64-step representation,
  decoupling the model from any specific rate.
- **Patch size**: Specified in seconds per dataset. During training, supports **patch size
  augmentation** — randomly samples from a `(min_sec, max_sec, step_sec)` range, forcing the
  model to learn resolution-robust representations. Fixed at 1.0s during evaluation.
- **Channel descriptions**: Text strings from dataset manifest (e.g., "Accelerometer X-axis",
  "Chest acceleration X-axis from wearable sensor"). Encoded by frozen SentenceBERT and fused
  into sensor features via `ChannelSemanticEncoding`, giving the model semantic awareness of
  what each channel measures.

### Pretrained Model
- Checkpoint: `training_output/semantic_alignment/{run_id}/best.pt`
- Trained on: 10 HAR datasets (87 activity labels)

### Key Implementation Details

**Fixed 1.0s patch size**:
TSFM supports variable patch sizes, but we use a fixed 1.0s for all test datasets — no
per-dataset sweep or test-time tuning. This is a metadata-only decision: at native 50Hz, 1.0s
patches (50 timesteps) are interpolated to 64 fixed steps, giving fine temporal resolution.
Sensitivity analysis shows results are robust (max 9% range across patch sizes on the easiest
dataset, <2% on the hardest).

**384-dim embeddings**:
After the semantic alignment head (channel fusion + temporal pooling), embeddings are 384-dim
and L2-normalized.

**Text embeddings from label bank**:
Zero-shot evaluation uses the trained `LearnableLabelBank` to produce text embeddings for each
activity label. These are L2-normalized for cosine similarity retrieval.

**End-to-end fine-tuning**:
For supervised metrics, the encoder + semantic alignment head are fine-tuned end-to-end.
Classification is via cosine similarity with frozen text embeddings from the label bank
(no separate classifier head). The label bank stays frozen during fine-tuning as the text
anchor space.

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
| Zero-shot GRU training (LiMU-BERT) | 512 | 128 | Speed optimization, applied uniformly |
| Zero-shot Transformer_ft (CrossHAR) | 512 | 128 | Speed optimization, applied uniformly |
| SVM (MOMENT) | N/A | N/A | CPU, sklearn |
| Stage 2 (LanHAR) | 256 | 256 | Matches original paper |
| End-to-end fine-tuning (all) | 32 | N/A | Small for gradient stability with encoder |
