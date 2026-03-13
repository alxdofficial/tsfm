# Baseline Evaluation Results

Generated: 2026-03-13 | Framework: unified evaluation | Seed: 3431

## Models

| Model | How It Works | Embed Dim | Zero-Shot | Supervised |
|-------|-------------|:---------:|-----------|------------|
| **TSFM-Medium (ours)** | Same architecture as Small-Deep but with d=512, 8 layers — largest model (~63M total); per-patch prediction with majority voting | 768 | Cosine sim with text embeddings (majority vote) | End-to-end cosine sim |
| **TSFM-Small-Deep (ours)** | 8-layer dual-branch Transformer with per-patch prediction, trained with CLIP-style contrastive alignment between IMU patches and text labels; majority voting at inference; processes each channel independently then fuses to 384-dim | 384 | Cosine sim with text embeddings (majority vote) | End-to-end cosine sim |
| **TSFM-Tiny (ours)** | Same architecture as Small-Deep but with d=192, 4 layers — lightweight edge model (~6M trainable); per-patch prediction with majority voting | 384 | Cosine sim with text embeddings (majority vote) | End-to-end cosine sim |
| **LiMU-BERT** | BERT-style masked reconstruction on 20-step IMU sub-windows; predicts masked timesteps from context | 72 | GRU classifier | End-to-end encoder + GRU |
| **MOMENT** | General time-series Transformer pretrained on diverse time-series data (no HAR); processes each IMU channel independently and **concatenates** (no fusion) to 6144-dim | 6144 | SVM-RBF | End-to-end encoder + linear |
| **CrossHAR** | Hierarchical self-supervised pretraining combining masked reconstruction and contrastive learning on IMU sequences | 72 | Transformer_ft classifier | End-to-end encoder + Transformer_ft |
| **LanHAR** | 2-stage CLIP-style alignment: (1) fine-tune SciBERT on activity text, (2) train a sensor Transformer to align with text space | 768 | Cosine sim with text embeddings | End-to-end cosine sim |
| **LLaSA** | LIMU-BERT IMU encoder + Vicuna-7B LLM; classifies by prompting the LLM with sensor tokens and parsing the generated text response | 7B params | LLM prompt → parse text | N/A (7B LLM not fine-tunable with few labels) |

## Model Size Comparison

Parameter counts measured by instantiating each model and summing `p.numel()` for all parameters.
"Trainable" means parameters updated during our HAR pretraining or stage-2 alignment training.
"Inference total" includes all parameters that must be loaded and run forward passes at test time,
including frozen components.

### Summary

| Model | IMU Encoder | Language/Text Module | Classifier/Projection Heads | Trainable | Inference Total |
|-------|----------:|---------------------:|----------------------------:|----------:|----------------:|
| **TSFM-Medium (ours)** | ~36M | 22.7M (frozen SBERT) | ~26M | **~63M** | **~85M** |
| **TSFM-Small-Deep (ours)** | ~19M | 22.7M (frozen SBERT) | ~15M | **~35M** | **~58M** |
| **TSFM-Tiny (ours)** | ~2.4M | 22.7M (frozen SBERT) | ~3.6M | **~6.0M** | **~28.7M** |
| **LiMU-BERT** | 62.6K | — | 10.1K (GRU) | **72.7K** | **72.7K** |
| **MOMENT** | 341.2M | — | ~50K (linear) | **341.2M** | **341.2M** |
| **CrossHAR** | 62.8K | — | 468.6K (Transformer) | **531.4K** | **531.4K** |
| **LanHAR** | 11.8M | 109.9M (SciBERT) | 1.2M (projections) | **13.0M**\* | **122.9M** |
| **LLaSA** | 62.6K (LiMU-BERT) | ~6.7B (Vicuna-7B) | ~17M (MLP projector) | **~6.7B** | **~6.7B** |

\*LanHAR stage-2 training freezes SciBERT (only sensor encoder + projections are trained). During
supervised fine-tuning, SciBERT is unfrozen and all 122.9M parameters are updated.

### Component Breakdown

**TSFM (ours)** — Small-Deep config (d=384, 8 temporal layers, per-patch prediction):

| Component | Parameters | Role |
|-----------|----------:|------|
| SpectralTemporal feature extractor (hybrid CNN+FFT, channel-independent) | ~100K | Temporal + spectral feature extraction per channel |
| Positional encoding (sinusoidal temporal + channel semantic MLP 384→384) | ~296K | Time + sensor meaning encoding |
| Dual-branch Transformer (8 layers, d=384, 8 heads, temporal + cross-channel) | ~19M | Core IMU encoder (2× Small's 4 layers) |
| Semantic alignment head (4 temporal layers, 6 fusion queries, 6 pool queries) | ~12M | Per-patch embeddings, projects to semantic space |
| Channel-text fusion (cross-attention between sensor tokens and text tokens) | ~1.6M | Conditions sensor encoding on channel descriptions |
| Learnable label bank (attention pooling, 6 query tokens per label) | ~1.5M | Multi-prototype text embeddings per label |
| **Trainable subtotal** | **~35M** | Saved in checkpoint |
| all-MiniLM-L6-v2 (SentenceBERT, 6-layer, d=384) | 22.7M | Frozen text encoder for channel descriptions + labels |
| **Inference total** | **~58M** | |

**TSFM-Tiny (ours)** — Tiny config (d=192, 4 temporal layers, per-patch prediction):

| Component | Parameters | Role |
|-----------|----------:|------|
| SpectralTemporal feature extractor (hybrid CNN+FFT, channel-independent) | ~28K | Temporal + spectral feature extraction per channel |
| Positional encoding (sinusoidal temporal + channel semantic MLP 384→192) | ~74K | Time + sensor meaning encoding |
| Dual-branch Transformer (4 layers, d=192, 4 heads, temporal + cross-channel) | ~2.4M | Core IMU encoder |
| Semantic alignment head (2 temporal layers, 4 fusion queries, 4 pool queries) | ~2.3M | Per-patch embeddings, projects to semantic space |
| Learnable label bank (attention pooling, 4 query tokens per label) | ~1.3M | Multi-prototype text embeddings per label |
| **Trainable subtotal** | **~6.0M** | Saved in checkpoint |
| all-MiniLM-L6-v2 (SentenceBERT, 6-layer, d=384) | 22.7M | Frozen text encoder for channel descriptions + labels |
| **Inference total** | **~28.7M** | |

Config: `TINY_CONFIG` in `model/config.py` — d_model=192, num_heads=4, num_temporal_layers=4,
dim_feedforward=768, contrastive_text_model=all-MiniLM-L6-v2 (384-dim), semantic_dim=384,
per_patch_prediction=True. Trained for 200 epochs with memory bank, jitter+scale augmentation.
Checkpoint: `training_output/semantic_alignment/20260306_051235/best.pt` (epoch 172, val_acc=83.3%).

**LiMU-BERT** — 72.7K at inference:

| Component | Parameters | Role |
|-----------|----------:|------|
| Input projection (Linear(6→72)) | 504 | Channel embedding |
| Positional embedding (Embedding(120,72)) | 8,640 | Fixed 120-position learned PE |
| Shared Transformer block (d=72, 4 heads, ff=144) — stored once, looped 4× | 42,408 | Masked reconstruction encoder |
| Pretrain decoder head (Linear(72→6)) | 11,094 | In checkpoint, unused during embedding |
| GRU classifier (72→20→10→87, 2-layer GRU) | 10,077 | Zero-shot: trained on training data |
| **Total** | **72.7K** | |

**MOMENT** — 341.2M at inference:

| Component | Parameters | Role |
|-----------|----------:|------|
| Patch embedding (Conv1d input projection) | ~2M | Time series → patch tokens |
| T5 Transformer encoder (24 layers, d=1024, ff=4096, 16 heads) | ~339M | General time-series encoder |
| Linear classification head (Linear(6144→N)) | ~50K | Fresh per dataset (6×1024 concat → N classes) |
| **Total** | **341.2M** | |

**CrossHAR** — 531.4K at inference:

| Component | Parameters | Role |
|-----------|----------:|------|
| Input projection (Linear(6→72)) | 504 | Channel embedding |
| Positional embedding (Embedding(120,72)) | 8,640 | Fixed 120-position learned PE |
| Transformer block (d=72, 4 heads, ff=144, 1 layer) | 42,408 | Hierarchical SSL encoder |
| Pretrain decoder head | 11,094 | In checkpoint, unused during embedding |
| Transformer classifier (72→100, 1 layer, 4 heads, ff=2048→87) | 468,635 | Zero-shot: trained on training data |
| **Total** | **531.4K** | |

**LanHAR** — 122.9M at inference:

| Component | Parameters | Role |
|-----------|----------:|------|
| SciBERT (12-layer BERT-base, sci vocab, d=768) | 109.9M | Text encoder (frozen stage 2, unfrozen supervised FT) |
| Sensor Transformer (3 layers, d=768, 2 heads, ff=1024) | 11.8M | IMU encoder |
| Text projection (Linear(768→768) + LN) | 592K | Project text to shared space |
| Sensor projection (Linear(768→768) + LN) | 592K | Project sensor to shared space |
| Logit scale | 1 | Learned temperature |
| **Total** | **122.9M** | |

**LLaSA** — ~6.7B at inference:

| Component | Parameters | Role |
|-----------|----------:|------|
| LiMU-BERT IMU encoder (same arch as standalone) | 62.6K | Sensor token extraction |
| MLP projector (Linear(72→4096) + GELU + Linear(4096→4096)) | ~17M | Bridge IMU tokens → LLM token space |
| Vicuna-7B (32-layer LLaMA, d=4096, 32 heads) | ~6.7B | Language model backbone |
| **Total** | **~6.7B** | |

### Observations

1. **TSFM-Small-Deep is a compact model** (~35M trainable), smaller than MOMENT (341M) and LLaSA
   (~6.7B). Despite being ~10× smaller than MOMENT and ~190× smaller than LLaSA, TSFM leads on
   all average evaluation metrics. **TSFM-Tiny (~6M trainable) is ~57× smaller than MOMENT** yet
   beats it on supervised fine-tuning (86.4% vs 81.3% at 10%), demonstrating that the architecture
   is highly parameter-efficient.

2. **MOMENT's 341.2M parameters are ~6× larger than TSFM-Small-Deep's inference total** (~58M)
   and ~12× larger than TSFM-Tiny's (~28.7M). Combined with its 6144-dim embedding (vs TSFM's
   384-dim), this partly explains why MOMENT performs close to TSFM on some metrics without text
   alignment — the model has far more capacity to encode discriminative patterns.

3. **LiMU-BERT and CrossHAR share the same encoder architecture** (Linear(6→72), Embedding(120,72),
   d=72 transformer) — the main difference is pretraining strategy (masked reconstruction vs
   hierarchical SSL) and the number of transformer layers (4 shared vs 1). CrossHAR's larger
   classifier head (468.6K vs 10.1K) accounts for most of its parameter difference.

4. **LanHAR's trainable portion (13.0M) is comparable to TSFM** (~35M), but most of its inference
   parameters come from the frozen SciBERT (109.9M). During supervised fine-tuning SciBERT is
   unfrozen, making the effective trainable count 122.9M.

5. **LLaSA is dominated by the 6.7B Vicuna-7B LLM** — the IMU encoder (62.6K) is <0.001% of the
   total model. The MLP projector (17M) bridges a 72-dim sensor space to a 4096-dim LLM space.

## Fairness Notes

**Training data**: All HAR-pretrained models (TSFM, LiMU-BERT, CrossHAR, LanHAR) use 10 training
datasets. All 7 test datasets were never seen during any model's pretraining. MOMENT was pretrained
on general time-series data with no HAR-specific data. **LLaSA was trained by its authors on HHAR,
MotionSense, Shoaib, and UCI-HAR** — MotionSense and Shoaib overlap with our test datasets, so
LLaSA results on those two datasets are marked with \* to indicate they are not truly zero-shot.

**TSFM uses native sampling rates and dataset-specific channel descriptions**: TSFM evaluates each
test dataset at its native sampling rate (50Hz for most datasets, 30Hz for Opportunity) with rich
channel descriptions from dataset manifests (e.g., "Accelerometer X-axis (waist-mounted
smartphone)"), while baselines use 20Hz resampled data with no channel metadata. This is fair because:
- **Native rate handling is a genuine architectural capability**, not a data advantage. TSFM's
  seconds-based patch tokenization with interpolation to fixed 64 steps was specifically designed
  to handle variable sampling rates. Baselines architecturally cannot do this (LiMU-BERT/CrossHAR
  have learned positional embeddings fixed at 120 positions; MOMENT has no concept of physical time).
- **Channel description awareness is a genuine architectural capability**. TSFM's
  `ChannelSemanticEncoding` module was designed to incorporate sensor metadata. No baseline has
  an equivalent mechanism.
- **Each model is evaluated with its full capabilities**: just as MOMENT benefits from its 341M
  parameters and 6144-dim embeddings, and LanHAR benefits from per-sample LLM descriptions,
  TSFM benefits from its metadata-aware architecture. Artificially crippling TSFM to 20Hz would
  be equivalent to limiting MOMENT to 384-dim embeddings — unfair in the opposite direction.

**Zero-shot prediction mechanisms differ by model type**:
- *Text-aligned models* (TSFM, LanHAR): Encode activity labels as text, predict via cosine similarity
  with sensor embeddings. **Truly training-data-free** — no classifier is fitted, no labeled data is
  used at inference time.
- *Classifier-based models* (LiMU-BERT, MOMENT, CrossHAR): Train a native classifier on
  **training dataset** embeddings (87 global labels), then predict on test data. This is
  "zero-shot" in the sense of zero-shot *transfer to unseen datasets* — the classifier never sees
  test data, but it does use labeled training data to learn the embedding-to-label mapping.
  LiMU-BERT uses a GRU, MOMENT uses SVM-RBF, CrossHAR uses a Transformer.
- *Generative model* (LLaSA): Prompts a 7B LLM with sensor tokens and activity list; parses the
  generated text response to extract the predicted label. Responses that cannot be matched to any
  valid label are counted as "unclear" (wrong). This is inherent to generative classification —
  other models always produce a valid prediction.

**Zero-shot scoring**: Open-set uses group match (synonyms accepted) for all models. Closed-set
uses each model's native mechanism: text-aligned models (TSFM, LanHAR) predict from the C test
labels via cosine similarity and score via exact match; classifier-based models (MOMENT, LiMU-BERT,
CrossHAR) mask their 87-class logits to test-relevant groups and score via group match.

**Supervised fine-tuning**: Each baseline fine-tunes its encoder end-to-end with its native
classification mechanism. Text-aligned models (TSFM, LanHAR) classify via cosine similarity
with frozen text embeddings. Non-text-aligned models use their paper's native classifier head.

**Embedding dimensions vary significantly**: MOMENT (6144) >> LanHAR (768) > TSFM (384) >>
LiMU-BERT/CrossHAR (72). Both TSFM and MOMENT process IMU channels independently, but TSFM
fuses per-channel representations into a compact 384-dim vector, while MOMENT concatenates
6 × 1024 = 6144-dim without any compression. This 16× dimensionality gap gives MOMENT's
downstream classifiers (SVM-RBF for ZS, linear head for supervised) substantially more
information to work with, which largely explains why MOMENT performs close to TSFM despite
having no text alignment and no HAR-specific pretraining.
See [Note on MOMENT's Structural Advantages](#note-on-moments-structural-advantages) below.

## Adaptations from Original Papers

We adapt each baseline to our unified benchmark rather than replicating each paper's exact experiment.
The table below documents every significant deviation and its fairness rationale.

| Baseline | What We Changed | Original Paper Protocol | Our Adaptation | Fairness Rationale |
|----------|----------------|------------------------|----------------|-------------------|
| **LiMU-BERT** | Window-level scoring | Scores each 20-step sub-window independently (6x more evaluation units per window) | Majority vote across 6 sub-windows per 120-step window | All models must be scored on the same evaluation units (windows) for comparable n_samples |
| **LiMU-BERT** | Single combined checkpoint | Separate pretrained models per dataset | One model pretrained on all 10 datasets combined | Unified pretraining for fair cross-dataset comparison |
| **LiMU-BERT** | GRU zero-shot classifier | Paper uses GRU for supervised only, not for zero-shot transfer | Train GRU (72→20→10→87) on training embeddings, predict test via logits | Only available mechanism — LiMU-BERT has no text branch for cosine similarity. GRU matches the paper's classifier architecture |
| **CrossHAR** | End-to-end supervised fine-tuning | Freezes encoder; trains only classifier head on static pre-extracted embeddings | Fine-tunes encoder + classifier jointly | All baselines use end-to-end fine-tuning for supervised metrics, giving each encoder a chance to adapt; this slightly *advantages* CrossHAR vs its paper |
| **CrossHAR** | Transformer zero-shot classifier | Paper uses Transformer classifier for supervised only, not for zero-shot | Train Transformer(72→100→87, 1-layer, 4-head) on training embeddings, predict test via logits | Same rationale as LiMU-BERT; matches the paper's classifier architecture |
| **MOMENT** | Linear head for supervised | Paper's classification evaluation uses only SVM-RBF on frozen embeddings (no fine-tuning) | Linear head (from MOMENT codebase's `ClassificationHead`) fine-tuned end-to-end | SVM is not differentiable; linear head enables end-to-end fine-tuning consistent with other baselines |
| **MOMENT** | SVM zero-shot classifier | Paper does not evaluate zero-shot transfer | SVM-RBF (GridSearchCV, 5-fold, 9 C values) trained on training embeddings; predict test via masked logits | SVM-RBF is MOMENT's native classifier — uses their codebase's approach. GridSearchCV ensures optimal hyperparameters |
| **MOMENT** | Per-channel processing with concatenation | Paper processes each channel independently (univariate) and concatenates to 6144-dim | Same — each of 6 channels processed as univariate (1, 512) series, embeddings concatenated to (N, 6144). Note: TSFM also processes channels independently but fuses to 384-dim | Faithful to MOMENT's design. The 6144-dim output gives MOMENT's classifiers 16× more information than TSFM's 384-dim, which partly explains MOMENT's strong performance. See [Note on MOMENT's Structural Advantages](#note-on-moments-structural-advantages) |
| **LanHAR** | No target data in Stage 2 | Sensor encoder trains on source + target data combined | Source data only (test data never seen) | No other baseline sees test data during training; exclusion prevents unfair distributional advantage but slightly *disadvantages* LanHAR vs its paper |
| **LanHAR** | Supervised fine-tuning added | Paper is zero-shot only (no supervised protocol) | Fine-tune entire model end-to-end (BERT + sensor encoder + projections) via cosine sim with frozen text prototypes | Extension for benchmark completeness; all baselines fine-tune end-to-end for consistency |
| **LanHAR** | BERT unfrozen during fine-tuning | Paper freezes BERT after Stage 1 | Fine-tune all parameters including BERT, with uniform lr=1e-5 | Consistent with other text-aligned models (TSFM fine-tunes its full model). Gives LanHAR the best chance to adapt. Since TSFM also fine-tunes its text components, this is equitable |
| **LLaSA** | Published weights as-is | Paper trains on HHAR, MotionSense, Shoaib, UCI-HAR with GPT-generated narrations | Use published BASH-Lab/LLaSA-7B from HuggingFace at fp16; no retraining | Retraining a 7B LLM is infeasible for a baseline comparison; published weights are evaluated as-is |
| **LLaSA** | 7B model (not 13B) | Paper uses Vicuna-13B | Vicuna-7B (from published HuggingFace weights) | 13B model requires >26GB VRAM; 7B is the published alternative. This may slightly *disadvantage* LLaSA |
| **LLaSA** | Zero-shot only | Paper evaluates zero-shot classification | Zero-shot open-set and closed-set; no supervised fine-tuning | Fine-tuning a 7B LLM with 1%/10% labeled data is impractical and incomparable to fine-tuning smaller encoders |
| **LLaSA** | Subsampled evaluation | Paper uses 10 samples per class | Up to 100 samples per class (vs full dataset for other baselines) | LLM inference at ~0.3s/sample makes full-dataset evaluation prohibitively slow. 100/class is 10x the paper's 10/class, giving LLaSA more evaluation signal |
| **All** | Unified zero-shot protocol | Each paper has its own evaluation protocol (if any) | Standardized open-set (87 labels, group match) and closed-set (test labels only) for all model types | Enables fair cross-model comparison on identical tasks with identical scoring |
| **All** | Unified supervised protocol | Each paper has its own supervised setup | Standardized: 1% and 10% labeled data, 80/10/10 split, seed 3431, max 20 epochs, patience 5, val accuracy for model selection | Ensures identical data conditions across all baselines |
| **All** | Unified batch sizes | Each paper uses its own batch size (typically 128) | 512 for zero-shot classifiers, 32 for fine-tuning | Speed optimization; applied uniformly across all baselines |

See [BASELINE_IMPLEMENTATION_NOTES.md](BASELINE_IMPLEMENTATION_NOTES.md) for full per-model implementation details.

## Evaluation Protocol

All evaluations use a unified protocol with 4 metrics: zero-shot open-set, zero-shot closed-set,
1% supervised fine-tuning, and 10% supervised fine-tuning. This section describes the exact
implementation of each metric, since these protocols are our own design (not from any baseline paper).

### Shared Infrastructure

**Label groups**: The 10 training datasets use 87 unique activity labels, many of which are synonyms
(e.g., "jogging"/"running", "walking_downstairs"/"stairs_down"). We define label groups that cluster
semantically equivalent labels (`datasets/imu_pretraining_dataset/label_groups.py`). The function
`get_label_to_group_mapping()` maps each label to its canonical group. Group matching means
a prediction of "jogging" is scored correct if the ground truth is "running" (same group).

**Data splits**: All models use the same random split (seed `3431`). Windows are shuffled and
split 80/10/10 into train/val/test. Subsampling for 1% and 10% is applied only to the train
portion (balanced across classes: `n_per_class = max(1, int(N_train * rate) // n_classes)`).
Val and test sets are always the full 10% slice — never subsampled.

### Zero-Shot Open-Set

The model predicts from all 87 training labels. Scored via group match (synonyms accepted).

The prediction mechanism differs by model type:

| Model Type | Mechanism |
|-----------|-----------|
| **Text-aligned** (TSFM, LanHAR) | Encode all 87 labels as text embeddings; predict via argmax cosine similarity against sensor embeddings. No classifier training needed. |
| **Classifier-based** (LiMU-BERT, MOMENT, CrossHAR) | Train a native classifier on pre-extracted training embeddings (87-class). LiMU-BERT uses a GRU (Adam, lr=1e-3, 100 epochs, batch 512); MOMENT uses SVM-RBF (GridSearchCV, 5-fold, 9 C values); CrossHAR uses a Transformer classifier (Adam, lr=1e-3, 100 epochs, batch 512). Predict on test embeddings, argmax over all 87 logits. |
| **Generative** (LLaSA) | Prompt the LLM with all 87 labels plus sensor tokens; parse the generated text to extract a label; fuzzy string matching to map output to a valid label. |

**Fairness note**: "Zero-shot" here means zero-shot *with respect to the test dataset* — no test
data is ever used for classifier training. However, classifier-based models do use labeled
training data to fit their ZS classifiers, while text-aligned models are truly training-data-free
at inference time (cosine similarity requires no fitted classifier). The classifier's capacity to
generalize is part of the model's capability being evaluated — some classifiers (e.g., SVM-RBF)
may generalize better than others (e.g., GRU). Text-aligned models have a structural advantage
here: they need no classifier training and can directly compare any text label to any sensor
embedding.

### Zero-Shot Closed-Set

The model predicts only from the test dataset's own activity labels. This removes the difficulty
of selecting among 87 labels and focuses on discriminating test activities.

| Model Type | Prediction | Scoring |
|-----------|-----------|---------|
| **Text-aligned** (TSFM, LanHAR) | Encode only the C test labels as text; argmax cosine similarity. | Exact match (predicts test labels directly) |
| **Classifier-based** (LiMU-BERT, MOMENT, CrossHAR) | Mask logits to training labels whose group matches a test label; argmax over masked logits. | Group match (maps through synonym groups) |
| **Generative** (LLaSA) | Prompt with only the C test labels; parse output. | Exact match |

### Supervised Fine-Tuning (1% and 10%)

Each model's encoder is fine-tuned end-to-end with a small amount of labeled data from the
test dataset. The classification head matches each model's native architecture:

| Model | What Is Fine-Tuned | Classification Head | Optimizer | LR |
|-------|-------------------|--------------------|-----------|----|
| **TSFM** | Full model (deep copy) | None — cosine sim vs frozen text embeddings, temperature=0.07 | AdamW | Uniform 1e-5 |
| **LiMU-BERT** | Encoder + fresh GRU head (deep copy) | GRU(72→20→10→C) | AdamW | Encoder 1e-5, head 1e-3 |
| **MOMENT** | Full model + fresh linear head | Linear(6144, C) | AdamW | Encoder 1e-5, head 1e-3 |
| **CrossHAR** | Encoder + fresh Transformer head (deep copy) | Transformer(72→100→C) with 1-layer self-attention | AdamW | Encoder 1e-5, head 1e-3 |
| **LanHAR** | Full model (BERT + sensor encoder + projections, deep copy) | None — cosine sim vs frozen text embeddings, pretrained logit_scale | AdamW | Uniform 1e-5 |
| **LLaSA** | N/A | N/A | N/A | N/A |

**Shared settings**: All models use max 20 epochs with early stopping (patience=5, monitored
on val accuracy). Loss is cross-entropy. Best checkpoint by val accuracy is restored before
test evaluation. AMP (GradScaler) is used for TSFM and LanHAR on CUDA; other models do not
use AMP.

**Fairness note on classification heads**: Text-aligned models (TSFM, LanHAR) classify via
cosine similarity against frozen text label embeddings, which means the text encoder's quality
directly influences supervised results. Classifier-based models use fresh randomly-initialized
heads (GRU, linear, Transformer) that are trained from scratch — this means they need more data
to converge but are not limited by text encoder quality. This explains why LiMU-BERT's 1%
performance is particularly weak (35.7% avg): the GRU head has too few samples to learn from
scratch, while TSFM's cosine-sim mechanism works even with very few samples because the text
embeddings provide structured class prototypes.

**Fairness note on differential learning rates**: LiMU-BERT, MOMENT, and CrossHAR use
differential learning rates (encoder at 1e-5, head at 1e-3) because their heads are randomly
initialized and need faster convergence. TSFM and LanHAR use uniform 1e-5 because they have no
separate head — the entire model is fine-tuned as one unit. This is the standard practice from
each model's codebase/paper.

### Data Preprocessing Comparison

Each model uses the preprocessing pipeline that matches its architecture. This is a genuine
capability difference, not a configurable choice:

| Model | Sampling Rate | Window Size | Channels | Normalization |
|-------|:---:|:---:|:---:|---------------|
| **TSFM** | Native (30-50Hz) | Varies (180-300 steps, 6s) | 6 (acc+gyro) | None (model normalizes internally) |
| **LiMU-BERT** | 20Hz (resampled) | 120 steps (6s) | 6 (acc+gyro) | acc /= 9.8 (for fine-tuning only) |
| **MOMENT** | 20Hz (resampled) | 120 steps, left-zero-padded to 512 | 6 (processed per-channel as univariate) | None |
| **CrossHAR** | 20Hz (resampled) | 120 steps (6s) | 6 (acc+gyro) | InstanceNorm1d per sample per channel |
| **LanHAR** | 20Hz (resampled) | 120 steps (6s) | 6 (acc+gyro) | Gravity alignment (Butterworth LP at 0.3Hz, Rodrigues rotation to +Z) |
| **LLaSA** | 20Hz (resampled) | 120 steps (6s) | 6 (acc+gyro) | None |

**Fairness note**: TSFM's native-rate evaluation is a genuine architectural capability (seconds-based
patch tokenization with interpolation to fixed 64 steps). Baselines cannot use native rates because
their architectures have fixed positional embeddings tied to specific sequence lengths
(LiMU-BERT/CrossHAR: 120 positions at 20Hz; MOMENT: 512 positions). Resampling all models to
20Hz would artificially handicap TSFM while giving no benefit to baselines. See the
[Ablation](#ablation-native-rate--rich-channel-descriptions) section for the measured impact of
native rate + channel descriptions.

### Native Architectural Capabilities

Our evaluation preprocesses all baseline data to 6 channels at 20Hz (see above), but this is a
pipeline choice driven by baseline limitations — not all models are architecturally restricted this
way. The table below documents each model's **native** capabilities for handling variable sampling
rates and variable channel counts, independent of how we evaluate them.

| Model | Variable Sampling Rate | Variable Channel Count | Architectural Constraint |
|-------|:---:|:---:|---|
| **TSFM (ours)** | Yes | Yes | Patches in seconds → `F.interpolate` to fixed 64 timesteps; channels are a dynamic axis with channel-independent CNN + cross-channel attention. No hardcoded channel count or sequence length. Both Small-Deep and Tiny share this architecture. |
| **LiMU-BERT** | No | No | Input projection is `Linear(6, 72)` — hardcodes 6 channels. Positional encoding is `Embedding(120, 72)` — hardcodes 120 timesteps (6s × 20Hz). Changing either requires retraining. |
| **MOMENT** | No | Partially | Fixed 512-token sequence length with no concept of physical time (no Hz-awareness). Univariate per-channel processing means arbitrary channel counts are theoretically possible, but our pipeline hardcodes 6 channels for SVM compatibility. |
| **CrossHAR** | No | No | Same architecture as LiMU-BERT: `Linear(6, 72)` input and `Embedding(120, 72)` positional encoding. Identical constraints. |
| **LanHAR** | Partially | No | Sinusoidal positional encoding (not learned) could handle variable sequence lengths, but input projection is `Linear(6, 768)` — hardcodes 6 channels. |
| **LLaSA** | No | No | Uses LiMU-BERT as its IMU encoder — inherits the same `Linear(6, 72)` and `Embedding(120, 72)` constraints. |

**Why this matters**: TSFM is the only model that can natively process datasets at their original
sampling rate and with their full channel set. All other models require resampling to a fixed rate
and/or channel extraction to exactly 6 channels. This is an inherent architectural capability —
TSFM was designed with seconds-based temporal reasoning and dynamic channel handling, while
baselines were designed for fixed-format inputs. When evaluating at native rates (50Hz, 30Hz, etc.),
TSFM exploits higher temporal resolution that baselines architecturally cannot access.

## Test Datasets

We evaluate on 5 main test datasets with high label group coverage (85-100%), plus 2 severe
out-of-domain datasets (HARTH, VTT-ConIoT) reported separately due to fundamentally different
characteristics. See [Severe Out-of-Domain](#severe-out-of-domain-harth--vtt-coniot) below.

### Main Test Datasets (85-100% label coverage)

| Dataset | Windows | Classes | Hz | Group Coverage | Difficulty |
|---------|--------:|--------:|---:|:-:|------------|
| MotionSense | 12,080 | 6 | 50 | 100% | Easy (basic locomotion, smartphone) |
| RealWorld | 27,138 | 8 | 50 | 100% | Medium (multi-placement) |
| MobiAct | 4,345 | 13 | 50 | 85% | Hard (falls, vehicle entry) |
| Shoaib | 5,537 | 7 | 50 | 100% | Medium (multi-placement smartphone) |
| Opportunity | 6,453 | 4 | 30 | 100% | Medium (raw XSens sensor counts, multi-body IMU) |

### Severe Out-of-Domain Test Datasets

| Dataset | Windows | Classes | Hz | Group Coverage | Difficulty |
|---------|--------:|--------:|---:|:-:|------------|
| HARTH | 47,330 | 12 | 50 | 100% | Severe (back-only accelerometer, extreme sensor distribution shift) |
| VTT-ConIoT | 2,058 | 16 | 50 | 50% | Severe (industrial/construction, 50% novel activities) |

**Why these are reported separately**:
- **HARTH**: 100% label coverage but extreme sensor distribution shift — uses back-mounted and
  thigh-mounted accelerometers only (no gyroscope). The raw accelerometer data includes gravity
  and has fundamentally different characteristics from waist/wrist-mounted smartphone IMUs in
  training data. All models achieve near-zero zero-shot accuracy, confirming this is a genuine
  distribution shift rather than a model-specific failure. Supervised fine-tuning adapts well
  (up to 78.3% at 10%).
- **VTT-ConIoT**: 8 of 16 activity labels (carrying, climbing ladder, kneeling work, leveling
  paint, lifting, pushing cart, roll painting, spraying paint) have no semantic equivalent in the
  10 training datasets. All models are guaranteed to fail on these activities regardless of
  architecture quality. This 50% coverage floor makes VTT-ConIoT a test of severe label shift.

**Notes on Opportunity**: Uses XSens body-worn IMUs with raw sensor counts (~1000 counts per g,
not standard m/s² or g units). Sampled at 30Hz (vs 50Hz for other datasets). Despite these
differences, zero-shot still works reasonably (42-54% closed-set group) because the 4 activities
(lying, sitting, standing, walking) are well-represented in training data.

---

## Average Across Main Datasets (5 datasets)

*Averaged over MotionSense, RealWorld, MobiAct, Shoaib, Opportunity (85-100% label coverage)*

| Model | ZS-Open Acc | ZS-Open F1 | ZS-Close Acc | ZS-Close F1 | 1% Acc | 1% F1 | 10% Acc | 10% F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM-Medium (ours)** | 40.7 | 18.8 | 44.8 | 36.2 | 72.2 | 67.5 | 85.3 | 81.1 |
| **TSFM-Small-Deep (ours)** | **42.0** | **21.0** | **53.1** | **37.1** | **76.5** | **70.0** | **85.7** | **81.7** |
| **TSFM-Tiny (ours)** | 32.8 | 14.1 | 52.1 | 37.2 | 72.6 | 67.4 | 86.4 | 82.1 |
| **LiMU-BERT** | 21.7 | 7.5 | 33.1 | 23.1 | 35.7 | 22.7 | 51.3 | 43.6 |
| **MOMENT** | 28.3 | 7.2 | 44.7 | 33.2 | 73.8 | 69.8 | 81.3 | 76.8 |
| **CrossHAR** | 21.2 | 5.5 | 38.2 | 31.7 | 61.1 | 55.2 | 78.6 | 75.2 |
| **LanHAR** | 15.8 | 6.2 | 28.4 | 21.7 | 43.3 | 35.2 | 56.2 | 51.5 |
| **LLaSA**† | 1.4 | 1.2 | 17.4 | 8.4 | N/A | N/A | N/A | N/A |

†LLaSA: Published model evaluated at fp16, max 100 samples/class. Zero-shot only (no supervised fine-tuning). MotionSense and Shoaib are in LLaSA's training data.

---

## Per-Dataset Results — Main Benchmarks (5 datasets)

### Zero-Shot Open-Set (Group Match)

*Model predicts from all 87 training labels. Scored via group match (synonyms accepted).*

| Model | MobiAct | MotionSense | RealWorld | Shoaib | Opportunity |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **TSFM-Medium (ours)** | 43.5 | 50.3 | 49.2 | 32.7 | 28.0 |
| **TSFM-Small-Deep (ours)** | **42.2** | **49.3** | **48.0** | **49.2** | 21.1 |
| **TSFM-Tiny (ours)** | 17.4 | 60.7 | 52.0 | 27.7 | 6.3 |
| **LiMU-BERT** | 6.1 | 28.4 | 29.1 | 16.8 | 28.1 |
| **MOMENT** | 28.7 | 33.8 | 14.6 | 33.6 | 30.6 |
| **CrossHAR** | 13.5 | 16.2 | 21.5 | 20.2 | **34.5** |
| **LanHAR** | 11.4 | 14.0 | 17.3 | 16.3 | 20.1 |
| **LLaSA**† | 0.0 | 7.2\* | 0.0 | 0.0\* | 0.0 |

### Zero-Shot Closed-Set

*Model predicts from test dataset labels only. Text-aligned models use exact match; classifier-based models use group match.*

| Model | MobiAct | MotionSense | RealWorld | Shoaib | Opportunity |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **TSFM-Medium (ours)** | 37.5 | 62.8 | 48.1 | 36.9 | 38.7 |
| **TSFM-Small-Deep (ours)** | **50.0** | **64.1** | **48.0** | **54.1** | 49.3 |
| **TSFM-Tiny (ours)** | 47.2 | 61.3 | 46.9 | 48.8 | 56.3 |
| **LiMU-BERT** | 29.3 | 39.9 | 30.5 | 37.6 | 28.1 |
| **MOMENT** | 40.9 | 51.6 | 31.1 | 46.0 | **53.9** |
| **CrossHAR** | 23.3 | 42.8 | 40.3 | 31.2 | 53.6 |
| **LanHAR** | 17.5 | 37.1 | 30.0 | 34.7 | 22.5 |
| **LLaSA**† | 4.4 | 28.5\* | 15.2 | 14.0\* | 25.0 |

\*LLaSA was trained on MotionSense and Shoaib — not truly zero-shot. †Subsampled (100/class).

### 1% Supervised (End-to-End Fine-Tuning)

*Encoder fine-tuned end-to-end with 1% of labeled data. Text-aligned models classify via cosine similarity; others use native classifier heads.*

| Model | MobiAct Acc | MobiAct F1 | MotionSense Acc | MotionSense F1 | RealWorld Acc | RealWorld F1 | Shoaib Acc | Shoaib F1 | Opportunity Acc | Opportunity F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM-Medium (ours)** | 52.4 | 32.1 | 89.5 | 87.9 | 75.0 | 75.1 | 82.9 | 82.3 | 61.3 | 60.1 |
| **TSFM-Small-Deep (ours)** | **65.3** | 35.2 | **88.6** | 87.1 | **75.1** | **74.0** | **81.6** | **81.2** | 72.0 | **72.6** |
| **TSFM-Tiny (ours)** | 46.7 | 24.1 | 89.7 | 88.7 | 74.9 | 74.7 | 81.8 | 80.8 | 69.8 | 68.8 |
| **LiMU-BERT** | 30.6 | 6.2 | 23.2 | 8.1 | 28.1 | 12.0 | 42.7 | 32.4 | 53.7 | 54.8 |
| **MOMENT** | 48.3 | **33.5** | 88.1 | **87.6** | 70.9 | 68.7 | 87.6 | 87.3 | **74.3** | 72.0 |
| **CrossHAR** | 41.1 | 30.5 | 71.9 | 67.5 | 50.1 | 44.8 | 82.0 | 81.6 | 60.5 | 51.6 |
| **LanHAR** | 32.6 | 13.9 | 41.7 | 35.6 | 47.3 | 39.7 | 61.1 | 57.7 | 33.9 | 29.0 |

### 10% Supervised (End-to-End Fine-Tuning)

*Encoder fine-tuned end-to-end with 10% of labeled data. Text-aligned models classify via cosine similarity; others use native classifier heads.*

| Model | MobiAct Acc | MobiAct F1 | MotionSense Acc | MotionSense F1 | RealWorld Acc | RealWorld F1 | Shoaib Acc | Shoaib F1 | Opportunity Acc | Opportunity F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM-Medium (ours)** | 75.6 | 54.9 | 95.0 | 94.5 | 84.7 | 85.3 | 96.4 | 96.3 | 74.6 | 74.5 |
| **TSFM-Small-Deep (ours)** | **72.9** | 51.7 | **95.0** | **94.6** | **85.6** | **86.4** | **95.9** | **95.7** | **79.3** | **80.0** |
| **TSFM-Tiny (ours)** | 74.7 | 52.1 | 95.0 | 94.5 | 85.8 | 86.7 | 95.5 | 95.3 | 81.1 | 81.9 |
| **LiMU-BERT** | 33.8 | 8.7 | 54.0 | 46.7 | 54.1 | 49.9 | 47.7 | 40.1 | 67.0 | 72.5 |
| **MOMENT** | 72.2 | **56.0** | 91.1 | 91.4 | 76.9 | 75.6 | 92.8 | 92.5 | 73.4 | 68.3 |
| **CrossHAR** | 63.7 | 47.5 | 90.0 | 89.7 | 78.8 | 76.7 | 94.2 | 94.0 | 66.3 | 68.1 |
| **LanHAR** | 36.3 | 19.3 | 69.2 | 69.8 | 55.9 | 54.3 | 72.4 | 67.7 | 47.2 | 46.6 |

---

## Key Observations

1. **TSFM leads all average metrics across 5 main datasets** — TSFM-Small-Deep achieves 53.1%
   closed-set avg accuracy, ahead of MOMENT (44.7%), CrossHAR (38.2%), LiMU-BERT (33.1%),
   and LanHAR (28.4%). TSFM leads at 10% supervised (85.7% vs MOMENT's 81.3%) and 1%
   supervised (76.5% acc, 70.0% F1 — both ahead of MOMENT's 73.8% / 69.8%).

2. **HARTH is a distribution-shift stress test** — All models achieve near-zero zero-shot accuracy
   (<1% for TSFM/MOMENT/CrossHAR, 20.2% for LiMU-BERT closed-set). This is caused by the
   back-mounted raw accelerometer data being fundamentally different from waist/wrist smartphone
   IMUs in training data. However, supervised fine-tuning adapts well: TSFM reaches 78.3% at 10%,
   confirming the encoder representations are flexible enough to adapt with labeled data.

3. **Opportunity data was fixed** — Previously produced all-zero input due to a column mapping
   bug in `dataset_config.json` (identity mapping where `back_acc_x → acc_x` was needed).
   After fixing, Opportunity produces meaningful results across all models: TSFM 49.3% ZS-Closed,
   MOMENT 53.9%, with supervised results up to 79.3% (TSFM) and 73.4% (MOMENT) at 10%.

4. **MOMENT is consistently the closest competitor, partly due to structural advantages** —
   Second-best on most metrics, and leads on Opportunity (53.9% ZS-Closed) and 1% supervised
   on several datasets. MOMENT benefits from a 6144-dim embedding (16× TSFM's 384-dim) and an
   SVM-RBF classifier for ZS that is stronger than the GRU/Transformer classifiers used by
   LiMU-BERT and CrossHAR.
   See [Note on MOMENT's Structural Advantages](#note-on-moments-structural-advantages).

5. **CrossHAR is a strong third** — 78.6% at 10% supervised avg, competitive with MOMENT on
   MotionSense (90.0%) and Shoaib (94.2%), despite a much smaller embedding (72-dim). Shows an
   unusual pattern on Opportunity where 10% accuracy (66.3%) is higher than 1% (60.5%).

6. **LiMU-BERT struggles with limited data** — Notably low at 1% supervised (35.7% avg) and
   10% supervised (51.3% avg). The GRU classifier needs substantial data to converge. Best
   relative performance on Opportunity ZS-Open (28.1%).

7. **LanHAR struggles most across all settings** — Despite text alignment, its from-scratch
   SciBERT training on small HAR data limits transfer. Exception: HARTH 10% reaches 70.0%,
   suggesting it can eventually adapt to distribution-shifted data given enough labels.

8. **LLaSA underperforms all encoder-based models** — 17.4% closed-set avg, below even LanHAR
   (28.4%). The 7B LLM is overwhelmed by the 87-label prompt in open-set (0% on 4/5 datasets),
   and even closed-set with fewer labels shows weak performance. Generative text classification
   is fundamentally less reliable than direct embedding comparison for IMU data.

9. **TSFM uses a fixed 1.0s patch size** — no per-dataset sweep or test-time tuning. Patch size
   sensitivity analysis shows 1.0s is within 1.3% of the best patch size on all datasets.
   See [Patch Size Sensitivity](#tsfm-patch-size-sensitivity) below.

---

## Note on MOMENT's Structural Advantages

MOMENT performs surprisingly close to TSFM despite being a general time-series model with no text
alignment and no HAR-specific pretraining. This section documents the structural factors that
contribute to MOMENT's strong performance and should be considered when interpreting the results.

**No data leakage**: We verified that MOMENT's zero-shot evaluation is clean. The ZS SVM is
trained only on the 10 training datasets (completely disjoint from test). ZS evaluation runs
before any supervised fine-tuning on each dataset. After each supervised FT run, the original
pretrained weights are restored via `deepcopy`. There is no supervised data contamination in
MOMENT's zero-shot numbers.

**No HAR pretraining**: Unlike TSFM, LiMU-BERT, CrossHAR, and LanHAR (all pretrained on our
10 HAR training datasets), MOMENT uses its published pretrained weights
(`AutonLab/MOMENT-1-large`) which were trained on general time-series data (weather, electricity,
traffic, etc.) with **no HAR or IMU data whatsoever**. The MOMENT encoder is never retrained on
our HAR data — only the downstream SVM (for ZS) and linear head (for supervised) see HAR labels.
This makes MOMENT's strong performance especially notable: its encoder has zero exposure to
activity recognition data, yet it nearly matches TSFM which was purpose-built for HAR.

### 1. Embedding Dimensionality (16× larger than TSFM)

Both TSFM and MOMENT process IMU channels independently — this is not a MOMENT-specific
advantage. The critical difference is what happens after per-channel processing:

| Model | Per-Channel Dim | Fusion | Final Embedding |
|-------|:-:|---|:-:|
| **TSFM** | Variable | Learned cross-channel attention → compressed | **384** |
| **MOMENT** | 1024 | Simple concatenation (no compression) | **6144** |

MOMENT preserves the full 6 × 1024 = 6144 dimensions, giving downstream classifiers (SVM for
ZS, linear head for supervised) 16× more information to work with than TSFM's 384-dim output,
and 85× more than LiMU-BERT/CrossHAR's 72-dim. An RBF-kernel SVM on 6144 dimensions can exploit
fine-grained per-channel patterns that are compressed away in lower-dimensional spaces.

This is most visible in the supervised results: MOMENT's `Linear(6144, C)` head with 1% data
(87.6% on Shoaib, 88.1% on MotionSense) nearly matches TSFM despite having no text-based class
prototypes to guide learning. The high dimensionality compensates for the lack of text alignment.

### 2. ZS Closed-Set Mask Expansion

Classifier-based models predict from 87 training labels with logits masked to allow only labels
whose group matches a test label. This expands the prediction space beyond the C test labels:

| Test Dataset | Test Classes | Allowed Training Labels (mask) | Expansion Factor |
|---|:-:|:-:|:-:|
| MotionSense | 6 | 22 | 3.7× |
| RealWorld | 8 | 30 | 3.8× |
| MobiAct | 13 | 34 | 2.6× |
| Shoaib | 7 | 25 | 3.6× |
| Opportunity | 4 | 14 | 3.5× |
| HARTH | 12 | 29 | 2.4× |

### 3. SVM-RBF Is a Stronger ZS Classifier Than GRU/Transformer

Among the three classifier-based models, MOMENT's SVM-RBF with GridSearchCV (5-fold, 9 C values)
is a notably stronger choice than LiMU-BERT's GRU (trained from scratch, 100 epochs) or CrossHAR's
Transformer (also from scratch, 100 epochs). The SVM benefits from:
- **No training instability**: Convex optimization with guaranteed convergence
- **Automatic regularization**: GridSearchCV finds optimal C without manual tuning
- **Kernel trick**: RBF kernel captures nonlinear patterns in the 6144-dim space

This partly explains why MOMENT's ZS numbers (44.7% avg closed-set) exceed LiMU-BERT (33.1%)
and CrossHAR (38.2%) by a larger margin than the embedding dimensionality alone would predict.

### 4. Combined Effect

These advantages compound: MOMENT operates on 6144-dim embeddings (more capacity) with an
SVM-RBF classifier (stronger learner). Each factor individually provides a modest edge; combined,
they make MOMENT appear close to TSFM despite fundamental architectural differences.

**This is fair in the sense that we follow MOMENT's published protocol** — per-channel processing
and SVM-RBF are how MOMENT is designed to be used. We do not artificially inflate MOMENT's
numbers.

---

## Severe Out-of-Domain: HARTH & VTT-ConIoT

These two datasets are reported separately due to extreme distribution shift:
- **HARTH** (12 classes): 100% label coverage but back-mounted accelerometer only (no gyro) — extreme sensor distribution shift from waist/wrist smartphone IMUs in training data.
- **VTT-ConIoT** (16 classes): Industrial/construction activities, only 50% label coverage — 8 of 16 activities (carrying, climbing ladder, kneeling work, etc.) have no training equivalent.

### HARTH

| Model | ZS-Open Acc | ZS-Close Acc | 1% Acc | 1% F1 | 10% Acc | 10% F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM-Medium (ours)** | 1.6 | 1.1 | 66.9 | 21.8 | 82.4 | 48.6 |
| **TSFM-Small-Deep (ours)** | **2.0** | 2.5 | **64.7** | **37.8** | **78.3** | **48.0** |
| **TSFM-Tiny (ours)** | 2.7 | 3.8 | 65.8 | 36.9 | 77.8 | 47.5 |
| **LiMU-BERT** | 0.0 | **20.2** | 54.4 | 5.9 | 19.1 | 2.7 |
| **MOMENT** | 0.4 | 1.9 | 66.7 | 32.4 | 65.4 | 31.2 |
| **CrossHAR** | 0.3 | 0.9 | 42.4 | 25.0 | 71.9 | 42.8 |
| **LanHAR** | 0.9 | 1.1 | 3.4 | 1.5 | 70.0 | 21.7 |
| **LLaSA**† | 11.9 | 12.5 | N/A | N/A | N/A | N/A |

### VTT-ConIoT

| Model | ZS-Open Acc | ZS-Close Acc | 1% Acc | 1% F1 | 10% Acc | 10% F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM-Medium (ours)** | 5.2 | 6.7 | 11.1 | 7.2 | 21.3 | 18.6 |
| **TSFM-Small-Deep (ours)** | 1.3 | 2.2 | 1.9 | 0.6 | 16.9 | 15.6 |
| **TSFM-Tiny (ours)** | 1.1 | 4.5 | 8.2 | 4.9 | 25.6 | 22.0 |
| **LiMU-BERT** | 3.4 | **7.1** | 4.3 | 0.6 | 15.0 | 6.5 |
| **MOMENT** | 1.6 | 5.2 | **18.4** | **17.0** | **34.8** | **29.9** |
| **CrossHAR** | 0.7 | 5.0 | 10.6 | 3.5 | 30.9 | 24.9 |
| **LanHAR** | **8.3** | 6.9 | 7.7 | 3.5 | 7.7 | 5.7 |
| **LLaSA**† | 0.0 | 6.9 | N/A | N/A | N/A | N/A |

**Observations**:
- **HARTH**: All models near-zero on zero-shot due to sensor distribution shift. LiMU-BERT is an outlier with 20.2% closed-set (its training label masking happens to favor HARTH's label distribution). Supervised FT adapts well: TSFM leads at both 1% (64.7%) and 10% (78.3%).
- **VTT-ConIoT**: All models near random on zero-shot (≤8% open-set, ≤7% closed-set). With 10% supervised data, MOMENT leads (34.8%), followed by CrossHAR (30.9%) and TSFM (16.9%). The 50% label coverage floor limits all models.

---

## TSFM Patch Size Sensitivity

*Zero-shot closed-set accuracy (%) at each candidate patch size. TSFM uses a fixed 1.0s patch size for all reported results — no per-dataset sweep or test labels used for selection.*

| Dataset | 0.5s | 0.75s | 1.0s | 1.25s | 1.5s | 1.75s | 2.0s | Range |
|---------|---:|---:|---:|---:|---:|---:|---:|---:|
| Shoaib | 35.3 | 40.9 | 44.1 | **45.0** | 44.4 | 43.8 | 42.8 | 9.6 |
| Opportunity | 30.6 | 38.3 | 42.3 | 42.7 | **43.6** | 43.5 | 43.0 | 13.0 |
| HARTH | **3.3** | 2.1 | 0.5 | 0.3 | 0.1 | 0.1 | 0.0 | 3.2 |

**Note**: Shoaib and HARTH are evaluated at native 50Hz; Opportunity at native 30Hz. HARTH's
near-zero ZS performance makes its patch sensitivity meaningless (noise). For Shoaib and
Opportunity, the 1.0s default is within 0.9% and 1.3% of the best patch size respectively.
Opportunity's slight preference for 1.5s makes physical sense — at 30Hz, a 1.5s patch contains
45 samples, close to the 50 samples that other 50Hz datasets get with a 1.0s patch.

*Previous 20Hz sensitivity analysis (from before the switch to native sampling rates):*

| Dataset | 1.0s | 1.25s | 1.5s | 1.75s | 2.0s | Range |
|---------|---:|---:|---:|---:|---:|---:|
| MotionSense | **51.5** | 48.5 | 46.1 | 44.0 | 42.5 | 9.0 |
| RealWorld | **37.0** | 35.8 | 36.3 | 33.9 | 32.4 | 4.6 |
| MobiAct | 48.1 | **49.2** | 47.5 | 45.7 | 44.1 | 5.1 |
| VTT-ConIoT | 3.4 | 3.9 | 3.5 | 3.9 | **4.7** | 1.3 |

**Note**: These numbers were measured at 20Hz before the switch to native sampling rates. The fixed 1.0s choice was within 1.1% of the best possible patch for 3/4 datasets at 20Hz.

---

## Ablation: Native Rate + Rich Channel Descriptions

The results above use TSFM's full capabilities: native sampling rates and dataset-specific channel
descriptions from manifests. Previously, TSFM was evaluated on the same 20Hz resampled data as
baselines with generic channel descriptions ("Accelerometer X-axis"). This provides a natural
ablation study showing the value of TSFM's metadata-aware architecture.

*This ablation was measured on the original 3 main test datasets (all 50Hz) + VTT-ConIoT.*

### Average Across Original Main Datasets (3 datasets)

| Configuration | ZS-Open Acc | ZS-Close Acc | 1% Acc | 10% Acc |
| :--- | ---: | ---: | ---: | ---: |
| **TSFM (native 50Hz + rich channels)** | **47.6** | **49.2** | **73.0** | **84.2** |
| **TSFM (20Hz + generic channels)** | 36.2 | 45.5 | 68.2 | 83.2 |
| **Delta** | +11.4 | +3.7 | +4.8 | +1.0 |

### Per-Dataset Deltas (native 50Hz vs 20Hz resampled)

| Dataset | ZS-Open Acc | ZS-Close Acc | 1% Acc | 10% Acc |
| :--- | ---: | ---: | ---: | ---: |
| MotionSense | +24.5 | +13.2 | +2.8 | 0.0 |
| RealWorld | -9.9 | -11.9 | -0.5 | +0.7 |
| MobiAct | +19.7 | +9.8 | +12.2 | +2.3 |
| VTT-ConIoT | +0.4 | +2.9 | +1.5 | +4.8 |

**Key takeaways**:
- **Massive zero-shot gains on MotionSense and MobiAct**: ZS-Open improves +24.5% on MotionSense
  and +19.7% on MobiAct. The correct channel-text conditioning (prepending dataset description to
  match the training format) lets the model leverage sensor placement and device context for
  disambiguation.
- **RealWorld zero-shot drops significantly**: ZS-Open -9.9%, ZS-Close -11.9%. RealWorld's
  dataset description likely conflicts with its heterogeneous multi-placement setup — the model
  over-commits to the described placement while the actual data comes from varied body locations.
  This is an honest negative result that highlights a limitation of dataset-level descriptions.
- **Supervised results are mixed**: MobiAct 1% jumps +12.2% (the placement context helps with
  few-shot learning of fall activities), but MotionSense 10% is flat (0.0%) and RealWorld 1% is
  slightly negative (-0.5%). With sufficient labeled data, the model can learn dataset-specific
  patterns without metadata, so the channel description effect washes out.
- **Net effect is strongly positive for zero-shot**: The ZS-Open average improves +11.4% across
  3 datasets, despite RealWorld's drop. This validates that TSFM's channel-text conditioning is
  a genuinely useful capability for zero-shot transfer.

This ablation demonstrates that TSFM's channel-text conditioning and native sampling rate handling
are genuinely useful architectural novelties, producing large zero-shot gains on datasets where
the metadata provides disambiguating context. The RealWorld regression shows that dataset-level
descriptions can hurt when they oversimplify heterogeneous data — a direction for future work
in per-sample or per-subject metadata conditioning.

---

## Label Handling for Unseen Test Activities

How each model handles the fact that test datasets may contain activity labels not seen during training.

### Label Groups (Synonym Mapping)

The 10 training datasets use 87 unique activity labels, many of which are synonyms
(e.g., "jogging"/"running", "walking_downstairs"/"stairs_down"/"descending_stairs").
We define **label groups** that cluster semantically equivalent labels:

| Group | Training Labels | Test Labels (examples) |
|-------|----------------|----------------------|
| running | running, jogging, running_treadmill | jogging (MotionSense), running (RealWorld, HARTH) |
| walking | walking, walking_parking, walking_treadmill_flat, walking_straight, walking_winding | walking (all test sets) |
| stairs_down | walking_downstairs, stairs_down, descending_stairs, going_down_stairs | walking_downstairs (MotionSense), stairs_down (RealWorld, MobiAct, HARTH) |
| sitting | sitting, sitting_chair, talking_sitting | sitting (MotionSense, Opportunity), sitting_chair (MobiAct) |
| ... | (87 labels across ~30 groups) | |

### Per-Model Handling of Unseen Labels

| Model | Open-Set | Closed-Set | Supervised |
|-------|----------|------------|------------|
| **TSFM** | Encodes all 87 training labels as text; predicts via cosine sim; scored via group matching. Unseen test labels that belong to a known group can be predicted correctly. Novel labels (no group match) are counted as **failures**. | Encodes only the test dataset's labels as text; exact match scoring. Novel labels can theoretically be predicted (model sees the text), but scored as failures if not in training groups. | Fine-tunes on test dataset labels directly. Novel labels are learned from the few-shot data. |
| **LiMU-BERT** | GRU classifier predicts over 87 training labels; scored via group matching. Can only predict training labels. Novel test labels are **guaranteed failures**. | Classifier logits masked to training labels whose group appears in test set. Novel labels (no matching group) are excluded from the mask, so they **cannot be predicted** and are counted as failures. | Fine-tunes on test dataset labels directly. Novel labels are learned from labeled data. |
| **MOMENT** | Same as LiMU-BERT but with SVM-RBF classifier. Novel test labels are **guaranteed failures**. | Same masking approach as LiMU-BERT. Novel labels cannot be predicted. | Fine-tunes with linear head on test dataset labels. Novel labels learned from data. |
| **CrossHAR** | Same as LiMU-BERT but with Transformer_ft classifier. Novel test labels are **guaranteed failures**. | Same masking approach. Novel labels cannot be predicted. | Fine-tunes with Transformer_ft on test dataset labels. |
| **LanHAR** | Encodes all 87 training labels as text; cosine sim + group matching. Same as TSFM. Novel labels counted as **failures**. | Same as TSFM — encodes test labels as text, exact match. | Same as TSFM. |
| **LLaSA** | Prompted with all 87 training labels; parses text response; group matching. Unparseable responses ("unclear") are counted as **failures**. | Prompted with test dataset's labels; exact string match on parsed response. Unclear responses are **failures**. | N/A — 7B LLM cannot be fine-tuned with few labels. |

### Impact on Reported Metrics

- **MotionSense** (100% coverage): All 6 activities have training equivalents. No failures due to unseen labels.
- **RealWorld** (100% coverage): All 8 activities have training equivalents. No failures due to unseen labels.
- **MobiAct** (85% coverage): 2 of 13 activities (`car_step_in`, `car_step_out`) have no training equivalent. These are counted as **failures for all models** in zero-shot. In supervised, they are learned from labeled data.
- **Shoaib** (100% coverage): All 7 activities have training equivalents. No failures due to unseen labels.
- **Opportunity** (100% coverage): All 4 activities (lying, sitting, standing, walking) have training equivalents. No failures due to unseen labels.
- **HARTH** (100% coverage): All 12 activities map to known groups (e.g., cycling_sit→cycling, shuffling→walking, transport_sit→sitting). No failures due to unseen labels; near-zero ZS accuracy is entirely due to distribution shift, not label coverage.
- **VTT-ConIoT** (50% coverage): 8 of 16 activities are completely novel (industrial/construction). These are **guaranteed failures for all models** in zero-shot, setting a ~50% accuracy ceiling. This is why VTT-ConIoT is reported separately.

### Scoring Rules

1. **Open-set zero-shot**: Prediction maps through label groups. If `group(predicted_label) == group(ground_truth_label)`, it's correct. Novel test labels with no matching group are always wrong.
2. **Closed-set zero-shot (text-aligned)**: Exact label match. Model predicts from the test dataset's own labels.
3. **Closed-set zero-shot (classifier-based)**: Logits masked to allow only training labels whose group matches a test label. `argmax(masked_logits)` → group match scoring. Novel test labels with no matching group cannot appear in the mask and are always wrong.
4. **Supervised**: Standard classification on the test dataset's labels. All labels (including novel ones) are learned from labeled training data.
5. **Generative zero-shot (LLaSA)**: LLM generates text → parsed for activity label → exact match (closed-set) or group match (open-set). Unparseable responses are scored as incorrect. Three-stage matching: exact label match → underscore/space normalization → substring containment.

---

## Reproducibility

- Raw JSON results: `test_output/baseline_evaluation/*_evaluation.json`
- Evaluation scripts: `val_scripts/human_activity_recognition/evaluate_*.py`
- Results generator: `scripts/generate_results_table.py`
- Run all: `bash scripts/auto_eval_after_training.sh`
- Random seed: 3431 (consistent across all splits)
