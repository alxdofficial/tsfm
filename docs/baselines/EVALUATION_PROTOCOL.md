# Evaluation Protocol

Unified 4-metric evaluation framework for comparing TSFM against baselines on 3 main test datasets plus 1 severe out-of-domain dataset.

## Design Principles

This protocol is designed to ensure **fair, reproducible comparison** across fundamentally different model architectures:

1. **No test data during training**: All 5 models train on the same 10 datasets; all 4 test datasets are strictly held out. No model ever sees test data during any training stage.
2. **Native architectures**: Each baseline uses its own paper's downstream classifier — we do not impose a uniform architecture that might favor or penalize any model.
3. **Identical test windows**: All models evaluate on the same 6-second windows and labels with the same data splits (same seed). TSFM uses native-rate windows (50Hz); baselines use 20Hz-resampled windows — see **Sampling Rate Policy** below.
4. **Multiple metrics**: 4 metrics spanning zero-shot to supervised fine-tuning capture different aspects of representation quality, avoiding cherry-picking a single favorable metric.
5. **Reproducibility**: All evaluators set global seeds (42) and classifier seeds (3431) for deterministic results.

## Test Datasets

All models are evaluated on 4 datasets that were **never seen during training**: 3 main test
datasets with high label group coverage (85-100%), plus 1 severe out-of-domain dataset.

### Main Test Datasets (85-100% label coverage)

| Dataset | Windows | Activities | Group Coverage | Difficulty |
|---------|---------|------------|:-:|------------|
| MotionSense | 12,080 | 6 | 100% | Easy (basic locomotion) |
| RealWorld | 27,138 | 8 | 100% | Medium (multi-placement) |
| MobiAct | 4,345 | 13 | 85% | Hard (includes falls, novel activities) |

### Severe Out-of-Domain Dataset (50% label coverage)

| Dataset | Windows | Activities | Group Coverage | Difficulty |
|---------|---------|------------|:-:|------------|
| VTT-ConIoT | 2,058 | 16 | 50% | Severe (industrial/construction) |

**Why VTT-ConIoT is reported separately**: 8 of 16 activity labels have no semantic equivalent
in the 10 training datasets. All models are guaranteed to fail on these activities regardless
of architecture quality. This 50% coverage floor makes VTT-ConIoT a test of severe domain
shift rather than cross-dataset generalization. Main results average over the 3 main datasets;
VTT-ConIoT is reported in a dedicated section.

Evaluation data format depends on each model's sampling rate capability (see **Sampling Rate Policy** below).

## Sampling Rate Policy

**Principle**: Each model should train and evaluate at each dataset's **native sampling rate** whenever
its architecture supports it. We only resample when the model architecturally requires a fixed rate.

### Native Sampling Rates

| Dataset | Native Hz | Role |
|---------|:---------:|------|
| UCI HAR | 50 | Train |
| HHAR | 50 | Train |
| PAMAP2 | 100 | Train |
| WISDM | 20 | Train |
| DSADS | 25 | Train |
| KU-HAR | 100 | Train |
| UniMiB SHAR | 50 | Train |
| HAPT | 50 | Train |
| MHEALTH | 50 | Train |
| RecGym | 20 | Train |
| MotionSense | 50 | Test |
| RealWorld HAR | 50 | Test |
| MobiAct | 50 | Test |
| VTT-ConIoT | 50 | Test |

### Per-Model Sampling Rate Handling

| Model | Supports Native Rates? | Training Rate | Evaluation Rate | Rationale |
|-------|:----------------------:|:-------------:|:---------------:|-----------|
| **TSFM** | Yes | Native per dataset | Native per dataset | Seconds-based patch tokenization + interpolation to fixed 64 steps decouples the model from any specific sampling rate. |
| **LiMU-BERT** | No | 20 Hz (resampled) | 20 Hz (resampled) | Learned positional embedding fixed at 120 positions; paper explicitly designed for 20 Hz. |
| **CrossHAR** | No | 20 Hz (resampled) | 20 Hz (resampled) | Inherits LiMU-BERT data format; learned positional embedding fixed at 120 positions. |
| **MOMENT** | No (rate-agnostic) | 20 Hz (resampled) | 20 Hz (resampled) | Processes any sequence of up to 512 numbers with no notion of physical time. Paper: *"We did not explicitly model temporal resolution."* Has no mechanism to adapt to or benefit from different rates, so resampled data is used for consistency with the benchmark format. |
| **LanHAR** | No | 20 Hz (resampled) | 20 Hz (resampled) | Paper designed for 50 Hz, but our benchmark standardizes non-TSFM data at 20 Hz (LiMU-BERT format). Butterworth filter frequencies are absolute Hz and work correctly at 20 Hz. Sensor encoder trains from scratch, so it learns patterns at whatever rate it receives. |

**Why LiMU-BERT/CrossHAR require 20 Hz**: Both use `nn.Embedding(120, hidden)` learned positional
embeddings that encode temporal relationships assuming 120 steps = 6 seconds at 20 Hz. Feeding data
at a different rate would change the physical duration each position represents, invalidating the
learned temporal patterns.

**Why MOMENT uses 20 Hz despite being rate-agnostic**: MOMENT treats all input as raw number sequences
with no frequency awareness. It would process 50 Hz and 20 Hz data identically (same 8-timestep patches,
same left-padding to 512). Since it cannot benefit from native rates and the benchmark data pipeline
already produces 20 Hz windows for LiMU-BERT/CrossHAR, we use the same 20 Hz data for simplicity.

**Why LanHAR uses 20 Hz instead of its paper's 50 Hz**: The LanHAR paper uses 50 Hz with 120-sample
windows (2.4 seconds per window). Our benchmark uses 20 Hz with 120-sample windows (6.0 seconds per
window). Since LanHAR trains its sensor encoder from scratch within our evaluation pipeline (no
pretrained weights from the paper), the encoder learns temporal patterns at the rate it receives.
The gravity alignment and Butterworth filter preprocessing correctly adapts to 20 Hz via the `fs`
parameter. Using 20 Hz allows LanHAR to share the same benchmark data files as LiMU-BERT and CrossHAR.

**Why TSFM uses native rates**: TSFM's `create_patches()` takes `sampling_rate_hz` as an explicit
parameter and specifies patch sizes in seconds (e.g., `patch_size_sec=1.0`). A 1-second patch at
50 Hz has 50 timesteps; at 100 Hz it has 100 timesteps. Both are interpolated to a fixed 64-step
representation via `F.interpolate`, producing rate-invariant patch tokens. This means TSFM sees
the full spectral content of each dataset at its native resolution — no information is lost to
downsampling. During training, the `MultiDatasetLoader` reads each dataset's native rate from
its manifest and passes it to the preprocessing pipeline.

### TSFM's Per-Dataset Metadata

TSFM uses three per-dataset metadata signals that no baseline uses:

| Metadata | Source | How It's Used |
|----------|--------|---------------|
| **Sampling rate** | Dataset manifest (`sampling_rate_hz`) | Converts seconds-based patch size to timesteps: `patch_timesteps = int(sampling_rate_hz * patch_size_sec)`. Ensures patches represent consistent physical durations across datasets with different rates. |
| **Patch size** | Per-dataset config (seconds) | Specifies temporal duration of each patch. Supports **patch size augmentation** during training: randomly samples from a configurable `(min_sec, max_sec, step_sec)` range per dataset, forcing the model to learn resolution-robust representations. Fixed at 1.0s during evaluation. |
| **Channel descriptions** | Dataset manifest (text strings, e.g., "Accelerometer X-axis") | Encoded by frozen SentenceBERT into dense vectors, then fused into sensor features via `ChannelSemanticEncoding`. Provides the model with semantic awareness of what each channel measures, supporting generalization across sensor configurations and placements. |

All three are passed through the training pipeline on every batch: `MultiDatasetLoader` reads them
from the manifest → training loop extracts them from metadata → `encoder.preprocess()` uses sampling
rate and patch size → `model.forward()` uses channel descriptions. Baselines have no equivalent —
they operate on fixed-format tensors with no dataset-level metadata.

## 4-Metric Framework

### Metric 1: Zero-Shot Open-Set

- **What**: Classify against ALL 87 training labels, score via synonym groups
- **Why**: Tests open-vocabulary generalization — can the model find the right activity among all possible labels?
- **Text-aligned (TSFM, LanHAR)**: `argmax(sensor_emb @ all_87_label_embs.T)` via cosine similarity with text embeddings, then group-match prediction to ground truth
- **Classifier-based (LiMU-BERT, MOMENT, CrossHAR)**: Each model's native classifier predicts over all 87 classes, then group-match prediction to ground truth

### Metric 2: Zero-Shot Closed-Set

- **What**: Classify against only labels relevant to the test dataset
- **Why**: Tests discriminative quality when the label space is constrained
- **Text-aligned (TSFM, LanHAR)**: Encode only the test dataset's activity labels as text, `argmax(sensor_emb @ test_label_embs.T)`, exact label matching
- **Classifier-based (LiMU-BERT, MOMENT, CrossHAR)**: Mask each model's native classifier logits/scores to training labels whose synonym group appears in the test dataset, `argmax(masked_logits)`, group-match scoring

**Closed-set mask details**: For each test dataset, a mask over the 87 training labels allows only those whose group is represented in the test set:

| Dataset | Test Groups | Allowed/87 | Notes |
|---------|-------------|------------|-------|
| MotionSense | 6 | 22 | All test labels mappable |
| RealWorld | 8 | 30 | All test labels mappable |
| MobiAct | 9 | 34 | vehicle_entry group has 0 training members |
| VTT-ConIoT | 11 | 20 | 4 groups (carrying, climbing, kneeling, painting) have 0 training members |

### Metric 3: 1% Supervised (End-to-End Fine-Tuning)

- **What**: Fine-tune the encoder end-to-end using 1% of labeled data from the test dataset
- **How**: Split 80/10/10 (train/val/test), subsample 1% of train via balanced_subsample, fine-tune encoder + classifier, evaluate on test split
- **Why**: Tests few-shot transfer — how well does the pretrained encoder adapt with minimal supervision?
- **Text-aligned (TSFM, LanHAR)**: Fine-tune sensor encoder, classify via cosine similarity with frozen text embeddings (no separate classifier head)
- **Non-text-aligned (LiMU-BERT, CrossHAR)**: Fine-tune encoder + native classifier head end-to-end
- **MOMENT**: Fine-tune encoder + linear head (paper's supervised fine-tuning protocol)

### Metric 4: 10% Supervised (End-to-End Fine-Tuning)

- **What**: Same as 1% but with 10% labeled data
- **Why**: Tests semi-supervised regime — practical for real deployments where some labels are available

## Fairness Justifications

### Why native classifiers instead of a shared architecture?

Each baseline's original paper evaluates with its own downstream classifier (e.g., MOMENT uses SVM-RBF, LiMU-BERT uses GRU). Imposing a uniform linear classifier would disadvantage models designed for non-linear classifiers and would not reflect each model's real-world performance. By using each paper's native classifier, we measure each model's representations in the way they were designed to be used.

### Why group-based scoring for classifier-based zero-shot?

Non-text-aligned models can only predict training labels (e.g., "jogging"), not test-specific labels (e.g., "running"). Since "jogging" and "running" are semantically equivalent, we map both through synonym groups before scoring. Without this, classifier-based models would be unfairly penalized for vocabulary mismatch even when their predictions are semantically correct.

### Why different zero-shot mechanisms for text-aligned vs classifier-based?

Text-aligned models (TSFM, LanHAR) can encode arbitrary label text at test time — this is a genuine architectural advantage and a core motivation for text alignment. Classifier-based models cannot do this, so they use their native classifiers trained on training data. This asymmetry is inherent to the model designs, not an evaluation bias. We include both zero-shot AND supervised metrics so readers can assess both capabilities.

### Why does TSFM use a fixed 1.0s patch size?

TSFM's variable-length architecture accepts any patch size, but we use a fixed 1.0s for all test datasets — no per-dataset sweep or test-time tuning. This is a metadata-only decision: at native 50Hz, 1.0s patches (50 timesteps) are interpolated to 64 fixed steps, giving fine temporal resolution. Sensitivity analysis across [1.0, 1.25, 1.5, 1.75, 2.0]s shows results are robust (max 9% range on the easiest dataset, <2% on the hardest), and smaller patches consistently perform best. Other baselines similarly use fixed embedding extraction with no per-dataset tuning.

### Why does zero-shot classifier training not violate the "zero-shot" definition?

For classifier-based baselines, the zero-shot classifier is trained exclusively on embeddings from the 10 **training** datasets — no test data is used. The "zero-shot" refers to the test dataset being unseen, not to the classifier being untrained. This is analogous to how CLIP trains on image-text pairs then evaluates zero-shot on ImageNet: the model is trained, but the test distribution is never seen.

## Adaptations from Original Papers

We adapt each baseline to our unified benchmark. The table below documents every significant
deviation from the original paper's evaluation protocol and our fairness rationale.

| Baseline | Deviation | Original Paper | Our Adaptation | Effect on Baseline |
|----------|-----------|---------------|----------------|-------------------|
| **LiMU-BERT** | Window-level scoring | Scores each 20-step sub-window independently | Majority vote across 6 sub-windows per 120-step window | Neutral (standardizes evaluation unit) |
| **LiMU-BERT** | Single combined model | Separate per-dataset pretrained models | One model pretrained on all 10 datasets | Neutral |
| **CrossHAR** | End-to-end fine-tuning | Freezes encoder; trains only Transformer_ft on static embeddings | Fine-tunes encoder + Transformer_ft end-to-end | Slight advantage (encoder adapts to target) |
| **MOMENT** | Linear head for supervised | Only SVM-RBF on frozen embeddings (no encoder fine-tuning for classification) | End-to-end fine-tuning with linear head from MOMENT codebase | Slight advantage (encoder adapts) |
| **LanHAR** | No target data in Stage 2 | Sensor encoder trains on source + target data combined | Source data only | Slight disadvantage (no target distribution) |
| **LanHAR** | Supervised fine-tuning | Not in paper (zero-shot only) | Cosine sim fine-tuning with frozen text prototypes | N/A (extension) |
| **LiMU-BERT, CrossHAR, LanHAR** | Resampled to 20 Hz | LiMU-BERT/CrossHAR: 20 Hz (same). LanHAR: 50 Hz. | All three use 20 Hz benchmark data | Neutral for LiMU-BERT/CrossHAR; LanHAR trains from scratch so adapts to any rate |
| **MOMENT** | Resampled to 20 Hz | Rate-agnostic (no frequency awareness) | 20 Hz benchmark data | Neutral (model has no concept of sampling rate) |
| **TSFM** | Native sampling rates | Native per dataset | Native per dataset | Slight advantage (preserves full spectral content) |
| **All** | Unified batch sizes | Each paper uses its own (typically 128) | 512 for classifiers, 32 for fine-tuning | Neutral |

**Why these deviations exist**: A cross-baseline benchmark requires standardized evaluation units,
identical data splits, and comparable fine-tuning conditions. Replicating each paper's exact
experiment would produce numbers on different datasets, different splits, and different evaluation
granularities — defeating the purpose of comparison. We prioritize fairness of comparison over
exact paper reproduction, and document every deviation transparently.

## Per-Baseline Summary

### Classifier and Training Overview

| Baseline | Text-Aligned? | ZS Method | Supervised Method | Extra Training for ZS | Embedding Dim |
|----------|:---:|-----------|----------------------|----------------------|:---:|
| **TSFM** | Yes | Cosine sim (LearnableLabelBank) | Fine-tune encoder, cosine sim | None (cosine sim) | 384 |
| **LanHAR** | Yes | Cosine sim (SciBERT prototypes) | Fine-tune sensor encoder, cosine sim | None (cosine sim) | 768 |
| **LiMU-BERT** | No | GRU classifier | Fine-tune encoder + GRU | Train GRU on 87-class training embeddings | 72 |
| **MOMENT** | No | SVM-RBF classifier | Fine-tune encoder + linear head | Train SVM on 87-class training embeddings | 6144 |
| **CrossHAR** | No | Transformer_ft classifier | Fine-tune encoder + Transformer_ft | Train Transformer on 87-class training embeddings | 72 |

**Key**: Text-aligned models require no extra classifier training for zero-shot — they use cosine similarity with text prototypes directly. Classifier-based models require training a classifier on the 10 training datasets' embeddings (using only training data, never test data). For supervised metrics, all models fine-tune their encoder end-to-end with their native classification mechanism.

### Label Group Mapping Coverage per Test Dataset

| Dataset | Total Activities | Mappable | Novel | Coverage | Closed-Set Mask | Unmappable Activities |
|---------|:---:|:---:|:---:|:---:|:---:|---|
| **MotionSense** | 6 | 6 | 0 | 100% | 22/87 | — |
| **RealWorld** | 8 | 8 | 0 | 100% | 30/87 | — |
| **MobiAct** | 13 | 11 | 2 | 85% | 34/87 | car_step_in, car_step_out |
| **VTT-ConIoT** | 16 | 8 | 8 | 50% | 20/87 | carrying, climbing_ladder, kneeling_work, leveling_paint, lifting, pushing_cart, roll_painting, spraying_paint |

**How to read this table**:
- **Mappable**: Test activities that have a semantically equivalent group in the training set (e.g., test "jogging" maps to training group "running")
- **Novel**: Test activities with no semantic equivalent in training — these are genuinely unseen and all models will struggle
- **Closed-set mask**: How many of the 87 training labels are "allowed" for closed-set prediction. Lower = easier discrimination
- For **classifier-based** models (LiMU-BERT, MOMENT, CrossHAR): novel activities are guaranteed wrong since no training label maps to them
- For **text-aligned** models (TSFM, LanHAR): novel activities can still be attempted via text similarity, though performance will be weak

**Difficulty ranking**: MotionSense (easiest) > RealWorld > MobiAct > VTT-ConIoT (hardest, 50% novel activities from construction domain)

## Per-Baseline Supervised Fine-Tuning

Each baseline fine-tunes its encoder end-to-end with its native classification mechanism:

| Baseline | Supervised Method | Epochs | Architecture |
|----------|------------------|--------|-------------|
| **TSFM** | Cosine sim fine-tune | 20 | Encoder + cosine sim with frozen text embeddings |
| **LiMU-BERT** | Encoder + GRU fine-tune | 20 | Encoder + 2-layer GRU(72->20->10) + Linear(10, num_classes) |
| **MOMENT** | Encoder + linear fine-tune | 20 | Encoder + Linear(6144, num_classes) |
| **CrossHAR** | Encoder + Transformer_ft fine-tune | 20 | Encoder + Linear(72->100) + TransformerEncoder(1L,4H) + Linear(100, num_classes) |
| **LanHAR** | Cosine sim fine-tune | 20 | Sensor encoder + cosine sim with frozen text prototypes |

### Fine-Tuning Hyperparameters (shared)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 20 | Small datasets, avoid overfitting |
| Encoder LR | 1e-5 | Conservative for pretrained models |
| Head LR | 1e-3 | Standard for randomly initialized heads |
| Batch size | 32 | Fits in GPU memory for all models |
| Early stopping | patience=5 | Monitor val accuracy |
| Weight decay | 1e-5 | Light regularization |
| Optimizer | AdamW | Standard |

### Why MOMENT uses linear head instead of SVM for supervised fine-tuning

SVM is not differentiable and cannot be trained end-to-end with the encoder. The MOMENT paper's
supervised fine-tuning protocol uses a linear head, which we adopt here. SVM-RBF remains the
zero-shot classifier (trained on frozen training embeddings).

### Zero-Shot Classifiers (non-text-aligned baselines)

Each non-text-aligned model uses its paper's native classifier architecture for zero-shot evaluation, trained on embeddings from all 10 training datasets with 87 global classes:

| Baseline | ZS Classifier | Input Format | Training | Selection |
|----------|--------------|--------------|----------|-----------|
| **LiMU-BERT** | GRU | (M, 20, 72) sub-windows | 90/10 split, 100 epochs | Best val accuracy |
| **MOMENT** | SVM-RBF | (N, 6144) flat | GridSearchCV, 5-fold CV | Best CV score |
| **CrossHAR** | Transformer_ft | (N, 120, 72) sequences | 90/10 split, 100 epochs | Best val loss |

- Same classifier used for both open-set (all 87 logits/scores) and closed-set (masked logits/scores)
- LiMU-BERT: Sub-windows created via `reshape_and_merge` (120-step windows split into 6 x 20-step sub-windows, filtered to uniform-label windows)
- MOMENT: SVM trained with GridSearchCV over C values, subsampled to 10K if needed
- CrossHAR: Transformer_ft architecture matches the paper's downstream classifier (Linear(72->100) + TransformerEncoder + Linear(100, 87))

## Data Split Protocol

For each test dataset:

1. **Random window-level split**: 80% train / 10% val / 10% test (seed=3431)
2. **Balanced subsampling**: For 1%/10% supervised, `balanced_subsample()` draws proportionally from each class, with `max(1, ...)` to ensure every class has at least 1 sample
3. **Consistent across baselines**: Same seed, same splits, same subsampling — every model sees identical train/val/test windows
4. **Global seeds**: All evaluators set `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)` at startup for full reproducibility
5. **TSFM patch size**: Fixed at 1.0s for all datasets (metadata-only decision, no test-time tuning)
6. **End-to-end fine-tuning**: Supervised metrics (1%, 10%) fine-tune the encoder jointly with the classifier, using differential learning rates (encoder: 1e-5, head: 1e-3) and early stopping

## Reported Metrics

For each dataset x metric combination:
- **Accuracy** (%)
- **F1 Macro** (%) — with `zero_division=0` for absent classes
- **F1 Weighted** (%) — weighted by class support, matches LanHAR paper's metric
- **N samples** — test set size
- **N train samples** — training samples used (varies by metric)

## Output Format

Each evaluation script writes a single JSON file per model to `test_output/baseline_evaluation/`:

```
{model_name}_evaluation.json
```

For example: `tsfm_evaluation.json`, `limubert_evaluation.json`, `moment_evaluation.json`, `crosshar_evaluation.json`, `lanhar_evaluation.json`. Each file contains results for all test datasets.

The `scripts/generate_results_table.py` script reads all JSON outputs and produces a combined comparison table.

## Running Evaluations

```bash
# Individual baselines
python val_scripts/human_activity_recognition/evaluate_moment.py
python val_scripts/human_activity_recognition/evaluate_limubert.py
python val_scripts/human_activity_recognition/evaluate_crosshar.py
python val_scripts/human_activity_recognition/evaluate_lanhar.py
python val_scripts/human_activity_recognition/evaluate_tsfm.py

# All at once (sequential)
bash scripts/run_all_evaluations.sh

# Generate combined table
python scripts/generate_results_table.py
```
