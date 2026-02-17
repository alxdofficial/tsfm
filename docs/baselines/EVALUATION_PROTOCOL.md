# Evaluation Protocol

Unified 5-metric evaluation framework for comparing TSFM against baselines on 4 zero-shot test datasets.

## Test Datasets

All models are evaluated on 4 datasets that were **never seen during training**:

| Dataset | Windows | Activities | Difficulty |
|---------|---------|------------|------------|
| MotionSense | 12,080 | 6 | Easy (basic locomotion) |
| RealWorld | 27,138 | 8 | Medium (multi-placement) |
| MobiAct | 4,345 | 13 | Hard (includes falls, novel activities) |
| VTT-ConIoT | 2,058 | 16 | Hard (industrial IoT context) |

All data is standardized to `(N, 120, 6)` windows at 20Hz with 6 IMU channels (acc_xyz + gyro_xyz).

## 5-Metric Framework

### Metric 1: Zero-Shot Open-Set

- **What**: Classify against ALL 87 training labels, score via synonym groups
- **Why**: Tests open-vocabulary generalization — can the model find the right activity among all possible labels?
- **Applies to**: All 5 models
- **Text-aligned (TSFM, LanHAR)**: `argmax(sensor_emb @ all_87_label_embs.T)` via cosine similarity with text embeddings, then group-match prediction to ground truth
- **Classifier-based (LiMU-BERT, MOMENT, CrossHAR)**: Train each model's native classifier on all 10 training datasets (87 global classes), predict over all 87, then group-match prediction to ground truth

### Metric 2: Zero-Shot Closed-Set

- **What**: Classify against only labels relevant to the test dataset
- **Why**: Tests discriminative quality when the label space is constrained
- **Applies to**: All 5 models
- **Text-aligned (TSFM, LanHAR)**: Encode only the test dataset's activity labels as text, `argmax(sensor_emb @ test_label_embs.T)`, exact label matching
- **Classifier-based (LiMU-BERT, MOMENT, CrossHAR)**: Mask each model's native classifier logits/scores to training labels whose synonym group appears in the test dataset, `argmax(masked_logits)`, group-match scoring

**Closed-set mask details**: For each test dataset, a mask over the 87 training labels allows only those whose group is represented in the test set:

| Dataset | Test Groups | Allowed/87 | Notes |
|---------|-------------|------------|-------|
| MotionSense | 6 | 22 | All test labels mappable |
| RealWorld | 8 | 30 | All test labels mappable |
| MobiAct | 9 | 34 | vehicle_entry group has 0 training members |
| VTT-ConIoT | 11 | 20 | 4 groups (carrying, climbing, kneeling, painting) have 0 training members |

**Fairness note**: Text-aligned and classifier-based models use different prediction mechanisms for zero-shot metrics, so direct comparison should be interpreted carefully. Text-aligned models have the advantage of encoding arbitrary label text; classifier-based models are limited to predicting among training labels and use group scoring to bridge the gap.

### Metric 3: 1% Supervised

- **What**: Train a downstream classifier using 1% of labeled data
- **How**: Split 80/10/10 (train/val/test), subsample 1% of train via balanced_subsample, train classifier, evaluate on test split
- **Why**: Tests few-shot transfer quality of frozen embeddings
- **Applies to**: All 5 models

### Metric 4: 10% Supervised

- **What**: Same as 1% but with 10% labeled data
- **Why**: Tests semi-supervised regime
- **Applies to**: All 5 models

### Metric 5: Linear Probe

- **What**: Train a linear classifier on full training split of frozen embeddings
- **How**: Split 80/10/10, train linear layer (100 epochs), evaluate on test split
- **Why**: Standard representation quality benchmark
- **Applies to**: All 5 models

## Per-Baseline Classifiers

Each baseline uses its original paper's downstream classifier to ensure fairness:

| Baseline | Supervised Classifier | Epochs | Architecture |
|----------|----------------------|--------|-------------|
| **TSFM** | Linear | 100 | Linear(384, num_classes) |
| **LiMU-BERT** | GRU | 100 | 2-layer GRU(72->20->10) + Linear(10, num_classes) |
| **MOMENT** | SVM-RBF | GridSearchCV | C in [1e-4..1e4], 5-fold CV, gamma=scale, 6144-dim input |
| **CrossHAR** | Transformer_ft | 100 | Linear(72->100) + TransformerEncoder(1L,4H) + Linear(100, num_classes) |
| **LanHAR** | Linear | 100 | Linear(768, num_classes) |

### Linear Probe Classifier (all baselines)

All baselines use the same linear probe architecture for Metric 5:
- `Linear(emb_dim, num_classes)` with dropout 0.3 during training
- 100 epochs, Adam, lr=1e-3, batch_size=512
- Best model selected by validation accuracy

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
3. **Consistent across baselines**: Same seed, same splits, same subsampling
4. **Global seeds**: All evaluators set `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)` at startup for full reproducibility
5. **TSFM patch-size sweep**: Uses a separate 20% held-out split (seed=42) per test dataset for patch selection; metrics are reported on the full dataset

## Reported Metrics

For each dataset x metric combination:
- **Accuracy** (%)
- **F1 Macro** (%) — with `zero_division=0` for absent classes
- **F1 Weighted** (%) — matches LanHAR paper's metric
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
