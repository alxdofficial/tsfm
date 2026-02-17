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

- **What**: Classify against ALL 87 training labels using cosine similarity
- **How**: `argmax(sensor_emb @ all_87_label_embs.T)` then group-match to test label
- **Why**: Tests open-vocabulary generalization — can the model find the right activity among all possible labels?
- **Applies to**: TSFM, LanHAR (text-aligned models only)
- **N/A for**: LiMU-BERT, MOMENT, CrossHAR (not text-aligned)

### Metric 2: Zero-Shot Closed-Set

- **What**: Classify against only the test dataset's own labels
- **How**: `argmax(sensor_emb @ test_label_embs.T)` with exact label matching
- **Why**: Tests discriminative quality when the label space is known
- **Applies to**: TSFM, LanHAR
- **N/A for**: LiMU-BERT, MOMENT, CrossHAR

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

## Data Split Protocol

For each test dataset:

1. **Random window-level split**: 80% train / 10% val / 10% test (seed=3431)
2. **Balanced subsampling**: For 1%/10% supervised, `balanced_subsample()` draws proportionally from each class, with `max(1, ...)` to ensure every class has at least 1 sample
3. **Consistent across baselines**: Same seed, same splits, same subsampling

## Reported Metrics

For each dataset x metric combination:
- **Accuracy** (%)
- **F1 Macro** (%) — with `zero_division=0` for absent classes
- **F1 Weighted** (%) — matches LanHAR paper's metric
- **N samples** — test set size
- **N train samples** — training samples used (varies by metric)

## Output Format

Each evaluation script writes a JSON file to `test_output/baseline_evaluation/`:

```
{dataset_name}_{model_name}_evaluation.json
```

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
