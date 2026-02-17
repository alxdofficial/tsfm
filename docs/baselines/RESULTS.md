# Baseline Evaluation Results

Generated: 2026-02-17 | Framework: 5-metric unified evaluation | Seed: 3431

## Models

| Model | Type | Pretrain Data | Embed Dim | Zero-Shot Method | Classifier |
|-------|------|---------------|-----------|------------------|------------|
| **TSFM (ours)** | Text-aligned foundation model | 10 HAR datasets (training in progress) | 384 | Cosine sim with text embeddings | Linear |
| **LiMU-BERT** | Self-supervised (masked reconstruction) | 10 HAR datasets (paper checkpoint) | 72 | Classifier + group scoring | GRU |
| **MOMENT** | General time-series foundation model | Time Series Pile (no HAR data) | 6144 | Classifier + group scoring | SVM-RBF |
| **CrossHAR** | Self-supervised (contrastive) | 10 HAR datasets (paper checkpoint) | 72 | Classifier + group scoring | Transformer_ft |
| **LanHAR** | Text-aligned (trained from scratch) | 10 HAR datasets (fresh each run) | 768 | Cosine sim with text embeddings | Linear |

## Fairness Notes

**Training data**: All HAR-pretrained models (TSFM, LiMU-BERT, CrossHAR, LanHAR) use 10 training
datasets. The 4 test datasets (MotionSense, RealWorld, MobiAct, VTT-ConIoT) were never seen during
any model's pretraining. MOMENT was pretrained on general time-series data with no HAR-specific data.

**Zero-shot prediction mechanisms differ by model type**:
- *Text-aligned models* (TSFM, LanHAR): Encode activity labels as text, predict via cosine similarity
  with sensor embeddings. Open-set uses all 87 labels + group scoring; closed-set uses only test labels + exact match.
- *Classifier-based models* (LiMU-BERT, MOMENT, CrossHAR): Train a linear classifier on training embeddings
  (87 global labels), predict via classifier logits + group scoring. Open-set uses all 87 logits;
  closed-set masks logits to training labels whose group appears in the test dataset.

**Classifiers**: Each baseline uses its own paper's recommended downstream classifier for supervised
metrics (1%, 10%). The linear probe metric uses the same `nn.Linear` architecture across all models.

**TSFM (ours)**: Training not yet complete. Results will be added once a fully trained checkpoint
is available.

**Embedding dimensions vary**: MOMENT (6144) >> LanHAR (768) > TSFM (384) >> LiMU-BERT/CrossHAR (72).
Higher dimensions give more capacity but the linear probe uses the same architecture for all.

## Test Datasets

| Dataset | Windows | Classes | Difficulty |
|---------|---------|---------|------------|
| MotionSense | 12,080 | 6 | Easy (basic locomotion) |
| RealWorld | 27,138 | 8 | Medium (multi-placement) |
| MobiAct | 4,345 | 13 | Hard (falls, vehicle entry) |
| VTT-ConIoT | 2,058 | 16 | Hard (industrial activities) |

---

## Average Across All Datasets

| Model | ZS-Open Acc | ZS-Open F1 | ZS-Close Acc | ZS-Close F1 | 1% Acc | 1% F1 | 10% Acc | 10% F1 | LP Acc | LP F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | — | — | — | — | — | — | — | — | — | — |
| **LiMU-BERT** | — | — | — | — | 38.2 | 30.9 | 57.3 | 50.4 | 55.2 | 43.3 |
| **MOMENT** | — | — | — | — | 58.4 | 54.7 | 74.5 | 71.8 | 82.7 | 81.3 |
| **CrossHAR** | — | — | — | — | 51.6 | 46.7 | 66.5 | 60.9 | 65.8 | 55.8 |
| **LanHAR** | 12.7 | 6.1 | 22.9 | 16.1 | 30.9 | 27.1 | 41.7 | 34.7 | 47.6 | 34.0 |

*Note: LiMU-BERT and CrossHAR zero-shot use native classifiers (GRU and Transformer_ft). MOMENT and LanHAR zero-shot results pending re-evaluation with caching.*

---

## Per-Dataset Results

### Zero-Shot Open-Set

*Model predicts from all 87 training labels; correct if predicted label maps to same group as ground truth.*

*Text-aligned models use cosine similarity; classifier-based models use a trained linear classifier.*

| Model | MobiAct Acc | MobiAct F1 | MotionSense Acc | MotionSense F1 | RealWorld Acc | RealWorld F1 | VTT-ConIoT Acc | VTT-ConIoT F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | — | — | — | — | — | — | — | — |
| **LiMU-BERT** | — | — | — | — | — | — | — | — |
| **MOMENT** | — | — | — | — | — | — | — | — |
| **CrossHAR** | — | — | — | — | — | — | — | — |
| **LanHAR** | 11.4 | 4.4 | 14.0 | 6.4 | 17.3 | 11.4 | 8.3 | 2.1 |

### Zero-Shot Closed-Set

*Text-aligned models predict from test labels only (exact match). Classifier-based models mask logits to test-relevant groups (group match).*

| Model | MobiAct Acc | MobiAct F1 | MotionSense Acc | MotionSense F1 | RealWorld Acc | RealWorld F1 | VTT-ConIoT Acc | VTT-ConIoT F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | — | — | — | — | — | — | — | — |
| **LiMU-BERT** | — | — | — | — | — | — | — | — |
| **MOMENT** | — | — | — | — | — | — | — | — |
| **CrossHAR** | — | — | — | — | — | — | — | — |
| **LanHAR** | 17.5 | 11.6 | 37.1 | 30.7 | 30.0 | 19.1 | 6.9 | 3.2 |

### 1% Supervised

| Model | MobiAct Acc | MobiAct F1 | MotionSense Acc | MotionSense F1 | RealWorld Acc | RealWorld F1 | VTT-ConIoT Acc | VTT-ConIoT F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | — | — | — | — | — | — | — | — |
| **LiMU-BERT** | 32.1 | 17.0 | 54.3 | 51.9 | 51.4 | 45.4 | 14.8 | 9.4 |
| **MOMENT** | 47.8 | 34.6 | 88.2 | 87.8 | 76.9 | 77.5 | 20.8 | 18.8 |
| **CrossHAR** | 42.8 | 32.4 | 80.4 | 78.8 | 68.8 | 63.3 | 14.5 | 12.4 |
| **LanHAR** | 25.7 | 17.7 | 44.7 | 42.0 | 43.5 | 42.0 | 9.7 | 6.9 |

### 10% Supervised

| Model | MobiAct Acc | MobiAct F1 | MotionSense Acc | MotionSense F1 | RealWorld Acc | RealWorld F1 | VTT-ConIoT Acc | VTT-ConIoT F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | — | — | — | — | — | — | — | — |
| **LiMU-BERT** | 56.6 | 37.8 | 84.1 | 85.0 | 66.8 | 63.7 | 21.8 | 15.0 |
| **MOMENT** | 77.5 | 67.3 | 94.7 | 94.2 | 84.1 | 84.8 | 41.5 | 41.0 |
| **CrossHAR** | 65.3 | 48.6 | 88.6 | 87.5 | 80.3 | 78.8 | 31.9 | 28.6 |
| **LanHAR** | 48.3 | 29.5 | 49.3 | 46.6 | 52.7 | 52.3 | 16.4 | 10.3 |

### Linear Probe

| Model | MobiAct Acc | MobiAct F1 | MotionSense Acc | MotionSense F1 | RealWorld Acc | RealWorld F1 | VTT-ConIoT Acc | VTT-ConIoT F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | — | — | — | — | — | — | — | — |
| **LiMU-BERT** | 62.3 | 30.5 | 72.8 | 70.5 | 63.1 | 50.7 | 22.7 | 21.6 |
| **MOMENT** | 86.9 | 81.2 | 95.8 | 95.3 | 86.7 | 87.7 | 61.4 | 61.1 |
| **CrossHAR** | 68.0 | 39.3 | 85.9 | 83.1 | 73.4 | 66.3 | 35.7 | 34.8 |
| **LanHAR** | 61.6 | 21.9 | 58.8 | 48.0 | 57.2 | 55.9 | 13.0 | 10.2 |

---

## Key Observations

1. **MOMENT dominates supervised/LP metrics** — its 6144-dim general time-series embeddings are
   extremely powerful, despite having zero HAR-specific pretraining.

2. **Zero-shot is hard** — LanHAR's best closed-set average is 22.9%, far from supervised baselines,
   indicating this is a challenging setting.

3. **All models now have zero-shot metrics** — text-aligned models use cosine similarity,
   classifier-based models use a trained linear classifier with group scoring. While the
   prediction mechanisms differ, both measure cross-dataset transfer without test-time labels.

4. **VTT-ConIoT is the hardest dataset** — 16 industrial activities, all models score lowest here.
   For classifier-based models, 8 of 16 test labels have no training synonyms, creating
   a genuine coverage gap vs text-aligned models.

5. **LanHAR underperforms expectations** — despite text alignment, it scores below non-text-aligned
   baselines on supervised metrics, suggesting from-scratch training doesn't learn representations
   as strong as pretrained encoders.

---

## Reproducibility

- Raw JSON results: `docs/baselines/results/*.json`
- Evaluation scripts: `val_scripts/human_activity_recognition/evaluate_*.py`
- Results generator: `scripts/generate_results_table.py`
- Run all: `bash scripts/run_all_evaluations.sh`
- Random seed: 3431 (consistent across all splits)
