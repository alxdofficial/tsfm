# Baseline Evaluation Results

Generated: 2026-02-17 | Framework: 5-metric unified evaluation | Seed: 3431

## Models Evaluated (Our Benchmark)

| Model | Type | Pretrain Data | Embed Dim | Zero-Shot? | Classifier |
|-------|------|---------------|-----------|------------|------------|
| **TSFM (ours)** | Text-aligned foundation model | 10 HAR datasets (epoch 73/100) | 384 | Yes | Linear |
| **LiMU-BERT** | Self-supervised (masked reconstruction) | 10 HAR datasets (paper checkpoint) | 72 | No | GRU |
| **MOMENT** | General time-series foundation model | Time Series Pile (no HAR data) | 6144 | No | SVM-RBF |
| **CrossHAR** | Self-supervised (contrastive) | 10 HAR datasets (paper checkpoint) | 72 | No | Transformer_ft |
| **LanHAR** | Text-aligned (trained from scratch) | 10 HAR datasets (fresh each run) | 768 | Yes | Linear |
| **NLS-HAR** * | Text-aligned (NLS) | Capture-24 (single dataset) | — | Yes | — |

## Fairness Notes

**Training data**: All HAR-pretrained models (TSFM, LiMU-BERT, CrossHAR, LanHAR) use 10 training
datasets. The 4 test datasets (MotionSense, RealWorld, MobiAct, VTT-ConIoT) were never seen during
any model's pretraining. MOMENT was pretrained on general time-series data with no HAR-specific data.

**Classifiers**: Each baseline uses its own paper's recommended downstream classifier for supervised
metrics (1%, 10%). The linear probe metric uses the same `nn.Linear` architecture across all models.

**TSFM training status**: Checkpoint at epoch 73 of 100 (training interrupted). Results may improve
with completed training. All other baselines use fully-trained checkpoints.

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
| **TSFM (ours)** | 21.0 | 5.7 | 29.4 | 16.3 | 45.3 | 36.4 | 52.9 | 46.2 | 63.2 | 54.4 |
| **LiMU-BERT** | N/A | N/A | N/A | N/A | 38.2 | 30.9 | 57.3 | 50.4 | 55.2 | 43.3 |
| **MOMENT** | N/A | N/A | N/A | N/A | 58.4 | 54.7 | 74.5 | 71.8 | 82.7 | 81.3 |
| **CrossHAR** | N/A | N/A | N/A | N/A | 51.6 | 46.7 | 66.5 | 60.9 | 65.8 | 55.8 |
| **LanHAR** | 12.7 | 6.1 | 22.9 | 16.1 | 30.9 | 27.1 | 41.7 | 34.7 | 47.6 | 34.0 |
| **NLS-HAR** * | — | 28.0 | — | — | — | — | — | — | — | — |

*\* NLS-HAR numbers from published paper, not re-evaluated on our pipeline. See [caveats below](#nls-har-caveats).*

---

## Per-Dataset Results

### Zero-Shot Open-Set

*Model predicts from all training labels; correct if predicted label maps to same group as ground truth.*

| Model | MobiAct Acc | MobiAct F1 | MotionSense Acc | MotionSense F1 | RealWorld Acc | RealWorld F1 | VTT-ConIoT Acc | VTT-ConIoT F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 30.7 | 7.2 | 26.7 | 7.4 | 22.1 | 7.0 | 4.4 | 1.1 |
| **LanHAR** | 11.4 | 4.4 | 14.0 | 6.4 | 17.3 | 11.4 | 8.3 | 2.1 |
| **NLS-HAR** * | — | 16.9 | — | 39.0 | — | — | — | — |

### Zero-Shot Closed-Set

*Model predicts from only the test dataset's own labels.*

| Model | MobiAct Acc | MobiAct F1 | MotionSense Acc | MotionSense F1 | RealWorld Acc | RealWorld F1 | VTT-ConIoT Acc | VTT-ConIoT F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 47.1 | 14.4 | 41.8 | 33.6 | 21.6 | 13.0 | 6.9 | 4.1 |
| **LanHAR** | 17.5 | 11.6 | 37.1 | 30.7 | 30.0 | 19.1 | 6.9 | 3.2 |

### 1% Supervised

| Model | MobiAct Acc | MobiAct F1 | MotionSense Acc | MotionSense F1 | RealWorld Acc | RealWorld F1 | VTT-ConIoT Acc | VTT-ConIoT F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 43.9 | 22.5 | 65.3 | 60.1 | 57.8 | 53.7 | 14.0 | 9.3 |
| **LiMU-BERT** | 32.1 | 17.0 | 54.3 | 51.9 | 51.4 | 45.4 | 14.8 | 9.4 |
| **MOMENT** | 47.8 | 34.6 | 88.2 | 87.8 | 76.9 | 77.5 | 20.8 | 18.8 |
| **CrossHAR** | 42.8 | 32.4 | 80.4 | 78.8 | 68.8 | 63.3 | 14.5 | 12.4 |
| **LanHAR** | 25.7 | 17.7 | 44.7 | 42.0 | 43.5 | 42.0 | 9.7 | 6.9 |

### 10% Supervised

| Model | MobiAct Acc | MobiAct F1 | MotionSense Acc | MotionSense F1 | RealWorld Acc | RealWorld F1 | VTT-ConIoT Acc | VTT-ConIoT F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 49.9 | 32.9 | 72.3 | 68.8 | 65.5 | 63.1 | 24.2 | 19.9 |
| **LiMU-BERT** | 56.6 | 37.8 | 84.1 | 85.0 | 66.8 | 63.7 | 21.8 | 15.0 |
| **MOMENT** | 77.5 | 67.3 | 94.7 | 94.2 | 84.1 | 84.8 | 41.5 | 41.0 |
| **CrossHAR** | 65.3 | 48.6 | 88.6 | 87.5 | 80.3 | 78.8 | 31.9 | 28.6 |
| **LanHAR** | 48.3 | 29.5 | 49.3 | 46.6 | 52.7 | 52.3 | 16.4 | 10.3 |

### Linear Probe

| Model | MobiAct Acc | MobiAct F1 | MotionSense Acc | MotionSense F1 | RealWorld Acc | RealWorld F1 | VTT-ConIoT Acc | VTT-ConIoT F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 66.2 | 37.0 | 84.5 | 81.8 | 77.0 | 75.0 | 25.1 | 23.6 |
| **LiMU-BERT** | 62.3 | 30.5 | 72.8 | 70.5 | 63.1 | 50.7 | 22.7 | 21.6 |
| **MOMENT** | 86.9 | 81.2 | 95.8 | 95.3 | 86.7 | 87.7 | 61.4 | 61.1 |
| **CrossHAR** | 68.0 | 39.3 | 85.9 | 83.1 | 73.4 | 66.3 | 35.7 | 34.8 |
| **LanHAR** | 61.6 | 21.9 | 58.8 | 48.0 | 57.2 | 55.9 | 13.0 | 10.2 |

---

## NLS-HAR Caveats

NLS-HAR zero-shot numbers are from the published paper
([Haresamudram et al., 2024](https://arxiv.org/abs/2408.12023)), **not re-evaluated on our
pipeline**. They are included in the zero-shot table for reference since the zero-shot setting
(pretrain on separate data, predict on target with no target training) is conceptually comparable.

### Protocol Differences

| Aspect | Our Benchmark | NLS-HAR Paper |
|--------|---------------|---------------|
| **Sensor channels** | 6 (acc + gyro) | 3 (acc only) |
| **Window size** | 6.0s (120 @ 20Hz) | 2.0s (100 @ 50Hz) |
| **Sampling rate** | 20 Hz | 50 Hz |
| **MobiAct classes** | 13 | 11 |
| **Pretrain data** | 10 HAR datasets | Capture-24 (single dataset) |
| **Metric reported** | Accuracy + F1 macro | F1 macro only |
| **Eval protocol** | Window-level random split | User-level 5-fold CV |

**Note on splits**: For zero-shot (no target training data), the split protocol does not affect the
result — both approaches evaluate on the full target dataset with no target-specific training. The
split difference only matters for supervised metrics, which is why NLS-HAR is only included in the
zero-shot table.

### NLS-HAR Additional Published Results (for reference)

| Method | MobiAct F1 | MotionSense F1 |
|--------|-----------|----------------|
| NLS Zero-Shot (Capture-24 pretrain) | 16.9 | 39.0 |
| NLS + target train pretrain | 59.1 | 73.4 |
| NLS + adaptation + improved text | 65.3 | — |
| Conv. Classifier (fully supervised) | 79.0 | 89.0 |

---

## Key Observations

1. **MOMENT dominates supervised/LP metrics** — its 6144-dim general time-series embeddings are
   extremely powerful, despite having zero HAR-specific pretraining.

2. **TSFM is the only model with both zero-shot AND competitive supervised performance**:
   - Beats LanHAR on zero-shot open-set by ~2x (21.0% vs 12.7% acc)
   - Beats LiMU-BERT on 1% supervised and linear probe
   - Note: training incomplete (73/100 epochs)

3. **Zero-shot is hard** — even the best (TSFM closed-set 29.4% avg) is far from supervised
   baselines, indicating significant room for improvement in text-sensor alignment.

4. **VTT-ConIoT is the hardest dataset** — 16 industrial activities, all models score lowest here.

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
- TSFM checkpoint: `training_output/semantic_alignment/20260216_225955/best.pt` (epoch 73)
