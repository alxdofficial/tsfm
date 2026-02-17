# Baseline Evaluation Results

Generated: 2026-02-17 | Framework: 4-metric unified evaluation | Seed: 3431

## Models

| Model | Type | Pretrain Data | Embed Dim | Zero-Shot Method | Classifier |
|-------|------|---------------|-----------|------------------|------------|
| **TSFM (ours)** | Text-aligned foundation model | 10 HAR datasets (training in progress) | 384 | Cosine sim with text embeddings | End-to-end cosine sim |
| **LiMU-BERT** | Self-supervised (masked reconstruction) | 10 HAR datasets (paper checkpoint) | 72 | Classifier + group scoring | End-to-end encoder + GRU |
| **MOMENT** | General time-series foundation model | Time Series Pile (no HAR data) | 6144 | Classifier + group scoring | End-to-end encoder + linear |
| **CrossHAR** | Self-supervised (contrastive) | 10 HAR datasets (paper checkpoint) | 72 | Classifier + group scoring | End-to-end encoder + Transformer_ft |
| **LanHAR** | Text-aligned (trained from scratch) | 10 HAR datasets (fresh each run) | 768 | Cosine sim with text embeddings | End-to-end cosine sim |

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

**Supervised fine-tuning**: Each baseline fine-tunes its encoder end-to-end with its native
classification mechanism. Text-aligned models (TSFM, LanHAR) classify via cosine similarity
with frozen text embeddings. Non-text-aligned models use their paper's native classifier head.

**TSFM (ours)**: Training not yet complete. Results will be added once a fully trained checkpoint
is available.

**Embedding dimensions vary**: MOMENT (6144) >> LanHAR (768) > TSFM (384) >> LiMU-BERT/CrossHAR (72).
Higher dimensions give more capacity for downstream tasks.

## Test Datasets

| Dataset | Windows | Classes | Difficulty |
|---------|---------|---------|------------|
| MotionSense | 12,080 | 6 | Easy (basic locomotion) |
| RealWorld | 27,138 | 8 | Medium (multi-placement) |
| MobiAct | 4,345 | 13 | Hard (falls, vehicle entry) |
| VTT-ConIoT | 2,058 | 16 | Hard (industrial activities) |

---

## Average Across All Datasets

| Model | ZS-Open Acc | ZS-Open F1 | ZS-Close Acc | ZS-Close F1 | 1% Acc | 1% F1 | 10% Acc | 10% F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | — | — | — | — | — | — | — | — |
| **LiMU-BERT** | — | — | — | — | — | — | — | — |
| **MOMENT** | — | — | — | — | — | — | — | — |
| **CrossHAR** | — | — | — | — | — | — | — | — |
| **LanHAR** | 12.7 | 6.1 | 22.9 | 16.1 | — | — | — | — |

*Note: Supervised metrics now use end-to-end fine-tuning. Previous frozen-embedding results are no longer comparable. Re-evaluation pending.*

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

### 1% Supervised (End-to-End Fine-Tuning)

*Encoder fine-tuned end-to-end with 1% of labeled data. Text-aligned models classify via cosine similarity; others use native classifier heads.*

| Model | MobiAct Acc | MobiAct F1 | MotionSense Acc | MotionSense F1 | RealWorld Acc | RealWorld F1 | VTT-ConIoT Acc | VTT-ConIoT F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | — | — | — | — | — | — | — | — |
| **LiMU-BERT** | — | — | — | — | — | — | — | — |
| **MOMENT** | — | — | — | — | — | — | — | — |
| **CrossHAR** | — | — | — | — | — | — | — | — |
| **LanHAR** | — | — | — | — | — | — | — | — |

### 10% Supervised (End-to-End Fine-Tuning)

*Encoder fine-tuned end-to-end with 10% of labeled data. Text-aligned models classify via cosine similarity; others use native classifier heads.*

| Model | MobiAct Acc | MobiAct F1 | MotionSense Acc | MotionSense F1 | RealWorld Acc | RealWorld F1 | VTT-ConIoT Acc | VTT-ConIoT F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | — | — | — | — | — | — | — | — |
| **LiMU-BERT** | — | — | — | — | — | — | — | — |
| **MOMENT** | — | — | — | — | — | — | — | — |
| **CrossHAR** | — | — | — | — | — | — | — | — |
| **LanHAR** | — | — | — | — | — | — | — | — |

---

## Key Observations

1. **Zero-shot is hard** — LanHAR's best closed-set average is 22.9%, far from supervised baselines,
   indicating this is a challenging setting.

2. **All models now have zero-shot metrics** — text-aligned models use cosine similarity,
   classifier-based models use a trained classifier with group scoring. While the
   prediction mechanisms differ, both measure cross-dataset transfer without test-time labels.

3. **VTT-ConIoT is the hardest dataset** — 16 industrial activities, all models score lowest here.
   For classifier-based models, 8 of 16 test labels have no training synonyms, creating
   a genuine coverage gap vs text-aligned models.

4. **Supervised metrics use end-to-end fine-tuning** — each model's encoder is fine-tuned with
   its native classification mechanism (cosine sim for text-aligned, classifier heads for others).
   This better reflects each model's real-world transfer learning capability compared to
   frozen-embedding evaluation.

---

## Reproducibility

- Raw JSON results: `docs/baselines/results/*.json`
- Evaluation scripts: `val_scripts/human_activity_recognition/evaluate_*.py`
- Results generator: `scripts/generate_results_table.py`
- Run all: `bash scripts/run_all_evaluations.sh`
- Random seed: 3431 (consistent across all splits)
