# Baseline Evaluation Results

Generated: 2026-02-17 | Framework: 4-metric unified evaluation | Seed: 3431

## Models

| Model | Type | Pretrain Data | Embed Dim | Zero-Shot Method | Classifier |
|-------|------|---------------|-----------|------------------|------------|
| **TSFM (ours)** | Text-aligned foundation model | 10 HAR datasets | 384 | Cosine sim with text embeddings | End-to-end cosine sim |
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
- *Classifier-based models* (LiMU-BERT, MOMENT, CrossHAR): Train a native classifier on training embeddings
  (87 global labels), predict via classifier logits + group scoring. Open-set uses all 87 logits;
  closed-set masks logits to training labels whose group appears in the test dataset.

**Supervised fine-tuning**: Each baseline fine-tunes its encoder end-to-end with its native
classification mechanism. Text-aligned models (TSFM, LanHAR) classify via cosine similarity
with frozen text embeddings. Non-text-aligned models use their paper's native classifier head.

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
| **TSFM (ours)** | 27.5 | 7.4 | 35.0 | 22.3 | 54.2 | 46.7 | 68.8 | 63.8 |
| **LiMU-BERT** | 16.7 | 5.2 | 27.0 | 18.4 | 24.1 | 15.5 | 51.8 | 43.2 |
| **MOMENT** | 19.7 | 5.4 | 32.2 | 21.9 | 58.9 | 53.3 | 70.6 | 66.4 |
| **CrossHAR** | 13.0 | 4.2 | 27.8 | 22.4 | 51.3 | 46.4 | 67.8 | 62.3 |
| **LanHAR** | 12.7 | 6.1 | 22.9 | 16.1 | 32.6 | 24.3 | 45.3 | 41.7 |

---

## Per-Dataset Results

### Zero-Shot Open-Set

*Model predicts from all 87 training labels; correct if predicted label maps to same group as ground truth.*

*Text-aligned models use cosine similarity; classifier-based models use a trained native classifier.*

| Model | MobiAct Acc | MobiAct F1 | MotionSense Acc | MotionSense F1 | RealWorld Acc | RealWorld F1 | VTT-ConIoT Acc | VTT-ConIoT F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 31.8 | 8.8 | 38.7 | 10.5 | 38.1 | 9.7 | 1.7 | 0.6 |
| **LiMU-BERT** | 7.0 | 2.1 | 28.0 | 10.1 | 28.9 | 7.6 | 3.1 | 0.8 |
| **MOMENT** | 28.7 | 7.0 | 33.8 | 8.0 | 14.6 | 6.0 | 1.6 | 0.4 |
| **CrossHAR** | 13.5 | 4.2 | 16.2 | 5.4 | 21.5 | 7.0 | 0.7 | 0.4 |
| **LanHAR** | 11.4 | 4.4 | 14.0 | 6.4 | 17.3 | 11.4 | 8.3 | 2.1 |

### Zero-Shot Closed-Set

*Text-aligned models predict from test labels only (exact match). Classifier-based models mask logits to test-relevant groups (group match).*

| Model | MobiAct Acc | MobiAct F1 | MotionSense Acc | MotionSense F1 | RealWorld Acc | RealWorld F1 | VTT-ConIoT Acc | VTT-ConIoT F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 48.1 | 16.5 | 51.5 | 45.3 | 37.0 | 24.8 | 3.4 | 2.6 |
| **LiMU-BERT** | 30.2 | 13.5 | 39.6 | 37.7 | 30.3 | 19.8 | 7.7 | 2.7 |
| **MOMENT** | 40.9 | 24.6 | 51.6 | 39.3 | 31.1 | 21.6 | 5.2 | 2.0 |
| **CrossHAR** | 23.3 | 17.7 | 42.8 | 39.5 | 40.3 | 29.5 | 5.0 | 2.7 |
| **LanHAR** | 17.5 | 11.6 | 37.1 | 30.7 | 30.0 | 19.1 | 6.9 | 3.2 |

### 1% Supervised (End-to-End Fine-Tuning)

*Encoder fine-tuned end-to-end with 1% of labeled data. Text-aligned models classify via cosine similarity; others use native classifier heads.*

| Model | MobiAct Acc | MobiAct F1 | MotionSense Acc | MotionSense F1 | RealWorld Acc | RealWorld F1 | VTT-ConIoT Acc | VTT-ConIoT F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 45.5 | 17.8 | 85.3 | 84.7 | 73.8 | 73.2 | 12.1 | 11.3 |
| **LiMU-BERT** | 6.7 | 2.6 | 38.7 | 29.4 | 45.0 | 27.4 | 5.8 | 2.5 |
| **MOMENT** | 54.9 | 36.8 | 87.4 | 87.5 | 72.1 | 70.2 | 21.3 | 18.6 |
| **CrossHAR** | 42.8 | 30.7 | 78.6 | 77.6 | 66.0 | 60.7 | 17.9 | 16.5 |
| **LanHAR** | 34.5 | 15.2 | 40.3 | 36.6 | 49.2 | 42.6 | 6.3 | 2.6 |

### 10% Supervised (End-to-End Fine-Tuning)

*Encoder fine-tuned end-to-end with 10% of labeled data. Text-aligned models classify via cosine similarity; others use native classifier heads.*

| Model | MobiAct Acc | MobiAct F1 | MotionSense Acc | MotionSense F1 | RealWorld Acc | RealWorld F1 | VTT-ConIoT Acc | VTT-ConIoT F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 72.6 | 53.8 | 93.3 | 92.4 | 83.7 | 83.9 | 25.6 | 25.1 |
| **LiMU-BERT** | 57.5 | 32.7 | 78.9 | 80.1 | 57.3 | 53.2 | 13.5 | 6.7 |
| **MOMENT** | 71.3 | 55.4 | 92.1 | 92.1 | 80.6 | 80.8 | 38.6 | 37.2 |
| **CrossHAR** | 69.7 | 54.7 | 91.6 | 91.0 | 80.5 | 79.2 | 29.5 | 24.3 |
| **LanHAR** | 35.9 | 24.3 | 75.8 | 76.1 | 56.5 | 55.4 | 13.0 | 10.9 |

---

## Key Observations

1. **TSFM leads zero-shot** — 35.0% closed-set avg accuracy, ahead of MOMENT (32.2%), CrossHAR (27.8%),
   LiMU-BERT (27.0%), and LanHAR (22.9%). The text-alignment approach provides a genuine advantage
   for cross-dataset transfer without any test-time labels.

2. **MOMENT leads supervised, TSFM close behind** — At 10%, MOMENT 70.6% vs TSFM 68.8% avg accuracy.
   At 1%, MOMENT leads more clearly (58.9% vs 54.2%), likely benefiting from its larger embedding
   dimension (6144 vs 384) and general time-series pretraining.

3. **CrossHAR is a strong third** — 67.8% at 10% supervised, competitive with TSFM/MOMENT on
   MotionSense (91.6%) and RealWorld (80.5%), despite a much smaller embedding (72-dim).

4. **VTT-ConIoT is the hardest dataset** — All models score lowest here. 8 of 16 test labels
   have no training synonyms (construction domain activities), creating a genuine coverage gap.
   MOMENT performs best here (38.6% at 10%), likely due to its general time-series pretraining.

5. **LiMU-BERT struggles at low data** — Only 24.1% at 1% supervised avg, the lowest among all models.
   Performance recovers at 10% (51.8%), suggesting the GRU classifier needs more data to converge.

6. **LanHAR underperforms across all metrics** — Despite being text-aligned, LanHAR's from-scratch
   SciBERT training on small HAR data limits its zero-shot and supervised transfer quality.

7. **TSFM uses a fixed 1.0s patch size** — no per-dataset sweep or test-time tuning. This is a
   metadata-only decision: at 20Hz, 1.0s patches give the finest temporal resolution while producing
   enough tokens for the encoder. Sensitivity analysis shows results are robust across patch sizes
   (max 9% range on easiest dataset, <2% on hardest).

---

## TSFM Patch Size Sensitivity

*Zero-shot closed-set accuracy (%) at each candidate patch size, evaluated on the full test set. TSFM uses a fixed 1.0s patch size for all reported results — no per-dataset sweep or test labels used for selection. This table demonstrates that the choice is robust.*

| Dataset | 1.0s | 1.25s | 1.5s | 1.75s | 2.0s | Range |
|---------|---:|---:|---:|---:|---:|---:|
| MotionSense | **51.5** | 48.5 | 46.1 | 44.0 | 42.5 | 9.0 |
| RealWorld | **37.0** | 35.8 | 36.3 | 33.9 | 32.4 | 4.6 |
| MobiAct | 48.1 | **49.2** | 47.5 | 45.7 | 44.1 | 5.1 |
| VTT-ConIoT | 3.4 | 3.9 | 3.5 | 3.9 | **4.7** | 1.3 |

**Bold** = best patch size per dataset. Smaller patches (1.0-1.25s) consistently perform best on easy/medium datasets. VTT-ConIoT is at noise level regardless of patch size (<5% for all). The fixed 1.0s choice is within 1.1% of the best possible patch for 3/4 datasets.

---

## Reproducibility

- Raw JSON results: `test_output/baseline_evaluation/*_evaluation.json`
- Evaluation scripts: `val_scripts/human_activity_recognition/evaluate_*.py`
- Results generator: `scripts/generate_results_table.py`
- Run all: `bash scripts/auto_eval_after_training.sh`
- Random seed: 3431 (consistent across all splits)
