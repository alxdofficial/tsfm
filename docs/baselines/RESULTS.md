# Baseline Evaluation Results

Generated: 2026-02-17 | Framework: 4-metric unified evaluation | Seed: 3431

## Models

| Model | How It Works | Embed Dim | Zero-Shot | Supervised |
|-------|-------------|:---------:|-----------|------------|
| **TSFM (ours)** | Dual-branch Transformer trained with CLIP-style contrastive alignment between IMU patches and text labels | 384 | Cosine sim with text embeddings | End-to-end cosine sim |
| **LiMU-BERT** | BERT-style masked reconstruction on 20-step IMU sub-windows; predicts masked timesteps from context | 72 | GRU classifier + group scoring | End-to-end encoder + GRU |
| **MOMENT** | General time-series Transformer pretrained on diverse time-series data (no HAR); processes each IMU channel independently | 6144 | SVM-RBF + group scoring | End-to-end encoder + linear |
| **CrossHAR** | Hierarchical self-supervised pretraining combining masked reconstruction and contrastive learning on IMU sequences | 72 | Transformer_ft classifier + group scoring | End-to-end encoder + Transformer_ft |
| **LanHAR** | 2-stage CLIP-style alignment: (1) fine-tune SciBERT on activity text, (2) train a sensor Transformer to align with text space | 768 | Cosine sim with text embeddings | End-to-end cosine sim |

## Fairness Notes

**Training data**: All HAR-pretrained models (TSFM, LiMU-BERT, CrossHAR, LanHAR) use 10 training
datasets. All 4 test datasets were never seen during any model's pretraining. MOMENT was pretrained
on general time-series data with no HAR-specific data.

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

## Adaptations from Original Papers

We adapt each baseline to our unified benchmark rather than replicating each paper's exact experiment.
The table below documents every significant deviation and its fairness rationale.

| Baseline | What We Changed | Original Paper Protocol | Our Adaptation | Fairness Rationale |
|----------|----------------|------------------------|----------------|-------------------|
| **LiMU-BERT** | Window-level scoring | Scores each 20-step sub-window independently (6x more evaluation units per window) | Majority vote across 6 sub-windows per 120-step window | All models must be scored on the same evaluation units (windows) for comparable n_samples |
| **LiMU-BERT** | Single combined checkpoint | Separate pretrained models per dataset | One model pretrained on all 10 datasets combined | Unified pretraining for fair cross-dataset comparison |
| **CrossHAR** | End-to-end supervised fine-tuning | Freezes encoder; trains only classifier head on static pre-extracted embeddings | Fine-tunes encoder + classifier jointly | All baselines use end-to-end fine-tuning for supervised metrics, giving each encoder a chance to adapt; this slightly *advantages* CrossHAR vs its paper |
| **MOMENT** | Linear head for supervised | Paper's classification evaluation uses only SVM-RBF on frozen embeddings (no fine-tuning) | Linear head (from MOMENT codebase's `ClassificationHead`) fine-tuned end-to-end | SVM is not differentiable; linear head enables end-to-end fine-tuning consistent with other baselines |
| **LanHAR** | No target data in Stage 2 | Sensor encoder trains on source + target data combined | Source data only (test data never seen) | No other baseline sees test data during training; exclusion prevents unfair distributional advantage but slightly *disadvantages* LanHAR vs its paper |
| **LanHAR** | Supervised fine-tuning added | Paper is zero-shot only (no supervised protocol) | Fine-tune sensor encoder via cosine sim with frozen text prototypes | Extension for benchmark completeness; uses LanHAR's native cosine-sim mechanism |
| **All** | Unified batch sizes | Each paper uses its own batch size (typically 128) | 512 for classifiers, 32 for fine-tuning | Speed optimization; applied uniformly across all baselines |

See [BASELINE_IMPLEMENTATION_NOTES.md](BASELINE_IMPLEMENTATION_NOTES.md) for full per-model implementation details.

## Test Datasets

We evaluate on 3 main test datasets with high label group coverage (85-100%), plus 1 severe
out-of-domain dataset (VTT-ConIoT) reported separately due to its fundamentally different
characteristics. See [Severe Out-of-Domain: VTT-ConIoT](#severe-out-of-domain-vtt-coniot) below.

### Main Test Datasets (85-100% label coverage)

| Dataset | Windows | Classes | Group Coverage | Difficulty |
|---------|---------|---------|:-:|------------|
| MotionSense | 12,080 | 6 | 100% | Easy (basic locomotion) |
| RealWorld | 27,138 | 8 | 100% | Medium (multi-placement) |
| MobiAct | 4,345 | 13 | 85% | Hard (falls, vehicle entry) |

### Out-of-Domain Test Dataset (50% label coverage)

| Dataset | Windows | Classes | Group Coverage | Difficulty |
|---------|---------|---------|:-:|------------|
| VTT-ConIoT | 2,058 | 16 | 50% | Severe (industrial/construction) |

**Why VTT-ConIoT is reported separately**: 8 of 16 activity labels (carrying, climbing ladder,
kneeling work, leveling paint, lifting, pushing cart, roll painting, spraying paint) have no
semantic equivalent in the 10 training datasets. All models are guaranteed to fail on these
activities regardless of architecture quality. This 50% coverage floor makes VTT-ConIoT a test
of severe domain shift rather than cross-dataset generalization.

---

## Average Across Main Datasets

| Model | ZS-Open Acc | ZS-Open F1 | ZS-Close Acc | ZS-Close F1 | 1% Acc | 1% F1 | 10% Acc | 10% F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 36.2 | 9.7 | 45.5 | 28.9 | 68.2 | 58.6 | 83.2 | 76.7 |
| **LiMU-BERT** | 21.2 | 6.7 | 33.2 | 23.6 | 25.6 | 15.9 | 62.6 | 52.1 |
| **MOMENT** | 25.7 | 7.0 | 41.2 | 28.5 | 71.5 | 64.8 | 81.3 | 76.1 |
| **CrossHAR** | 17.0 | 5.5 | 35.4 | 28.9 | 62.5 | 56.3 | 80.6 | 75.0 |
| **LanHAR** | 14.2 | 7.4 | 28.2 | 20.4 | 41.3 | 31.5 | 56.1 | 51.9 |

---

## Per-Dataset Results

### Zero-Shot Open-Set

*Model predicts from all 87 training labels; correct if predicted label maps to same group as ground truth.*

*Text-aligned models use cosine similarity; classifier-based models use a trained native classifier.*

| Model | MobiAct Acc | MobiAct F1 | MotionSense Acc | MotionSense F1 | RealWorld Acc | RealWorld F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 31.8 | 8.8 | 38.7 | 10.5 | 38.1 | 9.7 |
| **LiMU-BERT** | 6.1 | 2.0 | 28.4 | 10.3 | 29.1 | 7.7 |
| **MOMENT** | 28.7 | 7.0 | 33.8 | 8.0 | 14.6 | 6.0 |
| **CrossHAR** | 13.5 | 4.2 | 16.2 | 5.4 | 21.5 | 7.0 |
| **LanHAR** | 11.4 | 4.4 | 14.0 | 6.4 | 17.3 | 11.4 |

### Zero-Shot Closed-Set

*Text-aligned models predict from test labels only (exact match). Classifier-based models mask logits to test-relevant groups (group match).*

| Model | MobiAct Acc | MobiAct F1 | MotionSense Acc | MotionSense F1 | RealWorld Acc | RealWorld F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 48.1 | 16.5 | 51.5 | 45.3 | 37.0 | 24.8 |
| **LiMU-BERT** | 29.3 | 13.1 | 39.9 | 37.8 | 30.5 | 19.9 |
| **MOMENT** | 40.9 | 24.6 | 51.6 | 39.3 | 31.1 | 21.6 |
| **CrossHAR** | 23.3 | 17.7 | 42.8 | 39.5 | 40.3 | 29.5 |
| **LanHAR** | 17.5 | 11.6 | 37.1 | 30.7 | 30.0 | 19.1 |

### 1% Supervised (End-to-End Fine-Tuning)

*Encoder fine-tuned end-to-end with 1% of labeled data. Text-aligned models classify via cosine similarity; others use native classifier heads.*

| Model | MobiAct Acc | MobiAct F1 | MotionSense Acc | MotionSense F1 | RealWorld Acc | RealWorld F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 45.5 | 17.8 | 85.3 | 84.7 | 73.8 | 73.2 |
| **LiMU-BERT** | 8.3 | 1.2 | 22.6 | 6.1 | 45.9 | 40.3 |
| **MOMENT** | 54.9 | 36.8 | 87.4 | 87.5 | 72.1 | 70.2 |
| **CrossHAR** | 42.8 | 30.7 | 78.6 | 77.6 | 66.0 | 60.7 |
| **LanHAR** | 34.5 | 15.2 | 40.3 | 36.6 | 49.2 | 42.6 |

### 10% Supervised (End-to-End Fine-Tuning)

*Encoder fine-tuned end-to-end with 10% of labeled data. Text-aligned models classify via cosine similarity; others use native classifier heads.*

| Model | MobiAct Acc | MobiAct F1 | MotionSense Acc | MotionSense F1 | RealWorld Acc | RealWorld F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 72.6 | 53.8 | 93.3 | 92.4 | 83.7 | 83.9 |
| **LiMU-BERT** | 61.8 | 35.2 | 69.4 | 68.5 | 56.5 | 52.8 |
| **MOMENT** | 71.3 | 55.4 | 92.1 | 92.1 | 80.6 | 80.8 |
| **CrossHAR** | 69.7 | 54.7 | 91.6 | 91.0 | 80.5 | 79.2 |
| **LanHAR** | 35.9 | 24.3 | 75.8 | 76.1 | 56.5 | 55.4 |

---

## Key Observations

1. **TSFM leads zero-shot** — 45.5% closed-set avg accuracy on 3 main datasets, ahead of MOMENT (41.2%),
   CrossHAR (35.4%), LiMU-BERT (33.4%), and LanHAR (28.2%). Text alignment provides a genuine advantage
   for cross-dataset transfer without any test-time labels.

2. **MOMENT leads supervised, TSFM close behind** — At 10%, MOMENT 81.3% vs TSFM 83.2% avg accuracy
   (TSFM slightly ahead). At 1%, MOMENT leads (71.5% vs 68.2%), likely benefiting from its larger
   embedding dimension (6144 vs 384) and general time-series pretraining.

3. **CrossHAR is a strong third** — 80.6% at 10% supervised, competitive with TSFM/MOMENT on
   MotionSense (91.6%) and RealWorld (80.5%), despite a much smaller embedding (72-dim).

4. **LiMU-BERT struggles at low data** — Only 25.6% at 1% supervised avg, the lowest among all models.
   Performance recovers at 10% (62.6%), suggesting the GRU classifier needs more data to converge.

5. **LanHAR underperforms across all metrics** — Despite being text-aligned, LanHAR's from-scratch
   SciBERT training on small HAR data limits its zero-shot and supervised transfer quality.

6. **TSFM uses a fixed 1.0s patch size** — no per-dataset sweep or test-time tuning. This is a
   metadata-only decision: at 20Hz, 1.0s patches give the finest temporal resolution while producing
   enough tokens for the encoder. See [Patch Size Sensitivity](#tsfm-patch-size-sensitivity) below.

---

## Severe Out-of-Domain: VTT-ConIoT

VTT-ConIoT is an industrial/construction activity dataset with 16 classes, of which only 8 (50%)
have semantic equivalents in the 10 training datasets. The remaining 8 activities (carrying,
climbing ladder, kneeling work, leveling paint, lifting, pushing cart, roll painting, spraying paint)
are completely novel — no model can correctly classify them in zero-shot mode, and even supervised
fine-tuning has very limited training signal.

We report VTT-ConIoT separately because it tests a fundamentally different condition: **severe
domain shift** where the activity vocabulary is only half covered, rather than the near-complete
coverage (85-100%) of the 3 main test datasets.

| Model | ZS-Open Acc | ZS-Open F1 | ZS-Close Acc | ZS-Close F1 | 1% Acc | 1% F1 | 10% Acc | 10% F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | 1.7 | 0.6 | 3.4 | 2.6 | 12.1 | 11.3 | 25.6 | 25.1 |
| **LiMU-BERT** | 3.4 | 0.9 | 7.1 | 2.1 | 7.7 | 4.2 | 19.3 | 8.4 |
| **MOMENT** | 1.6 | 0.4 | 5.2 | 2.0 | 21.3 | 18.6 | 38.6 | 37.2 |
| **CrossHAR** | 0.7 | 0.4 | 5.0 | 2.7 | 17.9 | 16.5 | 29.5 | 24.3 |
| **LanHAR** | 8.3 | 2.1 | 6.9 | 3.2 | 6.3 | 2.6 | 13.0 | 10.9 |

**Observations**: All models score near random on zero-shot (<8% accuracy). With 10% supervised
data, MOMENT leads (38.6%), likely because its general time-series pretraining provides useful
signal even for novel activity types. TSFM and CrossHAR are in the 25-30% range. LiMU-BERT and
LanHAR struggle most (<14%).

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
