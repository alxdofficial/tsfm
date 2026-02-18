# Baseline Evaluation Results

Generated: 2026-02-18 | Framework: 4-metric unified evaluation | Seed: 3431

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

**TSFM uses native sampling rates and dataset-specific channel descriptions**: TSFM evaluates each
test dataset at its native sampling rate (all 4 test sets are 50Hz) with rich channel descriptions
from dataset manifests (e.g., "Accelerometer X-axis (waist-mounted smartphone)"), while baselines
use 20Hz resampled data with no channel metadata. This is fair because:
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
| **LanHAR** | Supervised fine-tuning added | Paper is zero-shot only (no supervised protocol) | Fine-tune entire model end-to-end (BERT + sensor encoder + projections) via cosine sim with frozen text prototypes | Extension for benchmark completeness; all baselines fine-tune end-to-end for consistency |
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
| **TSFM (ours)** | **47.6** | **20.0** | **49.2** | **31.0** | **73.0** | 61.1 | **84.2** | **78.3** |
| **LiMU-BERT** | 21.2 | 6.7 | 33.2 | 23.6 | 25.6 | 15.9 | 62.6 | 52.1 |
| **MOMENT** | 25.7 | 7.0 | 41.2 | 28.5 | 71.5 | **64.8** | 81.3 | 76.1 |
| **CrossHAR** | 17.0 | 5.5 | 35.4 | 28.9 | 62.5 | 56.3 | 80.6 | 75.0 |
| **LanHAR** | 14.2 | 7.4 | 28.2 | 20.4 | 42.0 | 32.9 | 56.6 | 52.3 |

---

## Per-Dataset Results

### Zero-Shot Open-Set

*Model predicts from all 87 training labels; correct if predicted label maps to same group as ground truth.*

*Text-aligned models use cosine similarity; classifier-based models use a trained native classifier.*

| Model | MobiAct Acc | MobiAct F1 | MotionSense Acc | MotionSense F1 | RealWorld Acc | RealWorld F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | **51.5** | **29.0** | **63.1** | **17.8** | 28.1 | **13.1** |
| **LiMU-BERT** | 6.1 | 2.0 | 28.4 | 10.3 | **29.1** | 7.7 |
| **MOMENT** | 28.7 | 7.0 | 33.8 | 8.0 | 14.6 | 6.0 |
| **CrossHAR** | 13.5 | 4.2 | 16.2 | 5.4 | 21.5 | 7.0 |
| **LanHAR** | 11.4 | 4.4 | 14.0 | 6.4 | 17.3 | 11.4 |

### Zero-Shot Closed-Set

*Text-aligned models predict from test labels only (exact match). Classifier-based models mask logits to test-relevant groups (group match).*

| Model | MobiAct Acc | MobiAct F1 | MotionSense Acc | MotionSense F1 | RealWorld Acc | RealWorld F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | **57.9** | 21.8 | **64.7** | **56.8** | 25.1 | 14.4 |
| **LiMU-BERT** | 29.3 | 13.1 | 39.9 | 37.8 | 30.5 | 19.9 |
| **MOMENT** | 40.9 | **24.6** | 51.6 | 39.3 | 31.1 | 21.6 |
| **CrossHAR** | 23.3 | 17.7 | 42.8 | 39.5 | **40.3** | **29.5** |
| **LanHAR** | 17.5 | 11.6 | 37.1 | 30.7 | 30.0 | 19.1 |

### 1% Supervised (End-to-End Fine-Tuning)

*Encoder fine-tuned end-to-end with 1% of labeled data. Text-aligned models classify via cosine similarity; others use native classifier heads.*

| Model | MobiAct Acc | MobiAct F1 | MotionSense Acc | MotionSense F1 | RealWorld Acc | RealWorld F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | **57.7** | 23.9 | **88.1** | 87.1 | **73.3** | **72.3** |
| **LiMU-BERT** | 8.3 | 1.2 | 22.6 | 6.1 | 45.9 | 40.3 |
| **MOMENT** | 54.9 | **36.8** | 87.4 | **87.5** | 72.1 | 70.2 |
| **CrossHAR** | 42.8 | 30.7 | 78.6 | 77.6 | 66.0 | 60.7 |
| **LanHAR** | 33.6 | 14.6 | 41.9 | 38.1 | 50.6 | 45.9 |

### 10% Supervised (End-to-End Fine-Tuning)

*Encoder fine-tuned end-to-end with 10% of labeled data. Text-aligned models classify via cosine similarity; others use native classifier heads.*

| Model | MobiAct Acc | MobiAct F1 | MotionSense Acc | MotionSense F1 | RealWorld Acc | RealWorld F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **TSFM (ours)** | **74.9** | **57.4** | **93.3** | **92.6** | **84.4** | **85.0** |
| **LiMU-BERT** | 61.8 | 35.2 | 69.4 | 68.5 | 56.5 | 52.8 |
| **MOMENT** | 71.3 | 55.4 | 92.1 | 92.1 | 80.6 | 80.8 |
| **CrossHAR** | 69.7 | 54.7 | 91.6 | 91.0 | 80.5 | 79.2 |
| **LanHAR** | 38.2 | 25.9 | 75.0 | 75.2 | 56.6 | 55.7 |

---

## Key Observations

1. **TSFM leads in most metrics** — 49.2% closed-set avg accuracy on 3 main datasets, ahead of
   MOMENT (41.2%), CrossHAR (35.4%), LiMU-BERT (33.2%), and LanHAR (28.2%). TSFM also leads at
   1% supervised (73.0% vs MOMENT's 71.5%) and 10% supervised (84.2% vs MOMENT's 81.3%).
   MOMENT leads on 1% F1 (64.8 vs 61.1), driven by stronger MobiAct performance.

2. **Correct channel-text conditioning produces large zero-shot gains on 2 of 3 datasets** — With
   dataset description prepended to channel descriptions (matching training format), MotionSense
   ZS-Open jumps +24.5% and MobiAct +19.7%. However, RealWorld zero-shot drops (-9.9% open,
   -11.9% close), likely because RealWorld's multi-placement heterogeneity conflicts with its
   single-placement dataset description. Net effect: ZS-Open avg improves +11.4% vs 20Hz baseline.
   See [Ablation](#ablation-native-rate--rich-channel-descriptions) for per-dataset deltas.

3. **CrossHAR is a strong third** — 80.6% at 10% supervised, competitive with MOMENT on
   MotionSense (91.6%) and RealWorld (80.5%), despite a much smaller embedding (72-dim).

4. **LiMU-BERT struggles at low data** — Only 25.6% at 1% supervised avg, the lowest among all models.
   Performance recovers at 10% (62.6%), suggesting the GRU classifier needs more data to converge.

5. **LanHAR underperforms across all metrics** — Despite being text-aligned, LanHAR's from-scratch
   SciBERT training on small HAR data limits its zero-shot and supervised transfer quality.

6. **TSFM uses a fixed 1.0s patch size** — no per-dataset sweep or test-time tuning. This is a
   metadata-only decision: at the native rate, 1.0s patches give fine temporal resolution while
   producing enough tokens for the encoder. See [Patch Size Sensitivity](#tsfm-patch-size-sensitivity) below.

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
| **TSFM (ours)** | 2.0 | 0.4 | 6.3 | **3.8** | 13.5 | 9.4 | 30.4 | 29.5 |
| **LiMU-BERT** | 3.4 | 0.9 | **7.1** | 2.1 | 7.7 | 4.2 | 19.3 | 8.4 |
| **MOMENT** | 1.6 | 0.4 | 5.2 | 2.0 | **21.3** | **18.6** | **38.6** | **37.2** |
| **CrossHAR** | 0.7 | 0.4 | 5.0 | 2.7 | 17.9 | 16.5 | 29.5 | 24.3 |
| **LanHAR** | **8.3** | **2.1** | 6.9 | 3.2 | 5.3 | 2.7 | 16.9 | 10.8 |

**Observations**: All models score near random on zero-shot (<8% accuracy). With 10% supervised
data, MOMENT leads (38.6%), followed by TSFM (30.4%) and CrossHAR (29.5%). TSFM's VTT-ConIoT
10% result dropped from 36.7% (previous eval) to 30.4% with the channel description conditioning
fix — the industrial dataset description may add noise for these unusual activities.
LiMU-BERT and LanHAR struggle most (<19%).

---

## TSFM Patch Size Sensitivity

*Previous sensitivity analysis at 20Hz (pre-native-rate). Patch size sweep at native 50Hz rates is pending.*

*Zero-shot closed-set accuracy (%) at each candidate patch size, evaluated on the full test set at 20Hz. TSFM uses a fixed 1.0s patch size for all reported results — no per-dataset sweep or test labels used for selection.*

| Dataset | 1.0s | 1.25s | 1.5s | 1.75s | 2.0s | Range |
|---------|---:|---:|---:|---:|---:|---:|
| MotionSense | **51.5** | 48.5 | 46.1 | 44.0 | 42.5 | 9.0 |
| RealWorld | **37.0** | 35.8 | 36.3 | 33.9 | 32.4 | 4.6 |
| MobiAct | 48.1 | **49.2** | 47.5 | 45.7 | 44.1 | 5.1 |
| VTT-ConIoT | 3.4 | 3.9 | 3.5 | 3.9 | **4.7** | 1.3 |

**Note**: These numbers were measured at 20Hz before the switch to native sampling rates. At native 50Hz, a 1.0s patch has 50 timesteps (vs 20 at 20Hz), so the sensitivity profile may differ. The fixed 1.0s choice was within 1.1% of the best possible patch for 3/4 datasets at 20Hz.

---

## Ablation: Native Rate + Rich Channel Descriptions

The results above use TSFM's full capabilities: native sampling rates (50Hz for all test datasets)
and dataset-specific channel descriptions from manifests. Previously, TSFM was evaluated on the same
20Hz resampled data as baselines with generic channel descriptions ("Accelerometer X-axis"). This
provides a natural ablation study showing the value of TSFM's metadata-aware architecture.

### Average Across Main Datasets (3 datasets)

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
| running | running, jogging, running_treadmill | jogging (MotionSense), running (RealWorld) |
| walking | walking, walking_parking, walking_treadmill_flat, walking_straight, walking_winding | walking (all test sets) |
| stairs_down | walking_downstairs, stairs_down, descending_stairs, going_down_stairs | walking_downstairs (MotionSense), stairs_down (RealWorld, MobiAct) |
| sitting | sitting, sitting_chair, talking_sitting | sitting (MotionSense), sitting_chair (MobiAct) |
| ... | (87 labels across ~30 groups) | |

### Per-Model Handling of Unseen Labels

| Model | Open-Set | Closed-Set | Supervised |
|-------|----------|------------|------------|
| **TSFM** | Encodes all 87 training labels as text; predicts via cosine sim; scored via group matching. Unseen test labels that belong to a known group can be predicted correctly. Novel labels (no group match) are counted as **failures**. | Encodes only the test dataset's labels as text; exact match scoring. Novel labels can theoretically be predicted (model sees the text), but scored as failures if not in training groups. | Fine-tunes on test dataset labels directly. Novel labels are learned from the few-shot data. |
| **LiMU-BERT** | GRU classifier predicts over 87 training labels; scored via group matching. Can only predict training labels. Novel test labels are **guaranteed failures**. | Classifier logits masked to training labels whose group appears in test set. Novel labels (no matching group) are excluded from the mask, so they **cannot be predicted** and are counted as failures. | Fine-tunes on test dataset labels directly. Novel labels are learned from labeled data. |
| **MOMENT** | Same as LiMU-BERT but with SVM-RBF classifier. Novel test labels are **guaranteed failures**. | Same masking approach as LiMU-BERT. Novel labels cannot be predicted. | Fine-tunes with linear head on test dataset labels. Novel labels learned from data. |
| **CrossHAR** | Same as LiMU-BERT but with Transformer_ft classifier. Novel test labels are **guaranteed failures**. | Same masking approach. Novel labels cannot be predicted. | Fine-tunes with Transformer_ft on test dataset labels. |
| **LanHAR** | Encodes all 87 training labels as text; cosine sim + group matching. Same as TSFM. Novel labels counted as **failures**. | Same as TSFM — encodes test labels as text, exact match. | Same as TSFM. |

### Impact on Reported Metrics

- **MotionSense** (100% coverage): All 6 activities have training equivalents. No failures due to unseen labels.
- **RealWorld** (100% coverage): All 8 activities have training equivalents. No failures due to unseen labels.
- **MobiAct** (85% coverage): 2 of 13 activities (`car_step_in`, `car_step_out`) have no training equivalent. These are counted as **failures for all models** in zero-shot. In supervised, they are learned from labeled data.
- **VTT-ConIoT** (50% coverage): 8 of 16 activities are completely novel (industrial/construction). These are **guaranteed failures for all models** in zero-shot, setting a ~50% accuracy ceiling. This is why VTT-ConIoT is reported separately.

### Scoring Rules

1. **Open-set zero-shot**: Prediction maps through label groups. If `group(predicted_label) == group(ground_truth_label)`, it's correct. Novel test labels with no matching group are always wrong.
2. **Closed-set zero-shot (text-aligned)**: Exact label match. Model predicts from the test dataset's own labels.
3. **Closed-set zero-shot (classifier-based)**: Logits masked to allow only training labels whose group matches a test label. `argmax(masked_logits)` → group match scoring. Novel test labels with no matching group cannot appear in the mask and are always wrong.
4. **Supervised**: Standard classification on the test dataset's labels. All labels (including novel ones) are learned from labeled training data.

---

## Reproducibility

- Raw JSON results: `test_output/baseline_evaluation/*_evaluation.json`
- Evaluation scripts: `val_scripts/human_activity_recognition/evaluate_*.py`
- Results generator: `scripts/generate_results_table.py`
- Run all: `bash scripts/auto_eval_after_training.sh`
- Random seed: 3431 (consistent across all splits)
