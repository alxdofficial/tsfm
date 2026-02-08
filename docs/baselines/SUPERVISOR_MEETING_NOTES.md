# Baseline Comparison Status - Meeting Notes

## TL;DR

We outperform the most directly comparable baseline (NLS-HAR from AAAI 2025) on overlapping test datasets:
- **MotionSense**: Our 53.7% F1 vs their 38.97% F1 (+15pp)
- **MobiAct**: Our 19.8% F1 vs their 16.93% F1 (+3pp)

**Caveat**: Our zero-shot protocols are similar but not identical (see Section 4).

---

## 1. Baseline Models (Detailed)

### NLS-HAR (AAAI 2025)
- **Paper**: "Limitations in Employing Natural Language Supervision for Sensor-Based HAR" (Haresamudram et al.)
- **Approach**: Contrastive IMU-text alignment (closest to ours). DistilBERT text encoder, 3-layer CNN sensor encoder.
- **Code**: **No** — not released, not on author's GitHub ([harkash](https://github.com/harkash)), CatalyzeX confirms unavailable
- **Checkpoints**: **No**
- **Train data**: Capture-24 (single dataset, 151 users, 177 activities)
- **Published zero-shot F1** (macro, Capture-24 pretrained):

| Dataset | F1 |
|---------|-----|
| HHAR | 31.05% |
| MotionSense | 38.97% |
| MobiAct | 16.93% |
| MHEALTH | 11.15% |
| PAMAP2 | 10.88% |
| Myogym | 1.47% |

- Also reports "unseen activity" zero-shot (Table 5): HHAR 55.01%, Myogym 36.96%, MobiAct 56.71%, MotionSense 40.68%, MHEALTH 39.54%, PAMAP2 55.33%
- With adaptation (minimal target labels): HHAR 63.43%, MobiAct 65.28%, PAMAP2 54.99%

### GOAT (IMWUT 2024)
- **Paper**: "A Generalized Cross-Dataset Activity Recognition Framework with Natural Language Supervision" (Miao et al.)
- **Approach**: Text-supervised cross-dataset HAR, leave-one-dataset-out. CLIP/BERT/GPT-2/Llama text encoders, Transformer activity encoder (0.499M params).
- **Code**: **Placeholder only** — [github.com/wdkhuans/GOAT](https://github.com/wdkhuans/GOAT) has 1 commit, only a README saying "Code being organized. Will be released soon." Still empty as of Feb 2026.
- **Checkpoints**: **No**
- **Published F1** (macro, leave-one-dataset-out, mean ± std over 5 seeds):

| Dataset | GOAT-CLIP | GOAT-BERT | Best |
|---------|-----------|-----------|------|
| RealWorld | **78.49 ± 0.69** | 72.17 ± 0.70 | CLIP |
| Realdisp | **81.39 ± 0.80** | 73.63 ± 1.19 | CLIP |
| Opportunity | 49.05 ± 0.63 | **54.35 ± 0.89** | BERT |
| PAMAP | **77.13 ± 0.98** | 68.94 ± 1.35 | CLIP |
| Daphnet FoG | 56.44 ± 1.17 | **64.14 ± 1.02** | BERT |

- Zero-shot on PAMAP (4 held-out activities): nordic walking 94.48%, vacuum cleaning 57.71%, ironing 33.79%, rope jumping 65.19%

### LanHAR (IMWUT 2025, arXiv 2024)
- **Paper**: "Large Language Model-Guided Semantic Alignment for HAR" (Yan et al., Lehigh University)
- **Approach**: Two-stage: (1) train SciBERT text encoder with LLM-generated semantic interpretations + triplet loss, (2) train Transformer sensor encoder to map into aligned language space. **Zero-shot cross-dataset** (no fine-tuning on target).
- **Code**: **Yes** — [github.com/DASHLab/LanHAR](https://github.com/DASHLab/LanHAR) (Apache-2.0, 2 stars). Requires LLM API to generate semantic interpretations first.
- **Checkpoints**: **No** — must train from scratch
- **Datasets**: HHAR, UCI-HAR, MotionSense, Shoaib (4 shared activities: walking, upstairs, downstairs, sitting/standing). PAMAP2 evaluated separately (within-dataset participant split, not cross-dataset).
- **Published F1** (cross-dataset zero-shot, single source → target):

| Source → Target | LanHAR F1 |
|-----------------|-----------|
| HHAR → UCI | 0.804 |
| HHAR → MotionSense | 0.795 |
| HHAR → Shoaib | 0.693 |
| UCI → HHAR | 0.808 |
| UCI → MotionSense | 0.765 |
| UCI → Shoaib | 0.718 |
| MotionSense → HHAR | 0.617 |
| MotionSense → UCI | 0.713 |
| MotionSense → Shoaib | 0.724 |
| Shoaib → HHAR | 0.687 |
| Shoaib → UCI | 0.730 |
| Shoaib → MotionSense | 0.721 |
| **Average** | **0.731** |

- Multi-source (3 datasets → 1 target, zero-shot): HHAR 0.704, MotionSense 0.760, UCI 0.749, Shoaib 0.712

**Important**: LanHAR evaluates only 4 activity classes (intersection across datasets). Our evaluation uses all activities per dataset.

### CrossHAR (IMWUT 2024)
- **Paper**: "CrossHAR: Generalizing Cross-Dataset HAR via Hierarchical Self-Supervised Pretraining" (Hong et al.)
- **Approach**: Hierarchical self-supervised pretraining (contrastive + masked reconstruction), NOT language-based. Pretrain on all datasets combined (unlabeled), fine-tune on source dataset labels, test on target.
- **Code**: **Yes** — [github.com/kingdomrush2/CrossHAR](https://github.com/kingdomrush2/CrossHAR) (19 stars, PyTorch 1.12)
- **Checkpoints**: **No** — must pretrain from scratch
- **Datasets**: HHAR, UCI-HAR, MotionSense, Shoaib (4 shared activities, resampled to 20Hz)
- **Published results** (accuracy, not F1; multi-source 3→1):

| Target | CrossHAR Accuracy |
|--------|-------------------|
| UCI | 88.68% |
| Shoaib | 73.67% |
| MotionSense | 78.26% |
| HHAR | 76.19% |

- Single-source best: HHAR→UCI 91.52%, UCI→MotionSense 87.90%
- **Note**: Paper reports accuracy primarily, not F1. Only 4 coarse activity classes evaluated.

### LIMU-BERT (SenSys 2021)
- **Paper**: "LIMU-BERT: Unleashing the Potential of Unlabeled Data for IMU Sensing Applications" (Xu et al., NTU)
- **Approach**: Self-supervised masked pretraining on unlabeled IMU, then GRU classifier fine-tuned with labeled data. Semi-supervised — headline results use only 1% of labels.
- **Code**: **Yes** — [github.com/dapowan/LIMU-BERT-Public](https://github.com/dapowan/LIMU-BERT-Public) (MIT license, PyTorch)
- **Checkpoints**: **Yes** — pretrained encoder + fine-tuned classifier `.pt` files in `saved/` directory for all 4 datasets. Model is tiny (~62K params encoder, ~9K params classifier).
- **Datasets**: HHAR, UCI-HAR, MotionSense, Shoaib (all at 20Hz, 6-second windows)
- **Published F1** (macro, 1% labeling rate, within-dataset supervised):

| Dataset | LIMU-GRU Acc | LIMU-GRU F1 |
|---------|-------------|-------------|
| HHAR | 96.4% | 96.2% |
| UCI-HAR | 92.4% | 92.3% |
| MotionSense | 92.7% | 89.9% |
| Shoaib | 90.0% | 89.9% |

- **These are supervised results** (within-dataset, 1% labels) — NOT zero-shot or cross-dataset. No cross-dataset zero-shot numbers published.

### IMU2CLIP (Findings of EMNLP 2023)
- **Paper**: "IMU2CLIP: Multimodal Contrastive Learning for IMU Motion Sensors from Egocentric Videos and Text" (Moon et al., Meta)
- **Approach**: Aligns IMU embeddings into CLIP space via contrastive learning with video+text. CNN+GRU encoder producing 128-dim embeddings.
- **Code**: **Yes but archived** — [github.com/facebookresearch/imu2clip](https://github.com/facebookresearch/imu2clip) (CC-BY-NC, archived July 2025)
- **Checkpoints**: **No** — promised but never released. Multiple GitHub issues requesting weights went unanswered. Requires Ego4D data (DUA required) to retrain.
- **Datasets**: Ego4D (head-mounted GoPro IMU, 6-axis, resampled to 200Hz, 5s windows) and Aria. **No standard HAR benchmarks evaluated.**
- **Published results** (Ego4D activity recognition, 4 classes):

| Setting | F1 | Accuracy |
|---------|-----|---------|
| Zero-shot (IMU↔text) | 31.89% | 36.38% |
| Probing (IMU↔text) | 45.12% | 58.01% |
| Fine-tuned (IMU↔text) | 45.15% | 63.14% |

- **Third-party evaluation on out-of-distribution HAR**: ~20% balanced accuracy zero-shot — poor transfer.
- **Not directly comparable** to our work on standard HAR benchmarks.

### Summary Table

| Baseline | Code | Checkpoints | Zero-Shot Numbers We Can Cite | Metric |
|----------|:----:|:-----------:|-------------------------------|--------|
| NLS-HAR | No | No | MotionSense 38.97%, MobiAct 16.93% | Macro F1 |
| GOAT | No | No | RealWorld 78.49% | Macro F1 |
| LanHAR | Yes | No | MotionSense 0.760*, Shoaib 0.712* | F1 (4 classes only) |
| CrossHAR | Yes | No | MotionSense 78.26%* (accuracy) | Accuracy (4 classes only) |
| LIMU-BERT | Yes | Yes | None (supervised only, no zero-shot) | Macro F1 |
| IMU2CLIP | Archived | No | None (Ego4D only, no HAR benchmarks) | F1 |

*LanHAR and CrossHAR numbers are multi-source (3→1 target) with only 4 activity classes — not directly comparable to our all-activities evaluation.

---

## 2. Datasets

### Our Training Set (11 datasets)
UCI-HAR, HHAR, MHEALTH, PAMAP2, WISDM, UniMiB-SHAR, DSADS, HAPT, KU-HAR, VTT-ConIoT, RecGym

**Total: ~93 unique activity labels**

### Our Zero-Shot Test Set (3 datasets)
MotionSense, RealWorld, MobiAct

**Model never sees any data from these during training.**

### NLS-HAR's Setup
- **Training**: Capture-24 dataset (single dataset, 151 users, 177 activities)
- **Testing**: 6 datasets: HHAR (31.05%), Myogym (1.47%), MobiAct (16.93%), MotionSense (38.97%), MHEALTH (11.15%), PAMAP2 (10.88%)

### LIMU-BERT's Test Datasets
HHAR, UCI-HAR, MotionSense, Shoaib

### Overlap for Comparison
We can compare on **MotionSense** and **MobiAct** (both zero-shot for us and for NLS-HAR). HHAR is in our training set so can't be compared directly unless we retrain without it.

---

## 3. Preliminary Results (Verified)

### Closed-Set F1 (Apples-to-Apples with NLS-HAR)

| Dataset | Our Model | NLS-HAR | Difference |
|---------|-----------|---------|------------|
| **MotionSense** | **53.72%** | 38.97% | **+14.75pp** |
| **MobiAct** | **19.83%** | 16.93% | **+2.90pp** |

### Our Full Zero-Shot Results

| Dataset | Accuracy (open) | F1 (open) | F1 (closed) | Labels |
|---------|-----------------|-----------|-------------|--------|
| MotionSense | 56.93% | 22.39% | 53.72% | 6 |
| RealWorld | 45.73% | 26.89% | 36.89% | 8 |
| MobiAct | 43.22% | 27.85% | 19.83% | 13 |

**Note on MobiAct**: Lower F1 because it has 6 novel activities (falls, vehicle entry) that don't exist in any training data.

---

## 4. Metric Methodology

### Two Evaluation Modes

**Closed-Set F1 (for NLS-HAR comparison)**
- Predict from ONLY the target dataset's labels (e.g., 6 labels for MotionSense)
- Compare raw predicted label vs raw ground truth label
- Macro F1 = average F1 across all classes
- **Similar to NLS-HAR but not identical** (see protocol difference below)

**Open-Set Accuracy/F1 (our harder evaluation)**
- Predict from ALL ~100 training labels
- Map predictions to semantic groups (e.g., "jogging" → "running" group)
- Give credit if predicted group matches ground truth group
- **Harder because model must find correct activity among 100 options**

### Semantic Groups (for Open-Set)

**The problem**: Different datasets use different names for the same activity (e.g., "walking_upstairs" vs "stairs_up" vs "ascending_stairs").

**Our solution**: ~25 manually defined groups that cluster synonyms:
- `running`: running, jogging, running_treadmill
- `ascending_stairs`: stairs_up, walking_upstairs, climbing_stairs
- `falling`: fall_forward, fall_backward, fall_sideways, ...

**Important caveat**: There's human discretion in deciding what's "close enough" to be a synonym. I could abuse this by making fewer, larger groups to inflate accuracy. To mitigate:
- Groups are fixed before running any experiments
- We also report **closed-set F1 without any grouping** (raw label match) for NLS-HAR comparison
- The closed-set metric has zero human discretion - it's just exact string matching

### Protocol Difference: Our Zero-Shot vs NLS-HAR's Zero-Shot

**NLS-HAR's approach**: Within each test dataset, they split activities into "seen" and "unseen" groups. They train on Capture-24, which contains some of the same activity types. At test time, they compute similarity against only the unseen classes. So their zero-shot is at the **activity level** — some activities within a dataset are seen, others unseen.

**Our approach**: We hold out **entire datasets**. The model never sees any IMU data from MotionSense, RealWorld, or MobiAct during training. At test time, we predict from the target dataset's labels. Our zero-shot is at the **dataset level** — the sensor data, subjects, and recording conditions are completely new.

**Both approaches** test generalization, but they're not identical. Our approach is arguably stricter (entirely new sensor conditions), but NLS-HAR's approach tests recognition of activity types the model has never been explicitly trained on. The comparison is still meaningful since both evaluate without fine-tuning on target data.

**Shared label overlap**: The model may have seen the same activity *names* in training (e.g., "walking" exists in both UCI-HAR and MotionSense), but the sensor data is completely different.

### Label Coverage in Zero-Shot Datasets

| Dataset | Labels with Training Equivalent | Truly Novel Labels |
|---------|--------------------------------|-------------------|
| MotionSense | 6/6 (100%) | 0 |
| RealWorld | 8/8 (100%) | 0 |
| MobiAct | 7/13 (54%) | 6 (falls, vehicle entry) |

---

## 5. Cross-Baseline Dataset Map

### All Datasets Used Across Baselines

| Dataset | Us | NLS-HAR | GOAT | LanHAR | CrossHAR | LIMU-BERT | IMU2CLIP |
|---------|:--:|:-------:|:----:|:------:|:--------:|:---------:|:--------:|
| UCI-HAR | train | — | — | eval | eval | eval | — |
| HHAR | train | eval | — | eval | eval | eval | — |
| MHEALTH | train | eval | — | — | — | — | — |
| PAMAP2 | train | eval | — | eval | — | — | — |
| WISDM | train | — | — | — | — | — | — |
| UniMiB-SHAR | train | — | — | — | eval | — | — |
| DSADS | train | — | — | — | — | — | — |
| HAPT | train | — | — | — | — | — | — |
| KU-HAR | train | — | — | — | — | — | — |
| VTT-ConIoT | train | — | — | — | — | — | — |
| RecGym | train | — | — | — | — | — | — |
| **MotionSense** | **test** | eval | — | eval | eval | eval | — |
| **RealWorld** | **test** | — | eval | — | — | — | — |
| **MobiAct** | **test** | eval | — | — | — | — | — |
| Shoaib | — | — | — | eval | eval | eval | — |
| Opportunity | — | — | eval | — | — | — | — |
| Realdisp | — | — | eval | — | — | — | — |
| PAMAP (original) | — | — | eval | — | — | — | — |
| Daphnet FoG | — | — | eval | — | — | — | — |
| Capture-24 | — | train | — | — | — | — | — |
| Myogym | — | eval | — | — | — | — | — |
| Ego4D | — | — | — | — | — | — | train |
| Aria | — | — | — | — | — | — | — |

### GOAT Details (IMWUT 2024)

Source: [GitHub](https://github.com/wdkhuans/GOAT)

| Dataset | Activities | Subjects | Devices | Rate |
|---------|-----------|----------|---------|------|
| RealWorld | 8 | 13 | 7 | 50Hz |
| Realdisp | 33 | 10 | 9 | 50Hz |
| Opportunity | 17 (ADL) | 4 | 5 | 30Hz |
| PAMAP | 12 | 8 | 3 | 100Hz |
| Daphnet FoG | 3 (gait) | 10 | 3 | 64Hz |

- Uses leave-one-dataset-out evaluation
- Reports macro F1 across 5 runs
- Zero-shot experiment on PAMAP with 4 held-out activities (nordic walking, vacuum cleaning, ironing, rope jumping)
- Best results: GOAT-CLIP on RealWorld (78.49%), Realdisp (81.39%), PAMAP (77.13%); GOAT-BERT on Opportunity (54.35%), Daphnet FoG (64.14%)

### Missing Datasets (Priority Order)

| Dataset | Priority | Reason | Used By |
|---------|----------|--------|---------|
| **Shoaib** | HIGH | 4 baselines use it for eval | LanHAR, CrossHAR, LIMU-BERT |
| **Opportunity** | HIGH | GOAT's primary benchmark | GOAT |
| Realdisp | MEDIUM | GOAT benchmark, 33 activities | GOAT |
| PAMAP (original) | LOW | We have PAMAP2 (superset) | GOAT |
| Daphnet FoG | LOW | Only 3 gait classes | GOAT |
| Capture-24 | MEDIUM | NLS-HAR trains on it (177 activities) | NLS-HAR |
| Myogym | LOW | Only NLS-HAR evals, scores 1.47% F1 | NLS-HAR |

### Comparison Strategy

**Standard practice**: Cite published numbers on overlapping test sets — don't retrain baselines. Train our model on the union of all datasets and report zero-shot results on test sets that overlap with baselines. Only retrain a baseline if (a) no published numbers exist on our test sets, or (b) a reviewer asks for it.

---

## 6. Ego4D / Aria IMU Feasibility

### Why Consider Ego4D?

IMU2CLIP (EMNLP 2023, Facebook Research) trains on Ego4D's head-mounted IMU data. No HAR baseline reports on Ego4D as a HAR benchmark, but it could demonstrate that our model generalizes beyond wrist/waist sensors to head-mounted IMU — a novelty contribution.

### Ego4D IMU Specs

- **Sensor**: BOSCH BMI260 (built into GoPro)
- **Axes**: 6-axis — 3-axis accelerometer + 3-axis gyroscope
- **Format**: CSV files, one per video clip, timestamps aligned to video
- **Mounting**: Head-mounted (GoPro on head rig)
- **Sampling rate**: Resampled to **200Hz** by IMU2CLIP (via `torchaudio.functional.resample`); raw GoPro rate unconfirmed
- **Scale**: ~3,670 hours of video (with IMU)

### Aria IMU Specs

- **Sensors**: Two separate IMUs (left and right)
- **Sampling rate**: 800Hz (left), 1000Hz (right)
- **Axes**: 3-axis accel + 3-axis gyro per IMU (12 channels total)

### Text Labels from Narrations

Ego4D includes timestamped narrations like "C picks up the cup", "C walks to the table". These become text labels for contrastive training. Narrations are aligned to video timestamps which are aligned to IMU timestamps.

### Why Our Architecture Handles Head-Mounted IMU

Our model uses **channel description semantics** — each sensor channel is described in natural language (e.g., "head accelerometer X"). The model conditions itself on these descriptions to interpret sensor data. This means:
- Head-mounted IMU data gets different channel descriptions than wrist/waist
- The model learns placement-specific patterns through language conditioning
- No architectural changes needed — just new channel descriptions

### Challenges

1. **Noisy narrations**: "C stirs the pot with their right hand" vs clean HAR labels like "cooking" — may need narration cleaning or mapping to activity categories
2. **Large scale**: Need to select relevant clips, can't use all 3,670 hours
3. **Resampling**: Must resample to match our pipeline's expected rate
4. **Preprocessing**: IMU2CLIP resamples to 200Hz, uses 5s windows (1000 samples), no normalization (raw values), fills NaN with 0, filters bad windows via `bad_imu_windows_*.json`
5. **Download/licensing**: Ego4D requires signing a data use agreement

### Verdict

**Nice-to-have, not essential for current paper.** No HAR baseline reports on Ego4D, so there's no comparison advantage. The main value is as a novelty demonstration that our channel description mechanism generalizes to head-mounted sensors. Could be a follow-up experiment or future work section.

---

## 7. Why We (Likely) Beat NLS-HAR

Their paper identifies two problems with natural language supervision for HAR:
1. **Sensor heterogeneity** - different devices, sampling rates, placements
2. **Lack of rich text descriptions** - simple labels like "walking" aren't descriptive enough

Our solutions:
1. **Multi-dataset training** - we train on 11 diverse datasets, they train on 1
2. **Label augmentation** - we generate synonyms and template variations
3. **Soft targets** - we don't treat "walking" and "jogging" as hard negatives

---

## Next Steps (TODO)

1. **Compute F1 scores** on zero-shot results for fair NLS-HAR comparison — **DONE** (see Section 3)
2. **Baseline code/checkpoints audit** — **DONE** (see Section 1)
3. **Add Shoaib dataset** as zero-shot test set (highest priority — enables comparison with LanHAR, CrossHAR, LIMU-BERT)
4. **Add Opportunity dataset** to training (enables stronger RealWorld comparison with GOAT)
5. **Retrain model** on expanded training set, run zero-shot on MotionSense, RealWorld, MobiAct, Shoaib
6. **Compile comparison table** — cite published baseline numbers on overlapping test sets
7. **Address comparability gaps**: LanHAR/CrossHAR only use 4 activity classes; our eval uses all activities. May need to also report 4-class subset results for fair comparison
8. **Run HHAR as zero-shot** (retrain without it) to compare with NLS-HAR's 31.05%
9. **Decide on Ego4D**: pursue as novelty experiment or defer to future work
