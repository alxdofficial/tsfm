# Paper Update Instructions - Table 3 and Validation Results

This document provides the specific updates needed for Table 3 (Per-Dataset Validation Accuracy) based on the latest evaluation run on 2026-01-27.

---

## UPDATE TABLE 3: Per-Dataset Validation Accuracy

**Current Table 3 in paper shows only 6 datasets. Replace with all 10 training datasets.**

> **Note:** VTT-ConIoT was moved from training to zero-shot evaluation to prevent data leakage. The numbers below are from the 11-dataset checkpoint (2026-01-27) and will be updated after retraining with 10 datasets.

### New Table 3: Per-Dataset Validation Accuracy (10-Dataset Model)

| Dataset | Val Group-Acc (%) |
|---------|-------------------|
| UCI-HAR | 74.77 |
| HHAR | 77.89 |
| UniMiB-SHAR | 83.44 |
| MHEALTH | 88.06 |
| PAMAP2 | 80.38 |
| WISDM | 60.69 |
| DSADS | 93.55 |
| HAPT | 73.20 |
| KU-HAR | 89.69 |
| RecGym | 98.31 |
| **Overall** | **TBD (retrain needed)** |

**Caption:** "Per-dataset validation accuracy (group-level) for the 10-dataset model. Accuracy varies by dataset complexity, label vocabulary overlap with other datasets, and the number of unique activities per dataset."

---

## UPDATE TABLE 2: Overall Validation Performance

**Replace current Table 2 with:**

| Metric | Value |
|--------|-------|
| Group Accuracy | 81.56% |
| MRR | 88.22% |
| Recall@1 | 73.17% |
| Recall@5 | 94.42% |
| Positive Similarity | 0.874 |
| Negative Similarity | 0.422 |
| Similarity Gap | 0.452 |

---

## CONFIRM TABLE 4: Zero-Shot Performance (Already Correct)

The paper should have these values for Table 4:

| Dataset | Group-Acc (%) | MRR (%) | Patch (s) |
|---------|---------------|---------|-----------|
| MotionSense | TBD | TBD | 1.5 |
| RealWorld | TBD | TBD | 1.5 |
| MobiAct | TBD | TBD | 1.25 |
| VTT-ConIoT | TBD | TBD | 2.0 |
| **Combined** | **TBD** | — | — |

> **Note:** VTT-ConIoT is now a zero-shot dataset (moved from training to prevent data leakage). Previous zero-shot numbers were from the 11-dataset model; retrain with 10 datasets needed.

---

## NEW TABLE: Dataset Scaling Ablation

**Add this new table showing the impact of training on more datasets:**

| Training Config | Training Val Acc (%) | Zero-Shot Acc (%) |
|-----------------|---------------------|-------------------|
| 6 datasets | 57.60 | 29.75 |
| 10 datasets | TBD (retrain needed) | TBD |
| **Improvement** | **TBD** | **TBD** |

> **Note:** Previous 11-dataset results included VTT-ConIoT in training. With VTT-ConIoT moved to zero-shot (4 zero-shot datasets total), scaling ablation numbers need re-evaluation after retraining.

### Per-Dataset Zero-Shot Improvement (from 6-dataset baseline):

| Dataset | 6 Datasets (%) | 10 Datasets (%) | Δ |
|---------|----------------|-----------------|-----|
| MotionSense | 34.64 | TBD | — |
| RealWorld | 27.40 | TBD | — |
| MobiAct | 25.46 | TBD | — |
| VTT-ConIoT | — | TBD | — |

---

## NEW TABLE: Expected vs Novel Activity Performance

**Add this table to Section 6.7 or as a new subsection:**

| Category | Samples | Accuracy (%) |
|----------|---------|--------------|
| Expected activities | 3039 (93.7%) | 51.07 |
| Novel activities | 205 (6.3%) | 25.37 |
| **Overall** | **3244** | **49.41** |

**Expected activities** are those with semantic overlap to training (walking, running, sitting, standing, stairs, jumping, lying).

**Novel activities** are activity categories NOT represented in training (falling, vehicle entry - only present in MobiAct).

---

## SECTION 6.8: Embedding Separability (Update Values)

Replace the embedding separability metrics in Section 6.8 with:

> On validation, the model achieves mean similarity **0.874** for matched IMU-text pairs and **0.422** for different-group pairs (gap **0.452**), indicating that the learned space clusters semantically aligned IMU and text representations while separating non-matching activities.

---

## NOTES ON DATASET PERFORMANCE VARIATION

The per-dataset validation accuracy varies significantly:

**High performers (>85%):**
- RecGym: 98.31% - Small vocabulary, distinct gym exercises
- DSADS: 93.55% - Well-separated daily/sports activities
- KU-HAR: 89.69% - Clear smartphone activity patterns
- MHEALTH: 88.06% - Multi-sensor body positions

**Moderate performers (70-85%):**
- UniMiB-SHAR: 83.44%
- PAMAP2: 80.38%
- HHAR: 77.89%
- UCI-HAR: 74.77%
- HAPT: 73.20%

**Lower performers (<70%):**
- WISDM: 60.69% - 18 activities with fine-grained distinctions

> **Note:** VTT-ConIoT (previously 51.35% validation) has been moved to zero-shot evaluation.

Consider adding a brief discussion of why some datasets are harder than others (label vocabulary overlap, activity granularity, sensor placement differences).

---

## CHECKLIST

- [ ] Replace Table 3 with 10-dataset validation accuracy (after retraining)
- [ ] Update Table 2 overall metrics
- [ ] Verify Table 4 zero-shot numbers match
- [ ] Add dataset scaling ablation table
- [ ] Add expected vs novel activity breakdown
- [ ] Update Section 6.8 embedding separability values
- [ ] Consider adding discussion of per-dataset performance variation
