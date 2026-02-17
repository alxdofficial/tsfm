# Baseline Setup Guide

How to set up and run all baseline evaluations. This guide is for reviewers and
anyone reproducing the results.

## Overview

Our evaluation compares 5 models. The evaluation code lives in this repository
(`val_scripts/human_activity_recognition/evaluate_*.py`), but 3 baselines require
their original code repos for model definitions and pretrained checkpoints.

| Model | Source | What's Needed |
|-------|--------|--------------|
| **TSFM (ours)** | This repo | Trained checkpoint |
| **LanHAR** | GitHub clone | Model definitions (trains from scratch during eval) |
| **LiMU-BERT** | GitHub clone | Model definitions + pretrained checkpoint |
| **CrossHAR** | GitHub clone | Model definitions + pretrained checkpoint |
| **MOMENT** | HuggingFace | Auto-downloaded at runtime |

## Step 1: Clone Baseline Repositories

```bash
mkdir -p auxiliary_repos && cd auxiliary_repos

# CrossHAR (Dang et al., IMWUT 2024)
git clone https://github.com/kingdomrush2/CrossHAR.git
cd CrossHAR && git checkout 77b63d3 && cd ..

# LiMU-BERT (Xu et al., SenSys 2021)
git clone https://github.com/dapowan/LIMU-BERT-Public.git
cd LIMU-BERT-Public && git checkout decffee && cd ..

# LanHAR (Hao et al., IMWUT 2025)
git clone https://github.com/DASHLab/LanHAR.git
cd LanHAR && git checkout 1fe98fa && cd ..
```

MOMENT (Goswami et al., ICML 2024) is auto-downloaded from HuggingFace
(`AutonLab/MOMENT-1-large`) on first run — no manual setup needed.

## Step 2: Pretrained Checkpoints

### LiMU-BERT
The pretrained combined checkpoint should be at:
```
auxiliary_repos/LIMU-BERT-Public/saved/pretrain_base_recgym_20_120/pretrained_combined.pt
```
This checkpoint was trained on all 10 training datasets using the LiMU-BERT
pretraining script. Pre-extracted embeddings for all datasets should be at:
```
auxiliary_repos/LIMU-BERT-Public/embed/embed_pretrained_combined_{dataset}_20_120.npy
```

### CrossHAR
The pretrained encoder checkpoint should be at:
```
auxiliary_repos/CrossHAR/saved/pretrain_base_combined_train_20_120/model_masked_6_1.pt
```
This checkpoint was trained on all 10 training datasets using CrossHAR's
masked pretraining.

### LanHAR
No pretrained checkpoint needed — LanHAR trains from scratch during evaluation.
It downloads SciBERT (`allenai/scibert_scivocab_uncased`) from HuggingFace
automatically.

### TSFM (ours)
Checkpoint is at:
```
training_output/semantic_alignment/{run_id}/best.pt
```
The evaluation script auto-discovers the latest run.

## Step 3: Benchmark Data

All models evaluate on the same standardized data:
```
benchmark_data/processed/tsfm/{dataset}/data.npy    # (N, 120, 6) windows
benchmark_data/processed/tsfm/{dataset}/labels.npy  # (N,) integer labels
```

Data is standardized to 20Hz, 6-second windows, 6 IMU channels (acc_xyz + gyro_xyz).

To prepare data from raw sources:
```bash
python datascripts/setup_all_datasets.py
```

## Step 4: Run Evaluations

```bash
# All baselines sequentially
bash scripts/run_all_evaluations.sh

# Or individually
python val_scripts/human_activity_recognition/evaluate_tsfm.py
python val_scripts/human_activity_recognition/evaluate_limubert.py
python val_scripts/human_activity_recognition/evaluate_moment.py
python val_scripts/human_activity_recognition/evaluate_crosshar.py
python val_scripts/human_activity_recognition/evaluate_lanhar.py

# Generate combined comparison table
python scripts/generate_results_table.py
```

Results are written to `test_output/baseline_evaluation/{model}_evaluation.json`.

### Expected Runtime (RTX 4090)

| Model | Time | Notes |
|-------|------|-------|
| TSFM | ~10 min | Fixed 1.0s patch, 4 test datasets |
| MOMENT | ~20-40 min | SVM GridSearchCV is slow |
| LiMU-BERT | ~1-2 hrs | 100-epoch GRU training for zero-shot |
| CrossHAR | ~2-4 hrs | 100-epoch Transformer training for zero-shot |
| LanHAR | ~4-8 hrs | Full 60-epoch training from scratch |

## What Reviewers Should Examine

1. **Evaluation framework**: `docs/baselines/EVALUATION_PROTOCOL.md`
2. **Fairness justifications**: Same file, "Fairness Justifications" section
3. **Per-baseline adaptations**: `docs/baselines/BASELINE_IMPLEMENTATION_NOTES.md`
4. **Our model evaluation**: `val_scripts/human_activity_recognition/evaluate_tsfm.py`
5. **A baseline evaluation** (for comparison): `val_scripts/human_activity_recognition/evaluate_moment.py`
6. **Results**: `docs/baselines/RESULTS.md`

## Repository Size

The git repository contains only source code and documentation (~5 MB). Large
artifacts are excluded via `.gitignore`:

| Directory | Size | Contents | In Git? |
|-----------|------|----------|---------|
| `auxiliary_repos/` | ~7.6 GB | Baseline code + checkpoints | No |
| `benchmark_data/` | ~6.3 GB | Preprocessed evaluation data | No |
| `training_output/` | ~14 GB | TSFM checkpoints and logs | No |
| `test_output/` | ~930 MB | Evaluation results (JSON, cached classifiers) | No |
| `data/` | ~32 GB | Raw dataset downloads | No |

All code needed to understand and review the evaluation is tracked in git.
The excluded directories contain data and model weights that must be
generated/downloaded following the steps above.
