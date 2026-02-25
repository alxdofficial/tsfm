# Baseline Setup Guide

How to set up and run all baseline evaluations. This guide is for reviewers and
anyone reproducing the results.

## Overview

Our evaluation compares 6 models on 7 test datasets using a unified 4-metric framework.
The evaluation code lives in this repository (`val_scripts/human_activity_recognition/evaluate_*.py`),
but 3 baselines require their original code repos for model definitions and pretrained checkpoints.

| Model | Source | What's Needed |
|-------|--------|--------------|
| **TSFM (ours)** | This repo | Trained checkpoint |
| **LanHAR** | GitHub clone | Model definitions (trains from scratch during eval) |
| **LiMU-BERT** | GitHub clone | Model definitions + pretrained checkpoint |
| **CrossHAR** | GitHub clone | Model definitions + pretrained checkpoint |
| **MOMENT** | HuggingFace | Auto-downloaded at runtime |
| **LLaSA** | HuggingFace | Auto-downloaded at runtime (~16GB VRAM required) |

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

# LLaSA (optional — requires ~16GB VRAM for 7B model)
git clone https://github.com/BASH-Lab/LLaSA.git
cd LLaSA && cd ..
```

MOMENT (Goswami et al., ICML 2024) is auto-downloaded from HuggingFace
(`AutonLab/MOMENT-1-large`) on first run — no manual setup needed.

LLaSA (Li et al., 2024) downloads its model (`BASH-Lab/LLaSA-7B`) from
HuggingFace at runtime.

## Step 2: Apply Patches to Baseline Repos

Some baseline repos need minor modifications for our evaluation framework:

```bash
# From project root:
# Apply LiMU-BERT patches (adds combined pretraining support)
cp docs/baselines/auxiliary_patches/limubert/* auxiliary_repos/LIMU-BERT-Public/

# Apply CrossHAR patches (adds combined pretraining support)
cp docs/baselines/auxiliary_patches/crosshar/* auxiliary_repos/CrossHAR/
```

See `docs/baselines/auxiliary_patches/` for the exact files and their purpose.

## Step 3: Pretrained Checkpoints

### LiMU-BERT
The pretrained combined checkpoint should be at:
```
auxiliary_repos/LIMU-BERT-Public/saved/pretrain_base_combined_train_20_120/pretrained_combined.pt
```
This checkpoint was trained on all 10 training datasets using the LiMU-BERT
pretraining script with the combined training configuration.

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
Checkpoint is auto-discovered at:
```
training_output/semantic_alignment/{latest_run}/best.pt
```
You can also set the `TSFM_CHECKPOINT` environment variable to override:
```bash
export TSFM_CHECKPOINT="training_output/semantic_alignment/20260217_113136/best.pt"
```

## Step 4: Benchmark Data

All models evaluate on preprocessed `.npy` windows. These are generated from the
standardized session data in `data/`.

**Baselines** (LiMU-BERT, CrossHAR, MOMENT, LanHAR, LLaSA) evaluate on 20Hz resampled data:
```
benchmark_data/processed/limubert/{dataset}/data_20_120.npy   # (N, 120, 6) windows at 20Hz
benchmark_data/processed/limubert/{dataset}/label_20_120.npy  # (N, 120, 2) labels
```

**TSFM** evaluates on native-rate data (50Hz for most datasets, 30Hz for Opportunity):
```
benchmark_data/processed/tsfm_eval/{dataset}/data_native.npy   # (N, W, 6) windows at native Hz
benchmark_data/processed/tsfm_eval/{dataset}/label_native.npy  # (N, W, 2) labels
benchmark_data/processed/tsfm_eval/{dataset}/metadata.json     # sampling rate, window size
```

To prepare data from raw sources:
```bash
# 1. Download and convert all datasets to standardized session format
python datascripts/setup_all_ts_datasets.py

# 2. Export raw per-subject CSVs for benchmark preprocessing
python benchmark_data/scripts/export_raw.py

# 3. Generate 20Hz .npy files for baselines
python benchmark_data/scripts/preprocess_limubert.py

# 4. Generate native-rate .npy files for TSFM evaluation
python benchmark_data/scripts/preprocess_tsfm_eval.py
```

## Step 5: Run Evaluations

```bash
# All baselines sequentially (including LLaSA)
bash scripts/run_all_evaluations.sh

# Or individually
python val_scripts/human_activity_recognition/evaluate_tsfm.py
python val_scripts/human_activity_recognition/evaluate_limubert.py
python val_scripts/human_activity_recognition/evaluate_moment.py
python val_scripts/human_activity_recognition/evaluate_crosshar.py
python val_scripts/human_activity_recognition/evaluate_lanhar.py
python val_scripts/human_activity_recognition/evaluate_llasa.py  # optional, needs ~16GB VRAM
```

Results are written to `test_output/baseline_evaluation/{model}_evaluation.json`.

### Expected Runtime (RTX 4090)

| Model | Time | Notes |
|-------|------|-------|
| TSFM | ~15 min | 7 test datasets, native rate |
| MOMENT | ~30-60 min | SVM GridSearchCV is slow |
| LiMU-BERT | ~2-4 hrs | 100-epoch GRU training for zero-shot |
| CrossHAR | ~3-6 hrs | 100-epoch Transformer training for zero-shot |
| LanHAR | ~6-12 hrs | Full 60-epoch training from scratch |
| LLaSA | ~2-4 hrs | 7B LLM inference, 100 samples/class |

## What Reviewers Should Examine

1. **Evaluation framework**: `docs/baselines/EVALUATION_PROTOCOL.md`
2. **Fairness justifications**: `docs/baselines/RESULTS.md` (Fairness Notes + MOMENT Advantages sections)
3. **Per-baseline adaptations**: `docs/baselines/RESULTS.md` (Adaptations table)
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
