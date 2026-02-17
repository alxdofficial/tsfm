# Benchmark Data for TSFM vs Baselines

Unified benchmark comparing TSFM against LIMU-BERT, LanHAR, and CrossHAR on 14 HAR datasets.

## Quick Start

```bash
# 1. Export raw per-subject CSVs from TSFM session data
python benchmark_data/scripts/export_raw.py

# 2. Generate LIMU-BERT .npy files
python benchmark_data/scripts/preprocess_limubert.py

# 3. Set up TSFM symlinks
python benchmark_data/scripts/preprocess_tsfm.py

# To zip for sharing (use --copy instead of symlinks):
python benchmark_data/scripts/preprocess_tsfm.py --copy
zip -r benchmark_data.zip benchmark_data/
```

## Folder Structure

```
benchmark_data/
├── README.md                   # This file
├── dataset_config.json         # Dataset metadata and train/test splits
├── raw/                        # Standardized per-subject CSVs
│   ├── uci_har/
│   │   ├── metadata.json
│   │   ├── subject_01.csv
│   │   └── ...
│   └── ... (14 datasets)
├── processed/
│   ├── tsfm/                   # Symlinks to data/{dataset}
│   ├── limubert/               # LIMU-BERT .npy format
│   │   ├── uci_har/
│   │   │   ├── data_20_120.npy
│   │   │   ├── label_20_120.npy
│   │   │   └── mapping.json
│   │   └── ...
│   ├── lanhar/                 # For collaborator to fill
│   └── crosshar/               # For collaborator to fill
└── scripts/
    ├── export_raw.py
    ├── preprocess_limubert.py
    └── preprocess_tsfm.py
```

## Datasets

### Training/Benchmark (10 datasets)

| Dataset | Sessions | Activities | Hz | Channels | Placement |
|---------|----------|------------|-----|----------|-----------|
| UCI-HAR | 10,299 | 6 | 50 | 9 (acc+gyro+total_acc) | Waist |
| HHAR | 15,000* | 6 | 50 | 6 (acc+gyro) | Pocket |
| PAMAP2 | 4,342 | 12 | 100 | 52 (multi-IMU) | Multi-body |
| WISDM | 15,000* | 18 | 20 | 12 (phone+watch) | Pocket+Wrist |
| DSADS | 9,120 | 19 | 25 | 9 (acc+gyro+mag) | Wrist |
| KU-HAR | 17,374 | 17 | 100 | 6 (acc+gyro) | Wrist |
| UniMiB-SHAR | 11,771 | 17 | 50 | 3 (acc only) | Pocket |
| HAPT | 2,546 | 12 | 50 | 6 (acc+gyro) | Waist |
| MHEALTH | 2,029 | 12 | 50 | 23 (multi+ECG) | Multi-body |
| RecGym | 7,150 | 11 | 20 | 6 (acc+gyro) | Chest |

*Subsampled from original count via stratified sampling.

### Zero-Shot (4 datasets, TSFM only)

| Dataset | Sessions | Activities | Difficulty |
|---------|----------|------------|------------|
| MotionSense | 7,989 | 6 | Easy |
| RealWorld | 16,830 | 8 | Medium |
| MobiAct | 3,646 | 13 | Hard |
| VTT-ConIoT | 207 | 16 | Hard |

## Raw CSV Format

Each `subject_XX.csv` contains continuous sensor data per subject:

```csv
timestamp_sec,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,activity
0.0000,0.120000,-0.030000,0.150000,0.001000,-0.002000,0.003000,walking
0.0200,0.150000,-0.020000,0.180000,0.002000,-0.001000,0.002000,walking
```

**Columns:**
- `timestamp_sec`: Seconds from start of recording (continuous)
- `acc_x`, `acc_y`, `acc_z`: Accelerometer (core, always present)
- `gyro_x`, `gyro_y`, `gyro_z`: Gyroscope (core, present except UniMiB-SHAR)
- Additional dataset-specific channels as extra columns (e.g., `mag_x`, `total_acc_x`)
- `activity`: Activity label string

**Notes:**
- Sessions are concatenated per subject, sorted by session name
- Timestamps are rebuilt as continuous (no gaps between sessions)
- Channel names are standardized: `acc_x/y/z` and `gyro_x/y/z` are always the core channels
- Original dataset-specific names are mapped to these standard names (see `dataset_config.json` for the mapping)

## LIMU-BERT Format

Each dataset produces:
- `data_20_120.npy`: Shape `(N, 120, 6)` — float32
  - 6 channels: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
  - Resampled to 20Hz, 120-sample windows (6 seconds)
  - Non-overlapping windows (stride = 120 samples)
  - UniMiB-SHAR: gyro channels are zero-padded
- `label_20_120.npy`: Shape `(N, 120, 2)` — int32
  - Per-timestep labels (same label repeated across window for uniform-activity windows)
  - Column 0: activity index (see `mapping.json`)
  - Column 1: subject index (see `mapping.json`)
- `mapping.json`: Activity and subject index mappings

## Train/Test Splits

**Training datasets** (10 datasets): Used for TSFM pre-training. Baselines use these for their own training procedures.

**Zero-shot test datasets** (4 datasets): MotionSense, RealWorld, MobiAct, VTT-ConIoT. Used for evaluation only — no training data from these datasets is seen during pre-training.

For supervised/linear-probe evaluations on zero-shot datasets, each evaluation script applies random window-level splits internally (not subject-based).

## Instructions for LanHAR/CrossHAR Preprocessing

Use the CSVs in `raw/` as your starting point. For each dataset:

1. Read all `subject_*.csv` files from `raw/{dataset}/`
2. Use `metadata.json` for sampling rate and channel info
3. Apply your model's required preprocessing (windowing, normalization, etc.)
4. Save output to `processed/lanhar/{dataset}/` or `processed/crosshar/{dataset}/`
5. See evaluation scripts for split logic (random window-level splits)

The `raw/` CSVs give you the **same data** used for TSFM and LIMU-BERT, just in a universal CSV format.
