# Dataset Pipeline Documentation

This directory contains scripts to download and convert 20 time series activity recognition
datasets into a standardized session format for training and evaluation.

## Quick Start

```bash
# Process all datasets (downloads auto-downloadable ones, converts all)
python datascripts/setup_all_ts_datasets.py

# Process a single dataset
python datascripts/setup_all_ts_datasets.py uci_har

# Verify conversions
python datascripts/verify_conversions.py
```

## All 20 Datasets

### Training Datasets (10)

| Dataset | Activities | Hz | Channels | Auto-Download? | Source |
|---------|:---:|:---:|:---:|:---:|-------------|
| **uci_har** | 6 | 50 | 9 | Yes | [UCI ML Repository](https://archive.ics.uci.edu/dataset/240) |
| **hhar** | 6 | 50 | 6 | Yes | [UCI ML Repository](https://archive.ics.uci.edu/dataset/344) |
| **pamap2** | 12 | 100 | 52 | Yes | [UCI ML Repository](https://archive.ics.uci.edu/dataset/231) |
| **wisdm** | 18 | 20 | 12 | Yes | [UCI ML Repository](https://archive.ics.uci.edu/dataset/507) |
| **mhealth** | 12 | 50 | 23 | Yes | [UCI ML Repository](https://archive.ics.uci.edu/dataset/319) |
| **dsads** | 19 | 25 | 9 | Yes | [UCI ML Repository](https://archive.ics.uci.edu/dataset/256) |
| **kuhar** | 17 | 100 | 6 | Manual | [Kaggle](https://www.kaggle.com/datasets/niloy333/ku-har-dataset) |
| **unimib_shar** | 17 | 50 | 3 | Manual | [UniMiB SHAR Project](http://www.sal.disco.unimib.it/technologies/unimib-shar/) |
| **hapt** | 12 | 50 | 6 | Manual | [UCI ML Repository](https://archive.ics.uci.edu/dataset/341) |
| **recgym** | 11 | 20 | 6 | Manual | Contact dataset authors |

### Zero-Shot Test Datasets (10)

| Dataset | Activities | Hz | Channels | Auto-Download? | Source |
|---------|:---:|:---:|:---:|:---:|-------------|
| **motionsense** | 6 | 50 | 6 | Yes | [GitHub](https://github.com/mmalekzadeh/motion-sense) |
| **realworld** | 8 | 50 | 6 | Manual | [Sensor.Informatik](http://sensor.informatik.uni-mannheim.de/#dataset_realworld) |
| **mobiact** | 13 | 50 | 6 | Manual | [BioSEC Group](https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/) |
| **vtt_coniot** | 16 | 50 | 6 | Manual | Contact VTT Finland |
| **shoaib** | 7 | 50 | 6 | Manual | [UTwente Research](https://research.utwente.nl/en/datasets/activity-recognition-data) |
| **opportunity** | 4 | 30 | 113 | Manual | [UCI ML Repository](https://archive.ics.uci.edu/dataset/226) |
| **harth** | 12 | 50 | 6 | Manual | [Machine Learning Repository](https://archive.ics.uci.edu/dataset/779) |
| **realdisp** | 33 | 50 | 6 | Manual | [UCI ML Repository](https://archive.ics.uci.edu/dataset/305) |
| **daphnet_fog** | 2 | 64 | 9 | Manual | [UCI ML Repository](https://archive.ics.uci.edu/dataset/245) |
| **usc_had** | 12 | 100 | 6 | Manual | [USC SIPI](https://sipi.usc.edu/had/) |

**Manual download**: Download the raw dataset to `datascripts/{dataset}/raw/` or the path
specified in `setup_all_ts_datasets.py`, then run the pipeline. The script will tell you
which datasets need manual download and where to get them.

## Standardized Output Format

All datasets are converted to a uniform session structure:

```
data/{dataset_name}/
    manifest.json       # Metadata: channels, sampling rate, description
    labels.json         # Session → activity label mapping
    sessions/
      session_001/
        data.parquet    # Sensor data with timestamp_sec column
      session_002/
        data.parquet
      ...
```

### `manifest.json`

```json
{
  "dataset_name": "Dataset Name",
  "description": "2-3 sentence description of dataset, sensors, subjects, activities",
  "channels": [
    {
      "name": "channel_name",
      "description": "How this channel was obtained and what it measures",
      "sampling_rate_hz": 50.0
    }
  ]
}
```

### `data.parquet`

Pandas DataFrame with:
- **First column**: `timestamp_sec` (real-world time in seconds from session start)
- **Other columns**: Sensor channels (e.g., `acc_x`, `gyro_y`)
- All reasoning uses real-world time (seconds), never timesteps

See [DATA_FORMAT.md](../DATA_FORMAT.md) for the full specification.

## Pipeline Steps

The master script `setup_all_ts_datasets.py` runs these steps per dataset:

1. **Download** (if auto-downloadable): `datascripts/shared/download_all_datasets.py`
2. **Convert**: `datascripts/{dataset}/convert.py` — transforms raw data to session format
3. **Validate**: Checks timestamp monotonicity, sampling rate, NaN values, etc.

After conversion, the benchmark preprocessing scripts generate the evaluation data:

4. **Export CSVs**: `benchmark_data/scripts/export_raw.py` — session data → per-subject CSVs
5. **Generate 20Hz .npy**: `benchmark_data/scripts/preprocess_limubert.py` — for all baselines
6. **Generate native .npy**: `benchmark_data/scripts/preprocess_tsfm_eval.py` — for TSFM evaluation

## Dataset-Specific Notes

- **UCI HAR**: Pre-segmented into 2.56s windows; each window is one session
- **PAMAP2**: Continuous recordings segmented by activity; ~40 channels from 3 IMU locations
- **Opportunity**: 113 raw channels; only 6 core channels (back-mounted acc+gyro) used for evaluation
- **HARTH**: Back + thigh accelerometers only (no gyroscope); gyro channels zero-padded
- **UniMiB SHAR**: Accelerometer only (3 channels); gyro channels zero-padded
- **ActionSense**: EMG-only at 200Hz (bilateral Myo armbands); not used in current evaluation

## Verify Conversions

```bash
# Verify all datasets
python datascripts/verify_conversions.py

# Verify specific dataset
python datascripts/verify_conversions.py uci_har
```

Checks: timestamp column exists and starts at 0, timestamps are monotonic,
sampling rates match manifest, no unexpected NaN values, reasonable session durations.
