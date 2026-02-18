# Dataset Pipeline Documentation

This directory contains scripts to download and convert time series activity recognition datasets into a standardized format.

## Purpose

This standardized format supports **both phases** of the tool-use-om architecture:

- **Phase 1 (Current):** EDA tool-use reasoning - LLM learns to explore and preprocess data
- **Phase 2 (Future):** Model selection - LLM learns to choose tokenizers and task heads

The format is designed to be:
- **Dataset-agnostic:** Works with any time series dataset
- **Token-efficient:** Minimal manifests to reduce LLM context usage
- **Real-world time:** All temporal reasoning uses seconds, not timesteps
- **Tool-friendly:** Easy for both symbolic tools (Phase 1) and tokenizers (Phase 2) to consume

📖 **See [ARCHITECTURE.md](../docs/ARCHITECTURE.md) for full system design**

## Standardized Format

All datasets are converted to a uniform structure for consistent processing:

```
data/
  {dataset_name}/
    manifest.json       # Minimal metadata (human/LLM-written)
    labels.json         # Session labels (activity annotations)
    sessions/
      session_001/
        data.parquet    # All channels as pandas DataFrame with timestamp
      session_002/
        data.parquet
      ...
```

### `manifest.json` Format

Minimal, token-efficient metadata:

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

**Key principles:**
- Brief descriptions to minimize token usage
- No file paths, subject lists, or episode enumerations
- Channel metadata includes name, description, and sampling rate
- Everything else discovered at runtime from session folders

### `labels.json` Format

Maps session IDs to activity labels:

```json
{
  "session_001": ["walking"],
  "session_002": ["running", "outdoor"],
  ...
}
```

- Each session can have one or more labels
- Labels are always lists (even for single-label datasets)

### `data.parquet` Format

Pandas DataFrame with:
- **First column**: `timestamp_sec` (real-world time in seconds from session start)
- **Other columns**: Sensor channels (e.g., `accel_x`, `gyro_y`, `emg_0`)
- All reasoning uses real-world time (seconds), never timesteps

## Available Datasets

| Dataset | Subjects | Activities | Channels | Rate | Description |
|---------|----------|------------|----------|------|-------------|
| **UCI HAR** | 30 | 6 | 9 | 50 Hz | Smartphone accel + gyro at waist |
| **PAMAP2** | 9 | 18 | ~40 | 100 Hz | 3 IMUs (hand, chest, ankle) + HR |
| **MHEALTH** | 10 | 12 | 23 | 50 Hz | 3 IMUs (chest, ankle, wrist) + ECG |
| **WISDM** | 51 | 18 | 12 | 20 Hz | Phone + watch accel + gyro |
| **ActionSense** | 9 | 23 | 16 | 200 Hz | Kitchen activities: bilateral EMG only |

## Usage

### Quick Start: Process All Datasets

```bash
# Install dependencies
pip install pandas pyarrow numpy matplotlib seaborn

# Download and convert all datasets
python3 datascripts/setup_all_ts_datasets.py
```

### Process Single Dataset

```bash
# Download and convert one dataset
python3 datascripts/setup_all_ts_datasets.py uci_har
```

### Verify Conversions

After conversion, verify data quality:

```bash
# Verify all datasets
python3 datascripts/verify_conversions.py

# Verify specific dataset
python3 datascripts/verify_conversions.py wisdm
```

The verification script checks:
- ✅ Timestamp column exists and starts at 0
- ✅ Timestamps are monotonically increasing
- ✅ Sampling rates match manifest specifications
- ✅ No unexpected NaN values in data
- ✅ Session durations are reasonable
- ✅ Data types are correct

### Manual Pipeline Steps

```bash
# 1. Download raw data
python3 datascripts/shared/download_all_datasets.py uci_har

# 2. Convert to standardized format
python3 datascripts/uci_har/convert.py

# 3. Verify output
python3 datascripts/verify_conversions.py uci_har
```

## Scripts

### Download Scripts

#### `shared/download_all_datasets.py`

Downloads raw datasets from UCI ML Repository and other sources.

```bash
# Download all
python datascripts/shared/download_all_datasets.py

# Download specific
python datascripts/shared/download_all_datasets.py pamap2
```

**Supported datasets:**
- `uci_har` - UCI HAR from archive.ics.uci.edu
- `pamap2` - PAMAP2 from UCI
- `mhealth` - MHEALTH from UCI
- `wisdm` - WISDM from UCI

**Note:** ActionSense requires manual download (see dataset documentation).

### Conversion Scripts

Each dataset has a dedicated converter at `datascripts/{dataset}/convert.py`:

- `uci_har/convert.py` - Converts UCI HAR windowed inertial signals
- `pamap2/convert.py` - Segments PAMAP2 continuous recordings by activity
- `mhealth/convert.py` - Segments MHEALTH log files by activity
- `wisdm/convert.py` - Organizes WISDM multi-device data into sessions
- `actionsense/convert.py` - Converts ActionSense multi-modal CSV files

All converters:
1. Load raw data
2. Segment into sessions (by activity, subject, or window)
3. Save as parquet with standardized column names
4. Generate manifest.json and labels.json

### Master Pipeline

#### `setup_all_ts_datasets.py`

Orchestrates complete pipeline: download → convert → validate.

```bash
# Process all datasets
python datascripts/setup_all_ts_datasets.py

# Process specific dataset
python datascripts/setup_all_ts_datasets.py mhealth
```

## Dataset-Specific Notes

### UCI HAR
- Pre-segmented into 2.56-second windows (128 samples @ 50 Hz)
- Each window is one session
- Includes train/test split (preserved in session IDs)
- 9 channels: body_acc (3), total_acc (3), body_gyro (3)

### PAMAP2
- Continuous recordings segmented by activity changes
- Filters out transient activities (activity_id=0)
- Minimum segment duration: 5 seconds
- 3 IMUs × 13 channels + heart rate = ~40 channels
- Orientation columns marked invalid (skipped)

### MHEALTH
- Log files with 24 columns per subject
- Segmented by activity changes
- Minimum segment duration: 3 seconds
- 23 sensor channels + 1 activity label
- 3 body locations: chest, left ankle, right wrist

### WISDM
- Separate files per device (phone/watch) and sensor (accel/gyro)
- Sessions organized by subject + activity + device + sensor
- Currently stores each device-sensor combo separately
- Future: Multi-modal alignment across devices

### ActionSense
- **EMG-only dataset** at 200 Hz (bilateral Myo armbands)
- 16 channels: 8 left forearm + 8 right forearm
- Other modalities (joints, gaze) excluded
- Target rate chosen based on EMG sensor hardware spec (Myo armband documented rate)

## Sampling Rate Policy

**Preprocessing Approach**: All variable-rate resampling is handled during dataset conversion (one-time preprocessing). Runtime code assumes all channels within a dataset have uniform sampling rates.

### Design Decisions

1. **Per-Dataset Target Rate**: Each dataset has a human-chosen target sampling rate specified in conversion scripts
2. **Uniform Within Dataset**: All channels in a dataset are resampled to the same target rate during conversion
3. **Hardware-Based Targets**: Target rates chosen based on sensor hardware specifications when available
4. **Linear Interpolation**: Resampling uses `np.interp()` for temporal alignment

### Dataset Target Rates

| Dataset | Target Rate | Rationale |
|---------|-------------|-----------|
| UCI HAR | 50 Hz | Native rate (pre-processed) |
| PAMAP2 | 100 Hz | Native rate (synchronized) |
| MHEALTH | 50 Hz | Native rate (synchronized) |
| WISDM | 20 Hz | Native rate (phone sensors) |
| ActionSense | 200 Hz | Myo armband hardware spec |

### Multi-Rate Datasets

For datasets with channels at different native rates (e.g., ActionSense):
- **Option 1 (Current)**: Select single modality per dataset (e.g., EMG only)
- **Option 2 (Future)**: Resample all modalities to common target rate with interpolation
- **Option 3 (Future)**: Support per-channel rates with NaN-sparse storage or metadata-driven resampling

Current implementation uses **Option 1** for simplicity and to avoid interpolation artifacts.

## Output Structure

After running the pipeline, you'll have:

```
data/
  uci_har/
    manifest.json
    labels.json
    sessions/session_001/data.parquet
    sessions/session_002/data.parquet
    ...
  pamap2/
    manifest.json
    labels.json
    sessions/subject1_seg001/data.parquet
    ...
  mhealth/
    ...
  wisdm/
    ...
  actionsense/
    ...
```

## Design Principles

1. **Dataset-Agnostic**: Uniform format works with any time series dataset
2. **Real-World Time**: All temporal reasoning uses seconds, not timesteps
3. **Token-Efficient**: Minimal manifests to reduce LLM context usage
4. **Session-Based**: Data organized as independent sessions with labels
5. **Parquet Storage**: Efficient binary format for DataFrames
6. **Runtime Discovery**: Session lists and structure discovered at load time

## Future Enhancements

- [ ] Multi-modal alignment with interpolation
- [ ] Automatic manifest generation from data
- [ ] Data quality validation
- [ ] Train/val/test split utilities
- [ ] Session merging and windowing utilities
- [ ] Additional datasets (OPPORTUNITY, REALDISP, etc.)

## Requirements

```bash
pip install pandas pyarrow requests
```

Optional for development:
```bash
pip install jupyter matplotlib seaborn  # For data exploration
```

## License

Datasets retain their original licenses (typically CC BY 4.0). See individual dataset documentation for details.

## References

- UCI HAR: https://archive.ics.uci.edu/dataset/240
- PAMAP2: https://archive.ics.uci.edu/dataset/231
- MHEALTH: https://archive.ics.uci.edu/dataset/319
- WISDM: https://archive.ics.uci.edu/dataset/507
- ActionSense: https://action-sense.csail.mit.edu
