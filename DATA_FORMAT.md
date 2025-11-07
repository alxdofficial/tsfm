# Data Format Specification

**Last Updated:** 2025-01-07
**Status:** ✅ Standardized across all datasets

---

## Overview

All time series datasets in this repository follow a **standardized format** based on three core concepts:

1. **Bag of Sessions**: Each dataset is a collection of independent sessions
2. **Pandas DataFrames**: Each session is stored as a parquet file containing a DataFrame
3. **Real-World Time**: All temporal operations use **seconds**, never timesteps

This format is designed to be:
- **Dataset-agnostic**: Works with any time series dataset (HAR, medical, industrial, etc.)
- **Tool-friendly**: Easy for both symbolic tools (Phase 1) and tokenizers (Phase 2) to consume
- **Token-efficient**: Minimal metadata to reduce LLM context usage
- **Human-readable**: Clear structure with descriptive manifests

---

## Directory Structure

Every dataset follows this exact structure:

```
data/{dataset_name}/
├── manifest.json          # Dataset metadata + channel descriptions + sampling rates
├── labels.json           # Session labels (activity/task annotations)
└── sessions/             # Directory containing all sessions
    ├── session_001/
    │   └── data.parquet  # Pandas DataFrame: timestamp_sec + channels
    ├── session_002/
    │   └── data.parquet
    └── ...
```

---

## File Specifications

### 1. `manifest.json`

**Purpose:** Provide metadata about the dataset and all channels, including textual descriptions and sampling rates.

**Format:**
```json
{
  "dataset_name": "Dataset Name",
  "description": "2-3 sentence description of the dataset, sensors, subjects, and activities",
  "channels": [
    {
      "name": "channel_name",
      "description": "How this channel was obtained and what it measures (e.g., 'X-axis acceleration from waist-mounted smartphone')",
      "sampling_rate_hz": 50.0
    },
    {
      "name": "another_channel",
      "description": "Description of another channel",
      "sampling_rate_hz": 100.0
    }
  ]
}
```

**Key Points:**
- **Textual descriptions**: Each channel must have a human-readable description explaining what it measures
- **Sampling rates**: Each channel specifies its sampling rate in Hz (can vary across channels)
- **Minimal metadata**: No file paths, subject lists, or episode enumerations (discovered at runtime)
- **Token-efficient**: Brief descriptions to minimize LLM context usage

**Example:**
```json
{
  "dataset_name": "UCI HAR",
  "description": "Human Activity Recognition dataset with 30 subjects performing 6 activities (walking, sitting, standing, etc.) while wearing a waist-mounted smartphone with accelerometer and gyroscope.",
  "channels": [
    {
      "name": "body_acc_x",
      "description": "Body acceleration in X-axis (gravity removed), from waist-mounted smartphone",
      "sampling_rate_hz": 50.0
    },
    {
      "name": "body_acc_y",
      "description": "Body acceleration in Y-axis (gravity removed)",
      "sampling_rate_hz": 50.0
    },
    {
      "name": "total_acc_x",
      "description": "Total acceleration in X-axis (gravity included)",
      "sampling_rate_hz": 50.0
    },
    {
      "name": "gyro_x",
      "description": "Angular velocity around X-axis from gyroscope",
      "sampling_rate_hz": 50.0
    }
  ]
}
```

---

### 2. `labels.json`

**Purpose:** Map each session ID to its activity/task labels.

**Format:**
```json
{
  "session_001": ["walking"],
  "session_002": ["running", "outdoor"],
  "session_003": ["sitting"],
  ...
}
```

**Key Points:**
- **Session IDs**: Keys are session folder names (e.g., `"session_001"`)
- **Label lists**: Values are **always lists**, even for single-label datasets
- **Multi-label support**: Sessions can have multiple labels (e.g., `["running", "outdoor"]`)
- **Flexible naming**: Session IDs can follow any naming convention (e.g., `subject1_activity3_trial2`)

**Example:**
```json
{
  "session_001": ["walking"],
  "session_002": ["walking_upstairs"],
  "session_003": ["walking_downstairs"],
  "session_004": ["sitting"],
  "session_005": ["standing"],
  "session_006": ["laying"]
}
```

---

### 3. `sessions/{session_id}/data.parquet`

**Purpose:** Store the actual time series data for a single session.

**Format:** Pandas DataFrame saved as parquet with the following structure:

| Column Name | Type | Description |
|-------------|------|-------------|
| `timestamp_sec` | float | Time in seconds from session start (required first column) |
| `channel_1` | float | First sensor channel (e.g., `accel_x`) |
| `channel_2` | float | Second sensor channel (e.g., `accel_y`) |
| ... | float | Additional channels |

**Key Points:**
- **First column is always `timestamp_sec`**: Real-world time in seconds from session start
- **All other columns are sensor channels**: Column names must match those in `manifest.json`
- **No timestep columns**: We use real-world time (seconds), never integer timesteps
- **Parquet format**: Efficient binary storage for DataFrames
- **Uniform sampling**: Within a session, all channels have the same sampling rate (resampled if needed)

**Example DataFrame:**

```python
import pandas as pd

# Example session with 3 channels sampled at 50 Hz
df = pd.DataFrame({
    'timestamp_sec': [0.00, 0.02, 0.04, 0.06, 0.08],  # 50 Hz → 0.02 sec intervals
    'accel_x': [0.12, 0.15, 0.13, 0.11, 0.14],
    'accel_y': [-0.03, -0.02, -0.04, -0.03, -0.02],
    'gyro_z': [0.001, 0.002, 0.001, 0.003, 0.002]
})

# Save as parquet
df.to_parquet('data/my_dataset/sessions/session_001/data.parquet', index=False)
```

---

## Design Principles

### 1. **Bag of Sessions**
- Each dataset is a **collection of independent sessions**
- A session represents a single continuous recording (e.g., one activity trial, one subject-activity combo)
- Sessions are stored in separate folders: `sessions/session_001/`, `sessions/session_002/`, etc.
- Session IDs are discovered at runtime (no need to enumerate them in manifest)

### 2. **Pandas DataFrames**
- Each session's data is stored as a **pandas DataFrame** in parquet format
- First column is always `timestamp_sec` (real-world time in seconds)
- Remaining columns are sensor channels (accelerometer, gyroscope, EMG, etc.)
- Parquet provides efficient compression and fast loading

### 3. **Real-World Time (Seconds), Never Timesteps**
- **All temporal reasoning uses seconds**, not integer timesteps
- When filtering by time: `filter_by_time(start_sec=5.0, end_sec=15.0)`
- When computing durations: `duration_sec = end_sec - start_sec`
- When specifying patch sizes (Phase 2): `patch_size_sec = 2.0` (not "128 timesteps")
- **Why?** Real-world time is dataset-agnostic and makes reasoning transferable across datasets with different sampling rates

### 4. **Dataset-Agnostic**
- Same structure works for any time series dataset (HAR, medical, industrial, etc.)
- Tools operate on this standard format without dataset-specific logic
- Easy to add new datasets by converting to this format

### 5. **Minimal Metadata**
- Manifests contain only essential information
- No file paths, subject lists, or episode enumerations
- Session lists discovered at runtime by scanning `sessions/` folder
- Reduces token usage when feeding metadata to LLM

### 6. **Channel Metadata**
- Every channel has a **textual description** explaining what it measures
- Every channel has a **sampling rate in Hz**
- This metadata helps LLMs reason about which channels to use for a task

---

## Real-World Time Examples

### ✅ Correct (Real-World Time)
```python
# Filter to first 10 seconds of each session
filter_by_time(dataset_name="uci_har", start_sec=0.0, end_sec=10.0)

# Select a 5-second window from 2 to 7 seconds
filter_by_time(dataset_name="pamap2", start_sec=2.0, end_sec=7.0)

# Compute duration in seconds
duration_sec = 15.5 - 3.2  # 12.3 seconds

# Patch size in seconds (Phase 2)
patch_size_sec = 1.5  # 1.5 seconds per patch
```

### ❌ Incorrect (Timesteps)
```python
# DON'T use timesteps
filter_by_timesteps(dataset_name="uci_har", start_step=0, end_step=500)

# DON'T convert to timesteps
num_steps = int(duration_sec * sampling_rate)  # Avoid this!

# DON'T specify patch size in timesteps
patch_size_steps = 128  # This is dataset-specific!
```

**Why real-world time?**
- Different datasets have different sampling rates (20 Hz, 50 Hz, 100 Hz, 200 Hz)
- "10 seconds" means the same thing across all datasets
- "500 timesteps" means 10 sec @ 50 Hz, but 2.5 sec @ 200 Hz!
- Real-world time makes reasoning transferable

---

## Phase 1 vs Phase 2 Usage

### Phase 1: EDA Tools (Current)
Tools operate on this standardized format:
- `show_session_stats`: Compute session counts and durations in seconds
- `show_channel_stats`: Compute statistics for each channel
- `select_channels`: Filter to specific channels
- `filter_by_time`: Extract temporal windows in seconds

**Example workflow:**
```python
# 1. Get overview
stats = show_session_stats(dataset_name="uci_har")
# Returns: {num_sessions: 7352, avg_duration_sec: 2.56, ...}

# 2. Explore channels
channels = show_channel_stats(dataset_name="uci_har")
# Returns: {body_acc_x: {mean: 0.08, std: 0.15, ...}, ...}

# 3. Focus on relevant channels
select_channels(dataset_name="uci_har", channel_names=["body_acc_x", "body_acc_y", "gyro_z"])

# 4. Extract temporal window (real-world time!)
filter_by_time(dataset_name="uci_har", start_sec=0.0, end_sec=2.0)
```

### Phase 2: Model Selection (Future)
Tokenizers consume this format:
- Load session parquet files
- Use `timestamp_sec` to create patches with **real-world durations** (e.g., 1.5 sec patches)
- Use channel metadata to select relevant channels
- Use sampling rates for preprocessing decisions

**Example workflow:**
```python
# 1. Select tokenizer based on task
select_tokenizer(dataset_name="uci_har", task_type="classification",
                 sequence_duration_sec=2.56, num_channels=3, priority="speed")
# Returns: {model_id: "Model-A", latent_dim: 64, ...}

# 2. Configure tokenizer parameters (real-world time!)
configure_tokenizer(model_id="Model-A", patch_size_sec=0.5, stride_sec=0.25)

# 3. Select task head
select_task_head(model_id="Model-A", task_type="classification", num_classes=6)
```

---

## Multi-Rate Data (Future Enhancement)

Some datasets have channels with **different sampling rates** (e.g., ActionSense: 60 Hz joints, 200 Hz EMG, 120 Hz gaze).

**Current approach:** Convert to single session per modality or resample to common rate

**Future approach:** Support multi-rate DataFrames with interpolation
```json
{
  "channels": [
    {"name": "joint_0_x", "sampling_rate_hz": 60.0},
    {"name": "emg_0", "sampling_rate_hz": 200.0},
    {"name": "gaze_x", "sampling_rate_hz": 120.0}
  ]
}
```

The DataFrame would have `timestamp_sec` with irregular spacing, and tools would interpolate as needed.

---

## Validation Checklist

When adding a new dataset, verify:

- [ ] `manifest.json` exists with dataset description
- [ ] Every channel in manifest has `name`, `description`, and `sampling_rate_hz`
- [ ] `labels.json` exists with all session IDs as keys
- [ ] All labels are **lists** (even single labels)
- [ ] `sessions/` folder contains one subfolder per session
- [ ] Each session has `data.parquet` file
- [ ] Each parquet has `timestamp_sec` as first column
- [ ] All other columns match channel names in manifest
- [ ] `timestamp_sec` starts at 0.0 for each session
- [ ] Real-world time is used (seconds), not timesteps

---

## References

- **ARCHITECTURE.md**: Full system design (3-layer architecture, Option 2)
- **tools/schemas.json**: Phase 1 EDA tool specifications
- **tools/PHASE2_DESIGN.md**: Phase 2 model selection tool specifications
- **datascripts/README.md**: Dataset pipeline and conversion scripts
- **docs/option2_rationale.md**: Why this architecture was chosen

---

## Summary

✅ **Bag of sessions**: Each dataset is a collection of independent sessions
✅ **Pandas DataFrames**: Each session is stored as parquet with `timestamp_sec` + channels
✅ **Labels file**: `labels.json` maps session IDs to activity labels (always lists)
✅ **Manifest file**: `manifest.json` contains textual descriptions and sampling rates per channel
✅ **Real-world time**: All operations use **seconds**, never timesteps
✅ **Dataset-agnostic**: Same structure works for any time series dataset
✅ **Tool-friendly**: Easy for both symbolic tools and tokenizers to consume
