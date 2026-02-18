#!/usr/bin/env python3
"""Preprocess raw CSVs into LIMU-BERT format .npy files.

LIMU-BERT expects:
  - data_20_120.npy: shape (N, 120, 6) - 6-second windows at 20Hz
  - label_20_120.npy: shape (N, 120, 2) - [activity_index, subject_index] repeated per timestep

Channels: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
Accelerometer values must be in m/s² (the LIMU-BERT training pipeline divides by 9.8).

Processing steps (matching original LIMU-BERT paper):
  1. Read per-subject CSVs from benchmark_data/raw/{dataset}/
  2. Downsample to 20Hz via temporal averaging (bin-and-mean), matching original implementation
  3. Extract 6 core channels (acc_x/y/z, gyro_x/y/z); zero-pad gyro for acc-only datasets
  4. Convert accelerometer to m/s² if stored in g units
  5. Window into non-overlapping 120-sample segments
  6. Save as .npy files

Usage:
    python benchmark_data/scripts/preprocess_limubert.py [--datasets uci_har hhar ...]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "benchmark_data"
RAW_DIR = BENCHMARK_DIR / "raw"
LIMUBERT_DIR = BENCHMARK_DIR / "processed" / "limubert"

# LIMU-BERT parameters
TARGET_HZ = 20
WINDOW_SIZE = 120  # 6 seconds at 20Hz
WINDOW_STRIDE = 120  # Non-overlapping (matches original LIMU-BERT)

CORE_CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

GRAVITY_MS2 = 9.80665

# Datasets where accelerometer values are stored in g units (not m/s²).
# These need to be multiplied by GRAVITY_MS2 before saving, since the
# LIMU-BERT training pipeline expects m/s² and normalizes by dividing by 9.8.
ACC_IN_G_UNITS = {"motionsense", "vtt_coniot"}

# Load dataset config
with open(BENCHMARK_DIR / "dataset_config.json") as f:
    CONFIG = json.load(f)

ALL_DATASETS = CONFIG["train_datasets"] + CONFIG["zero_shot_datasets"]


def downsample_by_averaging(data: np.ndarray, original_hz: float) -> np.ndarray:
    """Downsample sensor data to TARGET_HZ using temporal averaging.

    Matches the original LIMU-BERT downsampling approach: groups consecutive
    samples into bins and takes the mean of each bin. This acts as a low-pass
    anti-aliasing filter, unlike linear interpolation.

    Args:
        data: (T, C) array of sensor readings
        original_hz: original sampling rate

    Returns:
        Downsampled (T', C) array at TARGET_HZ
    """
    if original_hz == TARGET_HZ:
        return data

    ratio = original_hz / TARGET_HZ
    n_samples = data.shape[0]

    if ratio < 1:
        # Upsampling case (rare) — fall back to repeat-nearest since the
        # original LIMU-BERT code only handles downsampling.
        n_target = int(round(n_samples / ratio))
        indices = np.linspace(0, n_samples - 1, n_target).astype(int)
        return data[indices]

    # Downsampling by bin-and-mean, matching original LIMU-BERT down_sample()
    result = []
    if ratio == int(ratio):
        # Integer ratio: simple non-overlapping bins
        window = int(ratio)
        for i in range(0, n_samples - window + 1, window):
            result.append(np.mean(data[i : i + window], axis=0))
    else:
        # Non-integer ratio: variable-width bins (original LIMU-BERT approach)
        window = int(ratio)
        remainder = 0.0
        i = 0
        while i + window < n_samples:
            remainder += ratio - window
            if remainder >= 1:
                remainder -= 1
                result.append(np.mean(data[i : i + window + 1], axis=0))
                i += window + 1
            else:
                result.append(np.mean(data[i : i + window], axis=0))
                i += window

    if not result:
        return np.empty((0, data.shape[1]))
    return np.array(result)


def extract_core_channels(df: pd.DataFrame, has_gyro: bool = True) -> np.ndarray:
    """Extract 6 core channels from a DataFrame.

    If gyro is not available (e.g. UniMiB SHAR), zero-pads the gyro channels.

    Returns:
        (T, 6) array: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    """
    n = len(df)
    result = np.zeros((n, 6))

    # Accelerometer (always present)
    for i, ch in enumerate(["acc_x", "acc_y", "acc_z"]):
        if ch in df.columns:
            result[:, i] = df[ch].values

    # Gyroscope (may not be present)
    if has_gyro:
        for i, ch in enumerate(["gyro_x", "gyro_y", "gyro_z"]):
            if ch in df.columns:
                result[:, 3 + i] = df[ch].values

    return result


def window_data(
    data: np.ndarray,
    labels: np.ndarray,
    window_size: int = WINDOW_SIZE,
) -> tuple:
    """Create non-overlapping windows from data, matching original LIMU-BERT.

    The original LIMU-BERT truncates data to be evenly divisible by window_size,
    then reshapes into non-overlapping windows. Each window gets the majority
    activity label.

    Args:
        data: (T, C) sensor data
        labels: (T,) activity indices per sample
        window_size: number of samples per window

    Returns:
        windows: (N, window_size, C) windowed data
        window_labels: (N,) activity index per window (majority vote)
    """
    n_samples = data.shape[0]
    if n_samples < window_size:
        return np.empty((0, window_size, data.shape[1])), np.empty((0,), dtype=int)

    # Truncate to be evenly divisible, then reshape (matches original)
    n_windows = n_samples // window_size
    usable = n_windows * window_size

    windows = data[:usable].reshape(n_windows, window_size, data.shape[1])
    label_windows = labels[:usable].reshape(n_windows, window_size)

    # Majority label per window
    window_labels = np.array(
        [np.bincount(lw).argmax() for lw in label_windows], dtype=int
    )

    return windows, window_labels


def process_dataset(dataset: str):
    """Process a single dataset into LIMU-BERT format."""
    ds_config = CONFIG["datasets"][dataset]
    raw_dataset_dir = RAW_DIR / dataset
    out_dir = LIMUBERT_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing {ds_config['display_name']} ({dataset}) for LIMU-BERT")
    print(f"{'='*60}")

    # Load metadata
    with open(raw_dataset_dir / "metadata.json") as f:
        metadata = json.load(f)

    original_hz = metadata["sampling_rate_hz"]
    activities = sorted(metadata["activities"])
    activity_to_idx = {a: i for i, a in enumerate(activities)}
    subjects = metadata["subjects"]
    subject_to_idx = {str(s): i for i, s in enumerate(sorted(subjects, key=str))}

    has_gyro = any("gyro" in ch for ch in metadata["channels"])
    print(f"  Original Hz: {original_hz}, Target Hz: {TARGET_HZ}")
    print(f"  Has gyro: {has_gyro}")
    print(f"  Activities ({len(activities)}): {activities}")
    print(f"  Subjects: {len(subjects)}")

    all_windows = []
    all_labels = []

    # Process each subject file
    csv_files = sorted(raw_dataset_dir.glob("subject_*.csv"))
    for csv_path in csv_files:
        subject_name = csv_path.stem.replace("subject_", "")
        # Try exact match first, then try as integer (handles "01" -> "1")
        subject_idx = subject_to_idx.get(subject_name, -1)
        if subject_idx == -1:
            try:
                subject_idx = subject_to_idx.get(str(int(subject_name)), -1)
            except ValueError:
                pass

        df = pd.read_csv(csv_path)

        # Extract core channels
        sensor_data = extract_core_channels(df, has_gyro=has_gyro)

        # Build per-sample activity labels
        activity_labels = np.array(
            [activity_to_idx.get(a, -1) for a in df["activity"]], dtype=int
        )

        # Drop rows where core channels are NaN (e.g., WISDM has interleaved
        # phone/watch data where phone sensor rows are sparse)
        valid_mask = ~np.isnan(sensor_data[:, 0])
        if not valid_mask.all():
            n_before = len(sensor_data)
            sensor_data = sensor_data[valid_mask]
            activity_labels = activity_labels[valid_mask]
            # Fill any remaining NaN in other channels (e.g., sporadic dropouts)
            if np.isnan(sensor_data).any():
                for c in range(sensor_data.shape[1]):
                    col = sensor_data[:, c]
                    nans = np.isnan(col)
                    if nans.any():
                        col[nans] = 0.0
            print(
                f"      Dropped {n_before - len(sensor_data)} NaN rows "
                f"({(n_before - len(sensor_data))/n_before*100:.0f}%)"
            )

        # Downsample to 20Hz via temporal averaging (matches original LIMU-BERT)
        if original_hz != TARGET_HZ:
            n_orig = len(sensor_data)
            sensor_data = downsample_by_averaging(sensor_data, original_hz)
            n_target = len(sensor_data)

            # Nearest-neighbor for labels (categorical, can't average)
            if n_target > 0 and n_orig > 0:
                orig_indices = np.linspace(0, n_orig - 1, n_target).astype(int)
                activity_labels = activity_labels[orig_indices]

        # Convert accelerometer from g to m/s² if needed.
        # Original LIMU-BERT stores acc in m/s²; its training normalization
        # divides by 9.8 to bring values to ~g scale.
        if dataset in ACC_IN_G_UNITS:
            sensor_data[:, :3] *= GRAVITY_MS2

        # Window the data (non-overlapping, matches original LIMU-BERT)
        windows, window_act_labels = window_data(sensor_data, activity_labels)

        if len(windows) > 0:
            # Build per-timestep label array: (N, 120, 2) = [activity_index, subject_index]
            # LIMU-BERT expects labels repeated across all timesteps
            n_win = len(windows)
            labels = np.zeros((n_win, WINDOW_SIZE, 2), dtype=int)
            labels[:, :, 0] = window_act_labels[:, np.newaxis]
            labels[:, :, 1] = subject_idx

            all_windows.append(windows)
            all_labels.append(labels)

            print(
                f"    {csv_path.name}: {len(windows)} windows "
                f"(from {len(df)} samples)"
            )

    if not all_windows:
        print(f"  WARNING: No windows generated for {dataset}!")
        return None

    # Concatenate all windows
    all_windows = np.concatenate(all_windows, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"\n  Total windows: {all_windows.shape[0]}")
    print(f"  Data shape: {all_windows.shape}  (N, {WINDOW_SIZE}, 6)")
    print(f"  Label shape: {all_labels.shape}  (N, {WINDOW_SIZE}, 2)")

    # Save
    data_path = out_dir / f"data_{TARGET_HZ}_{WINDOW_SIZE}.npy"
    label_path = out_dir / f"label_{TARGET_HZ}_{WINDOW_SIZE}.npy"
    np.save(data_path, all_windows.astype(np.float32))
    np.save(label_path, all_labels.astype(np.int32))

    # Also save the activity and subject mappings
    mapping = {
        "activity_to_idx": activity_to_idx,
        "subject_to_idx": subject_to_idx,
        "window_size": WINDOW_SIZE,
        "sampling_rate_hz": TARGET_HZ,
        "stride": WINDOW_STRIDE,
        "has_gyro": has_gyro,
    }
    with open(out_dir / "mapping.json", "w") as f:
        json.dump(mapping, f, indent=2, default=str)

    print(f"  Saved: {data_path.name}, {label_path.name}, mapping.json")

    # Return stats for data_config.json generation
    return {
        "sr": TARGET_HZ,
        "seq_len": WINDOW_SIZE,
        "dimension": 6,
        "activity_label_index": 0,
        "activity_label_size": len(activity_to_idx),
        "activity_label": [a for a, _ in sorted(activity_to_idx.items(), key=lambda x: x[1])],
        "user_label_index": 1,
        "user_label_size": len(subject_to_idx),
        "has_gyro": has_gyro,
        "size": int(all_windows.shape[0]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw CSVs into LIMU-BERT format"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ALL_DATASETS,
        help=f"Datasets to process (default: all). Options: {ALL_DATASETS}",
    )
    args = parser.parse_args()

    for ds in args.datasets:
        if ds not in ALL_DATASETS:
            print(f"Error: Unknown dataset '{ds}'. Valid: {ALL_DATASETS}")
            sys.exit(1)

    LIMUBERT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(args.datasets)} datasets for LIMU-BERT")
    print(f"Output: {LIMUBERT_DIR}")
    print(f"Parameters: {TARGET_HZ}Hz, window={WINDOW_SIZE}, stride={WINDOW_STRIDE}")

    data_config = {}
    for ds in args.datasets:
        stats = process_dataset(ds)
        if stats:
            key = f"{ds}_{TARGET_HZ}_{WINDOW_SIZE}"
            data_config[key] = stats

    # Write LIMU-BERT compatible data_config.json
    config_path = LIMUBERT_DIR / "data_config.json"
    with open(config_path, "w") as f:
        json.dump(data_config, f, indent=2)
    print(f"\nWrote {config_path} with {len(data_config)} dataset entries")

    print(f"Done! Processed {len(args.datasets)} datasets")


if __name__ == "__main__":
    main()
