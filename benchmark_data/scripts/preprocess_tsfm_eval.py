#!/usr/bin/env python3
"""Preprocess raw CSVs into native-rate .npy files for TSFM evaluation.

Unlike the LIMU-BERT preprocessing (preprocess_limubert.py), this script:
  - Does NOT downsample — keeps native sampling rate (e.g. 50Hz)
  - Does NOT convert accelerometer units — keeps native units matching TSFM training data
  - Windows at native rate: window_size = int(native_hz * 6.0)

Output per dataset in benchmark_data/processed/tsfm_eval/{dataset}/:
  - data_native.npy:  (N, window_size, 6) sensor windows
  - label_native.npy: (N, window_size, 2) [activity_index, subject_index] per timestep
  - metadata.json:    sampling_rate_hz, window_size, n_windows

Processing steps:
  1. Read per-subject CSVs from benchmark_data/raw/{dataset}/
  2. Extract 6 core channels (acc_xyz + gyro_xyz); zero-pad gyro for acc-only datasets
  3. Window at native rate (non-overlapping, 6-second windows)
  4. Majority-vote labels per window
  5. Save .npy + metadata

Usage:
    python benchmark_data/scripts/preprocess_tsfm_eval.py [--datasets motionsense realworld ...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "benchmark_data"
RAW_DIR = BENCHMARK_DIR / "raw"
OUTPUT_DIR = BENCHMARK_DIR / "processed" / "tsfm_eval"

# Window duration in seconds (same as LIMU-BERT: 6-second non-overlapping windows)
WINDOW_DURATION_SEC = 6.0

CORE_CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

# Load dataset config
with open(BENCHMARK_DIR / "dataset_config.json") as f:
    CONFIG = json.load(f)

ALL_DATASETS = CONFIG["train_datasets"] + CONFIG["zero_shot_datasets"]


def extract_core_channels(df: pd.DataFrame, has_gyro: bool = True) -> np.ndarray:
    """Extract 6 core channels from a DataFrame.

    If gyro is not available (e.g. UniMiB SHAR), zero-pads the gyro channels.

    Returns:
        (T, 6) array: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    """
    n = len(df)
    result = np.zeros((n, 6))

    for i, ch in enumerate(["acc_x", "acc_y", "acc_z"]):
        if ch in df.columns:
            result[:, i] = df[ch].values

    if has_gyro:
        for i, ch in enumerate(["gyro_x", "gyro_y", "gyro_z"]):
            if ch in df.columns:
                result[:, 3 + i] = df[ch].values

    return result


def window_data(
    data: np.ndarray,
    labels: np.ndarray,
    window_size: int,
) -> tuple:
    """Create non-overlapping windows from data.

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

    n_windows = n_samples // window_size
    usable = n_windows * window_size

    windows = data[:usable].reshape(n_windows, window_size, data.shape[1])
    label_windows = labels[:usable].reshape(n_windows, window_size)

    window_labels = np.array(
        [np.bincount(lw).argmax() for lw in label_windows], dtype=int
    )

    return windows, window_labels


def process_dataset(dataset: str):
    """Process a single dataset into native-rate .npy files for TSFM evaluation."""
    ds_config = CONFIG["datasets"][dataset]
    raw_dataset_dir = RAW_DIR / dataset
    out_dir = OUTPUT_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing {ds_config['display_name']} ({dataset}) for TSFM eval")
    print(f"{'='*60}")

    # Load metadata
    with open(raw_dataset_dir / "metadata.json") as f:
        metadata = json.load(f)

    native_hz = metadata["sampling_rate_hz"]
    window_size = int(native_hz * WINDOW_DURATION_SEC)
    activities = sorted(metadata["activities"])
    activity_to_idx = {a: i for i, a in enumerate(activities)}
    subjects = metadata["subjects"]
    subject_to_idx = {str(s): i for i, s in enumerate(sorted(subjects, key=str))}

    has_gyro = any("gyro" in ch for ch in metadata["channels"])
    print(f"  Native Hz: {native_hz}, Window size: {window_size} ({WINDOW_DURATION_SEC}s)")
    print(f"  Has gyro: {has_gyro}")
    print(f"  Activities ({len(activities)}): {activities}")
    print(f"  Subjects: {len(subjects)}")

    all_windows = []
    all_labels = []

    csv_files = sorted(raw_dataset_dir.glob("subject_*.csv"))
    for csv_path in csv_files:
        subject_name = csv_path.stem.replace("subject_", "")
        subject_idx = subject_to_idx.get(subject_name, -1)
        if subject_idx == -1:
            try:
                subject_idx = subject_to_idx.get(str(int(subject_name)), -1)
            except ValueError:
                pass

        df = pd.read_csv(csv_path)

        sensor_data = extract_core_channels(df, has_gyro=has_gyro)

        activity_labels = np.array(
            [activity_to_idx.get(a, -1) for a in df["activity"]], dtype=int
        )

        # Drop rows where core channels are NaN
        valid_mask = ~np.isnan(sensor_data[:, 0])
        if not valid_mask.all():
            n_before = len(sensor_data)
            sensor_data = sensor_data[valid_mask]
            activity_labels = activity_labels[valid_mask]
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

        # NO downsampling — keep native rate
        # NO unit conversion — keep native units matching TSFM training data

        # Window the data (non-overlapping, 6-second windows at native rate)
        windows, window_act_labels = window_data(sensor_data, activity_labels, window_size)

        if len(windows) > 0:
            n_win = len(windows)
            labels = np.zeros((n_win, window_size, 2), dtype=int)
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

    all_windows = np.concatenate(all_windows, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"\n  Total windows: {all_windows.shape[0]}")
    print(f"  Data shape: {all_windows.shape}  (N, {window_size}, 6)")
    print(f"  Label shape: {all_labels.shape}  (N, {window_size}, 2)")

    # Save
    np.save(out_dir / "data_native.npy", all_windows.astype(np.float32))
    np.save(out_dir / "label_native.npy", all_labels.astype(np.int32))

    meta_out = {
        "sampling_rate_hz": native_hz,
        "window_size": window_size,
        "window_duration_sec": WINDOW_DURATION_SEC,
        "n_windows": int(all_windows.shape[0]),
        "has_gyro": has_gyro,
        "activity_to_idx": activity_to_idx,
        "subject_to_idx": subject_to_idx,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta_out, f, indent=2, default=str)

    print(f"  Saved: data_native.npy, label_native.npy, metadata.json")

    return meta_out


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw CSVs into native-rate .npy files for TSFM evaluation"
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(args.datasets)} datasets for TSFM evaluation (native rate)")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Window duration: {WINDOW_DURATION_SEC}s (non-overlapping)")

    for ds in args.datasets:
        process_dataset(ds)

    print(f"\nDone! Processed {len(args.datasets)} datasets")


if __name__ == "__main__":
    main()
