#!/usr/bin/env python3
"""Create a combined dataset for CrossHAR pretraining from all 10 training datasets.

Reads preprocessed LiMU-BERT format data and creates a pooled dataset
in CrossHAR's expected format.

CrossHAR expects:
  - dataset/{name}/data_20_120.npy: shape (N, 120, 6)
  - dataset/{name}/label_20_120.npy: shape (N, 120, K) where K >= 2

Usage:
    python create_combined_dataset.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LIMUBERT_DIR = PROJECT_ROOT / "benchmark_data" / "processed" / "limubert"
CROSSHAR_DIR = Path(__file__).resolve().parent
DATASET_DIR = CROSSHAR_DIR / "dataset"

with open(PROJECT_ROOT / "benchmark_data" / "dataset_config.json") as f:
    CONFIG = json.load(f)

TRAIN_DATASETS = CONFIG["train_datasets"]
TEST_DATASETS = CONFIG["zero_shot_datasets"]
ALL_DATASETS = TRAIN_DATASETS + TEST_DATASETS


def setup_dataset(dataset_name, target_name=None):
    """Symlink or copy a single dataset into CrossHAR's dataset directory."""
    if target_name is None:
        target_name = dataset_name

    src_dir = LIMUBERT_DIR / dataset_name
    dst_dir = DATASET_DIR / target_name

    data_src = src_dir / "data_20_120.npy"
    label_src = src_dir / "label_20_120.npy"

    if not data_src.exists():
        print(f"  SKIP {dataset_name}: data file not found")
        return None

    dst_dir.mkdir(parents=True, exist_ok=True)

    # Load data to get stats
    data = np.load(str(data_src))
    labels = np.load(str(label_src))

    # CrossHAR expects labels with at least 2 dimensions in axis 2
    # Our labels are (N, 120, 2) which is fine
    # But some CrossHAR code expects 3 label dimensions, so pad if needed
    if labels.shape[2] < 3:
        padded = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=labels.dtype)
        padded[:, :, :labels.shape[2]] = labels
        labels = padded

    # Save (copy, not symlink, since we might modify labels)
    np.save(str(dst_dir / "data_20_120.npy"), data.astype(np.float32))
    np.save(str(dst_dir / "label_20_120.npy"), labels.astype(np.int32))

    print(f"  {dataset_name} -> {target_name}: {data.shape[0]} windows, "
          f"data={data.shape}, label={labels.shape}")
    return data.shape[0]


def create_combined_dataset():
    """Pool all training datasets into a single combined dataset."""
    print("Creating combined training dataset...")

    all_data = []
    all_labels = []

    for ds in TRAIN_DATASETS:
        src_dir = LIMUBERT_DIR / ds
        data_path = src_dir / "data_20_120.npy"
        label_path = src_dir / "label_20_120.npy"

        if not data_path.exists():
            print(f"  SKIP {ds}: not found")
            continue

        data = np.load(str(data_path)).astype(np.float32)
        labels = np.load(str(label_path)).astype(np.int32)

        # Pad labels to 3 dimensions if needed
        if labels.shape[2] < 3:
            padded = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=labels.dtype)
            padded[:, :, :labels.shape[2]] = labels
            labels = padded

        all_data.append(data)
        all_labels.append(labels)
        print(f"  {ds}: {data.shape[0]} windows")

    combined_data = np.concatenate(all_data, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)

    # Shuffle
    rng = np.random.RandomState(3431)
    idx = np.arange(len(combined_data))
    rng.shuffle(idx)
    combined_data = combined_data[idx]
    combined_labels = combined_labels[idx]

    # Save
    dst_dir = DATASET_DIR / "combined_train"
    dst_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(dst_dir / "data_20_120.npy"), combined_data)
    np.save(str(dst_dir / "label_20_120.npy"), combined_labels)

    print(f"\nCombined: {combined_data.shape[0]} windows")
    print(f"  Data shape: {combined_data.shape}")
    print(f"  Label shape: {combined_labels.shape}")
    print(f"  Saved to: {dst_dir}")

    return combined_data.shape[0]


def build_data_config():
    """Build data_config.json with all datasets."""
    limubert_config_path = LIMUBERT_DIR / "data_config.json"
    with open(limubert_config_path) as f:
        limubert_config = json.load(f)

    data_config = {}

    # Add all individual datasets
    for ds in ALL_DATASETS:
        key = f"{ds}_20_120"
        limu_key = f"{ds}_20_120"
        if limu_key in limubert_config:
            entry = limubert_config[limu_key].copy()
            # Ensure required CrossHAR fields exist
            if "user_label_index" not in entry:
                entry["user_label_index"] = 1
            data_config[key] = entry

    # Add combined dataset entry
    total_size = sum(
        limubert_config.get(f"{ds}_20_120", {}).get("size", 0)
        for ds in TRAIN_DATASETS
    )

    # Collect all unique activity labels across training datasets
    all_activities = set()
    for ds in TRAIN_DATASETS:
        ds_info = CONFIG["datasets"].get(ds, {})
        all_activities.update(ds_info.get("activities", []))

    all_activities = sorted(all_activities)

    data_config["combined_train_20_120"] = {
        "sr": 20,
        "seq_len": 120,
        "dimension": 6,
        "activity_label_index": 0,
        "activity_label_size": len(all_activities),
        "activity_label": all_activities,
        "user_label_index": 1,
        "user_label_size": 1,
        "has_gyro": True,
        "size": total_size,
    }

    # Save
    config_path = DATASET_DIR / "data_config.json"
    with open(config_path, "w") as f:
        json.dump(data_config, f, indent=2)
    print(f"\nSaved data_config.json with {len(data_config)} entries")

    return data_config


def main():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Set up individual datasets
    print("Setting up individual datasets...")
    for ds in ALL_DATASETS:
        setup_dataset(ds)

    # Create combined dataset
    print()
    n_combined = create_combined_dataset()

    # Build data_config.json
    build_data_config()

    print(f"\nDone! All datasets ready in {DATASET_DIR}")


if __name__ == "__main__":
    main()
