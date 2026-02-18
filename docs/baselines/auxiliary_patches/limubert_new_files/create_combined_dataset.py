#!/usr/bin/env python3
"""Create a combined training dataset for LiMU-BERT multi-dataset pretraining.

Concatenates all training datasets into a single data/label .npy pair.
Labels are dataset-local (not globally remapped) since pretraining is
self-supervised and doesn't use labels.
"""
import json
import numpy as np
from pathlib import Path

PROC = Path("/home/alex/code/tsfm/benchmark_data/processed/limubert")
DS_DIR = Path("/home/alex/code/tsfm/auxiliary_repos/LIMU-BERT-Public/dataset")

TRAIN_DATASETS = [
    "uci_har", "hhar", "pamap2", "wisdm", "dsads",
    "kuhar", "unimib_shar", "hapt", "mhealth", "recgym",
]

VERSION = "20_120"

print("Creating combined training dataset for LiMU-BERT pretraining")
print(f"Datasets: {len(TRAIN_DATASETS)}")

all_data = []
all_labels = []

for ds in TRAIN_DATASETS:
    data = np.load(PROC / ds / f"data_{VERSION}.npy")
    labels = np.load(PROC / ds / f"label_{VERSION}.npy")
    print(f"  {ds}: {data.shape[0]} windows, shape {data.shape}")
    all_data.append(data)
    all_labels.append(labels)

combined_data = np.concatenate(all_data, axis=0)
combined_labels = np.concatenate(all_labels, axis=0)

print(f"\nCombined: {combined_data.shape[0]} total windows")
print(f"  Data shape: {combined_data.shape}")
print(f"  Label shape: {combined_labels.shape}")

# Save combined dataset
out_dir = DS_DIR / "combined_train"
out_dir.mkdir(parents=True, exist_ok=True)

np.save(out_dir / f"data_{VERSION}.npy", combined_data.astype(np.float32))
np.save(out_dir / f"label_{VERSION}.npy", combined_labels.astype(np.float32))

print(f"\nSaved to {out_dir}")

# Add entry to data_config.json
config_path = DS_DIR / "data_config.json"
config = json.load(open(config_path))
config[f"combined_train_{VERSION}"] = {
    "sr": 20,
    "seq_len": 120,
    "dimension": 6,
    "activity_label_index": 0,
    "activity_label_size": 0,  # Not meaningful for combined (pretraining only)
    "user_label_index": 1,
    "user_label_size": 0,
    "size": int(combined_data.shape[0]),
}
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"Updated {config_path} with combined_train_{VERSION} entry")
