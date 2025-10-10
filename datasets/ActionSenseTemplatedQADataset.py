import os
import random
import json
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import pandas as pd
import torch
from torch.utils.data import Dataset
import argparse

class ActionSenseTemplatedQADataset(Dataset):
    """
    Dataset for templated ActionSense QA pairs, supporting single and multi-activity contexts.

    - Loads QA pairs from a .jsonl file.
    - Caches sensor data from CSVs referenced in a manifest file.
    - For multi-activity questions, concatenates patches from all relevant sensor files.
    """

    def __init__(
        self,
        base_dir: str = "data/actionsenseqa/data",
        qa_jsonl_path: str = "data/actionsenseqa/data/qa_pairs_templated.jsonl",
        manifest_csv_path: str = "data/actionsenseqa/data/manifest.csv",
        split: str = "train",
        val_ratio: float = 0.2,
        split_seed: int = 42,
        patch_size: int = 1000,
        log_mode: str = "info",
    ) -> None:
        assert split in {"train", "val"}, "split must be 'train' or 'val'"

        self.base_dir = base_dir
        self.patch_size = int(patch_size)
        self.log_mode = log_mode

        all_records = self._load_qa_records(qa_jsonl_path)
        if not all_records:
            raise ValueError(f"No QA records could be loaded from {qa_jsonl_path}")

        self.manifest_map = self._load_manifest(manifest_csv_path)
        self.records = self._split_records(all_records, split, val_ratio, split_seed)
        sensor_paths = self._get_required_sensor_paths(self.records)
        self.sensor_cache = self._load_sensor_cache(sensor_paths)

        self._log(
            f"[INFO] ActionSenseTemplatedQADataset split={split}: {len(self.records)} QA pairs loaded.",
            level="info",
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        all_patches = []
        all_timestamps = []
        all_columns = []

        subject = record["subject"]
        split = record["split"]

        for activity_index in record["activity_indices"]:
            manifest_key = (subject, split, activity_index)
            if manifest_key not in self.manifest_map:
                raise KeyError(f"Activity key {manifest_key} not found in manifest.")

            csv_path = self.manifest_map[manifest_key]["csv_path"]
            cache_key = os.path.join(self.base_dir, csv_path)
            
            cache_entry = self.sensor_cache.get(cache_key)
            if cache_entry is None:
                raise KeyError(f"Sensor cache missing for path: {cache_key}")

            values = cache_entry["values"]
            timestamps = cache_entry["timestamps"]
            
            patches, trimmed_ts = self._segment_to_patches(values, timestamps)
            all_patches.append(patches)
            all_timestamps.append(trimmed_ts)
            if not all_columns:
                all_columns = cache_entry["columns"]

        final_patches = np.concatenate(all_patches, axis=0)
        final_timestamps = np.concatenate(all_timestamps, axis=0)

        sample = {
            "patches": torch.from_numpy(final_patches).float(),
            "question": record["question"],
            "answer": record["answer"],
            "metadata": {
                "question_type": record["question_type"],
                "subject": record["subject"],
                "split": record["split"],
                "activity_indices": record["activity_indices"],
                "activity_names": record["activity_names"],
                "timestamps": torch.from_numpy(final_timestamps.astype(np.float64)),
                "columns": all_columns,
            },
        }
        return sample

    def _load_qa_records(self, qa_jsonl_path: str) -> List[Dict[str, Any]]:
        records = []
        with open(qa_jsonl_path, 'r') as f:
            for line in f:
                records.append(json.loads(line))
        return records

    def _load_manifest(self, manifest_csv_path: str) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
        if not os.path.exists(manifest_csv_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_csv_path}")
        
        df = pd.read_csv(manifest_csv_path)
        mapping: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
        for _, row in df.iterrows():
            key = (str(row["subject"]), str(row["split"]), int(row["activity_index"]))
            mapping[key] = row.to_dict()
        return mapping

    def _get_required_sensor_paths(self, records: List[Dict[str, Any]]) -> List[str]:
        unique_paths = set()
        for record in records:
            subject = record["subject"]
            split = record["split"]
            for activity_index in record["activity_indices"]:
                manifest_key = (subject, split, activity_index)
                if manifest_key in self.manifest_map:
                    csv_path = self.manifest_map[manifest_key]["csv_path"]
                    full_path = os.path.join(self.base_dir, csv_path)
                    unique_paths.add(full_path)
        return sorted(list(unique_paths))

    def _split_records(
        self,
        records: List[Dict[str, Any]],
        split: str,
        val_ratio: float,
        split_seed: int
    ) -> List[Dict[str, Any]]:
        n = len(records)
        rng = npr.RandomState(split_seed)
        indices = np.arange(n)
        rng.shuffle(indices)

        n_val = int(round(val_ratio * n))
        n_val = min(max(n_val, 0), n)

        chosen_indices = indices[n_val:] if split == "train" else indices[:n_val]
        return [records[i] for i in chosen_indices]

    def _load_sensor_cache(self, sensor_paths: List[str]) -> Dict[str, Dict[str, Any]]:
        cache: Dict[str, Dict[str, Any]] = {}
        for path in sensor_paths:
            if not os.path.exists(path):
                self._log(f"[WARN] Sensor CSV missing: {path}", level="warn")
                continue
            df = pd.read_csv(path)
            if "time_s" not in df.columns:
                raise ValueError(f"Sensor CSV missing 'time_s' column: {path}")

            timestamps = df["time_s"].to_numpy(dtype=np.float64)
            numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "time_s"]
            values = df[numeric_cols].to_numpy(dtype=np.float32)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

            cache[path] = {
                "timestamps": timestamps,
                "values": values,
                "columns": numeric_cols,
            }
        self._log(f"[INFO] Cached sensor data for {len(cache)} files", level="info")
        return cache

    def _log(self, message: str, level: str = "info") -> None:
        if self.log_mode == "silent":
            return
        if level == "info" and self.log_mode in {"info", "debug"}:
            print(message)
        elif level == "warn":
            print(message)

    def _segment_to_patches(
        self, segment: np.ndarray, timestamps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if segment.size == 0:
            return np.array([]).reshape(0, self.patch_size, segment.shape[1] if segment.ndim > 1 else 1), np.array([])

        if segment.shape[0] != timestamps.shape[0]:
            raise ValueError("Sensor values and timestamps must have identical length")

        if self.patch_size <= 0:
            return segment[np.newaxis, ...], timestamps

        T = segment.shape[0]
        if T < self.patch_size:
            pad = self.patch_size - T
            pad_vals = np.zeros((pad, segment.shape[1]), dtype=segment.dtype)
            pad_ts = np.full((pad,), timestamps[0] if T > 0 else 0, dtype=timestamps.dtype)
            segment = np.concatenate([pad_vals, segment], axis=0)
            timestamps = np.concatenate([pad_ts, timestamps], axis=0)
            return segment[np.newaxis, ...], timestamps

        remainder = T % self.patch_size
        if remainder != 0:
            usable = T - remainder
            segment = segment[-usable:]
            timestamps = timestamps[-usable:]

        patches = segment.reshape(-1, self.patch_size, segment.shape[1])
        return patches, timestamps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug and visualize the ActionSenseTemplatedQADataset.")
    parser.add_argument("--qa_path", type=str, default="data/actionsenseqa/data/qa_pairs_templated.jsonl")
    parser.add_argument("--manifest_path", type=str, default="data/actionsenseqa/data/manifest.csv")
    parser.add_argument("--base_dir", type=str, default="data/actionsenseqa/data")
    parser.add_argument("--outdir", type=str, default="debug/templated_dataset_samples")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of random samples to visualize.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Initializing ActionSenseTemplatedQADataset...")
    dataset = ActionSenseTemplatedQADataset(
        base_dir=args.base_dir,
        qa_jsonl_path=args.qa_path,
        manifest_csv_path=args.manifest_path,
        split="train",
        val_ratio=0.2,
        split_seed=args.seed,
    )
    print(f"Dataset loaded with {len(dataset)} samples.")

    num_to_plot = min(args.num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_to_plot)

    for i, idx in enumerate(indices):
        print(f"\n--- Preparing sample {i+1}/{num_to_plot} (index: {idx}) ---")
        sample = dataset[idx]
        metadata = sample["metadata"]
        patches = sample["patches"].numpy()
        
        print(f"  Question Type: {metadata['question_type']}")
        print(f"  Question: {sample['question']}")
        print(f"  Answer: {sample['answer']}")
        print(f"  Patches shape: {patches.shape}")

        # --- Plotting --- #
        num_activities = len(metadata["activity_indices"])
        fig, axes = plt.subplots(num_activities, 1, figsize=(14, 4 * num_activities), sharex=True, squeeze=False)
        axes = axes.flatten()

        fig.suptitle(f"Q: {sample['question']}\nA: {sample['answer']}", fontsize=12, wrap=True)

        patch_counts = []
        for act_idx in metadata["activity_indices"]:
            manifest_key = (metadata["subject"], metadata["split"], act_idx)
            csv_path = dataset.manifest_map[manifest_key]['csv_path']
            cache_key = os.path.join(dataset.base_dir, csv_path)
            sensor_vals = dataset.sensor_cache[cache_key]['values']
            num_patches = 0
            if dataset.patch_size > 0:
                if len(sensor_vals) < dataset.patch_size:
                    num_patches = 1
                else:
                    num_patches = (len(sensor_vals) - (len(sensor_vals) % dataset.patch_size)) // dataset.patch_size
            else:
                 num_patches = 1
            patch_counts.append(num_patches)

        current_patch_offset = 0
        for j, act_idx in enumerate(metadata["activity_indices"]):
            ax = axes[j]
            num_patches_for_activity = patch_counts[j]
            activity_patches = patches[current_patch_offset : current_patch_offset + num_patches_for_activity]
            current_patch_offset += num_patches_for_activity

            if activity_patches.size == 0:
                ax.set_title(f"Session {act_idx}: {metadata['activity_names'][j]} (No Data)")
                continue

            flat_sequence = activity_patches.reshape(-1, activity_patches.shape[-1])
            time_axis = np.arange(flat_sequence.shape[0])

            for d in range(min(10, flat_sequence.shape[1])):
                label = metadata["columns"][d] if metadata["columns"] and d < len(metadata["columns"]) else f"ch{d}"
                ax.plot(time_axis, flat_sequence[:, d], label=label)
            
            ax.set_title(f"Session {act_idx}: {metadata['activity_names'][j]}")
            ax.set_ylabel("Sensor Value")
            ax.legend(loc="upper right", ncol=2)

        axes[-1].set_xlabel("Time (points)")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        save_path = os.path.join(args.outdir, f"sample_{idx}.png")
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"  Plot saved to {save_path}")
