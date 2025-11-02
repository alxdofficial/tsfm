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


class ActionSenseTemplatedQADataset_NativeRate(Dataset):
    """
    Dataset for templated ActionSense QA pairs with NATIVE SAMPLING RATES.

    DIFFERENCES from ActionSenseTemplatedQADataset:
    - Loads sensor streams at their native sampling rates (no upsampling)
    - Each stream (joints, emg_left, emg_right, gaze) loaded separately from individual CSVs
    - Uses patch_duration_s (temporal span) instead of patch_size (sample count)
    - Returns dict of patches: {stream_name: (P, T_native, D_stream)}

    - Loads QA pairs from a .jsonl file.
    - Caches sensor data from multiple CSVs per activity (one per stream).
    - For multi-activity questions, concatenates patches from all relevant sensor files.
    """

    # Stream names expected in manifest
    STREAM_NAMES = ["joints", "emg_left", "emg_right", "gaze"]

    # Nominal sampling rates for each stream type (Hz)
    # Used to ensure consistent T_native across activities for successful patch concatenation
    #
    # NOTE: These are TARGET rates from sensor specs. The download script resamples all
    # sensor streams to these regular rates, ensuring perfect temporal alignment.
    # - All CSVs are now resampled to regular grids at these rates
    # - No irregular sampling or gaps - data is interpolated during download
    # - Patches represent exact time durations across all streams
    NOMINAL_RATES = {
        "joints": 60.0,      # Xsens IMU: 60 Hz spec → 120 samples per 2s patch
        "emg_left": 200.0,   # Myo left: 200 Hz spec → 400 samples per 2s patch
        "emg_right": 200.0,  # Myo right: 200 Hz spec → 400 samples per 2s patch
        "gaze": 120.0,       # Tobii: 120 Hz spec → 240 samples per 2s patch
    }

    def __init__(
        self,
        base_dir: str = "data/actionsenseqa_native/data",
        qa_jsonl_path: str = "data/actionsenseqa_native/data/qa_pairs_templated.jsonl",
        manifest_csv_path: str = "data/actionsenseqa_native/data/manifest.csv",
        split: str = "train",
        val_ratio: float = 0.2,
        split_seed: int = 42,
        patch_duration_s: float = 2.0,  # Temporal span of each patch (seconds)
        log_mode: str = "info",
    ) -> None:
        assert split in {"train", "val"}, "split must be 'train' or 'val'"

        self.base_dir = base_dir
        self.patch_duration_s = float(patch_duration_s)
        self.log_mode = log_mode

        all_records = self._load_qa_records(qa_jsonl_path)
        if not all_records:
            raise ValueError(f"No QA records could be loaded from {qa_jsonl_path}")

        self.manifest_map = self._load_manifest(manifest_csv_path)
        self.records = self._split_records(all_records, split, val_ratio, split_seed)
        sensor_paths = self._get_required_sensor_paths(self.records)
        self.sensor_cache = self._load_sensor_cache(sensor_paths)

        self._log(
            f"[INFO] ActionSenseTemplatedQADataset_NativeRate split={split}: {len(self.records)} QA pairs loaded.",
            level="info",
        )
        self._log(
            f"[INFO] Using patch_duration_s={self.patch_duration_s}s (native rates will vary per stream)",
            level="info",
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]

        # Multi-rate patches: Dict[stream_name -> List[(P, T_native, D_stream)]]
        patches_per_stream = {stream: [] for stream in self.STREAM_NAMES}
        metadata_per_stream = {stream: [] for stream in self.STREAM_NAMES}

        subject = record["subject"]
        split = record["split"]

        for activity_index in record["activity_indices"]:
            manifest_key = (subject, split, activity_index)
            if manifest_key not in self.manifest_map:
                raise KeyError(f"Activity key {manifest_key} not found in manifest.")

            manifest_row = self.manifest_map[manifest_key]

            # Load each stream separately at native rate
            for stream_name in self.STREAM_NAMES:
                csv_path = manifest_row.get(f"{stream_name}_csv")

                # Skip if stream not available
                if not csv_path or pd.isna(csv_path) or csv_path == "":
                    continue

                rate_hz = manifest_row.get(f"{stream_name}_rate_hz", 0.0)
                num_channels = manifest_row.get(f"{stream_name}_channels", 0)

                if rate_hz == 0.0 or num_channels == 0:
                    continue

                # Load from cache
                cache_key = os.path.join(self.base_dir, csv_path)
                cache_entry = self.sensor_cache.get(cache_key)
                if cache_entry is None:
                    self._log(f"[WARN] Sensor cache missing for: {cache_key}", level="warn")
                    continue

                values = cache_entry["values"]  # (T, D)
                timestamps = cache_entry["timestamps"]  # (T,)

                # Compute patch_size using nominal rate for consistent T_native across activities
                # This prevents shape mismatches when concatenating patches from multiple activities
                nominal_rate = self.NOMINAL_RATES.get(stream_name, rate_hz)
                T_native = int(self.patch_duration_s * nominal_rate)
                T_native = max(1, T_native)  # Ensure at least 1 sample

                # Create patches at native rate
                patches, trimmed_ts = self._segment_to_patches_native(
                    values, timestamps, T_native
                )

                if patches.shape[0] > 0:  # Only add if patches exist
                    patches_per_stream[stream_name].append(patches)
                    metadata_per_stream[stream_name].append({
                        "rate_hz": rate_hz,
                        "num_channels": num_channels,
                        "T_native": T_native,
                        "timestamps": trimmed_ts,
                    })

        # Concatenate patches from multiple activities (if multi-activity question)
        final_patches = {}
        final_metadata = {}
        for stream_name in self.STREAM_NAMES:
            if patches_per_stream[stream_name]:
                # Concatenate along patch dimension
                final_patches[stream_name] = torch.from_numpy(
                    np.concatenate(patches_per_stream[stream_name], axis=0)
                ).float()

                # Merge metadata (use first entry's rate/channels, assuming consistent)
                final_metadata[stream_name] = {
                    "rate_hz": metadata_per_stream[stream_name][0]["rate_hz"],
                    "num_channels": metadata_per_stream[stream_name][0]["num_channels"],
                    "T_native": metadata_per_stream[stream_name][0]["T_native"],
                }

        # Skip samples with no valid streams
        if not final_patches:
            self._log(
                f"[WARN] Skipping sample with no valid streams: "
                f"subject={record['subject']}, split={record['split']}, "
                f"activities={record['activity_indices']}",
                level="warn"
            )
            # Return None to signal this sample should be skipped
            return None

        # Verify and align patches across streams (trims to minimum P if misaligned)
        final_patches = self._verify_patch_alignment(final_patches, record)

        sample = {
            "patches": final_patches,  # Dict[stream -> (P, T_native, D)]
            "question": record["question"],
            "answer": record["answer"],
            "metadata": {
                "question_type": record["question_type"],
                "subject": record["subject"],
                "split": record["split"],
                "activity_indices": record["activity_indices"],
                "activity_names": record["activity_names"],
                "stream_info": final_metadata,  # Per-stream rate/channels/T info
                "patch_duration_s": self.patch_duration_s,
            },
        }
        return sample

    def _load_qa_records(self, qa_jsonl_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(qa_jsonl_path):
            raise FileNotFoundError(f"QA JSONL not found: {qa_jsonl_path}")
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

        self._log(f"[INFO] Loaded manifest with {len(mapping)} activities", level="info")
        return mapping

    def _get_required_sensor_paths(self, records: List[Dict[str, Any]]) -> List[str]:
        unique_paths = set()
        for record in records:
            subject = record["subject"]
            split = record["split"]
            for activity_index in record["activity_indices"]:
                manifest_key = (subject, split, activity_index)
                if manifest_key in self.manifest_map:
                    manifest_row = self.manifest_map[manifest_key]

                    # Collect all stream CSVs
                    for stream_name in self.STREAM_NAMES:
                        csv_path = manifest_row.get(f"{stream_name}_csv", "")
                        if csv_path and not pd.isna(csv_path) and csv_path != "":
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
                self._log(f"[WARN] Sensor CSV missing 'time_s' column: {path}", level="warn")
                continue

            timestamps = df["time_s"].to_numpy(dtype=np.float64)
            numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "time_s"]
            values = df[numeric_cols].to_numpy(dtype=np.float32)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

            cache[path] = {
                "timestamps": timestamps,
                "values": values,
                "columns": numeric_cols,
            }

        self._log(f"[INFO] Cached sensor data for {len(cache)} stream files", level="info")
        return cache

    def _log(self, message: str, level: str = "info") -> None:
        if self.log_mode == "silent":
            return
        if level == "info" and self.log_mode in {"info", "debug"}:
            print(message)
        elif level == "warn":
            print(message)

    def _verify_patch_alignment(
        self,
        patches_dict: Dict[str, torch.Tensor],
        record: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Verify and align patches across streams by trimming to minimum patch count.
        This handles temporal misalignment in multi-rate sensor data.

        Args:
            patches_dict: Dict[stream_name -> (P, T_native, D)]
            record: The QA record being processed (for logging)

        Returns:
            Aligned patches_dict with all streams having the same P (minimum across streams)
        """
        if not patches_dict:
            return patches_dict  # No streams to verify

        # Count patches per stream
        patch_counts = {
            stream_name: patches.shape[0]
            for stream_name, patches in patches_dict.items()
        }

        unique_counts = set(patch_counts.values())

        if len(unique_counts) > 1:
            # Misalignment detected - trim to minimum P
            P_min = min(patch_counts.values())

            # self._log(
            #     f"[WARN] Patch misalignment detected - trimming to P_min={P_min}: "
            #     f"subject={record.get('subject')}, split={record.get('split')}, "
            #     f"activities={record.get('activity_indices')}, "
            #     f"counts={patch_counts}",
            #     level="warn"
            # )

            # Trim all streams to P_min (keep last P_min patches for temporal consistency)
            aligned_patches = {}
            for stream_name, patches in patches_dict.items():
                P_current = patches.shape[0]
                if P_current > P_min:
                    # Trim to last P_min patches
                    aligned_patches[stream_name] = patches[-P_min:]
                else:
                    aligned_patches[stream_name] = patches

            return aligned_patches

        # Already aligned - log success in debug mode
        num_patches = list(patch_counts.values())[0]
        self._log(
            f"[DEBUG] Patch alignment verified: {len(patches_dict)} streams, {num_patches} patches each",
            level="debug"
        )

        return patches_dict

    def _segment_to_patches_native(
        self,
        segment: np.ndarray,
        timestamps: np.ndarray,
        T_native: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment data into patches of size T_native (native sample count for this stream).

        Args:
            segment: (T, D) sensor values
            timestamps: (T,) timestamps
            T_native: patch size in samples for this stream's native rate

        Returns:
            patches: (P, T_native, D)
            trimmed_timestamps: (P * T_native,) timestamps for valid data
        """
        if segment.size == 0:
            D = segment.shape[1] if segment.ndim > 1 else 1
            return np.array([]).reshape(0, T_native, D), np.array([])

        if segment.shape[0] != timestamps.shape[0]:
            raise ValueError("Sensor values and timestamps must have identical length")

        if T_native <= 0:
            return segment[np.newaxis, ...], timestamps

        T = segment.shape[0]
        if T < T_native:
            # Pad if too short
            pad = T_native - T
            pad_vals = np.zeros((pad, segment.shape[1]), dtype=segment.dtype)
            pad_ts = np.full((pad,), timestamps[0] if T > 0 else 0.0, dtype=timestamps.dtype)
            segment = np.concatenate([pad_vals, segment], axis=0)
            timestamps = np.concatenate([pad_ts, timestamps], axis=0)
            return segment[np.newaxis, ...], timestamps

        # Trim to multiple of T_native
        remainder = T % T_native
        if remainder != 0:
            usable = T - remainder
            segment = segment[-usable:]
            timestamps = timestamps[-usable:]

        # Reshape into patches
        patches = segment.reshape(-1, T_native, segment.shape[1])
        return patches, timestamps


def actionsenseqa_native_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for ActionSenseTemplatedQADataset_NativeRate.

    Handles multi-stream native rate data where patches is a dict:
        {stream_name: (P, T_native, D_stream)}

    Pads sequences to max P in batch and creates a pad mask.
    Filters out None samples (from samples with no valid streams).
    """
    # Filter out None samples (from samples with no valid sensor streams)
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return {}

    # Multi-stream native rate: patches is dict {stream_name: (P, T_stream, D_stream)}
    first_patches = batch[0]["patches"]
    stream_names = list(first_patches.keys())

    # Handle edge case: empty patches dict
    if len(stream_names) == 0:
        raise ValueError(
            f"Empty patches dict in batch! This means no valid sensor streams were found.\n"
            f"Check that the data files exist and contain valid sensor data.\n"
            f"Sample metadata: {batch[0].get('metadata', {})}"
        )

    # Get P from first stream (all streams have same P due to alignment check)
    P_list = [item["patches"][stream_names[0]].shape[0] for item in batch]
    P_max = max(P_list)

    # Prepare output dicts for each stream
    patches_out_dict = {stream_name: [] for stream_name in stream_names}
    padmask_out = []

    for item in batch:
        patches_dict = item["patches"]
        P_i = patches_dict[stream_names[0]].shape[0]
        pad = P_max - P_i

        # Pad each stream independently
        for stream_name in stream_names:
            stream_patches = patches_dict[stream_name]  # (P_i, T_stream, D_stream)
            T_stream = stream_patches.shape[1]
            D_stream = stream_patches.shape[2]

            if pad > 0:
                pad_tensor = torch.zeros((pad, T_stream, D_stream), dtype=stream_patches.dtype)
                patches_pad = torch.cat([pad_tensor, stream_patches], dim=0)
            else:
                patches_pad = stream_patches

            patches_out_dict[stream_name].append(patches_pad)

        # Create pad mask (same for all streams)
        if pad > 0:
            mask = torch.cat([torch.zeros(pad, dtype=torch.bool), torch.ones(P_i, dtype=torch.bool)], dim=0)
        else:
            mask = torch.ones(P_i, dtype=torch.bool)
        padmask_out.append(mask)

    # Stack each stream
    patches_batched = {
        stream_name: torch.stack(patches_out_dict[stream_name], dim=0)
        for stream_name in stream_names
    }

    # Collect questions, answers, metadata
    questions, answers = [], []
    metadata_list: Dict[str, List[Any]] = {
        "subject": [],
        "split": [],
        "activity_indices": [],
        "activity_names": [],
        "question_type": [],
        "stream_info": [],
        "patch_duration_s": [],
    }

    for item in batch:
        questions.append(item["question"])
        answers.append(item["answer"])

        md = item["metadata"]
        for key in metadata_list:
            metadata_list[key].append(md.get(key))

    return {
        "patches": patches_batched,  # dict {stream_name: (B, P, T_stream, D_stream)}
        "pad_mask": torch.stack(padmask_out, dim=0),  # (B, P)
        "questions": questions,
        "answers": answers,
        "metadata": metadata_list,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug and visualize the ActionSenseTemplatedQADataset_NativeRate.")
    parser.add_argument("--qa_path", type=str, default="data/actionsenseqa_native/data/qa_pairs_templated.jsonl")
    parser.add_argument("--manifest_path", type=str, default="data/actionsenseqa_native/data/manifest.csv")
    parser.add_argument("--base_dir", type=str, default="data/actionsenseqa_native/data")
    parser.add_argument("--outdir", type=str, default="debug/templated_dataset_native_rate")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of random samples to visualize.")
    parser.add_argument("--patch_duration_s", type=float, default=5.0, help="Patch temporal duration in seconds")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Initializing ActionSenseTemplatedQADataset_NativeRate...")
    try:
        dataset = ActionSenseTemplatedQADataset_NativeRate(
            base_dir=args.base_dir,
            qa_jsonl_path=args.qa_path,
            manifest_csv_path=args.manifest_path,
            split="train",
            val_ratio=0.2,
            split_seed=args.seed,
            patch_duration_s=args.patch_duration_s,
        )
        print(f"Dataset loaded with {len(dataset)} samples.")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if len(dataset) == 0:
        print("[WARN] Dataset is empty. Exiting.")
        sys.exit(0)

    num_to_plot = min(args.num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_to_plot)

    for i, idx in enumerate(indices):
        print(f"\n--- Preparing sample {i+1}/{num_to_plot} (index: {idx}) ---")
        try:
            sample = dataset[idx]
        except Exception as e:
            print(f"[ERROR] Failed to load sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

        metadata = sample["metadata"]
        patches_dict = sample["patches"]

        print(f"  Question Type: {metadata['question_type']}")
        print(f"  Question: {sample['question']}")
        print(f"  Answer: {sample['answer']}")
        print(f"  Patch Duration: {metadata['patch_duration_s']}s")
        print(f"  Available Streams: {list(patches_dict.keys())}")

        for stream_name, patches in patches_dict.items():
            stream_info = metadata["stream_info"][stream_name]
            print(f"    {stream_name}: shape={tuple(patches.shape)}, "
                  f"rate={stream_info['rate_hz']:.1f} Hz, "
                  f"T_native={stream_info['T_native']}, "
                  f"channels={stream_info['num_channels']}")

        # --- Plotting --- #
        num_streams = len(patches_dict)
        if num_streams == 0:
            print("  No streams to plot. Skipping.")
            continue

        fig, axes = plt.subplots(num_streams, 1, figsize=(14, 4 * num_streams), squeeze=False)
        axes = axes.flatten()

        fig.suptitle(f"Q: {sample['question']}\nA: {sample['answer']}", fontsize=12, wrap=True)

        for ax_idx, (stream_name, patches) in enumerate(patches_dict.items()):
            ax = axes[ax_idx]
            patches_np = patches.numpy()  # (P, T_native, D)

            if patches_np.size == 0:
                ax.set_title(f"{stream_name} (No Data)")
                continue

            # Flatten patches into continuous sequence
            flat_sequence = patches_np.reshape(-1, patches_np.shape[-1])  # (P*T_native, D)

            stream_info = metadata["stream_info"][stream_name]
            rate_hz = stream_info["rate_hz"]

            # Create time axis in seconds
            time_axis = np.arange(flat_sequence.shape[0]) / rate_hz

            # Plot first 10 channels
            for d in range(min(10, flat_sequence.shape[1])):
                ax.plot(time_axis, flat_sequence[:, d], label=f"ch{d}", alpha=0.7)

            ax.set_title(f"{stream_name} ({stream_info['rate_hz']:.1f} Hz, {stream_info['num_channels']} channels)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Sensor Value")
            ax.legend(loc="upper right", ncol=5, fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        save_path = os.path.join(args.outdir, f"sample_{idx}.png")
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"  Plot saved to {save_path}")

    print(f"\n[DONE] Visualization complete. Outputs saved to {args.outdir}")
