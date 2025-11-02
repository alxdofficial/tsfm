"""
Chronos-2 specific dataset for ActionSense QA tasks.

Key features:
- Uses BOTH ARM joint rotations: left + right (shoulder, forearm, wrist)
- 18 channels total: 6 joints × 3 axes (X, Y, Z rotations)
- Native sampling rate: 60Hz (joints)
- Extracts from START of each activity
- Dynamic padding: pads to max length in batch (up to 2016, divisible by 48)
- Output: (D=18, T≤2016) ready for Chronos-2 multivariate input
"""

import os
import random
import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ActionSenseChronos2QA(Dataset):
    """
    Chronos-2 specific dataset for ActionSense QA training.

    Bilateral arm joint rotations:
    1. 18 channels: both arms (left + right) × 3 joints (shoulder, forearm, wrist) × 3 axes (X,Y,Z)
    2. Native sampling rate: 60Hz (joints only)
    3. Extract from START of activity (as much as available)
    4. Dynamic padding: pad to max length in batch (up to 2016, rounded down to nearest 48)
    5. Output format: (D=18, T≤2016) for Chronos-2 multivariate input
    """

    # Arm joint indices (both left and right arms)
    # Left arm kinematic chain
    LEFT_SHOULDER_IDX = [21, 22, 23]   # Joint 7: X, Y, Z rotation
    LEFT_FOREARM_IDX = [27, 28, 29]    # Joint 9: X, Y, Z rotation
    LEFT_WRIST_IDX = [30, 31, 32]      # Joint 10: X, Y, Z rotation

    # Right arm kinematic chain
    RIGHT_SHOULDER_IDX = [24, 25, 26]  # Joint 8: X, Y, Z rotation
    RIGHT_FOREARM_IDX = [33, 34, 35]   # Joint 11: X, Y, Z rotation (adjusted)
    RIGHT_WRIST_IDX = [36, 37, 38]     # Joint 12: X, Y, Z rotation (adjusted)

    # Combined arm indices (18 total)
    ARM_JOINT_INDICES = (
        LEFT_SHOULDER_IDX + LEFT_FOREARM_IDX + LEFT_WRIST_IDX +
        RIGHT_SHOULDER_IDX + RIGHT_FOREARM_IDX + RIGHT_WRIST_IDX
    )

    # Constants
    SAMPLING_RATE = 60.0          # Hz (joints native rate)
    MAX_TIMESTEPS = 2016          # Maximum timesteps (divisible by 48)
    TOTAL_CHANNELS = 18           # 6 joints × 3 axes

    def __init__(
        self,
        base_dir: str = "data/actionsenseqa_native/data",
        qa_jsonl_path: str = "data/actionsenseqa_native/data/qa_pairs_templated.jsonl",
        manifest_csv_path: str = "data/actionsenseqa_native/data/manifest.csv",
        split: str = "train",
        val_ratio: float = 0.2,
        split_seed: int = 42,
        target_fps: int = 30,
        window_seconds: float = 10.0,
        random_window: bool = True,
        log_mode: str = "info",
    ):
        """
        Args:
            base_dir: Root directory for sensor CSVs
            qa_jsonl_path: Path to QA pairs JSONL
            manifest_csv_path: Path to manifest CSV
            split: 'train' or 'val'
            val_ratio: Validation split ratio
            split_seed: Random seed for train/val split
            target_fps: Target sampling rate (downsample from native 60Hz)
            window_seconds: Window duration in seconds
            random_window: If True, randomly select window offset; if False, use start
            log_mode: 'info', 'debug', or 'silent'
        """
        assert split in {"train", "val"}, "split must be 'train' or 'val'"

        self.base_dir = base_dir
        self.split = split
        self.log_mode = log_mode

        # Window sampling configuration
        self.target_fps = target_fps
        self.window_seconds = window_seconds
        self.window_timesteps = int(window_seconds * target_fps)
        self.random_window = random_window
        self.downsample_factor = int(self.SAMPLING_RATE / target_fps)

        # Load QA pairs
        all_records = self._load_qa_records(qa_jsonl_path)
        if not all_records:
            raise ValueError(f"No QA records loaded from {qa_jsonl_path}")

        # Load manifest
        self.manifest_map = self._load_manifest(manifest_csv_path)

        # Split train/val
        self.records = self._split_records(all_records, split, val_ratio, split_seed)

        # Build cache of joint streams only
        sensor_paths = self._get_required_sensor_paths(self.records)
        self.sensor_cache = self._load_joint_streams(sensor_paths)

        self._log(
            f"[INFO] ActionSenseChronos2QA split={split}: {len(self.records)} QA pairs loaded.",
            level="info",
        )
        self._log(
            f"[INFO] Using {self.TOTAL_CHANNELS} arm joint channels (bilateral shoulder/forearm/wrist)",
            level="info",
        )
        self._log(
            f"[INFO] Native rate: {self.SAMPLING_RATE}Hz → Target: {self.target_fps}Hz (downsample factor: {self.downsample_factor}x)",
            level="info",
        )
        self._log(
            f"[INFO] Window: {self.window_seconds}s = {self.window_timesteps} timesteps at {self.target_fps}Hz, random={self.random_window}",
            level="info",
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            Dict with:
                - continuous_stream: (D=18, T_actual) FloatTensor (NO padding yet)
                - question: str
                - answer: str
                - metadata: dict with activity info
        """
        record = self.records[idx]

        # Concatenate joint streams from all activities in this QA pair
        all_joints = []

        for activity_index in record["activity_indices"]:
            manifest_key = (record["subject"], record["split"], activity_index)

            if manifest_key not in self.manifest_map:
                self._log(
                    f"[WARN] Activity key {manifest_key} not in manifest, skipping",
                    level="warn"
                )
                continue

            joints_csv = self.manifest_map[manifest_key].get("joints_csv")
            if not joints_csv:
                continue

            cache_key = os.path.join(self.base_dir, joints_csv)
            if cache_key not in self.sensor_cache:
                self._log(f"[WARN] Cache missing for {cache_key}", level="warn")
                continue

            joint_data = self.sensor_cache[cache_key]  # (T, 18)
            all_joints.append(joint_data)

        if not all_joints:
            # Fallback: return minimal zeros if no valid data
            self._log(
                f"[WARN] No valid joint data for sample {idx}, returning zeros",
                level="warn"
            )
            segment_final = torch.zeros(self.TOTAL_CHANNELS, 48)  # Minimum valid length
        else:
            # Concatenate temporally across activities
            full_stream = np.concatenate(all_joints, axis=0)  # (T_total, 18) at native 60Hz

            # Downsample to target FPS (e.g., 60Hz → 30Hz)
            # All 18 channels stay aligned since they share the same timestamps
            if self.downsample_factor > 1:
                full_stream = full_stream[::self.downsample_factor]  # Take every Nth sample

            # Extract window (random offset or from start)
            T_available = full_stream.shape[0]

            if T_available >= self.window_timesteps:
                if self.random_window:
                    # Random window selection for data augmentation
                    max_offset = T_available - self.window_timesteps
                    start_offset = np.random.randint(0, max_offset + 1)
                    segment = full_stream[start_offset:start_offset + self.window_timesteps]
                else:
                    # Fixed window from start (deterministic)
                    segment = full_stream[:self.window_timesteps]
            else:
                # If shorter than target window, use all available (will pad in collate)
                segment = full_stream

            # Transpose to (D, T) format
            segment_final = torch.from_numpy(segment.T).float()  # (18, T_actual)

        return {
            "continuous_stream": segment_final,  # (18, T_actual) - variable length!
            "question": record["question"],
            "answer": record["answer"],
            "metadata": {
                "subject": record["subject"],
                "activity_names": record.get("activity_names", []),
                "activity_indices": record["activity_indices"],
                "question_type": record.get("question_type", "unknown"),
            },
        }

    def _load_qa_records(self, qa_jsonl_path: str) -> List[Dict[str, Any]]:
        """Load QA pairs from JSONL file."""
        if not os.path.exists(qa_jsonl_path):
            raise FileNotFoundError(f"QA JSONL not found: {qa_jsonl_path}")

        records = []
        with open(qa_jsonl_path, 'r') as f:
            for line in f:
                records.append(json.loads(line))

        return records

    def _load_manifest(self, manifest_csv_path: str) -> Dict:
        """Load manifest CSV into lookup dict."""
        if not os.path.exists(manifest_csv_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_csv_path}")

        df = pd.read_csv(manifest_csv_path)
        manifest_map = {}

        for _, row in df.iterrows():
            key = (row["subject"], row["split"], row["activity_index"])
            manifest_map[key] = row.to_dict()

        return manifest_map

    def _split_records(
        self,
        all_records: List[Dict],
        split: str,
        val_ratio: float,
        seed: int
    ) -> List[Dict]:
        """Split records into train/val."""
        rng = random.Random(seed)
        shuffled = all_records.copy()
        rng.shuffle(shuffled)

        n_val = int(len(shuffled) * val_ratio)

        if split == "val":
            return shuffled[:n_val]
        else:
            return shuffled[n_val:]

    def _get_required_sensor_paths(self, records: List[Dict]) -> List[str]:
        """Extract unique joint CSV paths needed for these records."""
        paths = set()

        for record in records:
            for activity_index in record["activity_indices"]:
                key = (record["subject"], record["split"], activity_index)
                if key in self.manifest_map:
                    joints_csv = self.manifest_map[key].get("joints_csv")
                    if joints_csv:
                        full_path = os.path.join(self.base_dir, joints_csv)
                        paths.add(full_path)

        return list(paths)

    def _load_joint_streams(self, sensor_paths: List[str]) -> Dict[str, np.ndarray]:
        """
        Load joint streams and extract arm channels only.

        Args:
            sensor_paths: List of joint CSV paths

        Returns:
            cache: Dict mapping path -> (T, 18) numpy array of arm joint channels
        """
        cache = {}

        for path in sensor_paths:
            if not os.path.exists(path):
                self._log(f"[WARN] Joints CSV not found: {path}", level="warn")
                continue

            try:
                df = pd.read_csv(path)

                # Extract arm joint channels only
                arm_cols = [f"joints_{i}" for i in self.ARM_JOINT_INDICES]

                # Check if columns exist
                missing_cols = [col for col in arm_cols if col not in df.columns]
                if missing_cols:
                    self._log(
                        f"[WARN] Missing arm joint columns in {path}: {missing_cols}",
                        level="warn"
                    )
                    continue

                arm_data = df[arm_cols].values  # (T, 18)
                cache[path] = arm_data

            except Exception as e:
                self._log(f"[ERROR] Failed to load {path}: {e}", level="warn")
                continue

        self._log(f"[INFO] Loaded {len(cache)} joint streams (arm channels only)", level="info")
        return cache

    def _log(self, message: str, level: str = "info") -> None:
        """Log message based on log_mode."""
        if level == "error" or self.log_mode == "debug":
            print(message)
        elif level == "warn" and self.log_mode != "silent":
            print(message)
        elif level == "info" and self.log_mode in {"info", "debug"}:
            print(message)


def chronos2_qa_collate(batch: List[Dict]) -> Dict:
    """
    Collate function for Chronos-2 QA dataset with dynamic padding.

    Dynamically pads samples to the maximum length in the batch:
    1. Find max length across all samples in batch
    2. Round down to nearest multiple of 48 (for Chronos-2 patching)
    3. Cap at 2016 (max valid length divisible by 48)
    4. Left-pad all samples to this computed length

    Args:
        batch: List of dict, each with:
            - continuous_stream: (D=18, T_actual) variable length
            - question: str
            - answer: str
            - metadata: dict

    Returns:
        Batched dict with:
            - continuous_stream: (B, D=18, T_padded) where T_padded is batch-specific
            - questions: List[str]
            - answers: List[str]
            - metadata: Dict[str, List]
    """
    # Filter out None samples (if any)
    batch = [item for item in batch if item is not None]

    if not batch:
        # Return empty batch
        return {
            "continuous_stream": torch.zeros(0, 18, 48),
            "questions": [],
            "answers": [],
            "metadata": {},
        }

    # Find max length in batch
    max_len = max(item["continuous_stream"].shape[1] for item in batch)

    # Round down to nearest multiple of 16 (Chronos-2 patch size)
    # Cap at reasonable maximum to prevent OOM from occasional long sequences
    MAX_ALLOWED_TIMESTEPS = 512  # Hard cap to prevent OOM
    max_len = min(max_len, MAX_ALLOWED_TIMESTEPS)

    target_len = (max_len // 16) * 16
    target_len = max(target_len, 16)  # Minimum 16 timesteps (1 patch)

    # Left-pad all samples to target_len
    padded_streams = []
    for item in batch:
        stream = item["continuous_stream"]  # (18, T_actual)
        T_actual = stream.shape[1]

        if T_actual < target_len:
            # Left-pad (prepend zeros)
            pad_width = target_len - T_actual
            stream = torch.nn.functional.pad(stream, (pad_width, 0), value=0.0)
        elif T_actual > target_len:
            # Truncate to target_len (take first target_len timesteps)
            stream = stream[:, :target_len]

        padded_streams.append(stream)

    streams = torch.stack(padded_streams)  # (B, 18, target_len)
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]

    # Aggregate metadata
    metadata = {
        "subject": [item["metadata"]["subject"] for item in batch],
        "activity_names": [item["metadata"]["activity_names"] for item in batch],
        "activity_indices": [item["metadata"]["activity_indices"] for item in batch],
        "question_type": [item["metadata"]["question_type"] for item in batch],
    }

    return {
        "continuous_stream": streams,
        "questions": questions,
        "answers": answers,
        "metadata": metadata,
    }
