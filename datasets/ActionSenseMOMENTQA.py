"""
MOMENT-specific dataset for ActionSense QA tasks.

Key features:
- Uses BOTH ARM joint rotations: left + right (shoulder, forearm, wrist)
- 18 channels total: 6 joints × 3 axes (X, Y, Z rotations)
- Fixed 512 timesteps (MOMENT requirement)
- Adaptive sampling strategy (inherited from MOMENT CLS):
  1. If T < 512: use all data and left-pad
  2. If T >= 512 and equiv_rate >= 10Hz: uniform sampling
  3. If T >= 512 and equiv_rate < 10Hz: windowing at 10Hz cap
- Handles multi-activity QA pairs by concatenating activities temporally
- Output: (D=18, T=512) ready for MOMENT multivariate input
"""

import os
import random
import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ActionSenseMOMENTQA(Dataset):
    """
    MOMENT-specific dataset for ActionSense QA training.

    Bilateral arm joint rotations:
    1. 18 channels: both arms (left + right) × 3 joints (shoulder, forearm, wrist) × 3 axes (X,Y,Z)
    2. Native sampling rate: 60Hz (joints only)
    3. Adaptive sampling to exactly 512 timesteps (same as MOMENT CLS)
    4. Output format: (D=18, T=512) for MOMENT multivariate input
    """

    # Arm joint indices (both left and right arms)
    # Left arm kinematic chain
    LEFT_SHOULDER_IDX = [21, 22, 23]   # Joint 7: X, Y, Z rotation
    LEFT_FOREARM_IDX = [27, 28, 29]    # Joint 9: X, Y, Z rotation
    LEFT_WRIST_IDX = [30, 31, 32]      # Joint 10: X, Y, Z rotation

    # Right arm kinematic chain
    RIGHT_SHOULDER_IDX = [24, 25, 26]  # Joint 8: X, Y, Z rotation
    RIGHT_FOREARM_IDX = [33, 34, 35]   # Joint 11: X, Y, Z rotation
    RIGHT_WRIST_IDX = [36, 37, 38]     # Joint 12: X, Y, Z rotation

    # Combined arm indices (18 total)
    ARM_JOINT_INDICES = (
        LEFT_SHOULDER_IDX + LEFT_FOREARM_IDX + LEFT_WRIST_IDX +
        RIGHT_SHOULDER_IDX + RIGHT_FOREARM_IDX + RIGHT_WRIST_IDX
    )

    # Constants
    SAMPLING_RATE = 60.0          # Hz (joints native rate)
    FIXED_TIMESTEPS = 512         # MOMENT requirement
    TOTAL_CHANNELS = 18           # 6 joints × 3 axes

    def __init__(
        self,
        base_dir: str = "data/actionsenseqa_native/data",
        qa_jsonl_path: str = "data/actionsenseqa_native/data/qa_pairs_templated.jsonl",
        manifest_csv_path: str = "data/actionsenseqa_native/data/manifest.csv",
        split: str = "train",
        val_ratio: float = 0.2,
        split_seed: int = 42,
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
            random_window: If True, randomly select window offset when using windowing; if False, use start
            log_mode: 'info', 'debug', or 'silent'
        """
        assert split in {"train", "val"}, "split must be 'train' or 'val'"

        self.base_dir = base_dir
        self.split = split
        self.log_mode = log_mode
        self.random_window = random_window

        # Load QA pairs
        all_records = self._load_qa_records(qa_jsonl_path)
        if not all_records:
            raise ValueError(f"No QA records loaded from {qa_jsonl_path}")

        # Load manifest
        self.manifest_map = self._load_manifest(manifest_csv_path)

        # Split QA records into train/val
        self.records = self._split_records(all_records, split, val_ratio, split_seed)

        # Build cache of joint streams
        sensor_paths = self._get_required_sensor_paths(self.records)
        self.sensor_cache = self._load_joint_streams(sensor_paths)

        self._log(
            f"[INFO] ActionSenseMOMENTQA split={split}: {len(self.records)} QA pairs loaded.",
            level="info",
        )
        self._log(
            f"[INFO] Using {self.TOTAL_CHANNELS} arm joint channels (bilateral shoulder/forearm/wrist)",
            level="info",
        )
        self._log(
            f"[INFO] Adaptive sampling to {self.FIXED_TIMESTEPS} timesteps:",
            level="info",
        )
        self._log(
            f"       - If T < 512: use all and pad",
            level="info",
        )
        self._log(
            f"       - If T >= 512 and equiv_rate >= 10Hz: uniform sample",
            level="info",
        )
        self._log(
            f"       - If T >= 512 and equiv_rate < 10Hz: window at 10Hz cap (random={self.random_window})",
            level="info",
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            Dict with:
                - continuous_stream: (D=18, T=512) FloatTensor
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

            joint_data = self.sensor_cache[cache_key]  # (T, 18) at native 60Hz
            all_joints.append(joint_data)

        if not all_joints:
            # Fallback: return zeros if no valid data
            self._log(
                f"[WARN] No valid joint data for sample {idx}, returning zeros",
                level="warn"
            )
            segment_final = torch.zeros(self.TOTAL_CHANNELS, self.FIXED_TIMESTEPS)
        else:
            # Concatenate temporally across activities
            full_stream = np.concatenate(all_joints, axis=0)  # (T_total, 18) at native 60Hz
            T_available = full_stream.shape[0]

            # Apply adaptive sampling to fit into 512 timesteps
            segment_final = self._adaptive_sample(full_stream, T_available)

        return {
            "continuous_stream": segment_final,  # (18, 512) - fixed length!
            "question": record["question"],
            "answer": record["answer"],
            "metadata": {
                "subject": record["subject"],
                "activity_names": record.get("activity_names", []),
                "question_type": record.get("question_type", "unknown"),
            },
        }

    def _adaptive_sample(self, joint_data: np.ndarray, T_available: int) -> torch.Tensor:
        """
        Adaptive sampling strategy (same as ActionSenseMOMENTCLS).

        Args:
            joint_data: (T, 18) numpy array
            T_available: Number of available timesteps

        Returns:
            segment: (18, 512) torch tensor
        """
        MIN_SAMPLING_RATE = 10.0  # Hz - minimum acceptable sampling rate

        if T_available < self.FIXED_TIMESTEPS:
            # Case 1: Session shorter than 512 timesteps
            # Use all available data and left-pad with zeros
            segment = joint_data
            pad_len = self.FIXED_TIMESTEPS - T_available
            segment = np.pad(segment, ((pad_len, 0), (0, 0)), mode='constant', constant_values=0)
            segment_final = torch.from_numpy(segment.T).float()  # (18, 512)

        else:
            # Calculate what the equivalent sampling rate would be for uniform sampling
            equivalent_fps = (T_available / self.FIXED_TIMESTEPS) * self.SAMPLING_RATE

            if equivalent_fps >= MIN_SAMPLING_RATE:
                # Case 2: Uniform sampling yields acceptable rate (>= 10Hz)
                # Uniformly sample 512 indices from T_available
                indices = np.linspace(0, T_available - 1, self.FIXED_TIMESTEPS, dtype=int)
                segment = joint_data[indices]  # (512, 18)
                segment_final = torch.from_numpy(segment.T).float()  # (18, 512)

            else:
                # Case 3: Uniform sampling would be < 10Hz
                # Use windowing: cap at 10Hz, extract 512 timesteps
                window_samples_at_10hz = int((self.FIXED_TIMESTEPS / MIN_SAMPLING_RATE) * self.SAMPLING_RATE)
                # window_samples_at_10hz = (512 / 10) * 60 = 3072 samples

                if T_available >= window_samples_at_10hz:
                    # Extract window (random or from start)
                    if self.random_window:
                        max_offset = T_available - window_samples_at_10hz
                        start_offset = np.random.randint(0, max_offset + 1)
                        window = joint_data[start_offset:start_offset + window_samples_at_10hz]
                    else:
                        window = joint_data[:window_samples_at_10hz]

                    # Downsample window to 512 timesteps
                    indices = np.linspace(0, window_samples_at_10hz - 1, self.FIXED_TIMESTEPS, dtype=int)
                    segment = window[indices]  # (512, 18)
                    segment_final = torch.from_numpy(segment.T).float()  # (18, 512)
                else:
                    # Fallback: session too short even for 10Hz window
                    # Use all available and uniformly sample
                    indices = np.linspace(0, T_available - 1, self.FIXED_TIMESTEPS, dtype=int)
                    segment = joint_data[indices]  # (512, 18)
                    segment_final = torch.from_numpy(segment.T).float()  # (18, 512)

        return segment_final

    def _load_qa_records(self, qa_jsonl_path: str) -> List[Dict]:
        """Load QA pairs from JSONL file."""
        if not os.path.exists(qa_jsonl_path):
            raise FileNotFoundError(f"QA JSONL not found: {qa_jsonl_path}")

        records = []
        with open(qa_jsonl_path, "r") as f:
            for line in f:
                record = json.loads(line.strip())
                records.append(record)

        self._log(f"[INFO] Loaded {len(records)} QA records from {qa_jsonl_path}", level="info")
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
        """Split QA records into train/val."""
        rng = random.Random(seed)
        shuffled = all_records.copy()
        rng.shuffle(shuffled)

        n_val = int(len(shuffled) * val_ratio)

        if split == "val":
            return shuffled[:n_val]
        else:
            return shuffled[n_val:]

    def _get_required_sensor_paths(self, records: List[Dict]) -> List[str]:
        """Extract unique joint CSV paths needed for these QA records."""
        paths = set()

        for record in records:
            for activity_index in record["activity_indices"]:
                manifest_key = (record["subject"], record["split"], activity_index)
                if manifest_key in self.manifest_map:
                    joints_csv = self.manifest_map[manifest_key].get("joints_csv")
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


def moment_qa_collate(batch: List[Dict]) -> Dict:
    """
    Collate function for MOMENT QA dataset.

    Since all samples are already fixed at 512 timesteps, we just stack them.

    Args:
        batch: List of dict, each with:
            - continuous_stream: (D=18, T=512) fixed length
            - question: str
            - answer: str
            - metadata: dict

    Returns:
        Batched dict with:
            - continuous_stream: (B, D=18, T=512)
            - question: List[str]
            - answer: List[str]
            - metadata: Dict[str, List]
    """
    # Filter out None samples (if any)
    batch = [item for item in batch if item is not None]

    if not batch:
        # Return empty batch
        return {
            "continuous_stream": torch.zeros(0, 18, 512),
            "question": [],
            "answer": [],
            "metadata": {},
        }

    # Stack all samples (all already 512 timesteps)
    streams = torch.stack([item["continuous_stream"] for item in batch])  # (B, 18, 512)
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]

    # Aggregate metadata
    metadata = {
        "subject": [item["metadata"]["subject"] for item in batch],
        "activity_names": [item["metadata"]["activity_names"] for item in batch],
        "question_type": [item["metadata"]["question_type"] for item in batch],
    }

    return {
        "continuous_stream": streams,
        "question": questions,
        "answer": answers,
        "metadata": metadata,
    }
