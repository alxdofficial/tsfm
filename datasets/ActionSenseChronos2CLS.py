"""
Chronos-2 specific dataset for ActionSense activity classification.

Key features:
- Uses BOTH ARM joint rotations: left + right (shoulder, forearm, wrist)
- 18 channels total: 6 joints × 3 axes (X, Y, Z rotations)
- Native sampling rate: 60Hz (joints) by default
- Optional downsampling via target_fps parameter
- Extracts fixed-duration windows from activities
- Returns activity labels for classification
- Output: (D=18, T) ready for Chronos-2 multivariate input
"""

import os
import random
import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ActionSenseChronos2CLS(Dataset):
    """
    Chronos-2 specific dataset for ActionSense activity classification.

    Bilateral arm joint rotations:
    1. 18 channels: both arms (left + right) × 3 joints (shoulder, forearm, wrist) × 3 axes (X,Y,Z)
    2. Native sampling rate: 60Hz (joints only)
    3. Extract from activity windows (fixed duration)
    4. Dynamic padding: pad to max length in batch
    5. Output format: (D=18, T) for Chronos-2 multivariate input
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
    MAX_TIMESTEPS = 2016          # Maximum timesteps (divisible by 48) at native rate
    TOTAL_CHANNELS = 18           # 6 joints × 3 axes

    def __init__(
        self,
        base_dir: str = "data/actionsenseqa_native/data",
        manifest_csv_path: str = "data/actionsenseqa_native/data/manifest.csv",
        split: str = "train",
        val_ratio: float = 0.2,
        split_seed: int = 42,
        target_fps: int = None,  # None = native 60Hz; specify for downsampling
        window_seconds: float = 10.0,
        random_window: bool = True,
        log_mode: str = "info",
    ):
        """
        Args:
            base_dir: Root directory for sensor CSVs
            manifest_csv_path: Path to manifest CSV (must have 'activity_name' column)
            split: 'train' or 'val'
            val_ratio: Validation split ratio
            split_seed: Random seed for train/val split
            target_fps: Target sampling rate (None=native 60Hz, or specify for downsampling)
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
        self.random_window = random_window

        if target_fps is None:
            self.downsample_factor = 1  # No downsampling
            self.window_timesteps = int(window_seconds * self.SAMPLING_RATE)
            self._log(f"[INFO] Using NATIVE sampling rate: {self.SAMPLING_RATE}Hz", level="info")
        else:
            self.downsample_factor = int(self.SAMPLING_RATE / target_fps)
            self.window_timesteps = int(window_seconds * target_fps)
            self._log(
                f"[INFO] Downsampling: {self.SAMPLING_RATE}Hz → {target_fps}Hz (factor: {self.downsample_factor}x)",
                level="info"
            )

        # Load manifest and build activity mapping
        self.manifest_map = self._load_manifest(manifest_csv_path)

        # Build activity_to_id from manifest
        all_activities = set()
        for key, row in self.manifest_map.items():
            activity = row.get("activity_name")
            if activity and pd.notna(activity):
                all_activities.add(activity)

        unique_activities = sorted(list(all_activities))
        if not unique_activities:
            raise ValueError(f"No activity labels found in manifest: {manifest_csv_path}")

        self.activity_to_id: Dict[str, int] = {name: i for i, name in enumerate(unique_activities)}
        self.id_to_activity: List[str] = unique_activities
        self.num_classes: int = len(self.activity_to_id)

        # Split manifest records into train/val
        all_keys = list(self.manifest_map.keys())
        self.record_keys = self._split_records(all_keys, split, val_ratio, split_seed)

        # Build cache of joint streams only
        sensor_paths = self._get_required_sensor_paths(self.record_keys)
        self.sensor_cache = self._load_joint_streams(sensor_paths)

        self._log(
            f"[INFO] ActionSenseChronos2CLS split={split}: {len(self.record_keys)} samples loaded.",
            level="info",
        )
        self._log(
            f"[INFO] Found {self.num_classes} activity classes: {self.id_to_activity}",
            level="info",
        )
        self._log(
            f"[INFO] Using {self.TOTAL_CHANNELS} arm joint channels (bilateral shoulder/forearm/wrist)",
            level="info",
        )
        self._log(
            f"[INFO] Window: {self.window_seconds}s = {self.window_timesteps} timesteps, random={self.random_window}",
            level="info",
        )

    def __len__(self) -> int:
        return len(self.record_keys)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            Dict with:
                - continuous_stream: (D=18, T_actual) FloatTensor (NO padding yet)
                - activity_id: int (class label)
                - metadata: dict with activity info
        """
        manifest_key = self.record_keys[idx]
        manifest_row = self.manifest_map[manifest_key]

        # Get activity label
        activity_name = manifest_row.get("activity_name")
        if not activity_name or pd.isna(activity_name):
            # Fallback to zeros if no label
            self._log(
                f"[WARN] No activity label for sample {idx}, returning zeros",
                level="warn"
            )
            segment_final = torch.zeros(self.TOTAL_CHANNELS, 48)
            activity_id = 0
        else:
            activity_id = self.activity_to_id.get(activity_name, 0)

            # Load joint data
            joints_csv = manifest_row.get("joints_csv")
            if not joints_csv:
                self._log(f"[WARN] No joints CSV for sample {idx}", level="warn")
                segment_final = torch.zeros(self.TOTAL_CHANNELS, 48)
            else:
                cache_key = os.path.join(self.base_dir, joints_csv)
                if cache_key not in self.sensor_cache:
                    self._log(f"[WARN] Cache missing for {cache_key}", level="warn")
                    segment_final = torch.zeros(self.TOTAL_CHANNELS, 48)
                else:
                    joint_data = self.sensor_cache[cache_key]  # (T, 18)

                    # Downsample to target FPS (if specified)
                    if self.downsample_factor > 1:
                        joint_data = joint_data[::self.downsample_factor]

                    # Extract window (random offset or from start)
                    T_available = joint_data.shape[0]

                    if T_available >= self.window_timesteps:
                        if self.random_window:
                            # Random window selection for data augmentation
                            max_offset = T_available - self.window_timesteps
                            start_offset = np.random.randint(0, max_offset + 1)
                            segment = joint_data[start_offset:start_offset + self.window_timesteps]
                        else:
                            # Fixed window from start (deterministic)
                            segment = joint_data[:self.window_timesteps]
                    else:
                        # If shorter than target window, use all available (will pad in collate)
                        segment = joint_data

                    # Transpose to (D, T) format
                    segment_final = torch.from_numpy(segment.T).float()  # (18, T_actual)

        return {
            "continuous_stream": segment_final,  # (18, T_actual) - variable length!
            "activity_id": activity_id,
            "metadata": {
                "subject": manifest_key[0],
                "split_name": manifest_key[1],
                "activity_index": manifest_key[2],
                "activity_name": activity_name if activity_name and pd.notna(activity_name) else "unknown",
            },
        }

    def _load_manifest(self, manifest_csv_path: str) -> Dict:
        """Load manifest CSV into lookup dict."""
        if not os.path.exists(manifest_csv_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_csv_path}")

        df = pd.read_csv(manifest_csv_path)

        # Check for activity_name column
        if "activity_name" not in df.columns:
            raise ValueError(
                f"Manifest CSV must have 'activity_name' column for classification. "
                f"Found columns: {list(df.columns)}"
            )

        manifest_map = {}
        for _, row in df.iterrows():
            key = (row["subject"], row["split"], row["activity_index"])
            manifest_map[key] = row.to_dict()

        return manifest_map

    def _split_records(
        self,
        all_keys: List[tuple],
        split: str,
        val_ratio: float,
        seed: int
    ) -> List[tuple]:
        """Split records into train/val."""
        rng = random.Random(seed)
        shuffled = all_keys.copy()
        rng.shuffle(shuffled)

        n_val = int(len(shuffled) * val_ratio)

        if split == "val":
            return shuffled[:n_val]
        else:
            return shuffled[n_val:]

    def _get_required_sensor_paths(self, record_keys: List[tuple]) -> List[str]:
        """Extract unique joint CSV paths needed for these records."""
        paths = set()

        for key in record_keys:
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


def chronos2_cls_collate(batch: List[Dict]) -> Dict:
    """
    Collate function for Chronos-2 CLS dataset with dynamic padding.

    Dynamically pads samples to the maximum length in the batch:
    1. Find max length across all samples in batch
    2. Round down to nearest multiple of 16 (for Chronos-2 patching)
    3. Cap at reasonable maximum to prevent OOM
    4. Left-pad all samples to this computed length

    Args:
        batch: List of dict, each with:
            - continuous_stream: (D=18, T_actual) variable length
            - activity_id: int
            - metadata: dict

    Returns:
        Batched dict with:
            - continuous_stream: (B, D=18, T_padded) where T_padded is batch-specific
            - activity_ids: (B,) LongTensor
            - metadata: Dict[str, List]
    """
    # Filter out None samples (if any)
    batch = [item for item in batch if item is not None]

    if not batch:
        # Return empty batch
        return {
            "continuous_stream": torch.zeros(0, 18, 48),
            "activity_ids": torch.zeros(0, dtype=torch.long),
            "metadata": {},
        }

    # Find max length in batch
    max_len = max(item["continuous_stream"].shape[1] for item in batch)

    # Round down to nearest multiple of 16 (Chronos-2 patch size)
    # Cap at reasonable maximum to prevent OOM from occasional long sequences
    MAX_ALLOWED_TIMESTEPS = 1024  # Hard cap to prevent OOM
    max_len = min(max_len, MAX_ALLOWED_TIMESTEPS)

    target_len = (max_len // 16) * 16
    target_len = max(target_len, 16)  # Minimum 16 timesteps (1 patch)

    # Left-pad all samples to target_len
    padded_streams = []
    activity_ids = []

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
        activity_ids.append(item["activity_id"])

    streams = torch.stack(padded_streams)  # (B, 18, target_len)
    activity_ids = torch.tensor(activity_ids, dtype=torch.long)  # (B,)

    # Aggregate metadata
    metadata = {
        "subject": [item["metadata"]["subject"] for item in batch],
        "split_name": [item["metadata"]["split_name"] for item in batch],
        "activity_index": [item["metadata"]["activity_index"] for item in batch],
        "activity_name": [item["metadata"]["activity_name"] for item in batch],
    }

    return {
        "continuous_stream": streams,
        "activity_ids": activity_ids,
        "metadata": metadata,
    }
