"""
Convert HHAR (Heterogeneity Activity Recognition) dataset to standardized format.

Input: data/raw/hhar/Activity recognition exp/
Output: data/hhar/
  - manifest.json (minimal, human-readable)
  - labels.json (activity labels per session)
  - sessions/session_XXX/data.parquet (all channels as DataFrame)

HHAR Dataset Info:
- 9 users (a-i)
- 6 activities: stand, walk, bike, sit, stairsup, stairsdown
- Multiple device models (phones and watches)
- We use phone data for consistency with other datasets
- ~43.9 million samples total
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


# Activity mapping (original names to standardized)
ACTIVITIES = {
    "stand": "standing",
    "sit": "sitting",
    "walk": "walking",
    "bike": "cycling",
    "stairsup": "walking_upstairs",
    "stairsdown": "walking_downstairs",
}

# Paths
RAW_DIR = Path("data/raw/hhar/Activity recognition exp")
OUTPUT_DIR = Path("data/hhar")

# Processing parameters
WINDOW_SIZE = 128  # ~2.56 seconds at 50Hz
WINDOW_STRIDE = 64  # 50% overlap
TARGET_SAMPLE_RATE = 50  # Hz (HHAR varies between devices, we'll resample)


def load_and_merge_sensor_data(acc_path: Path, gyro_path: Path) -> pd.DataFrame:
    """
    Load and merge accelerometer and gyroscope data.

    Returns DataFrame with columns:
    - timestamp_ns: Creation_Time in nanoseconds
    - acc_x, acc_y, acc_z: Accelerometer data
    - gyro_x, gyro_y, gyro_z: Gyroscope data
    - user: User ID (a-i)
    - activity: Ground truth activity
    """
    print("  Loading accelerometer data...")
    # Read in chunks to handle large file
    acc_chunks = []
    for chunk in pd.read_csv(acc_path, chunksize=1000000):
        # Filter out null activities and keep only needed columns
        chunk = chunk[chunk['gt'].notna() & (chunk['gt'] != 'null')]
        chunk = chunk[['Creation_Time', 'x', 'y', 'z', 'User', 'Device', 'gt']]
        acc_chunks.append(chunk)

    acc_df = pd.concat(acc_chunks, ignore_index=True)
    acc_df.columns = ['timestamp_ns', 'acc_x', 'acc_y', 'acc_z', 'user', 'device', 'activity']
    print(f"    Loaded {len(acc_df):,} accelerometer samples")

    print("  Loading gyroscope data...")
    gyro_chunks = []
    for chunk in pd.read_csv(gyro_path, chunksize=1000000):
        chunk = chunk[chunk['gt'].notna() & (chunk['gt'] != 'null')]
        chunk = chunk[['Creation_Time', 'x', 'y', 'z', 'User', 'Device', 'gt']]
        gyro_chunks.append(chunk)

    gyro_df = pd.concat(gyro_chunks, ignore_index=True)
    gyro_df.columns = ['timestamp_ns', 'gyro_x', 'gyro_y', 'gyro_z', 'user', 'device', 'activity']
    print(f"    Loaded {len(gyro_df):,} gyroscope samples")

    # Merge on timestamp, user, and activity (approximate matching)
    print("  Merging sensor data...")

    # Sort by user, DEVICE, activity, timestamp. Streams are physical: never merge
    # across devices (their rates AND biases differ — the whole point of HHAR).
    acc_df = acc_df.sort_values(['user', 'device', 'activity', 'timestamp_ns']).reset_index(drop=True)
    gyro_df = gyro_df.sort_values(['user', 'device', 'activity', 'timestamp_ns']).reset_index(drop=True)

    merged_data = []
    keys = acc_df[['user', 'device', 'activity']].drop_duplicates()
    for user, device, activity in keys.itertuples(index=False):
        acc_subset = acc_df[(acc_df['user'] == user) & (acc_df['device'] == device)
                            & (acc_df['activity'] == activity)].sort_values('timestamp_ns')
        gyro_subset = gyro_df[(gyro_df['user'] == user) & (gyro_df['device'] == device)
                              & (gyro_df['activity'] == activity)].sort_values('timestamp_ns')
        if len(acc_subset) == 0 or len(gyro_subset) == 0:
            continue

        # Match gyro to acc timestamps within one physical device (50ms tolerance)
        merged = pd.merge_asof(
            acc_subset,
            gyro_subset[['timestamp_ns', 'gyro_x', 'gyro_y', 'gyro_z']],
            on='timestamp_ns',
            direction='nearest',
            tolerance=50_000_000,  # 50ms in nanoseconds
        )
        merged = merged.dropna(subset=['gyro_x', 'gyro_y', 'gyro_z'])
        merged_data.append(merged)

    result = pd.concat(merged_data, ignore_index=True)
    print(f"    Merged: {len(result):,} samples with both acc and gyro")

    return result


def resample_stream(group, rate=TARGET_SAMPLE_RATE):
    """Resample one physical (user,device,activity) stream to a uniform `rate` Hz on
    its REAL Creation_Time clock via linear interpolation, BEFORE windowing. Native
    HHAR rates are 50-200 Hz; without this a 128-sample window from a 200 Hz device
    spans only ~0.64 s of motion yet was stamped 2.56 s @ 50 Hz."""
    g = group.sort_values('timestamp_ns').drop_duplicates('timestamp_ns')
    if len(g) < 2:
        return None
    t = g['timestamp_ns'].to_numpy(np.float64) / 1e9
    t = t - t[0]
    duration = float(t[-1])
    if duration <= 0:
        return None
    n = int(duration * rate) + 1
    grid = np.arange(n) / rate
    out = {'timestamp_sec': grid}
    for c in ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']:
        out[c] = np.interp(grid, t, g[c].to_numpy(np.float64))
    return pd.DataFrame(out)


def create_windows(df: pd.DataFrame) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Create fixed-size windows from continuous sensor data.

    Returns:
        sessions: List of session IDs
        labels_dict: Dict mapping session ID to list of activity labels
    """
    sessions = []
    labels_dict = {}
    session_idx = 0

    # Group by user, DEVICE, activity — resample each physical stream to a true
    # 50 Hz on its real clock before windowing (native rates are 50-200 Hz).
    for (user, device, activity), group in df.groupby(['user', 'device', 'activity']):
        group = resample_stream(group)
        if group is None or len(group) < WINDOW_SIZE:
            continue

        # Map activity to standardized name
        std_activity = ACTIVITIES.get(activity, activity)

        # Create sliding windows over the resampled (true 50 Hz) stream
        num_samples = len(group)
        num_windows = max(0, (num_samples - WINDOW_SIZE) // WINDOW_STRIDE + 1)

        for win_idx in range(num_windows):
            start_idx = win_idx * WINDOW_STRIDE
            end_idx = start_idx + WINDOW_SIZE

            window_data = group.iloc[start_idx:end_idx]

            # Create session ID
            session_id = f"hhar_{user}_{activity}_{session_idx:05d}"

            # Create DataFrame for this window
            window_df = pd.DataFrame({
                'timestamp_sec': np.arange(WINDOW_SIZE) * (1.0 / TARGET_SAMPLE_RATE),
                'acc_x': window_data['acc_x'].values,
                'acc_y': window_data['acc_y'].values,
                'acc_z': window_data['acc_z'].values,
                'gyro_x': window_data['gyro_x'].values,
                'gyro_y': window_data['gyro_y'].values,
                'gyro_z': window_data['gyro_z'].values,
            })

            # Save to parquet
            session_dir = OUTPUT_DIR / "sessions" / session_id
            session_dir.mkdir(parents=True, exist_ok=True)

            parquet_path = session_dir / "data.parquet"
            window_df.to_parquet(parquet_path, index=False)

            # Store label
            labels_dict[session_id] = [std_activity]
            sessions.append(session_id)
            session_idx += 1

        if num_windows > 0:
            print(f"    User {user}, {std_activity}: {num_windows} windows")

    return sessions, labels_dict


def create_manifest():
    """Create minimal manifest.json."""
    manifest = {
        "dataset_name": "HHAR",
        "description": "Heterogeneity Human Activity Recognition dataset. 9 users performing 6 activities with smartphones (Nexus 4, Galaxy S+, Galaxy S3, S3 mini) carried in a WAIST POUCH. Phones sampled at their native 50-200 Hz; resampled to a true 50 Hz here. Triaxial accelerometer and gyroscope (phone data only; watch streams not used).",
        "channels": [
            {
                "name": "acc_x",
                "description": "Accelerometer X-axis",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "acc_y",
                "description": "Accelerometer Y-axis",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "acc_z",
                "description": "Accelerometer Z-axis",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "gyro_x",
                "description": "Gyroscope X-axis",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "gyro_y",
                "description": "Gyroscope Y-axis",
                "sampling_rate_hz": 50.0
            },
            {
                "name": "gyro_z",
                "description": "Gyroscope Z-axis",
                "sampling_rate_hz": 50.0
            }
        ]
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Created manifest: {manifest_path}")


def main():
    """Convert HHAR to standardized format."""
    print("=" * 80)
    print("HHAR -> Standardized Format Converter")
    print("=" * 80)

    # Check input
    acc_path = RAW_DIR / "Phones_accelerometer.csv"
    gyro_path = RAW_DIR / "Phones_gyroscope.csv"

    if not acc_path.exists():
        print(f"ERROR: Accelerometer data not found at {acc_path}")
        print("Run: python datascripts/shared/download_all_datasets.py hhar")
        return

    if not gyro_path.exists():
        print(f"ERROR: Gyroscope data not found at {gyro_path}")
        print("Run: python datascripts/shared/download_all_datasets.py hhar")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and merge sensor data
    print("\nStep 1: Loading and merging sensor data...")
    merged_df = load_and_merge_sensor_data(acc_path, gyro_path)

    # Create windows
    print("\nStep 2: Creating windows...")
    sessions, labels_dict = create_windows(merged_df)

    # Save labels.json
    labels_path = OUTPUT_DIR / "labels.json"
    with open(labels_path, 'w') as f:
        json.dump(labels_dict, f, indent=2)

    print(f"\nCreated labels: {labels_path}")
    print(f"  Total sessions: {len(labels_dict)}")

    # Create manifest
    create_manifest()

    # Print summary
    print(f"\n{'=' * 80}")
    print("Conversion complete!")
    print(f"{'=' * 80}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"  - {len(labels_dict)} sessions")
    print(f"  - 6 channels (acc + gyro)")
    print(f"  - {TARGET_SAMPLE_RATE} Hz sampling rate")
    print(f"  - {WINDOW_SIZE} samples per window (~{WINDOW_SIZE/TARGET_SAMPLE_RATE:.2f}s)")

    # Activity distribution
    activity_counts = {}
    for session_id, labels in labels_dict.items():
        for label in labels:
            activity_counts[label] = activity_counts.get(label, 0) + 1

    print("\nActivity distribution:")
    for activity, count in sorted(activity_counts.items()):
        print(f"  {activity}: {count}")


if __name__ == "__main__":
    main()
