"""
Convert Daphnet Freezing of Gait (FoG) dataset to standardized format.

Input: data/raw/daphnet_fog/dataset/
Output: data/daphnet_fog/
  - manifest.json
  - labels.json
  - sessions/session_XXX/data.parquet

Daphnet FoG Dataset Info:
- 10 subjects with Parkinson's disease
- 2 usable activity classes: walking (normal gait), freezing_gait (FoG episodes)
- 3 accelerometers (ankle, thigh, trunk)
- 64 Hz sampling rate
- Used for TRAINING (fall detection / gait analysis)

Reference:
Bachlin et al., "Wearable Assistant for Parkinson's Disease Patients With the Freezing of Gait Symptom"
https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add datascripts to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.windowing import create_variable_windows, get_window_range


# Activity mapping
# 0: not experiment, 1: no freeze (walking), 2: freeze
ACTIVITIES = {
    1: "walking",         # Normal walking (no freeze)
    2: "freezing_gait",   # Freezing of gait episode
}

# Column names (11 columns)
# Time, ankle acc (3), thigh acc (3), trunk acc (3), label
COLUMN_NAMES = [
    'timestamp',
    'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z',
    'thigh_acc_x', 'thigh_acc_y', 'thigh_acc_z',
    'trunk_acc_x', 'trunk_acc_y', 'trunk_acc_z',
    'activity_code'
]

# Sensor columns (for interpolation)
SENSOR_COLUMNS = [
    'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z',
    'thigh_acc_x', 'thigh_acc_y', 'thigh_acc_z',
    'trunk_acc_x', 'trunk_acc_y', 'trunk_acc_z',
]

# Paths
RAW_DIR = Path("data/raw/daphnet_fog/dataset")
OUTPUT_DIR = Path("data/daphnet_fog")

# Sampling rate
SAMPLE_RATE = 64.0  # Hz


def segment_continuous_activity(df: pd.DataFrame, min_duration_sec: float = 2.0):
    """Segment continuous recording into sessions based on activity changes."""
    sessions = []

    activity_changes = df['activity'].ne(df['activity'].shift())
    segment_ids = activity_changes.cumsum()

    for seg_id in segment_ids.unique():
        segment = df[segment_ids == seg_id].copy()

        activity = segment['activity'].iloc[0]
        if pd.isna(activity):
            continue

        duration_sec = len(segment) / SAMPLE_RATE
        if duration_sec < min_duration_sec:
            continue

        sessions.append({
            'data': segment,
            'activity': activity,
            'duration_sec': duration_sec
        })

    return sessions


def load_subject_file(filepath: Path) -> pd.DataFrame:
    """Load a single subject recording file."""
    try:
        # Space-separated, no header
        df = pd.read_csv(filepath, sep=r'\s+', header=None, names=COLUMN_NAMES)

        # Map activity codes to names (filter out 0 = not experiment)
        df['activity'] = df['activity_code'].map(ACTIVITIES)
        df = df[df['activity'].notna()].copy()

        # Handle missing sensor data
        df[SENSOR_COLUMNS] = df[SENSOR_COLUMNS].replace(-1, np.nan)
        df[SENSOR_COLUMNS] = df[SENSOR_COLUMNS].interpolate(method='linear', limit_direction='both')
        df[SENSOR_COLUMNS] = df[SENSOR_COLUMNS].fillna(0)

        return df

    except Exception as e:
        print(f"    Error loading {filepath.name}: {e}")
        return pd.DataFrame()


def convert_subject(filepath: Path):
    """Convert one subject's data to sessions with variable-length windowing."""
    print(f"  Processing: {filepath.name}")

    df = load_subject_file(filepath)

    if df.empty:
        return [], {}

    print(f"    Loaded {len(df)} samples")

    # Segment by activity
    sessions = segment_continuous_activity(df)
    print(f"    Found {len(sessions)} activity segments")

    # Extract subject ID from filename (e.g., S01R01.txt -> S01)
    subject_id = filepath.stem[:3] if len(filepath.stem) >= 3 else filepath.stem

    session_data = []
    labels_dict = {}
    total_windows = 0

    for idx, session in enumerate(sessions):
        activity_name = session['activity']
        base_session_id = f"daphnet_{subject_id}_seg{idx:03d}"

        # Prepare DataFrame
        data = session['data'].copy()

        # Create timestamp from index (original timestamp might not start at 0)
        data.insert(0, 'timestamp_sec', np.arange(len(data)) / SAMPLE_RATE)

        # Keep only sensor columns
        cols_to_keep = ['timestamp_sec'] + SENSOR_COLUMNS
        data = data[cols_to_keep]

        # Apply variable-length windowing
        windows = create_variable_windows(
            df=data,
            session_prefix=base_session_id,
            activity=activity_name,
            sample_rate=SAMPLE_RATE,
        )

        # Save each window
        for window_id, window_df, window_activity in windows:
            session_dir = OUTPUT_DIR / "sessions" / window_id
            session_dir.mkdir(parents=True, exist_ok=True)

            parquet_path = session_dir / "data.parquet"
            window_df.to_parquet(parquet_path, index=False)

            labels_dict[window_id] = [window_activity]
            session_data.append(window_id)
            total_windows += 1

    print(f"    Created {total_windows} windows")
    return session_data, labels_dict


def create_manifest():
    """Create minimal manifest.json."""
    channels = []

    body_parts = {
        "ankle": "ankle-mounted",
        "thigh": "thigh-mounted",
        "trunk": "trunk-mounted (lower back)",
    }

    for pos in ["ankle", "thigh", "trunk"]:
        body_desc = body_parts[pos]

        for axis in ['x', 'y', 'z']:
            channels.append({
                "name": f"{pos}_acc_{axis}",
                "description": f"Accelerometer {axis.upper()}-axis from {body_desc} sensor",
                "sampling_rate_hz": SAMPLE_RATE
            })

    manifest = {
        "dataset_name": "Daphnet FoG",
        "description": "Freezing of gait detection in Parkinson's disease patients. 10 subjects with accelerometers on ankle, thigh, and trunk. Activities: normal walking and freezing episodes.",
        "channels": channels
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Created manifest: {manifest_path}")


def main():
    """Convert Daphnet FoG to standardized format."""
    print("=" * 80)
    print("Daphnet FoG -> Standardized Format Converter")
    print("=" * 80)

    # Check input - try multiple possible locations
    possible_dirs = [
        RAW_DIR,
        RAW_DIR.parent,
        Path("data/raw/daphnet_fog"),
    ]

    raw_dir = None
    for d in possible_dirs:
        if d.exists():
            files = list(d.glob("*.txt")) + list(d.rglob("*.txt"))
            if files:
                raw_dir = d
                break

    if raw_dir is None:
        print(f"ERROR: Raw data not found")
        print("Download from: https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all subject files (format: S01R01.txt, S01R02.txt, etc.)
    subject_files = sorted(raw_dir.glob("S*.txt"))

    if not subject_files:
        subject_files = sorted(raw_dir.rglob("S*.txt"))

    print(f"\nFound {len(subject_files)} recording files")

    all_labels = {}
    all_sessions = []

    for subject_file in subject_files:
        sessions, labels = convert_subject(subject_file)
        all_sessions.extend(sessions)
        all_labels.update(labels)

    if not all_labels:
        print("\nNo sessions created. Check the raw data format.")
        return

    # Save labels.json
    labels_path = OUTPUT_DIR / "labels.json"
    with open(labels_path, 'w') as f:
        json.dump(all_labels, f, indent=2)

    print(f"\nCreated labels: {labels_path}")
    print(f"  Total sessions: {len(all_labels)}")

    # Create manifest
    create_manifest()

    print(f"\n{'=' * 80}")
    print("Conversion complete!")
    print(f"{'=' * 80}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"  - {len(all_labels)} sessions")
    print(f"  - 9 channels (3 accelerometers x 3 axes)")
    print(f"  - {SAMPLE_RATE} Hz sampling rate")

    # Activity distribution
    activity_counts = {}
    for session_id, labels in all_labels.items():
        for label in labels:
            activity_counts[label] = activity_counts.get(label, 0) + 1

    print("\nActivity distribution:")
    for activity, count in sorted(activity_counts.items()):
        print(f"  {activity}: {count}")

    # Generate debug visualizations
    try:
        from shared.visualization_utils import generate_debug_visualizations

        generate_debug_visualizations(OUTPUT_DIR)
    except ImportError as e:
        print(f"\nCould not generate visualizations: {e}")
        print("Install matplotlib: pip install matplotlib")


if __name__ == "__main__":
    main()
