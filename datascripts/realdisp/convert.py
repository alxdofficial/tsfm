"""
Convert REALDISP dataset to standardized format.

Input: data/raw/realdisp/
Output: data/realdisp/
  - manifest.json
  - labels.json
  - sessions/session_XXX/data.parquet

REALDISP Dataset Info:
- 17 subjects
- 33 fitness/warm-up activities (A1-A33)
- 9 body-worn sensors (each: acc3 + gyro3 + mag3 + quaternion4 = 13 channels)
- 120 columns per row: 2 timestamp + 117 sensor + 1 label
- 50 Hz sampling rate
- File format: space-separated .log files, no header
- File naming: subject[N]_[scenario].log (ideal, self, mutual[N])
- Used for TRAINING

Reference:
Banos et al., "REALDISP Activity Recognition Dataset"
https://archive.ics.uci.edu/dataset/305/realdisp+activity+recognition+dataset
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


# Activity mapping (33 activities) - EXACT labels from REALDISP documentation
# These are all fitness/warm-up exercises
ACTIVITIES = {
    1: "walking",
    2: "jogging",
    3: "running",
    4: "jump_up",
    5: "jump_front_back",
    6: "jump_sideways",
    7: "jump_legs_arms",
    8: "jump_rope",
    9: "trunk_twist_arms_out",
    10: "trunk_twist_elbows_bent",
    11: "waist_bends_forward",
    12: "waist_rotation",
    13: "waist_bend_cross",
    14: "reach_heels_backwards",
    15: "lateral_bend",
    16: "lateral_bend_arm_up",
    17: "forward_stretching",
    18: "upper_lower_twist",
    19: "lateral_arm_elevation",
    20: "frontal_arm_elevation",
    21: "frontal_hand_claps",
    22: "frontal_crossing_arms",
    23: "shoulders_high_rotation",
    24: "shoulders_low_rotation",
    25: "arms_inner_rotation",
    26: "knees_to_breast",
    27: "heels_to_backside",
    28: "knees_bending_crouching",
    29: "knees_alternating_forward",
    30: "rotation_on_knees",
    31: "rowing",
    32: "elliptical_bike",
    33: "cycling",
}

# Sensor positions in order they appear in the data file (S1-S9)
SENSOR_POSITIONS = [
    "left_calf",       # S1
    "left_thigh",      # S2
    "right_calf",      # S3
    "right_thigh",     # S4
    "back",            # S5
    "left_forearm",    # S6 (left lower arm)
    "left_arm",        # S7 (left upper arm)
    "right_forearm",   # S8 (right lower arm)
    "right_arm",       # S9 (right upper arm)
]

# Paths
RAW_DIR = Path("data/raw/realdisp")
OUTPUT_DIR = Path("data/realdisp")

# Sampling rate
SAMPLE_RATE = 50.0  # Hz

# Per-sensor channel order (13 channels each)
# AccX, AccY, AccZ, GyrX, GyrY, GyrZ, MagX, MagY, MagZ, Q1, Q2, Q3, Q4
CHANNELS_PER_SENSOR = 13
TOTAL_COLUMNS = 120  # 2 timestamp + 9*13 sensor + 1 label


def get_column_names():
    """Generate column names matching the actual REALDISP file layout.

    Layout: timestamp_s, timestamp_us, [9 sensors x 13 channels], activity_code
    Total: 120 columns
    """
    columns = ['timestamp_s', 'timestamp_us']

    for pos in SENSOR_POSITIONS:
        # Accelerometer (3)
        for axis in ['x', 'y', 'z']:
            columns.append(f"{pos}_acc_{axis}")
        # Gyroscope (3)
        for axis in ['x', 'y', 'z']:
            columns.append(f"{pos}_gyro_{axis}")
        # Magnetometer (3)
        for axis in ['x', 'y', 'z']:
            columns.append(f"{pos}_mag_{axis}")
        # Quaternion orientation (4) - we skip these in output
        for i in range(4):
            columns.append(f"{pos}_ori_{i}")

    columns.append('activity_code')  # Last column

    return columns


def load_subject_file(filepath: Path) -> pd.DataFrame:
    """Load a single subject .log file."""
    try:
        # Space-separated, no header
        df = pd.read_csv(filepath, sep=r'\s+', header=None)

        actual_cols = len(df.columns)

        if actual_cols != TOTAL_COLUMNS:
            print(f"    Warning: {filepath.name} has {actual_cols} columns, expected {TOTAL_COLUMNS}")
            if actual_cols < 10:
                return pd.DataFrame()

        # Assign column names
        all_col_names = get_column_names()
        if actual_cols == len(all_col_names):
            df.columns = all_col_names
        elif actual_cols < len(all_col_names):
            # Fewer columns - assign what we can
            df.columns = all_col_names[:actual_cols]
            print(f"    Partial columns: {actual_cols}/{len(all_col_names)}")
        else:
            # More columns than expected
            df = df.iloc[:, :len(all_col_names)]
            df.columns = all_col_names

        # Activity code is the LAST column
        if 'activity_code' in df.columns:
            df['activity'] = df['activity_code'].map(ACTIVITIES)
        else:
            print(f"    No activity_code column found")
            return pd.DataFrame()

        # Filter out unknown activities (code 0 or unmapped)
        df = df[df['activity'].notna()].copy()

        # Interpolate NaN sensor values
        sensor_cols = [c for c in df.columns if any(s in c for s in ['acc_', 'gyro_', 'mag_'])]
        df[sensor_cols] = df[sensor_cols].interpolate(method='linear', limit_direction='both')
        df[sensor_cols] = df[sensor_cols].fillna(0)

        return df

    except Exception as e:
        print(f"    Error loading {filepath.name}: {e}")
        return pd.DataFrame()


def segment_continuous_activity(df: pd.DataFrame, min_duration_sec: float = 3.0):
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


def convert_subject(filepath: Path):
    """Convert one subject's data to sessions with variable-length windowing."""
    print(f"  Processing: {filepath.name}")

    df = load_subject_file(filepath)

    if df.empty:
        return [], {}

    print(f"    Loaded {len(df)} samples, {df['activity'].nunique()} activities")

    # Segment by activity
    sessions = segment_continuous_activity(df)
    print(f"    Found {len(sessions)} activity segments")

    # Extract subject/scenario from filename (e.g., subject1_ideal.log)
    subject_id = filepath.stem
    session_data = []
    labels_dict = {}
    total_windows = 0

    for idx, session in enumerate(sessions):
        activity_name = session['activity']
        base_session_id = f"realdisp_{subject_id}_seg{idx:03d}"

        # Prepare DataFrame
        data = session['data'].copy()

        # Create timestamp from index (more reliable than raw timestamps)
        data.insert(0, 'timestamp_sec', np.arange(len(data)) / SAMPLE_RATE)

        # Keep only sensor columns (acc, gyro, mag - not orientation or timestamps)
        sensor_cols = [c for c in data.columns if any(s in c for s in ['acc_', 'gyro_', 'mag_'])]
        cols_to_keep = ['timestamp_sec'] + sensor_cols
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
        "left_calf": "left lower leg",
        "left_thigh": "left upper leg",
        "right_calf": "right lower leg",
        "right_thigh": "right upper leg",
        "back": "lower back",
        "left_forearm": "left forearm",
        "left_arm": "left upper arm",
        "right_forearm": "right forearm",
        "right_arm": "right upper arm",
    }

    for pos in SENSOR_POSITIONS:
        body_desc = body_parts.get(pos, pos)

        # Accelerometer
        for axis in ['x', 'y', 'z']:
            channels.append({
                "name": f"{pos}_acc_{axis}",
                "description": f"Accelerometer {axis.upper()}-axis from {body_desc} sensor",
                "sampling_rate_hz": SAMPLE_RATE
            })

        # Gyroscope
        for axis in ['x', 'y', 'z']:
            channels.append({
                "name": f"{pos}_gyro_{axis}",
                "description": f"Gyroscope {axis.upper()}-axis from {body_desc} sensor",
                "sampling_rate_hz": SAMPLE_RATE
            })

        # Magnetometer
        for axis in ['x', 'y', 'z']:
            channels.append({
                "name": f"{pos}_mag_{axis}",
                "description": f"Magnetometer {axis.upper()}-axis from {body_desc} sensor",
                "sampling_rate_hz": SAMPLE_RATE
            })

    manifest = {
        "dataset_name": "REALDISP",
        "description": "Realistic sensor displacement effects on activity recognition. 17 subjects performing 33 fitness/warm-up exercises. 9 body-worn sensors (calves, thighs, back, forearms, upper arms) with accelerometer, gyroscope, and magnetometer.",
        "channels": channels
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Created manifest: {manifest_path}")


def main():
    """Convert REALDISP to standardized format."""
    print("=" * 80)
    print("REALDISP -> Standardized Format Converter")
    print("=" * 80)

    # Check input
    if not RAW_DIR.exists():
        print(f"ERROR: Raw data not found at {RAW_DIR}")
        print("Download from: https://archive.ics.uci.edu/dataset/305/realdisp+activity+recognition+dataset")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all subject files (naming: subject[N]_[scenario].log)
    subject_files = sorted(RAW_DIR.glob("subject*.log"))

    if not subject_files:
        # Try other patterns
        subject_files = sorted(RAW_DIR.rglob("subject*.log"))

    if not subject_files:
        subject_files = sorted(RAW_DIR.glob("*.log")) + sorted(RAW_DIR.glob("*.txt"))

    print(f"\nFound {len(subject_files)} subject files")

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
    print(f"  - {len(SENSOR_POSITIONS) * 9} channels (9 positions x 9 sensors, excl. quaternions)")
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
        print(f"\n⚠ Could not generate visualizations: {e}")
        print("Install matplotlib: pip install matplotlib")


if __name__ == "__main__":
    main()
