"""
Convert OPPORTUNITY dataset to standardized format.

Input: data/raw/opportunity/OpportunityUCIDataset/
Output: data/opportunity/
  - manifest.json
  - labels.json
  - sessions/session_XXX/data.parquet

OPPORTUNITY Dataset Info:
- 4 subjects (S1-S4)
- 5 ADL runs + 5 drill runs per subject
- Locomotion (4 usable classes): Stand(1), Walk(2), Sit(4), Lie(5)
- Gesture (17 classes): Various kitchen activities (not used here)
- 250 columns: 1 timestamp + 242 sensor + 7 label columns
- 30 Hz sampling rate
- Body-worn: 5 XSens IMUs (Back, RUA, RLA, LUA, LLA)
  each with acc(3) + gyro(3) + mag(3) + quaternion(4) = 13 channels
- Used for TRAINING

The column_names.txt file in the dataset directory maps column indices to
sensor names. This script dynamically reads it if available, falling back
to hardcoded indices verified against the DeepConvLSTM reference implementation.

Reference:
Roggen et al., "Collecting complex activity datasets in highly rich networked sensor environments"
https://archive.ics.uci.edu/ml/datasets/OPPORTUNITY+Activity+Recognition
"""

import os
import sys
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

# Add datascripts to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.windowing import create_variable_windows, get_window_range


# Locomotion labels (column 244, 1-indexed = 243 0-indexed)
# Note: code 3 is intentionally absent in the dataset
LOCOMOTION_LABELS = {
    1: "standing",
    2: "walking",
    4: "sitting",
    5: "lying",
}

# Gesture labels (column 250, 1-indexed = mid-level gestures)
# We use the simpler locomotion labels for training
GESTURE_LABELS = {
    1: "open_door1",
    2: "open_door2",
    3: "close_door1",
    4: "close_door2",
    5: "open_fridge",
    6: "close_fridge",
    7: "open_dishwasher",
    8: "close_dishwasher",
    9: "open_drawer1",
    10: "close_drawer1",
    11: "open_drawer2",
    12: "close_drawer2",
    13: "open_drawer3",
    14: "close_drawer3",
    15: "clean_table",
    16: "drink_cup",
    17: "toggle_switch",
}

# Body-worn IMU positions
IMU_POSITIONS = ["back", "rua", "rla", "lua", "lla"]

# Fallback column indices (0-indexed) for body-worn IMU acc+gyro channels.
# These are used if column_names.txt is not found.
# Verified: BACK IMU starts at column 2 (1-indexed), each IMU has 13 channels
# (acc3 + gyro3 + mag3 + quaternion4). We extract only acc(3) + gyro(3).
FALLBACK_SENSOR_COLUMNS = {
    # Back IMU: columns 2-14 (1-indexed), acc at 2-4, gyro at 5-7
    "back_acc_x": 1, "back_acc_y": 2, "back_acc_z": 3,
    "back_gyro_x": 4, "back_gyro_y": 5, "back_gyro_z": 6,

    # RUA IMU: columns 15-27 (1-indexed), acc at 15-17, gyro at 18-20
    "rua_acc_x": 14, "rua_acc_y": 15, "rua_acc_z": 16,
    "rua_gyro_x": 17, "rua_gyro_y": 18, "rua_gyro_z": 19,

    # RLA IMU: columns 28-40 (1-indexed), acc at 28-30, gyro at 31-33
    "rla_acc_x": 27, "rla_acc_y": 28, "rla_acc_z": 29,
    "rla_gyro_x": 30, "rla_gyro_y": 31, "rla_gyro_z": 32,

    # LUA IMU: columns 41-53 (1-indexed), acc at 41-43, gyro at 44-46
    "lua_acc_x": 40, "lua_acc_y": 41, "lua_acc_z": 42,
    "lua_gyro_x": 43, "lua_gyro_y": 44, "lua_gyro_z": 45,

    # LLA IMU: columns 54-66 (1-indexed), acc at 54-56, gyro at 57-59
    "lla_acc_x": 53, "lla_acc_y": 54, "lla_acc_z": 55,
    "lla_gyro_x": 56, "lla_gyro_y": 57, "lla_gyro_z": 58,
}

# Label column indices (0-indexed)
LOCOMOTION_COL = 243  # Column 244 in 1-indexed
GESTURE_COL = 249     # Column 250 in 1-indexed (mid-level)

# Paths
RAW_DIR = Path("data/raw/opportunity/OpportunityUCIDataset/dataset")
OUTPUT_DIR = Path("data/opportunity")

# Sampling rate
SAMPLE_RATE = 30.0  # Hz


def load_column_names(dataset_dir: Path) -> dict:
    """Load column_names.txt and build sensor column index mapping.

    The column_names.txt file maps each column number to its sensor description.
    We parse it to find the exact indices for each body-worn IMU's acc and gyro channels.

    Returns:
        Dict mapping channel_name -> 0-indexed column index, or empty dict if file not found.
    """
    col_names_file = dataset_dir / "column_names.txt"
    if not col_names_file.exists():
        # Try parent directory
        col_names_file = dataset_dir.parent / "column_names.txt"
    if not col_names_file.exists():
        return {}

    print(f"  Found column_names.txt, parsing sensor layout...")

    # IMU name patterns in column_names.txt
    imu_patterns = {
        "back": re.compile(r'Accelerometer\s+BACK|InertialMeasurementUnit\s+BACK', re.IGNORECASE),
        "rua": re.compile(r'Accelerometer\s+RUA|InertialMeasurementUnit\s+RUA', re.IGNORECASE),
        "rla": re.compile(r'Accelerometer\s+RLA|InertialMeasurementUnit\s+RLA', re.IGNORECASE),
        "lua": re.compile(r'Accelerometer\s+LUA|InertialMeasurementUnit\s+LUA', re.IGNORECASE),
        "lla": re.compile(r'Accelerometer\s+LLA|InertialMeasurementUnit\s+LLA', re.IGNORECASE),
    }

    sensor_columns = {}
    try:
        with open(col_names_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                # Format: "Column: N name" or "N: name"
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue

                # Extract column number
                col_str = parts[0].rstrip(':')
                if col_str.startswith("Column"):
                    col_str = parts[1].split()[0].rstrip(':')
                    description = ' '.join(parts[1].split()[1:])
                else:
                    try:
                        int(col_str)
                        description = parts[1]
                    except ValueError:
                        continue

                try:
                    col_idx_1indexed = int(col_str)
                    col_idx = col_idx_1indexed - 1  # Convert to 0-indexed
                except ValueError:
                    continue

                desc_lower = description.lower()

                # Match each IMU position
                for pos_name, pattern in imu_patterns.items():
                    if pattern.search(description):
                        # Determine sensor type and axis
                        if 'accx' in desc_lower or 'accelerometer' in desc_lower and 'x' in desc_lower.split()[-1]:
                            sensor_columns[f"{pos_name}_acc_x"] = col_idx
                        elif 'accy' in desc_lower:
                            sensor_columns[f"{pos_name}_acc_y"] = col_idx
                        elif 'accz' in desc_lower:
                            sensor_columns[f"{pos_name}_acc_z"] = col_idx
                        elif 'gyrox' in desc_lower or ('gyroscope' in desc_lower and 'x' in desc_lower.split()[-1]):
                            sensor_columns[f"{pos_name}_gyro_x"] = col_idx
                        elif 'gyroy' in desc_lower:
                            sensor_columns[f"{pos_name}_gyro_y"] = col_idx
                        elif 'gyroz' in desc_lower:
                            sensor_columns[f"{pos_name}_gyro_z"] = col_idx

        if sensor_columns:
            print(f"    Resolved {len(sensor_columns)} sensor columns from column_names.txt")
            # Validate we got all expected channels
            expected = len(IMU_POSITIONS) * 6  # 5 IMUs x 6 channels (acc3 + gyro3)
            if len(sensor_columns) < expected:
                print(f"    Warning: only found {len(sensor_columns)}/{expected} expected channels")
                print(f"    Falling back to hardcoded indices for missing channels")

    except Exception as e:
        print(f"    Error parsing column_names.txt: {e}")

    return sensor_columns


def segment_continuous_activity(df: pd.DataFrame, min_duration_sec: float = 3.0):
    """Segment continuous recording into sessions based on activity changes."""
    sessions = []

    # Find activity boundaries
    activity_changes = df['activity'].ne(df['activity'].shift())
    segment_ids = activity_changes.cumsum()

    for seg_id in segment_ids.unique():
        segment = df[segment_ids == seg_id].copy()

        activity = segment['activity'].iloc[0]
        if pd.isna(activity) or activity == 'null':
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


def load_dat_file(filepath: Path, sensor_columns: dict) -> pd.DataFrame:
    """Load a single .dat file from OPPORTUNITY dataset.

    Args:
        filepath: Path to the .dat file
        sensor_columns: Dict mapping channel_name -> 0-indexed column index
    """
    try:
        # Space-separated, no header
        df = pd.read_csv(filepath, sep=r'\s+', header=None)

        # Expect 250 columns (1 timestamp + 242 sensor + 7 labels)
        if len(df.columns) < LOCOMOTION_COL + 1:
            print(f"    Unexpected columns in {filepath.name}: {len(df.columns)}")
            return pd.DataFrame()

        # Extract sensor columns
        sensor_data = {}
        for name, idx in sensor_columns.items():
            if idx < len(df.columns):
                sensor_data[name] = df.iloc[:, idx].values
            else:
                print(f"    Warning: column {idx} ({name}) out of range in {filepath.name}")

        if not sensor_data:
            return pd.DataFrame()

        # Create sensor DataFrame
        result = pd.DataFrame(sensor_data)

        # Add locomotion label
        result['activity_code'] = df.iloc[:, LOCOMOTION_COL].values

        # Map to activity names
        result['activity'] = result['activity_code'].map(LOCOMOTION_LABELS)

        # Validate: check NaN percentage per sensor
        sensor_cols = list(sensor_columns.keys())
        nan_pct = result[sensor_cols].isna().mean()
        high_nan = nan_pct[nan_pct > 0.5]
        if not high_nan.empty:
            print(f"    Warning: high NaN channels in {filepath.name}:")
            for ch, pct in high_nan.items():
                print(f"      {ch}: {pct*100:.1f}% NaN")

        # Interpolate NaN sensor values (brief sensor dropouts are common in OPPORTUNITY)
        result[sensor_cols] = result[sensor_cols].interpolate(method='linear', limit_direction='both')
        result[sensor_cols] = result[sensor_cols].fillna(0)

        return result

    except Exception as e:
        print(f"    Error loading {filepath.name}: {e}")
        return pd.DataFrame()


def convert_subject(subject_id: str, data_files: list, sensor_columns: dict):
    """Convert one subject's data to sessions with variable-length windowing."""
    print(f"  Processing subject: {subject_id}")

    all_data = []
    for filepath in data_files:
        df = load_dat_file(filepath, sensor_columns)
        if not df.empty:
            df['source_file'] = filepath.stem
            all_data.append(df)
            print(f"    Loaded {filepath.name}: {len(df)} samples")

    if not all_data:
        return [], {}

    combined_df = pd.concat(all_data, ignore_index=True)

    # Segment by activity
    sessions = segment_continuous_activity(combined_df)
    print(f"    Found {len(sessions)} activity segments")

    session_data = []
    labels_dict = {}
    total_windows = 0

    for idx, session in enumerate(sessions):
        activity_name = session['activity']
        base_session_id = f"opportunity_{subject_id}_seg{idx:03d}"

        # Prepare DataFrame
        data = session['data'].copy()
        data.insert(0, 'timestamp_sec', np.arange(len(data)) / SAMPLE_RATE)

        # Keep only sensor columns
        cols_to_keep = ['timestamp_sec'] + list(sensor_columns.keys())
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

    print(f"    Created {total_windows} windows from {len(sessions)} segments")
    return session_data, labels_dict


def create_manifest(sensor_columns: dict):
    """Create minimal manifest.json."""
    channels = []

    body_parts = {
        "back": "back-mounted (lower back)",
        "rua": "right upper arm",
        "rla": "right lower arm (forearm)",
        "lua": "left upper arm",
        "lla": "left lower arm (forearm)",
    }

    for sensor_name in sensor_columns.keys():
        parts = sensor_name.split('_')
        body_part = parts[0]
        sensor_type = parts[1]
        axis = parts[2]

        body_desc = body_parts.get(body_part, body_part)
        sensor_desc = {"acc": "Accelerometer", "gyro": "Gyroscope"}[sensor_type]

        channels.append({
            "name": sensor_name,
            "description": f"{sensor_desc} {axis.upper()}-axis from {body_desc} XSens IMU",
            "sampling_rate_hz": SAMPLE_RATE
        })

    manifest = {
        "dataset_name": "OPPORTUNITY",
        "description": "Activities of daily living in a sensor-rich kitchen environment. "
                       "4 subjects performing locomotion activities (stand, walk, sit, lie). "
                       "5 body-worn XSens IMUs on back, upper arms, and lower arms at 30 Hz.",
        "channels": channels
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Created manifest: {manifest_path}")
    print(f"  {len(channels)} channels ({len(IMU_POSITIONS)} IMUs x acc+gyro)")


def main():
    """Convert OPPORTUNITY to standardized format."""
    print("=" * 80)
    print("OPPORTUNITY -> Standardized Format Converter")
    print("=" * 80)

    # Check input - try multiple locations
    raw_dir = RAW_DIR
    if not raw_dir.exists():
        # Try without the nested 'dataset' directory
        alt_dir = RAW_DIR.parent
        if alt_dir.exists() and list(alt_dir.glob("*.dat")):
            raw_dir = alt_dir
        else:
            print(f"ERROR: Raw data not found at {RAW_DIR}")
            print("Download from: https://archive.ics.uci.edu/ml/datasets/OPPORTUNITY+Activity+Recognition")
            return

    # Try to dynamically resolve sensor column indices from column_names.txt
    dynamic_columns = load_column_names(raw_dir)
    if dynamic_columns:
        sensor_columns = dynamic_columns
        # Fill in any missing channels from fallback
        for name, idx in FALLBACK_SENSOR_COLUMNS.items():
            if name not in sensor_columns:
                sensor_columns[name] = idx
        print(f"  Using dynamically resolved column indices")
    else:
        sensor_columns = FALLBACK_SENSOR_COLUMNS.copy()
        print(f"  Using fallback column indices (column_names.txt not found)")
        print(f"  NOTE: Non-BACK IMU indices may need verification after download")

    print(f"  Sensor columns: {len(sensor_columns)}")
    for pos in IMU_POSITIONS:
        acc_idx = sensor_columns.get(f"{pos}_acc_x", "?")
        gyro_idx = sensor_columns.get(f"{pos}_gyro_x", "?")
        print(f"    {pos.upper()}: acc starts at col {acc_idx}, gyro starts at col {gyro_idx}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all .dat files and group by subject
    dat_files = sorted(raw_dir.glob("*.dat"))
    if not dat_files:
        dat_files = sorted(raw_dir.rglob("*.dat"))

    print(f"\nFound {len(dat_files)} .dat files")

    # Group files by subject (e.g., S1-ADL1.dat, S1-Drill.dat, ...)
    subjects = {}
    for filepath in dat_files:
        # Filename format: S1-ADL1.dat, S1-Drill.dat, etc.
        parts = filepath.stem.split('-')
        if parts:
            subject_id = parts[0]
            if subject_id not in subjects:
                subjects[subject_id] = []
            subjects[subject_id].append(filepath)

    print(f"Found {len(subjects)} subjects")

    all_labels = {}
    all_sessions = []

    for subject_id, files in sorted(subjects.items()):
        sessions, labels = convert_subject(subject_id, files, sensor_columns)
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
    create_manifest(sensor_columns)

    print(f"\n{'=' * 80}")
    print("Conversion complete!")
    print(f"{'=' * 80}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"  - {len(all_labels)} sessions")
    print(f"  - {len(sensor_columns)} channels ({len(IMU_POSITIONS)} IMUs x acc+gyro)")
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
