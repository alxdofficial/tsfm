"""
Convert Shoaib dataset to standardized format.

Input: data/raw/shoaib/
Output: data/shoaib/
  - manifest.json
  - labels.json
  - sessions/session_XXX/data.parquet

Shoaib 2014 Fusion Dataset Info:
- 10 subjects
- 7 activities: Biking, Sitting, Standing, Walking, Stairsup, Stairsdown, Jogging
- 5 body positions: right_pocket, left_pocket, belt, arm, wrist
- 4 sensors per position: accelerometer, gyroscope, magnetometer, linear acceleration
  (we use acc + gyro + mag = 9 channels per position, skip linear acceleration)
- 50 Hz sampling rate
- Used as ZERO-SHOT TEST set (not for training)

Note: Multiple Shoaib datasets exist (2013 PA, 2014 Fusion, 2016 SA).
This script targets the 2014 "Fusion of Smartphone Motion Sensors" dataset.
The actual file format (delimiter, column order) is described in the README
within the dataset archive. This script auto-detects delimiter and adapts
to both 4-position and 5-position variants.

Reference:
Shoaib et al., "Fusion of Smartphone Motion Sensors for Physical Activity Recognition"
Sensors 2014, 14(6), 10146-10176. DOI: 10.3390/s140610146
https://www.utwente.nl/en/eemcs/ps/research/dataset/
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


# Activity mapping (original names to standardized)
# The 2014 paper uses "Jogging" not "Running" — handle both
ACTIVITIES = {
    "Biking": "cycling",
    "Sitting": "sitting",
    "Standing": "standing",
    "Walking": "walking",
    "Stairsup": "walking_upstairs",
    "Stairsdown": "walking_downstairs",
    "Jogging": "jogging",
    "Running": "running",
}

# Body positions in the 2014 Fusion dataset (5 positions)
# The 2013 PA dataset had only 4 positions (no left_pocket)
BODY_POSITIONS_5 = ["right_pocket", "left_pocket", "belt", "arm", "wrist"]
BODY_POSITIONS_4 = ["pocket", "belt", "arm", "wrist"]  # Fallback for 2013 variant

# Sensors we extract per position (acc, gyro, mag — skip linear acceleration)
SENSORS = ["acc", "gyro", "mag"]
AXES = ["x", "y", "z"]
CHANNELS_PER_POSITION = len(SENSORS) * len(AXES)  # 9


def get_column_names(positions):
    """Generate column names for all sensors at given positions."""
    columns = []
    for pos in positions:
        for sensor in SENSORS:
            for axis in AXES:
                columns.append(f"{pos}_{sensor}_{axis}")
    return columns

# Paths
RAW_DIR = Path("data/raw/shoaib")
OUTPUT_DIR = Path("data/shoaib")

# Sampling rate
SAMPLE_RATE = 50.0  # Hz


def detect_positions(num_columns: int):
    """Detect whether this is a 4-position or 5-position variant based on column count.

    The 2014 dataset has 4 sensors (acc, gyro, mag, linear_acc) x 3 axes = 12 channels/position.
    The 2013 dataset has 3 sensors (acc, gyro, mag) x 3 axes = 9 channels/position.

    We extract only acc+gyro+mag (9 channels) from each position regardless.
    """
    # 5 positions x 12 channels = 60 (2014 Fusion with linear acc)
    # 4 positions x 12 channels = 48 (2013 PA with linear acc)
    # 5 positions x 9 channels = 45 (2014 Fusion without linear acc)
    # 4 positions x 9 channels = 36 (2013 PA without linear acc)
    if num_columns >= 60:
        return BODY_POSITIONS_5, 12  # 5 positions, 12 raw channels each
    elif num_columns >= 48:
        return BODY_POSITIONS_4, 12  # 4 positions, 12 raw channels each
    elif num_columns >= 45:
        return BODY_POSITIONS_5, 9   # 5 positions, 9 raw channels each
    elif num_columns >= 36:
        return BODY_POSITIONS_4, 9   # 4 positions, 9 raw channels each
    else:
        return BODY_POSITIONS_4, 9   # Default fallback


def load_subject_data(subject_dir: Path) -> tuple:
    """
    Load all activity data for a subject.

    The Shoaib dataset has one file per activity per subject.
    Each file contains sensor data from all body positions.

    Returns:
        Tuple of (DataFrame, positions_used) where positions_used is the
        list of body positions detected from the data.
    """
    all_data = []
    detected_positions = None

    for activity_name, std_activity in ACTIVITIES.items():
        # Try different possible file patterns (case-insensitive)
        patterns = [
            f"*{activity_name}*.txt",
            f"*{activity_name}*.csv",
            f"{activity_name}.txt",
            f"{activity_name}.csv",
            f"*{activity_name.lower()}*.txt",
            f"*{activity_name.lower()}*.csv",
        ]

        activity_file = None
        for pattern in patterns:
            files = list(subject_dir.glob(pattern))
            if files:
                activity_file = files[0]
                break

        if activity_file is None:
            continue

        try:
            # Auto-detect delimiter
            df = None
            for sep in [r'\s+', ',', ';', '\t']:
                try:
                    df = pd.read_csv(activity_file, sep=sep, header=None)
                    if len(df.columns) >= 36:
                        break
                    df = None
                except Exception:
                    continue

            if df is None:
                print(f"    Skipping {activity_name}: could not parse file")
                continue

            # Detect position layout from column count
            positions, raw_channels_per_pos = detect_positions(len(df.columns))
            if detected_positions is None:
                detected_positions = positions
                print(f"    Detected {len(positions)} body positions, "
                      f"{raw_channels_per_pos} raw channels/position "
                      f"({len(df.columns)} total columns)")

            # Extract acc+gyro+mag (first 9 of each position's channels)
            col_names = get_column_names(positions)
            extracted_cols = []
            for pos_idx in range(len(positions)):
                pos_start = pos_idx * raw_channels_per_pos
                # acc(3) + gyro(3) + mag(3) = first 9 channels of each position
                for ch_offset in range(CHANNELS_PER_POSITION):
                    col_idx = pos_start + ch_offset
                    if col_idx < len(df.columns):
                        extracted_cols.append(col_idx)

            if len(extracted_cols) != len(col_names):
                print(f"    Warning: column mismatch for {activity_name}: "
                      f"expected {len(col_names)}, got {len(extracted_cols)}")
                continue

            data = df.iloc[:, extracted_cols].copy()
            data.columns = col_names

            # Validate sensor data ranges
            sensor_vals = data.values
            if np.all(sensor_vals == 0):
                print(f"    Warning: all-zero data in {activity_name}")
            elif np.any(np.isnan(sensor_vals)):
                nan_pct = np.isnan(sensor_vals).mean() * 100
                print(f"    Warning: {nan_pct:.1f}% NaN values in {activity_name}")

            # Interpolate any NaN values
            data = data.interpolate(method='linear', limit_direction='both')
            data = data.fillna(0)

            # Add activity and subject info
            data['activity'] = std_activity
            data['subject'] = subject_dir.name

            all_data.append(data)
            print(f"    Loaded {activity_name}: {len(data)} samples")

        except Exception as e:
            print(f"    Error loading {activity_file}: {e}")
            continue

    if not all_data:
        return pd.DataFrame(), BODY_POSITIONS_4

    return pd.concat(all_data, ignore_index=True), detected_positions or BODY_POSITIONS_4


def convert_subject(subject_dir: Path):
    """Convert one subject's data to sessions with variable-length windowing."""
    print(f"  Processing: {subject_dir.name}")

    # Load all activity data for this subject
    df, positions = load_subject_data(subject_dir)

    if df.empty:
        print(f"    No data found for {subject_dir.name}")
        return [], {}, []

    # Segment by activity (already separated in the raw data)
    session_data = []
    labels_dict = {}
    total_windows = 0

    for activity, group in df.groupby('activity'):
        group = group.reset_index(drop=True)
        subject_id = subject_dir.name

        # Create base session ID
        base_session_id = f"shoaib_{subject_id}_{activity}"

        # Prepare DataFrame with timestamp
        data = group.copy()
        data.insert(0, 'timestamp_sec', np.arange(len(data)) / SAMPLE_RATE)

        # Drop activity and subject columns (stored in labels.json)
        data = data.drop(columns=['activity', 'subject'])

        # Apply variable-length windowing
        windows = create_variable_windows(
            df=data,
            session_prefix=base_session_id,
            activity=activity,
            sample_rate=SAMPLE_RATE,
        )

        # Save each window
        for window_id, window_df, window_activity in windows:
            session_dir = OUTPUT_DIR / "sessions" / window_id
            session_dir.mkdir(parents=True, exist_ok=True)

            parquet_path = session_dir / "data.parquet"
            window_df.to_parquet(parquet_path, index=False)

            # Store label
            labels_dict[window_id] = [window_activity]
            session_data.append(window_id)
            total_windows += 1

    print(f"    Created {total_windows} windows")
    return session_data, labels_dict, positions


def create_manifest(positions):
    """Create minimal manifest.json."""
    channels = []

    pos_descriptions = {
        "right_pocket": "right trouser pocket",
        "left_pocket": "left trouser pocket",
        "pocket": "trouser pocket",
        "belt": "belt-mounted on right leg",
        "arm": "right upper arm",
        "wrist": "right wrist",
    }

    for pos in positions:
        pos_desc = pos_descriptions.get(pos, pos)

        # Accelerometer
        for axis in ["x", "y", "z"]:
            channels.append({
                "name": f"{pos}_acc_{axis}",
                "description": f"Accelerometer {axis.upper()}-axis from {pos_desc} smartphone sensor",
                "sampling_rate_hz": SAMPLE_RATE
            })

        # Gyroscope
        for axis in ["x", "y", "z"]:
            channels.append({
                "name": f"{pos}_gyro_{axis}",
                "description": f"Gyroscope {axis.upper()}-axis from {pos_desc} smartphone sensor",
                "sampling_rate_hz": SAMPLE_RATE
            })

        # Magnetometer
        for axis in ["x", "y", "z"]:
            channels.append({
                "name": f"{pos}_mag_{axis}",
                "description": f"Magnetometer {axis.upper()}-axis from {pos_desc} smartphone sensor",
                "sampling_rate_hz": SAMPLE_RATE
            })

    n_pos = len(positions)
    manifest = {
        "dataset_name": "Shoaib",
        "description": f"Activity recognition with multiple body-worn smartphones. "
                       f"10 subjects performing 7 physical activities. "
                       f"Sensors at {n_pos} body positions ({', '.join(positions)}) "
                       f"with accelerometer, gyroscope, and magnetometer at 50 Hz.",
        "channels": channels
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Created manifest: {manifest_path}")
    print(f"  {len(channels)} channels ({n_pos} positions x 9 sensors)")


def main():
    """Convert Shoaib dataset to standardized format."""
    print("=" * 80)
    print("Shoaib -> Standardized Format Converter")
    print("=" * 80)
    print("NOTE: This dataset is used for ZERO-SHOT TESTING (not training)")

    # Check input
    if not RAW_DIR.exists():
        print(f"ERROR: Raw data not found at {RAW_DIR}")
        print("Download from: https://www.utwente.nl/en/eemcs/ps/research/dataset/")
        print("File: sensors-activity-recognition-dataset-shoaib.rar")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find subject directories
    subject_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])

    if not subject_dirs:
        # Maybe the data is in the root directory with subject prefixes
        print("Looking for subject files in root directory...")
        subject_dirs = [RAW_DIR]

    print(f"\nFound {len(subject_dirs)} subject directories")

    all_labels = {}
    all_sessions = []
    detected_positions = None

    for subject_dir in subject_dirs:
        sessions, labels, positions = convert_subject(subject_dir)
        all_sessions.extend(sessions)
        all_labels.update(labels)
        if positions and detected_positions is None:
            detected_positions = positions

    if not all_labels:
        print("\nNo sessions created. Check the raw data format.")
        return

    positions = detected_positions or BODY_POSITIONS_4

    # Save labels.json
    labels_path = OUTPUT_DIR / "labels.json"
    with open(labels_path, 'w') as f:
        json.dump(all_labels, f, indent=2)

    print(f"\nCreated labels: {labels_path}")
    print(f"  Total sessions: {len(all_labels)}")

    # Create manifest
    create_manifest(positions)

    print(f"\n{'=' * 80}")
    print("Conversion complete!")
    print(f"{'=' * 80}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"  - {len(all_labels)} sessions")
    print(f"  - {len(positions) * CHANNELS_PER_POSITION} channels "
          f"({len(positions)} positions x {CHANNELS_PER_POSITION} sensors)")
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
