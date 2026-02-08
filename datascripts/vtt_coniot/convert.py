"""
Convert VTT-ConIoT dataset to standardized format.

Input: data/raw/vtt_coniot/
Output: data/vtt_coniot/
  - manifest.json (channel metadata)
  - labels.json (activity labels per session)
  - sessions/session_XXX/data.parquet (sensor data)

VTT-ConIoT format:
  - 13 construction workers
  - 16 activities grouped into 6 main tasks
  - 3 body positions: hip, upper arm, back of shoulder
  - Sensors: accelerometer (~103Hz), gyroscope (~97Hz), magnetometer (~97Hz)
  - Synchronized to ~100 Hz

We extract data from the HIP position (most similar to waist).
We resample to 50 Hz for consistency with other datasets.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Activity mapping (based on VTT-ConIoT paper)
# 16 activities grouped into 6 main categories
ACTIVITIES = {
    # Painting activities
    "1": "roll_painting",
    "2": "spraying_paint",
    "3": "leveling_paint",
    # Walking/displacement
    "4": "pushing_cart",
    "5": "walking_straight",
    "6": "walking_winding",
    # Standing/working
    "7": "standing_work",
    "8": "kneeling_work",
    "9": "laying_back",
    # Lifting/carrying
    "10": "lifting",
    "11": "carrying",
    "12": "raising_hands",
    # Miscellaneous
    "13": "sitting",
    "14": "jumping_down",
    "15": "climbing_ladder",
    "16": "stairs",
}

# Alternative activity name mapping (for text-based labels)
ACTIVITY_NAME_MAP = {
    "roll_painting": "roll_painting",
    "rollpainting": "roll_painting",
    "paint_roll": "roll_painting",
    "spray_painting": "spraying_paint",
    "spraypainting": "spraying_paint",
    "paint_spray": "spraying_paint",
    "leveling": "leveling_paint",
    "paint_leveling": "leveling_paint",
    "push_cart": "pushing_cart",
    "pushcart": "pushing_cart",
    "cart_push": "pushing_cart",
    "walking": "walking_straight",
    "walk_straight": "walking_straight",
    "walk_winding": "walking_winding",
    "winding_walk": "walking_winding",
    "standing": "standing_work",
    "kneeling": "kneeling_work",
    "laying": "laying_back",
    "lying": "laying_back",
    "lift": "lifting",
    "carry": "carrying",
    "raise_hands": "raising_hands",
    "hands_raised": "raising_hands",
    "sit": "sitting",
    "jump": "jumping_down",
    "jumpdown": "jumping_down",
    "climb": "climbing_ladder",
    "ladder": "climbing_ladder",
    "stair": "stairs",
}

# Target body position
TARGET_POSITION = "hip"

# Paths
RAW_DIR = Path("data/raw/vtt_coniot")
OUTPUT_DIR = Path("data/vtt_coniot")

# Dataset parameters
ORIGINAL_SAMPLE_RATE = 100.0  # Hz (synchronized)
TARGET_SAMPLE_RATE = 50.0  # Hz (for consistency)


def find_data_files() -> List[Path]:
    """Find all CSV data files in the raw directory."""
    csv_files = list(RAW_DIR.glob("**/*.csv"))
    return csv_files


def normalize_activity_name(activity: str) -> str:
    """Normalize activity name to standard format."""
    # Try direct mapping first
    activity_lower = str(activity).lower().strip()

    # Check numeric code
    if activity_lower in ACTIVITIES:
        return ACTIVITIES[activity_lower]

    # Check name mapping
    activity_clean = activity_lower.replace(" ", "_").replace("-", "_")
    if activity_clean in ACTIVITY_NAME_MAP:
        return ACTIVITY_NAME_MAP[activity_clean]

    # Check partial matches
    for key, value in ACTIVITY_NAME_MAP.items():
        if key in activity_clean or activity_clean in key:
            return value

    # Return cleaned version if no match
    return activity_clean


def parse_vtt_filename(filepath: Path) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse VTT-ConIoT filename to extract activity and user IDs.

    Filename format: activity_9_user_3_combined.csv
    Returns (activity_id, user_id) or (None, None) if parsing fails.
    """
    import re

    filename = filepath.stem.lower()

    # Try to match activity_X_user_Y pattern
    match = re.match(r"activity_(\d+)_user_(\d+)", filename)
    if match:
        return int(match.group(1)), int(match.group(2))

    return None, None


def load_vtt_csv(filepath: Path) -> Optional[Tuple[pd.DataFrame, str, int]]:
    """
    Load a VTT-ConIoT CSV file.

    Returns (DataFrame, activity_label, user_id) or None if failed.
    """
    try:
        # Parse filename for activity and user
        activity_id, user_id = parse_vtt_filename(filepath)
        if activity_id is None:
            return None

        # Get activity name from ID
        activity_name = ACTIVITIES.get(str(activity_id), f"activity_{activity_id}")

        # Read CSV
        df = pd.read_csv(filepath)

        if len(df) < 10:
            return None

        # VTT-ConIoT format uses trousers_ prefix for hip/pocket position
        # Column names: trousers_Ax_g, trousers_Ay_g, trousers_Az_g (accelerometer in g)
        #               trousers_Gx_dps, trousers_Gy_dps, trousers_Gz_dps (gyroscope in deg/s)
        #               trousers_Mx_uT, trousers_My_uT, trousers_Mz_uT (magnetometer in microTesla)

        result = pd.DataFrame()

        # Create timestamp from row index (100Hz = 0.01s per sample)
        result["timestamp_sec"] = np.arange(len(df)) / ORIGINAL_SAMPLE_RATE

        # Map VTT-ConIoT columns to standard names
        # Use trousers (hip/pocket position) for consistency with other datasets
        column_mapping = {
            "acc_x": "trousers_Ax_g",
            "acc_y": "trousers_Ay_g",
            "acc_z": "trousers_Az_g",
            "gyro_x": "trousers_Gx_dps",
            "gyro_y": "trousers_Gy_dps",
            "gyro_z": "trousers_Gz_dps",
            "mag_x": "trousers_Mx_uT",
            "mag_y": "trousers_My_uT",
            "mag_z": "trousers_Mz_uT",
        }

        for target_col, source_col in column_mapping.items():
            if source_col in df.columns:
                result[target_col] = df[source_col].values

        # Check we have at least accelerometer data
        if "acc_x" not in result.columns:
            return None

        return result, activity_name, user_id if user_id else 0

    except Exception as e:
        print(f"    ERROR loading {filepath}: {e}")
        return None


def resample_to_target_rate(df: pd.DataFrame, target_rate: float = 50.0) -> Optional[pd.DataFrame]:
    """Resample data to target sampling rate."""
    if df is None or len(df) == 0:
        return None

    duration = df["timestamp_sec"].iloc[-1] - df["timestamp_sec"].iloc[0]
    if duration <= 0:
        return None

    num_samples = int(duration * target_rate) + 1
    new_timestamps = np.linspace(0, duration, num_samples)

    result = pd.DataFrame()
    result["timestamp_sec"] = new_timestamps

    for col in df.columns:
        if col != "timestamp_sec":
            result[col] = np.interp(new_timestamps, df["timestamp_sec"].values, df[col].values)

    return result


def convert_vtt_coniot():
    """Convert VTT-ConIoT dataset to standardized format."""
    print("=" * 80)
    print("VTT-ConIoT → Standardized Format Converter")
    print("=" * 80)

    # Check input
    if not RAW_DIR.exists():
        print(f"ERROR: Raw data not found at {RAW_DIR}")
        print("Run: python datascripts/vtt_coniot/download.py")
        return False

    # Find data files
    csv_files = find_data_files()
    if not csv_files:
        print(f"ERROR: No CSV files found in {RAW_DIR}")
        return False

    print(f"Found {len(csv_files)} CSV files")

    # Analyze file structure
    print("\nAnalyzing file structure...")
    for i, f in enumerate(csv_files[:3]):
        print(f"  Sample file {i+1}: {f}")
        try:
            df = pd.read_csv(f, nrows=5)
            print(f"    Columns: {list(df.columns)[:10]}...")
        except Exception as e:
            print(f"    Error reading: {e}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sessions_dir = OUTPUT_DIR / "sessions"
    sessions_dir.mkdir(exist_ok=True)

    all_labels = {}
    session_count = 0
    skipped_count = 0

    # Process each file
    for csv_file in csv_files:
        # Load file
        result = load_vtt_csv(csv_file)
        if result is None:
            skipped_count += 1
            continue

        df, activity, user_id = result

        # Resample to target rate
        df = resample_to_target_rate(df, TARGET_SAMPLE_RATE)
        if df is None or len(df) < 10:
            skipped_count += 1
            continue

        # Create session ID: user_activity format
        session_id = f"u{user_id:02d}_{activity}"

        # Save to parquet
        session_path = sessions_dir / session_id
        session_path.mkdir(exist_ok=True)

        parquet_path = session_path / "data.parquet"
        df.to_parquet(parquet_path, index=False)

        # Store label
        all_labels[session_id] = [activity]
        session_count += 1

        if session_count % 50 == 0:
            print(f"  Processed {session_count} sessions...")

    # Save labels.json
    labels_path = OUTPUT_DIR / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(all_labels, f, indent=2)
    print(f"\n✓ Created labels.json ({len(all_labels)} sessions)")

    # Create manifest.json
    create_manifest()

    # Count unique activities
    activities = set()
    for session_labels in all_labels.values():
        activities.update(session_labels)

    # Summary
    print(f"\n{'=' * 80}")
    print("Conversion complete!")
    print(f"{'=' * 80}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"  - {session_count} sessions converted")
    print(f"  - {skipped_count} files skipped")
    print(f"  - {len(activities)} unique activities")
    print(f"  - {TARGET_SAMPLE_RATE} Hz sampling rate")

    # Generate debug visualizations
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from shared.visualization_utils import generate_debug_visualizations

        generate_debug_visualizations(OUTPUT_DIR)
    except ImportError as e:
        print(f"\n⚠ Could not generate visualizations: {e}")
        print("Install matplotlib: pip install matplotlib")

    return True


def create_manifest():
    """Create manifest.json with channel metadata."""
    manifest = {
        "dataset_name": "VTT-ConIoT",
        "description": "VTT-ConIoT: Construction Workers Activity Recognition Dataset. 13 workers performing 16 construction activities. IMU sensors at hip position. Triaxial accelerometer, gyroscope, and magnetometer resampled to 50Hz.",
        "source": "https://zenodo.org/record/4683703",
        "num_subjects": 13,
        "body_position": TARGET_POSITION,
        "channels": [
            {"name": "acc_x", "description": "Accelerometer X-axis (hip)", "sampling_rate_hz": TARGET_SAMPLE_RATE},
            {"name": "acc_y", "description": "Accelerometer Y-axis (hip)", "sampling_rate_hz": TARGET_SAMPLE_RATE},
            {"name": "acc_z", "description": "Accelerometer Z-axis (hip)", "sampling_rate_hz": TARGET_SAMPLE_RATE},
            {"name": "gyro_x", "description": "Gyroscope X-axis (hip)", "sampling_rate_hz": TARGET_SAMPLE_RATE},
            {"name": "gyro_y", "description": "Gyroscope Y-axis (hip)", "sampling_rate_hz": TARGET_SAMPLE_RATE},
            {"name": "gyro_z", "description": "Gyroscope Z-axis (hip)", "sampling_rate_hz": TARGET_SAMPLE_RATE},
            {"name": "mag_x", "description": "Magnetometer X-axis (hip)", "sampling_rate_hz": TARGET_SAMPLE_RATE},
            {"name": "mag_y", "description": "Magnetometer Y-axis (hip)", "sampling_rate_hz": TARGET_SAMPLE_RATE},
            {"name": "mag_z", "description": "Magnetometer Z-axis (hip)", "sampling_rate_hz": TARGET_SAMPLE_RATE},
        ],
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"✓ Created manifest.json")


def main():
    """Main entry point."""
    success = convert_vtt_coniot()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
