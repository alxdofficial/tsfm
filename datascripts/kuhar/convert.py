"""
Convert KU-HAR dataset to standardized format.

Input: data/raw/kuhar/
Output: data/kuhar/
  - manifest.json (channel metadata)
  - labels.json (activity labels per session)
  - sessions/session_XXX/data.parquet (sensor data)

KU-HAR format:
  - Folders organized as: N.ActivityName (e.g., "16.Stair-down")
  - CSV files: timestamp, acc_x, acc_y, acc_z, timestamp2, gyro_x, gyro_y, gyro_z
  - No headers in CSV files
  - 90 subjects, 18 activities
  - 100 Hz sampling rate
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add parent to path for shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.windowing import create_variable_windows


# Activity mapping - folder name to standardized name
ACTIVITIES = {
    "Stand": "standing",
    "Sit": "sitting",
    "Talk-sit": "talking_sitting",
    "Talk-stand": "talking_standing",
    "Stand-sit": "standing_up_from_sitting",
    "Lay": "lying",
    "Lay-stand": "standing_up_from_laying",
    "Pick": "picking_up",
    "Jump": "jumping",
    "Push-up": "push_up",
    "Sit-up": "sit_up",
    "Walk": "walking",
    "Walk-backwards": "walking_backwards",
    "Walk-circle": "walking",  # Map to walking
    "Run": "running",
    "Stair-up": "walking_upstairs",
    "Stair-down": "walking_downstairs",
    "Table-tennis": "playing_sports",
}

# Paths
RAW_DIR = Path("data/raw/kuhar")
OUTPUT_DIR = Path("data/kuhar")

# Dataset parameters
SAMPLE_RATE = 100.0  # Hz


def parse_folder_name(folder_name: str) -> Optional[str]:
    """Parse activity name from folder like '16.Stair-down'."""
    # Remove numeric prefix
    match = re.match(r'\d+\.(.+)', folder_name)
    if match:
        activity_key = match.group(1)
        return ACTIVITIES.get(activity_key)
    return None


def load_csv_file(filepath: Path) -> Optional[pd.DataFrame]:
    """Load and parse CSV file (no header)."""
    try:
        # CSV format: timestamp, acc_x, acc_y, acc_z, timestamp2, gyro_x, gyro_y, gyro_z
        df = pd.read_csv(filepath, header=None)

        if len(df.columns) < 7:
            return None

        # Build standardized DataFrame
        result = pd.DataFrame()

        # Use first timestamp column
        result["timestamp_sec"] = df.iloc[:, 0].values

        # Accelerometer (columns 1-3)
        result["acc_x"] = df.iloc[:, 1].values
        result["acc_y"] = df.iloc[:, 2].values
        result["acc_z"] = df.iloc[:, 3].values

        # Gyroscope (columns 5-7, skipping timestamp2 at column 4)
        result["gyro_x"] = df.iloc[:, 5].values
        result["gyro_y"] = df.iloc[:, 6].values
        result["gyro_z"] = df.iloc[:, 7].values

        return result

    except Exception as e:
        print(f"  ERROR loading {filepath}: {e}")
        return None


def find_data_files(raw_dir: Path) -> List[Dict]:
    """Find all data files with activity labels."""
    data_files = []

    # Look in Raw_time_domian_data folder
    raw_data_dir = raw_dir / "1.Raw_time_domian_data"
    if not raw_data_dir.exists():
        # Try without the typo
        raw_data_dir = raw_dir / "1.Raw_time_domain_data"
    if not raw_data_dir.exists():
        # Just use any subfolder with activity folders
        for subdir in raw_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("."):
                raw_data_dir = subdir
                break

    if not raw_data_dir or not raw_data_dir.exists():
        return []

    print(f"Using data directory: {raw_data_dir}")

    # Iterate through activity folders
    for activity_folder in raw_data_dir.iterdir():
        if not activity_folder.is_dir():
            continue

        activity = parse_folder_name(activity_folder.name)
        if not activity:
            continue

        # Find CSV files in this activity folder
        for csv_file in activity_folder.glob("*.csv"):
            # Extract subject ID from filename (e.g., "1038_T_1.csv" -> subject 1038)
            subject_match = re.match(r'(\d+)_', csv_file.stem)
            subject = int(subject_match.group(1)) if subject_match else hash(csv_file.stem) % 10000

            data_files.append({
                "filepath": csv_file,
                "activity": activity,
                "subject": subject,
            })

    return data_files


def convert_kuhar():
    """Convert KU-HAR dataset to standardized format."""
    print("=" * 80)
    print("KU-HAR → Standardized Format Converter")
    print("=" * 80)

    # Check input
    if not RAW_DIR.exists():
        print(f"ERROR: Raw data not found at {RAW_DIR}")
        print("Run: python datascripts/kuhar/download.py")
        return False

    # Find data files
    data_files = find_data_files(RAW_DIR)
    print(f"Found {len(data_files)} data files")

    if len(data_files) == 0:
        print("ERROR: No valid data files found")
        return False

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sessions_dir = OUTPUT_DIR / "sessions"
    sessions_dir.mkdir(exist_ok=True)

    all_labels = {}
    session_count = 0
    skipped_count = 0
    activity_counts = {}

    for i, file_info in enumerate(data_files):
        filepath = file_info["filepath"]
        activity = file_info["activity"]
        subject = file_info["subject"]

        # Load CSV
        df = load_csv_file(filepath)
        if df is None or len(df) < 20:
            skipped_count += 1
            continue

        # Create session ID prefix
        session_prefix = f"s{subject:04d}_{activity}_{i:04d}"

        # Apply windowing
        windows = create_variable_windows(
            df=df,
            session_prefix=session_prefix,
            activity=activity,
            sample_rate=SAMPLE_RATE,
            seed=42 + subject + i,
        )

        # Save each window
        for window_id, window_df, window_activity in windows:
            window_path = sessions_dir / window_id
            window_path.mkdir(exist_ok=True)

            parquet_path = window_path / "data.parquet"
            window_df.to_parquet(parquet_path, index=False)

            all_labels[window_id] = [window_activity]
            session_count += 1

            # Track activity counts
            activity_counts[window_activity] = activity_counts.get(window_activity, 0) + 1

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(data_files)} files, {session_count} sessions...")

    # Save labels.json
    labels_path = OUTPUT_DIR / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(all_labels, f, indent=2)
    print(f"\n✓ Created labels.json ({len(all_labels)} sessions)")

    # Create manifest.json
    create_manifest()

    print(f"\n{'=' * 80}")
    print("Conversion complete!")
    print(f"{'=' * 80}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"  - {session_count} sessions converted")
    print(f"  - {skipped_count} files skipped")
    print(f"  - {len(activity_counts)} unique activities")
    print(f"  - {SAMPLE_RATE} Hz sampling rate")
    print(f"\nActivity distribution:")
    for activity, count in sorted(activity_counts.items(), key=lambda x: -x[1]):
        print(f"  {activity}: {count}")

    # Generate visualizations
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from shared.visualization_utils import generate_debug_visualizations
        generate_debug_visualizations(OUTPUT_DIR)
    except ImportError:
        pass

    return True


def create_manifest():
    """Create manifest.json with channel metadata."""
    manifest = {
        "dataset_name": "KU-HAR",
        "description": "KU-HAR (Korea University Human Activity Recognition) dataset. 90 subjects performing 18 activities including walking, running, jumping, stairs, sitting, standing, lying, and exercises (push-ups, sit-ups). Smartphone IMU at 100Hz with triaxial accelerometer and gyroscope.",
        "source": "https://www.kaggle.com/datasets/niloy333/kuhar",
        "num_subjects": 90,
        "channels": [
            {"name": "acc_x", "description": "Accelerometer X-axis", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "acc_y", "description": "Accelerometer Y-axis", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "acc_z", "description": "Accelerometer Z-axis", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "gyro_x", "description": "Gyroscope X-axis", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "gyro_y", "description": "Gyroscope Y-axis", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "gyro_z", "description": "Gyroscope Z-axis", "sampling_rate_hz": SAMPLE_RATE},
        ],
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"✓ Created manifest.json")


def main():
    """Main entry point."""
    success = convert_kuhar()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
