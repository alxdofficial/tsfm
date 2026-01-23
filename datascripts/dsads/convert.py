"""
Convert DSADS (Daily and Sports Activities) dataset to standardized format.

Input: data/raw/dsads/data/
Output: data/dsads/
  - manifest.json (channel metadata)
  - labels.json (activity labels per session)
  - sessions/session_XXX/data.parquet (sensor data)

DSADS format:
  - 19 activities (a01-a19), 8 subjects (p1-p8), 60 segments (s01-s60)
  - Each segment: 45 columns (5 units × 9 sensors), 125 rows (5 sec × 25 Hz)
  - Column order: T (torso), RA (right arm), LA (left arm), RL (right leg), LL (left leg)
  - Each unit: xacc, yacc, zacc, xgyro, ygyro, zgyro, xmag, ymag, zmag
  - Sampling rate: 25 Hz

We extract only the TORSO (T) sensors for consistency with other datasets.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Activity mapping (a01-a19)
ACTIVITIES = {
    "a01": "sitting",
    "a02": "standing",
    "a03": "lying_back",
    "a04": "lying_side",
    "a05": "stairs_up",
    "a06": "stairs_down",
    "a07": "standing_elevator",
    "a08": "moving_elevator",
    "a09": "walking_parking",
    "a10": "walking_treadmill_flat",
    "a11": "walking_treadmill_incline",
    "a12": "running_treadmill",
    "a13": "exercising_stepper",
    "a14": "exercising_cross_trainer",
    "a15": "cycling_horizontal",
    "a16": "cycling_vertical",
    "a17": "rowing",
    "a18": "jumping",
    "a19": "playing_basketball",
}

# Column indices for torso sensors (first 9 columns, 0-indexed)
TORSO_COLUMNS = {
    "acc_x": 0,
    "acc_y": 1,
    "acc_z": 2,
    "gyro_x": 3,
    "gyro_y": 4,
    "gyro_z": 5,
    "mag_x": 6,
    "mag_y": 7,
    "mag_z": 8,
}

# Paths
RAW_DIR = Path("data/raw/dsads")
OUTPUT_DIR = Path("data/dsads")

# Dataset parameters
SAMPLE_RATE = 25.0  # Hz
SEGMENT_DURATION = 5.0  # seconds
SAMPLES_PER_SEGMENT = 125


def find_data_dir() -> Optional[Path]:
    """Find the data directory in raw folder."""
    for candidate in ["data", "daily+and+sports+activities", "Daily and Sports Activities"]:
        path = RAW_DIR / candidate
        if path.exists():
            return path
    return None


def load_segment_file(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Load a DSADS segment file.

    Each file has 45 columns (comma-separated) and 125 rows.
    We extract only the first 9 columns (torso sensors).
    """
    try:
        # Load all columns
        data = np.loadtxt(filepath, delimiter=",")

        if data.shape != (SAMPLES_PER_SEGMENT, 45):
            print(f"    Warning: Unexpected shape {data.shape} in {filepath}")
            # Try to handle anyway
            if data.shape[0] < 10:
                return None

        # Extract torso columns and build DataFrame
        result = pd.DataFrame()

        # Create timestamp column
        num_samples = data.shape[0]
        result["timestamp_sec"] = np.linspace(0, (num_samples - 1) / SAMPLE_RATE, num_samples)

        # Extract torso sensor columns
        for col_name, col_idx in TORSO_COLUMNS.items():
            if col_idx < data.shape[1]:
                result[col_name] = data[:, col_idx]

        return result

    except Exception as e:
        print(f"    ERROR loading {filepath}: {e}")
        return None


def convert_dsads():
    """Convert DSADS dataset to standardized format."""
    print("=" * 80)
    print("DSADS → Standardized Format Converter")
    print("=" * 80)

    # Find data directory
    data_dir = find_data_dir()
    if data_dir is None:
        print(f"ERROR: Raw data not found at {RAW_DIR}")
        print("Run: python datascripts/dsads/download.py")
        return False

    print(f"Data directory: {data_dir}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sessions_dir = OUTPUT_DIR / "sessions"
    sessions_dir.mkdir(exist_ok=True)

    # Find all activity folders
    activity_folders = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("a")])
    print(f"\nFound {len(activity_folders)} activities")

    all_labels = {}
    session_count = 0
    skipped_count = 0

    for activity_folder in activity_folders:
        activity_code = activity_folder.name.lower()

        if activity_code not in ACTIVITIES:
            print(f"  Skipping unknown activity: {activity_code}")
            continue

        activity_name = ACTIVITIES[activity_code]
        print(f"\n  Processing {activity_code} ({activity_name})")

        # Find subject folders
        subject_folders = sorted([d for d in activity_folder.iterdir() if d.is_dir() and d.name.startswith("p")])

        for subject_folder in subject_folders:
            subject_id = int(subject_folder.name[1:])  # p1 -> 1

            # Find segment files
            segment_files = sorted(subject_folder.glob("s*.txt"))

            for segment_file in segment_files:
                segment_id = segment_file.stem  # s01, s02, etc.

                # Load segment
                df = load_segment_file(segment_file)
                if df is None or len(df) < 10:
                    skipped_count += 1
                    continue

                # Create session ID
                session_id = f"{activity_code}_p{subject_id:02d}_{segment_id}"

                # Save to parquet
                session_path = sessions_dir / session_id
                session_path.mkdir(exist_ok=True)

                parquet_path = session_path / "data.parquet"
                df.to_parquet(parquet_path, index=False)

                # Store label
                all_labels[session_id] = [activity_name]
                session_count += 1

        print(f"    Sessions so far: {session_count}")

    # Save labels.json
    labels_path = OUTPUT_DIR / "labels.json"
    with open(labels_path, "w") as f:
        json.dump(all_labels, f, indent=2)
    print(f"\n✓ Created labels.json ({len(all_labels)} sessions)")

    # Create manifest.json
    create_manifest()

    # Summary
    print(f"\n{'=' * 80}")
    print("Conversion complete!")
    print(f"{'=' * 80}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"  - {session_count} sessions converted")
    print(f"  - {skipped_count} segments skipped")
    print(f"  - Body position: torso")
    print(f"  - {SAMPLE_RATE} Hz sampling rate")

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
        "dataset_name": "DSADS",
        "description": "Daily and Sports Activities Dataset from UCI ML Repository. 8 subjects performing 19 daily and sports activities. Xsens MTx sensors on torso. Triaxial accelerometer, gyroscope, and magnetometer at 25Hz.",
        "source": "https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities",
        "num_subjects": 8,
        "body_position": "torso",
        "channels": [
            {"name": "acc_x", "description": "Accelerometer X-axis (torso)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "acc_y", "description": "Accelerometer Y-axis (torso)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "acc_z", "description": "Accelerometer Z-axis (torso)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "gyro_x", "description": "Gyroscope X-axis (torso)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "gyro_y", "description": "Gyroscope Y-axis (torso)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "gyro_z", "description": "Gyroscope Z-axis (torso)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "mag_x", "description": "Magnetometer X-axis (torso)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "mag_y", "description": "Magnetometer Y-axis (torso)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "mag_z", "description": "Magnetometer Z-axis (torso)", "sampling_rate_hz": SAMPLE_RATE},
        ],
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"✓ Created manifest.json")


def main():
    """Main entry point."""
    success = convert_dsads()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
