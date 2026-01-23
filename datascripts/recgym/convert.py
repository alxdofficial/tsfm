"""
Convert RecGym dataset to standardized format.

Input: data/raw/recgym/
Output: data/recgym/
  - manifest.json (channel metadata)
  - labels.json (activity labels per session)
  - sessions/session_XXX/data.parquet (sensor data)

RecGym format:
  - Single CSV file with ~4.4M instances
  - Columns: Object, Workout, Position, A_x, A_y, A_z, G_x, G_y, G_z, C_1, Session
  - 12 activities including Null
  - 10 volunteers, 3 positions (wrist, pocket, calf)
  - 20 Hz sampling rate

We extract only the WRIST position for consistency with other datasets.
We exclude the "Null" activity (downtime between exercises).
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add parent to path for shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.windowing import create_variable_windows


# Activity mapping (standardized names)
ACTIVITIES = {
    "Adductor": "adductor_machine",
    "ArmCurl": "arm_curl",
    "BenchPress": "bench_press",
    "LegCurl": "leg_curl",
    "LegPress": "leg_press",
    "Riding": "cycling",
    "RopeSkipping": "rope_skipping",
    "Running": "running",
    "Squat": "squat",
    "StairsClimber": "stairclimber",
    "Walking": "walking",
    # "Null" is excluded - represents downtime
}

# Target body position
TARGET_POSITION = "wrist"

# Paths
RAW_DIR = Path("data/raw/recgym")
OUTPUT_DIR = Path("data/recgym")

# Dataset parameters
SAMPLE_RATE = 20.0  # Hz


def find_csv_file() -> Optional[Path]:
    """Find the RecGym CSV file."""
    # Try various possible filenames
    candidates = list(RAW_DIR.glob("*.csv")) + list(RAW_DIR.glob("**/*.csv"))
    if candidates:
        return candidates[0]
    return None


def convert_recgym():
    """Convert RecGym dataset to standardized format."""
    print("=" * 80)
    print("RecGym → Standardized Format Converter")
    print("=" * 80)

    # Find CSV file
    csv_file = find_csv_file()
    if csv_file is None:
        print(f"ERROR: No CSV file found in {RAW_DIR}")
        print("Run: python datascripts/recgym/download.py")
        return False

    print(f"Loading: {csv_file}")

    # Load the full dataset
    try:
        df = pd.read_csv(csv_file)
        print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
        print(f"  Columns: {list(df.columns)}")
    except Exception as e:
        print(f"ERROR: Failed to load CSV: {e}")
        return False

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Find relevant columns
    # Expected: Object, Workout, Position, A_x, A_y, A_z, G_x, G_y, G_z, C_1, Session
    required_cols = ["A_x", "A_y", "A_z", "G_x", "G_y", "G_z"]
    for col in required_cols:
        if col not in df.columns:
            # Try lowercase
            lower_col = col.lower()
            if lower_col in df.columns:
                df.rename(columns={lower_col: col}, inplace=True)
            else:
                print(f"ERROR: Required column '{col}' not found")
                return False

    # Find position column
    pos_col = None
    for candidate in ["Position", "position", "Pos"]:
        if candidate in df.columns:
            pos_col = candidate
            break

    if pos_col is None:
        print("WARNING: Position column not found, using all data")

    # Find workout/activity column
    workout_col = None
    for candidate in ["Workout", "workout", "Activity", "activity"]:
        if candidate in df.columns:
            workout_col = candidate
            break

    if workout_col is None:
        print("ERROR: Workout/Activity column not found")
        return False

    # Find subject column
    subject_col = None
    for candidate in ["Object", "object", "Subject", "subject", "Participant"]:
        if candidate in df.columns:
            subject_col = candidate
            break

    if subject_col is None:
        print("WARNING: Subject column not found, will use single subject")
        df["subject"] = 1
        subject_col = "subject"

    # Filter for target position (if position column exists)
    if pos_col:
        unique_positions = df[pos_col].unique()
        print(f"  Positions: {list(unique_positions)}")

        # Find wrist position (case-insensitive)
        target_pos = None
        for pos in unique_positions:
            if "wrist" in str(pos).lower():
                target_pos = pos
                break

        if target_pos:
            df = df[df[pos_col] == target_pos]
            print(f"  Filtered to {TARGET_POSITION}: {len(df):,} rows")
        else:
            print(f"  WARNING: No wrist position found, using first position: {unique_positions[0]}")
            df = df[df[pos_col] == unique_positions[0]]

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sessions_dir = OUTPUT_DIR / "sessions"
    sessions_dir.mkdir(exist_ok=True)

    # Get unique workouts and subjects
    unique_workouts = df[workout_col].unique()
    unique_subjects = df[subject_col].unique()
    print(f"\n  Workouts: {list(unique_workouts)}")
    print(f"  Subjects: {len(unique_subjects)}")

    all_labels = {}
    session_count = 0
    skipped_count = 0

    # Process each subject and workout combination
    for subject in unique_subjects:
        subject_df = df[df[subject_col] == subject]

        for workout in unique_workouts:
            # Skip Null activity
            if str(workout).lower() == "null":
                continue

            workout_df = subject_df[subject_df[workout_col] == workout]
            if len(workout_df) < 20:  # Skip very short segments
                skipped_count += 1
                continue

            # Map workout to standardized name
            activity_name = ACTIVITIES.get(workout, str(workout).lower().replace(" ", "_"))

            # Create session ID
            subject_id = int(subject) if isinstance(subject, (int, float)) else hash(str(subject)) % 1000
            session_id = f"s{subject_id:02d}_{activity_name}"

            # Build DataFrame with standardized column names
            result = pd.DataFrame()

            # Create timestamp column
            num_samples = len(workout_df)
            result["timestamp_sec"] = np.linspace(0, (num_samples - 1) / SAMPLE_RATE, num_samples)

            # Extract sensor columns
            result["acc_x"] = workout_df["A_x"].values
            result["acc_y"] = workout_df["A_y"].values
            result["acc_z"] = workout_df["A_z"].values
            result["gyro_x"] = workout_df["G_x"].values
            result["gyro_y"] = workout_df["G_y"].values
            result["gyro_z"] = workout_df["G_z"].values

            # Split long sessions into variable-length windows
            windows = create_variable_windows(
                df=result,
                session_prefix=session_id,
                activity=activity_name,
                sample_rate=SAMPLE_RATE,
                seed=42 + hash(session_id) % 1000,  # Reproducible but varied
            )

            # Save each window as a separate session
            for window_id, window_df, window_activity in windows:
                window_path = sessions_dir / window_id
                window_path.mkdir(exist_ok=True)

                parquet_path = window_path / "data.parquet"
                window_df.to_parquet(parquet_path, index=False)

                # Store label
                all_labels[window_id] = [window_activity]
                session_count += 1

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
    print(f"  - Body position: {TARGET_POSITION}")
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
        "dataset_name": "RecGym",
        "description": "RecGym: Gym Workouts Recognition Dataset. 10 volunteers performing 11 gym exercises over 5 sessions. IMU sensors at wrist position. Triaxial accelerometer and gyroscope at 20Hz.",
        "source": "https://archive.ics.uci.edu/dataset/1128/recgym:+gym+workouts+recognition+dataset+with+imu+and+capacitive+sensor-7",
        "num_subjects": 10,
        "body_position": TARGET_POSITION,
        "channels": [
            {"name": "acc_x", "description": "Accelerometer X-axis (wrist)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "acc_y", "description": "Accelerometer Y-axis (wrist)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "acc_z", "description": "Accelerometer Z-axis (wrist)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "gyro_x", "description": "Gyroscope X-axis (wrist)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "gyro_y", "description": "Gyroscope Y-axis (wrist)", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "gyro_z", "description": "Gyroscope Z-axis (wrist)", "sampling_rate_hz": SAMPLE_RATE},
        ],
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"✓ Created manifest.json")


def main():
    """Main entry point."""
    success = convert_recgym()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
