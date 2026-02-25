"""
Convert USC-HAD dataset to standardized format.

Input: data/raw/usc_had/
Output: data/usc_had/
  - manifest.json
  - labels.json
  - sessions/session_XXX/data.parquet

USC Human Activity Dataset Info:
- 14 subjects (7 male, 7 female, ages 21-49)
- 12 activities: Walking Forward, Walking Left, Walking Right,
  Walking Upstairs, Walking Downstairs, Running Forward,
  Jumping Up, Sitting, Standing, Sleeping,
  Elevator Up, Elevator Down
- 1 sensor: MotionNode IMU (waist-mounted, front right hip)
- 6 channels: 3-axis accelerometer + 3-axis gyroscope
- 100 Hz sampling rate
- MATLAB .mat format with per-trial files
- Used as ZERO-SHOT TEST set (not for training)

File structure:
  Subject{N}/  (N = 1..14)
    a{A}t{T}.mat  (A = activity 1..12, T = trial 1..5)
  Each .mat contains:
    - sensor_readings: Nx6 array [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    - activity_number: scalar activity ID

Reference:
Mi Zhang and Alexander A. Sawchuk, "USC-HAD: A Daily Activity Dataset for
Ubiquitous Activity Recognition Using Wearable Sensors", UbiComp 2012 Workshop.
http://sipi.usc.edu/had/
"""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add parent to path for shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.windowing import create_variable_windows


# Activity mapping (1-indexed activity numbers to standardized names)
ACTIVITIES = {
    1: "walking_forward",
    2: "walking_left",
    3: "walking_right",
    4: "walking_upstairs",
    5: "walking_downstairs",
    6: "running_forward",
    7: "jumping_up",
    8: "sitting",
    9: "standing",
    10: "sleeping",
    11: "elevator_up",
    12: "elevator_down",
}

# Paths
RAW_DIR = Path("data/raw/usc_had")
OUTPUT_DIR = Path("data/usc_had")

# Sampling rate (native, no resampling needed)
SAMPLE_RATE = 100.0

# Column names for the 6-channel sensor data
COLUMN_NAMES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def load_mat_trial(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Load a single trial .mat file.

    The .mat file contains 'sensor_readings' (Nx6 array).
    Columns: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]

    Returns:
        DataFrame with sensor data and timestamp_sec column, or None on error.
    """
    try:
        from scipy.io import loadmat

        mat = loadmat(str(filepath))

        # sensor_readings is the main data array (Nx6)
        if "sensor_readings" in mat:
            data = mat["sensor_readings"]
        else:
            # Try alternative key names
            for key in mat:
                if not key.startswith("_") and isinstance(mat[key], np.ndarray):
                    if mat[key].ndim == 2 and mat[key].shape[1] == 6:
                        data = mat[key]
                        break
            else:
                print(f"    Warning: No sensor_readings found in {filepath.name}")
                return None

        if data.shape[0] < 10:
            return None

        df = pd.DataFrame(data, columns=COLUMN_NAMES)
        df.insert(0, "timestamp_sec", np.arange(len(df)) / SAMPLE_RATE)

        return df

    except Exception as e:
        print(f"    Error loading {filepath}: {e}")
        return None


def convert_dataset():
    """Convert USC-HAD dataset to standardized format."""
    print("=" * 80)
    print("USC-HAD -> Standardized Format Converter")
    print("=" * 80)
    print("NOTE: This dataset is used for ZERO-SHOT TESTING (not training)")

    # Check input
    if not RAW_DIR.exists():
        print(f"ERROR: Raw data not found at {RAW_DIR}")
        print("Download from: http://sipi.usc.edu/had/")
        print("Extract to: data/raw/usc_had/")
        print("Expected structure: data/raw/usc_had/Subject{1..14}/a{1..12}t{1..5}.mat")
        return False

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sessions_dir = OUTPUT_DIR / "sessions"
    sessions_dir.mkdir(exist_ok=True)

    # Find subject directories
    subject_dirs = sorted([
        d for d in RAW_DIR.iterdir()
        if d.is_dir() and d.name.lower().startswith("subject")
    ])

    if not subject_dirs:
        print("ERROR: No Subject directories found")
        print(f"Expected: {RAW_DIR}/Subject1/, Subject2/, ...")
        return False

    print(f"\nFound {len(subject_dirs)} subjects")

    all_labels = {}
    session_count = 0
    skipped_count = 0

    for subject_dir in subject_dirs:
        # Extract subject number
        subject_num = int("".join(c for c in subject_dir.name if c.isdigit()))
        print(f"\n  Processing Subject {subject_num}...")

        # Find all trial files
        mat_files = sorted(subject_dir.glob("*.mat"))
        if not mat_files:
            # Try subdirectories
            mat_files = sorted(subject_dir.glob("**/*.mat"))

        for mat_file in mat_files:
            # Parse filename: a{activity}t{trial}.mat
            fname = mat_file.stem.lower()

            # Extract activity and trial numbers
            activity_num = None
            trial_num = None

            # Pattern: a{N}t{M}
            if "a" in fname and "t" in fname:
                try:
                    a_idx = fname.index("a")
                    t_idx = fname.index("t", a_idx + 1)
                    activity_num = int(fname[a_idx + 1:t_idx])
                    trial_num = int(fname[t_idx + 1:])
                except (ValueError, IndexError):
                    pass

            if activity_num is None or activity_num not in ACTIVITIES:
                skipped_count += 1
                continue

            activity_name = ACTIVITIES[activity_num]

            # Load data
            df = load_mat_trial(mat_file)
            if df is None or len(df) < 10:
                skipped_count += 1
                continue

            # Create session prefix
            session_prefix = f"usc_had_s{subject_num:02d}_{activity_name}_t{trial_num:02d}"

            # Split into variable-length windows
            windows = create_variable_windows(
                df=df,
                session_prefix=session_prefix,
                activity=activity_name,
                sample_rate=SAMPLE_RATE,
                seed=42 + subject_num * 100 + (trial_num or 0),
            )

            # Save each window
            for window_id, window_df, window_activity in windows:
                window_path = sessions_dir / window_id
                window_path.mkdir(exist_ok=True)
                window_df.to_parquet(window_path / "data.parquet", index=False)

                all_labels[window_id] = [window_activity]
                session_count += 1

        if session_count % 50 == 0 and session_count > 0:
            print(f"    {session_count} sessions so far...")

    if not all_labels:
        print("\nNo sessions created. Check the raw data format.")
        return False

    # Save labels.json
    with open(OUTPUT_DIR / "labels.json", "w") as f:
        json.dump(all_labels, f, indent=2)
    print(f"\nCreated labels.json ({len(all_labels)} sessions)")

    # Create manifest
    create_manifest()

    # Activity distribution
    activity_counts = {}
    for labels in all_labels.values():
        for label in labels:
            activity_counts[label] = activity_counts.get(label, 0) + 1

    print(f"\n{'=' * 80}")
    print("Conversion complete!")
    print(f"{'=' * 80}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"  - {session_count} sessions converted")
    print(f"  - {skipped_count} files skipped")
    print(f"  - {len(activity_counts)} unique activities")
    print(f"  - {SAMPLE_RATE} Hz sampling rate")

    print("\nActivity distribution:")
    for activity, count in sorted(activity_counts.items()):
        print(f"  {activity}: {count}")

    # Generate debug visualizations
    try:
        from shared.visualization_utils import generate_debug_visualizations
        generate_debug_visualizations(OUTPUT_DIR)
    except ImportError:
        pass

    return True


def create_manifest():
    """Create manifest.json."""
    manifest = {
        "dataset_name": "USC-HAD",
        "description": (
            "USC Human Activity Dataset. 14 subjects performing 12 daily activities "
            "with a waist-mounted MotionNode IMU (front right hip). "
            "Triaxial accelerometer and gyroscope at 100 Hz."
        ),
        "source": "http://sipi.usc.edu/had/",
        "num_subjects": 14,
        "channels": [
            {"name": "acc_x", "description": "Accelerometer X-axis", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "acc_y", "description": "Accelerometer Y-axis", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "acc_z", "description": "Accelerometer Z-axis", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "gyro_x", "description": "Gyroscope X-axis", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "gyro_y", "description": "Gyroscope Y-axis", "sampling_rate_hz": SAMPLE_RATE},
            {"name": "gyro_z", "description": "Gyroscope Z-axis", "sampling_rate_hz": SAMPLE_RATE},
        ],
    }

    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("Created manifest.json")


def main():
    success = convert_dataset()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
