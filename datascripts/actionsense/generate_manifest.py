"""
Generate comprehensive JSON manifest from ActionSense CSV manifest.

This script reads the existing manifest.csv and creates a rich JSON manifest
that includes dataset metadata, modality specifications, activity descriptions,
and episode information for use by the LLM agent.
"""

import os
import json
import pandas as pd
from collections import Counter
from typing import Dict, List

# Constants
BASE_DIR = "data/raw/actionsense"
MANIFEST_CSV = os.path.join(BASE_DIR, "manifest.csv")
OUTPUT_JSON = "data/actionsense/manifest.json"


def load_manifest_csv(csv_path: str) -> pd.DataFrame:
    """Load the CSV manifest."""
    print(f"Loading manifest from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} episodes")
    return df


def extract_unique_activities(df: pd.DataFrame) -> List[Dict]:
    """Extract unique activities with descriptions."""
    activities = df["activity_name"].unique().tolist()
    print(f"Found {len(activities)} unique activities")

    # Create activity list with descriptions
    activity_list = []
    for activity in sorted(activities):
        activity_list.append({
            "name": activity,
            "description": activity  # Using activity name as description
        })

    return activity_list


def extract_subjects(df: pd.DataFrame) -> List[str]:
    """Extract unique subjects."""
    subjects = sorted(df["subject"].unique().tolist())
    print(f"Found {len(subjects)} subjects: {subjects}")
    return subjects


def calculate_total_duration(df: pd.DataFrame) -> float:
    """Calculate total duration of all episodes in hours."""
    df["duration_sec"] = df["t1_abs"] - df["t0_abs"]
    total_seconds = df["duration_sec"].sum()
    total_hours = total_seconds / 3600.0
    print(f"Total duration: {total_hours:.2f} hours ({total_seconds:.0f} seconds)")
    return total_hours


def get_activity_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """Get episode count per activity."""
    distribution = Counter(df["activity_name"])
    print(f"Activity distribution: {len(distribution)} activities")
    return dict(distribution)


def create_episode_list(df: pd.DataFrame, base_dir: str) -> List[Dict]:
    """Create list of episode metadata."""
    episodes = []

    for idx, row in df.iterrows():
        # Calculate duration
        duration_sec = row["t1_abs"] - row["t0_abs"]

        # Create episode ID
        episode_id = f"{row['subject']}_{row['split']}_act{row['activity_index']}"

        # Build file paths dict (only include non-null modalities)
        files = {}
        if pd.notna(row["joints_csv"]):
            files["joints"] = os.path.join(base_dir, row["joints_csv"])
        if pd.notna(row["emg_left_csv"]):
            files["emg_left"] = os.path.join(base_dir, row["emg_left_csv"])
        if pd.notna(row["emg_right_csv"]):
            files["emg_right"] = os.path.join(base_dir, row["emg_right_csv"])
        if pd.notna(row["gaze_csv"]):
            files["gaze"] = os.path.join(base_dir, row["gaze_csv"])

        episode = {
            "id": episode_id,
            "subject": row["subject"],
            "split": row["split"],
            "activity": row["activity_name"],
            "activity_index": int(row["activity_index"]),
            "duration_sec": round(duration_sec, 2),
            "timestamp_start": row["t0_abs"],
            "timestamp_end": row["t1_abs"],
            "files": files
        }

        episodes.append(episode)

    print(f"Created {len(episodes)} episode entries")
    return episodes


def generate_channel_names(num_joints: int = 22) -> List[str]:
    """Generate joint channel names (assuming 22 joints × 3 axes = 66 channels)."""
    axes = ["x", "y", "z"]
    channels = []
    for joint_id in range(num_joints):
        for axis in axes:
            channels.append(f"joint_{joint_id}_{axis}")
    return channels


def create_manifest(csv_path: str, base_dir: str) -> Dict:
    """Create comprehensive manifest JSON."""
    # Load data
    df = load_manifest_csv(csv_path)

    # Extract information
    activities = extract_unique_activities(df)
    subjects = extract_subjects(df)
    total_duration_hours = calculate_total_duration(df)
    activity_distribution = get_activity_distribution(df)
    episodes = create_episode_list(df, base_dir)

    # Build manifest
    manifest = {
        "dataset_name": "ActionSense",
        "description": "Multi-modal human activity recognition dataset captured from kitchen activities. Includes full-body joint angles, bilateral EMG, and gaze tracking.",
        "paper_url": "https://action-sense.csail.mit.edu",
        "modalities": {
            "joints": {
                "description": "Full-body joint angles captured via motion capture system",
                "channels": 66,
                "channel_names": generate_channel_names(22),
                "sampling_rate_hz": 60.0,
                "units": "degrees",
                "note": "22 joints × 3 axes (X, Y, Z rotation)"
            },
            "emg_left": {
                "description": "Electromyography signals from left forearm",
                "channels": 8,
                "channel_names": [f"emg_left_{i}" for i in range(8)],
                "sampling_rate_hz": 200.0,
                "units": "microvolts",
                "note": "8-channel Myo armband on left forearm"
            },
            "emg_right": {
                "description": "Electromyography signals from right forearm",
                "channels": 8,
                "channel_names": [f"emg_right_{i}" for i in range(8)],
                "sampling_rate_hz": 200.0,
                "units": "microvolts",
                "note": "8-channel Myo armband on right forearm"
            },
            "gaze": {
                "description": "Eye gaze tracking (2D screen coordinates)",
                "channels": 2,
                "channel_names": ["gaze_x", "gaze_y"],
                "sampling_rate_hz": 120.0,
                "units": "pixels",
                "note": "Normalized screen coordinates"
            }
        },
        "activities": activities,
        "subjects": subjects,
        "episodes": episodes,
        "statistics": {
            "total_episodes": len(episodes),
            "total_duration_hours": round(total_duration_hours, 2),
            "activities_distribution": activity_distribution,
            "subjects_count": len(subjects),
            "modalities_count": 4
        }
    }

    return manifest


def main():
    """Generate manifest JSON from CSV."""
    print("=" * 80)
    print("ActionSense Manifest Generator")
    print("=" * 80)

    # Check if input exists
    if not os.path.exists(MANIFEST_CSV):
        print(f"ERROR: Manifest CSV not found at: {MANIFEST_CSV}")
        return

    # Generate manifest
    manifest = create_manifest(MANIFEST_CSV, BASE_DIR)

    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    # Save to JSON
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Manifest saved to: {OUTPUT_JSON}")
    print(f"{'=' * 80}")
    print(f"\nSummary:")
    print(f"  - Episodes: {manifest['statistics']['total_episodes']}")
    print(f"  - Activities: {len(manifest['activities'])}")
    print(f"  - Subjects: {manifest['statistics']['subjects_count']}")
    print(f"  - Modalities: {manifest['statistics']['modalities_count']}")
    print(f"  - Total duration: {manifest['statistics']['total_duration_hours']:.2f} hours")
    print(f"\nManifest is ready for use with tool-using LLM!")


if __name__ == "__main__":
    main()
