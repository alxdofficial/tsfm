"""
Convert ActionSense dataset to standardized format.

⚠️  MANUAL DOWNLOAD REQUIRED ⚠️

ActionSense dataset must be downloaded manually before running this script.

Download Options:
  1. Use the automated download script (requires authentication):
     python datascripts/download_actionsense.py

  2. Manual download from MIT CSAIL:
     https://action-sense.csail.mit.edu

Expected Input Directory: data/raw/actionsense/
  - manifest.csv (episode metadata)
  - CSV files organized by subject/split/modality

Output: data/actionsense/
  - manifest.json
  - labels.json
  - sessions/session_XXX/data.parquet

Note: This conversion processes ONLY EMG data at 200Hz target sampling rate.
  - EMG left: 8 channels from left forearm Myo armband
  - EMG right: 8 channels from right forearm Myo armband
  - Total: 16 EMG channels
Other modalities (joints, gaze) are excluded.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path


# Paths
RAW_DIR = Path("data/raw/actionsense")
MANIFEST_CSV = RAW_DIR / "manifest.csv"
OUTPUT_DIR = Path("data/actionsense")

# Target sampling rate for EMG data (Hz)
# This is the documented spec for Myo armband used in ActionSense
TARGET_SAMPLING_RATE = 200.0


def convert_episodes():
    """Convert all episodes to sessions."""
    print("\nLoading manifest...")

    if not MANIFEST_CSV.exists():
        print(f"ERROR: Manifest not found: {MANIFEST_CSV}")
        return {}, {}

    manifest_df = pd.read_csv(MANIFEST_CSV)
    print(f"  Found {len(manifest_df)} episodes")

    all_labels = {}
    session_count = 0

    for idx, row in manifest_df.iterrows():
        # Create session ID
        session_id = f"{row['subject']}_{row['split']}_act{row['activity_index']}"

        # Load EMG data only
        emg_dfs = []

        # Load EMG left
        emg_left_path = row.get('emg_left_csv')
        if pd.notna(emg_left_path):
            full_path = RAW_DIR / emg_left_path
            if full_path.exists():
                emg_left_df = pd.read_csv(full_path)
                emg_dfs.append(emg_left_df)

        # Load EMG right
        emg_right_path = row.get('emg_right_csv')
        if pd.notna(emg_right_path):
            full_path = RAW_DIR / emg_right_path
            if full_path.exists():
                emg_right_df = pd.read_csv(full_path)
                emg_dfs.append(emg_right_df)

        # Skip if no EMG data available
        if not emg_dfs:
            continue

        # Merge EMG left and right
        combined_emg = pd.concat(emg_dfs, axis=1)

        # Drop duplicate 'time_s' columns (both left and right EMG have this)
        # Keep only the sensor data columns
        cols_to_keep = [col for col in combined_emg.columns if col.startswith('emg_')]
        combined_emg = combined_emg[cols_to_keep]

        # Add timestamp column at 200Hz
        combined_emg.insert(0, 'timestamp_sec', np.arange(len(combined_emg)) / TARGET_SAMPLING_RATE)

        # Save to parquet
        session_dir = OUTPUT_DIR / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = session_dir / "data.parquet"
        combined_emg.to_parquet(parquet_path, index=False)

        # Store label
        activity_name = row['activity_name']
        all_labels[session_id] = [activity_name]

        session_count += 1

    print(f"\n✓ Created {session_count} sessions")
    return all_labels


def create_manifest():
    """Create minimal manifest.json."""
    manifest = {
        "dataset_name": "ActionSense",
        "description": "Kitchen activity dataset with bilateral EMG from Myo armbands. 9 subjects performing 23 kitchen activities. Target sampling rate: 200Hz.",
        "channels": [
            # EMG left (8 channels)
            *[{
                "name": f"emg_left_{i}",
                "description": f"Left forearm EMG channel {i} from Myo armband",
                "sampling_rate_hz": 200.0
            } for i in range(8)],

            # EMG right (8 channels)
            *[{
                "name": f"emg_right_{i}",
                "description": f"Right forearm EMG channel {i} from Myo armband",
                "sampling_rate_hz": 200.0
            } for i in range(8)]
        ]
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Created manifest: {manifest_path}")


def main():
    """Convert ActionSense to standardized format."""
    print("=" * 80)
    print("ActionSense → Standardized Format Converter")
    print("=" * 80)

    # Check input
    if not RAW_DIR.exists():
        print(f"ERROR: Raw data not found at {RAW_DIR}")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Convert episodes
    labels_dict = convert_episodes()

    # Save labels.json
    if labels_dict:
        labels_path = OUTPUT_DIR / "labels.json"
        with open(labels_path, 'w') as f:
            json.dump(labels_dict, f, indent=2)

        print(f"\n✓ Created labels: {labels_path}")
        print(f"  Total sessions: {len(labels_dict)}")

        # Create manifest
        create_manifest()

        print(f"\n{'=' * 80}")
        print("Conversion complete!")
        print(f"{'=' * 80}")
        print(f"Output: {OUTPUT_DIR}")
        print(f"  - {len(labels_dict)} sessions")
        print(f"  - 16 channels (bilateral EMG)")
        print(f"  - 200Hz uniform sampling rate")
        print(f"\nNOTE: Only EMG data is processed. Joints and gaze modalities excluded.")

        # Generate debug visualizations
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))  # Add datascripts/ to path
            from shared.visualization_utils import generate_debug_visualizations

            generate_debug_visualizations(OUTPUT_DIR)
        except ImportError as e:
            print(f"\n⚠ Could not generate visualizations: {e}")
            print("Install matplotlib: pip install matplotlib")


if __name__ == "__main__":
    main()
