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

Note: This dataset has multi-modal data with different sampling rates:
  - Joints: 60 Hz (66 channels)
  - EMG: 200 Hz (16 channels)
  - Gaze: 120 Hz (2 channels)
Currently, only one modality per session is loaded. Multi-modal alignment
with interpolation is planned for a future version.
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


def load_episode_data(episode_row):
    """Load all modalities for one episode and merge into single DataFrame."""
    dfs = {}
    sampling_rates = {}

    # Load each modality that exists for this episode
    modalities = {
        'joints': ('joints_csv', 60.0),
        'emg_left': ('emg_left_csv', 200.0),
        'emg_right': ('emg_right_csv', 200.0),
        'gaze': ('gaze_csv', 120.0)
    }

    for modality_name, (csv_col, sampling_rate) in modalities.items():
        csv_path = episode_row.get(csv_col)

        if pd.notna(csv_path) and os.path.exists(RAW_DIR / csv_path):
            df = pd.read_csv(RAW_DIR / csv_path)

            # Add timestamp column based on sampling rate
            df.insert(0, 'timestamp_sec', np.arange(len(df)) / sampling_rate)

            dfs[modality_name] = df
            sampling_rates[modality_name] = sampling_rate

    if not dfs:
        return None

    # For simplicity, use the first modality as base
    # In a more advanced version, we could interpolate to align all modalities
    base_modality = list(dfs.keys())[0]
    merged_df = dfs[base_modality]

    # Add other modalities (for now, just keep them separate by session)
    # In practice, multi-modal alignment would require interpolation
    for modality_name, df in dfs.items():
        if modality_name != base_modality:
            # Prefix column names
            df_renamed = df.rename(columns={
                col: f"{col}" for col in df.columns if col != 'timestamp_sec'
            })
            # For now, just keep the first modality
            # TODO: Implement proper alignment
            pass

    return merged_df


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

        # Load and merge modality data
        # For ActionSense, we'll process each modality separately
        # since they have different sampling rates

        modalities_to_process = [
            ('joints', 'joints_csv', 60.0),
            ('emg_left', 'emg_left_csv', 200.0),
            ('emg_right', 'emg_right_csv', 200.0),
            ('gaze', 'gaze_csv', 120.0)
        ]

        # Load first available modality as primary
        primary_data = None
        for modality_name, csv_col, sampling_rate in modalities_to_process:
            csv_path = row.get(csv_col)

            if pd.notna(csv_path):
                full_path = RAW_DIR / csv_path
                if full_path.exists():
                    df = pd.read_csv(full_path)

                    # Add timestamp column
                    df.insert(0, 'timestamp_sec', np.arange(len(df)) / sampling_rate)

                    if primary_data is None:
                        primary_data = df
                    # TODO: Merge other modalities with interpolation

                    break  # Use first available modality for now

        if primary_data is None:
            continue

        # Save to parquet
        session_dir = OUTPUT_DIR / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = session_dir / "data.parquet"
        primary_data.to_parquet(parquet_path, index=False)

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
        "description": "Multi-modal kitchen activity dataset with motion capture, bilateral EMG, and gaze tracking. 9 subjects performing 23 kitchen activities.",
        "channels": [
            # Joints (66 channels, flat numbering)
            *[{
                "name": f"joints_{i}",
                "description": f"Joint channel {i} from motion capture system",
                "sampling_rate_hz": 60.0
            } for i in range(66)],

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
            } for i in range(8)],

            # Gaze (2 channels)
            {
                "name": "gaze_x",
                "description": "Eye gaze X-coordinate (screen pixels)",
                "sampling_rate_hz": 120.0
            },
            {
                "name": "gaze_y",
                "description": "Eye gaze Y-coordinate (screen pixels)",
                "sampling_rate_hz": 120.0
            }
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
        print(f"  - 84 channels (joints + EMG + gaze)")
        print(f"  - Multi-rate: 60Hz (joints), 200Hz (EMG), 120Hz (gaze)")
        print(f"\nNOTE: Currently using first available modality per session.")
        print(f"      Multi-modal alignment coming in future version.")

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
