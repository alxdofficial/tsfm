#!/usr/bin/env python3
"""
Test script for artifact system end-to-end flow.

Tests:
1. Loading a session as an artifact
2. Using select_channels to create a filtered artifact
3. Using filter_by_time to create a time-filtered artifact
4. Verifying artifact chaining and metadata
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from tools.tool_executor import (
    load_session_as_artifact,
    execute_tool,
    show_session_stats,
    show_channel_stats
)
from artifacts import (
    get_artifact,
    get_artifact_metadata,
    list_artifacts,
    clear_all_artifacts
)


def test_artifact_workflow():
    """Test the complete artifact workflow."""
    print("=" * 80)
    print("Testing Artifact System")
    print("=" * 80)

    # Clear any existing artifacts
    clear_all_artifacts()
    print("\n✓ Cleared existing artifacts")

    # Step 1: Load a session as an artifact
    print("\n" + "=" * 80)
    print("Step 1: Load session as artifact")
    print("=" * 80)

    dataset_name = "uci_har"

    # First, let's see what sessions are available
    stats = show_session_stats(dataset_name)
    print(f"\nDataset stats:")
    print(f"  Total sessions: {stats['total_sessions']}")
    print(f"  Labels: {list(stats['label_distribution'].keys())}")

    # Get channel info
    channel_info = show_channel_stats(dataset_name)
    available_channels = [ch["name"] for ch in channel_info["channels"]]
    print(f"\nAvailable channels ({len(available_channels)}):")
    for ch in available_channels[:6]:  # Show first 6
        print(f"  - {ch}")
    if len(available_channels) > 6:
        print(f"  ... and {len(available_channels) - 6} more")

    # Load first session
    session_id = "train_01_0000"  # Known session from UCI HAR
    print(f"\nLoading session: {session_id}")

    try:
        artifact_id = load_session_as_artifact(
            dataset_name=dataset_name,
            session_id=session_id,
            created_at_turn=0,
            created_by="user"
        )
        print(f"✓ Created artifact: {artifact_id}")

        # Get artifact metadata
        metadata = get_artifact_metadata(artifact_id)
        print(f"\nArtifact metadata:")
        print(f"  Type: {metadata['type']}")
        print(f"  Samples: {metadata['num_samples']}")
        print(f"  Channels: {metadata['num_channels']}")
        print(f"  Duration: {metadata['duration_sec']:.2f} sec")
        print(f"  Sampling rate: {metadata['sampling_rate_hz']:.2f} Hz")
        print(f"  Shape: {metadata['shape']}")

    except FileNotFoundError as e:
        print(f"⚠ Session not found: {e}")
        print("\nPlease ensure UCI HAR dataset is available in data/uci_har/")
        return False

    # Step 2: Select specific channels
    print("\n" + "=" * 80)
    print("Step 2: Select subset of channels")
    print("=" * 80)

    selected_channels = ["body_acc_x", "body_acc_y", "body_acc_z"]
    print(f"\nSelecting channels: {selected_channels}")

    result = execute_tool(
        tool_name="select_channels",
        parameters={
            "artifact_id": artifact_id,
            "channel_names": selected_channels
        },
        turn_number=1
    )

    print(f"✓ Tool result:")
    print(f"  Type: {result['type']}")
    print(f"  Artifact ID: {result['artifact_id']}")

    filtered_artifact_id = result["artifact_id"]

    # Verify new artifact
    filtered_metadata = get_artifact_metadata(filtered_artifact_id)
    print(f"\nFiltered artifact metadata:")
    print(f"  Samples: {filtered_metadata['num_samples']}")
    print(f"  Channels: {filtered_metadata['num_channels']} (was {metadata['num_channels']})")
    print(f"  Channel names: {filtered_metadata['channels']}")
    print(f"  Parent: {filtered_metadata['parent_artifact_id']}")
    print(f"  Created by: {filtered_metadata['created_by']}")

    # Verify channels were filtered
    assert filtered_metadata['num_channels'] == len(selected_channels), \
        f"Expected {len(selected_channels)} channels, got {filtered_metadata['num_channels']}"
    assert filtered_metadata['channels'] == selected_channels, \
        f"Channel mismatch: {filtered_metadata['channels']} != {selected_channels}"
    assert filtered_metadata['parent_artifact_id'] == artifact_id, \
        f"Parent artifact mismatch"

    print("✓ Channel filtering verified")

    # Step 3: Filter by time
    print("\n" + "=" * 80)
    print("Step 3: Filter by time range")
    print("=" * 80)

    start_time = 0.0
    end_time = 2.0  # First 2 seconds
    print(f"\nFiltering time range: {start_time} - {end_time} seconds")

    result = execute_tool(
        tool_name="filter_by_time",
        parameters={
            "artifact_id": filtered_artifact_id,
            "start_time_sec": start_time,
            "end_time_sec": end_time
        },
        turn_number=2
    )

    print(f"✓ Tool result:")
    print(f"  Type: {result['type']}")
    print(f"  Artifact ID: {result['artifact_id']}")

    time_filtered_id = result["artifact_id"]

    # Verify time-filtered artifact
    time_filtered_metadata = get_artifact_metadata(time_filtered_id)
    print(f"\nTime-filtered artifact metadata:")
    print(f"  Samples: {time_filtered_metadata['num_samples']} (was {filtered_metadata['num_samples']})")
    print(f"  Duration: {time_filtered_metadata['duration_sec']:.2f} sec (was {filtered_metadata['duration_sec']:.2f} sec)")
    print(f"  Channels: {time_filtered_metadata['num_channels']}")
    print(f"  Parent: {time_filtered_metadata['parent_artifact_id']}")
    print(f"  Created by: {time_filtered_metadata['created_by']}")

    # Verify time filtering worked
    assert time_filtered_metadata['num_samples'] < filtered_metadata['num_samples'], \
        "Time filtering should reduce sample count"
    assert abs(time_filtered_metadata['duration_sec'] - (end_time - start_time)) < 0.1, \
        f"Duration mismatch: expected ~{end_time - start_time}, got {time_filtered_metadata['duration_sec']}"
    assert time_filtered_metadata['parent_artifact_id'] == filtered_artifact_id, \
        "Parent artifact mismatch"

    print("✓ Time filtering verified")

    # Step 4: Verify artifact lineage
    print("\n" + "=" * 80)
    print("Step 4: Verify artifact lineage")
    print("=" * 80)

    all_artifacts = list_artifacts()
    print(f"\nTotal artifacts in registry: {len(all_artifacts)}")

    print("\nArtifact chain:")
    print(f"  1. {artifact_id} (timeseries) - created by: user")
    print(f"     └─> 2. {filtered_artifact_id} (timeseries) - created by: select_channels")
    print(f"         └─> 3. {time_filtered_id} (timeseries) - created by: filter_by_time")

    # Verify we have exactly 3 artifacts
    assert len(all_artifacts) == 3, f"Expected 3 artifacts, found {len(all_artifacts)}"

    print("\n✓ Artifact lineage verified")

    # Step 5: Access actual artifact data
    print("\n" + "=" * 80)
    print("Step 5: Access artifact data")
    print("=" * 80)

    final_artifact = get_artifact(time_filtered_id)
    print(f"\nFinal artifact data shape: {final_artifact.data.shape}")
    print(f"Columns: {list(final_artifact.data.columns)}")
    print(f"\nFirst 3 rows:")
    print(final_artifact.data.head(3))

    print("\n✓ Artifact data access verified")

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print("\n✓ All tests passed!")
    print(f"\nCreated artifact chain:")
    print(f"  Original: {metadata['num_samples']} samples, {metadata['num_channels']} channels")
    print(f"  After channel filter: {filtered_metadata['num_samples']} samples, {filtered_metadata['num_channels']} channels")
    print(f"  After time filter: {time_filtered_metadata['num_samples']} samples, {time_filtered_metadata['duration_sec']:.2f}s")

    return True


if __name__ == "__main__":
    try:
        success = test_artifact_workflow()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
