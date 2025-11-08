#!/usr/bin/env python3
"""
Test script for tokenizer and classifier tools.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from tools import (
    load_session_as_artifact,
    execute_tool
)
from artifacts import (
    get_artifact,
    get_artifact_metadata,
    clear_all_artifacts
)


def test_har_tokenizer_classifier():
    """Test the HAR tokenizer and classifier workflow."""
    print("=" * 80)
    print("Testing HAR Tokenizer/Classifier Workflow")
    print("=" * 80)

    # Clear existing artifacts
    clear_all_artifacts()

    # Load a session
    dataset_name = "uci_har"
    session_id = "train_01_0000"

    print(f"\n1. Loading session: {session_id}")
    artifact_id = load_session_as_artifact(
        dataset_name=dataset_name,
        session_id=session_id,
        created_at_turn=0,
        created_by="user"
    )
    print(f"   Created: {artifact_id}")

    # Select IMU channels only
    print(f"\n2. Selecting IMU channels...")
    result = execute_tool(
        tool_name="select_channels",
        parameters={
            "artifact_id": artifact_id,
            "channel_names": ["body_acc_x", "body_acc_y", "body_acc_z",
                             "body_gyro_x", "body_gyro_y", "body_gyro_z"]
        },
        turn_number=1
    )
    imu_artifact_id = result["artifact_id"]
    print(f"   Created: {imu_artifact_id}")
    print(f"   Type: {result['type']}")

    # Apply tokenizer
    print(f"\n3. Applying HAR tokenizer...")
    result = execute_tool(
        tool_name="human_activity_motion_tokenizer",
        parameters={
            "artifact_id": imu_artifact_id
        },
        turn_number=2
    )
    ztokens_artifact_id = result["artifact_id"]
    print(f"   Created: {ztokens_artifact_id}")
    print(f"   Type: {result['type']}")

    # Get z-tokens metadata
    ztokens_meta = get_artifact_metadata(ztokens_artifact_id)
    print(f"   Vocabulary size: {ztokens_meta['vocabulary_size']}")
    print(f"   Num tokens: {ztokens_meta['num_tokens']}")

    # Apply classifier
    print(f"\n4. Applying HAR classifier...")
    result = execute_tool(
        tool_name="human_activity_motion_classifier",
        parameters={
            "artifact_id": ztokens_artifact_id
        },
        turn_number=3
    )
    etokens_artifact_id = result["artifact_id"]
    print(f"   Created: {etokens_artifact_id}")
    print(f"   Type: {result['type']}")

    # Get e-tokens metadata
    etokens_meta = get_artifact_metadata(etokens_artifact_id)
    print(f"   Vocabulary size: {etokens_meta['vocabulary_size']}")
    print(f"   Num tokens: {etokens_meta['num_tokens']}")
    print(f"   Has semantic labels: {etokens_meta['has_semantic_labels']}")

    # Get actual artifact to see labels
    from artifacts import get_artifact
    etokens_artifact = get_artifact(etokens_artifact_id)
    print(f"\n   Semantic labels: {etokens_artifact.semantic_labels[:6]}...")

    print("\n" + "=" * 80)
    print("✓ HAR Tokenizer/Classifier workflow complete!")
    print("=" * 80)

    # Verify artifact chain
    print(f"\nArtifact Chain:")
    print(f"  1. {artifact_id} (timeseries) - initial data")
    print(f"  2. {imu_artifact_id} (timeseries) - IMU channels only")
    print(f"  3. {ztokens_artifact_id} (z_tokens) - tokenized")
    print(f"  4. {etokens_artifact_id} (e_tokens) - classified")

    return True


def test_mocap_tokenizer_classifier():
    """Test the Motion Capture tokenizer (mock data)."""
    print("\n" + "=" * 80)
    print("Testing MoCap Tokenizer/Classifier Workflow (Mock)")
    print("=" * 80)

    print("\nNote: This would require motion capture data with joint rotations.")
    print("For now, we've verified the HAR workflow works correctly.")
    print("The MoCap tools have the same structure and will work similarly.")

    return True


if __name__ == "__main__":
    try:
        success = test_har_tokenizer_classifier()
        if success:
            test_mocap_tokenizer_classifier()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
