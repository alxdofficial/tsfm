"""
Tests for VTT-ConIoT dataset conversion.

Run: python datascripts/vtt_coniot/test_convert.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_DIR = Path("data/vtt_coniot")
EXPECTED_CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]  # Minimum expected
EXPECTED_SAMPLE_RATE = 50.0


def test_output_exists():
    """Test that output directory and required files exist."""
    print("Testing output structure...")

    assert OUTPUT_DIR.exists(), f"Output directory not found: {OUTPUT_DIR}"

    manifest_path = OUTPUT_DIR / "manifest.json"
    assert manifest_path.exists(), f"manifest.json not found"

    labels_path = OUTPUT_DIR / "labels.json"
    assert labels_path.exists(), f"labels.json not found"

    sessions_dir = OUTPUT_DIR / "sessions"
    assert sessions_dir.exists(), f"sessions/ directory not found"

    session_dirs = list(sessions_dir.iterdir())
    assert len(session_dirs) > 0, "No sessions found"

    print(f"  ✓ Found {len(session_dirs)} sessions")
    return True


def test_manifest_structure():
    """Test manifest.json has required fields."""
    print("Testing manifest structure...")

    with open(OUTPUT_DIR / "manifest.json") as f:
        manifest = json.load(f)

    assert "dataset_name" in manifest, "Missing dataset_name"
    assert manifest["dataset_name"] == "VTT-ConIoT", "Wrong dataset name"
    assert "description" in manifest, "Missing description"
    assert "channels" in manifest, "Missing channels"

    channels = manifest["channels"]
    assert len(channels) >= 6, f"Expected at least 6 channels, got {len(channels)}"

    channel_names = [c["name"] for c in channels]
    for expected in ["acc_x", "acc_y", "acc_z"]:  # Minimum requirement
        assert expected in channel_names, f"Missing channel: {expected}"

    print(f"  ✓ Manifest valid: {manifest['dataset_name']}")
    return True


def test_labels_mapping():
    """Test that all sessions have labels."""
    print("Testing labels mapping...")

    with open(OUTPUT_DIR / "labels.json") as f:
        labels = json.load(f)

    sessions_dir = OUTPUT_DIR / "sessions"
    session_ids = [d.name for d in sessions_dir.iterdir() if d.is_dir()]

    # All sessions should have labels
    for session_id in session_ids:
        assert session_id in labels, f"Session {session_id} missing from labels.json"
        assert len(labels[session_id]) > 0, f"Session {session_id} has empty labels"

    # Count unique activities
    activities = set()
    for session_labels in labels.values():
        activities.update(session_labels)

    print(f"  ✓ All {len(session_ids)} sessions have labels")
    print(f"  ✓ Found {len(activities)} unique activities")
    return True


def test_session_parquet_format():
    """Test parquet files have correct format."""
    print("Testing parquet format...")

    sessions_dir = OUTPUT_DIR / "sessions"
    session_dirs = list(sessions_dir.iterdir())

    # Test first 5 sessions
    for session_dir in session_dirs[:5]:
        parquet_path = session_dir / "data.parquet"
        assert parquet_path.exists(), f"data.parquet not found in {session_dir.name}"

        df = pd.read_parquet(parquet_path)

        # Check timestamp column
        assert "timestamp_sec" in df.columns, f"Missing timestamp_sec in {session_dir.name}"

        # Check at least accelerometer channels
        has_acc = any(col in df.columns for col in ["acc_x", "acc_y", "acc_z"])
        assert has_acc, f"No accelerometer data in {session_dir.name}"

        # Check data types
        assert df["timestamp_sec"].dtype in [np.float64, np.float32], "timestamp_sec should be float"

        # Check data is reasonable
        assert len(df) > 0, f"Empty dataframe in {session_dir.name}"
        assert df["timestamp_sec"].iloc[0] >= 0, "Timestamps should start >= 0"

    print(f"  ✓ Parquet format valid for {min(5, len(session_dirs))} sessions")
    return True


def test_sampling_rate():
    """Test that data matches expected sampling rate."""
    print("Testing sampling rate...")

    sessions_dir = OUTPUT_DIR / "sessions"
    session_dirs = list(sessions_dir.iterdir())

    rates = []
    for session_dir in session_dirs[:10]:
        parquet_path = session_dir / "data.parquet"
        df = pd.read_parquet(parquet_path)

        if len(df) > 1:
            duration = df["timestamp_sec"].iloc[-1] - df["timestamp_sec"].iloc[0]
            if duration > 0:
                actual_rate = (len(df) - 1) / duration
                rates.append(actual_rate)

    if rates:
        mean_rate = np.mean(rates)
        assert abs(mean_rate - EXPECTED_SAMPLE_RATE) < 5, f"Sampling rate {mean_rate:.1f}Hz differs from expected {EXPECTED_SAMPLE_RATE}Hz"
        print(f"  ✓ Sampling rate: {mean_rate:.1f} Hz (expected {EXPECTED_SAMPLE_RATE} Hz)")

    return True


def test_channel_statistics():
    """Test channel data has reasonable statistics."""
    print("Testing channel statistics...")

    sessions_dir = OUTPUT_DIR / "sessions"
    session_dirs = list(sessions_dir.iterdir())

    # Collect statistics from sample of sessions
    all_stats = {}

    for session_dir in session_dirs[:20]:
        parquet_path = session_dir / "data.parquet"
        df = pd.read_parquet(parquet_path)

        for col in df.columns:
            if col == "timestamp_sec":
                continue

            if col not in all_stats:
                all_stats[col] = []

            values = df[col].dropna().values
            if len(values) > 0:
                all_stats[col].extend(values[:100])

    for channel, values in all_stats.items():
        if values:
            arr = np.array(values)
            mean, std = np.mean(arr), np.std(arr)
            print(f"    {channel}: mean={mean:.3f}, std={std:.3f}")

    print("  ✓ Channel statistics collected")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("VTT-ConIoT Dataset Conversion Tests")
    print("=" * 80)

    if not OUTPUT_DIR.exists():
        print(f"\n✗ Output directory not found: {OUTPUT_DIR}")
        print("Run conversion first: python datascripts/vtt_coniot/convert.py")
        return False

    tests = [
        test_output_exists,
        test_manifest_structure,
        test_labels_mapping,
        test_session_parquet_format,
        test_sampling_rate,
        test_channel_statistics,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1

    print(f"\n{'=' * 80}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 80)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
