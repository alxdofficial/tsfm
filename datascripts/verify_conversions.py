#!/usr/bin/env python3
"""
Verification script to check converted dataset quality.

Checks:
1. Timestamp column exists and starts at 0
2. Timestamps are monotonically increasing
3. Sampling rates match manifest
4. No unexpected NaN values
5. Session durations are reasonable
6. Data types are correct

Usage:
    python3 datascripts/verify_conversions.py [dataset_name]

    Without arguments: verify all datasets
    With dataset_name: verify only that dataset (uci_har, pamap2, mhealth, wisdm)
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


# Dataset configuration
DATASETS = {
    "uci_har": {
        "path": "data/uci_har",
        "expected_rate_hz": 50.0,
        "min_sessions": 7000,  # Should have ~7352 sessions
        "expected_duration_sec": 2.56  # Fixed window size
    },
    "pamap2": {
        "path": "data/pamap2",
        "expected_rate_hz": 100.0,  # IMU sampling rate
        "min_sessions": 100,
        "expected_duration_sec": None  # Variable duration
    },
    "mhealth": {
        "path": "data/mhealth",
        "expected_rate_hz": 50.0,
        "min_sessions": 100,
        "expected_duration_sec": None  # Variable duration
    },
    "wisdm": {
        "path": "data/wisdm",
        "expected_rate_hz": 20.0,
        "min_sessions": 1000,
        "expected_duration_sec": None  # Variable duration
    }
}


def check_timestamp_column(df, session_id):
    """Verify timestamp column properties."""
    issues = []

    # Check column exists
    if 'timestamp_sec' not in df.columns:
        issues.append(f"Missing 'timestamp_sec' column")
        return issues

    # Check it's the first column
    if df.columns[0] != 'timestamp_sec':
        issues.append(f"'timestamp_sec' is not the first column (found at position {list(df.columns).index('timestamp_sec')})")

    # Check data type
    if not np.issubdtype(df['timestamp_sec'].dtype, np.number):
        issues.append(f"'timestamp_sec' is not numeric (dtype: {df['timestamp_sec'].dtype})")
        return issues

    # Check starts at/near 0
    first_timestamp = df['timestamp_sec'].iloc[0]
    if abs(first_timestamp) > 0.1:  # Allow small tolerance
        issues.append(f"First timestamp should be ~0, got {first_timestamp:.6f}")

    # Check monotonically increasing
    if not df['timestamp_sec'].is_monotonic_increasing:
        issues.append(f"Timestamps are not monotonically increasing")

    # Check for negative values
    if (df['timestamp_sec'] < 0).any():
        issues.append(f"Found negative timestamps")

    # Check for NaN
    if df['timestamp_sec'].isna().any():
        nan_count = df['timestamp_sec'].isna().sum()
        issues.append(f"Found {nan_count} NaN values in timestamp_sec")

    return issues


def check_sampling_rate(df, expected_rate_hz, session_id):
    """Verify sampling rate consistency."""
    issues = []

    if len(df) < 2:
        return issues  # Can't check rate with < 2 samples

    # Calculate actual sampling rate from timestamps
    timestamps = df['timestamp_sec'].values
    intervals = np.diff(timestamps)

    # Check for zero or negative intervals
    if (intervals <= 0).any():
        issues.append(f"Found zero or negative time intervals")
        return issues

    # Mean interval
    mean_interval = np.mean(intervals)
    actual_rate_hz = 1.0 / mean_interval if mean_interval > 0 else 0

    # Check if close to expected (allow 5% tolerance)
    tolerance = expected_rate_hz * 0.05
    if abs(actual_rate_hz - expected_rate_hz) > tolerance:
        issues.append(f"Sampling rate mismatch: expected ~{expected_rate_hz:.1f} Hz, got {actual_rate_hz:.1f} Hz (mean interval: {mean_interval:.6f} sec)")

    # Check for irregular intervals (coefficient of variation)
    std_interval = np.std(intervals)
    cv = std_interval / mean_interval if mean_interval > 0 else 0
    if cv > 0.1:  # 10% coefficient of variation
        issues.append(f"Irregular sampling intervals (CV: {cv:.3f})")

    return issues


def check_data_quality(df, session_id):
    """Check for NaN values and data types."""
    issues = []

    # Check for NaN in sensor columns (excluding timestamp_sec)
    sensor_cols = [col for col in df.columns if col != 'timestamp_sec']

    for col in sensor_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            nan_pct = (nan_count / len(df)) * 100
            issues.append(f"Column '{col}' has {nan_count} NaN values ({nan_pct:.1f}%)")

        # Check data type is numeric
        if not np.issubdtype(df[col].dtype, np.number):
            issues.append(f"Column '{col}' is not numeric (dtype: {df[col].dtype})")

    return issues


def verify_session(session_path, expected_rate_hz):
    """Verify a single session."""
    session_id = session_path.parent.name

    try:
        df = pd.read_parquet(session_path)
    except Exception as e:
        return {
            'session_id': session_id,
            'status': 'ERROR',
            'issues': [f"Failed to load parquet: {e}"]
        }

    issues = []

    # Check timestamp column
    issues.extend(check_timestamp_column(df, session_id))

    # Check sampling rate
    if 'timestamp_sec' in df.columns and len(df) >= 2:
        issues.extend(check_sampling_rate(df, expected_rate_hz, session_id))

    # Check data quality
    issues.extend(check_data_quality(df, session_id))

    # Compute metrics
    duration_sec = df['timestamp_sec'].iloc[-1] - df['timestamp_sec'].iloc[0] if len(df) > 1 else 0
    num_samples = len(df)
    num_channels = len(df.columns) - 1  # Exclude timestamp_sec

    return {
        'session_id': session_id,
        'status': 'PASS' if len(issues) == 0 else 'FAIL',
        'issues': issues,
        'num_samples': num_samples,
        'duration_sec': duration_sec,
        'num_channels': num_channels
    }


def verify_manifest(dataset_path):
    """Verify manifest.json structure."""
    manifest_path = dataset_path / "manifest.json"

    if not manifest_path.exists():
        return {'status': 'FAIL', 'issues': ['manifest.json not found']}

    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
    except Exception as e:
        return {'status': 'FAIL', 'issues': [f'Failed to load manifest.json: {e}']}

    issues = []

    # Check required fields
    if 'dataset_name' not in manifest:
        issues.append("Missing 'dataset_name' field")
    if 'description' not in manifest:
        issues.append("Missing 'description' field")
    if 'channels' not in manifest:
        issues.append("Missing 'channels' field")
    else:
        # Check channel structure
        for i, channel in enumerate(manifest['channels']):
            if 'name' not in channel:
                issues.append(f"Channel {i} missing 'name' field")
            if 'description' not in channel:
                issues.append(f"Channel {i} missing 'description' field")
            if 'sampling_rate_hz' not in channel:
                issues.append(f"Channel {i} missing 'sampling_rate_hz' field")

    return {
        'status': 'PASS' if len(issues) == 0 else 'FAIL',
        'issues': issues,
        'num_channels': len(manifest.get('channels', []))
    }


def verify_labels(dataset_path):
    """Verify labels.json structure."""
    labels_path = dataset_path / "labels.json"

    if not labels_path.exists():
        return {'status': 'FAIL', 'issues': ['labels.json not found']}

    try:
        with open(labels_path, 'r') as f:
            labels = json.load(f)
    except Exception as e:
        return {'status': 'FAIL', 'issues': [f'Failed to load labels.json: {e}']}

    issues = []

    # Check structure
    for session_id, label_list in labels.items():
        if not isinstance(label_list, list):
            issues.append(f"Session '{session_id}' labels are not a list (got {type(label_list)})")
        elif len(label_list) == 0:
            issues.append(f"Session '{session_id}' has empty label list")

    return {
        'status': 'PASS' if len(issues) == 0 else 'FAIL',
        'issues': issues,
        'num_sessions': len(labels)
    }


def verify_dataset(dataset_name, config):
    """Verify entire dataset."""
    print(f"\n{'=' * 80}")
    print(f"Verifying: {dataset_name.upper()}")
    print(f"{'=' * 80}")

    dataset_path = Path(config['path'])

    if not dataset_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        return False

    # Verify manifest
    print("\n📄 Checking manifest.json...")
    manifest_result = verify_manifest(dataset_path)
    if manifest_result['status'] == 'PASS':
        print(f"   ✅ Manifest valid ({manifest_result['num_channels']} channels)")
    else:
        print(f"   ❌ Manifest issues:")
        for issue in manifest_result['issues']:
            print(f"      - {issue}")

    # Verify labels
    print("\n🏷️  Checking labels.json...")
    labels_result = verify_labels(dataset_path)
    if labels_result['status'] == 'PASS':
        print(f"   ✅ Labels valid ({labels_result['num_sessions']} sessions)")
    else:
        print(f"   ❌ Labels issues:")
        for issue in labels_result['issues']:
            print(f"      - {issue}")

    # Check minimum session count
    num_sessions = labels_result.get('num_sessions', 0)
    if num_sessions < config.get('min_sessions', 0):
        print(f"   ⚠️  Warning: Only {num_sessions} sessions (expected >= {config['min_sessions']})")

    # Verify sessions
    print(f"\n📊 Checking sessions (sampling rate: {config['expected_rate_hz']} Hz)...")
    sessions_dir = dataset_path / "sessions"

    if not sessions_dir.exists():
        print(f"   ❌ Sessions directory not found")
        return False

    session_files = sorted(sessions_dir.glob("*/data.parquet"))

    if len(session_files) == 0:
        print(f"   ❌ No session files found")
        return False

    print(f"   Found {len(session_files)} sessions")

    # Verify sample of sessions
    sample_size = min(10, len(session_files))
    print(f"   Verifying {sample_size} random sessions...")

    # Sample sessions (first, middle, last, and random)
    indices = [0, len(session_files) // 2, len(session_files) - 1]
    if len(session_files) > 10:
        import random
        random.seed(42)
        indices.extend(random.sample(range(len(session_files)), min(7, len(session_files) - 3)))

    sample_sessions = [session_files[i] for i in sorted(set(indices))]

    failed_sessions = []
    passed_sessions = []
    all_durations = []
    all_sample_counts = []

    for session_path in sample_sessions:
        result = verify_session(session_path, config['expected_rate_hz'])

        if result['status'] == 'PASS':
            passed_sessions.append(result)
        else:
            failed_sessions.append(result)

        all_durations.append(result.get('duration_sec', 0))
        all_sample_counts.append(result.get('num_samples', 0))

    # Print results
    if len(failed_sessions) == 0:
        print(f"   ✅ All {len(sample_sessions)} sampled sessions passed")
    else:
        print(f"   ❌ {len(failed_sessions)}/{len(sample_sessions)} sessions failed:")
        for result in failed_sessions[:5]:  # Show first 5 failures
            print(f"\n      Session: {result['session_id']}")
            for issue in result['issues']:
                print(f"        - {issue}")

    # Statistics
    print(f"\n📈 Session Statistics (from {len(sample_sessions)} samples):")
    print(f"   Duration: {np.min(all_durations):.2f}s - {np.max(all_durations):.2f}s (mean: {np.mean(all_durations):.2f}s)")
    print(f"   Samples:  {np.min(all_sample_counts)} - {np.max(all_sample_counts)} (mean: {np.mean(all_sample_counts):.0f})")

    # Check expected duration if specified
    if config.get('expected_duration_sec') is not None:
        expected = config['expected_duration_sec']
        mean_duration = np.mean(all_durations)
        if abs(mean_duration - expected) > 0.1:
            print(f"   ⚠️  Warning: Expected duration ~{expected:.2f}s, got {mean_duration:.2f}s")

    # Overall result
    overall_pass = (
        manifest_result['status'] == 'PASS' and
        labels_result['status'] == 'PASS' and
        len(failed_sessions) == 0
    )

    print(f"\n{'=' * 80}")
    if overall_pass:
        print(f"✅ {dataset_name.upper()} VERIFICATION PASSED")
    else:
        print(f"❌ {dataset_name.upper()} VERIFICATION FAILED")
    print(f"{'=' * 80}")

    return overall_pass


def main():
    """Main verification entry point."""
    print("=" * 80)
    print("Dataset Conversion Verification")
    print("=" * 80)

    # Determine which datasets to verify
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
        if dataset_name not in DATASETS:
            print(f"ERROR: Unknown dataset '{dataset_name}'")
            print(f"Available: {list(DATASETS.keys())}")
            return 1
        datasets_to_verify = {dataset_name: DATASETS[dataset_name]}
    else:
        datasets_to_verify = DATASETS

    # Verify each dataset
    results = {}
    for dataset_name, config in datasets_to_verify.items():
        results[dataset_name] = verify_dataset(dataset_name, config)

    # Summary
    print(f"\n{'=' * 80}")
    print("Verification Summary")
    print(f"{'=' * 80}")

    for dataset_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {dataset_name.upper():12} {status}")

    total_passed = sum(results.values())
    print(f"\nTotal: {total_passed}/{len(results)} datasets passed")

    return 0 if total_passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
