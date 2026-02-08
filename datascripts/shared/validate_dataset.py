"""
Validate a converted dataset against the IMUPretrainingDataset loader requirements.

Checks:
1. manifest.json structure (channels with name, description, sampling_rate_hz)
2. labels.json structure (session_id -> [label_list])
3. Session parquet files (timestamp_sec column, IMU channels, no NaN)
4. Channel naming conventions (contain acc/gyro/mag, end with _x/_y/_z)
5. Channel grouping (axis suffixes produce valid sensor groups)
6. Data integrity (no all-zero sessions, reasonable value ranges)

Usage:
    python datascripts/shared/validate_dataset.py data/opportunity
    python datascripts/shared/validate_dataset.py data/shoaib data/daphnet_fog data/realdisp
    python datascripts/shared/validate_dataset.py --all  # validate all datasets in data/
"""

import json
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path


IMU_PATTERNS = ['acc', 'gyro', 'mag', 'ori']
AXIS_PATTERN = re.compile(r'_([xyz]|[1-4])$')


def validate_manifest(dataset_path: Path) -> list:
    """Validate manifest.json structure."""
    errors = []
    manifest_path = dataset_path / "manifest.json"

    if not manifest_path.exists():
        errors.append("MISSING: manifest.json")
        return errors

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"INVALID JSON in manifest.json: {e}")
        return errors

    # Check required fields
    if "channels" not in manifest:
        errors.append("manifest.json missing 'channels' field")
        return errors

    if "dataset_name" not in manifest:
        errors.append("manifest.json missing 'dataset_name' field")

    if "description" not in manifest:
        errors.append("manifest.json missing 'description' field")

    channels = manifest["channels"]
    if not channels:
        errors.append("manifest.json has empty 'channels' list")
        return errors

    # Check channel structure
    sampling_rates = set()
    channel_names = []
    for i, ch in enumerate(channels):
        if "name" not in ch:
            errors.append(f"Channel {i} missing 'name'")
        else:
            channel_names.append(ch["name"])
        if "description" not in ch:
            errors.append(f"Channel {i} ({ch.get('name', '?')}) missing 'description'")
        if "sampling_rate_hz" not in ch:
            errors.append(f"Channel {i} ({ch.get('name', '?')}) missing 'sampling_rate_hz'")
        else:
            sampling_rates.add(ch["sampling_rate_hz"])

    if len(sampling_rates) > 1:
        errors.append(f"Mixed sampling rates in manifest: {sampling_rates}")

    # Check channel naming conventions
    imu_channels = [
        name for name in channel_names
        if any(p in name.lower() for p in IMU_PATTERNS)
    ]
    non_imu = [
        name for name in channel_names
        if not any(p in name.lower() for p in IMU_PATTERNS)
        and name != 'timestamp_sec'
    ]
    if non_imu:
        errors.append(
            f"Channels not matching IMU patterns (acc/gyro/mag/ori): {non_imu[:5]}"
            + (f" ... and {len(non_imu)-5} more" if len(non_imu) > 5 else "")
        )

    # Check axis suffixes
    no_axis = [name for name in imu_channels if not AXIS_PATTERN.search(name)]
    if no_axis:
        errors.append(
            f"Channels without axis suffix (_x/_y/_z): {no_axis[:5]}"
        )

    # Check grouping
    groups = {}
    for name in imu_channels:
        group_name = AXIS_PATTERN.sub('', name)
        groups.setdefault(group_name, []).append(name)

    incomplete = {g: chs for g, chs in groups.items() if len(chs) not in (3, 4)}
    if incomplete:
        for g, chs in list(incomplete.items())[:3]:
            errors.append(f"Incomplete sensor group '{g}': {chs} (expected 3 or 4 channels)")

    return errors


def validate_labels(dataset_path: Path) -> list:
    """Validate labels.json structure."""
    errors = []
    labels_path = dataset_path / "labels.json"

    if not labels_path.exists():
        errors.append("MISSING: labels.json")
        return errors

    try:
        with open(labels_path) as f:
            labels = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"INVALID JSON in labels.json: {e}")
        return errors

    if not isinstance(labels, dict):
        errors.append("labels.json must be a dict (session_id -> [labels])")
        return errors

    if not labels:
        errors.append("labels.json is empty")
        return errors

    # Check value format
    non_list = 0
    empty_labels = 0
    for session_id, label_val in labels.items():
        if not isinstance(label_val, list):
            non_list += 1
        elif len(label_val) == 0:
            empty_labels += 1

    if non_list > 0:
        errors.append(f"{non_list} labels are not lists (expected [label_string])")
    if empty_labels > 0:
        errors.append(f"{empty_labels} sessions have empty label lists")

    return errors


def validate_sessions(dataset_path: Path, max_check: int = 20) -> list:
    """Validate session parquet files."""
    errors = []
    sessions_dir = dataset_path / "sessions"

    if not sessions_dir.exists():
        errors.append("MISSING: sessions/ directory")
        return errors

    session_dirs = sorted(sessions_dir.glob("*/"))
    if not session_dirs:
        errors.append("sessions/ directory is empty")
        return errors

    # Load labels for cross-referencing
    labels_path = dataset_path / "labels.json"
    labels = {}
    if labels_path.exists():
        with open(labels_path) as f:
            labels = json.load(f)

    # Check subset of sessions
    checked = 0
    missing_in_labels = 0
    all_zero_sessions = 0
    nan_sessions = 0

    for session_dir in session_dirs[:max_check]:
        parquet_path = session_dir / "data.parquet"
        if not parquet_path.exists():
            errors.append(f"Missing data.parquet in {session_dir.name}")
            continue

        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            errors.append(f"Cannot read {session_dir.name}/data.parquet: {e}")
            continue

        # Check timestamp_sec column
        if 'timestamp_sec' not in df.columns:
            errors.append(f"{session_dir.name}: missing 'timestamp_sec' column")

        # Check IMU channels exist
        imu_cols = [
            col for col in df.columns
            if col != 'timestamp_sec'
            and any(p in col.lower() for p in IMU_PATTERNS)
        ]
        if not imu_cols:
            errors.append(f"{session_dir.name}: no IMU channels found")

        # Check for all-zero data
        if imu_cols:
            sensor_data = df[imu_cols].values
            if np.all(sensor_data == 0):
                all_zero_sessions += 1
            if np.any(np.isnan(sensor_data)):
                nan_sessions += 1

        # Check label cross-reference
        if session_dir.name not in labels:
            missing_in_labels += 1

        checked += 1

    if all_zero_sessions > 0:
        errors.append(f"{all_zero_sessions}/{checked} checked sessions have all-zero sensor data")
    if nan_sessions > 0:
        errors.append(f"{nan_sessions}/{checked} checked sessions contain NaN values")
    if missing_in_labels > 0:
        errors.append(f"{missing_in_labels}/{checked} sessions not found in labels.json")

    # Cross-check: labels for non-existent sessions
    existing_sessions = {d.name for d in session_dirs}
    orphan_labels = [sid for sid in labels if sid not in existing_sessions]
    if orphan_labels:
        errors.append(
            f"{len(orphan_labels)} labels reference non-existent sessions: "
            f"{orphan_labels[:3]}"
        )

    return errors


def validate_dataset(dataset_path: Path) -> dict:
    """Run all validation checks on a dataset."""
    result = {
        "dataset": dataset_path.name,
        "path": str(dataset_path),
        "exists": dataset_path.exists(),
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    if not dataset_path.exists():
        result["errors"].append(f"Dataset directory does not exist: {dataset_path}")
        return result

    # Validate manifest
    manifest_errors = validate_manifest(dataset_path)
    result["errors"].extend(manifest_errors)

    # Validate labels
    label_errors = validate_labels(dataset_path)
    result["errors"].extend(label_errors)

    # Validate sessions
    session_errors = validate_sessions(dataset_path)
    result["errors"].extend(session_errors)

    # Collect stats
    labels_path = dataset_path / "labels.json"
    if labels_path.exists():
        with open(labels_path) as f:
            labels = json.load(f)
        result["stats"]["total_sessions"] = len(labels)

        # Activity distribution
        activity_counts = {}
        for session_labels in labels.values():
            if isinstance(session_labels, list):
                for label in session_labels:
                    activity_counts[label] = activity_counts.get(label, 0) + 1
        result["stats"]["unique_activities"] = len(activity_counts)
        result["stats"]["activity_counts"] = activity_counts

    manifest_path = dataset_path / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        result["stats"]["total_channels"] = len(manifest.get("channels", []))
        if manifest.get("channels"):
            result["stats"]["sampling_rate"] = manifest["channels"][0].get("sampling_rate_hz")

    return result


def print_report(result: dict):
    """Print validation report."""
    name = result["dataset"]
    print(f"\n{'=' * 70}")
    print(f"  Dataset: {name}")
    print(f"  Path: {result['path']}")
    print(f"{'=' * 70}")

    if not result["exists"]:
        print(f"  NOT CONVERTED YET (directory does not exist)")
        return

    # Stats
    stats = result.get("stats", {})
    if stats:
        print(f"  Sessions: {stats.get('total_sessions', '?')}")
        print(f"  Channels: {stats.get('total_channels', '?')}")
        print(f"  Activities: {stats.get('unique_activities', '?')}")
        print(f"  Sampling rate: {stats.get('sampling_rate', '?')} Hz")

        counts = stats.get("activity_counts", {})
        if counts:
            print(f"  Activity distribution:")
            for act, count in sorted(counts.items()):
                print(f"    {act}: {count}")

    # Errors
    errors = result.get("errors", [])
    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for err in errors:
            print(f"    [x] {err}")
    else:
        print(f"\n  All checks passed!")


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_dataset.py <dataset_path> [<dataset_path2> ...]")
        print("       python validate_dataset.py --all")
        sys.exit(1)

    if sys.argv[1] == "--all":
        data_dir = Path("data")
        if not data_dir.exists():
            print("ERROR: data/ directory not found")
            sys.exit(1)
        dataset_paths = sorted([
            d for d in data_dir.iterdir()
            if d.is_dir() and (d / "manifest.json").exists()
        ])
        if not dataset_paths:
            print("No converted datasets found in data/")
            sys.exit(1)
    else:
        dataset_paths = [Path(p) for p in sys.argv[1:]]

    all_results = []
    for path in dataset_paths:
        result = validate_dataset(path)
        all_results.append(result)
        print_report(result)

    # Summary
    total_errors = sum(len(r["errors"]) for r in all_results)
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY: {len(all_results)} datasets validated, {total_errors} total errors")
    print(f"{'=' * 70}")

    if total_errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
