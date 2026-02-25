#!/usr/bin/env python3
"""Export TSFM session data to standardized per-subject raw CSVs.

For each of the 14 datasets, reads session parquet files from data/{dataset}/sessions/,
groups them by subject, concatenates within each activity segment, and writes
per-subject CSVs to benchmark_data/raw/{dataset}/.

Also writes metadata.json for each dataset with channel info, activities, subjects, etc.

Usage:
    python benchmark_data/scripts/export_raw.py [--datasets uci_har hhar ...]
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BENCHMARK_DIR = PROJECT_ROOT / "benchmark_data"
RAW_DIR = BENCHMARK_DIR / "raw"

# Load dataset config
with open(BENCHMARK_DIR / "dataset_config.json") as f:
    CONFIG = json.load(f)

ALL_DATASETS = CONFIG["train_datasets"] + CONFIG["zero_shot_datasets"]


def extract_subject(session_name: str, dataset: str):
    """Extract subject identifier from session name."""
    if dataset == "uci_har":
        # train_01_0000 or test_02_0000 -> subject number
        return int(session_name.split("_")[1])
    elif dataset == "hhar":
        # hhar_a_bike_00000 -> 'a'
        return session_name.split("_")[1]
    elif dataset == "pamap2":
        # subject101_seg000_0000 -> 101
        return int(re.search(r"subject(\d+)", session_name).group(1))
    elif dataset == "wisdm":
        # phone_accel_data_1600_accel_phone_A_0000 -> 1600
        return int(session_name.split("_")[3])
    elif dataset == "dsads":
        # a01_p01_s01 -> 1 (from p01)
        return int(session_name.split("_")[1][1:])
    elif dataset == "kuhar":
        # s1001_jumping_0910_0000 -> 1001
        return int(re.search(r"s(\d+)", session_name).group(1))
    elif dataset == "unimib_shar":
        # session_00000 -> need to look up from raw labels
        return None  # handled specially
    elif dataset == "hapt":
        # exp01_user01_lying_0000 -> 1
        return int(re.search(r"user(\d+)", session_name).group(1))
    elif dataset == "mhealth":
        # subject10_seg000_0000 -> 10
        return int(re.search(r"subject(\d+)", session_name).group(1))
    elif dataset == "recgym":
        # s334_adductor_machine_0000 -> 's334'
        return session_name.split("_")[0]
    elif dataset == "motionsense":
        # sub01_dws_01_0000 -> 1
        return int(re.search(r"sub(\d+)", session_name).group(1))
    elif dataset == "realworld":
        # s01_climbingdown_0000 -> 1
        return int(re.search(r"s(\d+)", session_name).group(1))
    elif dataset == "mobiact":
        # BSC_01_01_0000 -> 1 (second field is subject)
        return int(session_name.split("_")[1])
    elif dataset == "vtt_coniot":
        # u01_carrying -> 'u01'
        return session_name.split("_")[0]
    elif dataset == "shoaib":
        # shoaib_P01_cycling_0000 -> 'P01'
        return session_name.split("_")[1]
    elif dataset == "opportunity":
        # opportunity_S1_seg000_0000 -> 'S1'
        return session_name.split("_")[1]
    elif dataset == "harth":
        # harth_S006_walking_0000_0000 -> 'S006'
        return session_name.split("_")[1]
    elif dataset == "usc_had":
        # usc_had_s01_walking_forward_t01_0000 -> 1
        return int(re.search(r"s(\d+)", session_name).group(1))
    elif dataset == "realdisp":
        # realdisp_s01_walking_0000 -> 1
        return int(re.search(r"s(\d+)", session_name).group(1))
    elif dataset == "daphnet_fog":
        # daphnet_S01R01_seg000_0000 -> 'S01R01'
        return session_name.split("_")[1]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def load_unimib_subject_map():
    """Build session index -> subject mapping for UniMiB SHAR from raw labels."""
    labels = np.load(DATA_DIR / "raw" / "unimib_shar" / "UniMiB" / "acc_labels.npy")
    # labels[:, 1] is subject (1-30), rows correspond to session_00000, session_00001, ...
    return {f"session_{i:05d}": int(labels[i, 1]) for i in range(len(labels))}


def load_sessions(dataset: str):
    """Load all sessions for a dataset, returning list of (session_name, df, activity)."""
    sessions_dir = DATA_DIR / dataset / "sessions"
    labels_path = DATA_DIR / dataset / "labels.json"

    with open(labels_path) as f:
        labels = json.load(f)

    session_names = sorted(os.listdir(sessions_dir))
    results = []
    for sname in session_names:
        parquet_path = sessions_dir / sname / "data.parquet"
        if not parquet_path.exists():
            continue
        df = pd.read_parquet(parquet_path)
        activity = labels.get(sname, ["unknown"])[0]
        results.append((sname, df, activity))

    return results


def group_by_subject(sessions, dataset: str):
    """Group sessions by subject. Returns dict: subject -> [(session_name, df, activity)]."""
    if dataset == "unimib_shar":
        subject_map = load_unimib_subject_map()

    groups = defaultdict(list)
    for sname, df, activity in sessions:
        if dataset == "unimib_shar":
            subject = subject_map.get(sname)
            if subject is None:
                continue
        else:
            subject = extract_subject(sname, dataset)
        groups[subject].append((sname, df, activity))

    return groups


def get_channel_rename_map(dataset: str):
    """Get column rename mapping: original_name -> standardized_name.

    Core channels are renamed to acc_x/y/z, gyro_x/y/z.
    Extra channels keep their original names.
    """
    ds_config = CONFIG["datasets"][dataset]
    core_map = ds_config["core_channels"]
    # Invert: original_col_name -> standard_name
    rename = {}
    for standard, original in core_map.items():
        rename[original] = standard
    return rename


def build_subject_csv(subject_sessions, dataset: str):
    """Build a single per-subject DataFrame from grouped sessions.

    Sessions are sorted by name and concatenated. Each session's activity label
    is added as a column. Timestamps are recomputed as continuous from 0.
    """
    ds_config = CONFIG["datasets"][dataset]
    sampling_rate = ds_config["sampling_rate_hz"]
    dt = 1.0 / sampling_rate

    rename_map = get_channel_rename_map(dataset)

    # Sort sessions by name for deterministic ordering
    subject_sessions.sort(key=lambda x: x[0])

    all_rows = []
    time_offset = 0.0

    for sname, df, activity in subject_sessions:
        df = df.copy()

        # Drop old timestamp, we'll rebuild it
        if "timestamp_sec" in df.columns:
            df = df.drop(columns=["timestamp_sec"])

        # Rename core channels
        df = df.rename(columns=rename_map)

        # Add activity column
        df["activity"] = activity

        # Add continuous timestamp
        n = len(df)
        df.insert(0, "timestamp_sec", np.round(time_offset + np.arange(n) * dt, 4))
        time_offset += n * dt

        all_rows.append(df)

    result = pd.concat(all_rows, ignore_index=True)

    # Reorder columns: timestamp_sec, core channels, extra channels, activity
    core_cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    present_core = [c for c in core_cols if c in result.columns]
    extra_cols = [
        c
        for c in result.columns
        if c not in (["timestamp_sec", "activity"] + core_cols)
    ]
    col_order = ["timestamp_sec"] + present_core + sorted(extra_cols) + ["activity"]
    # Only include columns that exist
    col_order = [c for c in col_order if c in result.columns]
    result = result[col_order]

    return result


def subsample_sessions(sessions, dataset: str, target: int, seed: int = 42):
    """Stratified subsample of sessions to target count."""
    if len(sessions) <= target:
        return sessions

    rng = np.random.RandomState(seed)

    # Group by (subject, activity) for stratification
    strata = defaultdict(list)
    for item in sessions:
        sname, df, activity = item
        if dataset == "unimib_shar":
            subject_map = load_unimib_subject_map()
            subject = subject_map.get(sname, "unknown")
        else:
            subject = extract_subject(sname, dataset)
        strata[(subject, activity)].append(item)

    # Proportional allocation per stratum
    total = len(sessions)
    selected = []
    remaining_budget = target

    strata_items = sorted(strata.items())
    for i, (key, items) in enumerate(strata_items):
        if i == len(strata_items) - 1:
            n_select = remaining_budget
        else:
            n_select = max(1, int(round(len(items) / total * target)))
            n_select = min(n_select, len(items), remaining_budget)
        remaining_budget -= n_select

        indices = rng.choice(len(items), size=n_select, replace=False)
        selected.extend(items[j] for j in indices)

    return selected


def write_metadata(dataset: str, subjects_written: list, out_dir: Path):
    """Write metadata.json for a dataset."""
    ds_config = CONFIG["datasets"][dataset]

    core_channels = list(ds_config["core_channels"].keys())
    extra_channels = ds_config.get("extra_channels", [])

    # Map extra channels to their standardized names (if renamed, they keep original)
    # Extra channels that weren't in rename map keep their original names
    rename_map = get_channel_rename_map(dataset)
    all_channels_out = list(core_channels)
    for ec in extra_channels:
        # Extra channels keep their original name if not in rename map
        out_name = rename_map.get(ec, ec)
        if out_name not in all_channels_out:
            all_channels_out.append(out_name)

    metadata = {
        "dataset_name": ds_config["display_name"],
        "sampling_rate_hz": ds_config["sampling_rate_hz"],
        "channels": core_channels,
        "extra_channels": [
            c for c in all_channels_out if c not in core_channels
        ],
        "activities": ds_config["activities"],
        "subjects": sorted(subjects_written, key=lambda x: str(x)),
        "placement": ds_config["placement"],
        "num_sessions_original": ds_config["num_sessions"],
        "train_subjects": ds_config["train_subjects"],
        "test_subjects": ds_config["test_subjects"],
    }

    if "num_sessions_subsampled" in ds_config:
        metadata["num_sessions_subsampled"] = ds_config["num_sessions_subsampled"]

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def format_subject_filename(subject) -> str:
    """Format subject ID into a filename like subject_01.csv."""
    if isinstance(subject, int):
        return f"subject_{subject:02d}.csv"
    else:
        return f"subject_{subject}.csv"


def export_dataset(dataset: str):
    """Export a single dataset to benchmark_data/raw/{dataset}/."""
    ds_config = CONFIG["datasets"][dataset]
    out_dir = RAW_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Exporting {ds_config['display_name']} ({dataset})")
    print(f"{'='*60}")

    # Load all sessions
    print(f"  Loading sessions from data/{dataset}/sessions/...")
    sessions = load_sessions(dataset)
    print(f"  Loaded {len(sessions)} sessions")

    # Subsample if needed
    subsample_config = CONFIG.get("subsampling", {}).get(dataset)
    if subsample_config:
        target = subsample_config["target_sessions"]
        print(f"  Subsampling from {len(sessions)} to ~{target} sessions...")
        sessions = subsample_sessions(sessions, dataset, target)
        print(f"  After subsampling: {len(sessions)} sessions")

    # Group by subject
    subject_groups = group_by_subject(sessions, dataset)
    print(f"  Found {len(subject_groups)} subjects")

    # Write per-subject CSVs
    subjects_written = []
    total_rows = 0
    for subject in sorted(subject_groups.keys(), key=lambda x: str(x)):
        subject_sessions = subject_groups[subject]
        df = build_subject_csv(subject_sessions, dataset)

        filename = format_subject_filename(subject)
        df.to_csv(out_dir / filename, index=False, float_format="%.6f")

        subjects_written.append(subject)
        total_rows += len(df)
        print(
            f"    {filename}: {len(subject_sessions)} sessions, "
            f"{len(df)} rows, {len(df.columns)-2} sensor cols"
        )

    # Write metadata
    write_metadata(dataset, subjects_written, out_dir)

    print(f"  Total: {total_rows} rows across {len(subjects_written)} subjects")
    print(f"  Output: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Export datasets to raw CSV format")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ALL_DATASETS,
        help=f"Datasets to export (default: all). Options: {ALL_DATASETS}",
    )
    args = parser.parse_args()

    # Validate datasets
    for ds in args.datasets:
        if ds not in ALL_DATASETS:
            print(f"Error: Unknown dataset '{ds}'. Valid: {ALL_DATASETS}")
            sys.exit(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {len(args.datasets)} datasets to {RAW_DIR}")
    print(f"Datasets: {args.datasets}")

    for ds in args.datasets:
        export_dataset(ds)

    print(f"\nDone! Exported {len(args.datasets)} datasets to {RAW_DIR}")


if __name__ == "__main__":
    main()
