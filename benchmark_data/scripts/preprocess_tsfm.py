#!/usr/bin/env python3
"""Set up TSFM processed data via symlinks (or copies) to existing data/ directories.

Creates symlinks from benchmark_data/processed/tsfm/{dataset}/ -> data/{dataset}/
so the TSFM model can find its data through the benchmark folder structure.

Usage:
    python benchmark_data/scripts/preprocess_tsfm.py           # Create symlinks
    python benchmark_data/scripts/preprocess_tsfm.py --copy     # Copy instead (for zipping)
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BENCHMARK_DIR = PROJECT_ROOT / "benchmark_data"
TSFM_DIR = BENCHMARK_DIR / "processed" / "tsfm"

with open(BENCHMARK_DIR / "dataset_config.json") as f:
    CONFIG = json.load(f)

ALL_DATASETS = CONFIG["train_datasets"] + CONFIG["zero_shot_datasets"]


def main():
    parser = argparse.ArgumentParser(
        description="Set up TSFM processed data links"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy data instead of symlinking (for creating a zip archive)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ALL_DATASETS,
        help=f"Datasets to link (default: all). Options: {ALL_DATASETS}",
    )
    args = parser.parse_args()

    TSFM_DIR.mkdir(parents=True, exist_ok=True)

    mode = "copy" if args.copy else "symlink"
    print(f"Setting up TSFM data ({mode} mode)")
    print(f"Source: {DATA_DIR}")
    print(f"Target: {TSFM_DIR}")

    for dataset in args.datasets:
        source = DATA_DIR / dataset
        target = TSFM_DIR / dataset

        if not source.exists():
            print(f"  WARNING: {source} does not exist, skipping")
            continue

        if target.exists() or target.is_symlink():
            if target.is_symlink():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)

        if args.copy:
            print(f"  Copying {dataset}...")
            shutil.copytree(source, target)
        else:
            # Create relative symlink
            rel_path = os.path.relpath(source, target.parent)
            target.symlink_to(rel_path)
            print(f"  {dataset} -> {rel_path}")

    print(f"\nDone! {len(args.datasets)} datasets set up in {TSFM_DIR}")


if __name__ == "__main__":
    main()
