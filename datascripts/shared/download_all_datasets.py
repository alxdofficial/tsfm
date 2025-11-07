"""
Unified dataset download script for all time series activity recognition datasets.

Downloads and organizes raw datasets from UCI and other sources.
Each dataset will be converted to standardized format in separate conversion scripts.
"""

import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path


# Dataset URLs
DATASETS = {
    "uci_har": {
        "name": "UCI HAR",
        "url": "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
        "raw_dir": "data/raw/uci_har",
        "extract_subdir": "UCI HAR Dataset"
    },
    "pamap2": {
        "name": "PAMAP2",
        "url": "http://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip",
        "raw_dir": "data/raw/pamap2",
        "extract_subdir": "PAMAP2_Dataset"
    },
    "mhealth": {
        "name": "MHEALTH",
        "url": "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip",
        "raw_dir": "data/raw/mhealth",
        "extract_subdir": "MHEALTHDATASET"
    },
    "wisdm": {
        "name": "WISDM",
        "url": "https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip",
        "raw_dir": "data/raw/wisdm",
        "extract_subdir": "wisdm-dataset"
    }
}


def download_file(url: str, dest_path: str):
    """Download file with progress reporting."""
    print(f"  Downloading from: {url}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(downloaded * 100.0 / total_size, 100)
            sys.stdout.write(f"\r  Progress: {percent:.1f}%")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, progress_hook)
        print()  # New line after progress
        return True
    except Exception as e:
        print(f"\n  ERROR: Failed to download: {e}")
        return False


def extract_zip(zip_path: str, extract_dir: str):
    """Extract ZIP archive."""
    print(f"  Extracting to: {extract_dir}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"  ✓ Extracted successfully")
        return True
    except Exception as e:
        print(f"  ERROR: Failed to extract: {e}")
        return False


def download_dataset(dataset_key: str):
    """Download and extract a single dataset."""
    if dataset_key not in DATASETS:
        print(f"ERROR: Unknown dataset: {dataset_key}")
        print(f"Available: {list(DATASETS.keys())}")
        return False

    dataset = DATASETS[dataset_key]
    print(f"\n{'=' * 80}")
    print(f"Downloading: {dataset['name']}")
    print(f"{'=' * 80}")

    # Create directories
    raw_dir = Path(dataset['raw_dir'])
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Download ZIP
    zip_path = raw_dir / "download.zip"
    if zip_path.exists():
        print(f"  ZIP already exists: {zip_path}")
    else:
        if not download_file(dataset['url'], str(zip_path)):
            return False

    # Extract
    extract_dir = raw_dir
    expected_subdir = raw_dir / dataset['extract_subdir']

    if expected_subdir.exists():
        print(f"  Dataset already extracted: {expected_subdir}")
    else:
        if not extract_zip(str(zip_path), str(extract_dir)):
            return False

        # Handle nested ZIPs (e.g., UCI HAR has "UCI HAR Dataset.zip" inside download.zip)
        nested_zip = raw_dir / f"{dataset['extract_subdir']}.zip"
        if nested_zip.exists() and not expected_subdir.exists():
            print(f"  Found nested ZIP: {nested_zip.name}, extracting...")
            if not extract_zip(str(nested_zip), str(extract_dir)):
                return False

    # Verify extraction
    if expected_subdir.exists():
        print(f"  ✓ Dataset ready at: {expected_subdir}")
        return True
    else:
        print(f"  WARNING: Expected directory not found: {expected_subdir}")
        print(f"  Check contents of: {extract_dir}")
        return False


def download_all():
    """Download all datasets."""
    print("=" * 80)
    print("Dataset Download Script")
    print("=" * 80)
    print(f"\nWill download {len(DATASETS)} datasets:")
    for key, info in DATASETS.items():
        print(f"  - {info['name']} ({key})")

    results = {}
    for dataset_key in DATASETS.keys():
        success = download_dataset(dataset_key)
        results[dataset_key] = success

    # Summary
    print(f"\n{'=' * 80}")
    print("Download Summary")
    print(f"{'=' * 80}")
    for dataset_key, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {DATASETS[dataset_key]['name']}: {status}")

    total_success = sum(results.values())
    print(f"\nCompleted: {total_success}/{len(DATASETS)} datasets")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Download specific dataset
        dataset_key = sys.argv[1]
        download_dataset(dataset_key)
    else:
        # Download all
        download_all()


if __name__ == "__main__":
    main()
