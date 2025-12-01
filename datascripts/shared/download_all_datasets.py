"""
Unified dataset download script for all time series activity recognition datasets.

Downloads and organizes raw datasets from UCI and other sources.
Each dataset will be converted to standardized format in separate conversion scripts.
"""

import os
import sys
import ssl
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
    },
    # New datasets for better balance
    "unimib_shar": {
        "name": "UniMiB SHAR",
        "url": None,  # Uses Kaggle API
        "kaggle_dataset": "wangboluo/unimib-shar-dataset",
        "raw_dir": "data/raw/unimib_shar",
        "extract_subdir": "",  # Kaggle extracts directly to raw_dir
        "verify_file": "unimib_train.csv"  # Check this file instead of subdir
    },
    "hhar": {
        "name": "HHAR",
        "url": "https://archive.ics.uci.edu/static/public/344/heterogeneity+activity+recognition.zip",
        "raw_dir": "data/raw/hhar",
        "extract_subdir": "Activity recognition exp"
    }
}


def download_file(url: str, dest_path: str, skip_ssl_verify: bool = False):
    """Download file with progress reporting."""
    print(f"  Downloading from: {url}")

    try:
        # Set up SSL context if needed
        context = None
        if skip_ssl_verify:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            print("  (SSL verification disabled for this download)")

        # Create request with User-Agent header to avoid 403 errors
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'}
        request = urllib.request.Request(url, headers=headers)

        # Download with progress
        if context:
            response = urllib.request.urlopen(request, context=context)
        else:
            response = urllib.request.urlopen(request)

        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0

        with open(dest_path, 'wb') as f:
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = min(downloaded * 100.0 / total_size, 100)
                    sys.stdout.write(f"\r  Progress: {percent:.1f}%")
                    sys.stdout.flush()

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


def download_kaggle_dataset(kaggle_dataset: str, dest_dir: str):
    """Download dataset from Kaggle using kaggle API."""
    print(f"  Downloading from Kaggle: {kaggle_dataset}")
    try:
        import subprocess
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", kaggle_dataset, "-p", dest_dir, "--unzip"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"  ERROR: Kaggle download failed: {result.stderr}")
            print("  Make sure you have kaggle installed (pip install kaggle)")
            print("  And have your API key at ~/.kaggle/kaggle.json")
            print("  Get it from: https://www.kaggle.com/settings/account")
            return False
        print(f"  ✓ Downloaded and extracted successfully")
        return True
    except FileNotFoundError:
        print("  ERROR: 'kaggle' command not found.")
        print("  Install with: pip install kaggle")
        print("  Then add your API key to ~/.kaggle/kaggle.json")
        return False
    except Exception as e:
        print(f"  ERROR: Kaggle download failed: {e}")
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

    # Determine what to check for completion
    verify_file = dataset.get('verify_file')
    extract_subdir = dataset.get('extract_subdir', '')
    expected_subdir = raw_dir / extract_subdir if extract_subdir else raw_dir

    if verify_file:
        expected_path = raw_dir / verify_file
    elif extract_subdir:
        expected_path = raw_dir / extract_subdir
    else:
        expected_path = raw_dir

    # Check if already downloaded
    if expected_path.exists():
        print(f"  Dataset already extracted: {expected_path}")
        return True

    # Handle Kaggle datasets
    if dataset.get('kaggle_dataset'):
        if not download_kaggle_dataset(dataset['kaggle_dataset'], str(raw_dir)):
            return False
        # Verify extraction
        if expected_path.exists():
            print(f"  ✓ Dataset ready at: {expected_path}")
            return True
        else:
            # Kaggle might extract to different name, check what was created
            print(f"  WARNING: Expected file/directory not found: {expected_path}")
            print(f"  Check contents of: {raw_dir}")
            return False

    # Handle multi-file datasets (if any dataset has 'files' dict)
    if dataset.get('files'):
        expected_subdir.mkdir(parents=True, exist_ok=True)
        all_success = True
        for file_key, file_url in dataset['files'].items():
            zip_path = raw_dir / f"{file_key}.zip"
            extract_marker = expected_subdir / f".{file_key}_extracted"

            if extract_marker.exists():
                print(f"  {file_key}: already extracted")
                continue

            print(f"  Downloading {file_key}...")
            if not zip_path.exists():
                if not download_file(file_url, str(zip_path)):
                    all_success = False
                    continue

            print(f"  Extracting {file_key}...")
            if extract_zip(str(zip_path), str(expected_subdir)):
                # Create marker file
                extract_marker.touch()
            else:
                all_success = False

        if all_success:
            print(f"  ✓ Dataset ready at: {expected_subdir}")
        return all_success

    # Download ZIP from URL
    zip_path = raw_dir / "download.zip"
    if zip_path.exists():
        print(f"  ZIP already exists: {zip_path}")
    else:
        skip_ssl = dataset.get('skip_ssl_verify', False)
        if not download_file(dataset['url'], str(zip_path), skip_ssl_verify=skip_ssl):
            return False

    # Extract
    extract_dir = raw_dir

    if not expected_subdir.exists():
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
