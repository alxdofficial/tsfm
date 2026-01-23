"""
Download script for KU-HAR dataset.

Source: Kaggle / GitHub
URL: https://www.kaggle.com/datasets/niloy333/kuhar
GitHub: https://github.com/Niloy333/KU-HAR

Dataset contains:
- 90 subjects (large subject diversity)
- 18 activities including: Stand, Sit, Talk-sit, Talk-stand, Walk, Lay,
  Walk-up, Walk-down, Pick, Jump, Run, Push-up, Sit-up, Walk-circle, etc.
- Smartphone IMU sensors
- 100 Hz sampling rate
- Accelerometer + Gyroscope (6 channels)

Note: Requires Kaggle API credentials for automatic download.
Manual download: https://www.kaggle.com/datasets/niloy333/kuhar
"""

import os
import sys
import zipfile
from pathlib import Path


RAW_DIR = Path("data/raw/kuhar")
ZIP_FILE = RAW_DIR / "kuhar.zip"


def check_kaggle_api():
    """Check if Kaggle API is available."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        return True
    except ImportError:
        return False


def download_from_kaggle():
    """Download dataset using Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        print("  Using Kaggle API...")
        api = KaggleApi()
        api.authenticate()

        print("  Downloading dataset...")
        api.dataset_download_files(
            "niloy333/kuhar",
            path=str(RAW_DIR),
            unzip=True
        )
        return True
    except Exception as e:
        print(f"  ERROR: Kaggle download failed: {e}")
        return False


def extract_zip(zip_path: Path, extract_dir: Path):
    """Extract ZIP archive."""
    print(f"  Extracting to: {extract_dir}")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"  ✓ Extracted successfully")
        return True
    except Exception as e:
        print(f"  ERROR: Failed to extract: {e}")
        return False


def download_kuhar():
    """Download and extract KU-HAR dataset."""
    print("=" * 80)
    print("KU-HAR Dataset Download")
    print("=" * 80)

    # Create directory
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    csv_files = list(RAW_DIR.glob("**/*.csv"))
    if csv_files:
        print(f"✓ Dataset already downloaded: {RAW_DIR}")
        print(f"  Found {len(csv_files)} CSV files")
        return True

    # Try Kaggle API
    if check_kaggle_api():
        print("\nDownloading from Kaggle...")
        if download_from_kaggle():
            csv_files = list(RAW_DIR.glob("**/*.csv"))
            if csv_files:
                print(f"\n✓ Download complete!")
                print(f"  Found {len(csv_files)} CSV files")
                return True

    # Check for manual download
    if ZIP_FILE.exists():
        print(f"\n  Found ZIP file: {ZIP_FILE}")
        if extract_zip(ZIP_FILE, RAW_DIR):
            csv_files = list(RAW_DIR.glob("**/*.csv"))
            if csv_files:
                print(f"\n✓ Extraction complete!")
                print(f"  Found {len(csv_files)} CSV files")
                return True

    # Manual download instructions
    print("\n" + "=" * 80)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 80)
    print("\nKaggle API not available or download failed.")
    print("\nOption 1: Install Kaggle API")
    print("  pip install kaggle")
    print("  # Set up ~/.kaggle/kaggle.json with your API credentials")
    print("  # Then run this script again")
    print("\nOption 2: Manual download")
    print("  1. Go to: https://www.kaggle.com/datasets/niloy333/kuhar")
    print("  2. Download the dataset")
    print(f"  3. Place ZIP file at: {ZIP_FILE}")
    print("  4. Run this script again")
    print("\nOption 3: GitHub download")
    print("  git clone https://github.com/Niloy333/KU-HAR.git data/raw/kuhar")

    return False


def main():
    """Main entry point."""
    success = download_kuhar()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
