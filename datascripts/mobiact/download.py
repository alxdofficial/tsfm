"""
Download script for MobiFall dataset (MobiAct alternative).

Source: Kaggle
URL: https://www.kaggle.com/datasets/kmknation/mobifall-dataset-v20

Note: The full MobiAct dataset requires registration at https://bmi.hmu.gr/
This script downloads MobiFall v2.0, a publicly available dataset with similar
format containing falls and ADL activities.

Dataset contains:
- 31 subjects
- 4 fall types: BSC (backward), FKL (front-knees-lying), FOL (forward), SDL (sideward)
- 9 ADL activities: CSI (car step-in), CSO (car step-out), JOG (jogging), JUM (jumping),
  SCH (sit-chair), STD (stand-up), STN (standing), STU (stairs up), WAL (walking)
- Sensors: accelerometer, gyroscope, orientation (separate files)
- ~87 Hz sampling rate (Samsung Galaxy S3)

File format:
- Separate .txt files for each sensor type per trial
- Path: sub{N}/{FALLS|ADL}/{activity_code}/{activity}_acc_{subject}_{trial}.txt
- Header lines start with # (metadata)
- Data: timestamp(ns), x, y, z

Requirements:
- kaggle package: pip install kaggle
- Kaggle API credentials in ~/.kaggle/kaggle.json
"""

import sys
from pathlib import Path


RAW_DIR = Path("data/raw/mobiact")
KAGGLE_DATASET = "kmknation/mobifall-dataset-v20"


def download_mobifall():
    """Download MobiFall dataset from Kaggle."""
    print("=" * 80)
    print("MobiFall Dataset Download (MobiAct alternative)")
    print("=" * 80)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    expected_dir = RAW_DIR / "MobiFall_Dataset_v2.0"
    if expected_dir.exists():
        # Count txt files
        txt_files = list(expected_dir.glob("**/*.txt"))
        if len(txt_files) > 100:
            print(f"Dataset already downloaded: {expected_dir}")
            print(f"  Found {len(txt_files)} data files")
            return True

    # Also check for old MobiAct format
    old_format = RAW_DIR / "Annotated Data"
    if old_format.exists():
        csv_files = list(old_format.glob("**/*.csv"))
        if csv_files:
            print(f"Found MobiAct format dataset: {old_format}")
            print(f"  Found {len(csv_files)} CSV files")
            return True

    # Try to import kaggle
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("ERROR: kaggle package not installed")
        print("Install with: pip install kaggle")
        return False

    # Check for credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("ERROR: Kaggle API credentials not found")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Click 'API' -> 'Create New Token'")
        print("3. Save kaggle.json to ~/.kaggle/kaggle.json")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False

    # Download from Kaggle
    print(f"\nDownloading from Kaggle: {KAGGLE_DATASET}")
    print("This may take a few minutes (~95 MB)...")

    try:
        api = KaggleApi()
        api.authenticate()

        api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(RAW_DIR),
            unzip=True
        )

        # Verify download
        expected_dir = RAW_DIR / "MobiFall_Dataset_v2.0"
        if expected_dir.exists():
            txt_files = list(expected_dir.glob("**/*.txt"))
            print(f"\nDownload complete!")
            print(f"  Location: {expected_dir}")
            print(f"  Data files: {len(txt_files)}")
            return True
        else:
            # Check if extracted differently
            txt_files = list(RAW_DIR.glob("**/*.txt"))
            if txt_files:
                print(f"\nDownload complete!")
                print(f"  Location: {RAW_DIR}")
                print(f"  Data files: {len(txt_files)}")
                return True

            print("\nERROR: Download completed but data files not found")
            print(f"Check contents of: {RAW_DIR}")
            return False

    except Exception as e:
        print(f"\nERROR: Failed to download: {e}")
        print("\nAlternative: Full MobiAct dataset")
        print("1. Go to https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/")
        print("2. Fill out the registration form")
        print("3. Download and extract to data/raw/mobiact/")
        return False


def main():
    """Main entry point."""
    success = download_mobifall()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
