"""
Download script for RecGym dataset.

Source: Kaggle (official RecGym dataset)
URL: https://www.kaggle.com/datasets/zhaxidelebsz/10-gym-exercises-with-615-abstracted-features

Dataset contains:
- 10 volunteers performing gym exercises over 5 sessions each
- 12 activities: Adductor, ArmCurl, BenchPress, LegCurl, LegPress, Riding,
  RopeSkipping, Running, Squat, StairsClimber, Walking, and Null
- 3 body positions: wrist, pocket, calf
- Sensors: accelerometer (xyz), gyroscope (xyz), capacitance
- 20 Hz sampling rate
- ~4.7 million instances

Requirements:
- kaggle package: pip install kaggle
- Kaggle API credentials in ~/.kaggle/kaggle.json
  (Get from https://www.kaggle.com/settings -> API -> Create New Token)
"""

import sys
from pathlib import Path


RAW_DIR = Path("data/raw/recgym")
KAGGLE_DATASET = "zhaxidelebsz/10-gym-exercises-with-615-abstracted-features"


def download_recgym():
    """Download RecGym dataset from Kaggle."""
    print("=" * 80)
    print("RecGym Dataset Download")
    print("=" * 80)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    csv_file = RAW_DIR / "RecGym.csv"
    if csv_file.exists():
        size_mb = csv_file.stat().st_size / (1024 * 1024)
        print(f"Dataset already downloaded: {csv_file}")
        print(f"  Size: {size_mb:.1f} MB")
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
    print("This may take a few minutes (~450 MB)...")

    try:
        api = KaggleApi()
        api.authenticate()

        api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(RAW_DIR),
            unzip=True
        )

        # Verify download
        if csv_file.exists():
            size_mb = csv_file.stat().st_size / (1024 * 1024)
            print(f"\nDownload complete!")
            print(f"  Location: {csv_file}")
            print(f"  Size: {size_mb:.1f} MB")
            return True
        else:
            print("\nERROR: Download completed but CSV file not found")
            print(f"Check contents of: {RAW_DIR}")
            return False

    except Exception as e:
        print(f"\nERROR: Failed to download: {e}")
        return False


def main():
    """Main entry point."""
    success = download_recgym()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
