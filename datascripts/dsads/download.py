"""
Download script for DSADS (Daily and Sports Activities) dataset.

Source: UCI Machine Learning Repository
URL: https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities

Dataset contains:
- 8 subjects (4 male, 4 female, ages 20-30)
- 19 activities: sitting, standing, lying, stairs, walking, running, exercising, cycling, etc.
- 5 body positions: torso (T), right arm (RA), left arm (LA), right leg (RL), left leg (LL)
- 9 sensors per position: accelerometer (xyz), gyroscope (xyz), magnetometer (xyz)
- 25 Hz sampling rate
- 5-second segments (125 samples each)
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path


DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/256/daily+and+sports+activities.zip"
RAW_DIR = Path("data/raw/dsads")
ZIP_FILE = RAW_DIR / "daily_and_sports_activities.zip"


def download_file(url: str, dest_path: Path):
    """Download file with progress reporting."""
    print(f"  Downloading from: {url}")

    try:
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        request = urllib.request.Request(url, headers=headers)

        response = urllib.request.urlopen(request, timeout=300)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 8192
        downloaded = 0

        with open(dest_path, "wb") as f:
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = min(downloaded * 100.0 / total_size, 100)
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    sys.stdout.write(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
                    sys.stdout.flush()

        print()
        return True
    except Exception as e:
        print(f"\n  ERROR: Failed to download: {e}")
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


def download_dsads():
    """Download and extract DSADS dataset."""
    print("=" * 80)
    print("DSADS (Daily and Sports Activities) Dataset Download")
    print("=" * 80)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    data_dir = RAW_DIR / "data"
    if not data_dir.exists():
        # Try alternative locations
        for candidate in ["data", "daily+and+sports+activities", "Daily and Sports Activities"]:
            candidate_path = RAW_DIR / candidate
            if candidate_path.exists():
                data_dir = candidate_path
                break

    if data_dir.exists():
        activity_folders = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("a")]
        if activity_folders:
            print(f"✓ Dataset already downloaded: {data_dir}")
            print(f"  Found {len(activity_folders)} activities")
            return True

    # Download if needed
    if not ZIP_FILE.exists():
        print(f"\nDownloading DSADS dataset (~27 MB)...")
        if not download_file(DOWNLOAD_URL, ZIP_FILE):
            return False
    else:
        print(f"  ZIP file already exists: {ZIP_FILE}")

    # Extract
    print(f"\nExtracting...")
    if not extract_zip(ZIP_FILE, RAW_DIR):
        return False

    # Find extracted data directory
    for candidate in ["data", "daily+and+sports+activities", "Daily and Sports Activities"]:
        candidate_path = RAW_DIR / candidate
        if candidate_path.exists():
            data_dir = candidate_path
            break

    if data_dir.exists():
        activity_folders = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("a")]
        print(f"\n✓ Download complete!")
        print(f"  Location: {data_dir}")
        print(f"  Activities: {len(activity_folders)}")
        return True
    else:
        print(f"\n✗ Expected data directory not found")
        print(f"  Check contents of: {RAW_DIR}")
        return False


def main():
    """Main entry point."""
    success = download_dsads()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
