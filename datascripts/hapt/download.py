"""
Download script for HAPT (Human Activities and Postural Transitions) dataset.

Source: UCI Machine Learning Repository
URL: https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions

Dataset contains:
- 30 subjects (19-48 years)
- 12 activities: 6 basic (walking, walking_upstairs, walking_downstairs, sitting, standing, laying)
                 + 6 postural transitions (stand_to_sit, sit_to_stand, sit_to_lie, lie_to_sit, stand_to_lie, lie_to_stand)
- Smartphone (Samsung Galaxy S II) on waist
- 50 Hz sampling rate
- Accelerometer + Gyroscope (6 channels)
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path


DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/341/smartphone+based+recognition+of+human+activities+and+postural+transitions.zip"
RAW_DIR = Path("data/raw/hapt")
ZIP_FILE = RAW_DIR / "hapt.zip"


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


def download_hapt():
    """Download and extract HAPT dataset."""
    print("=" * 80)
    print("HAPT Dataset Download")
    print("=" * 80)

    # Create directory
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    raw_data_dir = RAW_DIR / "RawData"
    if raw_data_dir.exists():
        txt_files = list(raw_data_dir.glob("*.txt"))
        if txt_files:
            print(f"✓ Dataset already downloaded: {raw_data_dir}")
            print(f"  Found {len(txt_files)} files")
            return True

    # Download if needed
    if not ZIP_FILE.exists():
        print(f"\nDownloading HAPT dataset...")
        if not download_file(DOWNLOAD_URL, ZIP_FILE):
            return False
    else:
        print(f"  ZIP file already exists: {ZIP_FILE}")

    # Extract
    print(f"\nExtracting...")
    if not extract_zip(ZIP_FILE, RAW_DIR):
        return False

    # Verify extraction
    # The ZIP may have nested structure
    for subdir in RAW_DIR.iterdir():
        if subdir.is_dir():
            raw_data = subdir / "RawData"
            if raw_data.exists():
                print(f"\n✓ Download complete!")
                print(f"  Location: {raw_data}")
                return True
            # Check if RawData is directly in subdir
            if (subdir / "acc_exp01_user01.txt").exists():
                print(f"\n✓ Download complete!")
                print(f"  Location: {subdir}")
                return True

    # Check RAW_DIR directly
    if raw_data_dir.exists():
        print(f"\n✓ Download complete!")
        print(f"  Location: {raw_data_dir}")
        return True

    print(f"\n✗ Could not verify extraction")
    return False


def main():
    """Main entry point."""
    success = download_hapt()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
