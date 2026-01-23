"""
Download script for RealWorld HAR dataset.

Source: University of Mannheim
URL: https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/

Dataset contains:
- 15 subjects (8 male, 7 female)
- 8 activities: walking, running, sitting, standing, lying, stairs_up, stairs_down, jumping
- 7 body positions: chest, forearm, head, shin, thigh, upperarm, waist
- Sensors: accelerometer, gyroscope, magnetometer, GPS, light, sound
- ~10 minutes per activity per subject (jumping ~1.7 minutes)
"""

import os
import sys
import ssl
import urllib.request
import zipfile
from pathlib import Path


DOWNLOAD_URL = "http://wifo5-14.informatik.uni-mannheim.de/sensor/dataset/realworld2016/realworld2016_dataset.zip"
RAW_DIR = Path("data/raw/realworld")
ZIP_FILE = RAW_DIR / "realworld2016_dataset.zip"


def download_file(url: str, dest_path: Path):
    """Download file with progress reporting."""
    print(f"  Downloading from: {url}")

    try:
        # Create request with User-Agent header
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

        print()  # New line after progress
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


def download_realworld():
    """Download and extract RealWorld HAR dataset."""
    print("=" * 80)
    print("RealWorld HAR Dataset Download")
    print("=" * 80)

    # Create directory
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already extracted - may be in RAW_DIR directly or in subdirectory
    expected_subdir = RAW_DIR / "realworld2016_dataset"

    # Check subdirectory first
    if expected_subdir.exists():
        subject_folders = [d for d in expected_subdir.iterdir() if d.is_dir() and d.name.startswith("proband")]
        if subject_folders:
            print(f"✓ Dataset already downloaded: {expected_subdir}")
            print(f"  Found {len(subject_folders)} subjects")
            return True

    # Check RAW_DIR directly (ZIP may extract without subdirectory)
    subject_folders = [d for d in RAW_DIR.iterdir() if d.is_dir() and d.name.startswith("proband")]
    if subject_folders:
        print(f"✓ Dataset already downloaded: {RAW_DIR}")
        print(f"  Found {len(subject_folders)} subjects")
        return True

    # Download if needed
    if not ZIP_FILE.exists():
        print(f"\nDownloading RealWorld dataset (~3.5 GB)...")
        if not download_file(DOWNLOAD_URL, ZIP_FILE):
            return False
    else:
        print(f"  ZIP file already exists: {ZIP_FILE}")

    # Extract
    print(f"\nExtracting...")
    if not extract_zip(ZIP_FILE, RAW_DIR):
        return False

    # Verify extraction - check both possible locations
    if expected_subdir.exists():
        subject_folders = [d for d in expected_subdir.iterdir() if d.is_dir() and d.name.startswith("proband")]
        if subject_folders:
            print(f"\n✓ Download complete!")
            print(f"  Location: {expected_subdir}")
            print(f"  Subjects: {len(subject_folders)}")
            return True

    # Check RAW_DIR directly
    subject_folders = [d for d in RAW_DIR.iterdir() if d.is_dir() and d.name.startswith("proband")]
    if subject_folders:
        print(f"\n✓ Download complete!")
        print(f"  Location: {RAW_DIR}")
        print(f"  Subjects: {len(subject_folders)}")
        return True

    print(f"\n✗ No proband directories found")
    print(f"  Check contents of: {RAW_DIR}")
    return False


def main():
    """Main entry point."""
    success = download_realworld()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
