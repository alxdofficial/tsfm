"""
Download script for VTT-ConIoT dataset.

Source: Zenodo
URL: https://zenodo.org/record/4683703

Dataset contains:
- 13 construction workers
- 16 activities grouped into 6 main tasks (painting, walking, etc.)
- 3 body positions: hip, upper arm, back of shoulder
- Sensors: 10-DOF IMU (accelerometer, gyroscope, magnetometer, barometer)
- ~100 Hz sampling rate (after synchronization)
- ~1 minute per activity per subject
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path


DOWNLOAD_URL = "https://zenodo.org/records/4683703/files/VTT_ConIot_Dataset.zip?download=1"
RAW_DIR = Path("data/raw/vtt_coniot")
ZIP_FILE = RAW_DIR / "VTT_ConIot_Dataset.zip"


def download_file(url: str, dest_path: Path):
    """Download file with progress reporting."""
    print(f"  Downloading from: {url}")

    try:
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        request = urllib.request.Request(url, headers=headers)

        response = urllib.request.urlopen(request, timeout=600)
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


def download_vtt_coniot():
    """Download and extract VTT-ConIoT dataset."""
    print("=" * 80)
    print("VTT-ConIoT Dataset Download")
    print("=" * 80)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already extracted (look for any CSV files or subdirectories)
    csv_files = list(RAW_DIR.glob("**/*.csv"))
    subdirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]

    if csv_files or subdirs:
        print(f"✓ Dataset already downloaded: {RAW_DIR}")
        if csv_files:
            print(f"  Found {len(csv_files)} CSV file(s)")
        if subdirs:
            print(f"  Found {len(subdirs)} subdirectory(ies)")
        return True

    # Download if needed
    if not ZIP_FILE.exists():
        print(f"\nDownloading VTT-ConIoT dataset (~363 MB)...")
        if not download_file(DOWNLOAD_URL, ZIP_FILE):
            print("\nAlternative: Download manually from:")
            print("  https://zenodo.org/record/4683703")
            print(f"\nExtract to: {RAW_DIR}/")
            return False
    else:
        print(f"  ZIP file already exists: {ZIP_FILE}")

    # Extract
    print(f"\nExtracting...")
    if not extract_zip(ZIP_FILE, RAW_DIR):
        return False

    # Verify extraction
    csv_files = list(RAW_DIR.glob("**/*.csv"))
    subdirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]

    if csv_files or subdirs:
        print(f"\n✓ Download complete!")
        print(f"  Location: {RAW_DIR}")
        if csv_files:
            print(f"  CSV files: {len(csv_files)}")
        if subdirs:
            print(f"  Subdirectories: {len(subdirs)}")
        return True
    else:
        print(f"\n✗ No data files found after extraction")
        print(f"  Check contents of: {RAW_DIR}")
        return False


def main():
    """Main entry point."""
    success = download_vtt_coniot()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
