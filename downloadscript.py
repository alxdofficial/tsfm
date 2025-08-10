#!/usr/bin/env python3
import os
import sys
import time
import math
import requests
from tqdm.auto import tqdm

# ---- Where to put the files (relative to this script) ----
DEST_DIR = os.path.join("data", "actionet")

# ---- Files to download ----
URLS = [
    "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-07_experiment_S00/2022-06-07_18-10-55_actionNet-wearables_S00/2022-06-07_18-11-37_streamLog_actionNet-wearables_S00.hdf5",
    "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-13_experiment_S01_recordingStopped/2022-06-13_18-13-12_actionNet-wearables_S01/2022-06-13_18-14-59_streamLog_actionNet-wearables_S01.hdf5",
    "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-13_experiment_S02/2022-06-13_21-47-57_actionNet-wearables_S02/2022-06-13_21-48-24_streamLog_actionNet-wearables_S02.hdf5",
    "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-13_experiment_S02/2022-06-13_22-34-45_actionNet-wearables_S02/2022-06-13_22-35-11_streamLog_actionNet-wearables_S02.hdf5",
    "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-13_experiment_S02/2022-06-13_23-22-21_actionNet-wearables_S02/2022-06-13_23-22-44_streamLog_actionNet-wearables_S02.hdf5",
    "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-14_experiment_S03/2022-06-14_13-11-44_actionNet-wearables_S03/2022-06-14_13-12-07_streamLog_actionNet-wearables_S03.hdf5",
    "https://data.csail.mit.edu/ActionNet/wearable_data/2022-06-14_experiment_S03/2022-06-14_13-52-21_actionNet-wearables_S03/2022-06-14_13-52-57_streamLog_actionNet-wearables_S03.hdf5",
]

HEADERS = {"User-Agent": "actionet-downloader/1.0"}

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def get_filename_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1]

def already_downloaded(path: str, expected_size: int | None) -> bool:
    if not os.path.exists(path):
        return False
    if expected_size is None or expected_size == 0:
        return True  # can't verify, assume it's fine
    return os.path.getsize(path) == expected_size

def download_file(url: str, dest_dir: str, timeout: int = 60, chunk_size: int = 1024 * 1024):
    ensure_dir(dest_dir)
    filename = get_filename_from_url(url)
    dest_path = os.path.join(dest_dir, filename)

    # HEAD to get size (may be missing)
    try:
        head = requests.head(url, allow_redirects=True, headers=HEADERS, timeout=timeout)
        total_size = int(head.headers.get("Content-Length", "0")) or None
    except Exception:
        total_size = None

    if already_downloaded(dest_path, total_size):
        print(f"[SKIP] {filename} already exists and matches size.")
        return

    with requests.get(url, stream=True, headers=HEADERS, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0")) or total_size or 0

        with open(dest_path, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=filename,
            initial=0,
            dynamic_ncols=True,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"[OK]   Downloaded: {filename} -> {dest_path}")

def main():
    print(f"[INFO] Saving to: {os.path.abspath(DEST_DIR)}")
    for url in URLS:
        try:
            download_file(url, DEST_DIR)
        except KeyboardInterrupt:
            print("\n[ABORTED] Interrupted by user.")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Failed to download {url}\n        {e}")

if __name__ == "__main__":
    main()
