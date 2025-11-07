"""
Master pipeline script to download and convert all datasets.

Runs the complete pipeline:
1. Download raw data from sources
2. Convert to standardized format (parquet)
3. Generate minimal manifests
4. Validate output

Usage:
    python datascripts/setup_all_ts_datasets.py [dataset_name]

    Without arguments: process all datasets
    With dataset_name: process only that dataset

Available datasets:
    - uci_har
    - pamap2
    - mhealth
    - wisdm
    - actionsense (requires manual download first)
"""

import sys
import subprocess
from pathlib import Path


DATASETS = {
    "uci_har": {
        "name": "UCI HAR",
        "download_script": "datascripts/shared/download_all_datasets.py",
        "convert_script": "datascripts/uci_har/convert.py",
        "requires_manual": False
    },
    "pamap2": {
        "name": "PAMAP2",
        "download_script": "datascripts/shared/download_all_datasets.py",
        "convert_script": "datascripts/pamap2/convert.py",
        "requires_manual": False
    },
    "mhealth": {
        "name": "MHEALTH",
        "download_script": "datascripts/shared/download_all_datasets.py",
        "convert_script": "datascripts/mhealth/convert.py",
        "requires_manual": False
    },
    "wisdm": {
        "name": "WISDM",
        "download_script": "datascripts/shared/download_all_datasets.py",
        "convert_script": "datascripts/wisdm/convert.py",
        "requires_manual": False
    },
    "actionsense": {
        "name": "ActionSense",
        "download_script": None,  # Manual download
        "convert_script": "datascripts/actionsense/convert.py",
        "requires_manual": True
    }
}


def run_command(command: list, description: str):
    """Run a command and return success status."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Command failed with exit code {e.returncode}")
        return False


def process_dataset(dataset_key: str):
    """Process one dataset through the complete pipeline."""
    if dataset_key not in DATASETS:
        print(f"ERROR: Unknown dataset: {dataset_key}")
        return False

    dataset = DATASETS[dataset_key]

    print(f"\n{'=' * 80}")
    print(f"Processing: {dataset['name']}")
    print(f"{'=' * 80}")

    # Step 1: Download (if not manual)
    if dataset['requires_manual']:
        print(f"\n⚠ {dataset['name']} requires manual download")
        print("Please ensure data is available before conversion")
    else:
        download_success = run_command(
            [sys.executable, dataset['download_script'], dataset_key],
            f"Downloading {dataset['name']}"
        )
        if not download_success:
            print(f"✗ Failed to download {dataset['name']}")
            return False

    # Step 2: Convert
    convert_success = run_command(
        [sys.executable, dataset['convert_script']],
        f"Converting {dataset['name']} to standardized format"
    )

    if not convert_success:
        print(f"✗ Failed to convert {dataset['name']}")
        return False

    # Step 3: Validate output
    output_dir = Path(f"data/{dataset_key}")
    manifest_path = output_dir / "manifest.json"
    labels_path = output_dir / "labels.json"
    sessions_dir = output_dir / "sessions"

    if manifest_path.exists() and labels_path.exists() and sessions_dir.exists():
        num_sessions = len(list(sessions_dir.glob("session_*")))
        print(f"\n✓ {dataset['name']} completed successfully")
        print(f"    Output: {output_dir}")
        print(f"    Sessions: {num_sessions}")
        return True
    else:
        print(f"\n✗ {dataset['name']} validation failed")
        print(f"    Missing files in {output_dir}")
        return False


def main():
    """Main pipeline execution."""
    print("=" * 80)
    print("Dataset Setup Pipeline")
    print("=" * 80)

    # Determine which datasets to process
    if len(sys.argv) > 1:
        datasets_to_process = [sys.argv[1]]
    else:
        datasets_to_process = list(DATASETS.keys())

    print(f"\nWill process {len(datasets_to_process)} dataset(s):")
    for key in datasets_to_process:
        if key in DATASETS:
            print(f"  - {DATASETS[key]['name']}")
        else:
            print(f"  - {key} (UNKNOWN)")

    # Process each dataset
    results = {}
    for dataset_key in datasets_to_process:
        success = process_dataset(dataset_key)
        results[dataset_key] = success

    # Summary
    print(f"\n{'=' * 80}")
    print("Pipeline Summary")
    print(f"{'=' * 80}")

    for dataset_key, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        dataset_name = DATASETS.get(dataset_key, {}).get('name', dataset_key)
        print(f"  {dataset_name}: {status}")

    total_success = sum(results.values())
    print(f"\nCompleted: {total_success}/{len(results)} datasets")

    # Overall success
    if total_success == len(results):
        print("\n✓ All datasets processed successfully!")
        return 0
    else:
        print(f"\n⚠ {len(results) - total_success} dataset(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
