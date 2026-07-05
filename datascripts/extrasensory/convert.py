"""
Convert ExtraSensory (phone accelerometer) to standardized format.

Input:
  data/raw/extrasensory/raw_acc/<UUID>/<timestamp>.m_raw_acc.dat   (raw phone accel)
  data/raw/extrasensory/labels/<UUID>.features_labels.csv.gz       (per-example labels)
Output: data/extrasensory/  (manifest.json, labels.json, sessions/*/data.parquet)

ExtraSensory (Vaizman, Ellis & Lanckriet 2017, IEEE Pervasive Computing):
- 60 users, IN-THE-WILD free-living data collected with a personal smartphone
  (worn/carried however the user chose — pocket/hand/bag/table) plus a Pebble watch.
- Each ~1-minute example has a ~20 s raw accelerometer recording (phone ~40 Hz)
  and a MULTI-LABEL context annotation (posture + activity + phone-location +
  environment simultaneously).
- We project the multi-label context to a SINGLE mutually-exclusive movement
  primitive (lying/sitting/standing/walking/running/cycling/stairs). Examples with
  zero or more-than-one active primitive are skipped (ambiguous). Context labels
  (phone location, environment, social) are discarded.

This is the free-living phone TEST set that measures the actual deployment condition
(the other phone test sets — motionsense, inclusivehar — are lab-scripted).

Phone raw accelerometer is in g and INCLUDES gravity (iOS raw acceleration). The
.dat files hold 3 columns (x, y, z) with no timestamp; we synthesize a uniform
40 Hz clock. License: publicly available, citation required (Vaizman2017a).
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.windowing import create_variable_windows

RAW_DIR = Path("data/raw/extrasensory")
ACC_DIR = RAW_DIR / "raw_acc"
LABELS_DIR = RAW_DIR / "labels"
OUTPUT_DIR = Path("data/extrasensory")
PHONE_RATE = 40.0  # iOS raw accelerometer nominal rate in ExtraSensory

# Multi-label -> single movement primitive. These are mutually exclusive; an example
# is kept only if EXACTLY ONE is active (== 1).
PRIMARY_LABELS = {
    "label:LYING_DOWN": "lying",
    "label:SITTING": "sitting",
    "label:OR_standing": "standing",
    "label:FIX_walking": "walking",
    "label:FIX_running": "running",
    "label:BICYCLING": "cycling",
    "label:STAIRS_-_GOING_UP": "stairs_up",
    "label:STAIRS_-_GOING_DOWN": "stairs_down",
}
MIN_SAMPLES = int(PHONE_RATE * 3)  # need >= 3 s of signal


def load_example_labels(csv_gz: Path):
    """Return {timestamp:int -> activity_str} for examples with exactly one primitive."""
    with gzip.open(csv_gz, "rt") as f:
        df = pd.read_csv(f)
    label_cols = [c for c in PRIMARY_LABELS if c in df.columns]
    out = {}
    for _, row in df.iterrows():
        active = [PRIMARY_LABELS[c] for c in label_cols if row[c] == 1]
        if len(active) == 1:
            out[int(row["timestamp"])] = active[0]
    return out


def read_dat(path: Path) -> np.ndarray:
    """Read a raw_acc .dat -> (N, 3) x/y/z. Auto-handles an optional leading time col."""
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] >= 4:          # (t, x, y, z) — drop the time column
        arr = arr[:, 1:4]
    return arr[:, :3].astype(np.float32)


def convert_dataset() -> bool:
    print("=" * 80)
    print("ExtraSensory (phone) -> Standardized Format Converter")
    print("=" * 80)
    print("NOTE: free-living phone TEST set; multi-label -> single-movement projection")

    if not ACC_DIR.exists() or not LABELS_DIR.exists():
        print(f"ERROR: expected {ACC_DIR} and {LABELS_DIR}.")
        print("Download + extract from http://extrasensory.ucsd.edu :")
        print("  raw_measurements/ExtraSensory.raw_measurements.raw_acc.zip -> data/raw/extrasensory/raw_acc/")
        print("  primary_data_files/ExtraSensory.per_uuid_features_labels.zip -> data/raw/extrasensory/labels/")
        return False

    sessions_dir = OUTPUT_DIR / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(LABELS_DIR.glob("*.features_labels.csv.gz"))
    print(f"Found {len(label_files)} users")

    all_labels = {}
    session_count = 0
    counts = {}

    for ui, lf in enumerate(label_files):
        uuid = lf.name.split(".")[0]
        user_acc = ACC_DIR / uuid
        if not user_acc.exists():
            continue
        ts_to_act = load_example_labels(lf)
        subj_windows = 0
        for idx, (ts, activity) in enumerate(sorted(ts_to_act.items())):
            dat = user_acc / f"{ts}.m_raw_acc.dat"
            if not dat.exists():
                continue
            try:
                xyz = read_dat(dat)
            except Exception:
                continue
            if len(xyz) < MIN_SAMPLES or not np.isfinite(xyz).all():
                continue
            seg = pd.DataFrame(xyz, columns=["acc_x", "acc_y", "acc_z"])
            seg.insert(0, "timestamp_sec", np.arange(len(seg)) / PHONE_RATE)

            # subject is ALWAYS field 1 of the session id; UUID has no underscores.
            prefix = f"extrasensory_{uuid}_{activity}_{idx:04d}"
            windows = create_variable_windows(
                df=seg, session_prefix=prefix, activity=activity,
                sample_rate=PHONE_RATE, seed=1000 + ui * 1000 + idx,
            )
            for window_id, window_df, window_activity in windows:
                window_df = window_df.copy()
                window_df["timestamp_sec"] = (
                    window_df["timestamp_sec"] - window_df["timestamp_sec"].iloc[0]
                )
                wp = sessions_dir / window_id
                wp.mkdir(exist_ok=True)
                window_df.to_parquet(wp / "data.parquet", index=False)
                all_labels[window_id] = [window_activity]
                counts[window_activity] = counts.get(window_activity, 0) + 1
                session_count += 1
                subj_windows += 1
        if (ui + 1) % 10 == 0 or ui == len(label_files) - 1:
            print(f"  [{ui + 1:2d}/{len(label_files)}] {uuid[:8]}: {session_count} windows so far")

    if not all_labels:
        print("\nNo sessions created — check raw layout.")
        return False

    with open(OUTPUT_DIR / "labels.json", "w") as f:
        json.dump(all_labels, f)
    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(create_manifest(len(label_files)), f, indent=2)

    print(f"\n{'=' * 80}\nConversion complete!\n{'=' * 80}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"  - {session_count} sessions across {len(label_files)} users")
    print("\nActivity distribution:")
    for a, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {a:14s} {c}")
    return True


def create_manifest(num_subjects: int) -> dict:
    return {
        "dataset_name": "ExtraSensory",
        "description": (
            "In-the-wild free-living human activity recognition from a personal "
            "smartphone. 60 users, phone worn/carried in a naturally-varying location "
            "(pocket/hand/bag/table). Raw triaxial accelerometer in g (includes "
            "gravity), ~40 Hz. Multi-label context projected to a single movement "
            "primitive. The free-living phone deployment condition."
        ),
        "source": "http://extrasensory.ucsd.edu/",
        "num_subjects": num_subjects,
        "channels": [
            {"name": "acc_x", "description": "Phone accelerometer X-axis in g (raw, includes gravity; in-the-wild placement)", "sampling_rate_hz": PHONE_RATE},
            {"name": "acc_y", "description": "Phone accelerometer Y-axis in g (raw, includes gravity; in-the-wild placement)", "sampling_rate_hz": PHONE_RATE},
            {"name": "acc_z", "description": "Phone accelerometer Z-axis in g (raw, includes gravity; in-the-wild placement)", "sampling_rate_hz": PHONE_RATE},
        ],
    }


def main() -> int:
    return 0 if convert_dataset() else 1


if __name__ == "__main__":
    sys.exit(main())
