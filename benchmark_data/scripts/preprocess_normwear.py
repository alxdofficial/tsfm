#!/usr/bin/env python3
"""Preprocess native data into NormWear's input format: 65 Hz, 6 s windows, REAL channels only.

NormWear's published pipeline is ~65 Hz / 6 s with per-channel detrend + amplitude normalization
(done in the adapter, evaluate_normwear.window_embeddings). It is channel-INDEPENDENT, so we feed
only the channels a dataset actually has (acc, plus gyro where present) — NEVER zero-padded phantom
channels, which would enter its cross-channel pool as fake observations. Rate matters: its Ricker
CWT scales are 65 Hz-tuned, so we resample each stream to 65 Hz (up OR down, via polyphase resample)
rather than feeding the 20 Hz LiMU-BERT grid.

Output per dataset: benchmark_data/processed/normwear/<ds>/data_65_390.npy
  shape (N, 390, C) — 6 s windows at 65 Hz, C = number of REAL channels (3 acc, or 6 acc+gyro).

N and window order are IDENTICAL to benchmark_data/processed/limubert/<ds>/label_20_120.npy so the
NormWear adapter's per-window outputs align 1:1 with base.load_gt / keep_idx. Labels are NOT
re-derived here — the adapter reuses the canonical GT.

Alignment: reads the SAME per-subject raw CSVs (benchmark_data/raw/<ds>/subject_*.csv) that
preprocess_limubert.py consumed, in the SAME sorted order + NaN policy, and keys each subject to its
exact 20 Hz limubert window count (no floor() recompute — a 1-sample drift would desync). Run
export_raw.py first for any dataset whose raw CSVs are absent.

Usage:
    python benchmark_data/scripts/preprocess_normwear.py [--datasets harth ...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "benchmark_data"
RAW_DIR = BENCHMARK_DIR / "raw"
LIMU_DIR = BENCHMARK_DIR / "processed" / "limubert"
OUT_DIR = BENCHMARK_DIR / "processed" / "normwear"

TARGET_HZ = 65
WINDOW_65 = TARGET_HZ * 6   # 390 samples = 6 s @ 65 Hz (matches the limubert 6 s / 120-sample grid)

with open(BENCHMARK_DIR / "dataset_config.json") as f:
    CONFIG = json.load(f)
ALL_DATASETS = CONFIG["train_datasets"] + CONFIG["zero_shot_datasets"]

ACC_COLS = ["acc_x", "acc_y", "acc_z"]
GYRO_COLS = ["gyro_x", "gyro_y", "gyro_z"]


def _resample_to(data: np.ndarray, orig_hz: float, target_hz: int) -> np.ndarray:
    """Resample (T, C) from orig_hz to target_hz (up OR down) via anti-aliased polyphase filtering.
    Reduce up/down by their gcd so the filter is cheap; identity when already at target."""
    if int(round(orig_hz)) == target_hz:
        return data.astype(np.float64)
    up, down = target_hz, int(round(orig_hz))
    g = np.gcd(up, down)
    return resample_poly(data.astype(np.float64), up // g, down // g, axis=0)


def process_dataset(ds: str):
    raw_dir = RAW_DIR / ds
    csvs = sorted(raw_dir.glob("subject_*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"{ds}: no benchmark_data/raw/{ds}/subject_*.csv found. "
            f"Run: python benchmark_data/scripts/export_raw.py --datasets {ds}")
    with open(raw_dir / "metadata.json") as f:
        orig_hz = json.load(f)["sampling_rate_hz"]

    # Authoritative per-subject window counts from the EXISTING 20 Hz limubert grid (exact; no
    # floor() recompute). Mirrors preprocess_ssl_wearables' subject keying.
    lab = np.load(str(LIMU_DIR / ds / "label_20_120.npy"))
    with open(LIMU_DIR / ds / "mapping.json") as f:
        subject_to_idx = json.load(f)["subject_to_idx"]
    idx_to_count = {int(k): int(v) for k, v in
                    zip(*np.unique(lab[:, 0, 1], return_counts=True))}

    def _subject_count(csv_path) -> int:
        name = csv_path.stem.replace("subject_", "")
        idx = subject_to_idx.get(name)
        if idx is None and name.lstrip("-").isdigit():
            idx = subject_to_idx.get(str(int(name)))
        return 0 if idx is None else idx_to_count.get(int(idx), 0)

    # Real channels only: acc always; gyro only if the export actually carries it (acc-only
    # datasets have no gyro columns -> C=3, never zero-padded). acc_* is total_acc for uci_har
    # (core_channels remap); NormWear detrends the DC away so gravity presence is immaterial.
    sample_cols = set(pd.read_csv(csvs[0], nrows=1).columns)
    cols = list(ACC_COLS) + ([*GYRO_COLS] if all(c in sample_cols for c in GYRO_COLS) else [])
    n_ch = len(cols)

    all_win = []
    for csv in csvs:
        n_sub = _subject_count(csv)
        if n_sub == 0:
            continue
        df = pd.read_csv(csv)
        x = df[cols].values.astype(np.float64)
        valid = ~np.isnan(x[:, 0])                       # same NaN policy as preprocess_limubert
        x = np.nan_to_num(x[valid], nan=0.0)

        r = _resample_to(x, orig_hz, TARGET_HZ)          # (T65, C)
        # Window to EXACTLY n_sub 6 s windows aligned to the 20 Hz grid (same stream; count matches
        # modulo resample rounding). Edge-pad only a short tail if r is a few samples short; never
        # duplicate/truncate whole windows (see #83).
        need = n_sub * WINDOW_65
        if len(r) < need:
            r = np.pad(r, ((0, need - len(r)), (0, 0)), mode="edge")
        w = r[:need].reshape(n_sub, WINDOW_65, n_ch)
        all_win.append(w.astype(np.float32))

    data = (np.concatenate(all_win, axis=0) if all_win
            else np.empty((0, WINDOW_65, n_ch), np.float32))

    # Hard alignment check against the canonical 20 Hz label grid.
    assert data.shape[0] == lab.shape[0], (
        f"{ds}: produced {data.shape[0]} windows but label_20_120 has {lab.shape[0]}. "
        "Alignment broken -- do not use.")

    out_dir = OUT_DIR / ds
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_dir / f"data_{TARGET_HZ}_{WINDOW_65}.npy"), data)
    print(f"  {ds}: {data.shape[0]} windows, {n_ch}ch ({'acc+gyro' if n_ch == 6 else 'acc'}) "
          f"@ {TARGET_HZ}Hz -> {out_dir.name}/data_{TARGET_HZ}_{WINDOW_65}.npy")
    return data.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"NormWear preprocess: {TARGET_HZ}Hz, {WINDOW_65}-sample (6s) windows, REAL channels only")
    for ds in args.datasets:
        if ds not in ALL_DATASETS:
            print(f"unknown dataset {ds}"); sys.exit(1)
        process_dataset(ds)


if __name__ == "__main__":
    main()
