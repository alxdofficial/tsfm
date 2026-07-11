#!/usr/bin/env python3
"""Preprocess native data into ssl-wearables (OxWearables harnet) input format.

harnet input contract (verified against the released hubconf.py + data_parsing/*.py):
  * 30 Hz sampling rate
  * 3-channel ACCELEROMETER ONLY (x, y, z)
  * g-units WITH gravity present (a still window has |acc| ~= 1 g, NOT ~9.8)
  * NO per-window standardization (sslearning NormalDataset feeds the raw g signal)

Output per dataset:  benchmark_data/processed/ssl_wearables/<ds>/data_30_180.npy
  shape (N, 180, 3)  -- 6-second windows at 30 Hz, acc in g with gravity.

N and window order are IDENTICAL to benchmark_data/processed/limubert/<ds>/label_20_120.npy
so the ssl-wearables adapter's per-window outputs align 1:1 with base.load_gt / keep_idx.
Labels are NOT re-derived here -- the adapter/head-fit reuse the limubert label files.

Alignment guarantee: this reads the SAME per-subject raw CSVs
(benchmark_data/raw/<ds>/subject_*.csv) that preprocess_limubert.py consumed, in the
SAME sorted order, applies the SAME NaN-drop, and reconciles the per-subject 30 Hz window
count to the authoritative 20 Hz count (floor(len_20/120)). Run export_raw.py first for any
dataset whose raw CSVs are absent.

Usage:
    python benchmark_data/scripts/preprocess_ssl_wearables.py [--datasets harth ...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "benchmark_data"
RAW_DIR = BENCHMARK_DIR / "raw"
LIMU_DIR = BENCHMARK_DIR / "processed" / "limubert"
OUT_DIR = BENCHMARK_DIR / "processed" / "ssl_wearables"

TARGET_HZ = 30
WINDOW_30 = 180          # 6 s @ 30 Hz  (matches the limubert 6 s / 120-sample @ 20 Hz grid)
LIMU_HZ = 20
LIMU_WINDOW = 120
GRAVITY_MS2 = 9.80665    # ssl data_parsing uses /9.81; kept consistent with limubert here
CLIP_G = 3.0             # ssl data_parsing (oppo.py/pamap.py) clips to +/-3 g; benign for HAR

with open(BENCHMARK_DIR / "dataset_config.json") as f:
    CONFIG = json.load(f)
ALL_DATASETS = CONFIG["train_datasets"] + CONFIG["zero_shot_datasets"]

# --- per-dataset unit / gravity handling to reach "g WITH gravity" (verified vs native) ---
# iOS CoreMotion: userAcceleration is g with GRAVITY REMOVED; add the separate unit-gravity
# vector back -> total specific force in g.
IOS_USERACC_PLUS_GRAVITY = {"motionsense", "inclusivehar"}
# uci_har's export stores BOTH body_acc (acc_*, gravity removed) AND total_acc (gravity
# present, ALREADY in g ~1.0). harnet needs gravity, so read total_acc_* (see acc_cols
# selection below) and treat as already-g. (hapt's export carries NO total_acc columns;
# its acc_* is already g-with-gravity at median ~1.02, so it is ACC_G_ASIS below.)
# uci_har's export now REMAPS acc_* -> total_acc_* (gravity present, in g) via core_channels (#85),
# so there is no separate total_acc_* column anymore — read acc_* directly and treat as already-g
# (see ACC_G_ASIS). USE_TOTAL_ACC_COL is now empty (kept for clarity / future datasets).
USE_TOTAL_ACC_COL = set()
# Already g WITH gravity: Axivity raw (harth/capture24), hapt acc_* (median ~1.02 g),
# and uci_har total_acc.
ACC_G_ASIS = {"harth", "capture24", "hapt", "uci_har"}   # uci_har acc_* is now total_acc (g)
# milli-g with gravity -> g.
ACC_MILLI_G = {"opportunity"}
# everything else: m/s^2 with gravity -> divide by g.

# Datasets whose accelerometer CANNOT be expressed as physical g-with-gravity (harnet's input
# contract) and has no channel/orientation to reconstruct it. harnet was pretrained on
# gravity-PRESENT Axivity signal, so these are structurally incompatible with ssl-wearables and
# are skipped loudly (never silently), each with its verified reason:
INCOMPATIBLE_ACCEL = {
    "kuhar": "gravity-removed linear accel (static |acc|~0.05); no gravity channel to reconstruct",
    "recgym": "min-max normalized to [0,1] per axis (non-physical: all axes ~0.5, |acc|~0.866 const); "
              "no recoverable gravity magnitude or direction",
}


def _downsample_bin_mean(data: np.ndarray, original_hz: float, target_hz: int) -> np.ndarray:
    """Bin-and-mean downsampler, matching preprocess_limubert.downsample_by_averaging
    (integer + variable-width-bin branches), but parametrized by target_hz."""
    if original_hz == target_hz:
        return data
    ratio = original_hz / target_hz
    n = data.shape[0]
    if ratio < 1:
        n_target = int(round(n / ratio))
        idx = np.linspace(0, n - 1, n_target).astype(int)
        return data[idx]
    result = []
    window = int(ratio)
    if ratio == int(ratio):
        for i in range(0, n - window + 1, window):
            result.append(np.mean(data[i:i + window], axis=0))
    else:
        remainder = 0.0
        i = 0
        while i + window < n:
            remainder += ratio - window
            if remainder >= 1:
                remainder -= 1
                result.append(np.mean(data[i:i + window + 1], axis=0))
                i += window + 1
            else:
                result.append(np.mean(data[i:i + window], axis=0))
                i += window
    if not result:
        return np.empty((0, data.shape[1]))
    return np.array(result)


def _to_g_with_gravity(ds: str, cols3: np.ndarray, grav3) -> np.ndarray:
    if ds in IOS_USERACC_PLUS_GRAVITY:
        assert grav3 is not None, f"{ds}: needs gravity_x/y/z columns"
        out = cols3 + grav3
    elif ds in ACC_G_ASIS:
        out = cols3
    elif ds in ACC_MILLI_G:
        out = cols3 / 1000.0
    else:
        out = cols3 / GRAVITY_MS2
    if CLIP_G is not None:
        out = np.clip(out, -CLIP_G, CLIP_G)
    return out


def process_dataset(ds: str):
    if ds in INCOMPATIBLE_ACCEL:
        raise ValueError(
            f"{ds}: {INCOMPATIBLE_ACCEL[ds]} — incompatible with harnet's gravity-present input "
            f"contract. Exclude it from the ssl-wearables corpus.")
    raw_dir = RAW_DIR / ds
    csvs = sorted(raw_dir.glob("subject_*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"{ds}: no benchmark_data/raw/{ds}/subject_*.csv found. "
            f"Run: python benchmark_data/scripts/export_raw.py --datasets {ds}")
    with open(raw_dir / "metadata.json") as f:
        orig_hz = json.load(f)["sampling_rate_hz"]

    # Authoritative per-subject window counts come from the EXISTING limubert label grid,
    # NOT from recomputing floor(len_20/120) off freshly-exported CSVs: a 1-sample drift in
    # the 20 Hz downsample flips a floor() and desyncs the count by one window (hit on hhar).
    # preprocess_limubert.py assigns subject_idx via mapping.json's subject_to_idx (sorted by
    # str(subject), with an int fallback) and iterates the SAME sorted glob we do here, so we
    # key each subject CSV to its exact limubert window count and emit blocks in that order.
    lab = np.load(str(LIMU_DIR / ds / "label_20_120.npy"))
    with open(LIMU_DIR / ds / "mapping.json") as f:
        subject_to_idx = json.load(f)["subject_to_idx"]
    idx_to_count = {int(k): int(v) for k, v in
                    zip(*np.unique(lab[:, 0, 1], return_counts=True))}

    def _subject_count(csv_path) -> int:
        name = csv_path.stem.replace("subject_", "")
        idx = subject_to_idx.get(name)
        if idx is None and name.lstrip("-").isdigit():  # int-vs-str key fallback (mirrors limubert)
            idx = subject_to_idx.get(str(int(name)))
        return 0 if idx is None else idx_to_count.get(int(idx), 0)

    need_grav = ds in IOS_USERACC_PLUS_GRAVITY
    # harnet needs gravity-present accel. UCI-HAR family's acc_* is body (gravity-removed);
    # its total_acc_* carries gravity, so read that instead.
    acc_cols = (["total_acc_x", "total_acc_y", "total_acc_z"]
                if ds in USE_TOTAL_ACC_COL else ["acc_x", "acc_y", "acc_z"])
    all_win = []
    for csv in csvs:
        df = pd.read_csv(csv)
        acc = df[acc_cols].values.astype(np.float64)
        grav = df[["gravity_x", "gravity_y", "gravity_z"]].values.astype(np.float64) if need_grav else None

        # Same NaN policy as preprocess_limubert: drop rows where acc_x is NaN, zero-fill the rest.
        valid = ~np.isnan(acc[:, 0])
        acc = acc[valid]
        if grav is not None:
            grav = grav[valid]
        stack = acc if grav is None else np.concatenate([acc, grav], axis=1)
        stack = np.nan_to_num(stack, nan=0.0)

        # Authoritative window count from the existing 20 Hz label grid (exact, no recompute).
        n_sub = _subject_count(csv)
        if n_sub == 0:
            continue

        d30 = _downsample_bin_mean(stack, orig_hz, TARGET_HZ)
        # Window to EXACTLY n_sub 6s windows aligned to the 20 Hz grid. The 20 Hz and 30 Hz windows
        # come from the SAME per-subject stream, so the count matches modulo resample rounding: take
        # n_sub*WINDOW_30 samples; if d30 is a few samples short (rounding at the stream tail),
        # edge-pad ONLY the missing tail to complete the last window. Never duplicate or truncate
        # WHOLE windows — the old repeat/truncate injected fake windows / dropped real ones (#83).
        need = n_sub * WINDOW_30
        if len(d30) < need:
            d30 = np.pad(d30, ((0, need - len(d30)), (0, 0)), mode="edge")
        w = d30[:need].reshape(n_sub, WINDOW_30, stack.shape[1]) if n_sub > 0 \
            else np.empty((0, WINDOW_30, stack.shape[1]))

        # channels -> 3-ch g-with-gravity
        acc_w = w[:, :, :3]
        grav_w = w[:, :, 3:6] if need_grav else None
        g_w = _to_g_with_gravity(ds, acc_w, grav_w)
        all_win.append(g_w.astype(np.float32))

    data = np.concatenate(all_win, axis=0) if all_win else np.empty((0, WINDOW_30, 3), np.float32)

    # Hard alignment check against the limubert label grid (lab loaded above).
    assert data.shape[0] == lab.shape[0], (
        f"{ds}: produced {data.shape[0]} windows but label_20_120 has {lab.shape[0]}. "
        "Alignment broken -- do not use.")

    out_ds = OUT_DIR / ds
    out_ds.mkdir(parents=True, exist_ok=True)
    np.save(str(out_ds / "data_30_180.npy"), data)
    med = float(np.median(np.linalg.norm(data.reshape(-1, 3), axis=1)))
    print(f"  {ds}: {data.shape} | median|acc|={med:.3f} g  (expect ~1, not ~9.8)")
    return data.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    skipped = []
    for ds in args.datasets:
        if ds not in ALL_DATASETS:
            print(f"unknown dataset {ds}"); sys.exit(1)
        if ds in INCOMPATIBLE_ACCEL:
            print(f"SKIP {ds}: {INCOMPATIBLE_ACCEL[ds]}")
            skipped.append(ds)
            continue
        print(f"Processing {ds} -> {TARGET_HZ}Hz/3ch/g-with-gravity, window={WINDOW_30}")
        process_dataset(ds)
    if skipped:
        print(f"\nSkipped (non-physical/gravity-removed accel, incompatible with ssl-wearables): {skipped}")


if __name__ == "__main__":
    main()
