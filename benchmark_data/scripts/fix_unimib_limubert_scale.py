#!/usr/bin/env python3
"""One-off, idempotent correction: rescale the LiMU-BERT unimib_shar accel grid g -> m/s².

WHY THIS EXISTS (and is not just a preprocess_limubert.py re-run):
  unimib_shar's raw CSVs are permanently LOST (benchmark_data/raw/unimib_shar is empty;
  the source subject-map acc_labels.npy is gone), so preprocess_limubert.py can no longer
  regenerate this grid — it silently skips the dataset. The on-disk grid
  benchmark_data/processed/limubert/unimib_shar/data_20_120.npy predates the #34 unit fix:
  its accel is stored in g (median|acc|≈1.38) while EVERY other gravity-present LiMU-BERT
  train grid is in m/s² (median≈9.8), because unimib_shar is in preprocess_limubert's
  ACC_IN_G_UNITS set (native g -> ×9.80665 -> m/s²) but that multiply never ran on this
  stale artifact.

WHY IT MATTERS:
  LiMU-BERT's fixed normalization (normalize_for_limubert: acc /= 9.8, NO per-window
  instance-norm) means a global scale error does NOT cancel — mis-scaled unimib windows
  enter the frozen encoder ~10× too small, producing degenerate embeddings that corrupt
  the shared ConSE head. (CrossHAR is immune: it InstanceNorm1d's every window, which
  divides the scale out. ssl-wearables excludes unimib_shar. HALO reads unimib from the
  parquet corpus via accel_units at scale 1.0, not this grid. So ONLY LiMU-BERT is affected.)

WHAT IT DOES:
  Multiplies the 3 accel channels (cols 0:3) of data_20_120.npy by GRAVITY_MS2, reproducing
  exactly what preprocess_limubert.py would have emitted (sensor_data[:, :3] *= GRAVITY_MS2).
  Verified the ONLY staleness is scale: labels (0..16) match the 17 current activities, window
  structure is intact, gyro is legitimately zero (accel-only dataset). Idempotent: refuses to
  run if the grid is already in m/s² (median|acc| already ~9.8).

Usage:
    python benchmark_data/scripts/fix_unimib_limubert_scale.py
"""

import sys
from pathlib import Path

import numpy as np

GRAVITY_MS2 = 9.80665
ROOT = Path(__file__).resolve().parent.parent.parent
GRID = ROOT / "benchmark_data" / "processed" / "limubert" / "unimib_shar" / "data_20_120.npy"


def main():
    if not GRID.exists():
        sys.exit(f"grid not found: {GRID}")
    d = np.load(str(GRID))  # (N, 120, 6)
    acc = d[:, :, :3].reshape(-1, 3)
    med = float(np.median(np.linalg.norm(acc, axis=1)))
    print(f"unimib_shar limubert grid: shape={d.shape}  accel median|.|={med:.3f}")

    if med > 5.0:
        print(f"  already in m/s² (median {med:.3f} > 5) — nothing to do (idempotent no-op).")
        return
    if not (0.5 <= med <= 3.0):
        sys.exit(f"  REFUSING: median {med:.3f} is neither g (~1.4) nor m/s² (~13.6); "
                 "grid is not what this fix expects — inspect manually.")

    d[:, :, :3] *= GRAVITY_MS2
    np.save(str(GRID), d)
    new_med = float(np.median(np.linalg.norm(d[:, :, :3].reshape(-1, 3), axis=1)))
    print(f"  rescaled accel ×{GRAVITY_MS2} -> new accel median|.|={new_med:.3f} m/s² (matches ~9.8 corpus). "
          f"gyro untouched.")


if __name__ == "__main__":
    main()
