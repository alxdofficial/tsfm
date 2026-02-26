#!/usr/bin/env python3
"""
Generate UMAP visualization of distribution gap between training and
zero-shot test HAR datasets (Fig. X in paper).

Methodology:
  1. Sample up to 400 sessions per dataset from all 17 HAR datasets
  2. Extract all available channels using core_channels mapping and
     compute RAW statistical features (no normalization — the point is
     to show the distributional gap as it exists in the data, which
     motivates the need for our normalization + semantic alignment)
  3. Resample each session to 128 timesteps via linear interpolation
  4. Extract 52 features per session:
     - Per-channel (acc_x/y/z + gyro_x/y/z, 6 channels × 7 stats):
       mean, std, min, max, skewness, kurtosis, zero-crossing rate
     - Acceleration magnitude: mean, std, min, max
     - Cross-channel correlations: acc_x-acc_y, acc_x-acc_z, acc_y-acc_z
     - Signal energy: RMS per channel (already captured via std+mean)
     - Spectral centroid per acc channel (3)
  5. Standardize features (zero-mean unit-variance across all sessions)
  6. Remove 4-sigma outliers
  7. Project to 2D via UMAP
  8. Plot per-dataset 2-sigma Gaussian confidence ellipses + centroids
     (NOT individual points) for a clean, readable figure

The figure shows that training and test datasets occupy distinct regions
of feature space, motivating the zero-shot challenge.

Usage:
    python figures/fig_dataset_discrepancy.py

Output:
    figures/fig_dataset_discrepancy.pdf  (300 DPI, vector)
    figures/fig_dataset_discrepancy.png  (200 DPI, raster)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy.stats import skew, kurtosis

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Dataset groups
# ---------------------------------------------------------------------------
TRAIN_DATASETS = [
    "uci_har", "hhar", "pamap2", "wisdm", "dsads",
    "kuhar", "unimib_shar", "hapt", "mhealth", "recgym",
]
ZEROSHOT_TEST = [
    "motionsense", "realworld", "mobiact", "harth", "vtt_coniot",
]
SEVERE_OOD = [
    "shoaib", "opportunity",
]
ALL_DATASETS = TRAIN_DATASETS + ZEROSHOT_TEST + SEVERE_OOD

DISPLAY_NAMES = {
    "uci_har": "UCI HAR", "hhar": "HHAR", "pamap2": "PAMAP2",
    "wisdm": "WISDM", "dsads": "DSADS", "kuhar": "KU-HAR",
    "unimib_shar": "UniMiB", "hapt": "HAPT", "mhealth": "MHEALTH",
    "recgym": "RecGym", "motionsense": "MotionSense",
    "realworld": "RealWorld", "mobiact": "MobiAct",
    "harth": "HARTH", "vtt_coniot": "VTT-ConIoT",
    "shoaib": "Shoaib", "opportunity": "OPPORT.",
}

CORE_CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
N_FEATURES = 52  # 6×7 + 4 + 3 + 3
FIXED_LEN = 128


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_core_channel_mapping():
    with open(PROJECT_ROOT / "benchmark_data" / "dataset_config.json") as f:
        cfg = json.load(f)
    return {name: info["core_channels"] for name, info in cfg["datasets"].items()}


def _resample(col, target_len):
    if len(col) == target_len:
        return col
    return np.interp(np.linspace(0, 1, target_len),
                     np.linspace(0, 1, len(col)), col)


def extract_features(df, core_map):
    """Extract 52 RAW statistical + spectral features from one session.

    No z-score normalization — we want to capture the actual distributional
    differences (sensor scale, calibration, placement) between datasets.
    """
    channels = {}
    for core_name in CORE_CHANNELS:
        if core_name in core_map and core_map[core_name] in df.columns:
            raw = df[core_map[core_name]].values.astype(np.float64)
        else:
            raw = np.zeros(len(df), dtype=np.float64)
        raw = np.nan_to_num(raw, nan=0.0)
        if len(raw) < 4:
            raw = np.zeros(FIXED_LEN, dtype=np.float64)
        else:
            raw = _resample(raw, FIXED_LEN)
        channels[core_name] = raw

    features = []

    # Per-channel stats (6 channels × 7 stats = 42)
    for name in CORE_CHANNELS:
        col = channels[name]
        s = np.std(col)
        features.extend([
            np.mean(col),
            s,
            np.min(col),
            np.max(col),
            float(skew(col)) if s > 1e-10 else 0.0,
            float(kurtosis(col)) if s > 1e-10 else 0.0,
            np.sum(np.diff(np.sign(col)) != 0) / max(len(col) - 1, 1),
        ])

    # Acceleration magnitude stats (4)
    acc_mag = np.sqrt(sum(channels[f"acc_{a}"] ** 2 for a in "xyz"))
    features.extend([np.mean(acc_mag), np.std(acc_mag),
                     np.min(acc_mag), np.max(acc_mag)])

    # Cross-channel correlations (3)
    for a, b in [("acc_x", "acc_y"), ("acc_x", "acc_z"), ("acc_y", "acc_z")]:
        if np.std(channels[a]) > 1e-10 and np.std(channels[b]) > 1e-10:
            r = np.corrcoef(channels[a], channels[b])[0, 1]
            features.append(r if np.isfinite(r) else 0.0)
        else:
            features.append(0.0)

    # Spectral centroid per acc channel (3)
    for a in ["acc_x", "acc_y", "acc_z"]:
        fft_mag = np.abs(np.fft.rfft(channels[a]))[1:]
        freqs = np.fft.rfftfreq(FIXED_LEN)[1:]
        if fft_mag.sum() > 1e-10:
            features.append(np.sum(freqs * fft_mag) / np.sum(fft_mag))
        else:
            features.append(0.0)

    return np.array(features, dtype=np.float32)


def load_dataset_features(dataset_name, data_dir, core_map,
                          max_sessions=400, seed=42):
    sessions_dir = data_dir / dataset_name / "sessions"
    if not sessions_dir.exists():
        return np.empty((0, N_FEATURES), dtype=np.float32)

    all_sessions = sorted(os.listdir(sessions_dir))
    rng = np.random.RandomState(seed)
    if len(all_sessions) > max_sessions:
        idx = rng.choice(len(all_sessions), max_sessions, replace=False)
        selected = [all_sessions[i] for i in sorted(idx)]
    else:
        selected = all_sessions

    feats = []
    for sess in selected:
        path = sessions_dir / sess / "data.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
            f = extract_features(df, core_map)
            if np.all(np.isfinite(f)):
                feats.append(f)
        except Exception:
            continue
    return np.stack(feats) if feats else np.empty((0, N_FEATURES), dtype=np.float32)


def draw_confidence_ellipse(ax, mean, cov, n_std=1.5, **kwargs):
    """Draw a 2D Gaussian confidence ellipse."""
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-6)  # numerical safety
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    width, height = 2 * n_std * np.sqrt(vals)

    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)
    return ellipse


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    data_dir = PROJECT_ROOT / "data"
    output_path = PROJECT_ROOT / "figures" / "fig_dataset_discrepancy.pdf"
    core_maps = load_core_channel_mapping()

    # --- Load ---
    print("Loading sessions...")
    all_features, all_ds, all_groups = [], [], []
    for ds in ALL_DATASETS:
        if ds not in core_maps:
            continue
        print(f"  {DISPLAY_NAMES[ds]:12s}", end="", flush=True)
        feats = load_dataset_features(ds, data_dir, core_maps[ds])
        print(f" {len(feats):>4d} sessions")
        if len(feats) == 0:
            continue
        all_features.append(feats)
        all_ds.extend([ds] * len(feats))
        grp = ("Training" if ds in TRAIN_DATASETS
               else "Zero-Shot Test" if ds in ZEROSHOT_TEST
               else "Severe OOD")
        all_groups.extend([grp] * len(feats))

    X = np.concatenate(all_features)
    ds_arr = np.array(all_ds)
    grp_arr = np.array(all_groups)
    print(f"\nTotal: {len(X)} samples")

    # Standardize + outlier removal
    mu, sigma = X.mean(0), X.std(0)
    sigma[sigma < 1e-10] = 1.0
    X_norm = (X - mu) / sigma
    keep = np.max(np.abs(X_norm), axis=1) < 4.0
    n_rm = np.sum(~keep)
    if n_rm:
        print(f"  Removed {n_rm} outliers ({n_rm / len(X) * 100:.1f}%)")
    X_norm, ds_arr, grp_arr = X_norm[keep], ds_arr[keep], grp_arr[keep]

    # UMAP
    print("Fitting UMAP...")
    import umap
    emb = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                     metric="euclidean", random_state=42).fit_transform(X_norm)

    # --- Compute per-dataset statistics in UMAP space ---
    ds_stats = {}
    for ds in ALL_DATASETS:
        mask = ds_arr == ds
        if np.sum(mask) < 5:
            continue
        pts = emb[mask]
        ds_stats[ds] = {
            "mean": pts.mean(axis=0),
            "cov": np.cov(pts.T),
            "n": int(np.sum(mask)),
        }

    # --- Plot ---
    print("Generating figure...")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.linewidth": 0.5,
    })

    # Colors — training cool, test warm, OOD dark
    TRAIN_CMAP = {
        "uci_har": "#4e79a7", "hhar": "#59a14f", "pamap2": "#76b7b2",
        "wisdm": "#6a9fd6", "dsads": "#9c755f", "kuhar": "#8db0ce",
        "unimib_shar": "#8cd17d", "hapt": "#a0cbe8", "mhealth": "#b6992d",
        "recgym": "#499894",
    }
    TEST_CMAP = {
        "motionsense": "#e15759", "realworld": "#f28e2b", "mobiact": "#b07aa1",
        "harth": "#ff9da7", "vtt_coniot": "#d4a6c8",
    }
    OOD_CMAP = {"shoaib": "#c44e52", "opportunity": "#333333"}
    ALL_CMAP = {**TRAIN_CMAP, **TEST_CMAP, **OOD_CMAP}

    fig, ax = plt.subplots(figsize=(7.16, 5.0))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Draw ellipses + centroids
    for ds in ALL_DATASETS:
        if ds not in ds_stats:
            continue
        st = ds_stats[ds]
        color = ALL_CMAP[ds]
        grp = ("Training" if ds in TRAIN_DATASETS
               else "Zero-Shot Test" if ds in ZEROSHOT_TEST
               else "Severe OOD")

        if grp == "Training":
            # Filled ellipse, muted
            draw_confidence_ellipse(
                ax, st["mean"], st["cov"], n_std=1.5,
                facecolor=color, edgecolor=color,
                alpha=0.15, linewidth=0.8, zorder=1,
            )
            ax.plot(*st["mean"], "o", color=color, markersize=5,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=5)
        elif grp == "Zero-Shot Test":
            # Stronger ellipse, dashed edge
            draw_confidence_ellipse(
                ax, st["mean"], st["cov"], n_std=1.5,
                facecolor=color, edgecolor=color,
                alpha=0.20, linewidth=1.2, linestyle="--", zorder=2,
            )
            ax.plot(*st["mean"], "^", color=color, markersize=7,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=6)
        else:
            # Severe OOD: bold ellipse
            draw_confidence_ellipse(
                ax, st["mean"], st["cov"], n_std=1.5,
                facecolor=color, edgecolor=color,
                alpha=0.22, linewidth=1.5, linestyle="-", zorder=3,
            )
            ax.plot(*st["mean"], "s", color=color, markersize=7,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=7)

    # Label every dataset centroid
    # Pre-compute all centroids to detect overlaps
    centroids = {ds: ds_stats[ds]["mean"] for ds in ALL_DATASETS if ds in ds_stats}
    overall_center = np.mean(list(centroids.values()), axis=0)

    texts = []
    for ds, c in centroids.items():
        color = ALL_CMAP[ds]
        grp = ("Training" if ds in TRAIN_DATASETS
               else "Zero-Shot Test" if ds in ZEROSHOT_TEST
               else "Severe OOD")
        fontsize = 6.0 if grp == "Training" else 7.0
        fontweight = "normal" if grp == "Training" else "bold"

        t = ax.text(
            c[0], c[1], DISPLAY_NAMES[ds],
            fontsize=fontsize, fontweight=fontweight, color=color,
            ha="center", va="center", zorder=10,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                      edgecolor="none", alpha=0.8),
        )
        texts.append(t)

    # Use adjustText to resolve label overlaps
    # Set axis limits BEFORE adjustText so it can respect bounds
    all_c = np.array(list(centroids.values()))
    for dim, setter in [(0, ax.set_xlim), (1, ax.set_ylim)]:
        lo, hi = all_c[:, dim].min(), all_c[:, dim].max()
        span = hi - lo
        setter(lo - 0.35 * span, hi + 0.35 * span)

    try:
        from adjustText import adjust_text
        centroid_x = [c[0] for c in centroids.values()]
        centroid_y = [c[1] for c in centroids.values()]
        adjust_text(texts, x=centroid_x, y=centroid_y, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="#666666",
                                    alpha=0.3, lw=0.4),
                    expand=(1.6, 1.8), force_text=(0.6, 0.8),
                    force_points=(0.4, 0.4),
                    only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
                    ensure_inside_axes=True)
    except (ImportError, TypeError):
        # Fallback: radial offset labels
        for ds, c in centroids.items():
            direction = c - overall_center
            n = np.linalg.norm(direction)
            if n > 1e-6:
                direction /= n
            texts_dict = {t.get_text(): t for t in texts}
            name = DISPLAY_NAMES[ds]
            if name in texts_dict:
                t = texts_dict[name]
                t.set_position((c[0] + direction[0] * 1.5,
                                c[1] + direction[1] * 1.5))

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4e79a7",
               markersize=5, markeredgecolor="white", markeredgewidth=0.4,
               label="Training (10 datasets)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#e15759",
               markersize=6, markeredgecolor="white", markeredgewidth=0.4,
               label="Zero-shot test (5 datasets)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#c44e52",
               markersize=6, markeredgecolor="white", markeredgewidth=0.4,
               label="Severe OOD (2 datasets)"),
    ]
    # Add ellipse explanation
    from matplotlib.patches import Patch
    legend_elements.append(Patch(facecolor="#aaaaaa", edgecolor="#aaaaaa",
                                  alpha=0.2, label="1.5$\\sigma$ confidence region"))

    ax.legend(handles=legend_elements, loc="lower right", fontsize=7,
              frameon=True, framealpha=0.95, edgecolor="#cccccc",
              handletextpad=0.4, labelspacing=0.5, borderpad=0.6)

    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.set_title(
        "Distribution of Raw IMU Signal Characteristics Across 17 HAR Datasets",
        fontsize=9, fontweight="bold", pad=8)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color("#999999")

    plt.tight_layout(pad=0.5)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    fig.savefig(str(output_path.with_suffix(".png")), dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path.with_suffix('.png')}")
    plt.close(fig)


if __name__ == "__main__":
    main()
