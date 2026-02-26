#!/usr/bin/env python3
"""
Generate UMAP visualization of distribution gap between training and
zero-shot test HAR datasets (Fig. X in paper).

Methodology:
  1. Sample up to 300 sessions per dataset from all 17 HAR datasets
  2. Extract 3 accelerometer channels (acc_x/y/z) using core_channels
     mapping — accelerometer is universal across all datasets, avoiding
     trivial separation from gyroscope availability differences
  3. Resample each session to 128 timesteps via linear interpolation
     (matches the model's patch preprocessing)
  4. Apply per-channel z-score normalization (matches the model's
     per-patch normalization) — this removes mean/scale differences
     from sensor calibration
  5. Extract 39 features per session (13 per channel):
     - Post-normalization: skewness, kurtosis, normalized min/max,
       IQR, zero-crossing rate, lag-1 autocorrelation,
       mean absolute difference, jerk (1st-derivative std)
     - Frequency domain: dominant frequency, spectral centroid,
       spectral entropy, low/high energy ratio
  6. Standardize features, remove 4-sigma outliers (~4%)
  7. Project to 2D via UMAP (n_neighbors=50, min_dist=0.5)
  8. Plot with group-coded markers and convex hulls

Sampling rate note: datasets range from 20-100 Hz. After resampling
to 128 fixed timesteps, frequency features are in normalized frequency
space (0 to Nyquist). This means the same physical motion at different
sampling rates produces different spectral signatures — this is a real
source of distribution shift the model must handle.

Usage:
    python figures/fig_dataset_discrepancy.py

Output:
    figures/fig_dataset_discrepancy.pdf  (300 DPI, vector)
    figures/fig_dataset_discrepancy.png  (200 DPI, for quick viewing)
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
from scipy.stats import skew, kurtosis
from scipy.spatial import ConvexHull

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
    "unimib_shar": "UniMiB SHAR", "hapt": "HAPT", "mhealth": "MHEALTH",
    "recgym": "RecGym", "motionsense": "MotionSense",
    "realworld": "RealWorld", "mobiact": "MobiAct",
    "harth": "HARTH", "vtt_coniot": "VTT-ConIoT",
    "shoaib": "Shoaib", "opportunity": "OPPORTUNITY",
}

# Accelerometer only — available in ALL datasets
CORE_CHANNELS = ["acc_x", "acc_y", "acc_z"]
N_FEATURES = 39   # 3 channels x 13 features
FIXED_LEN = 128   # Resample target (matches model patch size x2)


# ---------------------------------------------------------------------------
# Core channel mapping
# ---------------------------------------------------------------------------
def load_core_channel_mapping():
    config_path = PROJECT_ROOT / "benchmark_data" / "dataset_config.json"
    with open(config_path) as f:
        cfg = json.load(f)
    return {name: info["core_channels"] for name, info in cfg["datasets"].items()}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _zscore(col):
    mu, sigma = np.mean(col), np.std(col)
    return (col - mu) / sigma if sigma > 1e-10 else col - mu


def _resample(col, target_len):
    if len(col) == target_len:
        return col
    return np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(col)), col)


def extract_features(df, core_map):
    """Extract 39 post-normalization + frequency features from one session."""
    features = []
    for core_name in CORE_CHANNELS:
        if core_name in core_map and core_map[core_name] in df.columns:
            raw = df[core_map[core_name]].values.astype(np.float64)
        else:
            raw = np.zeros(len(df), dtype=np.float64)
        raw = np.nan_to_num(raw, nan=0.0)
        if len(raw) < 4:
            features.extend([0.0] * 13)
            continue

        raw = _resample(raw, FIXED_LEN)
        col = _zscore(raw)

        # Post-normalization stats (9)
        features.append(float(skew(col)) if np.std(col) > 1e-10 else 0.0)
        features.append(float(kurtosis(col)) if np.std(col) > 1e-10 else 0.0)
        features.append(np.min(col))
        features.append(np.max(col))
        q75, q25 = np.percentile(col, [75, 25])
        features.append(q75 - q25)
        features.append(np.sum(np.diff(np.sign(col)) != 0) / max(len(col) - 1, 1))
        ac1 = np.corrcoef(col[:-1], col[1:])[0, 1] if np.std(col) > 1e-10 else 0.0
        features.append(ac1 if np.isfinite(ac1) else 0.0)
        features.append(np.mean(np.abs(np.diff(col))))
        features.append(np.std(np.diff(col)))

        # Frequency domain (4)
        fft_mag = np.abs(np.fft.rfft(col))
        freqs = np.fft.rfftfreq(len(col))
        fft_mag_nodc, freqs_nodc = fft_mag[1:], freqs[1:]
        if len(fft_mag_nodc) > 0 and fft_mag_nodc.sum() > 1e-10:
            features.append(freqs_nodc[np.argmax(fft_mag_nodc)])
            features.append(np.sum(freqs_nodc * fft_mag_nodc) / np.sum(fft_mag_nodc))
            psd = fft_mag_nodc ** 2
            psd_norm = psd / psd.sum()
            psd_norm = psd_norm[psd_norm > 1e-20]
            features.append(-np.sum(psd_norm * np.log2(psd_norm)) / max(np.log2(len(psd_norm)), 1))
            low_mask = freqs_nodc < 0.1
            low_e, high_e = np.sum(fft_mag_nodc[low_mask] ** 2), np.sum(fft_mag_nodc[~low_mask] ** 2)
            features.append(low_e / max(low_e + high_e, 1e-10))
        else:
            features.extend([0.0, 0.0, 0.0, 0.5])

    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset_features(dataset_name, data_dir, core_map, max_sessions=300, seed=42):
    sessions_dir = data_dir / dataset_name / "sessions"
    if not sessions_dir.exists():
        return np.empty((0, N_FEATURES), dtype=np.float32)

    all_sessions = sorted(os.listdir(sessions_dir))
    rng = np.random.RandomState(seed)
    if len(all_sessions) > max_sessions:
        indices = rng.choice(len(all_sessions), max_sessions, replace=False)
        selected = [all_sessions[i] for i in sorted(indices)]
    else:
        selected = all_sessions

    features_list = []
    for sess_name in selected:
        parquet_path = sessions_dir / sess_name / "data.parquet"
        if not parquet_path.exists():
            continue
        try:
            df = pd.read_parquet(parquet_path)
            # Skip near-constant sessions (degenerate after z-score)
            acc_stds = []
            for ch in CORE_CHANNELS:
                if ch in core_map and core_map[ch] in df.columns:
                    acc_stds.append(df[core_map[ch]].std())
            if acc_stds and max(acc_stds) < 0.02:
                continue
            feat = extract_features(df, core_map)
            if np.all(np.isfinite(feat)):
                features_list.append(feat)
        except Exception:
            continue

    return np.stack(features_list) if features_list else np.empty((0, N_FEATURES), dtype=np.float32)


def draw_smooth_hull(ax, points, color, alpha=0.08, edge_alpha=0.4, linewidth=1.5, pad=0.5):
    if len(points) < 3:
        return
    try:
        hull = ConvexHull(points)
    except Exception:
        return
    hull_pts = np.vstack([points[hull.vertices], points[hull.vertices[0]]])
    centroid = points.mean(axis=0)
    directions = hull_pts - centroid
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    hull_pts = hull_pts + pad * directions / norms
    ax.fill(hull_pts[:, 0], hull_pts[:, 1], color=color, alpha=alpha, zorder=0)
    ax.plot(hull_pts[:, 0], hull_pts[:, 1], color=color, alpha=edge_alpha,
            linewidth=linewidth, zorder=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    data_dir = PROJECT_ROOT / "data"
    output_path = PROJECT_ROOT / "figures" / "fig_dataset_discrepancy.pdf"
    core_maps = load_core_channel_mapping()

    # --- Load and featurize ---
    print("Loading and featurizing sessions...")
    all_features, all_dataset_ids, all_groups = [], [], []
    dataset_counts = {}

    for ds_name in ALL_DATASETS:
        if ds_name not in core_maps:
            continue
        print(f"  {DISPLAY_NAMES.get(ds_name, ds_name)}...", end="", flush=True)
        feats = load_dataset_features(ds_name, data_dir, core_maps[ds_name])
        print(f" {len(feats)} sessions")
        if len(feats) == 0:
            continue
        all_features.append(feats)
        all_dataset_ids.extend([ds_name] * len(feats))
        dataset_counts[ds_name] = len(feats)
        if ds_name in TRAIN_DATASETS:
            all_groups.extend(["Training"] * len(feats))
        elif ds_name in ZEROSHOT_TEST:
            all_groups.extend(["Zero-Shot Test"] * len(feats))
        else:
            all_groups.extend(["Severe OOD"] * len(feats))

    X = np.concatenate(all_features, axis=0)
    dataset_ids = np.array(all_dataset_ids)
    groups = np.array(all_groups)
    print(f"\nTotal: {len(X)} samples from {len(dataset_counts)} datasets")

    # Standardize + outlier removal
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std < 1e-10] = 1.0
    X_norm = (X - mean) / std
    inlier = np.max(np.abs(X_norm), axis=1) < 4.0
    n_out = np.sum(~inlier)
    if n_out:
        print(f"  Removed {n_out} outliers ({n_out / len(X) * 100:.1f}%)")
        X_norm, dataset_ids, groups = X_norm[inlier], dataset_ids[inlier], groups[inlier]
        dataset_counts = {ds: int(np.sum(dataset_ids == ds)) for ds in np.unique(dataset_ids)}

    # UMAP
    print("Fitting UMAP...")
    import umap
    embedding = umap.UMAP(
        n_components=2, n_neighbors=50, min_dist=0.5,
        metric="euclidean", random_state=42,
    ).fit_transform(X_norm)

    # --- Plotting (paper format: double-column, ~7 in wide) ---
    print("Generating figure...")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.linewidth": 0.6,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
    })

    # Colors
    train_colors = {
        "uci_har": "#4e79a7", "hhar": "#59a14f", "pamap2": "#76b7b2",
        "wisdm": "#6a9fd6", "dsads": "#9c755f", "kuhar": "#bab0ac",
        "unimib_shar": "#8cd17d", "hapt": "#a0cbe8", "mhealth": "#b6992d",
        "recgym": "#499894",
    }
    test_colors = {
        "motionsense": "#e15759", "realworld": "#f28e2b", "mobiact": "#b07aa1",
        "harth": "#ff9da7", "vtt_coniot": "#d4a6c8",
    }
    ood_colors = {"shoaib": "#d62728", "opportunity": "#1a1a1a"}
    all_colors = {**train_colors, **test_colors, **ood_colors}

    fig, ax = plt.subplots(figsize=(7.16, 5.0))  # IEEE double-column width
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # View bounds (clip to main cluster)
    view_q = {}
    for dim in [0, 1]:
        q5, q95 = np.percentile(embedding[:, dim], [5, 95])
        iqr = q95 - q5
        view_q[dim] = (q5 - 0.3 * iqr, q95 + 0.3 * iqr)
    view_mask = (
        (embedding[:, 0] > view_q[0][0]) & (embedding[:, 0] < view_q[0][1]) &
        (embedding[:, 1] > view_q[1][0]) & (embedding[:, 1] < view_q[1][1])
    )

    # Convex hulls
    for group_name, hull_color, hull_alpha in [
        ("Training", "#4e79a7", 0.05),
        ("Zero-Shot Test", "#e15759", 0.04),
    ]:
        mask = (groups == group_name) & view_mask
        if np.sum(mask) >= 3:
            draw_smooth_hull(ax, embedding[mask], hull_color,
                             alpha=hull_alpha, edge_alpha=0.25, linewidth=1.2, pad=0.4)
    for ds_name in SEVERE_OOD:
        mask = (dataset_ids == ds_name) & view_mask
        if np.sum(mask) >= 3:
            draw_smooth_hull(ax, embedding[mask], ood_colors[ds_name],
                             alpha=0.06, edge_alpha=0.35, linewidth=1.0, pad=0.3)

    # Scatter — training (background)
    for ds_name in TRAIN_DATASETS:
        mask = dataset_ids == ds_name
        if not np.any(mask):
            continue
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=train_colors[ds_name], marker="o", s=4, alpha=0.35,
                   edgecolors="none", zorder=1, rasterized=True)

    # Scatter — zero-shot test
    for ds_name in ZEROSHOT_TEST:
        mask = dataset_ids == ds_name
        if not np.any(mask):
            continue
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=test_colors[ds_name], marker="^", s=14, alpha=0.55,
                   edgecolors=test_colors[ds_name], linewidths=0.2,
                   zorder=2, rasterized=True)

    # Scatter — severe OOD
    for ds_name in SEVERE_OOD:
        mask = dataset_ids == ds_name
        if not np.any(mask):
            continue
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=ood_colors[ds_name], marker="s", s=16, alpha=0.6,
                   edgecolors="white", linewidths=0.2, zorder=3, rasterized=True)

    # Labels — radial offset from overall centroid
    overall_center = embedding[view_mask].mean(axis=0) if view_mask.any() else embedding.mean(axis=0)
    for ds_name in ZEROSHOT_TEST + SEVERE_OOD:
        mask = dataset_ids == ds_name
        if not np.any(mask):
            continue
        centroid = embedding[mask].mean(axis=0)
        direction = centroid - overall_center
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction /= norm
        else:
            direction = np.array([1.0, 0.0])
        dx, dy = direction[0] * 30, direction[1] * 30
        ax.annotate(
            DISPLAY_NAMES[ds_name], xy=centroid,
            fontsize=6.5, fontweight="bold", color=all_colors[ds_name],
            ha="center", va="center",
            xytext=(dx, dy), textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color=all_colors[ds_name],
                            alpha=0.35, lw=0.5),
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                      edgecolor="none", alpha=0.85),
            zorder=10,
        )

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4e79a7",
               markersize=5, markeredgecolor="none",
               label="Training (10 datasets)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#e15759",
               markersize=5.5, markeredgecolor="#e15759", markeredgewidth=0.3,
               label="Zero-shot test (5 datasets)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#d62728",
               markersize=5.5, markeredgecolor="white", markeredgewidth=0.3,
               label="Severe OOD (2 datasets)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=7.5,
              frameon=True, framealpha=0.95, edgecolor="#cccccc",
              handletextpad=0.4, labelspacing=0.45, borderpad=0.6)

    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.set_title("Distribution of Accelerometer Signal Features Across 17 HAR Datasets",
                 fontsize=9.5, fontweight="bold", pad=8)

    # Zoom to main cluster
    for dim, setter in [(0, ax.set_xlim), (1, ax.set_ylim)]:
        q5, q95 = np.percentile(embedding[:, dim], [5, 95])
        iqr = q95 - q5
        setter(q5 - 0.6 * iqr, q95 + 0.6 * iqr)

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
