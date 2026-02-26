#!/usr/bin/env python3
"""
Generate UMAP visualization of distribution gap between training and
zero-shot test HAR datasets (Fig. X in paper).

Methodology:
  1. Sample up to 400 sessions per dataset from all 17 HAR datasets
  2. Extract 6 IMU channels (acc + gyro) via core_channels mapping
  3. Resample to 128 timesteps, compute 52 raw statistical + spectral
     features per session (no z-score — we want to show actual gap)
  4. Standardize, remove 4-sigma outliers, project via UMAP
  5. Plot: single aggregate training cloud (blue) with individual test
     dataset ellipses overlaid, clearly showing where test distributions
     fall relative to training coverage

The figure communicates: training data covers region X, but zero-shot
test data extends into regions Y and Z that training never covered.

Usage:  python figures/fig_dataset_discrepancy.py
Output: figures/fig_dataset_discrepancy.{pdf,png}
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
from matplotlib.patches import Ellipse, Patch
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
    "unimib_shar": "UniMiB", "hapt": "HAPT", "mhealth": "MHEALTH",
    "recgym": "RecGym", "motionsense": "MotionSense",
    "realworld": "RealWorld", "mobiact": "MobiAct",
    "harth": "HARTH", "vtt_coniot": "VTT-ConIoT",
    "shoaib": "Shoaib", "opportunity": "OPPORTUNITY",
}

CORE_CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
N_FEATURES = 52
FIXED_LEN = 128


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_core_channel_mapping():
    with open(PROJECT_ROOT / "benchmark_data" / "dataset_config.json") as f:
        return {n: i["core_channels"] for n, i in json.load(f)["datasets"].items()}


def _resample(col, n):
    if len(col) == n:
        return col
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(col)), col)


def extract_features(df, core_map):
    chs = {}
    for name in CORE_CHANNELS:
        if name in core_map and core_map[name] in df.columns:
            raw = df[core_map[name]].values.astype(np.float64)
        else:
            raw = np.zeros(len(df), dtype=np.float64)
        raw = np.nan_to_num(raw, nan=0.0)
        chs[name] = _resample(raw, FIXED_LEN) if len(raw) >= 4 else np.zeros(FIXED_LEN)

    features = []
    for name in CORE_CHANNELS:
        col = chs[name]
        s = np.std(col)
        features.extend([
            np.mean(col), s, np.min(col), np.max(col),
            float(skew(col)) if s > 1e-10 else 0.0,
            float(kurtosis(col)) if s > 1e-10 else 0.0,
            np.sum(np.diff(np.sign(col)) != 0) / max(len(col) - 1, 1),
        ])
    acc_mag = np.sqrt(sum(chs[f"acc_{a}"] ** 2 for a in "xyz"))
    features.extend([np.mean(acc_mag), np.std(acc_mag), np.min(acc_mag), np.max(acc_mag)])
    for a, b in [("acc_x", "acc_y"), ("acc_x", "acc_z"), ("acc_y", "acc_z")]:
        if np.std(chs[a]) > 1e-10 and np.std(chs[b]) > 1e-10:
            r = np.corrcoef(chs[a], chs[b])[0, 1]
            features.append(r if np.isfinite(r) else 0.0)
        else:
            features.append(0.0)
    for a in ["acc_x", "acc_y", "acc_z"]:
        fm = np.abs(np.fft.rfft(chs[a]))[1:]
        fr = np.fft.rfftfreq(FIXED_LEN)[1:]
        features.append(np.sum(fr * fm) / np.sum(fm) if fm.sum() > 1e-10 else 0.0)
    return np.array(features, dtype=np.float32)


def load_dataset_features(name, data_dir, core_map, max_sess=400, seed=42):
    sdir = data_dir / name / "sessions"
    if not sdir.exists():
        return np.empty((0, N_FEATURES), dtype=np.float32)
    sessions = sorted(os.listdir(sdir))
    rng = np.random.RandomState(seed)
    if len(sessions) > max_sess:
        idx = rng.choice(len(sessions), max_sess, replace=False)
        sessions = [sessions[i] for i in sorted(idx)]
    feats = []
    for s in sessions:
        p = sdir / s / "data.parquet"
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
            f = extract_features(df, core_map)
            if np.all(np.isfinite(f)):
                feats.append(f)
        except Exception:
            continue
    return np.stack(feats) if feats else np.empty((0, N_FEATURES), dtype=np.float32)


def confidence_ellipse(ax, mean, cov, n_std=1.5, **kw):
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-6)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    w, h = 2 * n_std * np.sqrt(vals)
    e = Ellipse(xy=mean, width=w, height=h, angle=angle, **kw)
    ax.add_patch(e)
    return e


def smooth_hull(ax, pts, **kw):
    if len(pts) < 3:
        return
    try:
        hull = ConvexHull(pts)
    except Exception:
        return
    hp = np.vstack([pts[hull.vertices], pts[hull.vertices[0]]])
    c = pts.mean(0)
    d = hp - c
    n = np.linalg.norm(d, axis=1, keepdims=True)
    n[n < 1e-8] = 1.0
    hp = hp + 0.4 * d / n
    ax.fill(hp[:, 0], hp[:, 1], **kw)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    data_dir = PROJECT_ROOT / "data"
    core_maps = load_core_channel_mapping()

    print("Loading sessions...")
    all_features, all_ds, all_groups = [], [], []
    for ds in ALL_DATASETS:
        if ds not in core_maps:
            continue
        print(f"  {DISPLAY_NAMES[ds]:12s}", end="", flush=True)
        feats = load_dataset_features(ds, data_dir, core_maps[ds])
        print(f" {len(feats):>4d}")
        if len(feats) == 0:
            continue
        all_features.append(feats)
        all_ds.extend([ds] * len(feats))
        grp = ("train" if ds in TRAIN_DATASETS
               else "test" if ds in ZEROSHOT_TEST else "ood")
        all_groups.extend([grp] * len(feats))

    X = np.concatenate(all_features)
    ds_arr, grp_arr = np.array(all_ds), np.array(all_groups)
    print(f"Total: {len(X)}")

    mu, sig = X.mean(0), X.std(0)
    sig[sig < 1e-10] = 1.0
    Xn = (X - mu) / sig
    # 4σ for training, 6σ for test/OOD (they're supposed to be outliers)
    max_abs = np.max(np.abs(Xn), 1)
    keep = np.where(
        grp_arr == "train",
        max_abs < 4.0,
        max_abs < 6.0,
    )
    print(f"  Outliers removed: {np.sum(~keep)}")
    Xn, ds_arr, grp_arr = Xn[keep], ds_arr[keep], grp_arr[keep]

    print("UMAP...")
    import umap
    emb = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                     metric="euclidean", random_state=42).fit_transform(Xn)

    # Per-dataset stats
    ds_stats = {}
    for ds in ALL_DATASETS:
        m = ds_arr == ds
        if m.sum() < 5:
            continue
        pts = emb[m]
        ds_stats[ds] = {"mean": pts.mean(0), "cov": np.cov(pts.T), "pts": pts}

    # ---------- PLOT ----------
    print("Plotting...")
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8})

    # Test/OOD color per dataset
    TEST_COLORS = {
        "motionsense": "#e15759", "realworld": "#f28e2b", "mobiact": "#b07aa1",
        "harth": "#ff9da7", "vtt_coniot": "#59a14f",
        "shoaib": "#c44e52", "opportunity": "#333333",
    }

    fig, ax = plt.subplots(figsize=(7.16, 5.0))
    ax.set_facecolor("white")

    # ---- Layer 1: Training point cloud (individual dots, very light) ----
    train_mask = grp_arr == "train"
    ax.scatter(emb[train_mask, 0], emb[train_mask, 1],
               c="#b8cce4", marker=".", s=3, alpha=0.25,
               edgecolors="none", zorder=1, rasterized=True)

    # ---- Layer 2: Training convex hull (shows coverage boundary) ----
    train_pts = emb[train_mask]
    smooth_hull(ax, train_pts, color="#4e79a7", alpha=0.08, zorder=0)
    # Also draw hull border
    try:
        hull = ConvexHull(train_pts)
        hp = np.vstack([train_pts[hull.vertices], train_pts[hull.vertices[0]]])
        c = train_pts.mean(0)
        d = hp - c
        n = np.linalg.norm(d, axis=1, keepdims=True)
        n[n < 1e-8] = 1.0
        hp = hp + 0.4 * d / n
        ax.plot(hp[:, 0], hp[:, 1], color="#4e79a7", alpha=0.4,
                linewidth=1.5, linestyle="-", zorder=1, label="_")
    except Exception:
        pass

    # ---- Layer 3: Each test/OOD dataset as filled ellipse ----
    for ds in ZEROSHOT_TEST + SEVERE_OOD:
        if ds not in ds_stats:
            continue
        st = ds_stats[ds]
        color = TEST_COLORS[ds]
        is_ood = ds in SEVERE_OOD

        # Filled confidence ellipse
        confidence_ellipse(
            ax, st["mean"], st["cov"], n_std=1.5,
            facecolor=color, edgecolor=color,
            alpha=0.25 if not is_ood else 0.30,
            linewidth=1.5 if not is_ood else 2.0,
            linestyle="--" if not is_ood else "-",
            zorder=4,
        )
        # Centroid marker
        marker = "s" if is_ood else "^"
        ax.plot(*st["mean"], marker=marker, color=color, markersize=7,
                markeredgecolor="white", markeredgewidth=0.7, zorder=8)

    # ---- Layer 4: Labels for test/OOD datasets ----
    test_centroids = {ds: ds_stats[ds]["mean"]
                      for ds in ZEROSHOT_TEST + SEVERE_OOD if ds in ds_stats}
    overall_c = emb.mean(0)

    texts = []
    for ds, c in test_centroids.items():
        color = TEST_COLORS[ds]
        is_ood = ds in SEVERE_OOD
        t = ax.text(c[0], c[1], DISPLAY_NAMES[ds],
                    fontsize=7.5 if is_ood else 7.0,
                    fontweight="bold",
                    color=color, ha="center", va="center", zorder=10,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.9))
        texts.append(t)

    # Set limits before adjustText
    all_c = np.array(list(test_centroids.values()))
    train_c = train_pts.mean(0).reshape(1, 2)
    all_c = np.vstack([all_c, train_c])
    for dim, setter in [(0, ax.set_xlim), (1, ax.set_ylim)]:
        lo, hi = all_c[:, dim].min(), all_c[:, dim].max()
        span = hi - lo
        setter(lo - 0.4 * span, hi + 0.4 * span)

    try:
        from adjustText import adjust_text
        cx = [c[0] for c in test_centroids.values()]
        cy = [c[1] for c in test_centroids.values()]
        adjust_text(texts, x=cx, y=cy, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="#444444",
                                    alpha=0.4, lw=0.6),
                    expand=(2.0, 2.2), force_text=(1.0, 1.2),
                    force_points=(0.5, 0.5),
                    ensure_inside_axes=True)
    except (ImportError, TypeError):
        pass

    # ---- Layer 5: "Training region" label ----
    tc = train_pts.mean(0)
    ax.text(tc[0], tc[1], "Training\nregion",
            fontsize=9, fontweight="bold", color="#4e79a7",
            ha="center", va="center", alpha=0.5, zorder=2,
            fontstyle="italic")

    # ---- Legend ----
    legend_handles = [
        Patch(facecolor="#b8cce4", edgecolor="#4e79a7", alpha=0.4,
              linewidth=1.5, label="Training (10 datasets)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#e15759",
               markersize=7, markeredgecolor="white", markeredgewidth=0.5,
               label="Zero-shot test (5)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#c44e52",
               markersize=7, markeredgecolor="white", markeredgewidth=0.5,
               label="Severe OOD (2)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7.5,
              frameon=True, framealpha=0.95, edgecolor="#cccccc",
              borderpad=0.6, handletextpad=0.4, labelspacing=0.5)

    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.set_title("Distribution Gap Between Training and Zero-Shot Test Datasets",
                 fontsize=9.5, fontweight="bold", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
        sp.set_color("#999999")

    plt.tight_layout(pad=0.5)
    out = PROJECT_ROOT / "figures" / "fig_dataset_discrepancy"
    fig.savefig(str(out.with_suffix(".pdf")), dpi=300, bbox_inches="tight")
    fig.savefig(str(out.with_suffix(".png")), dpi=200, bbox_inches="tight")
    print(f"Saved: {out}.pdf / .png")
    plt.close(fig)


if __name__ == "__main__":
    main()
