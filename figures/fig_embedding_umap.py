#!/usr/bin/env python3
"""Generate publication-quality UMAP figure of IMU-text embedding alignment.

Single-column academic paper figure (~3.5 inches wide, 300 DPI).
Loads a trained TSFM checkpoint, extracts embeddings on all test datasets,
computes per-label centroids, and plots a joint 2D UMAP.

Usage:
    TSFM_CHECKPOINT=training_output/semantic_alignment/small_deep_v2_4b3fdd6/best.pt \
        python figures/fig_embedding_umap.py

    # Or use default checkpoint path
    python figures/fig_embedding_umap.py
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

# ── Project imports ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from val_scripts.human_activity_recognition.model_loading import load_model, load_label_bank
from datasets.imu_pretraining_dataset.label_groups import LABEL_GROUPS_SIMPLE

# ── Config ──────────────────────────────────────────────────────────────────
_DEFAULT_CHECKPOINT = str(
    PROJECT_ROOT / "training_output" / "semantic_alignment"
    / "small_deep_v2_4b3fdd6" / "best.pt"
)
CHECKPOINT_PATH = os.environ.get("TSFM_CHECKPOINT", _DEFAULT_CHECKPOINT)
BATCH_SIZE = 64
OUTPUT_DIR = PROJECT_ROOT / "figures"

# Eval constants (matching evaluate_tsfm.py at this commit)
BENCHMARK_DIR = PROJECT_ROOT / "benchmark_data"
TSFM_EVAL_DIR = BENCHMARK_DIR / "processed" / "tsfm_eval"
LIMUBERT_DATA_DIR = BENCHMARK_DIR / "processed" / "limubert"
DATA_CHANNELS = 6
PATCH_SIZE_SEC = 1.0

# Load dataset & label config
DATASET_CONFIG_PATH = BENCHMARK_DIR / "dataset_config.json"
GLOBAL_LABEL_PATH = LIMUBERT_DATA_DIR / "global_label_mapping.json"

with open(DATASET_CONFIG_PATH) as f:
    DATASET_CONFIG = json.load(f)
TEST_DATASETS = DATASET_CONFIG["zero_shot_datasets"]

with open(GLOBAL_LABEL_PATH) as f:
    GLOBAL_LABELS = json.load(f)["labels"]

# UMAP parameters (tuned for ~170 centroid points)
UMAP_N_NEIGHBORS = 10
UMAP_MIN_DIST = 0.05
EXCLUDE_GROUPS = ('postural_transition',)


# ── Data loading (mirrors evaluate_tsfm.py) ────────────────────────────────

def get_dataset_metadata(dataset_name: str) -> dict:
    manifest_path = PROJECT_ROOT / "data" / dataset_name / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    channels = manifest.get('channels', manifest.get('channel_descriptions', []))
    if isinstance(channels[0], dict):
        ch_descs = [c['description'] for c in channels]
    else:
        ch_descs = channels

    sr = manifest.get('sampling_rate_hz', manifest.get('sampling_rate', 50.0))
    n_real = len(ch_descs)
    has_gyro = n_real >= 6

    # Pad to DATA_CHANNELS if fewer
    while len(ch_descs) < DATA_CHANNELS:
        ch_descs.append(f"unused_pad_channel_{len(ch_descs)}")

    return {
        'channel_descriptions': ch_descs[:DATA_CHANNELS],
        'sampling_rate_hz': sr,
        'n_real_channels': n_real,
        'has_gyro': has_gyro,
    }


def load_raw_data(dataset_name: str):
    """Load preprocessed numpy arrays from benchmark_data."""
    ds_dir = TSFM_EVAL_DIR / dataset_name
    data = np.load(str(ds_dir / "data_native.npy")).astype(np.float32)
    label_path = ds_dir / "label_native.npy"
    if not label_path.exists():
        label_path = ds_dir / "labels.npy"
    labels = np.load(str(label_path)).astype(np.float32)
    with open(ds_dir / "metadata.json") as f:
        meta = json.load(f)
    sr = meta.get('sampling_rate_hz', 50.0)
    return data, labels, sr


def get_window_labels(labels_raw: np.ndarray, label_index: int = 0) -> np.ndarray:
    """Extract per-window activity labels as integer indices (majority vote)."""
    act_labels = labels_raw[:, :, label_index]
    t = int(np.min(act_labels))
    act_labels = act_labels - t
    window_labels = np.array([
        np.bincount(row.astype(int)).argmax() for row in act_labels
    ], dtype=np.int64)
    return window_labels


def get_dataset_labels(dataset_name: str) -> List[str]:
    """Get sorted activity labels for a dataset."""
    return sorted(DATASET_CONFIG["datasets"][dataset_name]["activities"])


# ── Embedding extraction ───────────────────────────────────────────────────

def extract_embeddings(model, raw_data, device, sampling_rate, channel_descriptions,
                       has_gyro=True):
    """Extract session-level L2-normalised embeddings."""
    model.train(False)
    N = raw_data.shape[0]
    seq_len = raw_data.shape[1]
    all_embs = []

    for start in tqdm(range(0, N, BATCH_SIZE), desc="Extracting embeddings",
                      total=(N + BATCH_SIZE - 1) // BATCH_SIZE, leave=False):
        end = min(start + BATCH_SIZE, N)
        batch = torch.from_numpy(raw_data[start:end]).float().to(device)
        bs = batch.shape[0]

        ch_mask = torch.ones(bs, DATA_CHANNELS, dtype=torch.bool, device=device)
        if not has_gyro:
            ch_mask[:, 3:] = False
        att_mask = torch.ones(bs, seq_len, dtype=torch.bool, device=device)
        ch_descs = [channel_descriptions[:] for _ in range(bs)]
        srs = [sampling_rate] * bs
        ps = [PATCH_SIZE_SEC] * bs

        with torch.no_grad():
            with autocast('cuda', enabled=device.type == 'cuda'):
                emb = model.forward_from_raw(
                    batch, ch_descs, ch_mask, srs, ps,
                    attention_mask=att_mask,
                )
            all_embs.append(emb.float().cpu().numpy())

    return np.concatenate(all_embs, axis=0)


# ── Group color mapping ────────────────────────────────────────────────────

def get_label_to_group(labels: List[str]) -> Dict[str, str]:
    label_to_group = {}
    for group_name, members in LABEL_GROUPS_SIMPLE.items():
        for lbl in members:
            label_to_group[lbl] = group_name
    return label_to_group


# Distinct colors for LABEL_GROUPS_SIMPLE groups (17 groups)
_TAB20 = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2',
]


# ── Figure generation ──────────────────────────────────────────────────────

def make_paper_figure(
    imu_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    labels: List[str],
    output_path: Path,
    exclude_groups: Tuple[str, ...] = EXCLUDE_GROUPS,
):
    """Publication-quality single-column UMAP figure.

    Minimal design: circles (IMU) and stars (text) colored by activity group,
    with group labels placed at cluster centroids. No axis labels, no legend
    clutter — the visual tells the story.
    """
    import umap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 7,
    })

    label_to_group = get_label_to_group(labels)

    # Filter invalid embeddings
    valid = ~(np.isnan(imu_embeddings).any(axis=1) | np.isinf(imu_embeddings).any(axis=1)
              | np.isnan(text_embeddings).any(axis=1) | np.isinf(text_embeddings).any(axis=1))
    imu_embeddings = imu_embeddings[valid]
    text_embeddings = text_embeddings[valid]
    labels = [l for l, v in zip(labels, valid) if v]

    # Exclude specified groups
    if exclude_groups:
        keep = np.array([label_to_group.get(l, l) not in exclude_groups for l in labels])
        n_dropped = len(labels) - keep.sum()
        if n_dropped:
            print(f"  Excluding {n_dropped} samples from groups: {exclude_groups}")
        imu_embeddings = imu_embeddings[keep]
        text_embeddings = text_embeddings[keep]
        labels = [l for l, k in zip(labels, keep) if k]

    labels_arr = np.array(labels)
    unique_labels = sorted(set(labels))

    # Per-label centroids
    imu_centroids = np.zeros((len(unique_labels), imu_embeddings.shape[1]))
    text_centroids = np.zeros((len(unique_labels), text_embeddings.shape[1]))
    for i, lbl in enumerate(unique_labels):
        mask = labels_arr == lbl
        imu_centroids[i] = imu_embeddings[mask].mean(axis=0)
        text_centroids[i] = text_embeddings[mask].mean(axis=0)

    # Re-normalise
    imu_centroids /= np.maximum(np.linalg.norm(imu_centroids, axis=1, keepdims=True), 1e-8)
    text_centroids /= np.maximum(np.linalg.norm(text_centroids, axis=1, keepdims=True), 1e-8)

    centroid_groups = np.array([label_to_group.get(l, l) for l in unique_labels])
    present_groups = sorted(set(centroid_groups))

    # Assign colors per group
    group_colors = {g: _TAB20[i % len(_TAB20)] for i, g in enumerate(present_groups)}

    # Balanced subsample: equal samples per group so no group dominates visually
    MAX_SAMPLES = 5000
    rng = np.random.RandomState(42)
    groups_all = np.array([label_to_group.get(l, l) for l in labels])
    present_groups_arr = sorted(set(groups_all))
    per_group = max(1, MAX_SAMPLES // len(present_groups_arr))

    balanced_idx = []
    for g in present_groups_arr:
        g_indices = np.where(groups_all == g)[0]
        n_take = min(per_group, len(g_indices))
        balanced_idx.extend(rng.choice(g_indices, n_take, replace=False).tolist())
    rng.shuffle(balanced_idx)
    balanced_idx = np.array(balanced_idx)

    imu_sub = imu_embeddings[balanced_idx]
    labels_sub = [labels[i] for i in balanced_idx]
    labels_sub_arr = np.array(labels_sub)
    groups_sub = np.array([label_to_group.get(l, l) for l in labels_sub])

    # Joint UMAP on subsampled IMU + text centroids
    all_points = np.vstack([imu_sub, text_centroids])
    n_imu = len(imu_sub)
    n_text = len(text_centroids)
    actual_neighbors = min(15, len(all_points) - 1)
    print(f"  UMAP: {n_imu} IMU samples + {n_text} text centroids, "
          f"n_neighbors={actual_neighbors}")
    reducer = umap.UMAP(
        n_neighbors=actual_neighbors,
        min_dist=0.1,
        n_components=2,
        metric='cosine',
        random_state=42,
    )
    proj = reducer.fit_transform(all_points)
    imu_proj = proj[:n_imu]
    text_proj = proj[n_imu:]

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.set_facecolor('white')

    # Layer 1: IMU sample cloud (small dots, low alpha)
    for g in present_groups:
        c = group_colors[g]
        mask = groups_sub == g
        if not mask.any():
            continue
        ax.scatter(imu_proj[mask, 0], imu_proj[mask, 1],
                   c=c, marker='.', s=3, alpha=0.3, zorder=2,
                   rasterized=True)  # rasterize dots for small PDF

    # Layer 2: text centroids (bold stars)
    for g in present_groups:
        c = group_colors[g]
        mask = centroid_groups == g
        if not mask.any():
            continue
        ax.scatter(text_proj[mask, 0], text_proj[mask, 1],
                   c=c, marker='*', s=70, alpha=0.95,
                   edgecolors='black', linewidths=0.3, zorder=4)

    # Layer 3: group labels near the text centroid cluster center (auto-repelled)
    from adjustText import adjust_text
    texts = []
    for g in present_groups:
        mask = centroid_groups == g
        if not mask.any():
            continue
        # Anchor at IMU cluster center (visually dominant), not text centroid
        imu_mask = groups_sub == g
        if imu_mask.any():
            cx, cy = imu_proj[imu_mask, 0].mean(), imu_proj[imu_mask, 1].mean()
        else:
            cx, cy = text_proj[mask, 0].mean(), text_proj[mask, 1].mean()
        display = g.replace('_', ' ')
        t = ax.text(cx, cy, display, fontsize=7, fontweight='bold',
                    color=group_colors[g], ha='center', va='center', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                              alpha=0.8, edgecolor='none'))
        texts.append(t)

    adjust_text(texts, ax=ax,
                force_text=(0.8, 0.8), force_points=(0.3, 0.3),
                expand_text=(1.2, 1.4),
                arrowprops=dict(arrowstyle='-', color='#aaaaaa', lw=0.4))

    # Minimal modality legend
    legend_handles = [
        Line2D([0], [0], marker='o', color='#888', ms=3, lw=0,
               alpha=0.5, label='Sensor embedding'),
        Line2D([0], [0], marker='*', color='#888', ms=6, lw=0,
               markeredgecolor='black', markeredgewidth=0.3,
               label='Text centroid'),
    ]
    ax.legend(handles=legend_handles, fontsize=5.5, loc='lower right',
              framealpha=0.85, borderpad=0.3, handletextpad=0.2)

    # Axes
    ax.set_xlabel('UMAP 1', fontsize=7)
    ax.set_ylabel('UMAP 2', fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Tight axis limits
    all_pts = np.vstack([imu_proj, text_proj])
    pad = 0.08
    xr = all_pts[:, 0].max() - all_pts[:, 0].min()
    yr = all_pts[:, 1].max() - all_pts[:, 1].min()
    ax.set_xlim(all_pts[:, 0].min() - pad * xr, all_pts[:, 0].max() + pad * xr)
    ax.set_ylim(all_pts[:, 1].min() - pad * yr, all_pts[:, 1].max() + pad * yr)

    plt.tight_layout(pad=0.2)

    # Save PNG + PDF
    png_path = output_path.with_suffix('.png')
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {png_path.name} + {pdf_path.name}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    # Load model
    model, checkpoint, hp_path = load_model(CHECKPOINT_PATH, device)
    label_bank = load_label_bank(checkpoint, device, hp_path)
    model.train(False)
    label_bank.train(False)
    print("Model loaded")

    # Collect embeddings across all test datasets
    all_imu = []
    all_text = []
    all_labels = []

    # Pre-compute text embeddings for all global labels once
    with torch.no_grad():
        global_label_embs = label_bank.encode(GLOBAL_LABELS, normalize=True)
        if global_label_embs.ndim == 3:
            global_label_embs = global_label_embs.mean(dim=1)
        global_label_embs = global_label_embs.cpu().numpy()
    global_label_to_idx = {l: i for i, l in enumerate(GLOBAL_LABELS)}

    for ds_name in TEST_DATASETS:
        ds_dir = TSFM_EVAL_DIR / ds_name
        if not ds_dir.exists() or not (ds_dir / "data_native.npy").exists():
            print(f"  SKIP {ds_name}: no preprocessed data")
            continue

        print(f"\n  Processing {ds_name}...")
        meta = get_dataset_metadata(ds_name)
        raw_data, raw_labels, sr = load_raw_data(ds_name)

        # Integer label indices -> string activity names
        window_label_indices = get_window_labels(raw_labels)
        ds_activities = get_dataset_labels(ds_name)

        # Extract IMU embeddings (session-level)
        imu_embs = extract_embeddings(
            model, raw_data, device, sr,
            meta['channel_descriptions'],
            has_gyro=meta['has_gyro'],
        )

        # Map integer indices -> activity names -> global label embeddings
        valid_idx = []
        valid_labels = []
        for i, idx in enumerate(window_label_indices):
            if idx < len(ds_activities):
                activity_name = ds_activities[idx]
                if activity_name in global_label_to_idx:
                    valid_idx.append(i)
                    valid_labels.append(activity_name)

        if not valid_idx:
            print(f"    No matching labels in label bank -- skipping")
            continue

        imu_embs = imu_embs[valid_idx]
        text_embs = np.array([global_label_embs[global_label_to_idx[l]] for l in valid_labels])

        print(f"    {len(valid_labels)} samples, {len(set(valid_labels))} unique labels")
        all_imu.append(imu_embs)
        all_text.append(text_embs)
        all_labels.extend(valid_labels)

    if not all_imu:
        print("ERROR: No data loaded. Check benchmark_data/processed/ exists.")
        sys.exit(1)

    imu_all = np.concatenate(all_imu, axis=0)
    text_all = np.concatenate(all_text, axis=0)
    print(f"\nTotal: {len(all_labels)} samples, {len(set(all_labels))} unique labels")

    # Generate figure
    output_path = OUTPUT_DIR / "fig_embedding_umap"
    make_paper_figure(imu_all, text_all, all_labels, output_path)
    print("\nDone!")


if __name__ == '__main__':
    main()
