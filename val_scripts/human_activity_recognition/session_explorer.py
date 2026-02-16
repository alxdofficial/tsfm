"""
Interactive session explorer for semantic alignment model.

Loads random sessions, displays info, runs inference,
and shows top-5 predictions with similarities and 3D visualizations.

Usage:
    # Edit CHECKPOINT_PATH below, then run:
    python val_scripts/human_activity_recognition/session_explorer.py
"""

import torch
from torch.amp import autocast
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for projection='3d'
from sklearn.decomposition import PCA
from pathlib import Path
import sys
import json
import random

# Add project root to path (val_scripts -> tsfm)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'tools' / 'models'))

from val_scripts.human_activity_recognition.evaluation_metrics import get_label_to_group_mapping

# =============================================================================
# CONFIGURATION - Edit these values instead of using CLI args
# =============================================================================

# Path to checkpoint
CHECKPOINT_PATH = "training_output/semantic_alignment/20260107_102025/epoch_60.pt"

# Output directory for plots
OUTPUT_DIR = "test_output/session_explorer"

# Number of sessions to explore
NUM_SESSIONS = 5

# Datasets to sample from
from val_scripts.human_activity_recognition.eval_config import (
    PATCH_SIZE_PER_DATASET, TRAINING_DATASETS,
)
EVAL_DATASETS = TRAINING_DATASETS[:6]  # uci_har, hhar, mhealth, pamap2, wisdm, unimib_shar

# Whether to pause between sessions (False for batch mode)
INTERACTIVE = False

# Label grouping: True = simplified (~12 groups), False = fine-grained (~25 groups)
USE_SIMPLE_GROUPS = False

# All unique labels (populated dynamically)
ALL_LABELS = None


# =============================================================================
# Model Loading
# =============================================================================


def load_model_and_data(checkpoint_path: str, device: torch.device):
    """Load model and validation data."""
    from datasets.imu_pretraining_dataset.multi_dataset_loader import create_dataloaders
    from val_scripts.human_activity_recognition.model_loading import (
        load_model as _load_model, load_label_bank,
    )

    model, checkpoint, hyperparams_path = _load_model(checkpoint_path, device)
    epoch = checkpoint.get('epoch', 'unknown')

    # Load validation data
    _, val_loader, _ = create_dataloaders(
        data_root='data',
        datasets=EVAL_DATASETS,
        batch_size=1,  # Single sample at a time
        patch_size_per_dataset=PATCH_SIZE_PER_DATASET,
        max_sessions_per_dataset=10000,
        num_workers=0,
    )

    label_bank = load_label_bank(checkpoint, device, hyperparams_path)

    # Collect all unique labels from the dataset
    global ALL_LABELS
    print("Collecting unique labels from dataset...")
    all_labels_set = set()
    for batch in val_loader:
        all_labels_set.update(batch['label_texts'])
    ALL_LABELS = sorted(list(all_labels_set))
    print(f"Found {len(ALL_LABELS)} unique labels")

    return model, val_loader, label_bank, epoch


# =============================================================================
# Session Exploration
# =============================================================================


def get_unique_labels_from_groups(label: str, top_labels: list) -> list:
    """Get top predictions with unique groups (allows ground truth group)."""
    label_to_group = get_label_to_group_mapping(use_simple=USE_SIMPLE_GROUPS)

    unique_labels = []
    seen_groups = set()  # Don't exclude ground truth group

    for lbl in top_labels:
        lbl_group = label_to_group.get(lbl, lbl)
        if lbl_group not in seen_groups:
            unique_labels.append(lbl)
            seen_groups.add(lbl_group)

    return unique_labels


def explore_session(
    model,
    batch,
    label_bank,
    device: torch.device,
    output_dir: Path = None,
    session_idx: int = 0
):
    """Explore a single session."""

    data = batch['data'].to(device)
    channel_mask = batch['channel_mask'].to(device)
    label_text = batch['label_texts'][0]
    metadata = batch['metadata'][0]

    # Extract info
    dataset_name = metadata.get('dataset', 'unknown')
    sampling_rate = metadata['sampling_rate_hz']
    patch_size = metadata['patch_size_sec']
    channel_descriptions = metadata['channel_descriptions']

    print("\n" + "="*70)
    print("SESSION INFO")
    print("="*70)
    print(f"Dataset:        {dataset_name}")
    print(f"Ground Truth:   {label_text}")
    print(f"Sampling Rate:  {sampling_rate} Hz")
    print(f"Patch Size:     {patch_size} sec")
    print(f"Channels:       {len(channel_descriptions)}")
    print(f"Data Shape:     {data.shape}")
    print(f"Channels:       {', '.join(channel_descriptions[:6])}...")

    # Run inference (use autocast to match training precision)
    with torch.no_grad():
        with autocast('cuda', enabled=device.type == 'cuda'):
            imu_emb = model.forward_from_raw(
                data,
                [channel_descriptions],
                channel_mask,
                [sampling_rate],
                [patch_size]
            )

    # Encode all labels
    all_labels = ALL_LABELS  # Already unique and sorted
    all_text_embs = label_bank.encode(all_labels, normalize=True)  # (L, D) or (L, K, D)

    # Compute similarities (handles multi-prototype)
    if all_text_embs.dim() == 3:
        similarities = torch.einsum('nd,lkd->nlk', imu_emb, all_text_embs).max(dim=-1).values.squeeze(0)
    else:
        similarities = torch.matmul(imu_emb, all_text_embs.T).squeeze(0)

    # Get top predictions
    sorted_indices = torch.argsort(similarities, descending=True)
    top_labels = [all_labels[i] for i in sorted_indices[:20]]

    # Get unique top 5 (filtering by group)
    unique_top = get_unique_labels_from_groups(label_text, top_labels)[:5]

    print("\n" + "-"*70)
    print("TOP 5 PREDICTIONS (unique groups)")
    print("-"*70)

    label_to_group = get_label_to_group_mapping(use_simple=USE_SIMPLE_GROUPS)
    gt_group = label_to_group.get(label_text, label_text)

    for i, lbl in enumerate(unique_top):
        idx = all_labels.index(lbl)
        sim = similarities[idx].item()
        lbl_group = label_to_group.get(lbl, lbl)
        match = " <-- CORRECT" if lbl_group == gt_group else ""
        print(f"  {i+1}. {lbl:30s} (group: {lbl_group:20s}) sim: {sim:.4f}{match}")

    # Get ground truth similarity
    gt_idx = all_labels.index(label_text) if label_text in all_labels else -1
    gt_sim = similarities[gt_idx].item() if gt_idx >= 0 else 0.0
    print(f"\n  Ground truth '{label_text}' similarity: {gt_sim:.4f}")

    # Random 5 other labels (not in top predictions)
    remaining_labels = [l for l in all_labels if l not in top_labels[:10]]
    random_labels = random.sample(remaining_labels, min(5, len(remaining_labels)))

    print("\n" + "-"*70)
    print("RANDOM 5 OTHER LABELS (for comparison)")
    print("-"*70)

    for lbl in random_labels:
        idx = all_labels.index(lbl)
        sim = similarities[idx].item()
        lbl_group = label_to_group.get(lbl, lbl)
        print(f"  - {lbl:30s} (group: {lbl_group:20s}) sim: {sim:.4f}")

    # Create 3D visualization
    if output_dir:
        filename = f'session_{session_idx:03d}_{label_text}.png'
        # Add data shape to metadata for plotting
        plot_metadata = dict(metadata)
        plot_metadata['data_shape'] = list(data.shape)
        create_session_3d_plot(
            imu_emb.detach().cpu().numpy(),
            all_text_embs.detach().cpu().numpy(),
            all_labels,
            label_text,
            unique_top + random_labels,
            output_dir / filename,
            metadata=plot_metadata
        )

    return {
        'dataset': dataset_name,
        'ground_truth': label_text,
        'top_predictions': unique_top,
        'ground_truth_similarity': gt_sim,
        'top_similarities': {lbl: similarities[all_labels.index(lbl)].item() for lbl in unique_top}
    }


def create_session_3d_plot(
    imu_emb: np.ndarray,
    text_embs: np.ndarray,
    all_labels: list,
    ground_truth: str,
    selected_labels: list,
    output_path: str,
    metadata: dict = None
):
    """Create 3D plot of session embedding with selected text embeddings."""

    # Filter to only labels that exist in all_labels
    valid_labels = [l for l in selected_labels if l in all_labels]
    selected_indices = [all_labels.index(l) for l in valid_labels]
    selected_text_embs = text_embs[selected_indices]

    # Combine IMU and text embeddings for PCA
    combined = np.vstack([imu_emb, selected_text_embs])

    # Fit PCA
    pca = PCA(n_components=3)
    combined_3d = pca.fit_transform(combined)

    imu_3d = combined_3d[0]
    text_3d = combined_3d[1:]

    # Create plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot IMU embedding (large star)
    ax.scatter(
        [imu_3d[0]], [imu_3d[1]], [imu_3d[2]],
        c='red', s=300, marker='*', label=f'IMU ({ground_truth})',
        edgecolors='black', linewidths=1
    )

    # Color scheme
    label_to_group = get_label_to_group_mapping(use_simple=USE_SIMPLE_GROUPS)
    gt_group = label_to_group.get(ground_truth, ground_truth)

    # Determine which of the original selected_labels were top-5 predictions
    top5_labels = selected_labels[:5]

    colors = []
    for lbl in valid_labels:
        lbl_group = label_to_group.get(lbl, lbl)
        if lbl_group == gt_group:
            colors.append('green')  # Same group as ground truth
        elif lbl in top5_labels:
            colors.append('blue')  # Top predictions
        else:
            colors.append('gray')  # Random labels

    # Plot text embeddings
    for i, (x, y, z) in enumerate(text_3d):
        lbl = valid_labels[i]
        ax.scatter([x], [y], [z], c=colors[i], s=100, marker='o', alpha=0.8)
        ax.text(x, y, z, f'  {lbl}', fontsize=8)

    # Draw lines from IMU to each text embedding
    for i, (x, y, z) in enumerate(text_3d):
        ax.plot(
            [imu_3d[0], x], [imu_3d[1], y], [imu_3d[2], z],
            c=colors[i], alpha=0.3, linestyle='--'
        )

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    # Build title with metadata
    title = f'Session Embedding: {ground_truth}\n(Red star = IMU, Green = same group, Blue = top preds, Gray = random)'
    ax.set_title(title, fontsize=12)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=15, label='IMU Embedding'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Same Group'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Top Predictions'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Random Labels'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    # Add metadata text box
    if metadata:
        dataset = metadata.get('dataset', 'unknown')
        sampling_rate = metadata.get('sampling_rate_hz', '?')
        num_channels = len(metadata.get('channel_descriptions', []))
        data_shape = metadata.get('data_shape', '?')
        session_len_samples = data_shape[1] if isinstance(data_shape, (list, tuple)) and len(data_shape) > 1 else '?'
        session_len_sec = session_len_samples / sampling_rate if isinstance(session_len_samples, (int, float)) and isinstance(sampling_rate, (int, float)) else '?'

        info_text = (
            f"Dataset: {dataset}\n"
            f"Ground Truth: {ground_truth}\n"
            f"Sampling Rate: {sampling_rate} Hz\n"
            f"Channels: {num_channels}\n"
            f"Session Length: {session_len_samples} samples ({session_len_sec:.1f}s)" if isinstance(session_len_sec, float) else
            f"Dataset: {dataset}\n"
            f"Ground Truth: {ground_truth}\n"
            f"Sampling Rate: {sampling_rate} Hz\n"
            f"Channels: {num_channels}\n"
            f"Session Length: {session_len_samples} samples"
        )

        # Add text box in figure coordinates (bottom left)
        fig.text(0.02, 0.02, info_text, fontsize=9, family='monospace',
                 verticalalignment='bottom', horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved 3D visualization to {output_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Num sessions: {NUM_SESSIONS}")

    # Load model and data
    model, val_loader, label_bank, epoch = load_model_and_data(CHECKPOINT_PATH, device)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Explore sessions
    val_iter = iter(val_loader)
    for i in range(NUM_SESSIONS):
        try:
            batch = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            batch = next(val_iter)

        explore_session(model, batch, label_bank, device, output_dir, session_idx=i)

        if i < NUM_SESSIONS - 1 and INTERACTIVE:
            input("\nPress Enter for next session...")

    print(f"\n{'='*70}")
    print(f"Exploration complete. {NUM_SESSIONS} session plots saved to {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
