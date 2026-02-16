"""
3D visualization of semantic embeddings across training epochs.

Creates a 2x2 grid showing embedding space evolution with:
- Consistent PCA axes computed from final epoch
- Both IMU and text embeddings color-coded by activity class
- Activity groups for cleaner visualization

Usage:
    # Edit RUN_DIR and EPOCHS below, then run:
    python val_scripts/human_activity_recognition/visualization_3d.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for projection='3d'
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple
from pathlib import Path
import sys
import json

# Add project root to path (val_scripts -> tsfm)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'tools' / 'models'))

from val_scripts.human_activity_recognition.evaluation_metrics import get_label_to_group_mapping

# =============================================================================
# CONFIGURATION - Edit these values instead of using CLI args
# =============================================================================

# Training run directory containing checkpoints
RUN_DIR = "training_output/semantic_alignment/20260107_102025"

# Epochs to visualize (2x2 grid, so typically 4 epochs)
EPOCHS = [15, 30, 45, 60]

# Output directory for plots
OUTPUT_DIR = "test_output/visualization_3d"

# Number of samples to visualize (for clarity)
SAMPLE_SIZE = 1000

# Datasets to use for evaluation
from val_scripts.human_activity_recognition.eval_config import TRAINING_DATASETS
EVAL_DATASETS = TRAINING_DATASETS[:6]  # uci_har, hhar, mhealth, pamap2, wisdm, unimib_shar

# Label grouping: True = simplified (~12 groups), False = fine-grained (~25 groups)
USE_SIMPLE_GROUPS = False

# =============================================================================
# Constants
# =============================================================================

# Color scheme for activity groups
ACTIVITY_COLORS = {
    'walking': '#2ecc71',       # green
    'running': '#e74c3c',       # red
    'ascending_stairs': '#3498db',  # blue
    'descending_stairs': '#9b59b6', # purple
    'sitting': '#f39c12',       # orange
    'standing': '#1abc9c',      # teal
    'lying': '#34495e',         # dark gray
    'falling': '#e91e63',       # pink
    'jumping': '#ff5722',       # deep orange
    'eating': '#795548',        # brown
    'other': '#95a5a6',         # gray for ungrouped activities
}


# =============================================================================
# Model Loading
# =============================================================================


def load_checkpoint_embeddings(
    checkpoint_path: str,
    val_loader,
    label_bank,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Load a checkpoint and extract embeddings from validation set.

    Returns:
        imu_embeddings: (N, D) tensor
        text_embeddings: (N, D) tensor
        labels: List of label strings
    """
    from val_scripts.human_activity_recognition.model_loading import load_model

    model, checkpoint, _ = load_model(checkpoint_path, device)
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Loading epoch {epoch} from {checkpoint_path}")

    # Collect embeddings
    all_imu_embeddings = []
    all_text_embeddings = []
    all_labels = []

    total_batches = len(val_loader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            data = batch['data'].to(device)
            channel_mask = batch['channel_mask'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label_texts = batch['label_texts']
            metadata = batch['metadata']

            sampling_rates = [m['sampling_rate_hz'] for m in metadata]
            patch_sizes = [m['patch_size_sec'] for m in metadata]
            channel_descriptions = [m['channel_descriptions'] for m in metadata]

            imu_emb = model.forward_from_raw(data, channel_descriptions, channel_mask, sampling_rates, patch_sizes,
                                            attention_mask=attention_mask)
            text_emb = label_bank.encode(label_texts, normalize=True)
            # Handle multi-prototype: mean across prototypes for visualization
            if text_emb.dim() == 3:
                text_emb = text_emb.mean(dim=1)  # (B, K, D) -> (B, D)

            all_imu_embeddings.append(imu_emb.cpu())
            all_text_embeddings.append(text_emb.cpu())
            all_labels.extend(label_texts)

            # Progress indicator
            if (batch_idx + 1) % 25 == 0 or batch_idx == total_batches - 1:
                pct = (batch_idx + 1) / total_batches * 100
                print(f"    [{batch_idx + 1}/{total_batches}] {pct:.0f}%", end='\r')

    print()  # New line after progress
    all_imu_embeddings = torch.cat(all_imu_embeddings, dim=0)
    all_text_embeddings = torch.cat(all_text_embeddings, dim=0)

    return all_imu_embeddings, all_text_embeddings, all_labels


from val_scripts.human_activity_recognition.model_loading import load_label_bank  # noqa: E302


# =============================================================================
# PCA and Visualization
# =============================================================================


def compute_consistent_pca(
    reference_embeddings: np.ndarray,
    n_components: int = 3
) -> PCA:
    """
    Fit PCA on reference embeddings (typically final epoch).
    This PCA will be used to project all epochs for consistent visualization.
    """
    pca = PCA(n_components=n_components)
    pca.fit(reference_embeddings)
    print(f"PCA explained variance ratios: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_)*100:.1f}%")
    return pca


def get_label_color(label: str, label_to_group: Dict[str, str]) -> str:
    """Get color for a label based on its group."""
    group = label_to_group.get(label, 'other')
    return ACTIVITY_COLORS.get(group, ACTIVITY_COLORS['other'])


def build_group_labels_text(labels: List[str], label_to_group: Dict[str, str]) -> str:
    """
    Build a text showing which labels belong to which groups.
    Only includes groups that are actually present in the data.
    """
    from collections import defaultdict

    # Group labels by their group
    group_to_labels = defaultdict(set)
    for label in set(labels):
        group = label_to_group.get(label, label)
        group_to_labels[group].add(label)

    # Build text, sorted by group name
    lines = []
    for group in sorted(group_to_labels.keys()):
        group_labels = sorted(group_to_labels[group])
        # Show group and its member labels
        if len(group_labels) == 1 and group_labels[0] == group:
            # Single label that is its own group
            lines.append(f"{group}")
        else:
            # Group with multiple labels or different name
            labels_str = ', '.join(group_labels)
            lines.append(f"{group}: {labels_str}")

    return '\n'.join(lines)


def create_3d_visualization(
    epochs_data: Dict[int, Tuple[np.ndarray, np.ndarray, List[str]]],
    pca: PCA,
    output_path: str,
    show_text_embeddings: bool = True,
    sample_size: int = 1000
):
    """
    Create a 2x2 grid of 3D plots showing embedding evolution.

    Args:
        epochs_data: Dict mapping epoch -> (imu_embeddings, text_embeddings, labels)
        pca: Fitted PCA for consistent projection
        output_path: Where to save the figure
        show_text_embeddings: Whether to overlay text embeddings
        sample_size: Number of samples to visualize (for clarity)
    """
    label_to_group = get_label_to_group_mapping(use_simple=USE_SIMPLE_GROUPS)
    epochs = sorted(epochs_data.keys())

    fig = plt.figure(figsize=(16, 14))

    for idx, epoch in enumerate(epochs):
        imu_emb, text_emb, labels = epochs_data[epoch]

        # Subsample for visualization
        if len(labels) > sample_size:
            indices = np.random.choice(len(labels), sample_size, replace=False)
            imu_emb = imu_emb[indices]
            text_emb = text_emb[indices]
            labels = [labels[i] for i in indices]

        # Project to 3D using consistent PCA
        imu_3d = pca.transform(imu_emb)
        text_3d = pca.transform(text_emb)

        # Get colors for each point
        colors = [get_label_color(l, label_to_group) for l in labels]

        # Create subplot
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')

        # Plot IMU embeddings (solid circles)
        ax.scatter(
            imu_3d[:, 0], imu_3d[:, 1], imu_3d[:, 2],
            c=colors, alpha=0.6, s=20, label='IMU'
        )

        # Overlay text embeddings (hollow circles) if requested
        if show_text_embeddings:
            ax.scatter(
                text_3d[:, 0], text_3d[:, 1], text_3d[:, 2],
                c=colors, alpha=0.3, s=30, marker='^', label='Text'
            )

        ax.set_title(f'Epoch {epoch}', fontsize=14, fontweight='bold')
        ax.set_xlabel('PC1', fontsize=10)
        ax.set_ylabel('PC2', fontsize=10)
        ax.set_zlabel('PC3', fontsize=10)

        # Set consistent axis limits across all plots
        # We'll compute this after collecting all data

    # Compute consistent axis limits from all epochs
    all_points = []
    for epoch in epochs:
        imu_emb, text_emb, _ = epochs_data[epoch]
        imu_3d = pca.transform(imu_emb[:sample_size] if len(imu_emb) > sample_size else imu_emb)
        all_points.append(imu_3d)
    all_points = np.vstack(all_points)

    margin = 0.1
    xlim = (all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
    ylim = (all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
    zlim = (all_points[:, 2].min() - margin, all_points[:, 2].max() + margin)

    # Apply limits to all subplots
    for idx in range(len(epochs)):
        ax = fig.axes[idx]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

    # Create legend with group colors
    legend_elements = []
    for group, color in ACTIVITY_COLORS.items():
        if group != 'other':
            from matplotlib.lines import Line2D
            legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=color, markersize=10, label=group))

    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99),
               ncol=2, fontsize=9)

    # Add text box showing label-to-group mapping
    # Get all labels from the final epoch
    final_epoch = max(epochs_data.keys())
    all_labels = epochs_data[final_epoch][2]
    group_text = build_group_labels_text(all_labels, label_to_group)

    fig.text(0.01, 0.01, f"Label Groups:\n{group_text}", fontsize=7, family='monospace',
             verticalalignment='bottom', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.suptitle('Semantic Embedding Space Evolution During Training',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved 3D visualization to {output_path}")
    plt.close()


def create_2d_projections(
    epochs_data: Dict[int, Tuple[np.ndarray, np.ndarray, List[str]]],
    pca: PCA,
    output_path: str,
    sample_size: int = 1000
):
    """
    Create 2D projections (PC1 vs PC2, PC1 vs PC3, PC2 vs PC3) for each epoch.
    """
    label_to_group = get_label_to_group_mapping(use_simple=USE_SIMPLE_GROUPS)
    epochs = sorted(epochs_data.keys())

    fig, axes = plt.subplots(len(epochs), 3, figsize=(18, 4 * len(epochs)))

    projections = [(0, 1, 'PC1', 'PC2'), (0, 2, 'PC1', 'PC3'), (1, 2, 'PC2', 'PC3')]

    for row, epoch in enumerate(epochs):
        imu_emb, text_emb, labels = epochs_data[epoch]

        # Subsample
        if len(labels) > sample_size:
            indices = np.random.choice(len(labels), sample_size, replace=False)
            imu_emb = imu_emb[indices]
            labels = [labels[i] for i in indices]

        # Project to 3D
        imu_3d = pca.transform(imu_emb)
        colors = [get_label_color(l, label_to_group) for l in labels]

        for col, (i, j, xlabel, ylabel) in enumerate(projections):
            ax = axes[row, col] if len(epochs) > 1 else axes[col]
            ax.scatter(imu_3d[:, i], imu_3d[:, j], c=colors, alpha=0.5, s=15)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if col == 1:
                ax.set_title(f'Epoch {epoch}', fontsize=12, fontweight='bold')

    # Add text box showing label-to-group mapping
    final_epoch = max(epochs_data.keys())
    all_labels = epochs_data[final_epoch][2]
    group_text = build_group_labels_text(all_labels, label_to_group)

    fig.text(0.01, 0.01, f"Label Groups:\n{group_text}", fontsize=7, family='monospace',
             verticalalignment='bottom', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.suptitle('2D Projections of Embedding Space', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved 2D projections to {output_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================


def main():
    """Main visualization script."""
    from datasets.imu_pretraining_dataset.multi_dataset_loader import create_dataloaders

    run_dir = Path(RUN_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory: {run_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs to visualize: {EPOCHS}")
    print(f"Sample size: {SAMPLE_SIZE}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load validation data loader
    _, val_loader, _ = create_dataloaders(
        data_root='data',
        datasets=EVAL_DATASETS,
        batch_size=64,
        max_sessions_per_dataset=10000,
        num_workers=4
    )

    # Load label_bank from the final epoch checkpoint
    final_epoch = max(EPOCHS)
    final_checkpoint_path = run_dir / f'epoch_{final_epoch}.pt'
    hyperparams_path = run_dir / 'hyperparameters.json'

    if final_checkpoint_path.exists():
        final_checkpoint = torch.load(final_checkpoint_path, map_location='cpu')
    else:
        # Fall back to any available checkpoint
        final_checkpoint = {}
        for ep in sorted(EPOCHS, reverse=True):
            cp_path = run_dir / f'epoch_{ep}.pt'
            if cp_path.exists():
                final_checkpoint = torch.load(cp_path, map_location='cpu')
                break

    label_bank = load_label_bank(final_checkpoint, device, hyperparams_path)

    # Load embeddings for each epoch
    epochs_data = {}
    for epoch in EPOCHS:
        checkpoint_path = run_dir / f'epoch_{epoch}.pt'
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint for epoch {epoch} not found at {checkpoint_path}")
            continue

        imu_emb, text_emb, labels = load_checkpoint_embeddings(
            str(checkpoint_path), val_loader, label_bank, device
        )
        epochs_data[epoch] = (imu_emb.numpy(), text_emb.numpy(), labels)
        print(f"  Loaded {len(labels)} samples for epoch {epoch}")

    if not epochs_data:
        print("Error: No checkpoints found!")
        return

    # Fit PCA on final epoch (reference for consistent axes)
    final_epoch = max(epochs_data.keys())
    print(f"\nFitting PCA on epoch {final_epoch} embeddings...")
    reference_imu = epochs_data[final_epoch][0]
    pca = compute_consistent_pca(reference_imu)

    # Create visualizations
    print("\nCreating 3D visualization...")
    create_3d_visualization(
        epochs_data, pca,
        output_path=str(output_dir / 'embedding_evolution_3d.png'),
        sample_size=SAMPLE_SIZE
    )

    print("Creating 2D projections...")
    create_2d_projections(
        epochs_data, pca,
        output_path=str(output_dir / 'embedding_evolution_2d.png'),
        sample_size=SAMPLE_SIZE
    )

    print(f"\n{'='*70}")
    print(f"Visualization complete. Plots saved to {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
