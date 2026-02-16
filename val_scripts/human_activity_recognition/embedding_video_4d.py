"""
4D Embedding Visualization: Creates a rotating 3D video of IMU-text embedding alignment.

Generates a video (MP4/GIF) where each frame shows a 3D projection of:
- IMU embeddings (circles) colored by activity label
- Text embeddings (triangles) for the same activities

The camera rotates around the 3D embedding space to create the 4D effect.

Usage:
    # Edit CHECKPOINT_PATH below, then run:
    python val_scripts/human_activity_recognition/embedding_video_4d.py
"""

import torch
from torch.amp import autocast
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for projection='3d'
import matplotlib.animation as animation
from sklearn.decomposition import PCA
from pathlib import Path
import sys
import json
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# Add project root to path (val_scripts -> tsfm)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'tools' / 'models'))

# =============================================================================
# CONFIGURATION - Edit these values instead of using CLI args
# =============================================================================

# Path to checkpoint
CHECKPOINT_PATH = "training_output/semantic_alignment/20260107_102025/epoch_60.pt"

# Output directory for video
OUTPUT_DIR = "test_output/embedding_video"

# Datasets to visualize
DATASETS_TO_USE = ['uci_har', 'hhar', 'mhealth', 'pamap2', 'wisdm', 'unimib_shar']

# Patch size per dataset - MUST match training config
PATCH_SIZE_PER_DATASET = {
    'uci_har': 1.0,
    'hhar': 1.0,
    'mhealth': 2.0,
    'pamap2': 2.0,
    'wisdm': 2.0,
    'unimib_shar': 1.0,
    'motionsense': 2.0,  # For zero-shot
}

# Video settings
VIDEO_FORMAT = 'mp4'  # 'mp4' or 'gif'
VIDEO_FPS = 30
VIDEO_DURATION_SEC = 10  # Total duration of rotation
VIDEO_DPI = 150

# Visualization settings
MAX_SAMPLES_PER_LABEL = 50  # Limit samples per label to avoid clutter
USE_UMAP = True  # False = PCA (faster), True = UMAP (slower but often better)
POINT_SIZE_IMU = 30
POINT_SIZE_TEXT = 100


# =============================================================================
# Model Loading (identical to compare_models.py)
# =============================================================================


def load_model(checkpoint_path: str, device: torch.device) -> Tuple:
    """
    Load model from checkpoint with architecture from hyperparameters.json.

    Returns:
        model: The loaded SemanticAlignmentModel
        checkpoint: Full checkpoint dict (for loading label_bank state)
        hyperparams_path: Path to hyperparameters.json
    """
    from imu_activity_recognition_encoder.encoder import IMUActivityRecognitionEncoder
    from imu_activity_recognition_encoder.semantic_alignment import SemanticAlignmentHead
    from training_scripts.human_activity_recognition.semantic_alignment_train import SemanticAlignmentModel

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Loaded checkpoint from epoch {epoch}")

    # Load hyperparameters from checkpoint directory
    hyperparams_path = checkpoint_path.parent / 'hyperparameters.json'
    if hyperparams_path.exists():
        with open(hyperparams_path) as f:
            hyperparams = json.load(f)
        enc_cfg = hyperparams.get('encoder', {})
        head_cfg = hyperparams.get('semantic_head', {})
        token_cfg = hyperparams.get('token_level_text', {})
        print(f"Loaded architecture from {hyperparams_path}")
    else:
        raise FileNotFoundError(
            f"hyperparameters.json not found at {hyperparams_path}. "
            "This checkpoint may be from an older incompatible version."
        )

    # Create encoder
    encoder = IMUActivityRecognitionEncoder(
        d_model=enc_cfg.get('d_model', 384),
        num_heads=enc_cfg.get('num_heads', 8),
        num_temporal_layers=enc_cfg.get('num_temporal_layers', 4),
        dim_feedforward=enc_cfg.get('dim_feedforward', 1536),
        dropout=enc_cfg.get('dropout', 0.1),
        use_cross_channel=enc_cfg.get('use_cross_channel', True),
        cnn_channels=enc_cfg.get('cnn_channels', [32, 64]),
        cnn_kernel_sizes=enc_cfg.get('cnn_kernel_sizes', [5]),
        target_patch_size=enc_cfg.get('target_patch_size', 64),
        use_channel_encoding=enc_cfg.get('use_channel_encoding', False)
    )

    # Create semantic head
    semantic_head = SemanticAlignmentHead(
        d_model=enc_cfg.get('d_model', 384),
        d_model_fused=384,
        output_dim=384,
        num_temporal_layers=head_cfg.get('num_temporal_layers', 2),
        num_heads=enc_cfg.get('num_heads', 8),
        dim_feedforward=enc_cfg.get('dim_feedforward', 1536),
        dropout=enc_cfg.get('dropout', 0.1),
        num_fusion_queries=head_cfg.get('num_fusion_queries', 4),
        use_fusion_self_attention=head_cfg.get('use_fusion_self_attention', True),
        num_pool_queries=head_cfg.get('num_pool_queries', 4),
        use_pool_self_attention=head_cfg.get('use_pool_self_attention', True)
    )

    # Create full model with token-level text encoding
    model = SemanticAlignmentModel(
        encoder,
        semantic_head,
        num_heads=token_cfg.get('num_heads', 4),
        dropout=enc_cfg.get('dropout', 0.1)
    )

    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint['model_state_dict'], strict=False
    )
    if unexpected_keys:
        other_unexpected = [k for k in unexpected_keys if 'channel_encoding' not in k]
        if other_unexpected:
            print(f"  Warning: Unexpected keys: {other_unexpected[:5]}...")
    if missing_keys:
        print(f"  Warning: Missing keys: {missing_keys[:5]}...")

    model.eval()
    model = model.to(device)

    return model, checkpoint, hyperparams_path


def load_label_bank(checkpoint: dict, device: torch.device, hyperparams_path: Path):
    """Load LearnableLabelBank with trained state from checkpoint."""
    from imu_activity_recognition_encoder.token_text_encoder import LearnableLabelBank

    # Get config from hyperparameters
    if hyperparams_path.exists():
        with open(hyperparams_path) as f:
            hyperparams = json.load(f)
        token_cfg = hyperparams.get('token_level_text', {})
    else:
        token_cfg = {}

    label_bank = LearnableLabelBank(
        device=device,
        num_heads=token_cfg.get('num_heads', 4),
        num_queries=token_cfg.get('num_queries', 4),
        num_prototypes=token_cfg.get('num_prototypes', 1),
        dropout=0.1
    )

    # Load trained weights if available
    if 'label_bank_state_dict' in checkpoint:
        label_bank.load_state_dict(checkpoint['label_bank_state_dict'])
        print("Loaded trained LearnableLabelBank from checkpoint")
    else:
        print("Warning: No label_bank_state_dict in checkpoint, using untrained LearnableLabelBank")

    label_bank.eval()
    return label_bank


# =============================================================================
# Data Collection
# =============================================================================


def collect_embeddings(
    model,
    label_bank,
    dataloader,
    device: torch.device,
    max_samples_per_label: int = 50
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Collect IMU and text embeddings from validation data.

    Returns:
        imu_embeddings: (N, D) numpy array
        text_embeddings: (N, D) numpy array
        labels: List of N label strings
    """
    all_imu_embeddings = []
    all_labels = []
    label_counts = {}

    print("Collecting embeddings...")
    for batch in tqdm(dataloader, desc="Processing batches"):
        data = batch['data'].to(device)
        channel_mask = batch['channel_mask'].to(device)
        metadata_list = batch['metadata']
        label_texts = batch['label_texts']

        # Check if we should skip (label quota reached)
        skip_batch = True
        for label in label_texts:
            if label_counts.get(label, 0) < max_samples_per_label:
                skip_batch = False
                break

        if skip_batch:
            continue

        # Extract per-sample info
        sampling_rates = [m['sampling_rate_hz'] for m in metadata_list]
        patch_sizes = [m['patch_size_sec'] for m in metadata_list]
        channel_descriptions_list = [m['channel_descriptions'] for m in metadata_list]

        # Run inference with autocast (matching training)
        with torch.no_grad():
            with autocast('cuda', enabled=device.type == 'cuda'):
                imu_emb = model(
                    data,
                    channel_descriptions_list,
                    channel_mask,
                    sampling_rates,
                    patch_sizes
                )

        # Collect per-sample
        for i, label in enumerate(label_texts):
            if label_counts.get(label, 0) < max_samples_per_label:
                all_imu_embeddings.append(imu_emb[i].detach().cpu().numpy())
                all_labels.append(label)
                label_counts[label] = label_counts.get(label, 0) + 1

    # Convert to arrays
    imu_embeddings = np.stack(all_imu_embeddings, axis=0)

    # Encode unique labels
    unique_labels = sorted(set(all_labels))
    print(f"Found {len(unique_labels)} unique labels, {len(all_labels)} total samples")

    text_embeddings_dict = {}
    for label in unique_labels:
        text_emb = label_bank.encode([label], normalize=True)
        # Handle multi-prototype: mean across prototypes for visualization
        if text_emb.dim() == 3:
            text_emb = text_emb.mean(dim=1)  # (1, K, D) -> (1, D)
        text_embeddings_dict[label] = text_emb.detach().cpu().numpy().squeeze(0)

    # Create text embeddings array matching IMU order
    text_embeddings = np.stack([text_embeddings_dict[l] for l in all_labels], axis=0)

    return imu_embeddings, text_embeddings, all_labels


# =============================================================================
# 3D Dimensionality Reduction
# =============================================================================


def reduce_to_3d(
    imu_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    use_umap: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce embeddings to 3D using PCA or UMAP.

    Returns:
        imu_3d: (N, 3) array
        text_3d: (N, 3) array
    """
    # Combine for joint reduction
    combined = np.vstack([imu_embeddings, text_embeddings])
    n_imu = len(imu_embeddings)

    if use_umap:
        try:
            import umap
            print("Computing UMAP reduction to 3D...")
            reducer = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                n_components=3,
                metric='cosine',
                random_state=42
            )
            combined_3d = reducer.fit_transform(combined)
        except ImportError:
            print("UMAP not installed, falling back to PCA")
            use_umap = False

    if not use_umap:
        print("Computing PCA reduction to 3D...")
        pca = PCA(n_components=3, random_state=42)
        combined_3d = pca.fit_transform(combined)
        print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    imu_3d = combined_3d[:n_imu]
    text_3d = combined_3d[n_imu:]

    return imu_3d, text_3d


# =============================================================================
# Video Generation
# =============================================================================


def create_3d_frame(
    ax,
    imu_3d: np.ndarray,
    text_3d: np.ndarray,
    labels: List[str],
    elev: float,
    azim: float,
    epoch: int
):
    """Create a single 3D frame with given camera angle."""
    ax.clear()

    # Get unique labels and create color mapping
    unique_labels = sorted(set(labels))
    if len(unique_labels) <= 10:
        cmap = plt.cm.tab10
    else:
        cmap = plt.cm.tab20

    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    # Plot IMU embeddings (circles)
    for label in unique_labels:
        idx = label_to_idx[label]
        color = cmap(idx / len(unique_labels))
        mask = np.array([l == label for l in labels])

        ax.scatter(
            imu_3d[mask, 0], imu_3d[mask, 1], imu_3d[mask, 2],
            c=[color], marker='o', s=POINT_SIZE_IMU, alpha=0.6,
            label=f'{label[:15]}...' if len(label) > 15 else label
        )

        # Plot text embedding centroid for this label (single triangle per label)
        text_centroid = text_3d[mask].mean(axis=0)
        ax.scatter(
            [text_centroid[0]], [text_centroid[1]], [text_centroid[2]],
            c=[color], marker='^', s=POINT_SIZE_TEXT, alpha=0.9,
            edgecolors='black', linewidths=0.5
        )

    # Set view angle
    ax.view_init(elev=elev, azim=azim)

    # Labels and title
    ax.set_xlabel('Dim 1', fontsize=10)
    ax.set_ylabel('Dim 2', fontsize=10)
    ax.set_zlabel('Dim 3', fontsize=10)
    ax.set_title(
        f'IMU-Text Embedding Alignment (Epoch {epoch})\n'
        f'Circles: IMU samples | Triangles: Text label centroids',
        fontsize=12, fontweight='bold'
    )

    # Legend (limit to fit)
    if len(unique_labels) <= 12:
        ax.legend(loc='upper left', fontsize=7, framealpha=0.8)


def create_video(
    imu_3d: np.ndarray,
    text_3d: np.ndarray,
    labels: List[str],
    output_path: Path,
    epoch: int,
    fps: int = 30,
    duration_sec: float = 10,
    video_format: str = 'mp4',
    dpi: int = 150
):
    """
    Create a rotating 3D video.

    Args:
        imu_3d: (N, 3) IMU embeddings
        text_3d: (N, 3) text embeddings
        labels: List of N labels
        output_path: Where to save the video
        epoch: Epoch number for title
        fps: Frames per second
        duration_sec: Video duration in seconds
        video_format: 'mp4' or 'gif'
        dpi: Resolution
    """
    n_frames = int(fps * duration_sec)

    # Set up figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Animation parameters
    elev_start = 20
    elev_end = 40
    azim_start = 0
    azim_end = 360  # Full rotation

    def update(frame):
        """Update function for animation."""
        progress = frame / n_frames

        # Smooth height change
        elev = elev_start + (elev_end - elev_start) * np.sin(progress * np.pi)
        # Continuous rotation
        azim = azim_start + (azim_end - azim_start) * progress

        create_3d_frame(ax, imu_3d, text_3d, labels, elev, azim, epoch)
        return []

    print(f"Generating {n_frames} frames at {fps} FPS ({duration_sec}s duration)...")

    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=1000/fps, blit=False
    )

    # Save based on format
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if video_format == 'mp4':
        output_file = output_path.with_suffix('.mp4')
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        print(f"Saving MP4 to {output_file}...")
        try:
            anim.save(str(output_file), writer=writer, dpi=dpi)
            print(f"Saved: {output_file}")
        except Exception as e:
            print(f"MP4 save failed ({e}), trying GIF fallback...")
            video_format = 'gif'

    if video_format == 'gif':
        output_file = output_path.with_suffix('.gif')
        print(f"Saving GIF to {output_file}...")
        # Reduce frames for GIF (smaller file)
        anim_gif = animation.FuncAnimation(
            fig, update, frames=min(n_frames, 120),  # Limit GIF frames
            interval=1000/fps, blit=False
        )
        anim_gif.save(str(output_file), writer='pillow', fps=min(fps, 15), dpi=dpi//2)
        print(f"Saved: {output_file}")

    plt.close(fig)
    return output_file


# =============================================================================
# Main
# =============================================================================


def main():
    """Main entry point."""
    from datasets.imu_pretraining_dataset.multi_dataset_loader import create_dataloaders

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model (identical to compare_models.py)
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    model, checkpoint, hyperparams_path = load_model(CHECKPOINT_PATH, device)
    label_bank = load_label_bank(checkpoint, device, hyperparams_path)
    epoch = checkpoint.get('epoch', 'unknown')

    # Load validation data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    _, val_loader, _ = create_dataloaders(
        data_root='data',
        datasets=DATASETS_TO_USE,
        batch_size=32,
        patch_size_per_dataset=PATCH_SIZE_PER_DATASET,
        max_sessions_per_dataset=10000,
        num_workers=4,
    )

    # Collect embeddings
    print("\n" + "="*70)
    print("COLLECTING EMBEDDINGS")
    print("="*70)
    imu_embeddings, text_embeddings, labels = collect_embeddings(
        model, label_bank, val_loader, device,
        max_samples_per_label=MAX_SAMPLES_PER_LABEL
    )

    # Reduce to 3D
    print("\n" + "="*70)
    print("DIMENSIONALITY REDUCTION")
    print("="*70)
    imu_3d, text_3d = reduce_to_3d(imu_embeddings, text_embeddings, use_umap=USE_UMAP)

    # Generate video
    print("\n" + "="*70)
    print("GENERATING VIDEO")
    print("="*70)
    output_path = output_dir / f'embedding_video_epoch_{epoch}'
    video_file = create_video(
        imu_3d, text_3d, labels, output_path,
        epoch=epoch,
        fps=VIDEO_FPS,
        duration_sec=VIDEO_DURATION_SEC,
        video_format=VIDEO_FORMAT,
        dpi=VIDEO_DPI
    )

    # Also save a static frame for quick preview
    print("\nSaving static preview frame...")
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    create_3d_frame(ax, imu_3d, text_3d, labels, elev=30, azim=45, epoch=epoch)
    preview_path = output_dir / f'embedding_preview_epoch_{epoch}.png'
    plt.savefig(preview_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {preview_path}")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    print(f"Video: {video_file}")
    print(f"Preview: {preview_path}")


if __name__ == '__main__':
    main()
