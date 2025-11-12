"""
Main pretraining script for IMU Activity Recognition Encoder.

Implements dual objective pretraining:
1. Masked Autoencoding (MAE) - 50% random masking
2. Contrastive Learning - patch-level contrastive with augmentations

Usage:
    python pretrain.py --config config.yaml
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
try:
    # Try new API first (torch >= 2.0)
    from torch.amp import autocast, GradScaler
except ImportError:
    # Fall back to old API
    from torch.cuda.amp import autocast, GradScaler
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import sys
from typing import Optional, List

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from tools.models.imu_activity_recognition_encoder.encoder import IMUActivityRecognitionEncoder
from tools.models.imu_activity_recognition_encoder.config import get_config as get_encoder_config
from datasets.imu_pretraining_dataset.multi_dataset_loader import create_dataloaders
from datasets.imu_pretraining_dataset.augmentations import get_weak_augmentation
from training_scripts.imu_tool_pretraining.losses import (
    CombinedPretrainingLoss,
    create_random_mask
)
from training_scripts.imu_tool_pretraining.plot_utils import TrainingPlotter


# ======================== HYPERPARAMETERS ========================

# Data configuration
DATA_ROOT = "/home/alex/code/tsfm/data"
DATASETS = ['uci_har', 'mhealth', 'pamap2', 'wisdm']

# Per-dataset patch sizes (seconds) - optimized based on sampling rate and activity characteristics
# Research shows 5.12s is common for complex activities, but varies by dataset:
# - Higher sampling rate + complex activities → longer patches
# - Lower sampling rate → shorter patches to maintain enough samples
# - Pre-segmented data → use natural segmentation
PATCH_SIZE_PER_DATASET = {
    'uci_har': 2.56,   # 50 Hz, pre-segmented at 128 samples (2.56s), simple activities
    'mhealth': 4.0,    # 50 Hz, health monitoring, medium-duration activities
    'pamap2': 5.0,     # 100 Hz, complex activities (running, cycling), long sessions
    'wisdm': 10.0       # 20 Hz, phone-based, compensate for lower sampling rate
}

MAX_PATCHES_PER_SAMPLE = 48  # Limit to prevent extreme padding (PAMAP2 can have 300+ patches)

# Model configuration
PROJECTION_DIM = 256

# Encoder configuration
D_MODEL = 384  # Match Sentence-BERT output dimension (all-MiniLM-L6-v2)
NUM_HEADS = 8  # 48 dims per head
NUM_TEMPORAL_LAYERS = 4
DIM_FEEDFORWARD = 1536  # 4x d_model
DROPOUT = 0.1
USE_CROSS_CHANNEL = True
CNN_CHANNELS = [64, 128]
CNN_KERNEL_SIZES = [3, 5, 7]

# Training configuration
OUTPUT_DIR = "training_output/imu_pretraining"
SAVE_EVERY = 10  # Save checkpoint every N epochs
EPOCHS = 100
BATCH_SIZE = 8
NUM_WORKERS = 4
SEED = 42

# Optimizer
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
WARMUP_EPOCHS = 10

# Loss weights
MAE_WEIGHT = 1.0
CONTRASTIVE_WEIGHT = 1.0
TEMPERATURE = 0.2  # Standard contrastive learning temperature

# Masking
MASK_RATIO = 0.5  # 50% random masking

# Plotting
PLOT_EVERY_N_BATCHES = 10  # Generate plots every N batches during training

# =================================================================


class PretrainingModel(nn.Module):
    """
    Pretraining wrapper around IMU encoder.

    Adds projection head for contrastive learning and reconstruction head for MAE.
    """

    def __init__(
        self,
        encoder: IMUActivityRecognitionEncoder,
        projection_dim: int = 256,
        device: torch.device = torch.device('cpu')
    ):
        super().__init__()

        self.encoder = encoder
        self.d_model = encoder.d_model
        self.device = device

        # Projection head for contrastive learning (2-layer MLP)
        self.projection_head = nn.Sequential(
            nn.Linear(self.d_model, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        ).to(device)

        # Reconstruction head for MAE
        self.reconstruction_head = nn.Linear(self.d_model, 96).to(device)

    def forward(
        self,
        patches: torch.Tensor,
        attention_mask: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
        channel_descriptions: Optional[List[List[str]]] = None,
        return_reconstruction: bool = True,
        mae_mask: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Forward pass.

        Args:
            patches: (batch, patches, 96, channels)
            attention_mask: (batch, patches) valid patch mask
            channel_mask: (batch, channels) valid channel mask (True=valid, False=padded)
            channel_descriptions: List of channel description lists per sample in batch
            return_reconstruction: Whether to compute reconstruction
            mae_mask: (batch, patches) MAE mask - True=masked position, apply mask_token

        Returns:
            (features, projected_features, reconstructed_patches)
        """
        # Encode - tokens applied at feature level inside encoder
        features = self.encoder(
            patches,
            channel_descriptions=channel_descriptions,
            temporal_mask=None,
            channel_mask=channel_mask,
            mae_mask=mae_mask,
            patch_attention_mask=attention_mask
        )  # (batch, patches, channels, d_model)

        # Project for contrastive learning
        projected = self.projection_head(features)  # (batch, patches, channels, projection_dim)

        # Reconstruct for MAE
        if return_reconstruction:
            reconstructed = self.reconstruction_head(features)  # (batch, patches, channels, 96)
            # Transpose to match input format
            reconstructed = reconstructed.permute(0, 1, 3, 2)  # (batch, patches, 96, channels)
        else:
            reconstructed = None

        return features, projected, reconstructed


def train_epoch(
    model: PretrainingModel,
    dataloader: DataLoader,
    criterion: CombinedPretrainingLoss,
    augmentation,
    optimizer: optim.Optimizer,
    device: torch.device,
    mask_ratio: float = 0.5,
    epoch: int = 0,
    plotter: TrainingPlotter = None,
    scaler: GradScaler = None,
    scheduler = None,
    max_patches_per_sample: int = None
) -> dict:
    """Train for one epoch."""
    model.train()

    # Use mixed precision if scaler is provided
    use_amp = scaler is not None and device.type == 'cuda'

    total_loss = 0.0
    total_mae_loss = 0.0
    total_contrastive_loss = 0.0
    num_batches = 0

    # Per-dataset tracking
    dataset_losses = {}  # {dataset_name: {'loss': total, 'mae': total, 'contrastive': total, 'count': n}}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Train")

    for batch_idx, batch in enumerate(pbar):
        # Get data
        data = batch['data'].to(device)  # (batch, timesteps, channels)
        attention_mask_seq = batch['attention_mask'].to(device)  # (batch, timesteps)
        channel_mask = batch.get('channel_mask', None)  # (batch, channels) - may not exist in all batches
        if channel_mask is not None:
            channel_mask = channel_mask.to(device)
        metadata_list = batch['metadata']

        # Preprocess into patches for each sample in batch
        batch_patches = []
        batch_attention_masks = []
        batch_channel_descriptions = []  # Store channel descriptions per sample

        for i in range(data.shape[0]):
            meta = metadata_list[i]
            sampling_rate = meta['sampling_rate_hz']
            patch_size_sec = meta['patch_size_sec']

            # Use encoder's preprocessing (no gradients needed for data preprocessing)
            with torch.no_grad():
                patches, prep_metadata = model.encoder.preprocess(
                    data[i],
                    sampling_rate_hz=sampling_rate,
                    patch_size_sec=patch_size_sec
                )

            # Move patches back to device (preprocessing returns CPU tensors)
            patches = patches.to(device)

            # Truncate if exceeds max_patches_per_sample (to prevent extreme padding)
            if max_patches_per_sample is not None and patches.shape[0] > max_patches_per_sample:
                patches = patches[:max_patches_per_sample]

            batch_patches.append(patches)

            # Store channel descriptions for this sample
            batch_channel_descriptions.append(meta['channel_descriptions'])

            # Create attention mask for patches (all valid initially)
            patch_attention_mask = torch.ones(patches.shape[0], dtype=torch.bool).to(device)
            batch_attention_masks.append(patch_attention_mask)

        # Stack and pad patches
        max_patches = max(p.shape[0] for p in batch_patches)
        max_channels = max(p.shape[2] for p in batch_patches)           

        padded_patches = torch.zeros(
            len(batch_patches), max_patches, 96, max_channels
        ).to(device)

        patch_attention_mask = torch.zeros(
            len(batch_patches), max_patches, dtype=torch.bool
        ).to(device)

        # Create channel mask for the encoder
        channel_mask_for_model = torch.zeros(
            len(batch_patches), max_channels, dtype=torch.bool
        ).to(device)

        # Pad channel descriptions to match max_channels
        padded_channel_descriptions = []
        for i, (patches, mask) in enumerate(zip(batch_patches, batch_attention_masks)):
            num_patches, _, num_channels = patches.shape
            padded_patches[i, :num_patches, :, :num_channels] = patches
            patch_attention_mask[i, :num_patches] = mask
            channel_mask_for_model[i, :num_channels] = True  # Mark valid channels

            # Pad channel descriptions
            channel_descs = batch_channel_descriptions[i]
            padded_descs = channel_descs + ["[PAD]"] * (max_channels - len(channel_descs))
            padded_channel_descriptions.append(padded_descs)

        # Create augmented view with sample-level augmentation
        # IMPORTANT: Apply same augmentation parameters to all patches within a sample
        # to preserve temporal consistency
        aug_patches = padded_patches.clone()
        for i in range(aug_patches.shape[0]):
            valid_mask = patch_attention_mask[i]
            if valid_mask.any():
                # Get valid patches for this sample
                num_valid_patches = valid_mask.sum().item()
                valid_patches = aug_patches[i, valid_mask]  # (num_valid, 96, channels)

                # Flatten to single tensor: (num_valid * 96, channels)
                flat_shape = (num_valid_patches * 96, valid_patches.shape[-1])
                flat_data = valid_patches.reshape(flat_shape)

                # Apply augmentation ONCE with same parameters for entire sample
                augmented = augmentation.apply(flat_data.unsqueeze(0), None)
                augmented = augmented.squeeze(0)

                # Reshape back to patches: (num_valid, 96, channels)
                augmented_patches = augmented.reshape(num_valid_patches, 96, valid_patches.shape[-1])
                aug_patches[i, valid_mask] = augmented_patches

        # Create random mask for MAE (50%)
        mae_mask = create_random_mask(
            batch_size=padded_patches.shape[0],
            num_patches=max_patches,
            mask_ratio=mask_ratio,
            attention_mask=patch_attention_mask,
            device=device
        )

        # Forward pass with automatic mixed precision
        # Tokens (mask_token, pad_token) are applied at feature level inside the encoder
        with autocast(device_type='cuda', enabled=use_amp):
            # Forward pass - original with MAE masking
            features_1, projected_1, reconstructed = model(
                padded_patches,  # Pass original patches
                patch_attention_mask,
                channel_mask=channel_mask_for_model,
                channel_descriptions=padded_channel_descriptions,
                return_reconstruction=True,
                mae_mask=mae_mask  # Encoder will apply mask_token at feature level
            )

            # Forward pass - augmented (no MAE masking, only pad tokens)
            features_2, projected_2, _ = model(
                aug_patches,  # Pass augmented patches
                patch_attention_mask,
                channel_mask=channel_mask_for_model,
                channel_descriptions=padded_channel_descriptions,
                return_reconstruction=False,
                mae_mask=None  # No MAE masking for augmented view
            )

            # Compute loss
            loss, metrics = criterion(
                predictions=reconstructed,
                targets=padded_patches,
                features_1=projected_1,
                features_2=projected_2,
                attention_mask=patch_attention_mask,
                mae_mask=mae_mask,
                channel_mask=channel_mask_for_model
            )

        # Backward with gradient scaling
        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Step learning rate scheduler (per-batch for warmup)
        if scheduler is not None:
            scheduler.step()

        # Track metrics
        total_loss += metrics['total_loss']
        total_mae_loss += metrics['mae_loss']
        total_contrastive_loss += metrics['contrastive_loss']
        num_batches += 1

        # Track per-dataset metrics (assume single dataset per batch)
        if len(metadata_list) > 0:
            dataset_name = metadata_list[0]['dataset']
            if dataset_name not in dataset_losses:
                dataset_losses[dataset_name] = {
                    'loss': 0.0,
                    'mae': 0.0,
                    'contrastive': 0.0,
                    'count': 0
                }
            dataset_losses[dataset_name]['loss'] += metrics['total_loss']
            dataset_losses[dataset_name]['mae'] += metrics['mae_loss']
            dataset_losses[dataset_name]['contrastive'] += metrics['contrastive_loss']
            dataset_losses[dataset_name]['count'] += 1  # Count batches, not samples

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics['total_loss']:.4f}",
            'mae': f"{metrics['mae_loss']:.4f}",
            'contrast': f"{metrics['contrastive_loss']:.4f}"
        })

        # Log batch-level training losses for real-time monitoring
        if plotter is not None:
            global_batch = epoch * len(dataloader) + batch_idx
            plotter.add_scalar('batch/train_loss', metrics['total_loss'], global_batch)
            plotter.add_scalar('batch/train_mae_loss', metrics['mae_loss'], global_batch)
            plotter.add_scalar('batch/train_contrastive_loss', metrics['contrastive_loss'], global_batch)

        # Clear CUDA cache periodically to prevent memory buildup
        if device.type == 'cuda' and batch_idx % 10 == 0:
            torch.cuda.empty_cache()

        # Generate plots periodically for real-time monitoring
        if plotter is not None and batch_idx % PLOT_EVERY_N_BATCHES == 0 and batch_idx > 0:
            plotter.plot_all()

    # Epoch metrics
    if num_batches > 0:
        metrics = {
            'loss': total_loss / num_batches,
            'mae_loss': total_mae_loss / num_batches,
            'contrastive_loss': total_contrastive_loss / num_batches
        }
    else:
        metrics = {
            'loss': 0.0,
            'mae_loss': 0.0,
            'contrastive_loss': 0.0
        }

    # Compute per-dataset averages
    per_dataset_metrics = {}
    for dataset_name, stats in dataset_losses.items():
        if stats['count'] > 0:
            per_dataset_metrics[dataset_name] = {
                'loss': stats['loss'] / stats['count'],
                'mae_loss': stats['mae'] / stats['count'],
                'contrastive_loss': stats['contrastive'] / stats['count']
            }

            # Log to plotter
            if plotter is not None:
                plotter.add_scalar(f'train_per_dataset/{dataset_name}/loss',
                                per_dataset_metrics[dataset_name]['loss'], epoch)
                plotter.add_scalar(f'train_per_dataset/{dataset_name}/mae_loss',
                                per_dataset_metrics[dataset_name]['mae_loss'], epoch)
                plotter.add_scalar(f'train_per_dataset/{dataset_name}/contrastive_loss',
                                per_dataset_metrics[dataset_name]['contrastive_loss'], epoch)

    metrics['per_dataset'] = per_dataset_metrics

    return metrics


def validate(
    model: PretrainingModel,
    dataloader: DataLoader,
    criterion: CombinedPretrainingLoss,
    augmentation,
    device: torch.device,
    mask_ratio: float = 0.5,
    epoch: int = 0,
    plotter: TrainingPlotter = None,
    max_patches_per_sample: int = None
) -> dict:
    """Validate the model."""
    model.eval()  # Set to evaluation mode

    total_loss = 0.0
    total_mae_loss = 0.0
    total_contrastive_loss = 0.0
    num_batches = 0

    # Per-dataset tracking
    dataset_losses = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
            # Similar preprocessing as train (simplified for validation)
            data = batch['data'].to(device)
            channel_mask = batch.get('channel_mask', None)
            if channel_mask is not None:
                channel_mask = channel_mask.to(device)
            metadata_list = batch['metadata']

            # Preprocess into patches
            batch_patches = []
            batch_attention_masks = []
            batch_channel_descriptions = []

            for i in range(data.shape[0]):
                meta = metadata_list[i]
                patches, _ = model.encoder.preprocess(
                    data[i],
                    sampling_rate_hz=meta['sampling_rate_hz'],
                    patch_size_sec=meta['patch_size_sec']
                )
                # Move patches back to device (preprocessing returns CPU tensors)
                patches = patches.to(device)

                # Truncate if exceeds max_patches_per_sample (to prevent extreme padding)
                if max_patches_per_sample is not None and patches.shape[0] > max_patches_per_sample:
                    patches = patches[:max_patches_per_sample]

                batch_patches.append(patches)
                batch_channel_descriptions.append(meta['channel_descriptions'])
                batch_attention_masks.append(torch.ones(patches.shape[0], dtype=torch.bool).to(device))

            # Pad patches
            max_patches = max(p.shape[0] for p in batch_patches)
            max_channels = max(p.shape[2] for p in batch_patches)

            padded_patches = torch.zeros(len(batch_patches), max_patches, 96, max_channels).to(device)
            patch_attention_mask = torch.zeros(len(batch_patches), max_patches, dtype=torch.bool).to(device)

            # Create channel mask for the encoder
            channel_mask_for_model = torch.zeros(len(batch_patches), max_channels, dtype=torch.bool).to(device)

            # Pad channel descriptions
            padded_channel_descriptions = []
            for i, (patches, mask) in enumerate(zip(batch_patches, batch_attention_masks)):
                num_patches, _, num_channels = patches.shape
                padded_patches[i, :num_patches, :, :num_channels] = patches
                patch_attention_mask[i, :num_patches] = mask
                channel_mask_for_model[i, :num_channels] = True

                # Pad channel descriptions
                channel_descs = batch_channel_descriptions[i]
                padded_descs = channel_descs + ["[PAD]"] * (max_channels - len(channel_descs))
                padded_channel_descriptions.append(padded_descs)

            # Create augmented view (no augmentation in validation for consistency)
            aug_patches = padded_patches.clone()

            # Create mask
            mae_mask = create_random_mask(
                batch_size=padded_patches.shape[0],
                num_patches=max_patches,
                mask_ratio=mask_ratio,
                attention_mask=patch_attention_mask,
                device=device
            )

            # Forward with autocast for memory savings (even without gradients)
            with autocast(device_type='cuda', enabled=device.type == 'cuda'):
                features_1, projected_1, reconstructed = model(
                    padded_patches,
                    patch_attention_mask,
                    channel_mask=channel_mask_for_model,
                    channel_descriptions=padded_channel_descriptions,
                    return_reconstruction=True,
                    mae_mask=mae_mask
                )
                features_2, projected_2, _ = model(
                    aug_patches,
                    patch_attention_mask,
                    channel_mask=channel_mask_for_model,
                    channel_descriptions=padded_channel_descriptions,
                    return_reconstruction=False,
                    mae_mask=None
                )

                # Compute loss
                loss, metrics = criterion(
                    reconstructed, padded_patches,
                    projected_1, projected_2,
                    patch_attention_mask, mae_mask,
                    channel_mask_for_model
                )

            total_loss += metrics['total_loss']
            total_mae_loss += metrics['mae_loss']
            total_contrastive_loss += metrics['contrastive_loss']
            num_batches += 1

            # Track per-dataset metrics (assume single dataset per batch)
            if len(metadata_list) > 0:
                dataset_name = metadata_list[0]['dataset']
                if dataset_name not in dataset_losses:
                    dataset_losses[dataset_name] = {
                        'loss': 0.0,
                        'mae': 0.0,
                        'contrastive': 0.0,
                        'count': 0
                    }
                dataset_losses[dataset_name]['loss'] += metrics['total_loss']
                dataset_losses[dataset_name]['mae'] += metrics['mae_loss']
                dataset_losses[dataset_name]['contrastive'] += metrics['contrastive_loss']
                dataset_losses[dataset_name]['count'] += 1  # Count batches, not samples

            # Clear CUDA cache periodically to prevent memory buildup
            if device.type == 'cuda' and batch_idx % 10 == 0:
                torch.cuda.empty_cache()

    # Epoch metrics
    if num_batches > 0:
        metrics = {
            'loss': total_loss / num_batches,
            'mae_loss': total_mae_loss / num_batches,
            'contrastive_loss': total_contrastive_loss / num_batches
        }
    else:
        metrics = {
            'loss': 0.0,
            'mae_loss': 0.0,
            'contrastive_loss': 0.0
        }

    # Compute per-dataset averages
    per_dataset_metrics = {}
    for dataset_name, stats in dataset_losses.items():
        if stats['count'] > 0:
            per_dataset_metrics[dataset_name] = {
                'loss': stats['loss'] / stats['count'],
                'mae_loss': stats['mae'] / stats['count'],
                'contrastive_loss': stats['contrastive'] / stats['count']
            }

            # Log to plotter
            if plotter is not None:
                plotter.add_scalar(f'val_per_dataset/{dataset_name}/loss',
                                per_dataset_metrics[dataset_name]['loss'], epoch)
                plotter.add_scalar(f'val_per_dataset/{dataset_name}/mae_loss',
                                per_dataset_metrics[dataset_name]['mae_loss'], epoch)
                plotter.add_scalar(f'val_per_dataset/{dataset_name}/contrastive_loss',
                                per_dataset_metrics[dataset_name]['contrastive_loss'], epoch)

    metrics['per_dataset'] = per_dataset_metrics

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Pretrain IMU Activity Recognition Encoder')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Print configuration
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Data root: {DATA_ROOT}")
    print(f"Datasets: {DATASETS}")
    print(f"Per-dataset patch sizes:")
    for dataset, patch_size in PATCH_SIZE_PER_DATASET.items():
        print(f"  {dataset}: {patch_size} sec")
    print(f"Max patches per sample: {MAX_PATCHES_PER_SAMPLE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Mask ratio: {MASK_RATIO}")
    print(f"MAE weight: {MAE_WEIGHT}, Contrastive weight: {CONTRASTIVE_WEIGHT}")
    print("="*60 + "\n")

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(OUTPUT_DIR) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    config = {
        'data': {
            'data_root': DATA_ROOT,
            'datasets': DATASETS,
            'patch_size_per_dataset': PATCH_SIZE_PER_DATASET,
            'max_patches_per_sample': MAX_PATCHES_PER_SAMPLE
        },
        'model': {
            'projection_dim': PROJECTION_DIM
        },
        'encoder': {
            'd_model': D_MODEL,
            'num_heads': NUM_HEADS,
            'num_temporal_layers': NUM_TEMPORAL_LAYERS,
            'dim_feedforward': DIM_FEEDFORWARD,
            'dropout': DROPOUT,
            'use_cross_channel': USE_CROSS_CHANNEL,
            'cnn_channels': CNN_CHANNELS,
            'cnn_kernel_sizes': CNN_KERNEL_SIZES
        },
        'training': {
            'output_dir': OUTPUT_DIR,
            'save_every': SAVE_EVERY,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'num_workers': NUM_WORKERS,
            'seed': SEED,
            'lr': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'warmup_epochs': WARMUP_EPOCHS,
            'mae_weight': MAE_WEIGHT,
            'contrastive_weight': CONTRASTIVE_WEIGHT,
            'temperature': TEMPERATURE,
            'mask_ratio': MASK_RATIO
        }
    }

    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Setup plotter for loss curves
    plotter = TrainingPlotter(output_dir=output_dir / 'plots')

    # Create encoder
    encoder_config = get_encoder_config('default')
    encoder_config.update({
        'd_model': D_MODEL,
        'num_heads': NUM_HEADS,
        'num_temporal_layers': NUM_TEMPORAL_LAYERS,
        'dim_feedforward': DIM_FEEDFORWARD,
        'dropout': DROPOUT,
        'use_cross_channel': USE_CROSS_CHANNEL,
        'cnn_channels': CNN_CHANNELS,
        'cnn_kernel_sizes': CNN_KERNEL_SIZES
    })
    encoder = IMUActivityRecognitionEncoder(**encoder_config)

    # Create pretraining model
    model = PretrainingModel(
        encoder=encoder,
        projection_dim=PROJECTION_DIM,
        device=device
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=DATA_ROOT,
        datasets=DATASETS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        patch_size_sec=2.0,  # Default (not used since we provide per-dataset sizes)
        patch_size_per_dataset=PATCH_SIZE_PER_DATASET,
        seed=SEED
    )

    # Create augmentation (weak: jitter, scale, time_shift)
    augmentation = get_weak_augmentation()

    # Create criterion
    criterion = CombinedPretrainingLoss(
        mae_weight=MAE_WEIGHT,
        contrastive_weight=CONTRASTIVE_WEIGHT,
        temperature=TEMPERATURE
    )

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = len(train_loader) * WARMUP_EPOCHS

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Create gradient scaler for mixed precision training
    scaler = GradScaler() if device.type == 'cuda' else None
    if scaler is not None:
        print("Using mixed precision training (AMP)")

    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, augmentation,
            optimizer, device, MASK_RATIO,
            epoch, plotter, scaler, scheduler,
            max_patches_per_sample=MAX_PATCHES_PER_SAMPLE
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion, augmentation,
            device, MASK_RATIO, epoch, plotter,
            max_patches_per_sample=MAX_PATCHES_PER_SAMPLE
        )

        # Log epoch metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae_loss']:.4f}, Contrast: {train_metrics['contrastive_loss']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae_loss']:.4f}, Contrast: {val_metrics['contrastive_loss']:.4f}")

        # Log to plotter
        plotter.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
        plotter.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
        plotter.add_scalar('epoch/train_mae_loss', train_metrics['mae_loss'], epoch)
        plotter.add_scalar('epoch/val_mae_loss', val_metrics['mae_loss'], epoch)
        plotter.add_scalar('epoch/train_contrastive_loss', train_metrics['contrastive_loss'], epoch)
        plotter.add_scalar('epoch/val_contrastive_loss', val_metrics['contrastive_loss'], epoch)
        plotter.add_scalar('epoch/lr', optimizer.param_groups[0]['lr'], epoch)

        # Generate plots after every epoch (already plotting every 10 batches too)
        plotter.plot_all()

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': config
        }
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()

        # Save latest
        torch.save(checkpoint, output_dir / 'latest.pt')

        # Save best
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(checkpoint, output_dir / 'best.pt')
            print(f"✓ Saved best model (val_loss: {best_val_loss:.4f})")

        # Save periodic checkpoints
        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch+1}.pt')

    # Generate final plots
    plotter.close()
    print(f"\nTraining complete! Output saved to {output_dir}")


if __name__ == "__main__":
    main()
