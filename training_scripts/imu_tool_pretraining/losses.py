"""
Loss functions for IMU pretraining.

Implements:
1. Masked Reconstruction Loss (MAE) - predicts masked patches
2. Patch Contrastive Loss (InfoNCE) - contrastive learning across batch
3. Combined loss for joint training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MaskedReconstructionLoss(nn.Module):
    """
    Masked autoencoding reconstruction loss.

    Computes MSE loss between predicted and target values, but only on:
    - Masked positions (mae_mask=True)
    - Valid positions (attention_mask=True)

    Applies per-patch normalization to targets for stable training.
    """

    def __init__(self, norm_target: bool = True):
        """
        Args:
            norm_target: Whether to normalize targets per-patch (recommended)
        """
        super().__init__()
        self.norm_target = norm_target

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: torch.Tensor,
        mae_mask: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute masked reconstruction loss.

        Args:
            predictions: Predicted values (batch, num_patches, 96, num_channels)
            targets: Ground truth values (batch, num_patches, 96, num_channels)
            attention_mask: Valid patch mask (batch, num_patches) - True=valid
            mae_mask: MAE mask (batch, num_patches) - True=masked (should reconstruct)
            channel_mask: Optional channel validity mask (batch, num_channels) - True=valid, False=padded

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Normalize targets per-patch if requested
        if self.norm_target:
            # Compute mean and std per patch, per channel
            # Only over valid patches (exclude padding)
            targets_normalized = self._normalize_patches(targets, attention_mask)
        else:
            targets_normalized = targets

        # Compute MSE
        mse = F.mse_loss(predictions, targets_normalized, reduction='none')
        # Shape: (batch, num_patches, 96, num_channels)

        # Average over timesteps: (batch, num_patches, 96, num_channels) -> (batch, num_patches, num_channels)
        mse_per_channel = mse.mean(dim=2)

        # Average over channels with masking to exclude padding
        if channel_mask is not None:
            # Expand channel mask: (batch, num_channels) -> (batch, 1, num_channels)
            mask_expanded = channel_mask.unsqueeze(1).float()

            # Masked average: sum valid channels / count valid channels
            mse_per_patch = (mse_per_channel * mask_expanded).sum(dim=2) / mask_expanded.sum(dim=2).clamp(min=1)
        else:
            # No channel masking - simple average
            mse_per_patch = mse_per_channel.mean(dim=2)

        # Now: (batch, num_patches)

        # Create combined mask: valid AND masked positions
        combined_mask = attention_mask & mae_mask

        # Mask the loss
        masked_mse = mse_per_patch * combined_mask.float()

        # Compute loss (only on masked, valid positions)
        num_masked = combined_mask.sum()
        if num_masked > 0:
            loss = masked_mse.sum() / num_masked
        else:
            loss = torch.tensor(0.0, device=mse.device)

        # Metrics
        metrics = {
            'mae_loss': loss.item(),
            'num_masked_patches': num_masked.item(),
            'mask_ratio': (mae_mask.sum().float() / mae_mask.numel()).item()
        }

        return loss, metrics

    def _normalize_patches(self, patches: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Normalize patches to zero mean and unit variance, excluding padding.

        Args:
            patches: (batch, num_patches, 96, num_channels)
            attention_mask: (batch, num_patches) - True=valid, False=padding

        Returns:
            Normalized patches of same shape
        """
        # Expand mask for broadcasting: (batch, num_patches) -> (batch, num_patches, 1, 1)
        mask = attention_mask.unsqueeze(-1).unsqueeze(-1).float()

        # Mask out padded patches
        masked_patches = patches * mask

        # Compute masked mean and std over timesteps (dim=2)
        # Only compute statistics where mask is True
        count = mask.sum(dim=2, keepdim=True).clamp(min=1)  # Avoid division by zero
        mean = masked_patches.sum(dim=2, keepdim=True) / count

        # Compute variance
        variance = ((masked_patches - mean * mask) ** 2).sum(dim=2, keepdim=True) / count
        std = torch.sqrt(variance).clamp(min=1e-6)

        # Normalize (all patches including padding, but using stats from valid patches only)
        normalized = (patches - mean) / std

        return normalized


class PatchContrastiveLoss(nn.Module):
    """
    Patch-level contrastive loss using InfoNCE / NT-Xent.

    Positive pairs: Same patch position from original and augmented version
    Negative pairs: Same patch position from different samples in batch

    This implements the contrastive objective where each patch's features
    should be similar to its augmented version and different from other
    samples' patches at the same position.
    """

    def __init__(self, temperature: float = 0.2):
        """
        Args:
            temperature: Temperature parameter for InfoNCE loss
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features_1: torch.Tensor,
        features_2: torch.Tensor,
        attention_mask: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
        mae_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute patch-level contrastive loss.

        Args:
            features_1: Features from original data (batch, patches, channels, d_model)
            features_2: Features from augmented data (batch, patches, channels, d_model)
            attention_mask: Valid patch mask (batch, patches) - True=valid
            channel_mask: Optional channel validity mask (batch, channels) - True=valid, False=padded
            mae_mask: Optional MAE mask (batch, patches) - True=masked. Masked patches excluded from negatives.

        Returns:
            Tuple of (loss, metrics_dict)
        """
        batch_size, num_patches, num_channels, d_model = features_1.shape

        # Pool over channels to get patch-level features
        # (batch, patches, channels, d_model) -> (batch, patches, d_model)
        if channel_mask is not None:
            # Mask out padded channels before pooling
            # Expand mask: (batch, channels) -> (batch, 1, channels, 1)
            mask = channel_mask.unsqueeze(1).unsqueeze(-1).float()  # (batch, 1, channels, 1)

            # Apply mask and compute masked average
            feat1 = (features_1 * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)  # (batch, patches, d_model)
            feat2 = (features_2 * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)
        else:
            # No masking - simple average
            feat1 = features_1.mean(dim=2)  # Average over channels
            feat2 = features_2.mean(dim=2)

        # Normalize features
        feat1 = F.normalize(feat1, dim=-1)  # (batch, patches, d_model)
        feat2 = F.normalize(feat2, dim=-1)

        # Create mask for valid patches
        # Only exclude padding (not MAE-masked patches)
        # MAE-masked patches still have valid encoder features for contrastive learning
        if mae_mask is not None:
            # Only exclude padding, keep MAE-masked patches
            valid_for_contrast = attention_mask
        else:
            valid_for_contrast = attention_mask

        # Flatten to patch level: (batch, patches, d_model) -> (num_valid_patches, d_model)
        feat1_flat = feat1[valid_for_contrast]  # Gather only valid, unmasked patches
        feat2_flat = feat2[valid_for_contrast]

        # Handle empty batch case (all patches masked/padded) to prevent NaN
        if feat1_flat.shape[0] == 0:
            # No valid patches - return zero loss with warning
            loss = torch.tensor(0.0, device=feat1.device, dtype=feat1.dtype)
            metrics = {
                'contrastive_loss': 0.0,
                'effective_batch_size': 0,
                'warning': 'empty_batch'
            }
            return loss, metrics

        # Compute InfoNCE loss at patch level
        loss = self._info_nce_loss(feat1_flat, feat2_flat)

        # Metrics
        effective_batch_size = feat1_flat.shape[0]
        metrics = {
            'contrastive_loss': loss.item(),
            'effective_batch_size': effective_batch_size
        }

        return loss, metrics

    def _info_nce_loss(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss for a single patch position.

        Args:
            z_i: Features from view 1 (N, d_model)
            z_j: Features from view 2 (N, d_model)

        Returns:
            Scalar loss
        """
        N = z_i.shape[0]

        # Concatenate to form (2N, d_model)
        z = torch.cat([z_i, z_j], dim=0)

        # Compute similarity matrix (2N, 2N)
        sim_matrix = torch.mm(z, z.t()) / self.temperature

        # Create labels: i and i+N are positives
        # For i in [0, N), positive is at i+N
        # For i in [N, 2N), positive is at i-N
        labels = torch.cat([
            torch.arange(N, 2*N),  # For first N samples, positives are in second N
            torch.arange(0, N)      # For second N samples, positives are in first N
        ]).to(z.device)

        # Mask out self-similarities (diagonal)
        mask = torch.eye(2*N, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

        # Compute cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


class CombinedPretrainingLoss(nn.Module):
    """
    Combined loss for pretraining: MAE + Contrastive.

    Balances masked autoencoding and contrastive learning objectives.
    Supports dynamic loss balancing via EMA normalization.
    """

    def __init__(
        self,
        mae_weight: float = 1.0,
        contrastive_weight: float = 1.0,
        temperature: float = 0.2,
        norm_target: bool = True,
        dynamic_balance: bool = True,
        ema_decay: float = 0.99
    ):
        """
        Args:
            mae_weight: Weight for MAE loss (relative importance)
            contrastive_weight: Weight for contrastive loss (relative importance)
            temperature: Temperature for contrastive loss
            norm_target: Whether to normalize MAE targets
            dynamic_balance: If True, normalize losses by their EMA to balance magnitudes
            ema_decay: Decay rate for EMA (0.99 = smooth, 0.9 = faster adaptation)
        """
        super().__init__()

        self.mae_weight = mae_weight
        self.contrastive_weight = contrastive_weight
        self.dynamic_balance = dynamic_balance
        self.ema_decay = ema_decay

        self.mae_loss_fn = MaskedReconstructionLoss(norm_target=norm_target)
        self.contrastive_loss_fn = PatchContrastiveLoss(temperature=temperature)

        # EMA trackers for dynamic balancing (not saved in state_dict)
        self.register_buffer('mae_ema', torch.tensor(1.0))
        self.register_buffer('contrastive_ema', torch.tensor(1.0))
        self.register_buffer('ema_initialized', torch.tensor(False))

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        features_1: torch.Tensor,
        features_2: torch.Tensor,
        attention_mask: torch.Tensor,
        mae_mask: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Args:
            predictions: Reconstructed patches (batch, patches, 96, channels)
            targets: Ground truth patches (batch, patches, 96, channels)
            features_1: Features from original (batch, patches, channels, d_model)
            features_2: Features from augmented (batch, patches, channels, d_model)
            attention_mask: Valid patch mask (batch, patches)
            mae_mask: MAE mask (batch, patches)
            channel_mask: Optional channel validity mask (batch, channels)

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Compute MAE loss (with channel masking to exclude padding)
        mae_loss, mae_metrics = self.mae_loss_fn(
            predictions, targets, attention_mask, mae_mask,
            channel_mask=channel_mask
        )

        # Compute contrastive loss (patch-level, excluding masked and padded patches)
        contrastive_loss, contrastive_metrics = self.contrastive_loss_fn(
            features_1, features_2, attention_mask,
            channel_mask=channel_mask,
            mae_mask=mae_mask
        )

        # Dynamic loss balancing via EMA normalization
        if self.dynamic_balance:
            with torch.no_grad():
                mae_val = mae_loss.detach()
                cont_val = contrastive_loss.detach()

                if not self.ema_initialized:
                    # Initialize EMA with first values
                    self.mae_ema = mae_val
                    self.contrastive_ema = cont_val
                    self.ema_initialized = torch.tensor(True)
                else:
                    # Update EMA
                    self.mae_ema = self.ema_decay * self.mae_ema + (1 - self.ema_decay) * mae_val
                    self.contrastive_ema = self.ema_decay * self.contrastive_ema + (1 - self.ema_decay) * cont_val

            # Normalize losses by their EMA (so both contribute ~1.0 on average)
            # Then apply user-specified weights for relative importance
            mae_normalized = mae_loss / (self.mae_ema + 1e-8)
            contrastive_normalized = contrastive_loss / (self.contrastive_ema + 1e-8)

            total_loss = (
                self.mae_weight * mae_normalized +
                self.contrastive_weight * contrastive_normalized
            )

            # Effective weights for logging
            effective_mae_weight = self.mae_weight / (self.mae_ema.item() + 1e-8)
            effective_contrastive_weight = self.contrastive_weight / (self.contrastive_ema.item() + 1e-8)
        else:
            # Static weighting (original behavior)
            total_loss = (
                self.mae_weight * mae_loss +
                self.contrastive_weight * contrastive_loss
            )
            effective_mae_weight = self.mae_weight
            effective_contrastive_weight = self.contrastive_weight

        # Combined metrics
        metrics = {
            'total_loss': total_loss.item(),
            **mae_metrics,
            **contrastive_metrics,
            'mae_weight': self.mae_weight,
            'contrastive_weight': self.contrastive_weight,
            'effective_mae_weight': effective_mae_weight,
            'effective_contrastive_weight': effective_contrastive_weight,
        }

        if self.dynamic_balance:
            metrics['mae_ema'] = self.mae_ema.item()
            metrics['contrastive_ema'] = self.contrastive_ema.item()

        return total_loss, metrics


def create_random_mask(
    batch_size: int,
    num_patches: int,
    mask_ratio: float = 0.5,
    attention_mask: Optional[torch.Tensor] = None,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Create random mask for masked autoencoding.

    Args:
        batch_size: Batch size
        num_patches: Number of patches per sample
        mask_ratio: Ratio of patches to mask (0.5 = 50%)
        attention_mask: Valid patch mask (batch, patches) - don't mask invalid patches
        device: Device to create mask on

    Returns:
        Boolean mask (batch, patches) where True = should be masked/reconstructed
    """
    # Create random mask
    random_values = torch.rand(batch_size, num_patches, device=device)
    mae_mask = random_values < mask_ratio

    # Don't mask invalid (padded) patches
    if attention_mask is not None:
        mae_mask = mae_mask & attention_mask

    return mae_mask


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")

    batch_size = 4
    num_patches = 10
    patch_size = 96
    num_channels = 9
    d_model = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dummy data (with requires_grad for backward test)
    predictions = torch.randn(batch_size, num_patches, patch_size, num_channels, requires_grad=True).to(device)
    targets = torch.randn(batch_size, num_patches, patch_size, num_channels).to(device)
    features_1 = torch.randn(batch_size, num_patches, num_channels, d_model, requires_grad=True).to(device)
    features_2 = torch.randn(batch_size, num_patches, num_channels, d_model, requires_grad=True).to(device)

    # Create masks
    attention_mask = torch.ones(batch_size, num_patches, dtype=torch.bool).to(device)
    attention_mask[0, 8:] = False  # Simulate padding for first sample
    attention_mask[1, 9:] = False

    mae_mask = create_random_mask(batch_size, num_patches, 0.5, attention_mask, device)

    # Test MAE loss
    print("\n1. Testing Masked Reconstruction Loss...")
    mae_loss_fn = MaskedReconstructionLoss()
    mae_loss, mae_metrics = mae_loss_fn(predictions, targets, attention_mask, mae_mask)
    print(f"   MAE Loss: {mae_loss.item():.4f}")
    print(f"   Metrics: {mae_metrics}")

    # Test Contrastive loss
    print("\n2. Testing Patch Contrastive Loss...")
    contrastive_loss_fn = PatchContrastiveLoss(temperature=0.2)
    contrastive_loss, contrastive_metrics = contrastive_loss_fn(
        features_1, features_2, attention_mask
    )
    print(f"   Contrastive Loss: {contrastive_loss.item():.4f}")
    print(f"   Metrics: {contrastive_metrics}")

    # Test Combined loss
    print("\n3. Testing Combined Loss...")
    combined_loss_fn = CombinedPretrainingLoss(mae_weight=1.0, contrastive_weight=0.5)
    total_loss, combined_metrics = combined_loss_fn(
        predictions, targets, features_1, features_2, attention_mask, mae_mask
    )
    print(f"   Total Loss: {total_loss.item():.4f}")
    print(f"   Metrics: {combined_metrics}")

    # Test backward pass
    print("\n4. Testing backward pass...")
    total_loss.backward()
    print("   ✓ Backward pass successful")

    print("\n✓ All loss function tests passed!")
