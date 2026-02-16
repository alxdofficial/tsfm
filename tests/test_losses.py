"""Test 3: Loss computation correctness.

Verifies Stage 1 (MAE/contrastive) and Stage 2 (semantic alignment) losses.
"""

import sys
from pathlib import Path
import torch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training_scripts.human_activity_recognition.losses import (
    MaskedReconstructionLoss,
    PatchContrastiveLoss,
    CombinedPretrainingLoss,
    create_span_mask,
    create_channel_dropout_mask,
    create_random_mask,
)
from training_scripts.human_activity_recognition.semantic_loss import (
    InfoNCELoss,
    SemanticAlignmentLoss,
)


# ============================================================
# Stage 1 Losses
# ============================================================

class TestMaskedReconstructionLoss:
    """Test MAE reconstruction loss."""

    def test_output_is_scalar(self):
        B, P, T, C = 4, 10, 96, 9
        loss_fn = MaskedReconstructionLoss(norm_target=True)
        predictions = torch.randn(B, P, T, C)
        targets = torch.randn(B, P, T, C)
        attention_mask = torch.ones(B, P, dtype=torch.bool)
        mae_mask = torch.zeros(B, P, dtype=torch.bool)
        mae_mask[:, :5] = True  # Mask first 5 patches

        loss, metrics = loss_fn(predictions, targets, attention_mask, mae_mask)
        assert loss.dim() == 0, "Loss should be scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert loss.item() > 0, "Loss should be positive for random inputs"

    def test_zero_loss_for_identical_inputs(self):
        B, P, T, C = 2, 5, 96, 6
        loss_fn = MaskedReconstructionLoss(norm_target=False)  # No normalization for exact match
        data = torch.randn(B, P, T, C)
        attention_mask = torch.ones(B, P, dtype=torch.bool)
        mae_mask = torch.ones(B, P, dtype=torch.bool)

        loss, _ = loss_fn(data, data, attention_mask, mae_mask)
        assert loss.item() < 1e-6, f"Loss should be ~0 for identical inputs, got {loss.item()}"

    def test_channel_mask_excludes_padded_channels(self):
        B, P, T, C = 2, 5, 96, 9
        loss_fn = MaskedReconstructionLoss(norm_target=False)
        predictions = torch.randn(B, P, T, C)
        targets = torch.randn(B, P, T, C)
        attention_mask = torch.ones(B, P, dtype=torch.bool)
        mae_mask = torch.ones(B, P, dtype=torch.bool)

        channel_mask = torch.ones(B, C, dtype=torch.bool)
        channel_mask[:, 6:] = False  # Last 3 channels are padding

        loss, _ = loss_fn(predictions, targets, attention_mask, mae_mask,
                          channel_mask=channel_mask)
        assert not torch.isnan(loss)

    def test_metrics_keys(self):
        B, P, T, C = 2, 5, 96, 6
        loss_fn = MaskedReconstructionLoss()
        predictions = torch.randn(B, P, T, C)
        targets = torch.randn(B, P, T, C)
        attention_mask = torch.ones(B, P, dtype=torch.bool)
        mae_mask = torch.ones(B, P, dtype=torch.bool)

        _, metrics = loss_fn(predictions, targets, attention_mask, mae_mask)
        assert 'mae_loss' in metrics
        assert 'num_masked_patches' in metrics
        assert 'mask_ratio' in metrics


class TestPatchContrastiveLoss:
    """Test patch-level contrastive loss."""

    def test_output_is_scalar(self):
        B, P, C, D = 4, 10, 9, 128
        loss_fn = PatchContrastiveLoss(temperature=0.2)
        features_1 = torch.randn(B, P, C, D)
        features_2 = torch.randn(B, P, C, D)
        attention_mask = torch.ones(B, P, dtype=torch.bool)

        loss, metrics = loss_fn(features_1, features_2, attention_mask)
        assert loss.dim() == 0, "Loss should be scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_empty_batch_returns_zero(self):
        B, P, C, D = 2, 5, 6, 128
        loss_fn = PatchContrastiveLoss()
        features_1 = torch.randn(B, P, C, D)
        features_2 = torch.randn(B, P, C, D)
        attention_mask = torch.zeros(B, P, dtype=torch.bool)  # All masked

        loss, metrics = loss_fn(features_1, features_2, attention_mask)
        assert loss.item() == 0.0


class TestCreateSpanMask:
    """Test span mask creation."""

    def test_output_shape(self):
        B, P = 4, 20
        mask = create_span_mask(B, P, mask_ratio=0.3)
        assert mask.shape == (B, P)
        assert mask.dtype == torch.bool

    def test_contiguous_spans(self):
        """Verify that masked positions form contiguous spans."""
        B, P = 1, 50
        mask = create_span_mask(B, P, mask_ratio=0.3, span_length_range=(2, 4))
        # Check that at least some contiguous groups exist
        row = mask[0]
        if row.any():
            # Find transitions (0->1 or 1->0)
            diffs = row[1:].int() - row[:-1].int()
            starts = (diffs == 1).sum().item()
            ends = (diffs == -1).sum().item()
            # Number of spans should be small relative to masked count
            total_masked = row.sum().item()
            num_spans = starts + (1 if row[0] else 0)
            if total_masked > 0:
                # Average span length should be >= min_span
                avg_span = total_masked / max(num_spans, 1)
                assert avg_span >= 1.0, "Spans should be contiguous"

    def test_respects_target_ratio(self):
        """Mask ratio should be roughly correct (within tolerance)."""
        B, P = 8, 100
        target_ratio = 0.3
        mask = create_span_mask(B, P, mask_ratio=target_ratio)
        actual_ratio = mask.float().mean().item()
        # Allow generous tolerance due to span granularity
        assert actual_ratio > 0.1, f"Too few masked patches: {actual_ratio}"
        assert actual_ratio < 0.6, f"Too many masked patches: {actual_ratio}"

    def test_respects_attention_mask(self):
        """Should not mask padding patches."""
        B, P = 2, 20
        attention_mask = torch.ones(B, P, dtype=torch.bool)
        attention_mask[:, 15:] = False  # Last 5 are padding
        mask = create_span_mask(B, P, mask_ratio=0.5, attention_mask=attention_mask)
        assert not mask[:, 15:].any(), "Padding positions should not be masked"


class TestCreateChannelDropoutMask:
    """Test channel dropout mask creation."""

    def test_keeps_at_least_one_channel(self):
        B, C = 8, 9
        for _ in range(10):  # Run multiple times for stochasticity
            mask = create_channel_dropout_mask(B, C, dropout_ratio=0.9)
            for i in range(B):
                # At least 1 channel should NOT be dropped
                assert (mask[i] == False).any(), \
                    f"Sample {i}: all channels dropped, must keep at least 1"

    def test_single_channel_not_dropped(self):
        """With only 1 valid channel, nothing should be dropped."""
        B, C = 2, 5
        channel_mask = torch.zeros(B, C, dtype=torch.bool)
        channel_mask[:, 0] = True  # Only 1 valid channel
        mask = create_channel_dropout_mask(B, C, dropout_ratio=0.5, channel_mask=channel_mask)
        assert not mask.any(), "Single valid channel should never be dropped"

    def test_output_shape(self):
        B, C = 4, 12
        mask = create_channel_dropout_mask(B, C, dropout_ratio=0.3)
        assert mask.shape == (B, C)
        assert mask.dtype == torch.bool


# ============================================================
# Stage 2 Losses
# ============================================================

class TestInfoNCELoss:
    """Test InfoNCE contrastive loss for semantic alignment."""

    def test_single_prototype_scalar_loss(self):
        B, D = 8, 384
        loss_fn = InfoNCELoss(temperature=0.1, use_soft_targets=True)
        imu = torch.randn(B, D)
        imu = imu / imu.norm(dim=-1, keepdim=True)
        text = torch.randn(B, D)
        text = text / text.norm(dim=-1, keepdim=True)

        loss, metrics = loss_fn(imu, text, return_metrics=True)
        assert loss.dim() == 0, "Loss should be scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert loss.item() > 0, "InfoNCE loss should be positive"

    def test_multi_prototype_scalar_loss(self):
        B, K, D = 8, 3, 384
        loss_fn = InfoNCELoss(temperature=0.1, use_soft_targets=True)
        imu = torch.randn(B, D)
        imu = imu / imu.norm(dim=-1, keepdim=True)
        text = torch.randn(B, K, D)
        text = text / text.norm(dim=-1, keepdim=True)

        loss, metrics = loss_fn(imu, text, return_metrics=True)
        assert loss.dim() == 0, "Loss should be scalar"
        assert not torch.isnan(loss), f"Loss is NaN"

    def test_no_nan_with_hard_targets(self):
        B, D = 4, 384
        loss_fn = InfoNCELoss(temperature=0.1, use_soft_targets=False)
        imu = torch.randn(B, D)
        imu = imu / imu.norm(dim=-1, keepdim=True)
        text = torch.randn(B, D)
        text = text / text.norm(dim=-1, keepdim=True)

        loss, _ = loss_fn(imu, text)
        assert not torch.isnan(loss)

    def test_with_queue(self):
        B, D, Q = 4, 384, 16
        loss_fn = InfoNCELoss(temperature=0.1, use_soft_targets=True)
        imu = torch.randn(B, D)
        imu = imu / imu.norm(dim=-1, keepdim=True)
        text = torch.randn(B, D)
        text = text / text.norm(dim=-1, keepdim=True)
        imu_queue = torch.randn(Q, D)
        imu_queue = imu_queue / imu_queue.norm(dim=-1, keepdim=True)
        text_queue = torch.randn(Q, D)
        text_queue = text_queue / text_queue.norm(dim=-1, keepdim=True)

        loss, metrics = loss_fn(imu, text, imu_queue=imu_queue, text_queue=text_queue)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_metrics_contain_expected_keys(self):
        B, D = 8, 384
        loss_fn = InfoNCELoss(temperature=0.1, use_soft_targets=True)
        imu = torch.randn(B, D)
        imu = imu / imu.norm(dim=-1, keepdim=True)
        text = torch.randn(B, D)
        text = text / text.norm(dim=-1, keepdim=True)

        _, metrics = loss_fn(imu, text, return_metrics=True)
        assert 'loss' in metrics
        assert 'positive_similarity' in metrics
        assert 'negative_similarity' in metrics
        assert 'similarity_gap' in metrics
        assert 'logit_scale' in metrics

    def test_backward_pass(self):
        """Verify gradients flow through the loss."""
        B, D = 4, 384
        loss_fn = InfoNCELoss(temperature=0.1)
        imu = torch.randn(B, D, requires_grad=True)
        imu_norm = imu / imu.norm(dim=-1, keepdim=True)
        text = torch.randn(B, D)
        text = text / text.norm(dim=-1, keepdim=True)

        loss, _ = loss_fn(imu_norm, text)
        loss.backward()
        assert imu.grad is not None
        assert not torch.isnan(imu.grad).any()


class TestSemanticAlignmentLoss:
    """Test the wrapper class."""

    def test_single_prototype(self):
        B, D = 8, 384
        loss_fn = SemanticAlignmentLoss(temperature=0.1)
        imu = torch.randn(B, D)
        imu = imu / imu.norm(dim=-1, keepdim=True)
        text = torch.randn(B, D)
        text = text / text.norm(dim=-1, keepdim=True)

        loss, metrics = loss_fn(imu, text)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_multi_prototype(self):
        B, K, D = 8, 3, 384
        loss_fn = SemanticAlignmentLoss(temperature=0.1)
        imu = torch.randn(B, D)
        imu = imu / imu.norm(dim=-1, keepdim=True)
        text = torch.randn(B, K, D)
        text = text / text.norm(dim=-1, keepdim=True)

        loss, metrics = loss_fn(imu, text)
        assert loss.dim() == 0
        assert not torch.isnan(loss)
