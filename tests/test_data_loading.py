"""Test 8: Dataset loading pipeline.

Tests collate_patches_fn, compute_group_weights, and ChannelBucketBatchSampler
with synthetic data (no data directory required).
"""

import sys
from pathlib import Path
import torch
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.imu_pretraining_dataset.multi_dataset_loader import (
    group_channels_by_sensor,
)


class TestGroupChannelsBySensor:
    """Test channel grouping utility."""

    def test_basic_triads(self):
        channels = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        groups = group_channels_by_sensor(channels)
        assert 'acc' in groups
        assert 'gyro' in groups
        assert groups['acc'] == ['acc_x', 'acc_y', 'acc_z']
        assert groups['gyro'] == ['gyro_x', 'gyro_y', 'gyro_z']

    def test_body_location_prefix(self):
        channels = ['chest_acc_x', 'chest_acc_y', 'chest_acc_z']
        groups = group_channels_by_sensor(channels)
        assert 'chest_acc' in groups
        assert len(groups['chest_acc']) == 3

    def test_quaternion_channels(self):
        channels = ['ori_1', 'ori_2', 'ori_3', 'ori_4']
        groups = group_channels_by_sensor(channels)
        assert 'ori' in groups
        assert len(groups['ori']) == 4

    def test_single_channel_no_suffix(self):
        channels = ['temperature']
        groups = group_channels_by_sensor(channels)
        assert 'temperature' in groups
        assert groups['temperature'] == ['temperature']

    def test_sorted_within_groups(self):
        channels = ['acc_z', 'acc_x', 'acc_y']
        groups = group_channels_by_sensor(channels)
        assert groups['acc'] == ['acc_x', 'acc_y', 'acc_z']


class TestCollatePatchesFn:
    """Test the collate function with synthetic batch data."""

    def _make_sample(self, num_patches, target_patch_size, num_channels, label_text='walking'):
        return {
            'patches': torch.randn(num_patches, target_patch_size, num_channels),
            'metadata': {
                'dataset': 'test',
                'channel_descriptions': [f'ch_{i}' for i in range(num_channels)],
            },
            'label_text': label_text,
        }

    def test_uniform_batch(self):
        """All samples same shape -> no padding needed."""
        from datasets.imu_pretraining_dataset.multi_dataset_loader import IMUPretrainingDataset

        batch = [self._make_sample(5, 64, 9, 'walking') for _ in range(4)]
        result = IMUPretrainingDataset.collate_patches_fn(batch)

        assert result['patches'].shape == (4, 5, 64, 9)
        assert result['patch_mask'].shape == (4, 5)
        assert result['channel_mask'].shape == (4, 9)
        assert result['patch_mask'].all()
        assert result['channel_mask'].all()
        assert len(result['label_texts']) == 4

    def test_variable_patches(self):
        """Samples with different patch counts -> pad to max."""
        from datasets.imu_pretraining_dataset.multi_dataset_loader import IMUPretrainingDataset

        batch = [
            self._make_sample(3, 64, 6, 'walking'),
            self._make_sample(7, 64, 6, 'running'),
        ]
        result = IMUPretrainingDataset.collate_patches_fn(batch)

        assert result['patches'].shape == (2, 7, 64, 6)
        assert result['patch_mask'][0, :3].all()
        assert not result['patch_mask'][0, 3:].any()
        assert result['patch_mask'][1, :7].all()

    def test_variable_channels(self):
        """Samples with different channel counts -> pad to max."""
        from datasets.imu_pretraining_dataset.multi_dataset_loader import IMUPretrainingDataset

        batch = [
            self._make_sample(5, 64, 3, 'sitting'),
            self._make_sample(5, 64, 9, 'standing'),
        ]
        result = IMUPretrainingDataset.collate_patches_fn(batch)

        assert result['patches'].shape == (2, 5, 64, 9)
        assert result['channel_mask'][0, :3].all()
        assert not result['channel_mask'][0, 3:].any()
        assert result['channel_mask'][1, :9].all()


class TestComputeGroupWeights:
    """Test group weight computation with synthetic data.

    Since compute_group_weights is a method on IMUPretrainingDataset,
    we test the math logic directly with synthetic data.
    """

    def _compute_weights_from_labels(self, labels, temperature=0.0):
        """Compute weights using the same math as compute_group_weights."""
        from collections import defaultdict
        from datasets.imu_pretraining_dataset.label_groups import get_group_for_label

        group_counts = defaultdict(int)
        sample_groups = []
        for label in labels:
            group = get_group_for_label(label)
            group_counts[group] += 1
            sample_groups.append(group)

        weights = torch.zeros(len(labels))
        alpha = temperature  # temperature=0 -> alpha=0 -> uniform over groups
        for i, group in enumerate(sample_groups):
            count = group_counts[group]
            if alpha == 0:
                weights[i] = 1.0 / count
            elif alpha == 1:
                weights[i] = 1.0
            else:
                weights[i] = count ** (alpha - 1)

        # Normalize so weights sum to num_samples
        weights = weights / weights.sum() * len(labels)
        return weights

    def test_uniform_with_alpha_zero(self):
        """alpha=0 (temperature=0) -> balanced groups (uniform over groups)."""
        labels = ['walking'] * 100 + ['sitting'] * 10
        weights = self._compute_weights_from_labels(labels, temperature=0.0)

        # Walking (100 samples) should get much lower weight per sample than sitting (10)
        walking_weight = weights[0].item()
        sitting_weight = weights[100].item()
        assert sitting_weight > walking_weight, \
            "Rare group should get higher weight with alpha=0"

    def test_no_rebalancing_with_alpha_one(self):
        """alpha=1 (temperature=1) -> no rebalancing (uniform over samples)."""
        labels = ['walking'] * 100 + ['sitting'] * 10
        weights = self._compute_weights_from_labels(labels, temperature=1.0)

        # All weights should be roughly the same
        assert torch.allclose(weights, weights[0].expand_as(weights), atol=1e-5), \
            "With alpha=1, all weights should be roughly the same"

    def test_sqrt_with_alpha_half(self):
        """alpha=0.5 -> sqrt balancing (compromise)."""
        labels = ['walking'] * 100 + ['sitting'] * 10
        weights = self._compute_weights_from_labels(labels, temperature=0.5)

        walking_weight = weights[0].item()
        sitting_weight = weights[100].item()
        assert sitting_weight > walking_weight, \
            "Rare group should still get higher weight with alpha=0.5"

        # But the ratio should be less extreme than alpha=0
        weights_balanced = self._compute_weights_from_labels(labels, temperature=0.0)
        ratio_balanced = weights_balanced[100].item() / weights_balanced[0].item()
        ratio_sqrt = sitting_weight / walking_weight
        assert ratio_sqrt < ratio_balanced, \
            "sqrt balancing should be less extreme than full balancing"


class TestChannelBucketBatchSampler:
    """Test channel bucket batch sampler."""

    def test_produces_valid_batches(self):
        from training_scripts.human_activity_recognition.semantic_alignment_train import ChannelBucketBatchSampler

        # Create synthetic channel counts: some 3ch, some 6ch, some 9ch
        channel_counts = [3] * 20 + [6] * 30 + [9] * 50
        sample_weights = torch.ones(100)
        batch_size = 8

        sampler = ChannelBucketBatchSampler(
            channel_counts=channel_counts,
            sample_weights=sample_weights,
            batch_size=batch_size,
        )

        batches = list(sampler)
        assert len(batches) > 0, "Should produce at least one batch"

        for batch in batches:
            assert len(batch) == batch_size, f"Batch size should be {batch_size}, got {len(batch)}"
            # All indices should be valid
            for idx in batch:
                assert 0 <= idx < 100

    def test_batches_have_same_channel_count(self):
        """All samples in a batch should have the same channel count."""
        from training_scripts.human_activity_recognition.semantic_alignment_train import ChannelBucketBatchSampler

        channel_counts = [3] * 20 + [6] * 30 + [9] * 50
        sample_weights = torch.ones(100)
        batch_size = 4

        sampler = ChannelBucketBatchSampler(
            channel_counts=channel_counts,
            sample_weights=sample_weights,
            batch_size=batch_size,
        )

        for batch in sampler:
            ch_counts_in_batch = [channel_counts[idx] for idx in batch]
            assert len(set(ch_counts_in_batch)) == 1, \
                f"All samples in batch should have same channel count, got {ch_counts_in_batch}"

    def test_respects_num_samples(self):
        """Total samples seen should match num_samples."""
        from training_scripts.human_activity_recognition.semantic_alignment_train import ChannelBucketBatchSampler

        channel_counts = [6] * 100
        sample_weights = torch.ones(100)
        batch_size = 10
        num_samples = 50

        sampler = ChannelBucketBatchSampler(
            channel_counts=channel_counts,
            sample_weights=sample_weights,
            batch_size=batch_size,
            num_samples=num_samples,
        )

        batches = list(sampler)
        total_samples = sum(len(b) for b in batches)
        assert total_samples == num_samples, \
            f"Expected {num_samples} total samples, got {total_samples}"
