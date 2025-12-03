"""
Multi-Dataset Loader for IMU Pretraining.

Loads from multiple activity recognition datasets with:
- Random dataset selection per batch
- Random channel subset selection
- Train/val/test splits (70/15/15)
- Padding and attention masks for variable-length sequences

Supported datasets:
- UCI HAR: 6 activities, 6 channels (acc + gyro), 50Hz
- MHEALTH: 12 activities, 6 channels (acc + gyro), 50Hz
- PAMAP2: 12 activities, 27 IMU channels, 100Hz
- WISDM: 18 activities, 6 channels (acc + gyro), 20Hz
- UniMiB SHAR: 9 activities, 3 channels (acc only), 50Hz
- HHAR: 6 activities, 6 channels (acc + gyro), 50Hz
"""

import re
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

from datasets.imu_pretraining_dataset.label_augmentation import augment_label


def group_channels_by_sensor(channel_names: List[str]) -> Dict[str, List[str]]:
    """
    Group channels by sensor type (accelerometer, gyroscope, etc.) and location.

    Channels are grouped by their prefix, excluding the axis suffix (x/y/z or 1/2/3/4).
    For example:
        - acc_x, acc_y, acc_z -> group "acc"
        - hand_gyro_x, hand_gyro_y, hand_gyro_z -> group "hand_gyro"
        - chest_acc_x, chest_acc_y, chest_acc_z -> group "chest_acc"

    Args:
        channel_names: List of channel names

    Returns:
        Dict mapping group name to list of channel names in that group
    """
    groups = {}

    # Pattern to match axis suffix: _x, _y, _z, _1, _2, _3, _4
    axis_pattern = re.compile(r'_([xyz]|[1-4])$')

    for channel in channel_names:
        # Extract group name by removing axis suffix
        match = axis_pattern.search(channel)
        if match:
            group_name = channel[:match.start()]
        else:
            # Channel without axis suffix (treat as its own group)
            group_name = channel

        if group_name not in groups:
            groups[group_name] = []
        groups[group_name].append(channel)

    # Sort channels within each group for consistency (x before y before z, etc.)
    for group_name in groups:
        groups[group_name] = sorted(groups[group_name])

    return groups


def select_channel_groups(
    channel_groups: Dict[str, List[str]],
    min_groups: int = 1,
    max_groups: int = None
) -> List[str]:
    """
    Randomly select channel groups and return flattened channel list.

    Args:
        channel_groups: Dict mapping group name to list of channels
        min_groups: Minimum number of groups to select
        max_groups: Maximum number of groups to select (None = all)

    Returns:
        Flattened list of selected channel names
    """
    group_names = list(channel_groups.keys())

    if max_groups is None:
        max_groups = len(group_names)

    # Clamp to available groups
    max_groups = min(max_groups, len(group_names))
    min_groups = min(min_groups, max_groups)

    # Randomly select number of groups
    num_groups = random.randint(min_groups, max_groups)

    # Randomly select which groups
    selected_group_names = random.sample(group_names, num_groups)

    # Flatten to channel list
    selected_channels = []
    for group_name in sorted(selected_group_names):  # Sort for consistency
        selected_channels.extend(channel_groups[group_name])

    return selected_channels


class IMUPretrainingDataset(Dataset):
    """
    Multi-dataset loader for pretraining IMU encoder.

    Loads from multiple datasets with variable channel sampling.
    Uses dataset-specific label augmentation with synonyms and templates.
    """

    def __init__(
        self,
        data_root: str = "/home/alex/code/tsfm/data",
        datasets: List[str] = ['uci_har', 'mhealth', 'pamap2', 'wisdm', 'unimib_shar', 'hhar'],
        split: str = 'train',
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        patch_size_sec: float = 2.0,
        patch_size_per_dataset: Optional[Dict[str, float]] = None,
        min_channel_groups: int = 1,  # Minimum number of sensor groups to select
        max_channel_groups: int = None,  # Maximum groups (None = all available)
        max_sessions_per_dataset: Optional[int] = None,  # Limit sessions per dataset for faster experiments
        seed: int = 42
    ):
        """
        Args:
            data_root: Root directory containing dataset folders
            datasets: List of dataset names to use
            split: 'train', 'val', or 'test'
            split_ratios: (train, val, test) split ratios
            patch_size_sec: Default patch size in seconds (used if patch_size_per_dataset not provided)
            patch_size_per_dataset: Optional dict mapping dataset name to patch size in seconds
            min_channel_groups: Minimum number of sensor groups to sample (e.g., acc, gyro)
            max_channel_groups: Maximum number of sensor groups (None = all available)
            max_sessions_per_dataset: Maximum sessions to load per dataset (None = all).
                                      Useful for faster experimentation with large datasets.
            seed: Random seed for reproducibility

        Note on channel groups:
            Channels are grouped by sensor type and location. For example:
            - acc_x, acc_y, acc_z -> group "acc"
            - hand_gyro_x, hand_gyro_y, hand_gyro_z -> group "hand_gyro"

            When sampling, entire groups are selected (not individual channels).
            This ensures physically meaningful data (e.g., all 3 axes of an accelerometer).
        """
        self.data_root = Path(data_root)
        self.datasets = datasets
        self.split = split
        self.split_ratios = split_ratios
        self.patch_size_sec = patch_size_sec
        self.patch_size_per_dataset = patch_size_per_dataset or {}
        self.min_channel_groups = min_channel_groups
        self.max_channel_groups = max_channel_groups
        self.max_sessions_per_dataset = max_sessions_per_dataset

        # Set random seed
        random.seed(seed)
        np.random.seed(seed)

        # Load dataset metadata
        self.dataset_info = {}
        self.sessions = []
        self._load_datasets()

        # Create splits
        self._create_splits()

        print(f"Loaded {len(self.sessions)} sessions for {split} split from {len(self.datasets)} datasets")

    def _load_datasets(self):
        """Load metadata from all datasets."""
        for dataset_name in self.datasets:
            dataset_path = self.data_root / dataset_name

            if not dataset_path.exists():
                print(f"Warning: Dataset {dataset_name} not found at {dataset_path}")
                continue

            # Load manifest
            manifest_path = dataset_path / "manifest.json"
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            # Load labels
            labels_path = dataset_path / "labels.json"
            with open(labels_path, 'r') as f:
                labels = json.load(f)

            # Store dataset info
            self.dataset_info[dataset_name] = {
                'manifest': manifest,
                'labels': labels,
                'path': dataset_path,
                'channels': [ch['name'] for ch in manifest['channels']],
                'channel_info': {ch['name']: ch for ch in manifest['channels']},
                'sampling_rates': {ch['name']: ch['sampling_rate_hz'] for ch in manifest['channels']}
            }

            # Collect all sessions
            sessions_dir = dataset_path / "sessions"
            dataset_sessions = []
            for session_dir in sorted(sessions_dir.iterdir()):
                if session_dir.is_dir():
                    session_id = session_dir.name
                    dataset_sessions.append({
                        'dataset': dataset_name,
                        'session_id': session_id,
                        'path': session_dir / "data.parquet",
                        'label': labels.get(session_id, ['unknown'])
                    })

            # Apply max_sessions_per_dataset limit if specified
            if self.max_sessions_per_dataset is not None and len(dataset_sessions) > self.max_sessions_per_dataset:
                # Shuffle before limiting to get diverse samples
                random.shuffle(dataset_sessions)
                dataset_sessions = dataset_sessions[:self.max_sessions_per_dataset]
                print(f"  {dataset_name}: limited to {self.max_sessions_per_dataset} sessions (from {len(list(sessions_dir.iterdir()))})")

            self.sessions.extend(dataset_sessions)

    def _create_splits(self):
        """Create train/val/test splits."""
        # Shuffle sessions
        random.shuffle(self.sessions)

        # Calculate split indices
        n_total = len(self.sessions)
        n_train = int(n_total * self.split_ratios[0])
        n_val = int(n_total * self.split_ratios[1])

        # Split sessions
        if self.split == 'train':
            self.sessions = self.sessions[:n_train]
        elif self.split == 'val':
            self.sessions = self.sessions[n_train:n_train + n_val]
        elif self.split == 'test':
            self.sessions = self.sessions[n_train + n_val:]
        else:
            raise ValueError(f"Invalid split: {self.split}")

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample with random channel selection.

        Returns:
            Dictionary with:
            - data: Tensor (timesteps, num_channels)
            - attention_mask: Boolean tensor (timesteps,) - True=valid, False=padding
            - metadata: Dict with dataset, session_id, channels, sampling_rate, etc.
        """
        session_info = self.sessions[idx]
        dataset_name = session_info['dataset']
        dataset_info = self.dataset_info[dataset_name]

        # Load session data
        df = pd.read_parquet(session_info['path'])

        # Get available channels (exclude timestamp_sec and non-IMU channels)
        # Filter to only IMU channels: accelerometer, gyroscope, magnetometer, orientation
        # This excludes heart_rate (9 Hz in PAMAP2) and temperature sensors
        IMU_PATTERNS = ['acc', 'gyro', 'mag', 'ori']
        available_channels = [
            col for col in df.columns
            if col != 'timestamp_sec'
            and any(p in col.lower() for p in IMU_PATTERNS)
        ]

        # Group channels by sensor type and location
        # e.g., acc_x, acc_y, acc_z -> group "acc"
        # e.g., hand_gyro_x, hand_gyro_y, hand_gyro_z -> group "hand_gyro"
        channel_groups = group_channels_by_sensor(available_channels)

        # Randomly select channel groups (not individual channels)
        # This ensures we get complete sensor data (all axes together)
        selected_channels = select_channel_groups(
            channel_groups,
            min_groups=self.min_channel_groups,
            max_groups=self.max_channel_groups
        )
        num_channels = len(selected_channels)

        # Extract data for selected channels
        data = df[selected_channels].values  # (timesteps, num_channels)

        # Get sampling rate (assume same for all channels in a dataset)
        sampling_rate = dataset_info['sampling_rates'][selected_channels[0]]

        # Convert to tensor
        data = torch.from_numpy(data).float()

        # Create attention mask (all valid for now, padding handled in collate)
        attention_mask = torch.ones(len(data), dtype=torch.bool)

        # Get dataset description for context
        dataset_desc = dataset_info['manifest'].get('description', '')

        # Get channel descriptions with dataset context prepended
        channel_descriptions = []
        for ch in selected_channels:
            if ch in dataset_info['channel_info']:
                ch_desc = dataset_info['channel_info'][ch]['description']
            else:
                # Fallback for missing channel info
                ch_desc = f"Channel: {ch}"

            # Prepend dataset description for richer semantic context
            if dataset_desc:
                full_desc = f"{dataset_desc} {ch_desc}"
            else:
                full_desc = ch_desc
            channel_descriptions.append(full_desc)

        # Get patch size for this dataset (use per-dataset if available, otherwise default)
        patch_size_sec = self.patch_size_per_dataset.get(dataset_name, self.patch_size_sec)

        # Convert label to text string
        # Labels are stored as lists, join them with space if multiple
        label_list = session_info['label']
        if isinstance(label_list, list):
            base_label = ' '.join(str(l) for l in label_list)
        else:
            base_label = str(label_list)

        # Apply dataset-specific label augmentation (only for training split)
        # Uses synonyms and templates tailored to each dataset's activities
        augmentation_rate = 0.8 if self.split == 'train' else 0.0
        label_text = augment_label(
            label=base_label,
            dataset_name=dataset_name,
            augmentation_rate=augmentation_rate,
            use_synonyms=True,
            use_templates=True
        )

        return {
            'data': data,
            'attention_mask': attention_mask,
            'label_text': label_text,  # Add label text at top level
            'metadata': {
                'dataset': dataset_name,
                'session_id': session_info['session_id'],
                'label': session_info['label'],
                'label_text': label_text,  # Also keep in metadata for backwards compatibility
                'channels': selected_channels,
                'channel_descriptions': channel_descriptions,
                'sampling_rate_hz': sampling_rate,
                'patch_size_sec': patch_size_sec,
                'num_channels': num_channels
            }
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Collate batch with padding for variable-length sequences and channels.

        Args:
            batch: List of samples from __getitem__

        Returns:
            Batched dictionary with:
            - data: (batch, max_timesteps, max_channels) with zero padding
            - attention_mask: (batch, max_timesteps) Boolean mask
            - channel_mask: (batch, max_channels) Boolean mask for valid channels
            - label_texts: List of label text strings
            - metadata: List of metadata dicts
        """
        # Find max dimensions
        max_timesteps = max(sample['data'].shape[0] for sample in batch)
        max_channels = max(sample['data'].shape[1] for sample in batch)

        batch_size = len(batch)

        # Initialize padded tensors
        padded_data = torch.zeros(batch_size, max_timesteps, max_channels)
        attention_mask = torch.zeros(batch_size, max_timesteps, dtype=torch.bool)
        channel_mask = torch.zeros(batch_size, max_channels, dtype=torch.bool)

        metadata_list = []
        label_texts = []

        for i, sample in enumerate(batch):
            data = sample['data']
            timesteps, num_channels = data.shape

            # Copy data
            padded_data[i, :timesteps, :num_channels] = data

            # Set masks
            attention_mask[i, :timesteps] = True
            channel_mask[i, :num_channels] = True

            metadata_list.append(sample['metadata'])
            label_texts.append(sample['label_text'])

        return {
            'data': padded_data,
            'attention_mask': attention_mask,
            'channel_mask': channel_mask,
            'label_texts': label_texts,
            'metadata': metadata_list
        }


def worker_init_fn(worker_id: int) -> None:
    """
    Initialize random seeds for DataLoader workers to ensure reproducibility.

    Each worker gets a unique seed based on worker_id to ensure:
    1. Different workers produce different random sequences
    2. Results are reproducible across runs with the same seed

    Args:
        worker_id: Worker ID (0 to num_workers-1)
    """
    # Get the dataset instance from the worker
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        # Single-process data loading, no need to reseed
        return

    # Get base seed from dataset
    dataset = worker_info.dataset
    base_seed = getattr(dataset, 'seed', 42)

    # Create unique seed for this worker
    worker_seed = base_seed + worker_id

    # Seed all random number generators
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def create_dataloaders(
    data_root: str = "/home/alex/code/tsfm/data",
    datasets: List[str] = ['uci_har', 'mhealth', 'pamap2', 'wisdm', 'unimib_shar', 'hhar'],
    batch_size: int = 32,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    patch_size_sec: float = 2.0,
    patch_size_per_dataset: Optional[Dict[str, float]] = None,
    max_sessions_per_dataset: Optional[int] = None,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, and test dataloaders.

    Args:
        data_root: Root directory containing datasets
        datasets: List of dataset names
        batch_size: Batch size
        num_workers: Number of worker processes
        patch_size_sec: Default patch duration in seconds
        patch_size_per_dataset: Optional dict mapping dataset name to patch size
        max_sessions_per_dataset: Max sessions per dataset (None = all)
        seed: Random seed

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = IMUPretrainingDataset(
        data_root=data_root,
        datasets=datasets,
        split='train',
        patch_size_sec=patch_size_sec,
        patch_size_per_dataset=patch_size_per_dataset,
        max_sessions_per_dataset=max_sessions_per_dataset,
        seed=seed
    )

    val_dataset = IMUPretrainingDataset(
        data_root=data_root,
        datasets=datasets,
        split='val',
        patch_size_sec=patch_size_sec,
        patch_size_per_dataset=patch_size_per_dataset,
        max_sessions_per_dataset=max_sessions_per_dataset,
        seed=seed
    )

    test_dataset = IMUPretrainingDataset(
        data_root=data_root,
        datasets=datasets,
        split='test',
        patch_size_sec=patch_size_sec,
        patch_size_per_dataset=patch_size_per_dataset,
        max_sessions_per_dataset=max_sessions_per_dataset,
        seed=seed
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=IMUPretrainingDataset.collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=IMUPretrainingDataset.collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=IMUPretrainingDataset.collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import sys

    # Check if debug mode requested
    debug_labels = '--debug-labels' in sys.argv or '--labels' in sys.argv or len(sys.argv) == 1

    if debug_labels:
        print("=" * 80)
        print("LABEL AUGMENTATION DEBUG MODE")
        print("=" * 80)

        # Create dataset
        dataset = IMUPretrainingDataset(split='train', seed=42)

        print(f"\nDataset size: {len(dataset)} samples")
        print(f"Augmentation rate: 80% for training split")

        # Sample 50 items and track augmentations
        print("\n" + "=" * 80)
        print("SAMPLING 50 EXAMPLES (showing original label → augmented text)")
        print("=" * 80)

        from collections import defaultdict
        augmentation_examples = defaultdict(list)
        dataset_counts = defaultdict(int)

        for i in range(50):
            sample = dataset[i]
            original_label = sample['metadata']['label']
            # Handle label list or string
            if isinstance(original_label, list):
                original_label = ' '.join(str(l) for l in original_label)
            else:
                original_label = str(original_label)
            augmented_label = sample['label_text']
            dataset_name = sample['metadata']['dataset']

            # Track
            dataset_counts[dataset_name] += 1
            key = (dataset_name, original_label)
            if len(augmentation_examples[key]) < 5:  # Keep up to 5 variations
                augmentation_examples[key].append(augmented_label)

            # Print
            if original_label != augmented_label:
                print(f"{i+1:2d}. [{dataset_name:10s}] {original_label:25s} → {augmented_label}")
            else:
                print(f"{i+1:2d}. [{dataset_name:10s}] {original_label:25s} (no augmentation)")

        # Show dataset distribution
        print("\n" + "=" * 80)
        print("DATASET DISTRIBUTION IN SAMPLES:")
        print("=" * 80)
        for dataset_name, count in sorted(dataset_counts.items()):
            print(f"  {dataset_name:15s}: {count:2d} samples ({count/50*100:.0f}%)")

        # Show all variations collected for each label
        print("\n" + "=" * 80)
        print("LABEL VARIATION EXAMPLES (grouped by dataset and activity):")
        print("=" * 80)

        for (dataset_name, original_label), variations in sorted(augmentation_examples.items()):
            if len(variations) > 1 or variations[0] != original_label:
                print(f"\n{dataset_name.upper()} - '{original_label}':")
                for j, var in enumerate(variations, 1):
                    marker = "✓" if var != original_label else "○"
                    print(f"  {marker} {var}")

        # Sample more to show diversity
        print("\n" + "=" * 80)
        print("TESTING AUGMENTATION DIVERSITY (100 more samples per dataset)")
        print("=" * 80)

        from collections import Counter

        for target_dataset in ['uci_har', 'mhealth', 'pamap2', 'wisdm', 'unimib_shar']:
            variations_seen = []
            samples_checked = 0
            idx = 0

            while samples_checked < 100 and idx < len(dataset):
                sample = dataset[idx]
                if sample['metadata']['dataset'] == target_dataset:
                    variations_seen.append(sample['label_text'])
                    samples_checked += 1
                idx += 1

            unique_count = len(set(variations_seen))
            total_count = len(variations_seen)

            print(f"\n{target_dataset.upper()}:")
            print(f"  Samples checked: {total_count}")
            print(f"  Unique texts:    {unique_count}")
            print(f"  Diversity:       {unique_count/total_count*100:.1f}%")

            # Show most common variations
            most_common = Counter(variations_seen).most_common(5)
            print(f"  Most common variations:")
            for text, count in most_common:
                print(f"    {count:2d}× {text}")

        print("\n" + "=" * 80)
        print("✓ Label augmentation debug complete!")
        print("=" * 80)

    else:
        # Standard test mode
        print("=" * 80)
        print("STANDARD DATASET LOADER TEST")
        print("=" * 80)
        print("(Run with --debug-labels to see augmentation details)")

        dataset = IMUPretrainingDataset(split='train', seed=42)
        print(f"\nDataset size: {len(dataset)}")

        # Test single sample
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  Data shape: {sample['data'].shape}")
        print(f"  Attention mask shape: {sample['attention_mask'].shape}")
        print(f"  Dataset: {sample['metadata']['dataset']}")
        print(f"  Channels: {sample['metadata']['num_channels']}")
        print(f"  Label: {sample['label_text']}")

        # Test dataloader with batching
        print("\nTesting dataloader with batching...")
        train_loader, _, _ = create_dataloaders(batch_size=4, num_workers=0)

        batch = next(iter(train_loader))
        print(f"\nBatch shapes:")
        print(f"  Data: {batch['data'].shape}")
        print(f"  Attention mask: {batch['attention_mask'].shape}")
        print(f"  Channel mask: {batch['channel_mask'].shape}")

        print("\nBatch labels:")
        for i, (label, meta) in enumerate(zip(batch['label_texts'], batch['metadata'])):
            print(f"  {i+1}. [{meta['dataset']:10s}] {label}")

        print("\n" + "=" * 80)
        print("✓ Dataset loader test passed!")
        print("=" * 80)
