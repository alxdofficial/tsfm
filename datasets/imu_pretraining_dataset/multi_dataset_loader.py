"""
Multi-Dataset Loader for IMU Pretraining.

Loads from UCI HAR, MHEALTH, PAMAP2, and WISDM datasets with:
- Random dataset selection per batch
- Random channel subset selection
- Train/val/test splits (70/15/15)
- Padding and attention masks for variable-length sequences
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random


class IMUPretrainingDataset(Dataset):
    """
    Multi-dataset loader for pretraining IMU encoder.

    Loads from multiple datasets with variable channel sampling.
    """

    def __init__(
        self,
        data_root: str = "/home/alex/code/tsfm/data",
        datasets: List[str] = ['uci_har', 'mhealth', 'pamap2', 'wisdm'],
        split: str = 'train',
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        patch_size_sec: float = 2.0,
        patch_size_per_dataset: Optional[Dict[str, float]] = None,
        min_channels: int = 6,
        max_channels: int = 40,
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
            min_channels: Minimum number of channels to sample
            max_channels: Maximum number of channels to sample
            seed: Random seed for reproducibility
        """
        self.data_root = Path(data_root)
        self.datasets = datasets
        self.split = split
        self.split_ratios = split_ratios
        self.patch_size_sec = patch_size_sec
        self.patch_size_per_dataset = patch_size_per_dataset or {}
        self.min_channels = min_channels
        self.max_channels = max_channels

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
            for session_dir in sorted(sessions_dir.iterdir()):
                if session_dir.is_dir():
                    session_id = session_dir.name
                    self.sessions.append({
                        'dataset': dataset_name,
                        'session_id': session_id,
                        'path': session_dir / "data.parquet",
                        'label': labels.get(session_id, ['unknown'])
                    })

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

        # Get available channels (exclude timestamp_sec)
        available_channels = [col for col in df.columns if col != 'timestamp_sec']

        # Determine channel count for this sample
        max_possible = min(len(available_channels), self.max_channels)
        min_possible = min(self.min_channels, max_possible)
        num_channels = random.randint(min_possible, max_possible)

        # Randomly select channels
        selected_channels = random.sample(available_channels, num_channels)
        selected_channels = sorted(selected_channels)  # Sort for consistency

        # Extract data for selected channels
        data = df[selected_channels].values  # (timesteps, num_channels)

        # Get sampling rate (assume same for all channels in a dataset)
        sampling_rate = dataset_info['sampling_rates'][selected_channels[0]]

        # Convert to tensor
        data = torch.from_numpy(data).float()

        # Create attention mask (all valid for now, padding handled in collate)
        attention_mask = torch.ones(len(data), dtype=torch.bool)

        # Get channel descriptions (with fallback for missing entries)
        channel_descriptions = []
        for ch in selected_channels:
            if ch in dataset_info['channel_info']:
                channel_descriptions.append(dataset_info['channel_info'][ch]['description'])
            else:
                # Fallback for missing channel info
                channel_descriptions.append(f"Channel: {ch}")

        # Get patch size for this dataset (use per-dataset if available, otherwise default)
        patch_size_sec = self.patch_size_per_dataset.get(dataset_name, self.patch_size_sec)

        # Convert label to text string
        # Labels are stored as lists, join them with space if multiple
        label_list = session_info['label']
        if isinstance(label_list, list):
            label_text = ' '.join(str(l) for l in label_list)
        else:
            label_text = str(label_list)

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


def create_dataloaders(
    data_root: str = "/home/alex/code/tsfm/data",
    datasets: List[str] = ['uci_har', 'mhealth', 'pamap2', 'wisdm'],
    batch_size: int = 32,
    num_workers: int = 4,
    patch_size_sec: float = 2.0,
    patch_size_per_dataset: Optional[Dict[str, float]] = None,
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
        seed=seed
    )

    val_dataset = IMUPretrainingDataset(
        data_root=data_root,
        datasets=datasets,
        split='val',
        patch_size_sec=patch_size_sec,
        patch_size_per_dataset=patch_size_per_dataset,
        seed=seed
    )

    test_dataset = IMUPretrainingDataset(
        data_root=data_root,
        datasets=datasets,
        split='test',
        patch_size_sec=patch_size_sec,
        patch_size_per_dataset=patch_size_per_dataset,
        seed=seed
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=IMUPretrainingDataset.collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=IMUPretrainingDataset.collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=IMUPretrainingDataset.collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset loader
    print("Testing IMU Pretraining Dataset...")

    # Create dataset
    dataset = IMUPretrainingDataset(
        split='train',
        patch_size_sec=2.0,
        seed=42
    )

    print(f"\nDataset size: {len(dataset)}")

    # Test single sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Data shape: {sample['data'].shape}")
    print(f"  Attention mask shape: {sample['attention_mask'].shape}")
    print(f"  Dataset: {sample['metadata']['dataset']}")
    print(f"  Channels: {sample['metadata']['num_channels']}")
    print(f"  Selected channels: {sample['metadata']['channels'][:3]}...")
    print(f"  Sampling rate: {sample['metadata']['sampling_rate_hz']} Hz")

    # Test dataloader with batching
    print("\nTesting dataloader with batching...")
    train_loader, _, _ = create_dataloaders(batch_size=4, num_workers=0)

    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Data: {batch['data'].shape}")
    print(f"  Attention mask: {batch['attention_mask'].shape}")
    print(f"  Channel mask: {batch['channel_mask'].shape}")

    print("\nBatch metadata:")
    for i, meta in enumerate(batch['metadata'][:2]):
        print(f"  Sample {i}: {meta['dataset']}, {meta['num_channels']} channels")

    print("\n✓ Dataset loader test passed!")
