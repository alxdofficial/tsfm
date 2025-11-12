"""
IMU Pretraining Dataset

Multi-dataset loader for pretraining IMU activity recognition encoder.
Supports UCI HAR, MHEALTH, PAMAP2, and WISDM datasets with variable channel sampling.
"""

from .multi_dataset_loader import IMUPretrainingDataset
from .augmentations import IMUAugmentation

__all__ = ['IMUPretrainingDataset', 'IMUAugmentation']
