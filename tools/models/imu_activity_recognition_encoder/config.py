"""
Default configuration for IMU Activity Recognition Encoder.

This file defines the default hyperparameters for the model.
You can override these when instantiating the encoder.
"""

from typing import Dict, Any


# Default configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    # Model architecture
    "d_model": 384,  # Match Sentence-BERT output dimension (all-MiniLM-L6-v2)
    "num_heads": 8,  # 48 dims per head
    "num_temporal_layers": 4,
    "dim_feedforward": 1536,  # 4x d_model
    "dropout": 0.1,
    "use_cross_channel": False,  # Enable cross-channel attention (default: False for backward compatibility)

    # CNN parameters
    "cnn_channels": [64, 128],
    "cnn_kernel_sizes": [3, 5, 7],

    # Preprocessing parameters
    "target_patch_size": 64,
    "normalization_method": "zscore",
    "interpolation_method": "linear",

    # Positional encoding
    "temporal_init_scale": 0.1,
    "channel_init_scale": 0.1,
    "use_channel_encoding": True,
    "sentence_bert_model": "all-MiniLM-L6-v2",

    # Other
    "max_patches": 5000
}


# Small model configuration (for quick experiments)
SMALL_CONFIG: Dict[str, Any] = {
    "d_model": 64,
    "num_heads": 4,
    "num_temporal_layers": 2,
    "dim_feedforward": 256,
    "dropout": 0.1,
    "use_cross_channel": False,  # Enable cross-channel attention
    "cnn_channels": [32, 64],
    "cnn_kernel_sizes": [3, 5, 7],
    "target_patch_size": 64,
    "normalization_method": "zscore",
    "interpolation_method": "linear",
    "temporal_init_scale": 0.1,
    "channel_init_scale": 0.1,
    "use_channel_encoding": False,  # Disable for speed
    "max_patches": 5000
}


# Large model configuration (for best performance)
LARGE_CONFIG: Dict[str, Any] = {
    "d_model": 256,
    "num_heads": 8,
    "num_temporal_layers": 6,
    "dim_feedforward": 1024,
    "dropout": 0.1,
    "use_cross_channel": False,  # Enable cross-channel attention
    "cnn_channels": [64, 128, 256],
    "cnn_kernel_sizes": [3, 5, 7],
    "target_patch_size": 64,
    "normalization_method": "zscore",
    "interpolation_method": "linear",
    "temporal_init_scale": 0.1,
    "channel_init_scale": 0.1,
    "use_channel_encoding": True,
    "sentence_bert_model": "all-MiniLM-L6-v2",
    "max_patches": 5000
}


def get_config(size: str = "default") -> Dict[str, Any]:
    """
    Get a model configuration by size.

    Args:
        size: One of "small", "default", "large"

    Returns:
        Configuration dictionary

    Example:
        >>> config = get_config("small")
        >>> encoder = IMUActivityRecognitionEncoder(**config)
    """
    configs = {
        "small": SMALL_CONFIG,
        "default": DEFAULT_CONFIG,
        "large": LARGE_CONFIG
    }

    if size not in configs:
        raise ValueError(f"Unknown config size: {size}. Choose from {list(configs.keys())}")

    return configs[size].copy()
