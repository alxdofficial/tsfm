"""
Model configuration tiers for TSFM.

Three deployment tiers:
  - Small  (d=384, 4 layers)  : ~21M trainable, ~44M inference  — mobile-friendly
  - Medium (d=768, 8 layers)  : ~122M trainable, ~232M inference — balanced
  - Large  (d=1024, 12 layers): ~350M trainable, ~700M inference — maximum quality

Text encoder weights are frozen and used only during training to produce
channel/label embeddings. At inference, precomputed embeddings are stored
as constants, so the text encoder is never deployed on-device.
"""

from typing import Dict, Any


# ---------------------------------------------------------------------------
# Small: d=384, 4 layers, all-MiniLM-L6-v2 (384-dim, 22.7M params)
# Trainable: ~21M | Inference (excl. precomputed text): ~21M
# Suitable for smartphone / edge deployment.
# ---------------------------------------------------------------------------
SMALL_CONFIG: Dict[str, Any] = {
    "d_model": 384,
    "num_heads": 8,              # 48 dims per head
    "num_temporal_layers": 4,
    "dim_feedforward": 1536,     # 4x d_model
    "dropout": 0.1,
    "use_cross_channel": True,
    "cnn_channels": [32, 64],
    "cnn_kernel_sizes": [5],
    "target_patch_size": 64,
    "normalization_method": "zscore",
    "interpolation_method": "linear",
    "temporal_init_scale": 0.1,
    "channel_init_scale": 0.1,
    "use_channel_encoding": True,
    "sentence_bert_model": "all-MiniLM-L6-v2",   # 384-dim embeddings
    "max_patches": 5000
}


# ---------------------------------------------------------------------------
# Medium: d=768, 8 layers, all-mpnet-base-v2 (768-dim, 109M params)
# Trainable: ~122M | Inference (incl. frozen text encoder): ~232M
# Balanced quality-to-cost ratio for GPU servers.
# ---------------------------------------------------------------------------
MEDIUM_CONFIG: Dict[str, Any] = {
    "d_model": 768,
    "num_heads": 12,             # 64 dims per head
    "num_temporal_layers": 8,
    "dim_feedforward": 3072,     # 4x d_model
    "dropout": 0.1,
    "use_cross_channel": True,
    "cnn_channels": [64, 128],
    "cnn_kernel_sizes": [5],
    "target_patch_size": 64,
    "normalization_method": "zscore",
    "interpolation_method": "linear",
    "temporal_init_scale": 0.1,
    "channel_init_scale": 0.1,
    "use_channel_encoding": True,
    "sentence_bert_model": "all-mpnet-base-v2",   # 768-dim embeddings
    "max_patches": 5000
}


# ---------------------------------------------------------------------------
# Large: d=1024, 12 layers, BAAI/bge-large-en-v1.5 (1024-dim, 335M params)
# Trainable: ~350M | Inference (incl. frozen text encoder): ~700M
# Maximum quality; requires >=40GB GPU for training.
# ---------------------------------------------------------------------------
LARGE_CONFIG: Dict[str, Any] = {
    "d_model": 1024,
    "num_heads": 16,             # 64 dims per head
    "num_temporal_layers": 12,
    "dim_feedforward": 4096,     # 4x d_model
    "dropout": 0.1,
    "use_cross_channel": True,
    "cnn_channels": [128, 256],
    "cnn_kernel_sizes": [5],
    "target_patch_size": 64,
    "normalization_method": "zscore",
    "interpolation_method": "linear",
    "temporal_init_scale": 0.1,
    "channel_init_scale": 0.1,
    "use_channel_encoding": True,
    "sentence_bert_model": "BAAI/bge-large-en-v1.5",  # 1024-dim embeddings
    "max_patches": 5000
}


# ---------------------------------------------------------------------------
# Tiny: d=64 (for unit tests and quick experiments only)
# ---------------------------------------------------------------------------
TINY_CONFIG: Dict[str, Any] = {
    "d_model": 64,
    "num_heads": 4,
    "num_temporal_layers": 2,
    "dim_feedforward": 256,
    "dropout": 0.1,
    "use_cross_channel": False,
    "cnn_channels": [32, 64],
    "cnn_kernel_sizes": [5],
    "target_patch_size": 64,
    "normalization_method": "zscore",
    "interpolation_method": "linear",
    "temporal_init_scale": 0.1,
    "channel_init_scale": 0.1,
    "use_channel_encoding": False,
    "max_patches": 5000
}


def get_config(size: str = "small") -> Dict[str, Any]:
    """
    Get a model configuration by size.

    Args:
        size: One of "tiny", "small", "medium", "large"

    Returns:
        Configuration dictionary

    Example:
        >>> config = get_config("small")
        >>> encoder = IMUActivityRecognitionEncoder(**config)
    """
    configs = {
        "tiny": TINY_CONFIG,
        "small": SMALL_CONFIG,
        "medium": MEDIUM_CONFIG,
        "large": LARGE_CONFIG,
    }

    if size not in configs:
        raise ValueError(f"Unknown config size: {size}. Choose from {list(configs.keys())}")

    return configs[size].copy()
