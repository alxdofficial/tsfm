"""
Model configuration tiers for TSFM.

Three deployment tiers:
  - Small  (d=384, 4 layers)  : ~21M trainable, ~44M inference  — mobile-friendly
  - Medium (d=768, 8 layers)  : ~122M trainable, ~232M inference — balanced
  - Large  (d=1024, 12 layers): ~350M trainable, ~700M inference — maximum quality

Text encoder weights are frozen and used only during training to produce
channel/label embeddings. At inference, precomputed embeddings are stored
as constants, so the text encoder is never deployed on-device.

Each config is the single source of truth for all architecture hyperparameters:
encoder, semantic head, channel-text fusion, and label bank.
"""

from typing import Dict, Any


# ---------------------------------------------------------------------------
# Small: d=384, 4 layers, all-MiniLM-L6-v2 (384-dim, 22.7M params)
# Trainable: ~21M | Inference (excl. precomputed text): ~21M
# Suitable for smartphone / edge deployment.
# ---------------------------------------------------------------------------
SMALL_CONFIG: Dict[str, Any] = {
    # --- Encoder ---
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
    "max_patches": 5000,

    # --- Text encoder (frozen, not deployed on-device) ---
    "sentence_bert_model": "all-MiniLM-L6-v2",   # 384-dim embeddings
    "text_dim": 384,                              # Text encoder output dimension

    # --- Semantic alignment head ---
    "semantic_dim": 384,                   # Final embedding dim (matches SBERT)
    "d_model_fused": 384,                  # Cross-channel fusion output dim
    "num_semantic_temporal_layers": 2,     # Temporal attention layers in head
    "num_fusion_queries": 4,               # Query tokens for channel fusion
    "use_fusion_self_attention": True,
    "num_pool_queries": 4,                 # Query tokens for temporal pooling
    "use_pool_self_attention": True,

    # --- Channel-text fusion ---
    "channel_text_num_heads": 4,           # Cross-attention heads

    # --- Label bank ---
    "label_bank_num_heads": 4,             # Attention heads for label pooling
    "label_bank_num_queries": 4,           # Learnable query tokens per label
    "label_bank_num_prototypes": 1,        # Prototype embeddings per label
}


# ---------------------------------------------------------------------------
# Small-Wide: d=384, 4 layers (same encoder as Small), but 768-dim semantic space.
# Trainable: ~36M | Inference: ~36M
# Tests whether wider representation space helps without scaling the encoder.
# Small encoder = low VRAM → can train at BS=32+ (vs BS=8 for Medium).
# ---------------------------------------------------------------------------
SMALL_WIDE_CONFIG: Dict[str, Any] = {
    # --- Encoder (identical to SMALL — cheap, proven) ---
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
    "max_patches": 5000,

    # --- Text encoder (frozen, 768-dim for richer label representations) ---
    "sentence_bert_model": "all-mpnet-base-v2",   # 768-dim embeddings
    "text_dim": 768,                               # Text encoder output dimension

    # --- Semantic alignment head (WIDER: 768-dim semantic space) ---
    "semantic_dim": 768,                   # 2x Small — more room for fine-grained distinctions
    "d_model_fused": 768,                  # CrossChannelFusion projects 384→768
    "num_semantic_temporal_layers": 3,     # Between Small(2) and Medium(4)
    "num_fusion_queries": 4,               # Same as Small (proven)
    "use_fusion_self_attention": True,
    "num_pool_queries": 4,                 # Same as Small
    "use_pool_self_attention": True,

    # --- Channel-text fusion ---
    "channel_text_num_heads": 4,           # Same as Small

    # --- Label bank (768-dim to match semantic space) ---
    "label_bank_num_heads": 8,             # 768/8=96 dims per head
    "label_bank_num_queries": 4,           # Same as Small
    "label_bank_num_prototypes": 1,
}


# ---------------------------------------------------------------------------
# Medium: d=768, 8 layers, all-mpnet-base-v2 (768-dim, 109M params)
# Trainable: ~122M | Inference (incl. frozen text encoder): ~232M
# Balanced quality-to-cost ratio for GPU servers.
# ---------------------------------------------------------------------------
MEDIUM_CONFIG: Dict[str, Any] = {
    # --- Encoder ---
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
    "max_patches": 5000,

    # --- Text encoder (frozen, not deployed on-device) ---
    "sentence_bert_model": "all-mpnet-base-v2",   # 768-dim embeddings
    "text_dim": 768,                               # Text encoder output dimension

    # --- Semantic alignment head ---
    "semantic_dim": 768,
    "d_model_fused": 768,
    "num_semantic_temporal_layers": 4,     # Scaled up from 2
    "num_fusion_queries": 6,               # Scaled up from 4
    "use_fusion_self_attention": True,
    "num_pool_queries": 6,                 # Scaled up from 4
    "use_pool_self_attention": True,

    # --- Channel-text fusion ---
    "channel_text_num_heads": 8,           # Scaled up from 4

    # --- Label bank ---
    "label_bank_num_heads": 8,             # Scaled up from 4
    "label_bank_num_queries": 6,           # Scaled up from 4
    "label_bank_num_prototypes": 1,
}


# ---------------------------------------------------------------------------
# Large: d=1024, 12 layers, BAAI/bge-large-en-v1.5 (1024-dim, 335M params)
# Trainable: ~350M | Inference (incl. frozen text encoder): ~700M
# Maximum quality; requires >=40GB GPU for training.
# ---------------------------------------------------------------------------
LARGE_CONFIG: Dict[str, Any] = {
    # --- Encoder ---
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
    "max_patches": 5000,

    # --- Text encoder (frozen, not deployed on-device) ---
    "sentence_bert_model": "BAAI/bge-large-en-v1.5",  # 1024-dim embeddings
    "text_dim": 1024,                                   # Text encoder output dimension

    # --- Semantic alignment head ---
    "semantic_dim": 1024,
    "d_model_fused": 1024,
    "num_semantic_temporal_layers": 6,     # Scaled up from 2
    "num_fusion_queries": 8,               # Scaled up from 4
    "use_fusion_self_attention": True,
    "num_pool_queries": 8,                 # Scaled up from 4
    "use_pool_self_attention": True,

    # --- Channel-text fusion ---
    "channel_text_num_heads": 16,          # 1024/16=64 dims per head

    # --- Label bank ---
    "label_bank_num_heads": 16,            # 1024/16=64 dims per head
    "label_bank_num_queries": 8,           # Scaled up from 4
    "label_bank_num_prototypes": 1,
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
    "max_patches": 5000,

    # --- Downstream (minimal for tests) ---
    "sentence_bert_model": "all-MiniLM-L6-v2",
    "text_dim": 384,
    "semantic_dim": 64,
    "d_model_fused": 64,
    "num_semantic_temporal_layers": 1,
    "num_fusion_queries": 2,
    "use_fusion_self_attention": False,
    "num_pool_queries": 2,
    "use_pool_self_attention": False,
    "channel_text_num_heads": 2,
    "label_bank_num_heads": 2,
    "label_bank_num_queries": 2,
    "label_bank_num_prototypes": 1,
}


def get_config(size: str = "small") -> Dict[str, Any]:
    """
    Get a model configuration by size.

    Args:
        size: One of "tiny", "small", "medium", "large"

    Returns:
        Configuration dictionary containing all architecture hyperparameters
        for encoder, semantic head, channel-text fusion, and label bank.

    Example:
        >>> config = get_config("small")
        >>> encoder = IMUActivityRecognitionEncoder(**config)
    """
    configs = {
        "tiny": TINY_CONFIG,
        "small": SMALL_CONFIG,
        "small_wide": SMALL_WIDE_CONFIG,
        "medium": MEDIUM_CONFIG,
        "large": LARGE_CONFIG,
    }

    if size not in configs:
        raise ValueError(f"Unknown config size: {size}. Choose from {list(configs.keys())}")

    return configs[size].copy()
