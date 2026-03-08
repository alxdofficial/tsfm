"""
Model configuration tiers for TSFM.

Five tiers:
  - Tiny       (d=192, 4 layers)  : ~5-6M trainable  — lightweight edge model
  - Small      (d=384, 4 layers)  : ~21M trainable    — baseline, proven 85% val_acc
  - Small-Deep (d=384, 8 layers)  : ~25M trainable    — per-patch prediction + majority voting
  - Medium     (d=512, 8 layers)  : ~67M trainable  — wider encoder, same proven depth
  - Large      (d=512, 12 layers) : ~92M trainable  — research/server model (needs GradCache)

Text encoder weights are frozen and used only during training to produce
channel/label embeddings. At inference, precomputed embeddings are stored
as constants, so the text encoder is never deployed on-device.

Each config is the single source of truth for all architecture hyperparameters:
encoder, semantic head, channel-text fusion, and label bank.
"""

from typing import Dict, Any


# ---------------------------------------------------------------------------
# Tiny: d=192, 4 layers, all-MiniLM-L6-v2 (384-dim → projected to 192)
# Trainable: ~5-6M | Lightweight edge model
# Uses sbert_to_model projection (384→192) in ChannelSemanticEncoding.
# ---------------------------------------------------------------------------
TINY_CONFIG: Dict[str, Any] = {
    # --- Encoder ---
    "d_model": 192,
    "num_heads": 4,              # 48 dims per head
    "num_temporal_layers": 4,
    "dim_feedforward": 768,      # 4x d_model
    "dropout": 0.1,
    "use_cross_channel": True,
    "cnn_channels": [32, 64],
    "cnn_kernel_sizes": [5],
    "target_patch_size": 64,
    "feature_extractor_type": "spectral_temporal",
    "spectral_ratio": 0.25,
    "normalization_method": "zscore",
    "interpolation_method": "linear",
    "temporal_init_scale": 0.1,
    "channel_init_scale": 0.1,
    "use_channel_encoding": True,
    "max_patches": 5000,

    # --- Text encoder (MiniLM — 384-dim, projected to d_model=192 in positional encoding) ---
    "sentence_bert_model": "all-MiniLM-L6-v2",
    "text_dim": 384,

    # --- Contrastive text encoder (MiniLM — keeps tiny model lightweight) ---
    "contrastive_text_model": "all-MiniLM-L6-v2",
    "contrastive_text_dim": 384,

    # --- Semantic alignment head ---
    "semantic_dim": 384,                   # Matches contrastive_text_dim
    "d_model_fused": 192,                  # Matches d_model
    "num_semantic_temporal_layers": 2,
    "num_fusion_queries": 4,
    "use_fusion_self_attention": True,
    "num_pool_queries": 4,
    "use_pool_self_attention": True,

    # --- Channel-text fusion ---
    "channel_text_num_heads": 4,

    # --- Label bank ---
    "label_bank_num_heads": 4,
    "label_bank_num_queries": 4,
    "label_bank_num_prototypes": 1,

    # --- Per-patch prediction ---
    "per_patch_prediction": True,
}


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
    "feature_extractor_type": "cnn",  # "cnn" or "spectral_temporal"
    "spectral_ratio": 0.25,           # fraction of d_model for spectral features
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
# Small-Deep: d=384, 8 layers, all-MiniLM-L6-v2 (384-dim)
# Small model's proven 384-dim contrastive dynamics (dense similarity, learns
# to push negatives apart) but with deeper/wider components from Medium.
# Per-patch prediction: each patch independently produces a contrastive
# embedding; majority voting at inference.
# ---------------------------------------------------------------------------
SMALL_DEEP_CONFIG: Dict[str, Any] = {
    # --- Encoder (384-dim like Small, but DEEPER: 8 layers) ---
    "d_model": 384,
    "num_heads": 8,              # 48 dims per head (same as Small)
    "num_temporal_layers": 8,    # 2x Small
    "dim_feedforward": 1536,     # 4x d_model
    "dropout": 0.1,
    "use_cross_channel": True,
    "cnn_channels": [32, 64],    # Same as Small (matched to d_model=384)
    "cnn_kernel_sizes": [5],
    "target_patch_size": 64,
    "feature_extractor_type": "spectral_temporal",  # Use hybrid extractor
    "spectral_ratio": 0.25,
    "normalization_method": "zscore",
    "interpolation_method": "linear",
    "temporal_init_scale": 0.1,
    "channel_init_scale": 0.1,
    "use_channel_encoding": True,
    "max_patches": 5000,

    # --- Text encoder (MiniLM — used for channel positional encoding, must match d_model=384) ---
    "sentence_bert_model": "all-MiniLM-L6-v2",
    "text_dim": 384,

    # --- Contrastive text encoder (MPNet — used for label bank + channel fusion) ---
    "contrastive_text_model": "all-mpnet-base-v2",  # 768-dim, richer label representations
    "contrastive_text_dim": 768,

    # --- Semantic alignment head (scaled up from Small) ---
    "semantic_dim": 768,                   # Matches contrastive_text_dim (MPNet output)
    "d_model_fused": 384,                  # Keep same as d_model
    "num_semantic_temporal_layers": 4,     # 2x Small
    "num_fusion_queries": 6,               # Scaled from 4
    "use_fusion_self_attention": True,
    "num_pool_queries": 6,                 # Scaled from 4
    "use_pool_self_attention": True,

    # --- Channel-text fusion ---
    "channel_text_num_heads": 4,

    # --- Label bank ---
    "label_bank_num_heads": 4,
    "label_bank_num_queries": 6,           # Scaled from 4
    "label_bank_num_prototypes": 1,

    # --- Per-patch prediction ---
    "per_patch_prediction": True,          # Each patch predicts independently; majority vote at inference
}


# ---------------------------------------------------------------------------
# Medium: d=512, 8 layers, all-mpnet-base-v2 (768-dim)
# Same proven depth as Small-Deep but wider encoder (512 vs 384).
# Trainable: ~67M | Tests width scaling independently of depth.
# ---------------------------------------------------------------------------
MEDIUM_CONFIG: Dict[str, Any] = {
    # --- Encoder (512-dim, 8 layers — same depth as Small-Deep) ---
    "d_model": 512,
    "num_heads": 8,              # 64 dims per head
    "num_temporal_layers": 8,    # Same as Small-Deep
    "dim_feedforward": 2048,     # 4x d_model
    "dropout": 0.1,
    "use_cross_channel": True,
    "cnn_channels": [32, 64],
    "cnn_kernel_sizes": [5],
    "target_patch_size": 64,
    "feature_extractor_type": "spectral_temporal",
    "spectral_ratio": 0.25,
    "normalization_method": "zscore",
    "interpolation_method": "linear",
    "temporal_init_scale": 0.1,
    "channel_init_scale": 0.1,
    "use_channel_encoding": True,
    "max_patches": 5000,

    # --- Text encoder (MiniLM — used for channel positional encoding, projected to d_model=512) ---
    "sentence_bert_model": "all-MiniLM-L6-v2",
    "text_dim": 384,

    # --- Contrastive text encoder (MPNet — used for label bank + channel fusion) ---
    "contrastive_text_model": "all-mpnet-base-v2",
    "contrastive_text_dim": 768,

    # --- Semantic alignment head (same structure as Small-Deep, wider fused dim) ---
    "semantic_dim": 768,                   # Matches contrastive_text_dim (MPNet output)
    "d_model_fused": 512,                  # Matches d_model
    "num_semantic_temporal_layers": 4,     # Same as Small-Deep
    "num_fusion_queries": 6,               # Same as Small-Deep
    "use_fusion_self_attention": True,
    "num_pool_queries": 6,                 # Same as Small-Deep
    "use_pool_self_attention": True,

    # --- Channel-text fusion ---
    "channel_text_num_heads": 4,           # Same as Small-Deep

    # --- Label bank ---
    "label_bank_num_heads": 4,             # Same as Small-Deep
    "label_bank_num_queries": 6,           # Same as Small-Deep
    "label_bank_num_prototypes": 1,

    # --- Per-patch prediction ---
    "per_patch_prediction": True,
}


# ---------------------------------------------------------------------------
# Large: d=512, 12 layers, all-mpnet-base-v2 (768-dim)
# Trainable: ~92M | Research/server model. Requires GradCache for stable
# training — without it, gradients vanish through 12 post-norm layers.
# ---------------------------------------------------------------------------
LARGE_CONFIG: Dict[str, Any] = {
    # --- Encoder (512-dim, 12 layers) ---
    "d_model": 512,
    "num_heads": 8,              # 64 dims per head
    "num_temporal_layers": 12,
    "dim_feedforward": 2048,     # 4x d_model
    "dropout": 0.1,
    "use_cross_channel": True,
    "cnn_channels": [32, 64],
    "cnn_kernel_sizes": [5],
    "target_patch_size": 64,
    "feature_extractor_type": "spectral_temporal",
    "spectral_ratio": 0.25,
    "normalization_method": "zscore",
    "interpolation_method": "linear",
    "temporal_init_scale": 0.1,
    "channel_init_scale": 0.1,
    "use_channel_encoding": True,
    "max_patches": 5000,

    # --- Text encoder (MiniLM — used for channel positional encoding, projected to d_model=512) ---
    "sentence_bert_model": "all-MiniLM-L6-v2",
    "text_dim": 384,

    # --- Contrastive text encoder (MPNet — used for label bank + channel fusion) ---
    "contrastive_text_model": "all-mpnet-base-v2",
    "contrastive_text_dim": 768,

    # --- Semantic alignment head ---
    "semantic_dim": 768,                   # Matches contrastive_text_dim (MPNet output)
    "d_model_fused": 512,                  # Matches d_model
    "num_semantic_temporal_layers": 6,
    "num_fusion_queries": 8,
    "use_fusion_self_attention": True,
    "num_pool_queries": 8,
    "use_pool_self_attention": True,

    # --- Channel-text fusion ---
    "channel_text_num_heads": 8,

    # --- Label bank ---
    "label_bank_num_heads": 8,
    "label_bank_num_queries": 8,
    "label_bank_num_prototypes": 1,

    # --- Per-patch prediction ---
    "per_patch_prediction": True,
}


def get_config(size: str = "small") -> Dict[str, Any]:
    """
    Get a model configuration by size.

    Args:
        size: One of "tiny", "small", "small_deep", "medium", "large"

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
        "small_deep": SMALL_DEEP_CONFIG,
        "medium": MEDIUM_CONFIG,
        "large": LARGE_CONFIG,
    }

    if size not in configs:
        raise ValueError(f"Unknown config size: {size}. Choose from {list(configs.keys())}")

    return configs[size].copy()
