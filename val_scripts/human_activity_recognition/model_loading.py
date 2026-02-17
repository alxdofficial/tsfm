"""Shared model and label bank loading utilities.

Consolidates the model loading logic that was duplicated across
evaluate_tsfm.py, compare_models.py, benchmark_baselines.py,
session_explorer.py, embedding_video_4d.py, and visualization_3d.py.
"""

import json
from pathlib import Path
from typing import Tuple

import torch

from model.encoder import IMUActivityRecognitionEncoder
from model.semantic_alignment import SemanticAlignmentHead
from training_scripts.human_activity_recognition.semantic_alignment_train import SemanticAlignmentModel
from model.token_text_encoder import LearnableLabelBank


def load_model(
    checkpoint_path: str,
    device: torch.device,
    verbose: bool = True,
) -> Tuple[SemanticAlignmentModel, dict, Path]:
    """Load a SemanticAlignmentModel from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint .pt file.
        device: Device to load the model onto.
        verbose: Whether to print loading information.

    Returns:
        model: The loaded SemanticAlignmentModel in inference mode.
        checkpoint: The full checkpoint dict (contains label_bank_state_dict, epoch, etc.).
        hyperparams_path: Path to hyperparameters.json alongside the checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = checkpoint_path.resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    epoch = checkpoint.get('epoch', 'unknown')

    # Load hyperparameters from checkpoint directory
    hyperparams_path = checkpoint_path.parent / 'hyperparameters.json'
    if not hyperparams_path.exists():
        raise FileNotFoundError(
            f"hyperparameters.json not found at {hyperparams_path}. "
            "This checkpoint may be from an older incompatible version."
        )

    with open(hyperparams_path) as f:
        hyperparams = json.load(f)
    enc_cfg = hyperparams.get('encoder', {})
    head_cfg = hyperparams.get('semantic_head', {})
    token_cfg = hyperparams.get('token_level_text', {})

    # Create encoder
    encoder = IMUActivityRecognitionEncoder(
        d_model=enc_cfg.get('d_model', 384),
        num_heads=enc_cfg.get('num_heads', 8),
        num_temporal_layers=enc_cfg.get('num_temporal_layers', 4),
        dim_feedforward=enc_cfg.get('dim_feedforward', 1536),
        dropout=enc_cfg.get('dropout', 0.1),
        use_cross_channel=enc_cfg.get('use_cross_channel', True),
        cnn_channels=enc_cfg.get('cnn_channels', [32, 64]),
        cnn_kernel_sizes=enc_cfg.get('cnn_kernel_sizes', [5]),
        target_patch_size=enc_cfg.get('target_patch_size', 64),
        use_channel_encoding=enc_cfg.get('use_channel_encoding', False),
    )

    # Create semantic head
    semantic_head = SemanticAlignmentHead(
        d_model=enc_cfg.get('d_model', 384),
        d_model_fused=384,
        output_dim=384,
        num_temporal_layers=head_cfg.get('num_temporal_layers', 2),
        num_heads=enc_cfg.get('num_heads', 8),
        dim_feedforward=enc_cfg.get('dim_feedforward', 1536),
        dropout=enc_cfg.get('dropout', 0.1),
        num_fusion_queries=head_cfg.get('num_fusion_queries', 4),
        use_fusion_self_attention=head_cfg.get('use_fusion_self_attention', True),
        num_pool_queries=head_cfg.get('num_pool_queries', 4),
        use_pool_self_attention=head_cfg.get('use_pool_self_attention', True),
    )

    # Create full model with token-level text encoding
    model = SemanticAlignmentModel(
        encoder,
        semantic_head,
        num_heads=token_cfg.get('num_heads', 4),
        dropout=enc_cfg.get('dropout', 0.1),
    )

    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint['model_state_dict'], strict=False
    )
    if unexpected_keys:
        other_unexpected = [k for k in unexpected_keys if 'channel_encoding' not in k]
        if other_unexpected and verbose:
            print(f"  Warning: Unexpected keys: {other_unexpected[:5]}...")
    if missing_keys and verbose:
        print(f"  Warning: Missing keys: {missing_keys[:5]}...")

    model.train(False)
    model = model.to(device)

    if verbose:
        print(f"  Loaded checkpoint from epoch {epoch}")
        print(f"  Encoder: d_model={enc_cfg.get('d_model', 384)}, "
              f"layers={enc_cfg.get('num_temporal_layers', 4)}, "
              f"heads={enc_cfg.get('num_heads', 8)}")

    return model, checkpoint, hyperparams_path


def load_label_bank(
    checkpoint: dict,
    device: torch.device,
    hyperparams_path: Path,
    verbose: bool = True,
) -> LearnableLabelBank:
    """Load a LearnableLabelBank with trained state from a checkpoint.

    Args:
        checkpoint: The checkpoint dict (from torch.load or load_model).
        device: Device to load the label bank onto.
        hyperparams_path: Path to hyperparameters.json.
        verbose: Whether to print loading information.

    Returns:
        label_bank: The loaded LearnableLabelBank in inference mode.
    """
    if hyperparams_path.exists():
        with open(hyperparams_path) as f:
            hyperparams = json.load(f)
        token_cfg = hyperparams.get('token_level_text', {})
    else:
        token_cfg = {}

    label_bank = LearnableLabelBank(
        device=device,
        num_heads=token_cfg.get('num_heads', 4),
        num_queries=token_cfg.get('num_queries', 4),
        num_prototypes=token_cfg.get('num_prototypes', 1),
        dropout=0.1,
    )

    if 'label_bank_state_dict' in checkpoint:
        label_bank.load_state_dict(checkpoint['label_bank_state_dict'])
        if verbose:
            print("  Loaded trained LearnableLabelBank state")
    else:
        if verbose:
            print("  Warning: No label_bank_state_dict in checkpoint, using untrained weights")

    label_bank.train(False)
    return label_bank
