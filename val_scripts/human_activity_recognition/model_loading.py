"""Shared model and label bank loading utilities.

Consolidates the model loading logic that was duplicated across
evaluate_tsfm.py, compare_models.py, benchmark_baselines.py,
session_explorer.py, embedding_video_4d.py, and visualization_3d.py.

Reads architecture hyperparameters from hyperparameters.json saved alongside
the checkpoint. Supports both new format (with 'config' key) and legacy
format (with separate 'encoder'/'semantic_head'/'token_level_text' keys).
"""

import json
from pathlib import Path
from typing import Tuple

import torch

from model.encoder import IMUActivityRecognitionEncoder
from model.semantic_alignment import SemanticAlignmentHead
from training_scripts.human_activity_recognition.semantic_alignment_train import SemanticAlignmentModel
from model.token_text_encoder import LearnableLabelBank, TokenTextEncoder


def _load_hyperparams(hyperparams_path: Path) -> dict:
    """Load and normalize hyperparameters from JSON, handling both formats."""
    with open(hyperparams_path) as f:
        hp = json.load(f)

    # New format: 'config' contains the full architecture dict
    if 'config' in hp:
        cfg = hp['config']
        return {
            'encoder': {
                'd_model': cfg['d_model'],
                'num_heads': cfg['num_heads'],
                'num_temporal_layers': cfg['num_temporal_layers'],
                'dim_feedforward': cfg['dim_feedforward'],
                'dropout': cfg.get('dropout', 0.1),
                'use_cross_channel': cfg.get('use_cross_channel', True),
                'cnn_channels': cfg.get('cnn_channels', [32, 64]),
                'cnn_kernel_sizes': cfg.get('cnn_kernel_sizes', [5]),
                'target_patch_size': cfg.get('target_patch_size', 64),
                'use_channel_encoding': False,
            },
            'semantic_head': {
                'd_model_fused': cfg.get('d_model_fused', cfg['d_model']),
                'semantic_dim': cfg.get('semantic_dim', cfg['d_model']),
                'num_temporal_layers': cfg.get('num_semantic_temporal_layers', 2),
                'num_fusion_queries': cfg.get('num_fusion_queries', 4),
                'use_fusion_self_attention': cfg.get('use_fusion_self_attention', True),
                'num_pool_queries': cfg.get('num_pool_queries', 4),
                'use_pool_self_attention': cfg.get('use_pool_self_attention', True),
            },
            'channel_text_fusion': {
                'num_heads': cfg.get('channel_text_num_heads', 4),
            },
            'label_bank': {
                'sentence_bert_model': cfg.get('sentence_bert_model', 'all-MiniLM-L6-v2'),
                'd_model': cfg.get('semantic_dim', cfg['d_model']),
                'num_heads': cfg.get('label_bank_num_heads', 4),
                'num_queries': cfg.get('label_bank_num_queries', 4),
                'num_prototypes': cfg.get('label_bank_num_prototypes', 1),
            },
        }

    # Legacy format: separate sections
    enc = hp.get('encoder', {})
    sem = hp.get('semantic', {})
    head = hp.get('semantic_head', {})
    tok = hp.get('token_level_text', {})
    d_model = enc.get('d_model', 384)
    return {
        'encoder': {
            'd_model': d_model,
            'num_heads': enc.get('num_heads', 8),
            'num_temporal_layers': enc.get('num_temporal_layers', 4),
            'dim_feedforward': enc.get('dim_feedforward', 1536),
            'dropout': enc.get('dropout', 0.1),
            'use_cross_channel': enc.get('use_cross_channel', True),
            'cnn_channels': enc.get('cnn_channels', [32, 64]),
            'cnn_kernel_sizes': enc.get('cnn_kernel_sizes', [5]),
            'target_patch_size': enc.get('target_patch_size', 64),
            'use_channel_encoding': enc.get('use_channel_encoding', False),
        },
        'semantic_head': {
            'd_model_fused': sem.get('d_model_fused', d_model),
            'semantic_dim': sem.get('semantic_dim', d_model),
            'num_temporal_layers': head.get('num_temporal_layers', 2),
            'num_fusion_queries': head.get('num_fusion_queries', 4),
            'use_fusion_self_attention': head.get('use_fusion_self_attention', True),
            'num_pool_queries': head.get('num_pool_queries', 4),
            'use_pool_self_attention': head.get('use_pool_self_attention', True),
        },
        'channel_text_fusion': {
            'num_heads': tok.get('num_heads', 4),
        },
        'label_bank': {
            'sentence_bert_model': sem.get('sentence_bert_model', 'all-MiniLM-L6-v2'),
            'd_model': sem.get('semantic_dim', d_model),
            'num_heads': tok.get('num_heads', 4),
            'num_queries': tok.get('num_queries', 4),
            'num_prototypes': tok.get('num_prototypes', 1),
        },
    }


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

    hp = _load_hyperparams(hyperparams_path)
    enc_cfg = hp['encoder']
    head_cfg = hp['semantic_head']
    fusion_cfg = hp['channel_text_fusion']
    lb_cfg = hp['label_bank']

    d_model = enc_cfg['d_model']

    # Create encoder
    encoder = IMUActivityRecognitionEncoder(
        d_model=d_model,
        num_heads=enc_cfg['num_heads'],
        num_temporal_layers=enc_cfg['num_temporal_layers'],
        dim_feedforward=enc_cfg['dim_feedforward'],
        dropout=enc_cfg['dropout'],
        use_cross_channel=enc_cfg['use_cross_channel'],
        cnn_channels=enc_cfg['cnn_channels'],
        cnn_kernel_sizes=enc_cfg['cnn_kernel_sizes'],
        target_patch_size=enc_cfg['target_patch_size'],
        use_channel_encoding=enc_cfg['use_channel_encoding'],
    )

    # Create semantic head
    semantic_head = SemanticAlignmentHead(
        d_model=d_model,
        d_model_fused=head_cfg['d_model_fused'],
        output_dim=head_cfg['semantic_dim'],
        num_temporal_layers=head_cfg['num_temporal_layers'],
        num_heads=enc_cfg['num_heads'],
        dim_feedforward=head_cfg['d_model_fused'] * 4,
        dropout=enc_cfg['dropout'],
        num_fusion_queries=head_cfg['num_fusion_queries'],
        use_fusion_self_attention=head_cfg['use_fusion_self_attention'],
        num_pool_queries=head_cfg['num_pool_queries'],
        use_pool_self_attention=head_cfg['use_pool_self_attention'],
    )

    # Create shared text encoder
    shared_text_encoder = TokenTextEncoder(model_name=lb_cfg['sentence_bert_model'])

    # Create full model with token-level text encoding
    model = SemanticAlignmentModel(
        encoder,
        semantic_head,
        num_heads=fusion_cfg['num_heads'],
        dropout=enc_cfg['dropout'],
        text_encoder=shared_text_encoder,
    )

    # Convert legacy combined gate weights to split gate format if needed
    # Old: channel_fusion.gate.0.weight (d, 2*d), gate.0.bias (d,)
    # New: gate_sensor.weight (d, d), gate_channel.weight (d, d), gate_channel.bias (d,)
    state_dict = checkpoint['model_state_dict']
    old_gate_w = 'channel_fusion.gate.0.weight'
    old_gate_b = 'channel_fusion.gate.0.bias'
    if old_gate_w in state_dict and old_gate_b in state_dict:
        W = state_dict.pop(old_gate_w)  # (d, 2*d)
        b = state_dict.pop(old_gate_b)  # (d,)
        d = W.shape[0]
        state_dict['channel_fusion.gate_sensor.weight'] = W[:, :d]    # first d cols
        state_dict['channel_fusion.gate_channel.weight'] = W[:, d:]   # last d cols
        state_dict['channel_fusion.gate_channel.bias'] = b

    # Load state dict (strict=False only to tolerate removed channel_encoding keys)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # Filter out known benign mismatches
    benign_patterns = ('channel_encoding',)
    benign_unexpected = [k for k in unexpected_keys if any(p in k for p in benign_patterns)]
    critical_unexpected = [k for k in unexpected_keys if not any(p in k for p in benign_patterns)]
    if critical_unexpected:
        raise RuntimeError(
            f"Checkpoint has {len(critical_unexpected)} unexpected keys "
            f"(architecture mismatch): {critical_unexpected[:10]}"
        )
    if missing_keys:
        raise RuntimeError(
            f"Checkpoint is missing {len(missing_keys)} keys "
            f"(architecture mismatch): {missing_keys[:10]}"
        )
    if benign_unexpected and verbose:
        print(f"  Note: Ignored {len(benign_unexpected)} legacy channel_encoding keys")

    model.train(False)
    model = model.to(device)

    if verbose:
        print(f"  Loaded checkpoint from epoch {epoch}")
        print(f"  Encoder: d_model={d_model}, "
              f"layers={enc_cfg['num_temporal_layers']}, "
              f"heads={enc_cfg['num_heads']}")
        print(f"  Semantic head: d_fused={head_cfg['d_model_fused']}, "
              f"layers={head_cfg['num_temporal_layers']}, "
              f"fusion_q={head_cfg['num_fusion_queries']}, "
              f"pool_q={head_cfg['num_pool_queries']}")

    return model, checkpoint, hyperparams_path


def load_label_bank(
    checkpoint: dict,
    device: torch.device,
    hyperparams_path: Path,
    verbose: bool = True,
    text_encoder: TokenTextEncoder = None,
) -> LearnableLabelBank:
    """Load a LearnableLabelBank with trained state from a checkpoint.

    Args:
        checkpoint: The checkpoint dict (from torch.load or load_model).
        device: Device to load the label bank onto.
        hyperparams_path: Path to hyperparameters.json.
        verbose: Whether to print loading information.
        text_encoder: Optional shared TokenTextEncoder (avoids loading a second copy).

    Returns:
        label_bank: The loaded LearnableLabelBank in inference mode.
    """
    hp = _load_hyperparams(hyperparams_path)
    lb_cfg = hp['label_bank']

    label_bank = LearnableLabelBank(
        model_name=lb_cfg['sentence_bert_model'],
        device=device,
        d_model=lb_cfg['d_model'],
        num_heads=lb_cfg['num_heads'],
        num_queries=lb_cfg['num_queries'],
        num_prototypes=lb_cfg['num_prototypes'],
        dropout=0.0,
        text_encoder=text_encoder,
    )

    if 'label_bank_state_dict' in checkpoint:
        label_bank.load_state_dict(checkpoint['label_bank_state_dict'])
        if verbose:
            print(f"  Loaded LearnableLabelBank state (d={lb_cfg['d_model']}, "
                  f"heads={lb_cfg['num_heads']}, queries={lb_cfg['num_queries']})")
    else:
        if verbose:
            print("  Warning: No label_bank_state_dict in checkpoint, using untrained weights")

    label_bank.train(False)
    return label_bank
