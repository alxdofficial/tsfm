"""
Shared TSFM eval helpers (data loading, embedding/forward, metadata) reused by
evaluate_tsfm_v2. Extracted from the retired v1 evaluate_tsfm.py; the v1 scoring
(open/closed-set + synonym groups) is gone — see eval_v2.py / evaluate_tsfm_v2.py.

Extracts embeddings from the trained TSFM/HALO semantic alignment model for the
v2 zero-shot and subject-disjoint few-shot evaluators.

Zero-shot uses cosine similarity between IMU embeddings and text label
prototypes — no classifier training needed.

Supervised fine-tuning: deep-copies the model, fine-tunes the sensor encoder
end-to-end with cross-entropy on cosine similarity logits against frozen text
label embeddings. No separate classifier head — uses the model's native
text-alignment mechanism.

Uses native sampling rates and dataset-specific channel descriptions from
manifests, giving TSFM the same rich metadata it uses during training.

Usage:
    python val_scripts/human_activity_recognition/evaluate_tsfm.py
"""

import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.imu_pretraining_dataset.label_groups import (
    LABEL_GROUPS,
    get_label_to_group_mapping,
)
from val_scripts.human_activity_recognition.model_loading import load_model, load_label_bank
from val_scripts.human_activity_recognition.evaluation_metrics import compute_similarity
from val_scripts.human_activity_recognition.grouped_zero_shot import (
    map_local_to_global_labels,
)
from training_scripts.human_activity_recognition.semantic_alignment_train import SemanticAlignmentModel
from model.token_text_encoder import LearnableLabelBank

# =============================================================================
# Configuration
# =============================================================================

BENCHMARK_DIR = PROJECT_ROOT / "benchmark_data"
DATA_DIR = PROJECT_ROOT / "data"
LIMUBERT_DATA_DIR = BENCHMARK_DIR / "processed" / "limubert"
TSFM_EVAL_DIR = BENCHMARK_DIR / "processed" / "tsfm_eval"
DATASET_CONFIG_PATH = BENCHMARK_DIR / "dataset_config.json"
GLOBAL_LABEL_PATH = LIMUBERT_DATA_DIR / "global_label_mapping.json"
OUTPUT_DIR = PROJECT_ROOT / "test_output" / "baseline_evaluation"

# TSFM checkpoint - set TSFM_CHECKPOINT env var, or update this default path
_DEFAULT_CHECKPOINT = str(PROJECT_ROOT / "training_output" / "semantic_alignment" / "small_v1_best_20260217" / "best.pt")
CHECKPOINT_PATH = os.environ.get("TSFM_CHECKPOINT", _DEFAULT_CHECKPOINT)

# Data specs
DATA_CHANNELS = 6          # 6-channel IMU (3 accel + 3 gyro)
TSFM_EMB_DIM = None        # Auto-detected from model (set after loading)
TSFM_BATCH_SIZE = 32       # Batch size for embedding extraction

# Fixed patch size for evaluation
# 1.0s chosen as smallest valid size — gives finest temporal resolution.
PATCH_SIZE_SEC = 1.0

# Core channel names (used for manifest lookup)
CORE_CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

# Fallback channel descriptions (used only when manifest is unavailable)
FALLBACK_CHANNEL_DESCRIPTIONS = [
    "Accelerometer X-axis",
    "Accelerometer Y-axis",
    "Accelerometer Z-axis",
    "Gyroscope X-axis",
    "Gyroscope Y-axis",
    "Gyroscope Z-axis",
]

# Fine-tuning hyperparameters (end-to-end, cosine sim with frozen text embeddings)
FINETUNE_EPOCHS = 20
FINETUNE_BATCH_SIZE = 32
FINETUNE_ENCODER_LR = 1e-5
FINETUNE_WEIGHT_DECAY = 1e-5
FINETUNE_PATIENCE = 5  # Early stopping patience (monitor val accuracy)
FINETUNE_TEMPERATURE = 0.07  # Same as training temperature

CLASSIFIER_SEED = 3431

# Data split parameters
TRAINING_RATE = 0.8
VALI_RATE = 0.1
SUPERVISED_LABEL_RATE_1PCT = 0.01  # 1% of training portion
SUPERVISED_LABEL_RATE_10PCT = 0.10  # 10% of training portion

# Load configs
with open(DATASET_CONFIG_PATH) as f:
    DATASET_CONFIG = json.load(f)

with open(GLOBAL_LABEL_PATH) as f:
    GLOBAL_LABELS = json.load(f)["labels"]

TRAIN_DATASETS = DATASET_CONFIG["train_datasets"]
TEST_DATASETS = DATASET_CONFIG["zero_shot_datasets"]


def get_dataset_metadata(dataset_name: str) -> dict:
    """Get native sampling rate, channel descriptions, and gyro availability from manifest.

    Uses dataset_config's core_channels mapping to resolve location-prefixed
    manifest channel names (e.g. shoaib's right_pocket_acc_x → acc_x).
    Appends sampling rate and patch size suffix to match training format.
    """
    manifest_path = DATA_DIR / dataset_name / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    sampling_rate = manifest['channels'][0]['sampling_rate_hz']
    dataset_desc = manifest.get('description', '')

    # Build manifest channel name → description lookup
    ch_map = {ch['name']: ch['description'] for ch in manifest['channels']}

    # Get core_channels mapping from dataset_config (standard_name → original_name)
    ds_config = DATASET_CONFIG["datasets"][dataset_name]
    core_channel_map = ds_config.get("core_channels", {})

    # Determine which core channels are real (have actual sensor data)
    has_gyro = "gyro_x" in core_channel_map

    # Build descriptions for real channels only (model pads with [PAD] for missing)
    real_channels = ["acc_x", "acc_y", "acc_z"]
    if has_gyro:
        real_channels += ["gyro_x", "gyro_y", "gyro_z"]

    channel_descriptions = []
    for ch in real_channels:
        # Use core_channels mapping to find the original manifest channel name
        original_name = core_channel_map.get(ch, ch)
        ch_desc = ch_map.get(original_name, ch_map.get(ch, f"Channel: {ch}"))

        # Keep channel text compact so axis/placement/unit/rate semantics are not
        # truncated by the 64-token text encoder.
        full_desc = ch_desc

        # Append sampling rate and patch window size to match training format
        # (matches multi_dataset_loader.py:383)
        full_desc = f"{full_desc} (sampled at {sampling_rate:.0f}Hz, {PATCH_SIZE_SEC:.1f}s window)"
        channel_descriptions.append(full_desc)

    return {
        'sampling_rate_hz': sampling_rate,
        'channel_descriptions': channel_descriptions,
        'dataset_description': dataset_desc,
        'has_gyro': has_gyro,
        'n_real_channels': len(real_channels),
    }


# =============================================================================
# TSFM Model Loading & Embedding Extraction
# =============================================================================

load_tsfm_model = load_model  # backwards-compatible alias


def extract_tsfm_embeddings(
    model: SemanticAlignmentModel,
    raw_data: np.ndarray,
    device: torch.device,
    sampling_rate: float,
    channel_descriptions: List[str],
    batch_size: int = TSFM_BATCH_SIZE,
    patch_size_sec: float = PATCH_SIZE_SEC,
    has_gyro: bool = True,
) -> np.ndarray:
    """Extract TSFM embeddings from raw sensor data.

    For per-patch models, forward_from_raw automatically mean-pools to session-level.

    Args:
        model: Loaded TSFM SemanticAlignmentModel
        raw_data: (N, seq_len, 6) raw sensor data at native rate
        device: torch device
        sampling_rate: native sampling rate in Hz
        channel_descriptions: per-channel descriptions (3 for acc-only, 6 for acc+gyro)
        batch_size: batch size for inference
        patch_size_sec: patch duration in seconds
        has_gyro: whether gyro channels are real (False = zero-padded, mask them out)

    Returns:
        embeddings: (N, D) L2-normalized session-level embeddings
    """
    model.train(False)
    N = raw_data.shape[0]
    seq_len = raw_data.shape[1]

    all_embeddings = []
    for start in tqdm(range(0, N, batch_size), desc="TSFM | Extracting embeddings",
                      total=(N + batch_size - 1) // batch_size, leave=True):
        end = min(start + batch_size, N)
        batch_data = torch.from_numpy(raw_data[start:end]).float().to(device)
        bs = batch_data.shape[0]

        channel_mask = torch.ones(bs, DATA_CHANNELS, dtype=torch.bool, device=device)
        if not has_gyro:
            channel_mask[:, 3:] = False  # Mask out zero-padded gyro channels
        attention_mask = torch.ones(bs, seq_len, dtype=torch.bool, device=device)

        channel_descs = [channel_descriptions[:] for _ in range(bs)]
        sampling_rates = [sampling_rate] * bs
        patch_sizes = [patch_size_sec] * bs

        with torch.no_grad():
            with autocast('cuda', enabled=device.type == 'cuda'):
                emb = model.forward_from_raw(
                    batch_data, channel_descs, channel_mask,
                    sampling_rates, patch_sizes,
                    attention_mask=attention_mask
                )
            all_embeddings.append(emb.float().cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def extract_tsfm_per_patch_embeddings(
    model: SemanticAlignmentModel,
    raw_data: np.ndarray,
    device: torch.device,
    sampling_rate: float,
    channel_descriptions: List[str],
    batch_size: int = TSFM_BATCH_SIZE,
    patch_size_sec: float = PATCH_SIZE_SEC,
    has_gyro: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract per-patch TSFM embeddings for majority-vote zero-shot.

    Only meaningful for per-patch models. Session-level models return (N, 1, D).

    Returns:
        patch_embeddings: (N, max_P, D) padded per-patch embeddings
        patch_masks: (N, max_P) boolean masks (True=valid)
    """
    model.train(False)
    is_per_patch = hasattr(model, 'semantic_head') and model.semantic_head.per_patch_prediction
    N = raw_data.shape[0]
    seq_len = raw_data.shape[1]

    all_embeddings = []
    all_masks = []
    for start in tqdm(range(0, N, batch_size), desc="TSFM | Extracting per-patch embeddings",
                      total=(N + batch_size - 1) // batch_size, leave=True):
        end = min(start + batch_size, N)
        batch_data = torch.from_numpy(raw_data[start:end]).float().to(device)
        bs = batch_data.shape[0]

        channel_mask = torch.ones(bs, DATA_CHANNELS, dtype=torch.bool, device=device)
        if not has_gyro:
            channel_mask[:, 3:] = False  # Mask out zero-padded gyro channels
        attention_mask = torch.ones(bs, seq_len, dtype=torch.bool, device=device)

        channel_descs = [channel_descriptions[:] for _ in range(bs)]
        sampling_rates = [sampling_rate] * bs
        patch_sizes = [patch_size_sec] * bs

        with torch.no_grad():
            with autocast('cuda', enabled=device.type == 'cuda'):
                if is_per_patch:
                    emb, pmask = model.forward_from_raw(
                        batch_data, channel_descs, channel_mask,
                        sampling_rates, patch_sizes,
                        attention_mask=attention_mask,
                        return_per_patch=True,
                    )
                else:
                    emb = model.forward_from_raw(
                        batch_data, channel_descs, channel_mask,
                        sampling_rates, patch_sizes,
                        attention_mask=attention_mask,
                    )
                    # Wrap session-level (B, D) as (B, 1, D) for uniform API
                    emb = emb.unsqueeze(1)
                    pmask = torch.ones(bs, 1, dtype=torch.bool, device=device)

            all_embeddings.append(emb.float().cpu())
            all_masks.append(pmask.cpu())

    # Pad to max_P across all batches
    max_P = max(e.shape[1] for e in all_embeddings)
    padded_embs = []
    padded_masks = []
    for emb, mask in zip(all_embeddings, all_masks):
        pad_P = max_P - emb.shape[1]
        if pad_P > 0:
            padded_embs.append(F.pad(emb, (0, 0, 0, pad_P)))
            padded_masks.append(F.pad(mask, (0, pad_P), value=False))
        else:
            padded_embs.append(emb)
            padded_masks.append(mask)

    return torch.cat(padded_embs, dim=0), torch.cat(padded_masks, dim=0)


# =============================================================================
# Data Loading (identical to baselines)
# =============================================================================

def load_raw_data(dataset_name: str, use_native_rate: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """Load raw sensor data and labels for a dataset.

    Returns: (data, labels, sampling_rate_hz)
    """
    if use_native_rate:
        tsfm_eval_dir = TSFM_EVAL_DIR / dataset_name
        meta_path = tsfm_eval_dir / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        data = np.load(str(tsfm_eval_dir / "data_native.npy")).astype(np.float32)
        labels = np.load(str(tsfm_eval_dir / "label_native.npy")).astype(np.float32)
        return data, labels, float(meta['sampling_rate_hz'])
    else:
        ds_dir = LIMUBERT_DATA_DIR / dataset_name
        data = np.load(str(ds_dir / "data_20_120.npy")).astype(np.float32)
        labels = np.load(str(ds_dir / "label_20_120.npy")).astype(np.float32)
        return data, labels, 20.0






# =============================================================================
# Data Splitting (identical to baselines)
# =============================================================================





# =============================================================================
# Classifier Training (identical to baselines)
# =============================================================================



# =============================================================================
# End-to-End Fine-Tuning (cosine sim with frozen text embeddings)
# =============================================================================

def _forward_batch(model, batch_data, device, sampling_rate, channel_descriptions, seq_len,
                    has_gyro=True):
    """Run TSFM forward pass on a batch of raw data, returning embeddings.

    Args:
        model: SemanticAlignmentModel
        batch_data: (B, seq_len, 6) tensor on device
        device: torch device
        sampling_rate: native sampling rate in Hz
        channel_descriptions: per-channel descriptions from manifest
        seq_len: sequence length (timesteps per window)
        has_gyro: whether gyro channels are real (False = zero-padded, mask them out)

    Returns:
        (B, 384) L2-normalized embeddings
    """
    bs = batch_data.shape[0]
    channel_mask = torch.ones(bs, DATA_CHANNELS, dtype=torch.bool, device=device)
    if not has_gyro:
        channel_mask[:, 3:] = False
    attention_mask = torch.ones(bs, seq_len, dtype=torch.bool, device=device)
    channel_descs = [channel_descriptions[:] for _ in range(bs)]
    sampling_rates = [sampling_rate] * bs
    patch_sizes = [PATCH_SIZE_SEC] * bs

    with autocast('cuda', enabled=device.type == 'cuda'):
        emb = model.forward_from_raw(
            batch_data, channel_descs, channel_mask,
            sampling_rates, patch_sizes,
            attention_mask=attention_mask,
        )
    return emb


def compute_cosine_accuracy(model, data_loader, text_embs, device,
                            sampling_rate, channel_descriptions, seq_len,
                            has_gyro=True):
    """Compute cosine-similarity accuracy on a data loader.

    Args:
        model: SemanticAlignmentModel
        data_loader: yields (batch_data, batch_labels)
        text_embs: (C, 384) frozen text label embeddings
        device: torch device
        sampling_rate: native sampling rate in Hz
        channel_descriptions: per-channel descriptions from manifest
        seq_len: sequence length (timesteps per window)
        has_gyro: whether gyro channels are real

    Returns:
        accuracy (float, 0-1)
    """
    model.train(False)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            emb = _forward_batch(model, batch_data, device, sampling_rate, channel_descriptions, seq_len,
                                 has_gyro=has_gyro)
            logits = emb @ text_embs.T / FINETUNE_TEMPERATURE
            preds = logits.argmax(dim=1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.shape[0]
    return correct / max(total, 1)


# =============================================================================
# Zero-Shot Evaluation Functions (cosine similarity with label bank)
# =============================================================================









# =============================================================================
# Main
# =============================================================================





if __name__ == '__main__':
    main()
