"""
TSFM (our model) evaluation using the unified framework.

Extracts embeddings from the trained TSFM semantic alignment model
and evaluates with:
  1. Zero-shot open-set (cosine sim against all 87 training labels, group matching)
  2. Zero-shot closed-set (cosine sim against test dataset labels only, exact match)
  3. 1% supervised (end-to-end fine-tuning, cosine sim with frozen text embeddings)
  4. 10% supervised (end-to-end fine-tuning, cosine sim with frozen text embeddings)

Zero-shot uses cosine similarity between IMU embeddings and text label
embeddings from the trained LearnableLabelBank — no classifier training needed.

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
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "models"))

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
def _find_latest_checkpoint():
    """Find most recent best.pt in training_output/semantic_alignment/."""
    sa_dir = PROJECT_ROOT / "training_output" / "semantic_alignment"
    if not sa_dir.exists():
        return str(sa_dir / "TIMESTAMP" / "best.pt")  # placeholder
    runs = sorted(sa_dir.iterdir(), reverse=True)
    for run in runs:
        best = run / "best.pt"
        if best.exists():
            return str(best)
    return str(sa_dir / "TIMESTAMP" / "best.pt")

CHECKPOINT_PATH = os.environ.get("TSFM_CHECKPOINT", _find_latest_checkpoint())

# Data specs
DATA_CHANNELS = 6          # 6-channel IMU (3 accel + 3 gyro)
TSFM_EMB_DIM = 384         # TSFM embedding dimension
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
    """Get native sampling rate and channel descriptions from manifest."""
    manifest_path = DATA_DIR / dataset_name / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    sampling_rate = manifest['channels'][0]['sampling_rate_hz']

    dataset_desc = manifest.get('description', '')

    ch_map = {ch['name']: ch['description'] for ch in manifest['channels']}
    channel_descriptions = []
    for ch in CORE_CHANNELS:
        ch_desc = ch_map.get(ch, f"Channel: {ch}")
        # Prepend dataset description to match training format
        # (multi_dataset_loader.py:368 does f"{dataset_desc} {ch_desc}")
        if dataset_desc:
            channel_descriptions.append(f"{dataset_desc} {ch_desc}")
        else:
            channel_descriptions.append(ch_desc)

    return {
        'sampling_rate_hz': sampling_rate,
        'channel_descriptions': channel_descriptions,
        'dataset_description': dataset_desc,
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
) -> np.ndarray:
    """Extract TSFM embeddings from raw sensor data.

    Args:
        model: Loaded TSFM SemanticAlignmentModel
        raw_data: (N, seq_len, 6) raw sensor data at native rate
        device: torch device
        sampling_rate: native sampling rate in Hz
        channel_descriptions: per-channel descriptions from manifest
        batch_size: batch size for inference
        patch_size_sec: patch duration in seconds

    Returns:
        embeddings: (N, 384) L2-normalized embeddings
    """
    model.eval()
    N = raw_data.shape[0]
    seq_len = raw_data.shape[1]

    all_embeddings = []
    for start in tqdm(range(0, N, batch_size), desc="TSFM | Extracting embeddings",
                      total=(N + batch_size - 1) // batch_size, leave=True):
        end = min(start + batch_size, N)
        batch_data = torch.from_numpy(raw_data[start:end]).float().to(device)
        bs = batch_data.shape[0]

        channel_mask = torch.ones(bs, DATA_CHANNELS, dtype=torch.bool, device=device)
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


def get_window_labels(labels_raw: np.ndarray, label_index: int = 0) -> np.ndarray:
    """Extract per-window activity labels (majority vote)."""
    act_labels = labels_raw[:, :, label_index]
    t = int(np.min(act_labels))
    act_labels = act_labels - t

    window_labels = np.array([
        np.bincount(row.astype(int)).argmax() for row in act_labels
    ], dtype=np.int64)

    return window_labels


def get_dataset_labels(dataset_name: str) -> List[str]:
    """Get sorted activity labels for a dataset."""
    return sorted(DATASET_CONFIG["datasets"][dataset_name]["activities"])


# =============================================================================
# Data Splitting (identical to baselines)
# =============================================================================

def balanced_subsample(data, labels, rate, rng):
    """Balanced subsampling: equal samples per class."""
    unique_labels = np.unique(labels)
    n_total = max(1, int(len(data) * rate))
    n_per_class = max(1, n_total // len(unique_labels))

    selected_idx = []
    for lbl in unique_labels:
        class_idx = np.where(labels == lbl)[0]
        rng.shuffle(class_idx)
        selected_idx.extend(class_idx[:n_per_class])

    selected_idx = np.array(selected_idx)
    rng.shuffle(selected_idx)
    return data[selected_idx], labels[selected_idx]


def prepare_train_test_split(data, labels, training_rate=0.8, vali_rate=0.1,
                              label_rate=1.0, seed=CLASSIFIER_SEED, balance=True):
    """Split data into train/val/test."""
    rng = np.random.RandomState(seed)
    idx = np.arange(len(data))
    rng.shuffle(idx)
    data = data[idx]
    labels = labels[idx]

    train_n = int(len(data) * training_rate)
    vali_n = int(len(data) * vali_rate)

    train_data = data[:train_n]
    train_labels = labels[:train_n]
    vali_data = data[train_n:train_n + vali_n]
    vali_labels = labels[train_n:train_n + vali_n]
    test_data = data[train_n + vali_n:]
    test_labels = labels[train_n + vali_n:]

    if label_rate < 1.0:
        if balance:
            train_data, train_labels = balanced_subsample(
                train_data, train_labels, label_rate, rng)
        else:
            n_labeled = max(1, int(len(train_data) * label_rate))
            train_data = train_data[:n_labeled]
            train_labels = train_labels[:n_labeled]

    return train_data, train_labels, vali_data, vali_labels, test_data, test_labels


# =============================================================================
# Classifier Training (identical to baselines)
# =============================================================================



# =============================================================================
# End-to-End Fine-Tuning (cosine sim with frozen text embeddings)
# =============================================================================

def _forward_batch(model, batch_data, device, sampling_rate, channel_descriptions, seq_len):
    """Run TSFM forward pass on a batch of raw data, returning embeddings.

    Args:
        model: SemanticAlignmentModel
        batch_data: (B, seq_len, 6) tensor on device
        device: torch device
        sampling_rate: native sampling rate in Hz
        channel_descriptions: per-channel descriptions from manifest
        seq_len: sequence length (timesteps per window)

    Returns:
        (B, 384) L2-normalized embeddings
    """
    bs = batch_data.shape[0]
    channel_mask = torch.ones(bs, DATA_CHANNELS, dtype=torch.bool, device=device)
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
                            sampling_rate, channel_descriptions, seq_len):
    """Compute cosine-similarity accuracy on a data loader.

    Args:
        model: SemanticAlignmentModel
        data_loader: yields (batch_data, batch_labels)
        text_embs: (C, 384) frozen text label embeddings
        device: torch device
        sampling_rate: native sampling rate in Hz
        channel_descriptions: per-channel descriptions from manifest
        seq_len: sequence length (timesteps per window)

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
            emb = _forward_batch(model, batch_data, device, sampling_rate, channel_descriptions, seq_len)
            logits = emb @ text_embs.T / FINETUNE_TEMPERATURE
            preds = logits.argmax(dim=1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.shape[0]
    return correct / max(total, 1)


# =============================================================================
# Zero-Shot Evaluation Functions (cosine similarity with label bank)
# =============================================================================

def evaluate_zero_shot_open_set(
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    test_dataset: str,
    label_bank: LearnableLabelBank,
    device: torch.device,
) -> Dict[str, float]:
    """True zero-shot open-set: cosine similarity against all 87 training labels.

    No classifier training. Uses label bank to encode ALL training labels as
    text embeddings, then argmax cosine similarity. Match via synonym groups.
    """
    label_to_group = get_label_to_group_mapping()
    test_activities = get_dataset_labels(test_dataset)

    print(f"  [Zero-shot open-set] Encoding {len(GLOBAL_LABELS)} training labels...")
    with torch.no_grad():
        label_embeddings = label_bank.encode(GLOBAL_LABELS, normalize=True).to(device)

    # Convert test embeddings to tensor
    test_emb_t = torch.from_numpy(test_embeddings).float().to(device)

    # Cosine similarity (both are L2-normalized)
    similarity = compute_similarity(test_emb_t, label_embeddings)  # (N, 87)
    pred_indices = similarity.argmax(dim=1).cpu().numpy()

    # Map through synonym groups
    pred_groups = []
    gt_groups = []
    for i in range(len(test_labels)):
        local_idx = test_labels[i]
        if local_idx < len(test_activities):
            gt_name = test_activities[local_idx]
        else:
            continue
        gt_group = label_to_group.get(gt_name, gt_name)

        pred_idx = pred_indices[i]
        pred_name = GLOBAL_LABELS[pred_idx] if pred_idx < len(GLOBAL_LABELS) else "unknown"
        pred_group = label_to_group.get(pred_name, pred_name)

        gt_groups.append(gt_group)
        pred_groups.append(pred_group)

    acc = accuracy_score(gt_groups, pred_groups) * 100
    f1 = f1_score(gt_groups, pred_groups, average='macro', zero_division=0) * 100
    f1_w = f1_score(gt_groups, pred_groups, average='weighted', zero_division=0) * 100

    return {'accuracy': acc, 'f1_macro': f1, 'f1_weighted': f1_w, 'n_samples': len(gt_groups),
            'n_classes_train': len(GLOBAL_LABELS)}


def evaluate_zero_shot_closed_set(
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    test_dataset: str,
    label_bank: LearnableLabelBank,
    device: torch.device,
) -> Dict[str, float]:
    """True zero-shot closed-set: cosine similarity against test dataset labels only.

    No classifier training. Encode only the test dataset's activity labels,
    then argmax cosine similarity. Exact match evaluation.
    """
    test_activities = get_dataset_labels(test_dataset)
    num_test_classes = len(test_activities)

    print(f"  [Zero-shot closed-set] Encoding {num_test_classes} test labels...")
    with torch.no_grad():
        label_embeddings = label_bank.encode(test_activities, normalize=True).to(device)

    # Convert test embeddings to tensor
    test_emb_t = torch.from_numpy(test_embeddings).float().to(device)

    # Cosine similarity
    similarity = compute_similarity(test_emb_t, label_embeddings)  # (N, C)
    pred_indices = similarity.argmax(dim=1).cpu().numpy()

    gt_names = []
    pred_names = []
    for i in range(len(test_labels)):
        local_idx = test_labels[i]
        if local_idx < len(test_activities):
            gt_names.append(test_activities[local_idx])
        else:
            continue
        pred_idx = pred_indices[i]
        pred_names.append(test_activities[pred_idx] if pred_idx < len(test_activities) else "unknown")

    acc = accuracy_score(gt_names, pred_names) * 100
    f1 = f1_score(
        gt_names, pred_names, labels=test_activities,
        average='macro', zero_division=0,
    ) * 100
    f1_w = f1_score(
        gt_names, pred_names, labels=test_activities,
        average='weighted', zero_division=0,
    ) * 100

    return {'accuracy': acc, 'f1_macro': f1, 'f1_weighted': f1_w, 'n_samples': len(gt_names),
            'n_classes': num_test_classes}


def evaluate_supervised_finetune(
    model: SemanticAlignmentModel,
    label_bank: LearnableLabelBank,
    raw_data: np.ndarray,
    test_labels: np.ndarray,
    test_dataset: str,
    device: torch.device,
    sampling_rate: float,
    channel_descriptions: List[str],
    label_rate: float = 0.01,
    label_tag: str = "1%",
) -> Dict[str, float]:
    """Supervised evaluation via end-to-end fine-tuning with cosine similarity.

    Deep-copies the model, fine-tunes encoder + semantic head end-to-end.
    Classification is via cosine similarity against frozen text label embeddings
    (no separate classifier head). This matches TSFM's native learning mechanism.

    Args:
        model: Pretrained SemanticAlignmentModel (will be deep-copied, not modified)
        label_bank: Trained LearnableLabelBank for text embeddings
        raw_data: (N, seq_len, 6) raw sensor data at native rate
        test_labels: (N,) integer class indices
        test_dataset: dataset name for looking up activity labels
        device: torch device
        sampling_rate: native sampling rate in Hz
        channel_descriptions: per-channel descriptions from manifest
        label_rate: Fraction of training portion to use (0.01 = 1%, 0.10 = 10%)
        label_tag: Display tag for logging
    """
    test_activities = get_dataset_labels(test_dataset)
    num_test_classes = len(test_activities)

    print(f"  [{label_tag} supervised FT] Total samples: {len(raw_data)}, {num_test_classes} classes")

    # Split raw data into 80/10/10
    train_data, train_labels_arr, val_data, val_labels_arr, test_data, test_labels_arr = \
        prepare_train_test_split(
            raw_data, test_labels,
            training_rate=TRAINING_RATE,
            vali_rate=VALI_RATE,
            label_rate=label_rate,
            seed=CLASSIFIER_SEED,
            balance=True,
        )

    print(f"  [{label_tag} supervised FT] Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    if len(train_data) == 0:
        return {'accuracy': 0.0, 'f1_macro': 0.0, 'f1_weighted': 0.0,
                'n_samples': 0, 'n_classes': num_test_classes}

    # Get frozen text embeddings for this dataset's labels
    with torch.no_grad():
        text_embs = label_bank.encode(test_activities, normalize=True).to(device)  # (C, 384)

    # Deep-copy model for fine-tuning (don't modify the original)
    ft_model = copy.deepcopy(model)
    ft_model.train()

    # Set up optimizer (AdamW, conservative LR for pretrained encoder)
    optimizer = torch.optim.AdamW(
        ft_model.parameters(), lr=FINETUNE_ENCODER_LR, weight_decay=FINETUNE_WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # Data loaders
    train_ds = TensorDataset(
        torch.from_numpy(train_data).float(),
        torch.from_numpy(train_labels_arr).long(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(val_data).float(),
        torch.from_numpy(val_labels_arr).long(),
    )
    test_ds = TensorDataset(
        torch.from_numpy(test_data).float(),
        torch.from_numpy(test_labels_arr).long(),
    )
    train_loader = DataLoader(train_ds, batch_size=FINETUNE_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=FINETUNE_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=FINETUNE_BATCH_SIZE, shuffle=False)

    # Fine-tune loop with early stopping
    best_val_acc = -1.0
    best_state = None
    patience_counter = 0

    pbar = tqdm(range(FINETUNE_EPOCHS), desc=f"TSFM | {test_dataset} | FT {label_tag}", leave=True)
    for epoch in pbar:
        ft_model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            seq_len = batch_data.shape[1]
            emb = _forward_batch(ft_model, batch_data, device, sampling_rate, channel_descriptions, seq_len)  # (B, 384)
            logits = emb @ text_embs.T / FINETUNE_TEMPERATURE  # (B, C)
            loss = criterion(logits, batch_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        seq_len = raw_data.shape[1]
        val_acc = compute_cosine_accuracy(ft_model, val_loader, text_embs, device,
                                          sampling_rate, channel_descriptions, seq_len)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(ft_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        avg_loss = epoch_loss / max(1, n_batches)
        pbar.set_postfix(loss=f"{avg_loss:.4f}", val_acc=f"{val_acc:.3f}", best=f"{best_val_acc:.3f}")

        if patience_counter >= FINETUNE_PATIENCE:
            print(f"    Early stopping at epoch {epoch + 1} (patience={FINETUNE_PATIENCE})")
            break

    # Restore best model
    if best_state is not None:
        ft_model.load_state_dict(best_state)

    # Evaluate on test split
    ft_model.train(False)
    all_preds = []
    all_gt = []
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            seq_len = batch_data.shape[1]
            emb = _forward_batch(ft_model, batch_data, device, sampling_rate, channel_descriptions, seq_len)
            logits = emb @ text_embs.T / FINETUNE_TEMPERATURE
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_gt.extend(batch_labels.numpy())

    # Convert to activity names for scoring
    gt_names = []
    pred_names = []
    for i in range(len(all_gt)):
        local_idx = all_gt[i]
        if local_idx < len(test_activities):
            gt_names.append(test_activities[local_idx])
        else:
            continue
        pred_idx = all_preds[i]
        pred_names.append(test_activities[pred_idx] if pred_idx < len(test_activities) else "unknown")

    acc = accuracy_score(gt_names, pred_names) * 100
    f1 = f1_score(
        gt_names, pred_names, labels=test_activities,
        average='macro', zero_division=0,
    ) * 100
    f1_w = f1_score(
        gt_names, pred_names, labels=test_activities,
        average='weighted', zero_division=0,
    ) * 100

    # Clean up to free GPU memory
    del best_state
    del ft_model
    torch.cuda.empty_cache()

    return {'accuracy': acc, 'f1_macro': f1, 'f1_weighted': f1_w, 'n_samples': len(gt_names),
            'n_train_samples': len(train_data), 'n_classes': num_test_classes}


# =============================================================================
# Main
# =============================================================================

def print_results_table(all_results):
    """Print results table."""
    print()
    print("=" * 130)
    print("TSFM EVALUATION RESULTS")
    print("=" * 130)

    header = (f"{'Dataset':<16}"
              f"{'ZS-Open Acc':>13}{'ZS-Open F1':>13}"
              f"{'ZS-Close Acc':>14}{'ZS-Close F1':>13}"
              f"{'1%FT Acc':>11}{'1%FT F1':>10}"
              f"{'10%FT Acc':>12}{'10%FT F1':>11}")
    print(header)
    print("-" * 130)

    for ds in TEST_DATASETS:
        if ds not in all_results:
            continue
        r = all_results[ds]

        def g(key, metric):
            return r.get(key, {}).get(metric, 0.0)

        print(f"{ds:<16}"
              f"{g('zero_shot_open_set','accuracy'):>12.1f}%{g('zero_shot_open_set','f1_macro'):>12.1f}%"
              f"{g('zero_shot_closed_set','accuracy'):>13.1f}%{g('zero_shot_closed_set','f1_macro'):>12.1f}%"
              f"{g('1pct_supervised','accuracy'):>10.1f}%{g('1pct_supervised','f1_macro'):>9.1f}%"
              f"{g('10pct_supervised','accuracy'):>11.1f}%{g('10pct_supervised','f1_macro'):>10.1f}%")

    print("=" * 130)
    print()
    print("Details:")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  Embedding dim: {TSFM_EMB_DIM}")
    print(f"  Native sampling rates, dataset-specific channel descriptions from manifests")
    print(f"  Patch size: {PATCH_SIZE_SEC}s (fixed, no per-dataset sweep)")
    print(f"  Zero-shot: Cosine similarity with LearnableLabelBank text embeddings")
    print(f"  Supervised: End-to-end fine-tuning, cosine sim with frozen text embeddings, "
          f"{FINETUNE_EPOCHS} epochs, lr={FINETUNE_ENCODER_LR}")


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load TSFM model + label bank
    print(f"\nLoading TSFM model from {CHECKPOINT_PATH}...")
    model, checkpoint, hyperparams_path = load_tsfm_model(CHECKPOINT_PATH, device)
    label_bank = load_label_bank(checkpoint, device, hyperparams_path)
    print("Model and label bank loaded successfully")

    # Run scoring on each test dataset
    all_results = {}

    for test_ds in TEST_DATASETS:
        print(f"\n{'='*60}")
        print(f"Testing TSFM on {test_ds}")
        print(f"{'='*60}")

        # Check if processed data exists
        processed_dir = TSFM_EVAL_DIR / test_ds
        if not processed_dir.exists() or not (processed_dir / "data_native.npy").exists():
            print(f"  SKIPPED: No processed data found at {processed_dir}")
            print(f"  Run: python benchmark_data/scripts/preprocess_tsfm_eval.py --datasets {test_ds}")
            continue

        # Load per-dataset metadata from manifest
        meta = get_dataset_metadata(test_ds)
        raw_data, raw_labels, sr = load_raw_data(test_ds)
        test_labels = get_window_labels(raw_labels)
        test_activities = get_dataset_labels(test_ds)
        ch_descs = meta['channel_descriptions']

        print(f"  Sampling rate: {sr}Hz, Seq len: {raw_data.shape[1]}, Patch: {PATCH_SIZE_SEC}s")
        print(f"  Channel descriptions: {ch_descs}")

        # Extract embeddings with native rate and manifest channel descriptions
        test_emb = extract_tsfm_embeddings(
            model, raw_data, device,
            sampling_rate=sr,
            channel_descriptions=ch_descs,
            patch_size_sec=PATCH_SIZE_SEC,
        )

        print(f"  Test data: {test_emb.shape[0]} windows -> embeddings {test_emb.shape}, "
              f"{len(test_activities)} classes")
        print(f"  Classes: {test_activities}")

        ds_results = {'patch_size': PATCH_SIZE_SEC, 'sampling_rate_hz': sr}

        # 1. Zero-shot open-set (cosine similarity)
        print(f"\n  --- Zero-shot Open-Set (cosine sim) ---")
        ds_results['zero_shot_open_set'] = evaluate_zero_shot_open_set(
            test_emb, test_labels, test_ds, label_bank, device)
        print(f"  ZS Open-set: Acc={ds_results['zero_shot_open_set']['accuracy']:.1f}%, "
              f"F1={ds_results['zero_shot_open_set']['f1_macro']:.1f}%")

        # 2. Zero-shot closed-set (cosine similarity)
        print(f"\n  --- Zero-shot Closed-Set (cosine sim) ---")
        ds_results['zero_shot_closed_set'] = evaluate_zero_shot_closed_set(
            test_emb, test_labels, test_ds, label_bank, device)
        print(f"  ZS Closed-set: Acc={ds_results['zero_shot_closed_set']['accuracy']:.1f}%, "
              f"F1={ds_results['zero_shot_closed_set']['f1_macro']:.1f}%")

        # 3. 1% supervised (end-to-end fine-tuning)
        print(f"\n  --- 1% Supervised (End-to-End Fine-Tuning) ---")
        ds_results['1pct_supervised'] = evaluate_supervised_finetune(
            model, label_bank, raw_data, test_labels, test_ds, device,
            sampling_rate=sr, channel_descriptions=ch_descs,
            label_rate=SUPERVISED_LABEL_RATE_1PCT, label_tag="1%")
        print(f"  1% supervised: "
              f"Acc={ds_results['1pct_supervised']['accuracy']:.1f}%, "
              f"F1={ds_results['1pct_supervised']['f1_macro']:.1f}%")

        # 4. 10% supervised (end-to-end fine-tuning)
        print(f"\n  --- 10% Supervised (End-to-End Fine-Tuning) ---")
        ds_results['10pct_supervised'] = evaluate_supervised_finetune(
            model, label_bank, raw_data, test_labels, test_ds, device,
            sampling_rate=sr, channel_descriptions=ch_descs,
            label_rate=SUPERVISED_LABEL_RATE_10PCT, label_tag="10%")
        print(f"  10% supervised: "
              f"Acc={ds_results['10pct_supervised']['accuracy']:.1f}%, "
              f"F1={ds_results['10pct_supervised']['f1_macro']:.1f}%")

        all_results[test_ds] = ds_results

    # Print summary table
    print_results_table(all_results)

    # Save results
    results_path = OUTPUT_DIR / "tsfm_evaluation.json"
    save_data = {}
    for ds, metrics in all_results.items():
        save_data[ds] = {}
        for metric_name, metric_vals in metrics.items():
            if isinstance(metric_vals, dict):
                save_data[ds][metric_name] = {
                    k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in metric_vals.items()
                }
            else:
                save_data[ds][metric_name] = float(metric_vals) if isinstance(metric_vals, (np.floating, float)) else metric_vals

    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
