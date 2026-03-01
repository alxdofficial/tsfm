"""
LiMU-BERT scoring using the unified framework.

Evaluates with:
  1. Zero-shot open-set (classifier on 87 training labels, group scoring)
  2. Zero-shot closed-set (classifier with masked logits, group scoring)
  3. 1% supervised end-to-end fine-tuning (encoder + GRU)
  4. 10% supervised end-to-end fine-tuning (encoder + GRU)

Uses LiMU-BERT's pretrained embeddings (N, 120, 72) for zero-shot.
End-to-end fine-tuning loads the pretrained encoder and raw sensor data,
fine-tuning encoder + GRU jointly with differential learning rates.
GRU classifier uses sub-windows (M, 20, 72) matching LiMU-BERT's original format.
Zero-shot uses mean-pooled (N, 72) embeddings.

Usage:
    python val_scripts/human_activity_recognition/evaluate_limubert.py
"""

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LIMUBERT_REPO = PROJECT_ROOT / "auxiliary_repos" / "LIMU-BERT-Public"
sys.path.insert(0, str(LIMUBERT_REPO))

from val_scripts.human_activity_recognition.grouped_zero_shot import (
    load_global_labels, map_local_to_global_labels,
    get_closed_set_mask, score_with_groups,
)

# =============================================================================
# Configuration
# =============================================================================

LIMUBERT_DIR = PROJECT_ROOT / "auxiliary_repos" / "LIMU-BERT-Public"
BENCHMARK_DIR = PROJECT_ROOT / "benchmark_data"
DATASET_CONFIG_PATH = BENCHMARK_DIR / "dataset_config.json"
OUTPUT_DIR = PROJECT_ROOT / "test_output" / "baseline_evaluation"

# LiMU-BERT settings
LIMUBERT_EMBED_DIR = LIMUBERT_DIR / "embed"
LIMUBERT_DATASET_DIR = LIMUBERT_DIR / "dataset"
LIMUBERT_PRETRAIN_NAME = "pretrained_combined"
LIMUBERT_VERSION = "20_120"

# Training hyperparameters (match LiMU-BERT config/train.json)
GRU_EPOCHS = 100  # Original repo provides train_100ep.json (same config, fewer epochs)
CLASSIFIER_BATCH_SIZE = 512
CLASSIFIER_LR = 1e-3
CLASSIFIER_SEED = 3431

# Fine-tuning hyperparameters (encoder + GRU, end-to-end)
FINETUNE_EPOCHS = 20
FINETUNE_BATCH_SIZE = 32
FINETUNE_ENCODER_LR = 1e-5
FINETUNE_HEAD_LR = 1e-3
FINETUNE_WEIGHT_DECAY = 1e-5
FINETUNE_PATIENCE = 5

# Embedding parameters
EMB_DIM = 72       # LiMU-BERT hidden dimension
MERGE_LEN = 20     # Sub-window length for GRU classifier (matches LiMU-BERT)
SEQ_LEN = 120      # Full window length

# Data split parameters
TRAINING_RATE = 0.8
VALI_RATE = 0.1
SUPERVISED_LABEL_RATE_1PCT = 0.01
SUPERVISED_LABEL_RATE_10PCT = 0.10

# Load configs
with open(DATASET_CONFIG_PATH) as f:
    DATASET_CONFIG = json.load(f)

TRAIN_DATASETS = DATASET_CONFIG["train_datasets"]
TEST_DATASETS = DATASET_CONFIG["zero_shot_datasets"]


# =============================================================================
# GRU Classifier (matches LiMU-BERT architecture)
# =============================================================================

class GRUClassifier(nn.Module):
    """GRU classifier matching LiMU-BERT's ClassifierGRU (gru_v2 config).

    Architecture:
        GRU1: (input_dim, 20) x 2 layers
        GRU2: (20, 10) x 1 layer
        Dropout
        Linear: (10, num_classes)
    """
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.gru1 = nn.GRU(input_dim, 20, num_layers=2, batch_first=True)
        self.gru2 = nn.GRU(20, 10, num_layers=1, batch_first=True)
        self.linear = nn.Linear(10, num_classes)

    def forward(self, x, training=False):
        h, _ = self.gru1(x)
        h, _ = self.gru2(h)
        h = h[:, -1, :]  # Last timestep
        if training:
            h = F.dropout(h, p=0.5, training=True)
        h = self.linear(h)
        return h


# =============================================================================
# Model Loading
# =============================================================================

def load_limubert_model(device):
    """Load pretrained LiMU-BERT model for end-to-end fine-tuning."""
    from models import LIMUBertModel4Pretrain
    from config import PretrainModelConfig

    cfg = PretrainModelConfig(
        hidden=72, hidden_ff=144, feature_num=6,
        n_layers=4, n_heads=4, seq_len=120, emb_norm=True
    )
    model = LIMUBertModel4Pretrain(cfg, output_embed=True)
    checkpoint_path = LIMUBERT_REPO / "saved" / "pretrain_base_recgym_20_120" / "pretrained_combined.pt"
    model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
    model.to(device)
    return model


# =============================================================================
# Data Loading and Preparation
# =============================================================================

def load_embeddings(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load LiMU-BERT embeddings and labels for a dataset."""
    embed_path = LIMUBERT_EMBED_DIR / f"embed_{LIMUBERT_PRETRAIN_NAME}_{dataset_name}_{LIMUBERT_VERSION}.npy"
    label_path = LIMUBERT_DATASET_DIR / dataset_name / f"label_{LIMUBERT_VERSION}.npy"

    embeddings = np.load(str(embed_path)).astype(np.float32)
    labels = np.load(str(label_path)).astype(np.float32)
    return embeddings, labels


def load_raw_data(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load raw sensor data and labels for a dataset."""
    ds_dir = PROJECT_ROOT / "benchmark_data" / "processed" / "limubert" / dataset_name
    data = np.load(str(ds_dir / "data_20_120.npy")).astype(np.float32)
    labels = np.load(str(ds_dir / "label_20_120.npy")).astype(np.float32)
    return data, labels


def normalize_for_limubert(data: np.ndarray) -> np.ndarray:
    """Normalize raw data for LiMU-BERT: acc channels / 9.8.

    Matches LiMU-BERT's Preprocess4Normalization.
    """
    normed = data.copy()
    normed[:, :, :3] = normed[:, :, :3] / 9.8  # acc channels
    return normed


def get_dataset_labels(dataset_name: str) -> List[str]:
    """Get sorted activity labels for a dataset."""
    return sorted(DATASET_CONFIG["datasets"][dataset_name]["activities"])


def reshape_and_merge(embeddings: np.ndarray, labels_raw: np.ndarray,
                       label_index: int = 0, label_offset: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape embeddings and labels to LiMU-BERT classifier format.

    1. Extract activity labels: (N, 120, 2) -> (N, 120)
    2. Reshape into sub-windows of MERGE_LEN: (N*K, MERGE_LEN, D) and (N*K, MERGE_LEN)
    3. Filter to keep only windows where all timesteps have the same label
    4. Return (M, MERGE_LEN, D) data and (M,) 1D labels

    Args:
        label_offset: Pre-computed global label minimum (t). If None, computed
            from this partition. Must be provided when calling on individual
            splits to ensure consistent label indices across train/val/test.
    """
    N = embeddings.shape[0]
    D = embeddings.shape[2]
    K = SEQ_LEN // MERGE_LEN  # 120/20 = 6

    act_labels = labels_raw[:, :, label_index]  # (N, 120)
    if label_offset is None:
        t = int(np.min(act_labels))
    else:
        t = label_offset
    act_labels = act_labels - t

    data = embeddings.reshape(N * K, MERGE_LEN, D)
    labels = act_labels.reshape(N * K, MERGE_LEN)

    keep = []
    label_out = []
    for i in range(labels.shape[0]):
        unique = np.unique(labels[i])
        if unique.size == 1:
            keep.append(i)
            label_out.append(int(unique[0]))

    keep = np.array(keep)
    parent_window_ids = keep // K  # which original window each sub-window belongs to
    data = data[keep]
    label_out = np.array(label_out, dtype=np.int64)

    return data, label_out, parent_window_ids


def get_window_labels(labels_raw: np.ndarray, label_index: int = 0) -> np.ndarray:
    """Extract per-window activity labels via majority vote."""
    act_labels = labels_raw[:, :, label_index]
    t = int(np.min(act_labels))
    act_labels = act_labels - t
    window_labels = np.array(
        [np.bincount(row.astype(int)).argmax() for row in act_labels],
        dtype=np.int64
    )
    return window_labels


def majority_vote_subwindows(sub_preds: np.ndarray, parent_ids: np.ndarray,
                             n_windows: int) -> np.ndarray:
    """Aggregate sub-window predictions to window-level via majority vote.

    Args:
        sub_preds: (M,) per-sub-window predictions (global label indices)
        parent_ids: (M,) parent window index for each sub-window
        n_windows: N, total number of original windows

    Returns:
        (N,) window-level predictions. Windows with no valid sub-windows
        get the most common prediction as fallback.
    """
    window_preds = np.full(n_windows, -1, dtype=np.int64)
    for w in range(n_windows):
        mask = parent_ids == w
        if mask.any():
            votes = sub_preds[mask]
            window_preds[w] = np.bincount(votes).argmax()

    # Fallback for windows with no valid sub-windows (all sub-windows filtered out)
    if (window_preds == -1).any():
        fallback = np.bincount(sub_preds).argmax()
        window_preds[window_preds == -1] = fallback

    return window_preds


def mean_pool_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Mean-pool (N, 120, 72) embeddings to (N, 72)."""
    return embeddings.mean(axis=1)


def balanced_subsample(data: np.ndarray, labels: np.ndarray,
                        rate: float, rng: np.random.RandomState):
    """Balanced subsampling matching LiMU-BERT's prepare_simple_dataset_balance."""
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
    """Split data into train/val/test and optionally subsample training labels."""
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
# GRU Classifier Training (paper's original architecture)
# =============================================================================

def train_gru_classifier(
    train_data, train_labels, val_data, val_labels,
    num_classes, epochs=GRU_EPOCHS, batch_size=CLASSIFIER_BATCH_SIZE,
    lr=CLASSIFIER_LR, device=None, verbose=False, desc="GRU",
):
    """Train a GRU classifier on embeddings."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GRUClassifier(input_dim=train_data.shape[-1], num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.from_numpy(train_data).float(),
        torch.from_numpy(train_labels).long()
    )
    val_ds = TensorDataset(
        torch.from_numpy(val_data).float(),
        torch.from_numpy(val_labels).long()
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_acc = 0.0
    best_state = None

    pbar = tqdm(range(epochs), desc=desc, leave=True)
    for epoch in pbar:
        model.train()
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_data, training=True)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

        model.train(False)
        val_preds = []
        val_gt = []
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                logits = model(batch_data, training=False)
                preds = logits.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_gt.extend(batch_labels.numpy())

        val_acc = accuracy_score(val_gt, val_preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        pbar.set_postfix(val_acc=f"{val_acc:.3f}", best=f"{best_val_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.train(False)
    return model


def predict_gru(model, data, batch_size=CLASSIFIER_BATCH_SIZE, device=None):
    """Get predictions from a trained GRU classifier."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.train(False)
    ds = TensorDataset(torch.from_numpy(data).float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for (batch_data,) in loader:
            batch_data = batch_data.to(device)
            logits = model(batch_data, training=False)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    return np.array(all_preds)


def predict_gru_global(model, data, device, logit_mask=None,
                       batch_size=CLASSIFIER_BATCH_SIZE):
    """Predict global label indices using GRU classifier.

    Args:
        model: Trained GRUClassifier with num_classes=87
        data: (M, 20, 72) sub-window embeddings
        device: torch device
        logit_mask: Optional (87,) bool mask for closed-set scoring.
            If provided, masked-out logits are set to -inf before argmax.

    Returns:
        pred_global_indices: (M,) predicted global label indices
    """
    model.train(False)
    ds = TensorDataset(torch.from_numpy(data).float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    if logit_mask is not None:
        mask_tensor = torch.from_numpy(logit_mask).bool().to(device)

    all_preds = []
    with torch.no_grad():
        for (batch_data,) in loader:
            batch_data = batch_data.to(device)
            logits = model(batch_data, training=False)
            if logit_mask is not None:
                logits[:, ~mask_tensor] = float('-inf')
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    return np.array(all_preds)


# =============================================================================
# Evaluation Functions
# =============================================================================

def split_full_windows(embeddings, labels_raw, training_rate=TRAINING_RATE,
                       vali_rate=VALI_RATE, seed=CLASSIFIER_SEED):
    """Split full windows (N, 120, D) into train/val/test BEFORE reshaping.

    Matching the original LiMU-BERT partition_and_reshape() order:
    split full windows first, then reshape each partition into sub-windows.
    This prevents data leakage from sub-windows of the same original window
    appearing in different splits.
    """
    rng = np.random.RandomState(seed)
    N = len(embeddings)
    idx = np.arange(N)
    rng.shuffle(idx)

    train_n = int(N * training_rate)
    vali_n = int(N * vali_rate)

    train_emb = embeddings[idx[:train_n]]
    train_lab = labels_raw[idx[:train_n]]
    val_emb = embeddings[idx[train_n:train_n + vali_n]]
    val_lab = labels_raw[idx[train_n:train_n + vali_n]]
    test_emb = embeddings[idx[train_n + vali_n:]]
    test_lab = labels_raw[idx[train_n + vali_n:]]

    return train_emb, train_lab, val_emb, val_lab, test_emb, test_lab


def evaluate_supervised_finetune(
    bert_model,
    raw_data: np.ndarray,
    test_labels_raw: np.ndarray,
    test_dataset: str,
    device: torch.device,
    label_rate: float = 0.01,
    label_tag: str = "1%",
) -> Dict[str, float]:
    """End-to-end fine-tuning: encoder + GRU classifier.

    Loads the pretrained LiMU-BERT encoder and fine-tunes it jointly with a
    GRU classifier head. Uses raw sensor data (N, 120, 6), normalizes, then
    forward pass: encoder -> sub-window reshape -> GRU -> logits.

    Splits full windows FIRST (prevents data leakage), then subsamples
    training windows by label_rate.

    Args:
        bert_model: Pretrained LiMU-BERT model (will be deep-copied)
        raw_data: (N, 120, 6) raw sensor data
        test_labels_raw: (N, 120, 2) raw labels
        test_dataset: Dataset name
        device: torch device
        label_rate: Fraction of training portion to use (0.01 = 1%, 0.10 = 10%)
        label_tag: Display tag for logging
    """
    test_activities = get_dataset_labels(test_dataset)
    num_test_classes = len(test_activities)

    # Get window-level labels
    window_labels = get_window_labels(test_labels_raw)

    # Normalize raw data for LiMU-BERT
    normed_data = normalize_for_limubert(raw_data)

    # Split at window level
    train_data, train_labels, val_data, val_labels, test_data_split, test_labels_split = \
        prepare_train_test_split(normed_data, window_labels,
                                 training_rate=TRAINING_RATE, vali_rate=VALI_RATE,
                                 label_rate=label_rate, seed=CLASSIFIER_SEED, balance=True)

    print(f"  [{label_tag} fine-tune] Windows: train={len(train_data)}, "
          f"val={len(val_data)}, test={len(test_data_split)}, {num_test_classes} classes")

    if len(train_data) == 0:
        return {'accuracy': 0.0, 'f1_macro': 0.0, 'f1_weighted': 0.0,
                'n_samples': 0, 'n_classes': num_test_classes}

    # Deep-copy encoder
    ft_encoder = copy.deepcopy(bert_model)
    for p in ft_encoder.parameters():
        p.requires_grad_(True)
    ft_encoder.train()

    # Create GRU classifier
    gru_classifier = GRUClassifier(input_dim=EMB_DIM, num_classes=num_test_classes).to(device)

    optimizer = torch.optim.AdamW([
        {'params': ft_encoder.parameters(), 'lr': FINETUNE_ENCODER_LR},
        {'params': gru_classifier.parameters(), 'lr': FINETUNE_HEAD_LR},
    ], weight_decay=FINETUNE_WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # DataLoaders (full windows)
    train_ds = TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_labels).long())
    val_ds = TensorDataset(torch.from_numpy(val_data).float(), torch.from_numpy(val_labels).long())
    test_ds_loader = TensorDataset(torch.from_numpy(test_data_split).float(), torch.from_numpy(test_labels_split).long())
    train_loader = DataLoader(train_ds, batch_size=FINETUNE_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=FINETUNE_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds_loader, batch_size=FINETUNE_BATCH_SIZE, shuffle=False)

    K = SEQ_LEN // MERGE_LEN  # 6 sub-windows per window

    best_val_acc = -1.0
    best_enc_state = None
    best_gru_state = None
    patience_counter = 0

    pbar = tqdm(range(FINETUNE_EPOCHS), desc=f"LiMU-BERT | {test_dataset} | FT {label_tag}", leave=True)
    for epoch in pbar:
        ft_encoder.train()
        gru_classifier.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)   # (B, 120, 6)
            batch_labels = batch_labels.to(device)  # (B,)
            B = batch_data.shape[0]

            optimizer.zero_grad()
            h = ft_encoder(batch_data)  # (B, 120, 72)
            # Reshape to sub-windows: (B, 120, 72) -> (B*K, 20, 72)
            h_sub = h.reshape(B * K, MERGE_LEN, EMB_DIM)
            # Expand labels: each window's label applies to all K sub-windows
            labels_sub = batch_labels.unsqueeze(1).expand(B, K).reshape(B * K)

            logits = gru_classifier(h_sub, training=True)  # (B*K, num_classes)
            loss = criterion(logits, labels_sub)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        ft_encoder.train(False)
        gru_classifier.train(False)
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                B = batch_data.shape[0]
                h = ft_encoder(batch_data)
                h_sub = h.reshape(B * K, MERGE_LEN, EMB_DIM)
                logits = gru_classifier(h_sub)  # (B*K, num_classes)
                preds = logits.argmax(dim=1).reshape(B, K)
                # Majority vote across K sub-windows per window
                window_preds = torch.mode(preds, dim=1).values
                correct += (window_preds == batch_labels).sum().item()
                total += B
        val_acc = correct / max(total, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_enc_state = copy.deepcopy(ft_encoder.state_dict())
            best_gru_state = copy.deepcopy(gru_classifier.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        avg_loss = epoch_loss / max(1, n_batches)
        pbar.set_postfix(loss=f"{avg_loss:.4f}", val_acc=f"{val_acc:.3f}", best=f"{best_val_acc:.3f}")

        if patience_counter >= FINETUNE_PATIENCE:
            print(f"    Early stopping at epoch {epoch + 1}")
            break

    if best_enc_state is not None:
        ft_encoder.load_state_dict(best_enc_state)
        gru_classifier.load_state_dict(best_gru_state)

    # Test evaluation
    ft_encoder.train(False)
    gru_classifier.train(False)
    all_preds = []
    all_gt = []
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            B = batch_data.shape[0]
            h = ft_encoder(batch_data)
            h_sub = h.reshape(B * K, MERGE_LEN, EMB_DIM)
            logits = gru_classifier(h_sub)
            preds = logits.argmax(dim=1).reshape(B, K)
            window_preds = torch.mode(preds, dim=1).values.cpu().numpy()
            all_preds.extend(window_preds)
            all_gt.extend(batch_labels.numpy())

    # Score
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
    f1 = f1_score(gt_names, pred_names, labels=test_activities, average='macro', zero_division=0) * 100
    f1_w = f1_score(gt_names, pred_names, labels=test_activities, average='weighted', zero_division=0) * 100

    del best_enc_state, best_gru_state
    del ft_encoder, gru_classifier
    torch.cuda.empty_cache()

    return {'accuracy': acc, 'f1_macro': f1, 'f1_weighted': f1_w,
            'n_samples': len(gt_names), 'n_train_samples': len(train_data),
            'n_classes': num_test_classes}


# =============================================================================
# Grouped Zero-Shot: Training Embedding Loading
# =============================================================================

def load_limubert_training_embeddings(global_labels, device):
    """Load pre-extracted LiMU-BERT embeddings for all 10 training datasets.

    Uses reshape_and_merge to produce sub-windows (M, 20, 72) matching
    the GRU classifier's native input format. Labels are mapped to global
    indices (0..86).

    Returns:
        all_embeddings: (M_total, 20, 72) concatenated sub-window embeddings
        all_labels: (M_total,) global label indices
    """
    all_emb_list = []
    all_lab_list = []

    for ds in tqdm(TRAIN_DATASETS, desc="LiMU-BERT | Loading training embeddings", leave=True):
        emb, lab_raw = load_embeddings(ds)
        # reshape_and_merge returns sub-windows with local label indices
        sub_windows, local_labels, _ = reshape_and_merge(emb, lab_raw)  # (M, 20, 72), (M,)
        global_lab = map_local_to_global_labels(local_labels, ds, DATASET_CONFIG, global_labels)
        all_emb_list.append(sub_windows)
        all_lab_list.append(global_lab)

    return np.concatenate(all_emb_list, axis=0), np.concatenate(all_lab_list, axis=0)


# =============================================================================
# Output Formatting
# =============================================================================

def print_results_table(all_results: Dict):
    """Print results table."""
    print()
    print("=" * 130)
    print("LIMU-BERT EVALUATION RESULTS")
    print("=" * 130)

    header = (f"{'Dataset':<16}"
              f"{'ZS-Open Acc':>13}{'ZS-Open F1':>12}"
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
              f"{g('zero_shot_open_set','accuracy'):>12.1f}%{g('zero_shot_open_set','f1_macro'):>11.1f}%"
              f"{g('zero_shot_closed_set','accuracy'):>13.1f}%{g('zero_shot_closed_set','f1_macro'):>12.1f}%"
              f"{g('1pct_supervised','accuracy'):>10.1f}%{g('1pct_supervised','f1_macro'):>9.1f}%"
              f"{g('10pct_supervised','accuracy'):>11.1f}%{g('10pct_supervised','f1_macro'):>10.1f}%")

    print("=" * 130)
    print()
    print("Details:")
    print(f"  Zero-shot: GRU classifier on 10 training datasets (87 classes), sub-window format, group-matched scoring")
    print(f"  Open-set: all 87 logits; Closed-set: logits masked to test-relevant groups")
    print(f"  Supervised: End-to-end fine-tuning (encoder + GRU), {FINETUNE_EPOCHS} epochs, "
          f"encoder_lr={FINETUNE_ENCODER_LR}, head_lr={FINETUNE_HEAD_LR}, patience={FINETUNE_PATIENCE}")


# =============================================================================
# Main
# =============================================================================

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading LiMU-BERT model...")
    limubert_model = load_limubert_model(device)
    print("Model loaded successfully")

    # Load or train zero-shot GRU classifier
    global_labels = load_global_labels()
    zs_clf_path = OUTPUT_DIR / "limubert_zs_gru.pt"

    if zs_clf_path.exists():
        print(f"\nLoading cached zero-shot GRU from {zs_clf_path}...")
        zs_classifier = GRUClassifier(input_dim=EMB_DIM, num_classes=len(global_labels)).to(device)
        zs_classifier.load_state_dict(torch.load(str(zs_clf_path), map_location=device))
        zs_classifier.train(False)
        print(f"  Loaded GRU with {len(global_labels)} classes")
    else:
        print(f"\nLoading training embeddings for zero-shot ({len(TRAIN_DATASETS)} datasets)...")
        train_emb, train_lab = load_limubert_training_embeddings(global_labels, device)
        print(f"Training data: {train_emb.shape[0]} sub-windows, shape={train_emb.shape}, "
              f"{len(np.unique(train_lab))} classes")

        # 90/10 train/val split for zero-shot classifier
        rng = np.random.RandomState(CLASSIFIER_SEED)
        N = len(train_emb)
        indices = np.arange(N)
        rng.shuffle(indices)
        val_n = int(N * 0.1)
        val_idx = indices[:val_n]
        train_idx = indices[val_n:]

        print(f"\nTraining zero-shot GRU classifier "
              f"({len(train_idx)} train, {len(val_idx)} val, 87 classes)...")
        zs_classifier = train_gru_classifier(
            train_emb[train_idx], train_lab[train_idx],
            train_emb[val_idx], train_lab[val_idx],
            num_classes=len(global_labels), device=device,
            desc="LiMU-BERT | ZS GRU",
        )
        torch.save(zs_classifier.state_dict(), str(zs_clf_path))
        print(f"  Saved zero-shot GRU to {zs_clf_path}")

    all_results = {}

    for test_ds in TEST_DATASETS:
        print(f"\n{'='*60}")
        print(f"Evaluating LiMU-BERT on {test_ds}")
        print(f"{'='*60}")

        # Load both embeddings (for zero-shot) and raw data (for fine-tuning)
        try:
            test_emb, test_lab = load_embeddings(test_ds)
            raw_data, raw_labels = load_raw_data(test_ds)
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            continue
        test_activities = get_dataset_labels(test_ds)
        print(f"  Test data: {test_emb.shape[0]} windows, "
              f"{len(test_activities)} classes")
        print(f"  Classes: {test_activities}")

        ds_results = {}

        # Reshape test data into sub-windows for zero-shot GRU
        n_windows = test_emb.shape[0]
        test_sub_windows, test_sub_labels, parent_ids = reshape_and_merge(test_emb, test_lab)  # (M, 20, 72), (M,), (M,)
        # Get window-level labels for scoring (consistent with other models)
        window_labels = get_window_labels(test_lab)
        print(f"  Sub-windows for ZS: {test_sub_windows.shape[0]} from {n_windows} windows")

        # 0. Zero-shot open-set (GRU on sub-windows → majority vote → window-level)
        print(f"\n  --- Zero-Shot Open-Set (GRU) ---")
        sub_pred_open = predict_gru_global(zs_classifier, test_sub_windows, device)
        pred_open = majority_vote_subwindows(sub_pred_open, parent_ids, n_windows)
        ds_results['zero_shot_open_set'] = score_with_groups(
            pred_open, window_labels, test_ds, global_labels, DATASET_CONFIG)
        zs_open = ds_results['zero_shot_open_set']
        print(f"  ZS Open-set: Acc={zs_open['accuracy']:.1f}%, F1={zs_open['f1_macro']:.1f}%")

        # 1. Zero-shot closed-set (GRU with masked logits → majority vote → window-level)
        print(f"\n  --- Zero-Shot Closed-Set (GRU) ---")
        mask = get_closed_set_mask(test_ds, global_labels, DATASET_CONFIG)
        sub_pred_closed = predict_gru_global(zs_classifier, test_sub_windows, device, logit_mask=mask)
        pred_closed = majority_vote_subwindows(sub_pred_closed, parent_ids, n_windows)
        ds_results['zero_shot_closed_set'] = score_with_groups(
            pred_closed, window_labels, test_ds, global_labels, DATASET_CONFIG)
        ds_results['zero_shot_closed_set']['n_allowed'] = int(mask.sum())
        ds_results['zero_shot_closed_set']['n_masked'] = int((~mask).sum())
        zs_close = ds_results['zero_shot_closed_set']
        print(f"  ZS Closed-set: Acc={zs_close['accuracy']:.1f}%, F1={zs_close['f1_macro']:.1f}% "
              f"({zs_close['n_allowed']}/{len(global_labels)} labels allowed)")

        # 3. 1% supervised (end-to-end fine-tuning)
        print(f"\n  --- 1% Supervised (Fine-Tune) ---")
        ds_results['1pct_supervised'] = evaluate_supervised_finetune(
            limubert_model, raw_data, raw_labels, test_ds, device,
            label_rate=SUPERVISED_LABEL_RATE_1PCT, label_tag="1%")
        print(f"  1% fine-tune: "
              f"Acc={ds_results['1pct_supervised']['accuracy']:.1f}%, "
              f"F1={ds_results['1pct_supervised']['f1_macro']:.1f}%")

        # 4. 10% supervised (end-to-end fine-tuning)
        print(f"\n  --- 10% Supervised (Fine-Tune) ---")
        ds_results['10pct_supervised'] = evaluate_supervised_finetune(
            limubert_model, raw_data, raw_labels, test_ds, device,
            label_rate=SUPERVISED_LABEL_RATE_10PCT, label_tag="10%")
        print(f"  10% fine-tune: "
              f"Acc={ds_results['10pct_supervised']['accuracy']:.1f}%, "
              f"F1={ds_results['10pct_supervised']['f1_macro']:.1f}%")

        all_results[test_ds] = ds_results

    # Print summary table
    print_results_table(all_results)

    # Save results
    results_path = OUTPUT_DIR / "limubert_evaluation.json"
    save_data = {}
    for ds, metrics in all_results.items():
        save_data[ds] = {}
        for metric_name, metric_vals in metrics.items():
            save_data[ds][metric_name] = {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in metric_vals.items()
            }
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
