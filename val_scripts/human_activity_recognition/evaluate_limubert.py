"""
LiMU-BERT evaluation using the new evaluation framework.

LiMU-BERT is NOT text-aligned, so zero-shot evaluation is N/A.
Evaluates with:
  1. 1% supervised (GRU classifier on 1% of test data - paper's architecture)
  2. 10% supervised (GRU classifier on 10% of test data)
  3. Linear probe (linear classifier on frozen mean-pooled embeddings, full train split)

Uses LiMU-BERT's pretrained embeddings (N, 120, 72).
GRU classifier uses sub-windows (M, 20, 72) matching LiMU-BERT's original format.
Linear probe uses mean-pooled (N, 72) embeddings.

Usage:
    python val_scripts/human_activity_recognition/evaluate_limubert.py
"""

import json
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

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
GRU_EPOCHS = 700  # Original: 700 (train.json) - GRU downstream classifier
LINEAR_PROBE_EPOCHS = 100  # Standard for linear probes (not in original paper)
CLASSIFIER_BATCH_SIZE = 128
CLASSIFIER_LR = 1e-3
CLASSIFIER_SEED = 3431

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
# Linear Classifier (for linear probe)
# =============================================================================

class LinearClassifier(nn.Module):
    """Linear classifier for measuring representation quality.

    Architecture: Dropout(0.3) -> Linear(input_dim, num_classes)
    """
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x, training=False):
        if training:
            x = F.dropout(x, p=0.3, training=True)
        return self.linear(x)


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
    data = data[keep]
    label_out = np.array(label_out, dtype=np.int64)

    return data, label_out


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
    lr=CLASSIFIER_LR, device=None, verbose=False,
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

    for epoch in range(epochs):
        model.train()
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_data, training=True)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

        model.eval()
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

        if verbose and (epoch + 1) % 20 == 0:
            val_f1 = f1_score(val_gt, val_preds, average='macro', zero_division=0)
            print(f"    Epoch {epoch+1}/{epochs}: val_acc={val_acc:.3f}, val_f1={val_f1:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_gru(model, data, batch_size=CLASSIFIER_BATCH_SIZE, device=None):
    """Get predictions from a trained GRU classifier."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
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


# =============================================================================
# Linear Classifier Training (for linear probe)
# =============================================================================

def train_linear_classifier(
    train_data, train_labels, val_data, val_labels,
    num_classes, input_dim=EMB_DIM,
    epochs=LINEAR_PROBE_EPOCHS, batch_size=CLASSIFIER_BATCH_SIZE,
    lr=CLASSIFIER_LR, device=None, verbose=False,
):
    """Train a linear classifier on mean-pooled embeddings."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LinearClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
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

    for epoch in range(epochs):
        model.train()
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_data, training=True)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

        model.eval()
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

        if verbose and (epoch + 1) % 20 == 0:
            val_f1 = f1_score(val_gt, val_preds, average='macro', zero_division=0)
            print(f"    Epoch {epoch+1}/{epochs}: val_acc={val_acc:.3f}, val_f1={val_f1:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_linear(model, data, batch_size=CLASSIFIER_BATCH_SIZE, device=None):
    """Get predictions from a trained linear classifier."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
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


def evaluate_supervised_gru(
    test_embeddings: np.ndarray,
    test_labels_raw: np.ndarray,
    test_dataset: str,
    device: torch.device,
    label_rate: float = 0.01,
    label_tag: str = "1%",
) -> Dict[str, float]:
    """Supervised with GRU classifier (paper's architecture).

    Uses sub-window format (M, 20, 72) matching LiMU-BERT's original setup.
    Splits full windows FIRST, then reshapes each partition into sub-windows
    independently, matching the original LiMU-BERT partition_and_reshape() flow.

    Args:
        label_rate: Fraction of training portion to use (0.01 = 1%, 0.10 = 10%)
        label_tag: Display tag for logging
    """
    test_activities = get_dataset_labels(test_dataset)
    num_test_classes = len(test_activities)

    # Compute global label offset BEFORE splitting (matching original LiMU-BERT)
    global_label_offset = int(np.min(test_labels_raw[:, :, 0]))

    # Split full windows FIRST (prevents data leakage)
    train_emb, train_lab_raw, val_emb, val_lab_raw, test_emb, test_lab_raw = \
        split_full_windows(test_embeddings, test_labels_raw)

    # Reshape each partition into sub-windows independently, using global offset
    train_data, train_labels = reshape_and_merge(train_emb, train_lab_raw, label_offset=global_label_offset)
    val_data, val_labels = reshape_and_merge(val_emb, val_lab_raw, label_offset=global_label_offset)
    eval_data, eval_labels = reshape_and_merge(test_emb, test_lab_raw, label_offset=global_label_offset)

    print(f"  [{label_tag} supervised GRU] Total sub-windows: "
          f"train={len(train_data)}, val={len(val_data)}, test={len(eval_data)}, "
          f"{num_test_classes} classes")

    # Apply label_rate subsampling to train set only
    if label_rate < 1.0:
        rng = np.random.RandomState(CLASSIFIER_SEED + 1)
        train_data, train_labels = balanced_subsample(
            train_data, train_labels, label_rate, rng)

    print(f"  [{label_tag} supervised GRU] Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(eval_data)}")

    if len(train_data) == 0:
        return {'accuracy': 0.0, 'f1_macro': 0.0, 'n_samples': 0, 'n_classes': num_test_classes}

    print(f"  [{label_tag} supervised GRU] Training GRU classifier ({num_test_classes} classes)...")
    model = train_gru_classifier(
        train_data, train_labels, val_data, val_labels,
        num_classes=num_test_classes, device=device, verbose=True
    )

    pred_indices = predict_gru(model, eval_data, device=device)

    gt_names = []
    pred_names = []
    for i in range(len(eval_labels)):
        local_idx = eval_labels[i]
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

    return {'accuracy': acc, 'f1_macro': f1, 'n_samples': len(gt_names),
            'n_train_samples': len(train_data), 'n_classes': num_test_classes}


def evaluate_linear_probe(
    test_embeddings: np.ndarray,
    test_labels_raw: np.ndarray,
    test_dataset: str,
    device: torch.device,
) -> Dict[str, float]:
    """Linear probe: linear classifier on frozen mean-pooled embeddings, full train split.

    Measures representation quality. Uses mean-pooled (N, 72) embeddings with
    80% train / 10% val / 10% test split, no subsampling.
    """
    test_activities = get_dataset_labels(test_dataset)
    num_test_classes = len(test_activities)

    test_pooled = mean_pool_embeddings(test_embeddings)
    test_window_labels = get_window_labels(test_labels_raw)

    print(f"  [Linear probe] Total samples: {len(test_pooled)}, {num_test_classes} classes")

    train_data, train_labels, val_data, val_labels, eval_data, eval_labels = \
        prepare_train_test_split(
            test_pooled, test_window_labels,
            training_rate=TRAINING_RATE,
            vali_rate=VALI_RATE,
            label_rate=1.0,
            seed=CLASSIFIER_SEED,
            balance=False
        )

    print(f"  [Linear probe] Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(eval_data)}")

    if len(train_data) == 0:
        return {'accuracy': 0.0, 'f1_macro': 0.0, 'n_samples': 0, 'n_classes': num_test_classes}

    print(f"  [Linear probe] Training linear classifier ({num_test_classes} classes)...")
    model = train_linear_classifier(
        train_data, train_labels, val_data, val_labels,
        num_classes=num_test_classes, device=device, verbose=True
    )

    pred_indices = predict_linear(model, eval_data, device=device)

    gt_names = []
    pred_names = []
    for i in range(len(eval_labels)):
        local_idx = eval_labels[i]
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

    return {'accuracy': acc, 'f1_macro': f1, 'n_samples': len(gt_names),
            'n_train_samples': len(train_data), 'n_classes': num_test_classes}


# =============================================================================
# Output Formatting
# =============================================================================

def print_results_table(all_results: Dict):
    """Print results table."""
    print()
    print("=" * 100)
    print("LIMU-BERT EVALUATION RESULTS (New Framework)")
    print("=" * 100)

    header = (f"{'Dataset':<16}"
              f"{'1%Sup Acc':>11}{'1%Sup F1':>10}"
              f"{'10%Sup Acc':>12}{'10%Sup F1':>11}"
              f"{'LP Acc':>9}{'LP F1':>8}")
    print(header)
    print("-" * 100)

    for ds in TEST_DATASETS:
        if ds not in all_results:
            continue
        r = all_results[ds]

        def g(key, metric):
            return r.get(key, {}).get(metric, 0.0)

        print(f"{ds:<16}"
              f"{g('1pct_supervised','accuracy'):>10.1f}%{g('1pct_supervised','f1_macro'):>9.1f}%"
              f"{g('10pct_supervised','accuracy'):>11.1f}%{g('10pct_supervised','f1_macro'):>10.1f}%"
              f"{g('linear_probe','accuracy'):>8.1f}%{g('linear_probe','f1_macro'):>7.1f}%")

    print("=" * 100)
    print()
    print("Details:")
    print(f"  Supervised: GRU classifier (paper's architecture), {GRU_EPOCHS} epochs, lr={CLASSIFIER_LR}")
    print(f"  Linear probe: Linear classifier on mean-pooled embeddings, {LINEAR_PROBE_EPOCHS} epochs")
    print(f"  Zero-shot: N/A (LiMU-BERT is not text-aligned)")


# =============================================================================
# Main
# =============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for test_ds in TEST_DATASETS:
        print(f"\n{'='*60}")
        print(f"Evaluating LiMU-BERT on {test_ds}")
        print(f"{'='*60}")

        test_emb, test_lab = load_embeddings(test_ds)
        test_activities = get_dataset_labels(test_ds)
        print(f"  Test data: {test_emb.shape[0]} windows, "
              f"{len(test_activities)} classes")
        print(f"  Classes: {test_activities}")

        ds_results = {}

        # 1. 1% supervised (GRU)
        print(f"\n  --- 1% Supervised (GRU) ---")
        ds_results['1pct_supervised'] = evaluate_supervised_gru(
            test_emb, test_lab, test_ds, device,
            label_rate=SUPERVISED_LABEL_RATE_1PCT, label_tag="1%")
        print(f"  1% supervised: "
              f"Acc={ds_results['1pct_supervised']['accuracy']:.1f}%, "
              f"F1={ds_results['1pct_supervised']['f1_macro']:.1f}%")

        # 2. 10% supervised (GRU)
        print(f"\n  --- 10% Supervised (GRU) ---")
        ds_results['10pct_supervised'] = evaluate_supervised_gru(
            test_emb, test_lab, test_ds, device,
            label_rate=SUPERVISED_LABEL_RATE_10PCT, label_tag="10%")
        print(f"  10% supervised: "
              f"Acc={ds_results['10pct_supervised']['accuracy']:.1f}%, "
              f"F1={ds_results['10pct_supervised']['f1_macro']:.1f}%")

        # 3. Linear probe (mean-pooled + linear classifier, full train split)
        print(f"\n  --- Linear Probe ---")
        ds_results['linear_probe'] = evaluate_linear_probe(
            test_emb, test_lab, test_ds, device)
        print(f"  Linear probe: "
              f"Acc={ds_results['linear_probe']['accuracy']:.1f}%, "
              f"F1={ds_results['linear_probe']['f1_macro']:.1f}%")

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
