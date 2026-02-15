"""
Unified baseline evaluation framework.

For each baseline model x each test dataset, computes 3 metrics:
  1. Zero-shot open-set accuracy & F1:
     Predict from ALL training labels. Match via synonym groups.
  2. Closed-set accuracy & F1:
     Restrict predictions to test dataset's own labels. Exact match.
  3. 1% supervised accuracy & F1:
     Fine-tune on 1% of test data. Closed-set eval on remaining 99%.

Currently supports:
  - LiMU-BERT (GRU classifier on pretrained embeddings)

Usage:
    python val_scripts/human_activity_recognition/evaluate_baselines.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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

from datasets.imu_pretraining_dataset.label_groups import (
    LABEL_GROUPS,
    get_label_to_group_mapping,
    get_group_for_label,
)

# =============================================================================
# Configuration
# =============================================================================

LIMUBERT_DIR = PROJECT_ROOT / "auxiliary_repos" / "LIMU-BERT-Public"
BENCHMARK_DIR = PROJECT_ROOT / "benchmark_data"
DATASET_CONFIG_PATH = BENCHMARK_DIR / "dataset_config.json"
GLOBAL_LABEL_PATH = BENCHMARK_DIR / "processed" / "limubert" / "global_label_mapping.json"
OUTPUT_DIR = PROJECT_ROOT / "test_output" / "baseline_evaluation"

# LiMU-BERT settings
LIMUBERT_EMBED_DIR = LIMUBERT_DIR / "embed"
LIMUBERT_DATASET_DIR = LIMUBERT_DIR / "dataset"
LIMUBERT_PRETRAIN_NAME = "pretrained_combined"
LIMUBERT_VERSION = "20_120"

# Training hyperparameters (match LiMU-BERT)
CLASSIFIER_EPOCHS = 100
CLASSIFIER_BATCH_SIZE = 128
CLASSIFIER_LR = 1e-3
CLASSIFIER_SEED = 3431

# Embedding parameters
EMB_DIM = 72       # LiMU-BERT hidden dimension
MERGE_LEN = 20     # Sub-window length for classifier (matches LiMU-BERT)
SEQ_LEN = 120      # Full window length

# Data split parameters
TRAINING_RATE = 0.8   # For open-set/closed-set: fraction used as training pool
VALI_RATE = 0.1       # Validation fraction
SUPERVISED_LABEL_RATE = 0.01  # 1% supervised

# Load configs
with open(DATASET_CONFIG_PATH) as f:
    DATASET_CONFIG = json.load(f)

with open(GLOBAL_LABEL_PATH) as f:
    GLOBAL_LABELS = json.load(f)["labels"]

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
# Linear Classifier (fair comparison with other baselines)
# =============================================================================

class LinearClassifier(nn.Module):
    """Linear classifier matching MOMENT, LanHAR, CrossHAR baselines.

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
                       label_index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape embeddings and labels to LiMU-BERT classifier format.

    1. Extract activity labels: (N, 120, 2) -> (N, 120)
    2. Reshape into sub-windows of MERGE_LEN: (N*K, MERGE_LEN, D) and (N*K, MERGE_LEN)
    3. Filter to keep only windows where all timesteps have the same label
    4. Return (M, MERGE_LEN, D) data and (M,) 1D labels

    Args:
        embeddings: (N, 120, 72) embeddings
        labels_raw: (N, 120, 2) labels [activity_idx, subject_idx]
        label_index: which label dimension to use (0=activity)

    Returns:
        data: (M, MERGE_LEN, 72) reshaped embeddings
        labels: (M,) integer activity labels
    """
    N = embeddings.shape[0]
    D = embeddings.shape[2]
    K = SEQ_LEN // MERGE_LEN  # 120/20 = 6

    # Extract activity labels
    act_labels = labels_raw[:, :, label_index]  # (N, 120)

    # Subtract minimum label index (LiMU-BERT convention)
    t = int(np.min(act_labels))
    act_labels = act_labels - t

    # Reshape
    data = embeddings.reshape(N * K, MERGE_LEN, D)
    labels = act_labels.reshape(N * K, MERGE_LEN)

    # Merge: keep only sub-windows with uniform labels
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


def prepare_train_test_split(data: np.ndarray, labels: np.ndarray,
                              training_rate: float = 0.8,
                              vali_rate: float = 0.1,
                              label_rate: float = 1.0,
                              seed: int = CLASSIFIER_SEED,
                              balance: bool = True):
    """Split data into train/val/test and optionally subsample training labels.

    Args:
        data: (M, MERGE_LEN, D)
        labels: (M,) integer labels
        training_rate: fraction for training+validation pool
        vali_rate: fraction for validation
        label_rate: fraction of training data to use as labeled (1.0 = all, 0.01 = 1%)
        seed: random seed
        balance: if True, balance classes when subsampling

    Returns:
        train_data, train_labels, val_data, val_labels, test_data, test_labels
    """
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

    # Subsample training labels
    if label_rate < 1.0:
        if balance:
            # Balanced subsampling: equal samples per class
            train_data, train_labels = balanced_subsample(
                train_data, train_labels, label_rate, rng)
        else:
            n_labeled = max(1, int(len(train_data) * label_rate))
            train_data = train_data[:n_labeled]
            train_labels = train_labels[:n_labeled]

    return train_data, train_labels, vali_data, vali_labels, test_data, test_labels


def get_window_labels(labels_raw: np.ndarray, label_index: int = 0) -> np.ndarray:
    """Extract per-window activity labels via majority vote.

    Unlike reshape_and_merge, this keeps one label per original window (N,)
    rather than splitting into sub-windows and filtering.
    """
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


# =============================================================================
# GRU Classifier Training
# =============================================================================

def train_gru_classifier(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    val_data: np.ndarray,
    val_labels: np.ndarray,
    num_classes: int,
    epochs: int = CLASSIFIER_EPOCHS,
    batch_size: int = CLASSIFIER_BATCH_SIZE,
    lr: float = CLASSIFIER_LR,
    device: torch.device = None,
    verbose: bool = False,
) -> GRUClassifier:
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
        # Train
        model.train()
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_data, training=True)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

        # Validate
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


def predict_gru(model: GRUClassifier, data: np.ndarray,
                batch_size: int = CLASSIFIER_BATCH_SIZE,
                device: torch.device = None) -> np.ndarray:
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
# Linear Classifier Training (fair comparison path)
# =============================================================================

def train_linear_classifier(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    val_data: np.ndarray,
    val_labels: np.ndarray,
    num_classes: int,
    input_dim: int = EMB_DIM,
    epochs: int = CLASSIFIER_EPOCHS,
    batch_size: int = CLASSIFIER_BATCH_SIZE,
    lr: float = CLASSIFIER_LR,
    device: torch.device = None,
    verbose: bool = False,
) -> LinearClassifier:
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


def predict_linear(model: LinearClassifier, data: np.ndarray,
                   batch_size: int = CLASSIFIER_BATCH_SIZE,
                   device: torch.device = None) -> np.ndarray:
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
# Evaluation Functions (GRU - original LiMU-BERT architecture)
# =============================================================================

def evaluate_open_set(
    train_embeddings: Dict[str, Tuple[np.ndarray, np.ndarray]],
    test_embeddings: np.ndarray,
    test_labels_raw: np.ndarray,
    test_dataset: str,
    device: torch.device,
) -> Dict[str, float]:
    """Metric 1: Zero-shot open-set evaluation.

    Train a classifier on ALL training data with ALL 87 training labels.
    Evaluate on test data using synonym groups for matching.
    """
    print("  [Open-set] Preparing training data with all labels...")
    label_to_group = get_label_to_group_mapping()
    test_activities = get_dataset_labels(test_dataset)

    # Build global label index
    global_label_to_idx = {label: i for i, label in enumerate(GLOBAL_LABELS)}
    num_global_classes = len(GLOBAL_LABELS)

    # Combine all training embeddings with global labels
    all_train_data = []
    all_train_labels = []

    for ds_name in TRAIN_DATASETS:
        if ds_name not in train_embeddings:
            continue
        emb, lab_raw = train_embeddings[ds_name]
        ds_activities = get_dataset_labels(ds_name)

        data, local_labels = reshape_and_merge(emb, lab_raw)

        # Map local labels to global labels
        for i in range(len(local_labels)):
            local_idx = local_labels[i]
            if local_idx < len(ds_activities):
                activity_name = ds_activities[local_idx]
                global_idx = global_label_to_idx.get(activity_name, -1)
                if global_idx >= 0:
                    all_train_data.append(data[i])
                    all_train_labels.append(global_idx)

    all_train_data = np.array(all_train_data)
    all_train_labels = np.array(all_train_labels, dtype=np.int64)

    print(f"  [Open-set] Training data: {len(all_train_data)} samples, {num_global_classes} classes")

    # Split into train/val
    rng = np.random.RandomState(CLASSIFIER_SEED)
    idx = np.arange(len(all_train_data))
    rng.shuffle(idx)
    all_train_data = all_train_data[idx]
    all_train_labels = all_train_labels[idx]

    val_n = int(len(all_train_data) * 0.1)
    train_data = all_train_data[val_n:]
    train_labels = all_train_labels[val_n:]
    val_data = all_train_data[:val_n]
    val_labels = all_train_labels[:val_n]

    # Train classifier
    print(f"  [Open-set] Training GRU classifier ({num_global_classes} classes)...")
    model = train_gru_classifier(
        train_data, train_labels, val_data, val_labels,
        num_classes=num_global_classes, device=device, verbose=True
    )

    # Prepare test data
    test_data, test_local_labels = reshape_and_merge(test_embeddings, test_labels_raw)

    # Get predictions
    pred_global_indices = predict_gru(model, test_data, device=device)

    # Map predictions and ground truth through synonym groups
    pred_groups = []
    gt_groups = []
    for i in range(len(test_local_labels)):
        # Ground truth: local label -> activity name -> group
        local_idx = test_local_labels[i]
        if local_idx < len(test_activities):
            gt_name = test_activities[local_idx]
        else:
            continue
        gt_group = label_to_group.get(gt_name, gt_name)

        # Prediction: global label -> activity name -> group
        pred_idx = pred_global_indices[i]
        if pred_idx < len(GLOBAL_LABELS):
            pred_name = GLOBAL_LABELS[pred_idx]
        else:
            pred_name = "unknown"
        pred_group = label_to_group.get(pred_name, pred_name)

        gt_groups.append(gt_group)
        pred_groups.append(pred_group)

    acc = accuracy_score(gt_groups, pred_groups) * 100
    f1 = f1_score(gt_groups, pred_groups, average='macro', zero_division=0) * 100

    return {
        'accuracy': acc,
        'f1_macro': f1,
        'n_samples': len(gt_groups),
        'n_train_samples': len(train_data),
        'n_classes_train': num_global_classes,
        'n_classes_test': len(test_activities),
    }


def evaluate_closed_set(
    train_embeddings: Dict[str, Tuple[np.ndarray, np.ndarray]],
    test_embeddings: np.ndarray,
    test_labels_raw: np.ndarray,
    test_dataset: str,
    device: torch.device,
) -> Dict[str, float]:
    """Metric 2: Closed-set evaluation.

    Train a classifier using training data filtered to labels matching the
    test dataset's label space. Use synonym groups to identify matching
    training samples. Evaluate with exact match (no groups).
    """
    label_to_group = get_label_to_group_mapping()
    test_activities = get_dataset_labels(test_dataset)
    test_label_groups = {label_to_group.get(a, a) for a in test_activities}

    # Map test activities to their group, and build group-to-test-label mapping
    group_to_test_label = {}
    for act in test_activities:
        group = label_to_group.get(act, act)
        group_to_test_label[group] = act

    test_label_to_idx = {a: i for i, a in enumerate(test_activities)}
    num_test_classes = len(test_activities)

    print(f"  [Closed-set] Collecting training data for {num_test_classes} test classes...")

    # Collect training data whose labels match test label groups
    all_train_data = []
    all_train_labels = []

    for ds_name in TRAIN_DATASETS:
        if ds_name not in train_embeddings:
            continue
        emb, lab_raw = train_embeddings[ds_name]
        ds_activities = get_dataset_labels(ds_name)

        data, local_labels = reshape_and_merge(emb, lab_raw)

        for i in range(len(local_labels)):
            local_idx = local_labels[i]
            if local_idx < len(ds_activities):
                activity_name = ds_activities[local_idx]
                group = label_to_group.get(activity_name, activity_name)

                if group in test_label_groups:
                    # Map to the test dataset's label for this group
                    test_label = group_to_test_label.get(group)
                    if test_label is not None:
                        test_idx = test_label_to_idx[test_label]
                        all_train_data.append(data[i])
                        all_train_labels.append(test_idx)

    all_train_data = np.array(all_train_data)
    all_train_labels = np.array(all_train_labels, dtype=np.int64)

    # Check class coverage
    covered_classes = len(np.unique(all_train_labels))
    print(f"  [Closed-set] Training data: {len(all_train_data)} samples, "
          f"{covered_classes}/{num_test_classes} classes covered")

    if len(all_train_data) == 0:
        return {'accuracy': 0.0, 'f1_macro': 0.0, 'n_samples': 0, 'n_classes': num_test_classes}

    # Split train/val
    rng = np.random.RandomState(CLASSIFIER_SEED)
    idx = np.arange(len(all_train_data))
    rng.shuffle(idx)
    all_train_data = all_train_data[idx]
    all_train_labels = all_train_labels[idx]

    val_n = int(len(all_train_data) * 0.1)
    train_data = all_train_data[val_n:]
    train_labels = all_train_labels[val_n:]
    val_data = all_train_data[:val_n]
    val_labels = all_train_labels[:val_n]

    # Train classifier
    print(f"  [Closed-set] Training GRU classifier ({num_test_classes} classes)...")
    model = train_gru_classifier(
        train_data, train_labels, val_data, val_labels,
        num_classes=num_test_classes, device=device, verbose=True
    )

    # Prepare test data
    test_data, test_local_labels = reshape_and_merge(test_embeddings, test_labels_raw)

    # Get predictions
    pred_indices = predict_gru(model, test_data, device=device)

    # Map to label names for exact-match evaluation
    gt_names = []
    pred_names = []
    for i in range(len(test_local_labels)):
        local_idx = test_local_labels[i]
        if local_idx < len(test_activities):
            gt_names.append(test_activities[local_idx])
        else:
            continue

        pred_idx = pred_indices[i]
        if pred_idx < len(test_activities):
            pred_names.append(test_activities[pred_idx])
        else:
            pred_names.append("unknown")

    acc = accuracy_score(gt_names, pred_names) * 100
    f1 = f1_score(gt_names, pred_names, average='macro', zero_division=0) * 100

    return {
        'accuracy': acc,
        'f1_macro': f1,
        'n_samples': len(gt_names),
        'n_train_samples': len(train_data),
        'n_classes': num_test_classes,
        'covered_classes': covered_classes,
    }


def evaluate_1pct_supervised(
    test_embeddings: np.ndarray,
    test_labels_raw: np.ndarray,
    test_dataset: str,
    device: torch.device,
) -> Dict[str, float]:
    """Metric 3: 1% supervised evaluation.

    Train a classifier on 1% of the test dataset. Evaluate on the rest.
    Closed-set (test dataset labels only), exact match.
    """
    test_activities = get_dataset_labels(test_dataset)
    num_test_classes = len(test_activities)

    # Prepare test data
    test_data, test_local_labels = reshape_and_merge(test_embeddings, test_labels_raw)

    print(f"  [1% supervised] Total samples: {len(test_data)}, {num_test_classes} classes")

    # Split: 80% pool (subsample 1% for training), 10% val, 10% test
    train_data, train_labels, val_data, val_labels, eval_data, eval_labels = \
        prepare_train_test_split(
            test_data, test_local_labels,
            training_rate=TRAINING_RATE,
            vali_rate=VALI_RATE,
            label_rate=SUPERVISED_LABEL_RATE,
            seed=CLASSIFIER_SEED,
            balance=True
        )

    print(f"  [1% supervised] Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(eval_data)}")

    if len(train_data) == 0:
        return {'accuracy': 0.0, 'f1_macro': 0.0, 'n_samples': 0, 'n_classes': num_test_classes}

    # Train classifier
    print(f"  [1% supervised] Training GRU classifier ({num_test_classes} classes)...")
    model = train_gru_classifier(
        train_data, train_labels, val_data, val_labels,
        num_classes=num_test_classes, device=device, verbose=True
    )

    # Evaluate on test split
    pred_indices = predict_gru(model, eval_data, device=device)

    # Map to label names
    gt_names = []
    pred_names = []
    for i in range(len(eval_labels)):
        local_idx = eval_labels[i]
        if local_idx < len(test_activities):
            gt_names.append(test_activities[local_idx])
        else:
            continue

        pred_idx = pred_indices[i]
        if pred_idx < len(test_activities):
            pred_names.append(test_activities[pred_idx])
        else:
            pred_names.append("unknown")

    acc = accuracy_score(gt_names, pred_names) * 100
    f1 = f1_score(gt_names, pred_names, average='macro', zero_division=0) * 100

    return {
        'accuracy': acc,
        'f1_macro': f1,
        'n_samples': len(gt_names),
        'n_train_samples': len(train_data),
        'n_classes': num_test_classes,
    }


# =============================================================================
# Linear Probe Evaluation Functions (fair comparison path)
# =============================================================================

def evaluate_open_set_linear(
    train_embeddings: Dict[str, Tuple[np.ndarray, np.ndarray]],
    test_embeddings: np.ndarray,
    test_labels_raw: np.ndarray,
    test_dataset: str,
    device: torch.device,
) -> Dict[str, float]:
    """Metric 1: Zero-shot open-set with linear probe on mean-pooled embeddings."""
    print("  [Open-set] Preparing training data with all labels...")
    label_to_group = get_label_to_group_mapping()
    test_activities = get_dataset_labels(test_dataset)

    global_label_to_idx = {label: i for i, label in enumerate(GLOBAL_LABELS)}
    num_global_classes = len(GLOBAL_LABELS)

    all_train_data = []
    all_train_labels = []

    for ds_name in TRAIN_DATASETS:
        if ds_name not in train_embeddings:
            continue
        emb, lab_raw = train_embeddings[ds_name]
        ds_activities = get_dataset_labels(ds_name)

        # Mean-pool to (N, 72) and get window-level labels
        pooled = mean_pool_embeddings(emb)
        window_labels = get_window_labels(lab_raw)

        for i in range(len(window_labels)):
            local_idx = window_labels[i]
            if local_idx < len(ds_activities):
                activity_name = ds_activities[local_idx]
                global_idx = global_label_to_idx.get(activity_name, -1)
                if global_idx >= 0:
                    all_train_data.append(pooled[i])
                    all_train_labels.append(global_idx)

    all_train_data = np.array(all_train_data)
    all_train_labels = np.array(all_train_labels, dtype=np.int64)

    print(f"  [Open-set] Training data: {len(all_train_data)} samples, {num_global_classes} classes")

    rng = np.random.RandomState(CLASSIFIER_SEED)
    idx = np.arange(len(all_train_data))
    rng.shuffle(idx)
    all_train_data = all_train_data[idx]
    all_train_labels = all_train_labels[idx]

    val_n = int(len(all_train_data) * 0.1)
    train_data = all_train_data[val_n:]
    train_labels = all_train_labels[val_n:]
    val_data = all_train_data[:val_n]
    val_labels = all_train_labels[:val_n]

    print(f"  [Open-set] Training linear classifier ({num_global_classes} classes)...")
    model = train_linear_classifier(
        train_data, train_labels, val_data, val_labels,
        num_classes=num_global_classes, device=device, verbose=True
    )

    # Mean-pool test embeddings
    test_pooled = mean_pool_embeddings(test_embeddings)
    test_window_labels = get_window_labels(test_labels_raw)

    pred_global_indices = predict_linear(model, test_pooled, device=device)

    pred_groups = []
    gt_groups = []
    for i in range(len(test_window_labels)):
        local_idx = test_window_labels[i]
        if local_idx < len(test_activities):
            gt_name = test_activities[local_idx]
        else:
            continue
        gt_group = label_to_group.get(gt_name, gt_name)

        pred_idx = pred_global_indices[i]
        if pred_idx < len(GLOBAL_LABELS):
            pred_name = GLOBAL_LABELS[pred_idx]
        else:
            pred_name = "unknown"
        pred_group = label_to_group.get(pred_name, pred_name)

        gt_groups.append(gt_group)
        pred_groups.append(pred_group)

    acc = accuracy_score(gt_groups, pred_groups) * 100
    f1 = f1_score(gt_groups, pred_groups, average='macro', zero_division=0) * 100

    return {
        'accuracy': acc,
        'f1_macro': f1,
        'n_samples': len(gt_groups),
        'n_train_samples': len(train_data),
        'n_classes_train': num_global_classes,
        'n_classes_test': len(test_activities),
    }


def evaluate_closed_set_linear(
    train_embeddings: Dict[str, Tuple[np.ndarray, np.ndarray]],
    test_embeddings: np.ndarray,
    test_labels_raw: np.ndarray,
    test_dataset: str,
    device: torch.device,
) -> Dict[str, float]:
    """Metric 2: Closed-set with linear probe on mean-pooled embeddings."""
    label_to_group = get_label_to_group_mapping()
    test_activities = get_dataset_labels(test_dataset)
    test_label_groups = {label_to_group.get(a, a) for a in test_activities}

    group_to_test_label = {}
    for act in test_activities:
        group = label_to_group.get(act, act)
        group_to_test_label[group] = act

    test_label_to_idx = {a: i for i, a in enumerate(test_activities)}
    num_test_classes = len(test_activities)

    print(f"  [Closed-set] Collecting training data for {num_test_classes} test classes...")

    all_train_data = []
    all_train_labels = []

    for ds_name in TRAIN_DATASETS:
        if ds_name not in train_embeddings:
            continue
        emb, lab_raw = train_embeddings[ds_name]
        ds_activities = get_dataset_labels(ds_name)

        pooled = mean_pool_embeddings(emb)
        window_labels = get_window_labels(lab_raw)

        for i in range(len(window_labels)):
            local_idx = window_labels[i]
            if local_idx < len(ds_activities):
                activity_name = ds_activities[local_idx]
                group = label_to_group.get(activity_name, activity_name)

                if group in test_label_groups:
                    test_label = group_to_test_label.get(group)
                    if test_label is not None:
                        test_idx = test_label_to_idx[test_label]
                        all_train_data.append(pooled[i])
                        all_train_labels.append(test_idx)

    all_train_data = np.array(all_train_data)
    all_train_labels = np.array(all_train_labels, dtype=np.int64)

    covered_classes = len(np.unique(all_train_labels))
    print(f"  [Closed-set] Training data: {len(all_train_data)} samples, "
          f"{covered_classes}/{num_test_classes} classes covered")

    if len(all_train_data) == 0:
        return {'accuracy': 0.0, 'f1_macro': 0.0, 'n_samples': 0, 'n_classes': num_test_classes}

    rng = np.random.RandomState(CLASSIFIER_SEED)
    idx = np.arange(len(all_train_data))
    rng.shuffle(idx)
    all_train_data = all_train_data[idx]
    all_train_labels = all_train_labels[idx]

    val_n = int(len(all_train_data) * 0.1)
    train_data = all_train_data[val_n:]
    train_labels = all_train_labels[val_n:]
    val_data = all_train_data[:val_n]
    val_labels = all_train_labels[:val_n]

    print(f"  [Closed-set] Training linear classifier ({num_test_classes} classes)...")
    model = train_linear_classifier(
        train_data, train_labels, val_data, val_labels,
        num_classes=num_test_classes, device=device, verbose=True
    )

    # Mean-pool test embeddings
    test_pooled = mean_pool_embeddings(test_embeddings)
    test_window_labels = get_window_labels(test_labels_raw)

    pred_indices = predict_linear(model, test_pooled, device=device)

    gt_names = []
    pred_names = []
    for i in range(len(test_window_labels)):
        local_idx = test_window_labels[i]
        if local_idx < len(test_activities):
            gt_names.append(test_activities[local_idx])
        else:
            continue

        pred_idx = pred_indices[i]
        if pred_idx < len(test_activities):
            pred_names.append(test_activities[pred_idx])
        else:
            pred_names.append("unknown")

    acc = accuracy_score(gt_names, pred_names) * 100
    f1 = f1_score(gt_names, pred_names, average='macro', zero_division=0) * 100

    return {
        'accuracy': acc,
        'f1_macro': f1,
        'n_samples': len(gt_names),
        'n_train_samples': len(train_data),
        'n_classes': num_test_classes,
        'covered_classes': covered_classes,
    }


def evaluate_1pct_supervised_linear(
    test_embeddings: np.ndarray,
    test_labels_raw: np.ndarray,
    test_dataset: str,
    device: torch.device,
) -> Dict[str, float]:
    """Metric 3: 1% supervised with linear probe on mean-pooled embeddings."""
    test_activities = get_dataset_labels(test_dataset)
    num_test_classes = len(test_activities)

    # Mean-pool test embeddings
    test_pooled = mean_pool_embeddings(test_embeddings)
    test_window_labels = get_window_labels(test_labels_raw)

    print(f"  [1% supervised] Total samples: {len(test_pooled)}, {num_test_classes} classes")

    # Split with 1D data (N, 72) and per-window labels
    train_data, train_labels, val_data, val_labels, eval_data, eval_labels = \
        prepare_train_test_split(
            test_pooled, test_window_labels,
            training_rate=TRAINING_RATE,
            vali_rate=VALI_RATE,
            label_rate=SUPERVISED_LABEL_RATE,
            seed=CLASSIFIER_SEED,
            balance=True
        )

    print(f"  [1% supervised] Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(eval_data)}")

    if len(train_data) == 0:
        return {'accuracy': 0.0, 'f1_macro': 0.0, 'n_samples': 0, 'n_classes': num_test_classes}

    print(f"  [1% supervised] Training linear classifier ({num_test_classes} classes)...")
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
        if pred_idx < len(test_activities):
            pred_names.append(test_activities[pred_idx])
        else:
            pred_names.append("unknown")

    acc = accuracy_score(gt_names, pred_names) * 100
    f1 = f1_score(gt_names, pred_names, average='macro', zero_division=0) * 100

    return {
        'accuracy': acc,
        'f1_macro': f1,
        'n_samples': len(gt_names),
        'n_train_samples': len(train_data),
        'n_classes': num_test_classes,
    }


# =============================================================================
# Output Formatting
# =============================================================================

def print_results_table(all_results: Dict, classifier_name: str = "GRU"):
    """Print a unified results table."""
    print()
    print("=" * 94)
    print(f"LIMU-BERT BASELINE EVALUATION RESULTS ({classifier_name} classifier)")
    print("=" * 94)

    header = (f"{'Dataset':<16}"
              f"{'Open-Set Acc':>13}{'Open-Set F1':>13}"
              f"{'Closed Acc':>13}{'Closed F1':>13}"
              f"{'1% Sup Acc':>13}{'1% Sup F1':>13}")
    print(header)
    print("-" * 94)

    for ds in TEST_DATASETS:
        if ds not in all_results:
            continue
        r = all_results[ds]
        os_acc = r.get('open_set', {}).get('accuracy', 0.0)
        os_f1 = r.get('open_set', {}).get('f1_macro', 0.0)
        cs_acc = r.get('closed_set', {}).get('accuracy', 0.0)
        cs_f1 = r.get('closed_set', {}).get('f1_macro', 0.0)
        sup_acc = r.get('1pct_supervised', {}).get('accuracy', 0.0)
        sup_f1 = r.get('1pct_supervised', {}).get('f1_macro', 0.0)
        print(f"{ds:<16}"
              f"{os_acc:>12.1f}%{os_f1:>12.1f}%"
              f"{cs_acc:>12.1f}%{cs_f1:>12.1f}%"
              f"{sup_acc:>12.1f}%{sup_f1:>12.1f}%")

    print("=" * 94)

    print()
    print("Evaluation Details:")
    print(f"  Open-set: Classifier on ALL {len(GLOBAL_LABELS)} training labels. "
          f"Group-based matching.")
    print(f"  Closed-set: Classifier on training data matching test labels. "
          f"Exact match.")
    print(f"  1% supervised: Classifier on 1% of test data. Exact match.")
    print(f"  Classifier: {classifier_name}, {CLASSIFIER_EPOCHS} epochs, "
          f"lr={CLASSIFIER_LR}")


# =============================================================================
# Main
# =============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all training embeddings (needed for open-set and closed-set)
    print("\nLoading training embeddings...")
    train_embeddings = {}
    for ds in TRAIN_DATASETS:
        try:
            emb, lab = load_embeddings(ds)
            train_embeddings[ds] = (emb, lab)
            print(f"  {ds}: {emb.shape[0]} windows")
        except FileNotFoundError as e:
            print(f"  {ds}: MISSING ({e})")

    print(f"\nLoaded {len(train_embeddings)}/{len(TRAIN_DATASETS)} training datasets")

    # =====================================================================
    # Evaluation 1: Linear probe (fair comparison with other baselines)
    # Uses mean-pooled (N, 72) embeddings + LinearClassifier
    # =====================================================================
    print("\n" + "=" * 70)
    print("LINEAR PROBE EVALUATION (fair comparison)")
    print("=" * 70)

    linear_results = {}

    for test_ds in TEST_DATASETS:
        print(f"\n{'='*60}")
        print(f"Evaluating LiMU-BERT (linear probe) on {test_ds}")
        print(f"{'='*60}")

        test_emb, test_lab = load_embeddings(test_ds)
        test_activities = get_dataset_labels(test_ds)
        print(f"  Test data: {test_emb.shape[0]} windows, "
              f"{len(test_activities)} classes")
        print(f"  Classes: {test_activities}")

        ds_results = {}

        print(f"\n  --- Metric 1: Zero-shot Open-Set ---")
        ds_results['open_set'] = evaluate_open_set_linear(
            train_embeddings, test_emb, test_lab, test_ds, device)
        print(f"  Open-set: Acc={ds_results['open_set']['accuracy']:.1f}%, "
              f"F1={ds_results['open_set']['f1_macro']:.1f}%")

        print(f"\n  --- Metric 2: Closed-Set ---")
        ds_results['closed_set'] = evaluate_closed_set_linear(
            train_embeddings, test_emb, test_lab, test_ds, device)
        print(f"  Closed-set: Acc={ds_results['closed_set']['accuracy']:.1f}%, "
              f"F1={ds_results['closed_set']['f1_macro']:.1f}%")

        print(f"\n  --- Metric 3: 1% Supervised ---")
        ds_results['1pct_supervised'] = evaluate_1pct_supervised_linear(
            test_emb, test_lab, test_ds, device)
        print(f"  1% supervised: "
              f"Acc={ds_results['1pct_supervised']['accuracy']:.1f}%, "
              f"F1={ds_results['1pct_supervised']['f1_macro']:.1f}%")

        linear_results[test_ds] = ds_results

    print_results_table(linear_results, classifier_name="Linear")

    # Save linear probe results
    results_path = OUTPUT_DIR / "limubert_linear_evaluation.json"
    save_data = {}
    for ds, metrics in linear_results.items():
        save_data[ds] = {}
        for metric_name, metric_vals in metrics.items():
            save_data[ds][metric_name] = {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in metric_vals.items()
            }
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nLinear probe results saved to {results_path}")

    # =====================================================================
    # Evaluation 2: GRU classifier (original LiMU-BERT architecture)
    # Uses sub-windows (M, 20, 72) + GRUClassifier
    # =====================================================================
    print("\n" + "=" * 70)
    print("GRU EVALUATION (original LiMU-BERT architecture)")
    print("=" * 70)

    gru_results = {}

    for test_ds in TEST_DATASETS:
        print(f"\n{'='*60}")
        print(f"Evaluating LiMU-BERT (GRU) on {test_ds}")
        print(f"{'='*60}")

        test_emb, test_lab = load_embeddings(test_ds)
        test_activities = get_dataset_labels(test_ds)
        print(f"  Test data: {test_emb.shape[0]} windows, "
              f"{len(test_activities)} classes")
        print(f"  Classes: {test_activities}")

        ds_results = {}

        print(f"\n  --- Metric 1: Zero-shot Open-Set ---")
        ds_results['open_set'] = evaluate_open_set(
            train_embeddings, test_emb, test_lab, test_ds, device)
        print(f"  Open-set: Acc={ds_results['open_set']['accuracy']:.1f}%, "
              f"F1={ds_results['open_set']['f1_macro']:.1f}%")

        print(f"\n  --- Metric 2: Closed-Set ---")
        ds_results['closed_set'] = evaluate_closed_set(
            train_embeddings, test_emb, test_lab, test_ds, device)
        print(f"  Closed-set: Acc={ds_results['closed_set']['accuracy']:.1f}%, "
              f"F1={ds_results['closed_set']['f1_macro']:.1f}%")

        print(f"\n  --- Metric 3: 1% Supervised ---")
        ds_results['1pct_supervised'] = evaluate_1pct_supervised(
            test_emb, test_lab, test_ds, device)
        print(f"  1% supervised: "
              f"Acc={ds_results['1pct_supervised']['accuracy']:.1f}%, "
              f"F1={ds_results['1pct_supervised']['f1_macro']:.1f}%")

        gru_results[test_ds] = ds_results

    print_results_table(gru_results, classifier_name="GRU (original)")

    # Save GRU results
    results_path = OUTPUT_DIR / "limubert_gru_evaluation.json"
    save_data = {}
    for ds, metrics in gru_results.items():
        save_data[ds] = {}
        for metric_name, metric_vals in metrics.items():
            save_data[ds][metric_name] = {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in metric_vals.items()
            }
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nGRU results saved to {results_path}")


if __name__ == '__main__':
    main()
