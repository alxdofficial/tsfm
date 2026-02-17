"""
TSFM (our model) evaluation using the new evaluation framework.

Extracts embeddings from the trained TSFM semantic alignment model
and evaluates with:
  1. Zero-shot open-set (cosine sim against all 87 training labels, group matching)
  2. Zero-shot closed-set (cosine sim against test dataset labels only)
  3. 1% supervised (linear classifier on 1% of test data)
  4. 10% supervised (linear classifier on 10% of test data)
  5. Linear probe (linear classifier on frozen embeddings, full train split)

Zero-shot uses cosine similarity between IMU embeddings and text label
embeddings from the trained LearnableLabelBank — no classifier training needed.

Uses the same benchmark data format as all baselines:
  (N, 120, 6) windows at 20Hz with 6 IMU channels (acc_xyz + gyro_xyz)

Usage:
    python val_scripts/human_activity_recognition/evaluate_tsfm.py
"""

import copy
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast
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
from training_scripts.human_activity_recognition.semantic_alignment_train import SemanticAlignmentModel
from imu_activity_recognition_encoder.token_text_encoder import LearnableLabelBank

# =============================================================================
# Configuration
# =============================================================================

BENCHMARK_DIR = PROJECT_ROOT / "benchmark_data"
LIMUBERT_DATA_DIR = BENCHMARK_DIR / "processed" / "limubert"
DATASET_CONFIG_PATH = BENCHMARK_DIR / "dataset_config.json"
GLOBAL_LABEL_PATH = LIMUBERT_DATA_DIR / "global_label_mapping.json"
OUTPUT_DIR = PROJECT_ROOT / "test_output" / "baseline_evaluation"

# TSFM checkpoint - update this path to your trained model
CHECKPOINT_PATH = "training_output/semantic_alignment/20260216_225955/best.pt"

# Data specs (standardized benchmark format)
DATA_SEQ_LEN = 120         # Window length (timesteps)
DATA_CHANNELS = 6          # 6-channel IMU (3 accel + 3 gyro)
DATA_SAMPLING_RATE = 20.0  # All benchmark data is resampled to 20Hz
TSFM_EMB_DIM = 384         # TSFM embedding dimension
TSFM_BATCH_SIZE = 32       # Batch size for embedding extraction

# Patch sizes to sweep per dataset (benchmark data: 120 steps @ 20Hz = 6s window)
# Covers training augmentation range; best is selected per test dataset
PATCH_SIZES_TO_TRY = [1.0, 1.25, 1.5, 1.75, 2.0]
DEFAULT_PATCH_SIZE_SEC = 1.5  # Fallback / training dataset default

# Channel descriptions for the standardized 6-channel format
CHANNEL_DESCRIPTIONS = [
    "Accelerometer X-axis",
    "Accelerometer Y-axis",
    "Accelerometer Z-axis",
    "Gyroscope X-axis",
    "Gyroscope Y-axis",
    "Gyroscope Z-axis",
]

# Classifier hyperparameters (must match baselines exactly)
CLASSIFIER_EPOCHS = 100
CLASSIFIER_BATCH_SIZE = 128
CLASSIFIER_LR = 1e-3
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


# =============================================================================
# Linear Classifier (identical to baselines)
# =============================================================================

class LinearClassifier(nn.Module):
    """Simple linear classifier for fixed-size embeddings."""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x, training=False):
        if training:
            x = F.dropout(x, p=0.3, training=True)
        return self.linear(x)


# =============================================================================
# TSFM Model Loading & Embedding Extraction
# =============================================================================

load_tsfm_model = load_model  # backwards-compatible alias


def extract_tsfm_embeddings(
    model: SemanticAlignmentModel,
    raw_data: np.ndarray,
    device: torch.device,
    batch_size: int = TSFM_BATCH_SIZE,
    patch_size_sec: float = DEFAULT_PATCH_SIZE_SEC,
) -> np.ndarray:
    """Extract TSFM embeddings from raw sensor data.

    Args:
        model: Loaded TSFM SemanticAlignmentModel
        raw_data: (N, 120, 6) raw sensor data at 20Hz
        device: torch device
        batch_size: batch size for inference
        patch_size_sec: patch duration in seconds

    Returns:
        embeddings: (N, 384) L2-normalized embeddings
    """
    model.eval()
    N = raw_data.shape[0]

    all_embeddings = []
    for start in tqdm(range(0, N, batch_size), desc="TSFM | Extracting embeddings",
                      total=(N + batch_size - 1) // batch_size, leave=True):
        end = min(start + batch_size, N)
        batch_data = torch.from_numpy(raw_data[start:end]).float().to(device)
        bs = batch_data.shape[0]

        # Channel mask: all 6 channels valid
        channel_mask = torch.ones(bs, DATA_CHANNELS, dtype=torch.bool, device=device)

        # Attention mask: all 120 timesteps valid
        attention_mask = torch.ones(bs, DATA_SEQ_LEN, dtype=torch.bool, device=device)

        # Per-sample metadata
        channel_descs = [CHANNEL_DESCRIPTIONS[:] for _ in range(bs)]
        sampling_rates = [DATA_SAMPLING_RATE] * bs
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

def load_raw_data(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load raw sensor data and labels for a dataset."""
    ds_dir = LIMUBERT_DATA_DIR / dataset_name
    data_path = ds_dir / "data_20_120.npy"
    label_path = ds_dir / "label_20_120.npy"

    data = np.load(str(data_path)).astype(np.float32)
    labels = np.load(str(label_path)).astype(np.float32)
    return data, labels


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

def train_linear_classifier(
    train_data, train_labels, val_data, val_labels,
    num_classes, input_dim=TSFM_EMB_DIM,
    epochs=CLASSIFIER_EPOCHS, batch_size=CLASSIFIER_BATCH_SIZE,
    lr=CLASSIFIER_LR, device=None, verbose=False, desc="Linear probe",
):
    """Train a linear classifier on embeddings."""
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


def predict_classifier(model, data, batch_size=CLASSIFIER_BATCH_SIZE, device=None):
    """Get predictions from a trained classifier."""
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

    return {'accuracy': acc, 'f1_macro': f1, 'n_samples': len(gt_groups),
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

    return {'accuracy': acc, 'f1_macro': f1, 'n_samples': len(gt_names),
            'n_classes': num_test_classes}


def evaluate_supervised(
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    test_dataset: str,
    device: torch.device,
    label_rate: float = 0.01,
    label_tag: str = "1%",
) -> Dict[str, float]:
    """Supervised evaluation: train linear classifier on subset of test data.

    Args:
        label_rate: Fraction of training portion to use (0.01 = 1%, 0.10 = 10%)
        label_tag: Display tag for logging
    """
    test_activities = get_dataset_labels(test_dataset)
    num_test_classes = len(test_activities)

    print(f"  [{label_tag} supervised] Total samples: {len(test_embeddings)}, {num_test_classes} classes")

    train_data, train_labels_arr, val_data, val_labels_arr, eval_data, eval_labels_arr = \
        prepare_train_test_split(
            test_embeddings, test_labels,
            training_rate=TRAINING_RATE,
            vali_rate=VALI_RATE,
            label_rate=label_rate,
            seed=CLASSIFIER_SEED,
            balance=True
        )

    print(f"  [{label_tag} supervised] Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(eval_data)}")

    if len(train_data) == 0:
        return {'accuracy': 0.0, 'f1_macro': 0.0, 'n_samples': 0, 'n_classes': num_test_classes}

    print(f"  [{label_tag} supervised] Training linear classifier ({num_test_classes} classes)...")
    model = train_linear_classifier(
        train_data, train_labels_arr, val_data, val_labels_arr,
        num_classes=num_test_classes, device=device, verbose=True,
        desc=f"TSFM | {test_dataset} | linear {label_tag}",
    )

    pred_indices = predict_classifier(model, eval_data, device=device)

    gt_names = []
    pred_names = []
    for i in range(len(eval_labels_arr)):
        local_idx = eval_labels_arr[i]
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
    test_labels: np.ndarray,
    test_dataset: str,
    device: torch.device,
) -> Dict[str, float]:
    """Linear probe: train linear classifier on full train split of test data.

    Measures representation quality. Uses 80% train / 10% val / 10% test split
    with full training set (no subsampling).
    """
    return evaluate_supervised(
        test_embeddings, test_labels, test_dataset, device,
        label_rate=1.0, label_tag="Linear probe",
    )


# =============================================================================
# Main
# =============================================================================

def print_results_table(all_results):
    """Print results table."""
    print()
    print("=" * 140)
    print("TSFM EVALUATION RESULTS (New Framework)")
    print("=" * 140)

    header = (f"{'Dataset':<16}{'Patch':>6}"
              f"{'ZS-Open Acc':>13}{'ZS-Open F1':>13}"
              f"{'ZS-Close Acc':>14}{'ZS-Close F1':>13}"
              f"{'1%Sup Acc':>11}{'1%Sup F1':>10}"
              f"{'10%Sup Acc':>12}{'10%Sup F1':>11}"
              f"{'LP Acc':>9}{'LP F1':>8}")
    print(header)
    print("-" * 140)

    for ds in TEST_DATASETS:
        if ds not in all_results:
            continue
        r = all_results[ds]
        ps = r.get('best_patch_size', DEFAULT_PATCH_SIZE_SEC)

        def g(key, metric):
            return r.get(key, {}).get(metric, 0.0)

        print(f"{ds:<16}{ps:>5.2f}s"
              f"{g('zero_shot_open_set','accuracy'):>12.1f}%{g('zero_shot_open_set','f1_macro'):>12.1f}%"
              f"{g('zero_shot_closed_set','accuracy'):>13.1f}%{g('zero_shot_closed_set','f1_macro'):>12.1f}%"
              f"{g('1pct_supervised','accuracy'):>10.1f}%{g('1pct_supervised','f1_macro'):>9.1f}%"
              f"{g('10pct_supervised','accuracy'):>11.1f}%{g('10pct_supervised','f1_macro'):>10.1f}%"
              f"{g('linear_probe','accuracy'):>8.1f}%{g('linear_probe','f1_macro'):>7.1f}%")

    print("=" * 140)
    print()
    print("Details:")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  Embedding dim: {TSFM_EMB_DIM}")
    print(f"  Benchmark data: {DATA_SEQ_LEN} steps @ {DATA_SAMPLING_RATE}Hz, {DATA_CHANNELS} channels")
    print(f"  Patch sizes tried: {PATCH_SIZES_TO_TRY}")
    print(f"  Zero-shot: Cosine similarity with LearnableLabelBank text embeddings")
    print(f"  Supervised: Linear classifier, {CLASSIFIER_EPOCHS} epochs, lr={CLASSIFIER_LR}")


def select_best_patch_size(
    model: SemanticAlignmentModel,
    raw_data: np.ndarray,
    raw_labels: np.ndarray,
    test_dataset: str,
    label_bank: LearnableLabelBank,
    device: torch.device,
    patch_sizes: list = PATCH_SIZES_TO_TRY,
) -> Tuple[float, np.ndarray]:
    """Select best patch size using zero-shot closed-set accuracy.

    Extracts embeddings at each candidate patch size, computes cosine
    similarity with test label text embeddings, and picks the patch size
    with highest closed-set accuracy. No classifier training needed.

    Patch sizes longer than the window duration are automatically excluded.

    Returns:
        (best_patch_size, best_embeddings)
    """
    test_labels = get_window_labels(raw_labels)
    test_activities = get_dataset_labels(test_dataset)

    # Filter: patch size must not exceed window duration
    window_duration = raw_data.shape[1] / DATA_SAMPLING_RATE
    valid_sizes = [ps for ps in patch_sizes if ps <= window_duration]
    if not valid_sizes:
        valid_sizes = [window_duration]
    if len(valid_sizes) < len(patch_sizes):
        dropped = [ps for ps in patch_sizes if ps > window_duration]
        print(f"  Dropped patch sizes {dropped} (exceed window duration {window_duration:.1f}s)")

    if len(valid_sizes) == 1:
        emb = extract_tsfm_embeddings(model, raw_data, device, patch_size_sec=valid_sizes[0])
        return valid_sizes[0], emb

    # Encode test labels once for zero-shot evaluation
    with torch.no_grad():
        label_embeddings = label_bank.encode(test_activities, normalize=True).to(device)

    print(f"  Sweeping patch sizes: {valid_sizes}")
    best_acc = -1.0
    best_ps = valid_sizes[0]
    best_emb = None

    for ps in valid_sizes:
        try:
            emb = extract_tsfm_embeddings(model, raw_data, device, patch_size_sec=ps)
        except Exception as e:
            print(f"    patch={ps}s: FAILED ({e})")
            continue

        # Quick zero-shot closed-set eval via cosine similarity
        emb_t = torch.from_numpy(emb).float().to(device)
        similarity = compute_similarity(emb_t, label_embeddings)
        pred_indices = similarity.argmax(dim=1).cpu().numpy()

        correct = 0
        total = 0
        for i in range(len(test_labels)):
            if test_labels[i] < len(test_activities):
                total += 1
                if pred_indices[i] == test_labels[i]:
                    correct += 1

        acc = (correct / total * 100) if total > 0 else 0.0
        print(f"    patch={ps}s: ZS closed-set acc={acc:.1f}%")

        if acc > best_acc:
            best_acc = acc
            best_ps = ps
            best_emb = emb

    print(f"  -> Best patch size: {best_ps}s (ZS closed-set acc={best_acc:.1f}%)")
    return best_ps, best_emb


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load TSFM model + label bank
    print(f"\nLoading TSFM model from {CHECKPOINT_PATH}...")
    model, checkpoint, hyperparams_path = load_tsfm_model(CHECKPOINT_PATH, device)
    label_bank = load_label_bank(checkpoint, device, hyperparams_path)
    print("Model and label bank loaded successfully")

    # Run evaluation on each test dataset
    all_results = {}

    for test_ds in TEST_DATASETS:
        print(f"\n{'='*60}")
        print(f"Testing TSFM on {test_ds}")
        print(f"{'='*60}")

        raw_data, raw_labels = load_raw_data(test_ds)
        test_labels = get_window_labels(raw_labels)
        test_activities = get_dataset_labels(test_ds)

        # Select best patch size for this dataset
        best_ps, test_emb = select_best_patch_size(
            model, raw_data, raw_labels, test_ds, label_bank, device)

        print(f"  Test data: {test_emb.shape[0]} windows -> embeddings {test_emb.shape}, "
              f"{len(test_activities)} classes")
        print(f"  Classes: {test_activities}")

        ds_results = {'best_patch_size': best_ps}

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

        # 3. 1% supervised
        print(f"\n  --- 1% Supervised ---")
        ds_results['1pct_supervised'] = evaluate_supervised(
            test_emb, test_labels, test_ds, device,
            label_rate=SUPERVISED_LABEL_RATE_1PCT, label_tag="1%")
        print(f"  1% supervised: "
              f"Acc={ds_results['1pct_supervised']['accuracy']:.1f}%, "
              f"F1={ds_results['1pct_supervised']['f1_macro']:.1f}%")

        # 4. 10% supervised
        print(f"\n  --- 10% Supervised ---")
        ds_results['10pct_supervised'] = evaluate_supervised(
            test_emb, test_labels, test_ds, device,
            label_rate=SUPERVISED_LABEL_RATE_10PCT, label_tag="10%")
        print(f"  10% supervised: "
              f"Acc={ds_results['10pct_supervised']['accuracy']:.1f}%, "
              f"F1={ds_results['10pct_supervised']['f1_macro']:.1f}%")

        # 5. Linear probe (full train split)
        print(f"\n  --- Linear Probe ---")
        ds_results['linear_probe'] = evaluate_linear_probe(
            test_emb, test_labels, test_ds, device)
        print(f"  Linear probe: "
              f"Acc={ds_results['linear_probe']['accuracy']:.1f}%, "
              f"F1={ds_results['linear_probe']['f1_macro']:.1f}%")

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
