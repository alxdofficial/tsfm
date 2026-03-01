"""
MOMENT baseline scoring using the unified framework.

Evaluates with:
  1. Zero-shot open-set (classifier on 87 training labels, group scoring)
  2. Zero-shot closed-set (classifier with masked logits, group scoring)
  3. 1% supervised end-to-end fine-tuning with linear head (paper's supervised protocol)
  4. 10% supervised end-to-end fine-tuning with linear head (paper's supervised protocol)

Following the MOMENT paper (Goswami et al., ICML 2024), we use an SVM
classifier with RBF kernel for zero-shot scoring. Supervised metrics use
end-to-end fine-tuning of the MOMENT encoder with a linear classification
head, using separate learning rates for the encoder and head.

Usage:
    python val_scripts/human_activity_recognition/evaluate_moment.py
"""

import json
import joblib
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from val_scripts.human_activity_recognition.grouped_zero_shot import (
    load_global_labels, map_local_to_global_labels,
    get_closed_set_mask, score_with_groups,
)

# =============================================================================
# Configuration
# =============================================================================

BENCHMARK_DIR = PROJECT_ROOT / "benchmark_data"
LIMUBERT_DATA_DIR = BENCHMARK_DIR / "processed" / "limubert"
DATASET_CONFIG_PATH = BENCHMARK_DIR / "dataset_config.json"
OUTPUT_DIR = PROJECT_ROOT / "test_output" / "baseline_evaluation"

# MOMENT settings
MOMENT_MODEL_NAME = "AutonLab/MOMENT-1-large"
MOMENT_SEQ_LEN = 512       # MOMENT expects 512 timesteps
DATA_SEQ_LEN = 120         # Our data window length
DATA_CHANNELS = 6          # 6-channel IMU
MOMENT_EMB_DIM_PER_CHANNEL = 1024  # MOMENT-1-large per-channel embedding dim
MOMENT_EMB_DIM = DATA_CHANNELS * MOMENT_EMB_DIM_PER_CHANNEL  # 6144 (concat per-channel)
MOMENT_BATCH_SIZE = 128    # Batch size for embedding extraction (MOMENT-large fits ~128 on 24GB)

# SVM hyperparameters (matching MOMENT paper's fit_svm protocol)
SVM_C_VALUES = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
SVM_MAX_SAMPLES = 10000  # Subsample if training set exceeds this
CLASSIFIER_SEED = 3431

# Fine-tuning hyperparameters (encoder + linear head, paper's supervised protocol)
FINETUNE_EPOCHS = 20
FINETUNE_BATCH_SIZE = 32
FINETUNE_ENCODER_LR = 1e-5
FINETUNE_HEAD_LR = 1e-3
FINETUNE_WEIGHT_DECAY = 1e-5
FINETUNE_PATIENCE = 5

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
# Linear Classification Head (for end-to-end fine-tuning)
# =============================================================================

class LinearHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


# =============================================================================
# MOMENT Embedding Extraction
# =============================================================================

def load_moment_model(device: torch.device):
    """Load the pretrained MOMENT model in embedding mode."""
    from momentfm import MOMENTPipeline

    model = MOMENTPipeline.from_pretrained(
        MOMENT_MODEL_NAME,
        model_kwargs={"task_name": "embedding"},
    )
    model.init()
    model.to(device)
    return model


def extract_moment_embeddings(
    model,
    raw_data: np.ndarray,
    device: torch.device,
    batch_size: int = MOMENT_BATCH_SIZE,
) -> np.ndarray:
    """Extract MOMENT embeddings from raw sensor data.

    Following MOMENT's multivariate evaluation protocol: each channel is
    processed independently as a univariate series, producing per-channel
    1024-dim embeddings that are concatenated into (N, 6*1024) = (N, 6144).

    Returns:
        embeddings: (N, 6144) per-channel concatenated embeddings
    """
    model.eval()
    N = raw_data.shape[0]

    # Reshape: (N, 120, 6) -> (N, 6, 120) (channels first)
    data_channels_first = raw_data.transpose(0, 2, 1)  # (N, 6, 120)

    # Left-pad timesteps: (N, 6, 120) -> (N, 6, 512)
    # Paper specifies left-zero-padding for series shorter than 512
    padded = np.zeros((N, DATA_CHANNELS, MOMENT_SEQ_LEN), dtype=np.float32)
    padded[:, :, -DATA_SEQ_LEN:] = data_channels_first

    # Input mask: 1 for real data (right side), 0 for padding (left side)
    input_mask = np.zeros((N, MOMENT_SEQ_LEN), dtype=np.float32)
    input_mask[:, -DATA_SEQ_LEN:] = 1.0

    # Per-channel embedding extraction (matches MOMENT multivariate protocol):
    # Each channel is processed as an independent univariate series (B, 1, 512),
    # then per-channel 1024-dim embeddings are concatenated to (B, 6144).
    all_embeddings = []
    for start in tqdm(range(0, N, batch_size), desc="MOMENT | Extracting embeddings",
                      total=(N + batch_size - 1) // batch_size, leave=True):
        end = min(start + batch_size, N)
        B = end - start

        # Reshape (B, 6, 512) -> (B*6, 1, 512): each channel as separate sample
        batch_multi = torch.from_numpy(padded[start:end]).float()  # (B, 6, 512)
        batch_data = batch_multi.reshape(B * DATA_CHANNELS, 1, MOMENT_SEQ_LEN).to(device)

        # Expand mask: (B, 512) -> (B*6, 512)
        batch_mask_raw = torch.from_numpy(input_mask[start:end]).float()  # (B, 512)
        batch_mask = batch_mask_raw.unsqueeze(1).expand(
            B, DATA_CHANNELS, MOMENT_SEQ_LEN
        ).reshape(B * DATA_CHANNELS, MOMENT_SEQ_LEN).to(device)

        with torch.no_grad():
            output = model(x_enc=batch_data, input_mask=batch_mask)
            emb = output.embeddings.cpu().numpy()  # (B*6, 1024)

        # Reshape and concatenate channels: (B*6, 1024) -> (B, 6*1024)
        emb = emb.reshape(B, DATA_CHANNELS * MOMENT_EMB_DIM_PER_CHANNEL)
        all_embeddings.append(emb)

    return np.concatenate(all_embeddings, axis=0)


# =============================================================================
# Data Loading
# =============================================================================

def load_raw_data(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load raw sensor data and labels for a dataset."""
    ds_dir = LIMUBERT_DATA_DIR / dataset_name
    data = np.load(str(ds_dir / "data_20_120.npy")).astype(np.float32)
    labels = np.load(str(ds_dir / "label_20_120.npy")).astype(np.float32)
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
# Data Splitting
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
# SVM Classifier Training (matching MOMENT paper: fit_svm with RBF kernel)
# =============================================================================

def train_svm_classifier(train_data, train_labels, verbose=False):
    """Train SVM-RBF classifier matching MOMENT paper's protocol."""
    nb_classes = len(np.unique(train_labels))
    train_size = len(train_data)

    svm = SVC(C=100000, gamma="scale")

    # Small dataset fallback (matches MOMENT paper)
    if train_size // max(nb_classes, 1) < 5 or train_size < 50:
        if verbose:
            print(f"    Training SVM-RBF (small dataset, n={train_size}, no CV)...")
        return svm.fit(train_data, train_labels)

    if verbose:
        print(f"    Training SVM-RBF with GridSearchCV (n={train_size}, "
              f"{nb_classes} classes, 5-fold CV, 9 C values)..."
              f"\n    (SVM has no epoch progress bar — watch for completion)")

    grid_search = GridSearchCV(
        svm,
        {"C": SVM_C_VALUES, "kernel": ["rbf"], "gamma": ["scale"],
         "max_iter": [10000000]},
        cv=5, n_jobs=-1,
    )

    # Subsample if too large (matches MOMENT paper)
    if train_size > SVM_MAX_SAMPLES:
        if verbose:
            print(f"    Subsampling {SVM_MAX_SAMPLES}/{train_size} for GridSearchCV...")
        _, class_counts = np.unique(train_labels, return_counts=True)
        stratify = train_labels if class_counts.min() >= 2 else None
        train_data, _, train_labels, _ = train_test_split(
            train_data, train_labels, train_size=SVM_MAX_SAMPLES,
            random_state=0, stratify=stratify
        )

    grid_search.fit(train_data, train_labels)
    if verbose:
        print(f"    Best C={grid_search.best_params_['C']}")
    return grid_search.best_estimator_


def predict_svm_global(svm_model, data, logit_mask=None):
    """Predict global label indices using the SVM classifier.

    Args:
        svm_model: Trained SVC model (classes are global label indices)
        data: (N, 6144) embedding vectors
        logit_mask: Optional (87,) bool mask for closed-set. If provided,
            uses decision_function scores masked to allowed classes.

    Returns:
        pred_global_indices: (N,) predicted global label indices
    """
    if logit_mask is None:
        # Open-set: simple predict
        return svm_model.predict(data).astype(np.int64)

    # Closed-set: use decision_function to get per-class scores,
    # mask disallowed classes, then argmax
    scores = svm_model.decision_function(data)  # (N, n_svm_classes)
    svm_classes = svm_model.classes_  # global label indices the SVM knows

    # For binary classification, decision_function returns (N,) not (N,2)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)

    # Build mask over SVM's class columns (not all 87)
    # svm_classes[j] is the global label index for column j
    col_mask = np.array([logit_mask[c] for c in svm_classes], dtype=bool)

    # Mask disallowed columns to -inf
    scores[:, ~col_mask] = -np.inf

    # Argmax over SVM columns, map back through svm_classes
    best_col = np.argmax(scores, axis=1)
    pred_global = svm_classes[best_col].astype(np.int64)
    return pred_global


# =============================================================================
# End-to-End Fine-Tuning Helpers
# =============================================================================

def _forward_moment_batch(model, batch_data, input_mask, device):
    """Differentiable MOMENT forward pass for a batch.

    Args:
        model: MOMENT pipeline model
        batch_data: (B, 6, 512) padded data tensor on device
        input_mask: (B, 512) mask tensor on device

    Returns:
        (B, 6144) concatenated per-channel embeddings
    """
    B = batch_data.shape[0]
    # Reshape (B, 6, 512) -> (B*6, 1, 512)
    batch_flat = batch_data.reshape(B * DATA_CHANNELS, 1, MOMENT_SEQ_LEN)
    # Expand mask: (B, 512) -> (B*6, 512)
    mask_flat = input_mask.unsqueeze(1).expand(
        B, DATA_CHANNELS, MOMENT_SEQ_LEN
    ).reshape(B * DATA_CHANNELS, MOMENT_SEQ_LEN)
    # Forward through MOMENT (no torch.no_grad — gradients flow!)
    output = model(x_enc=batch_flat, input_mask=mask_flat)
    emb = output.embeddings  # (B*6, 1024)
    # Reshape to (B, 6*1024)
    return emb.reshape(B, DATA_CHANNELS * MOMENT_EMB_DIM_PER_CHANNEL)


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_supervised_finetune(
    moment_model,
    raw_data: np.ndarray,
    test_labels: np.ndarray,
    test_dataset: str,
    device: torch.device,
    label_rate: float = 0.01,
    label_tag: str = "1%",
) -> Dict[str, float]:
    """End-to-end fine-tuning: MOMENT encoder + linear head.

    Fine-tunes the full MOMENT encoder with a linear classification head using
    separate learning rates (encoder at FINETUNE_ENCODER_LR, head at
    FINETUNE_HEAD_LR). Early stopping monitors validation accuracy.

    Args:
        moment_model: Pretrained MOMENT pipeline model
        raw_data: (N, 120, 6) raw sensor windows
        test_labels: (N,) integer class labels
        test_dataset: Dataset name for label lookup
        device: Cuda/CPU device
        label_rate: Fraction of training portion to use (0.01 = 1%, 0.10 = 10%)
        label_tag: Display tag for logging
    """
    test_activities = get_dataset_labels(test_dataset)
    num_test_classes = len(test_activities)
    N = raw_data.shape[0]

    print(f"  [{label_tag} supervised fine-tune] Total samples: {N}, {num_test_classes} classes")

    # ------------------------------------------------------------------
    # 1. Preprocess all data once: left-pad to 512, build input mask
    # ------------------------------------------------------------------
    data_cf = raw_data.transpose(0, 2, 1)  # (N, 6, 120)
    padded = np.zeros((N, DATA_CHANNELS, MOMENT_SEQ_LEN), dtype=np.float32)
    padded[:, :, -DATA_SEQ_LEN:] = data_cf
    input_mask = np.zeros((N, MOMENT_SEQ_LEN), dtype=np.float32)
    input_mask[:, -DATA_SEQ_LEN:] = 1.0

    # ------------------------------------------------------------------
    # 2. Split into train / val / test
    # ------------------------------------------------------------------
    train_pad, train_labels_arr, val_pad, val_labels_arr, test_pad, test_labels_arr = \
        prepare_train_test_split(
            padded, test_labels,
            training_rate=TRAINING_RATE,
            vali_rate=VALI_RATE,
            label_rate=label_rate,
            seed=CLASSIFIER_SEED,
            balance=True,
        )
    # We also need the corresponding mask splits.  prepare_train_test_split
    # shuffles indices deterministically, so we replicate the shuffle to get
    # aligned masks.
    _, _, val_mask, _, test_mask, _ = prepare_train_test_split(
        input_mask, test_labels,
        training_rate=TRAINING_RATE,
        vali_rate=VALI_RATE,
        label_rate=label_rate,
        seed=CLASSIFIER_SEED,
        balance=True,
    )
    # For train, label_rate subsampling changes the set, so we derive
    # train_mask from train_pad shape (mask is identical across samples
    # since all windows have the same length).
    train_mask = np.zeros((len(train_pad), MOMENT_SEQ_LEN), dtype=np.float32)
    train_mask[:, -DATA_SEQ_LEN:] = 1.0

    print(f"  [{label_tag} supervised fine-tune] Train: {len(train_pad)}, "
          f"Val: {len(val_pad)}, Test: {len(test_pad)}")

    if len(train_pad) == 0:
        return {'accuracy': 0.0, 'f1_macro': 0.0, 'n_samples': 0,
                'n_classes': num_test_classes}

    # ------------------------------------------------------------------
    # 3. Build data loaders
    # ------------------------------------------------------------------
    train_ds = TensorDataset(
        torch.from_numpy(train_pad).float(),
        torch.from_numpy(train_mask).float(),
        torch.from_numpy(train_labels_arr).long(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(val_pad).float(),
        torch.from_numpy(val_mask).float(),
        torch.from_numpy(val_labels_arr).long(),
    )
    test_ds = TensorDataset(
        torch.from_numpy(test_pad).float(),
        torch.from_numpy(test_mask).float(),
        torch.from_numpy(test_labels_arr).long(),
    )
    train_loader = DataLoader(train_ds, batch_size=FINETUNE_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=FINETUNE_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=FINETUNE_BATCH_SIZE, shuffle=False)

    # ------------------------------------------------------------------
    # 4. Save a copy of original encoder weights for restoration later
    # ------------------------------------------------------------------
    original_state = copy.deepcopy(moment_model.state_dict())

    # ------------------------------------------------------------------
    # 5. Set up linear head and optimizer with two param groups
    # ------------------------------------------------------------------
    head = LinearHead(input_dim=MOMENT_EMB_DIM, num_classes=num_test_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [
            {"params": moment_model.parameters(), "lr": FINETUNE_ENCODER_LR},
            {"params": head.parameters(), "lr": FINETUNE_HEAD_LR},
        ],
        weight_decay=FINETUNE_WEIGHT_DECAY,
    )

    # ------------------------------------------------------------------
    # 6. Training loop with early stopping
    # ------------------------------------------------------------------
    best_val_acc = 0.0
    best_encoder_state = None
    best_head_state = None
    patience_counter = 0

    print(f"  [{label_tag} supervised fine-tune] Fine-tuning MOMENT encoder + head "
          f"({num_test_classes} classes, {FINETUNE_EPOCHS} epochs, "
          f"patience={FINETUNE_PATIENCE})...")

    pbar = tqdm(range(FINETUNE_EPOCHS),
                desc=f"MOMENT | {test_dataset} | {label_tag} fine-tune", leave=True)
    for epoch in pbar:
        # --- Train ---
        moment_model.train()
        head.train()
        for batch_pad, batch_mask, batch_labels in train_loader:
            batch_pad = batch_pad.to(device)
            batch_mask = batch_mask.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            emb = _forward_moment_batch(moment_model, batch_pad, batch_mask, device)
            logits = head(emb)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

        # --- Validate ---
        moment_model.eval()
        head.eval()
        val_preds = []
        val_gt = []
        with torch.no_grad():
            for batch_pad, batch_mask, batch_labels in val_loader:
                batch_pad = batch_pad.to(device)
                batch_mask = batch_mask.to(device)
                emb = _forward_moment_batch(moment_model, batch_pad, batch_mask, device)
                logits = head(emb)
                preds = logits.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_gt.extend(batch_labels.numpy())

        val_acc = accuracy_score(val_gt, val_preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_encoder_state = copy.deepcopy(moment_model.state_dict())
            best_head_state = copy.deepcopy(head.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        pbar.set_postfix(val_acc=f"{val_acc:.3f}", best=f"{best_val_acc:.3f}",
                         pat=f"{patience_counter}/{FINETUNE_PATIENCE}")

        if patience_counter >= FINETUNE_PATIENCE:
            print(f"    Early stopping at epoch {epoch + 1}")
            break

    # ------------------------------------------------------------------
    # 7. Test using best checkpoint
    # ------------------------------------------------------------------
    if best_encoder_state is not None:
        moment_model.load_state_dict(best_encoder_state)
    if best_head_state is not None:
        head.load_state_dict(best_head_state)

    moment_model.eval()
    head.eval()
    all_preds = []
    all_gt = []
    with torch.no_grad():
        for batch_pad, batch_mask, batch_labels in test_loader:
            batch_pad = batch_pad.to(device)
            batch_mask = batch_mask.to(device)
            emb = _forward_moment_batch(moment_model, batch_pad, batch_mask, device)
            logits = head(emb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_gt.extend(batch_labels.numpy())

    # ------------------------------------------------------------------
    # 8. Restore original encoder weights (so fine-tuning is isolated per run)
    # ------------------------------------------------------------------
    moment_model.load_state_dict(original_state)
    moment_model.eval()

    # Clean up GPU memory
    del head, optimizer, criterion
    del train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
    del best_encoder_state, best_head_state
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 9. Compute metrics
    # ------------------------------------------------------------------
    pred_indices = np.array(all_preds)
    gt_arr = np.array(all_gt)

    gt_names = []
    pred_names = []
    for i in range(len(gt_arr)):
        local_idx = gt_arr[i]
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

    return {'accuracy': acc, 'f1_macro': f1, 'f1_weighted': f1_w,
            'n_samples': len(gt_names), 'n_train_samples': len(train_pad),
            'n_classes': num_test_classes}


# =============================================================================
# Grouped Zero-Shot: Training Embedding Extraction
# =============================================================================

def load_moment_training_embeddings(model, global_labels, device):
    """Extract MOMENT embeddings for all 10 training datasets.

    Returns:
        all_embeddings: (N_total, 6144) concatenated embeddings
        all_labels: (N_total,) global label indices
    """
    all_emb_list = []
    all_lab_list = []

    for ds in TRAIN_DATASETS:
        print(f"  Extracting MOMENT embeddings for {ds}...")
        raw_data, raw_labels = load_raw_data(ds)
        emb = extract_moment_embeddings(model, raw_data, device)  # (N, 6144)
        labels = get_window_labels(raw_labels)
        global_lab = map_local_to_global_labels(labels, ds, DATASET_CONFIG, global_labels)
        all_emb_list.append(emb)
        all_lab_list.append(global_lab)
        print(f"    {ds}: {emb.shape[0]} samples")

    return np.concatenate(all_emb_list, axis=0), np.concatenate(all_lab_list, axis=0)


# =============================================================================
# Output Formatting
# =============================================================================

def print_results_table(all_results):
    """Print results table."""
    print()
    print("=" * 130)
    print("MOMENT EVALUATION RESULTS")
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
    print(f"  Model: {MOMENT_MODEL_NAME}")
    print(f"  Zero-shot: SVM-RBF on 10 training datasets (87 classes), group-matched scoring")
    print(f"  Open-set: SVM predict over all classes; Closed-set: decision scores masked to test-relevant groups")
    print(f"  Supervised: End-to-end fine-tuning (encoder lr={FINETUNE_ENCODER_LR}, "
          f"head lr={FINETUNE_HEAD_LR}, {FINETUNE_EPOCHS} epochs, patience={FINETUNE_PATIENCE})")


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

    # Load MOMENT model
    print(f"\nLoading MOMENT model: {MOMENT_MODEL_NAME}...")
    moment_model = load_moment_model(device)
    print("Model loaded successfully")

    # Load or train zero-shot SVM classifier
    global_labels = load_global_labels()
    zs_svm_path = OUTPUT_DIR / "moment_zs_svm.pkl"

    if zs_svm_path.exists():
        print(f"\nLoading cached zero-shot SVM from {zs_svm_path}...")
        zs_svm = joblib.load(zs_svm_path)
        print(f"  Loaded SVM with {len(zs_svm.classes_)} classes")
    else:
        print(f"\nExtracting training embeddings for zero-shot ({len(TRAIN_DATASETS)} datasets)...")
        train_emb, train_lab = load_moment_training_embeddings(moment_model, global_labels, device)
        print(f"Training data: {train_emb.shape[0]} samples, dim={train_emb.shape[1]}, "
              f"{len(np.unique(train_lab))} classes")

        print("\nTraining zero-shot SVM-RBF classifier...")
        zs_svm = train_svm_classifier(train_emb, train_lab, verbose=True)
        joblib.dump(zs_svm, zs_svm_path)
        print(f"  Saved zero-shot SVM to {zs_svm_path}")

    # Run scoring on each test dataset
    all_results = {}

    for test_ds in TEST_DATASETS:
        print(f"\n{'='*60}")
        print(f"Testing MOMENT on {test_ds}")
        print(f"{'='*60}")

        try:
            raw_data, raw_labels = load_raw_data(test_ds)
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            continue
        test_emb = extract_moment_embeddings(moment_model, raw_data, device)
        test_labels = get_window_labels(raw_labels)
        test_activities = get_dataset_labels(test_ds)

        print(f"  Test data: {test_emb.shape[0]} windows -> embeddings {test_emb.shape} "
              f"({DATA_CHANNELS}ch x {MOMENT_EMB_DIM_PER_CHANNEL}d concat), "
              f"{len(test_activities)} classes")
        print(f"  Classes: {test_activities}")

        ds_results = {}

        # 0. Zero-shot open-set (SVM on flat 6144-dim vectors)
        print(f"\n  --- Zero-Shot Open-Set (SVM-RBF) ---")
        pred_open = predict_svm_global(zs_svm, test_emb)
        ds_results['zero_shot_open_set'] = score_with_groups(
            pred_open, test_labels, test_ds, global_labels, DATASET_CONFIG)
        zs_open = ds_results['zero_shot_open_set']
        print(f"  ZS Open-set: Acc={zs_open['accuracy']:.1f}%, F1={zs_open['f1_macro']:.1f}%")

        # 1. Zero-shot closed-set (SVM with masked decision scores)
        print(f"\n  --- Zero-Shot Closed-Set (SVM-RBF) ---")
        mask = get_closed_set_mask(test_ds, global_labels, DATASET_CONFIG)
        pred_closed = predict_svm_global(zs_svm, test_emb, logit_mask=mask)
        ds_results['zero_shot_closed_set'] = score_with_groups(
            pred_closed, test_labels, test_ds, global_labels, DATASET_CONFIG)
        ds_results['zero_shot_closed_set']['n_allowed'] = int(mask.sum())
        ds_results['zero_shot_closed_set']['n_masked'] = int((~mask).sum())
        zs_close = ds_results['zero_shot_closed_set']
        print(f"  ZS Closed-set: Acc={zs_close['accuracy']:.1f}%, F1={zs_close['f1_macro']:.1f}% "
              f"({zs_close['n_allowed']}/{len(global_labels)} labels allowed)")

        # 3. 1% supervised fine-tuning (encoder + linear head)
        print(f"\n  --- 1% Supervised Fine-Tune ---")
        ds_results['1pct_supervised'] = evaluate_supervised_finetune(
            moment_model, raw_data, test_labels, test_ds, device,
            label_rate=SUPERVISED_LABEL_RATE_1PCT, label_tag="1%")
        print(f"  1% supervised: "
              f"Acc={ds_results['1pct_supervised']['accuracy']:.1f}%, "
              f"F1={ds_results['1pct_supervised']['f1_macro']:.1f}%")

        # 4. 10% supervised fine-tuning (encoder + linear head)
        print(f"\n  --- 10% Supervised Fine-Tune ---")
        ds_results['10pct_supervised'] = evaluate_supervised_finetune(
            moment_model, raw_data, test_labels, test_ds, device,
            label_rate=SUPERVISED_LABEL_RATE_10PCT, label_tag="10%")
        print(f"  10% supervised: "
              f"Acc={ds_results['10pct_supervised']['accuracy']:.1f}%, "
              f"F1={ds_results['10pct_supervised']['f1_macro']:.1f}%")

        all_results[test_ds] = ds_results

    # Print summary table
    print_results_table(all_results)

    # Save results
    results_path = OUTPUT_DIR / "moment_evaluation.json"
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
