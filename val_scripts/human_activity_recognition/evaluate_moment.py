"""
MOMENT baseline evaluation using the new evaluation framework.

MOMENT is NOT text-aligned, so zero-shot evaluation is N/A.
Evaluates with:
  1. 1% supervised (SVM-RBF classifier - paper's protocol)
  2. 10% supervised (SVM-RBF classifier)
  3. Linear probe (linear classifier on frozen embeddings, full train split)

Following the MOMENT paper (Goswami et al., ICML 2024), we use an SVM
classifier with RBF kernel and GridSearchCV over C values for supervised
evaluation. Linear probe uses a simple nn.Linear for fair comparison.

Usage:
    python val_scripts/human_activity_recognition/evaluate_moment.py
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
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
MOMENT_EMB_DIM_PER_CHANNEL = 1024  # MOMENT-1-large per-channel embedding dim
MOMENT_EMB_DIM = DATA_CHANNELS * MOMENT_EMB_DIM_PER_CHANNEL  # 6144 (concat per-channel)
DATA_SEQ_LEN = 120         # Our data window length
DATA_CHANNELS = 6          # 6-channel IMU
MOMENT_BATCH_SIZE = 128    # Batch size for embedding extraction (MOMENT-large fits ~128 on 24GB)

# SVM hyperparameters (matching MOMENT paper's fit_svm protocol)
SVM_C_VALUES = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
SVM_MAX_SAMPLES = 10000  # Subsample if training set exceeds this
CLASSIFIER_SEED = 3431

# Linear probe hyperparameters
LINEAR_EPOCHS = 100
LINEAR_BATCH_SIZE = 512
LINEAR_LR = 1e-3

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
# Linear Classifier (for linear probe)
# =============================================================================

class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x, training=False):
        if training:
            x = F.dropout(x, p=0.3, training=True)
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
    for start in range(0, N, batch_size):
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
              f"{nb_classes} classes, 5-fold CV)...")

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


# =============================================================================
# Linear Classifier Training (for linear probe)
# =============================================================================

def train_linear_classifier(
    train_data, train_labels, val_data, val_labels,
    num_classes, input_dim=MOMENT_EMB_DIM,
    epochs=LINEAR_EPOCHS, batch_size=LINEAR_BATCH_SIZE,
    lr=LINEAR_LR, device=None, verbose=False,
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


def predict_linear(model, data, batch_size=LINEAR_BATCH_SIZE, device=None):
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

def evaluate_supervised_svm(
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    test_dataset: str,
    label_rate: float = 0.01,
    label_tag: str = "1%",
) -> Dict[str, float]:
    """Supervised with SVM-RBF (paper's protocol).

    Args:
        label_rate: Fraction of training portion to use (0.01 = 1%, 0.10 = 10%)
        label_tag: Display tag for logging
    """
    test_activities = get_dataset_labels(test_dataset)
    num_test_classes = len(test_activities)

    print(f"  [{label_tag} supervised SVM] Total samples: {len(test_embeddings)}, {num_test_classes} classes")

    train_data, train_labels_arr, val_data, val_labels_arr, eval_data, eval_labels_arr = \
        prepare_train_test_split(
            test_embeddings, test_labels,
            training_rate=TRAINING_RATE,
            vali_rate=VALI_RATE,
            label_rate=label_rate,
            seed=CLASSIFIER_SEED,
            balance=True
        )

    print(f"  [{label_tag} supervised SVM] Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(eval_data)}")

    if len(train_data) == 0:
        return {'accuracy': 0.0, 'f1_macro': 0.0, 'n_samples': 0, 'n_classes': num_test_classes}

    print(f"  [{label_tag} supervised SVM] Training SVM-RBF ({num_test_classes} classes)...")
    model = train_svm_classifier(train_data, train_labels_arr, verbose=True)
    pred_indices = model.predict(eval_data)

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
    """Linear probe: linear classifier on frozen embeddings, full train split.

    MOMENT embeddings are (N, 6144) per-channel concatenated, no pooling needed.
    """
    test_activities = get_dataset_labels(test_dataset)
    num_test_classes = len(test_activities)

    print(f"  [Linear probe] Total samples: {len(test_embeddings)}, {num_test_classes} classes")

    train_data, train_labels_arr, val_data, val_labels_arr, eval_data, eval_labels_arr = \
        prepare_train_test_split(
            test_embeddings, test_labels,
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
        train_data, train_labels_arr, val_data, val_labels_arr,
        num_classes=num_test_classes, device=device, verbose=True
    )

    pred_indices = predict_linear(model, eval_data, device=device)

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


# =============================================================================
# Output Formatting
# =============================================================================

def print_results_table(all_results):
    """Print results table."""
    print()
    print("=" * 100)
    print("MOMENT EVALUATION RESULTS (New Framework)")
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
    print(f"  Model: {MOMENT_MODEL_NAME}")
    print(f"  Supervised: SVM-RBF with GridSearchCV (C={SVM_C_VALUES})")
    print(f"  Linear probe: Linear classifier, {LINEAR_EPOCHS} epochs, lr={LINEAR_LR}")
    print(f"  Zero-shot: N/A (MOMENT is not text-aligned)")


# =============================================================================
# Main
# =============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load MOMENT model
    print(f"\nLoading MOMENT model: {MOMENT_MODEL_NAME}...")
    moment_model = load_moment_model(device)
    print("Model loaded successfully")

    # Run evaluation on each test dataset
    all_results = {}

    for test_ds in TEST_DATASETS:
        print(f"\n{'='*60}")
        print(f"Testing MOMENT on {test_ds}")
        print(f"{'='*60}")

        raw_data, raw_labels = load_raw_data(test_ds)
        test_emb = extract_moment_embeddings(moment_model, raw_data, device)
        test_labels = get_window_labels(raw_labels)
        test_activities = get_dataset_labels(test_ds)

        print(f"  Test data: {test_emb.shape[0]} windows -> embeddings {test_emb.shape} "
              f"({DATA_CHANNELS}ch x {MOMENT_EMB_DIM_PER_CHANNEL}d concat), "
              f"{len(test_activities)} classes")
        print(f"  Classes: {test_activities}")

        ds_results = {}

        # 1. 1% supervised (SVM)
        print(f"\n  --- 1% Supervised (SVM) ---")
        ds_results['1pct_supervised'] = evaluate_supervised_svm(
            test_emb, test_labels, test_ds,
            label_rate=SUPERVISED_LABEL_RATE_1PCT, label_tag="1%")
        print(f"  1% supervised: "
              f"Acc={ds_results['1pct_supervised']['accuracy']:.1f}%, "
              f"F1={ds_results['1pct_supervised']['f1_macro']:.1f}%")

        # 2. 10% supervised (SVM)
        print(f"\n  --- 10% Supervised (SVM) ---")
        ds_results['10pct_supervised'] = evaluate_supervised_svm(
            test_emb, test_labels, test_ds,
            label_rate=SUPERVISED_LABEL_RATE_10PCT, label_tag="10%")
        print(f"  10% supervised: "
              f"Acc={ds_results['10pct_supervised']['accuracy']:.1f}%, "
              f"F1={ds_results['10pct_supervised']['f1_macro']:.1f}%")

        # 3. Linear probe (full train split)
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
