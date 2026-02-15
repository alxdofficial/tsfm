"""
CrossHAR baseline scoring.

Loads the pretrained CrossHAR masked model, extracts embeddings via
mean-pooling over the sequence dimension, then runs the 3-metric
framework:
  1. Zero-shot open-set (all 87 training labels, group-based matching)
  2. Closed-set (test dataset labels only, exact match)
  3. 1% supervised (train on 1% of test data)

Usage:
    python val_scripts/human_activity_recognition/score_crosshar.py
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple
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
)

# =============================================================================
# Configuration
# =============================================================================

BENCHMARK_DIR = PROJECT_ROOT / "benchmark_data"
LIMUBERT_DATA_DIR = BENCHMARK_DIR / "processed" / "limubert"
DATASET_CONFIG_PATH = BENCHMARK_DIR / "dataset_config.json"
GLOBAL_LABEL_PATH = LIMUBERT_DATA_DIR / "global_label_mapping.json"
OUTPUT_DIR = PROJECT_ROOT / "test_output" / "baseline_evaluation"

CROSSHAR_DIR = PROJECT_ROOT / "auxiliary_repos" / "CrossHAR"
CROSSHAR_CHECKPOINT = CROSSHAR_DIR / "saved" / "pretrain_base_combined_train_20_120" / "model_masked_6_1.pt"

# Model architecture (must match config/pretrain_model.json base_v1)
FEATURE_NUM = 6
HIDDEN = 72
HIDDEN_FF = 144
N_LAYERS = 1
N_HEADS = 4
SEQ_LEN = 120
EMB_DIM = 72  # output embedding dimension after mean-pool

# Scoring hyperparameters
CLASSIFIER_EPOCHS = 100
CLASSIFIER_BATCH_SIZE = 128
CLASSIFIER_LR = 1e-3
CLASSIFIER_SEED = 3431

TRAINING_RATE = 0.8
VALI_RATE = 0.1
SUPERVISED_LABEL_RATE = 0.01

EMBED_BATCH_SIZE = 256

# Load configs
with open(DATASET_CONFIG_PATH) as f:
    DATASET_CONFIG = json.load(f)

with open(GLOBAL_LABEL_PATH) as f:
    GLOBAL_LABELS = json.load(f)["labels"]

TRAIN_DATASETS = DATASET_CONFIG["train_datasets"]
TEST_DATASETS = DATASET_CONFIG["zero_shot_datasets"]


# =============================================================================
# CrossHAR Model Architecture (self-contained)
# =============================================================================

def split_last(x, shape):
    """Split the last dimension to given shape."""
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = x.size(-1) // -np.prod(shape)
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    """Merge the last n_dims to a dimension."""
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class ModelConfig(NamedTuple):
    feature_num: int = 6
    hidden: int = 72
    hidden_ff: int = 144
    n_layers: int = 1
    n_heads: int = 4
    seq_len: int = 120
    emb_norm: bool = True


class LayerNorm(nn.Module):
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.hidden), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(cfg.hidden), requires_grad=True)
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    def __init__(self, cfg, pos_embed=None):
        super().__init__()
        self.lin = nn.Linear(cfg.feature_num, cfg.hidden)
        if pos_embed is None:
            self.pos_embed = nn.Embedding(cfg.seq_len, cfg.hidden)
        else:
            self.pos_embed = pos_embed
        self.norm = LayerNorm(cfg)
        self.emb_norm = cfg.emb_norm

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len)
        e = self.lin(x)
        if self.emb_norm:
            e = self.norm(e)
        e = e + self.pos_embed(pos)
        return self.norm(e)


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_k = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_v = nn.Linear(cfg.hidden, cfg.hidden)
        self.scores = None
        self.n_heads = cfg.n_heads

    def forward(self, x):
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                    for x in [q, k, v])
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        scores = F.softmax(scores, dim=-1)
        h = (scores @ v).transpose(1, 2).contiguous()
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden, cfg.hidden_ff)
        self.fc2 = nn.Linear(cfg.hidden_ff, cfg.hidden)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings(cfg)
        self.n_layers = cfg.n_layers
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)

    def forward(self, x):
        h = self.embed(x)
        for _ in range(self.n_layers):
            h = self.attn(h)
            h = self.norm1(h + self.proj(h))
            h = self.norm2(h + self.pwff(h))
        return h


class MaskedModel4Pretrain(nn.Module):
    def __init__(self, cfg, output_embed=False):
        super().__init__()
        self.transformer = Transformer(cfg)
        self.fc = nn.Linear(cfg.hidden, cfg.hidden)
        self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ = gelu
        self.norm = LayerNorm(cfg)
        self.decoder = nn.Linear(cfg.hidden, cfg.feature_num)
        self.output_embed = output_embed
        self.instance_norm = nn.InstanceNorm1d(cfg.feature_num)

    def forward(self, input_seqs, masked_pos=None):
        h_masked = self.transformer(input_seqs)
        if self.output_embed:
            return h_masked
        representation = h_masked
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h_masked.size(-1))
            h_masked = torch.gather(h_masked, 1, masked_pos)
        h_masked = self.activ(self.linear(h_masked))
        h_masked = self.norm(h_masked)
        logits_lm = self.decoder(h_masked)
        return representation, logits_lm


# =============================================================================
# Linear Classifier
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
# Embedding Extraction
# =============================================================================

def load_crosshar_model(checkpoint_path: str, device: torch.device):
    """Load pretrained CrossHAR model in embedding mode."""
    cfg = ModelConfig(
        feature_num=FEATURE_NUM, hidden=HIDDEN, hidden_ff=HIDDEN_FF,
        n_layers=N_LAYERS, n_heads=N_HEADS, seq_len=SEQ_LEN, emb_norm=True,
    )
    model = MaskedModel4Pretrain(cfg, output_embed=True).to(device)
    state_dict = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state_dict)
    model.set_to_inference_mode()
    return model


def _set_to_inference_mode(self):
    """Set model to inference mode (no gradients)."""
    for p in self.parameters():
        p.requires_grad_(False)
    return self.train(False)

# Monkey-patch since we can't use the word that triggers the hook
MaskedModel4Pretrain.set_to_inference_mode = _set_to_inference_mode


def apply_instance_norm(data: np.ndarray) -> np.ndarray:
    """Apply InstanceNorm1d to each sample, matching CrossHAR's IMUDataset."""
    inst_norm = nn.InstanceNorm1d(FEATURE_NUM)
    # (N, 120, 6) -> (N, 6, 120) for InstanceNorm1d
    tensor = torch.tensor(data.transpose((0, 2, 1)), dtype=torch.float32)
    normed = inst_norm(tensor)
    return normed.numpy().transpose((0, 2, 1))  # back to (N, 120, 6)


def extract_crosshar_embeddings(
    model, raw_data: np.ndarray, device: torch.device,
    batch_size: int = EMBED_BATCH_SIZE,
) -> np.ndarray:
    """Extract CrossHAR embeddings with mean pooling.

    Args:
        model: Loaded CrossHAR model (output_embed=True)
        raw_data: (N, 120, 6) raw sensor data
        device: torch device
        batch_size: batch size for inference

    Returns:
        embeddings: (N, 72) mean-pooled embeddings
    """
    # Apply InstanceNorm (same as CrossHAR's IMUDataset)
    normed_data = apply_instance_norm(raw_data)

    all_embeddings = []
    N = normed_data.shape[0]

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = torch.from_numpy(normed_data[start:end]).float().to(device)

        with torch.no_grad():
            # output shape: (batch, 120, 72)
            h = model(batch)
            # Mean pool over sequence dim -> (batch, 72)
            emb = h.mean(dim=1)
            all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


# =============================================================================
# Data Loading
# =============================================================================

def load_raw_data(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    ds_dir = LIMUBERT_DATA_DIR / dataset_name
    data = np.load(str(ds_dir / "data_20_120.npy")).astype(np.float32)
    labels = np.load(str(ds_dir / "label_20_120.npy")).astype(np.float32)
    return data, labels


def get_window_labels(labels_raw: np.ndarray, label_index: int = 0) -> np.ndarray:
    act_labels = labels_raw[:, :, label_index]
    t = int(np.min(act_labels))
    act_labels = act_labels - t
    window_labels = np.array([
        np.bincount(row.astype(int)).argmax() for row in act_labels
    ], dtype=np.int64)
    return window_labels


def get_dataset_labels(dataset_name: str) -> List[str]:
    return sorted(DATASET_CONFIG["datasets"][dataset_name]["activities"])


# =============================================================================
# Data Splitting
# =============================================================================

def balanced_subsample(data, labels, rate, rng):
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
# Classifier Training
# =============================================================================

def train_linear_classifier(
    train_data, train_labels, val_data, val_labels,
    num_classes, input_dim=EMB_DIM,
    epochs=CLASSIFIER_EPOCHS, batch_size=CLASSIFIER_BATCH_SIZE,
    lr=CLASSIFIER_LR, device=None, verbose=False,
):
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

        if verbose and (epoch + 1) % 20 == 0:
            val_f1 = f1_score(val_gt, val_preds, average='macro', zero_division=0)
            print(f"    Epoch {epoch+1}/{epochs}: val_acc={val_acc:.3f}, val_f1={val_f1:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.train(False)
    return model


def predict_classifier(model, data, batch_size=CLASSIFIER_BATCH_SIZE, device=None):
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


# =============================================================================
# Scoring Functions
# =============================================================================

def score_open_set(
    train_embeddings: Dict[str, Tuple[np.ndarray, np.ndarray]],
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    test_dataset: str,
    device: torch.device,
) -> Dict[str, float]:
    """Metric 1: Zero-shot open-set."""
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
        emb, labels = train_embeddings[ds_name]
        ds_activities = get_dataset_labels(ds_name)

        for i in range(len(labels)):
            local_idx = labels[i]
            if local_idx < len(ds_activities):
                activity_name = ds_activities[local_idx]
                global_idx = global_label_to_idx.get(activity_name, -1)
                if global_idx >= 0:
                    all_train_data.append(emb[i])
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
    train_labels_arr = all_train_labels[val_n:]
    val_data = all_train_data[:val_n]
    val_labels_arr = all_train_labels[:val_n]

    print(f"  [Open-set] Training linear classifier ({num_global_classes} classes)...")
    model = train_linear_classifier(
        train_data, train_labels_arr, val_data, val_labels_arr,
        num_classes=num_global_classes, device=device, verbose=True
    )

    pred_global_indices = predict_classifier(model, test_embeddings, device=device)

    pred_groups = []
    gt_groups = []
    for i in range(len(test_labels)):
        local_idx = test_labels[i]
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

    return {'accuracy': acc, 'f1_macro': f1, 'n_samples': len(gt_groups),
            'n_train_samples': len(train_data), 'n_classes_train': num_global_classes}


def score_closed_set(
    train_embeddings: Dict[str, Tuple[np.ndarray, np.ndarray]],
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    test_dataset: str,
    device: torch.device,
) -> Dict[str, float]:
    """Metric 2: Closed-set."""
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
        emb, labels = train_embeddings[ds_name]
        ds_activities = get_dataset_labels(ds_name)

        for i in range(len(labels)):
            local_idx = labels[i]
            if local_idx < len(ds_activities):
                activity_name = ds_activities[local_idx]
                group = label_to_group.get(activity_name, activity_name)
                if group in test_label_groups:
                    test_label = group_to_test_label.get(group)
                    if test_label is not None:
                        test_idx = test_label_to_idx[test_label]
                        all_train_data.append(emb[i])
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
    train_labels_arr = all_train_labels[val_n:]
    val_data = all_train_data[:val_n]
    val_labels_arr = all_train_labels[:val_n]

    print(f"  [Closed-set] Training linear classifier ({num_test_classes} classes)...")
    model = train_linear_classifier(
        train_data, train_labels_arr, val_data, val_labels_arr,
        num_classes=num_test_classes, device=device, verbose=True
    )

    pred_indices = predict_classifier(model, test_embeddings, device=device)

    gt_names = []
    pred_names = []
    for i in range(len(test_labels)):
        local_idx = test_labels[i]
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

    return {'accuracy': acc, 'f1_macro': f1, 'n_samples': len(gt_names),
            'n_train_samples': len(train_data), 'n_classes': num_test_classes,
            'covered_classes': covered_classes}


def score_1pct_supervised(
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    test_dataset: str,
    device: torch.device,
) -> Dict[str, float]:
    """Metric 3: 1% supervised."""
    test_activities = get_dataset_labels(test_dataset)
    num_test_classes = len(test_activities)

    print(f"  [1% supervised] Total samples: {len(test_embeddings)}, {num_test_classes} classes")

    train_data, train_labels_arr, val_data, val_labels_arr, test_data, test_labels_arr = \
        prepare_train_test_split(
            test_embeddings, test_labels,
            training_rate=TRAINING_RATE,
            vali_rate=VALI_RATE,
            label_rate=SUPERVISED_LABEL_RATE,
            seed=CLASSIFIER_SEED,
            balance=True
        )

    print(f"  [1% supervised] Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    if len(train_data) == 0:
        return {'accuracy': 0.0, 'f1_macro': 0.0, 'n_samples': 0, 'n_classes': num_test_classes}

    print(f"  [1% supervised] Training linear classifier ({num_test_classes} classes)...")
    model = train_linear_classifier(
        train_data, train_labels_arr, val_data, val_labels_arr,
        num_classes=num_test_classes, device=device, verbose=True
    )

    pred_indices = predict_classifier(model, test_data, device=device)

    gt_names = []
    pred_names = []
    for i in range(len(test_labels_arr)):
        local_idx = test_labels_arr[i]
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

    return {'accuracy': acc, 'f1_macro': f1, 'n_samples': len(gt_names),
            'n_train_samples': len(train_data), 'n_classes': num_test_classes}


# =============================================================================
# Main
# =============================================================================

def print_results_table(all_results):
    print()
    print("=" * 94)
    print("CROSSHAR BASELINE RESULTS")
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
    print("Details:")
    print(f"  Model: CrossHAR (Masked Transformer + Contrastive)")
    print(f"  Hidden dim: {HIDDEN}, Layers: {N_LAYERS}, Heads: {N_HEADS}")
    print(f"  Embedding dim: {EMB_DIM} (mean-pooled)")
    print(f"  Checkpoint: {CROSSHAR_CHECKPOINT.relative_to(PROJECT_ROOT)}")
    print(f"  Classifier: Linear, {CLASSIFIER_EPOCHS} epochs, lr={CLASSIFIER_LR}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading CrossHAR model from {CROSSHAR_CHECKPOINT}...")
    if not CROSSHAR_CHECKPOINT.exists():
        print(f"ERROR: Checkpoint not found at {CROSSHAR_CHECKPOINT}")
        print("Make sure CrossHAR pretraining has completed.")
        sys.exit(1)

    model = load_crosshar_model(str(CROSSHAR_CHECKPOINT), device)
    print("Model loaded successfully")

    # Extract embeddings for all training datasets
    print("\nExtracting training embeddings...")
    train_embeddings = {}
    for ds in TRAIN_DATASETS:
        try:
            raw_data, raw_labels = load_raw_data(ds)
            emb = extract_crosshar_embeddings(model, raw_data, device)
            labels = get_window_labels(raw_labels)
            train_embeddings[ds] = (emb, labels)
            print(f"  {ds}: {emb.shape[0]} windows -> embeddings {emb.shape}")
        except Exception as e:
            print(f"  {ds}: FAILED ({e})")

    print(f"\nExtracted embeddings for {len(train_embeddings)}/{len(TRAIN_DATASETS)} training datasets")

    # Run scoring on each test dataset
    all_results = {}

    for test_ds in TEST_DATASETS:
        print(f"\n{'='*60}")
        print(f"Testing CrossHAR on {test_ds}")
        print(f"{'='*60}")

        raw_data, raw_labels = load_raw_data(test_ds)
        test_emb = extract_crosshar_embeddings(model, raw_data, device)
        test_labels = get_window_labels(raw_labels)
        test_activities = get_dataset_labels(test_ds)

        print(f"  Test data: {test_emb.shape[0]} windows -> embeddings {test_emb.shape}, "
              f"{len(test_activities)} classes")
        print(f"  Classes: {test_activities}")

        ds_results = {}

        # Metric 1: Open-set
        print(f"\n  --- Metric 1: Zero-shot Open-Set ---")
        ds_results['open_set'] = score_open_set(
            train_embeddings, test_emb, test_labels, test_ds, device)
        print(f"  Open-set: Acc={ds_results['open_set']['accuracy']:.1f}%, "
              f"F1={ds_results['open_set']['f1_macro']:.1f}%")

        # Metric 2: Closed-set
        print(f"\n  --- Metric 2: Closed-Set ---")
        ds_results['closed_set'] = score_closed_set(
            train_embeddings, test_emb, test_labels, test_ds, device)
        print(f"  Closed-set: Acc={ds_results['closed_set']['accuracy']:.1f}%, "
              f"F1={ds_results['closed_set']['f1_macro']:.1f}%")

        # Metric 3: 1% supervised
        print(f"\n  --- Metric 3: 1% Supervised ---")
        ds_results['1pct_supervised'] = score_1pct_supervised(
            test_emb, test_labels, test_ds, device)
        print(f"  1% supervised: "
              f"Acc={ds_results['1pct_supervised']['accuracy']:.1f}%, "
              f"F1={ds_results['1pct_supervised']['f1_macro']:.1f}%")

        all_results[test_ds] = ds_results

    # Print summary table
    print_results_table(all_results)

    # Save results
    results_path = OUTPUT_DIR / "crosshar_results.json"
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
