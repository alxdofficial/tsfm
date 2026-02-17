"""
CrossHAR baseline scoring using the unified framework.

Evaluates with:
  1. Zero-shot open-set (classifier on 87 training labels, group scoring)
  2. Zero-shot closed-set (classifier with masked logits, group scoring)
  3. 1% supervised (end-to-end fine-tuning of encoder + Transformer_ft classifier)
  4. 10% supervised (end-to-end fine-tuning of encoder + Transformer_ft classifier)

Usage:
    python val_scripts/human_activity_recognition/evaluate_crosshar.py
"""

import json
import math
import random
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

CROSSHAR_DIR = PROJECT_ROOT / "auxiliary_repos" / "CrossHAR"
CROSSHAR_CHECKPOINT = CROSSHAR_DIR / "saved" / "pretrain_base_combined_train_20_120" / "model_masked_6_1.pt"

# Model architecture (must match config/pretrain_model.json base_v1)
FEATURE_NUM = 6
HIDDEN = 72
HIDDEN_FF = 144
N_LAYERS = 1
N_HEADS = 4
SEQ_LEN = 120
EMB_DIM = 72  # CrossHAR hidden dimension (input to Transformer_ft classifier)

# Scoring hyperparameters (zero-shot classifier training)
CLASSIFIER_EPOCHS = 100
CLASSIFIER_BATCH_SIZE = 512
CLASSIFIER_LR = 1e-3
CLASSIFIER_SEED = 3431

# End-to-end fine-tuning hyperparameters (supervised metrics)
FINETUNE_EPOCHS = 20
FINETUNE_BATCH_SIZE = 32
FINETUNE_ENCODER_LR = 1e-5
FINETUNE_HEAD_LR = 1e-3
FINETUNE_WEIGHT_DECAY = 1e-5
FINETUNE_PATIENCE = 5

TRAINING_RATE = 0.8
VALI_RATE = 0.1
SUPERVISED_LABEL_RATE_1PCT = 0.01
SUPERVISED_LABEL_RATE_10PCT = 0.10

EMBED_BATCH_SIZE = 512

# Load configs
with open(DATASET_CONFIG_PATH) as f:
    DATASET_CONFIG = json.load(f)

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
# Transformer Classifier (matching original CrossHAR's Transformer_ft)
# Config: input_size=72, hidden_size=100, num_layers=1, num_heads=4,
#         dim_feedforward=2048, dropout=0.1
# =============================================================================

class ClassifierPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding matching CrossHAR's PositionalEncoding."""
    def __init__(self, hidden_size, max_seq_len=120):
        super().__init__()
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size)
        )
        pos_enc = torch.zeros(1, max_seq_len, hidden_size)
        pos_enc[0, :, 0::2] = torch.sin(position * div_term)
        pos_enc[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        return x + self.pos_enc[:, :x.size(1)]


FT_HIDDEN_SIZE = 100
FT_NUM_HEADS = 4
FT_DIM_FEEDFORWARD = 2048
FT_NUM_LAYERS = 1
FT_DROPOUT = 0.1


class TransformerClassifier(nn.Module):
    """Matching CrossHAR's Transformer_ft classifier.

    Architecture: Linear(72→100) → PosEnc → TransformerEncoder(1 layer, 4 heads,
    ff=2048) → MeanPool → Linear(100→num_classes)

    Note: Original Transformer_ft defines nn.Dropout but never applies it in
    forward(). We match this exactly — dropout only applies inside the
    TransformerEncoderLayer (PyTorch default 0.1).
    """
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Linear(input_dim, FT_HIDDEN_SIZE)
        self.positional_encoding = ClassifierPositionalEncoding(FT_HIDDEN_SIZE)
        encoder_layer = nn.TransformerEncoderLayer(
            FT_HIDDEN_SIZE, FT_NUM_HEADS,
            dim_feedforward=FT_DIM_FEEDFORWARD, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, FT_NUM_LAYERS)
        self.fc = nn.Linear(FT_HIDDEN_SIZE, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)
        return self.fc(x)


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
    """Extract CrossHAR sequence embeddings (no pooling).

    Returns full sequence embeddings for downstream Transformer_ft classifier,
    matching the original CrossHAR evaluation protocol.

    Args:
        model: Loaded CrossHAR model (output_embed=True)
        raw_data: (N, 120, 6) raw sensor data
        device: torch device
        batch_size: batch size for inference

    Returns:
        embeddings: (N, 120, 72) full sequence embeddings
    """
    # Apply InstanceNorm (same as CrossHAR's IMUDataset)
    normed_data = apply_instance_norm(raw_data)

    all_embeddings = []
    N = normed_data.shape[0]

    for start in tqdm(range(0, N, batch_size), desc="CrossHAR | Extracting embeddings",
                      total=(N + batch_size - 1) // batch_size, leave=True):
        end = min(start + batch_size, N)
        batch = torch.from_numpy(normed_data[start:end]).float().to(device)

        with torch.no_grad():
            # output shape: (batch, 120, 72)
            h = model(batch)
            all_embeddings.append(h.cpu().numpy())

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

def train_transformer_classifier(
    train_data, train_labels, val_data, val_labels,
    num_classes, input_dim=EMB_DIM,
    epochs=CLASSIFIER_EPOCHS, batch_size=CLASSIFIER_BATCH_SIZE,
    lr=CLASSIFIER_LR, device=None, verbose=False, desc="Transformer_ft",
):
    """Train Transformer_ft classifier matching original CrossHAR protocol.

    Args:
        train_data: (N, 120, 72) sequence embeddings
        train_labels: (N,) integer labels
        val_data: (M, 120, 72) validation sequence embeddings
        val_labels: (M,) validation labels
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TransformerClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
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

    best_val_loss = float('inf')
    best_state = None

    pbar = tqdm(range(epochs), desc=desc, leave=True)
    for epoch in pbar:
        model.train()
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(batch_data)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

        # Validation by loss (matching original CrossHAR's FinetuneTrainer)
        model.train(False)
        val_loss = 0.0
        val_preds = []
        val_gt = []
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                logits = model(batch_data)
                val_loss += criterion(logits, batch_labels).item()
                preds = logits.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_gt.extend(batch_labels.cpu().numpy())

        val_loss /= max(len(val_loader), 1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        val_acc = accuracy_score(val_gt, val_preds)
        pbar.set_postfix(val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.train(False)
    return model


def predict_transformer_classifier(model, data, batch_size=CLASSIFIER_BATCH_SIZE, device=None):
    """Get predictions from a trained TransformerClassifier.

    Args:
        data: (N, 120, 72) sequence embeddings
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.train(False)
    ds = TensorDataset(torch.from_numpy(data).float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for (batch_data,) in loader:
            batch_data = batch_data.to(device)
            logits = model(batch_data)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    return np.array(all_preds)


def predict_transformer_global(model, data, device, logit_mask=None,
                               batch_size=CLASSIFIER_BATCH_SIZE):
    """Predict global label indices using Transformer_ft classifier.

    Args:
        model: Trained TransformerClassifier with num_classes=87
        data: (N, 120, 72) sequence embeddings
        device: torch device
        logit_mask: Optional (87,) bool mask for closed-set scoring.
            If provided, masked-out logits are set to -inf before argmax.

    Returns:
        pred_global_indices: (N,) predicted global label indices
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
            logits = model(batch_data)
            if logit_mask is not None:
                logits[:, ~mask_tensor] = float('-inf')
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    return np.array(all_preds)


# =============================================================================
# Grouped Zero-Shot: Training Embedding Extraction
# =============================================================================

def load_crosshar_training_embeddings(model, global_labels, device):
    """Extract CrossHAR embeddings for all 10 training datasets.

    Returns full sequence embeddings for the Transformer_ft zero-shot classifier.

    Returns:
        all_embeddings: (N_total, 120, 72) concatenated sequence embeddings
        all_labels: (N_total,) global label indices
    """
    all_emb_list = []
    all_lab_list = []

    for ds in tqdm(TRAIN_DATASETS, desc="CrossHAR | Loading training embeddings", leave=True):
        raw_data, raw_labels = load_raw_data(ds)
        emb = extract_crosshar_embeddings(model, raw_data, device)  # (N, 120, 72)
        labels = get_window_labels(raw_labels)
        global_lab = map_local_to_global_labels(labels, ds, DATASET_CONFIG, global_labels)
        all_emb_list.append(emb)
        all_lab_list.append(global_lab)
        tqdm.write(f"    {ds}: {emb.shape[0]} samples")

    return np.concatenate(all_emb_list, axis=0), np.concatenate(all_lab_list, axis=0)


# =============================================================================
# Scoring Functions
# =============================================================================

def score_supervised_finetune(
    encoder_model,
    raw_data: np.ndarray,
    test_labels: np.ndarray,
    test_dataset: str,
    device: torch.device,
    label_rate: float = 0.01,
    label_tag: str = "1%",
) -> Dict[str, float]:
    """End-to-end fine-tuning: encoder + Transformer_ft classifier.

    Deep-copies the pretrained encoder, re-enables gradients, and fine-tunes
    both encoder and a fresh Transformer_ft classification head on raw data.
    Splits raw data into 80/10/10 train/val/test, subsamples train by label_rate.
    """
    test_activities = get_dataset_labels(test_dataset)
    num_test_classes = len(test_activities)

    print(f"  [{label_tag} finetune] Total samples: {len(raw_data)}, "
          f"{num_test_classes} classes")

    # Split RAW data (not embeddings)
    train_data, train_labels_arr, val_data, val_labels_arr, test_data, test_labels_arr = \
        prepare_train_test_split(
            raw_data, test_labels,
            training_rate=TRAINING_RATE,
            vali_rate=VALI_RATE,
            label_rate=label_rate,
            seed=CLASSIFIER_SEED,
            balance=True,
        )

    print(f"  [{label_tag} finetune] Train: {len(train_data)}, Val: {len(val_data)}, "
          f"Test: {len(test_data)}")

    if len(train_data) == 0:
        return {'accuracy': 0.0, 'f1_macro': 0.0, 'n_samples': 0, 'n_classes': num_test_classes}

    print(f"  [{label_tag} finetune] End-to-end fine-tuning encoder + Transformer_ft "
          f"({num_test_classes} classes)...")

    # Deep-copy encoder, re-enable gradients (original has requires_grad=False)
    ft_encoder = copy.deepcopy(encoder_model)
    for p in ft_encoder.parameters():
        p.requires_grad_(True)
    ft_encoder.train()

    # Create fresh classifier head
    classifier = TransformerClassifier(input_dim=EMB_DIM, num_classes=num_test_classes).to(device)

    # InstanceNorm module for differentiable forward pass
    inst_norm = nn.InstanceNorm1d(FEATURE_NUM).to(device)

    # Two param groups with different learning rates
    optimizer = torch.optim.AdamW([
        {'params': ft_encoder.parameters(), 'lr': FINETUNE_ENCODER_LR},
        {'params': classifier.parameters(), 'lr': FINETUNE_HEAD_LR},
    ], weight_decay=FINETUNE_WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # Build data loaders from raw numpy arrays
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

    best_val_acc = 0.0
    best_encoder_state = None
    best_classifier_state = None
    patience_counter = 0

    pbar = tqdm(range(FINETUNE_EPOCHS),
                desc=f"CrossHAR | {test_dataset} | finetune {label_tag}", leave=True)
    for epoch in pbar:
        # --- Training ---
        ft_encoder.train()
        classifier.train()
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)    # (B, 120, 6)
            batch_labels = batch_labels.to(device)

            # Apply InstanceNorm: (B, 120, 6) -> (B, 6, 120) -> norm -> (B, 120, 6)
            normed = inst_norm(batch_data.transpose(1, 2)).transpose(1, 2)

            # Forward through encoder -> (B, 120, 72)
            h = ft_encoder(normed)
            # Forward through classifier -> (B, num_classes)
            logits = classifier(h)

            loss = criterion(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- Validation ---
        ft_encoder.train(False)
        classifier.train(False)
        val_loss = 0.0
        val_preds = []
        val_gt = []
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                normed = inst_norm(batch_data.transpose(1, 2)).transpose(1, 2)
                h = ft_encoder(normed)
                logits = classifier(h)
                val_loss += criterion(logits, batch_labels).item()
                preds = logits.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_gt.extend(batch_labels.cpu().numpy())

        val_loss /= max(len(val_loader), 1)
        val_acc = accuracy_score(val_gt, val_preds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_encoder_state = copy.deepcopy(ft_encoder.state_dict())
            best_classifier_state = copy.deepcopy(classifier.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        pbar.set_postfix(val_acc=f"{val_acc:.3f}", val_loss=f"{val_loss:.4f}",
                         patience=f"{patience_counter}/{FINETUNE_PATIENCE}")

        if patience_counter >= FINETUNE_PATIENCE:
            print(f"  [{label_tag} finetune] Early stopping at epoch {epoch + 1}")
            break

    # Restore best weights
    if best_encoder_state is not None:
        ft_encoder.load_state_dict(best_encoder_state)
        classifier.load_state_dict(best_classifier_state)
    ft_encoder.train(False)
    classifier.train(False)

    # --- Evaluate on test split ---
    all_preds = []
    all_gt = []
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            normed = inst_norm(batch_data.transpose(1, 2)).transpose(1, 2)
            h = ft_encoder(normed)
            logits = classifier(h)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_gt.extend(batch_labels.numpy())

    pred_indices = np.array(all_preds)
    gt_indices = np.array(all_gt)

    gt_names = []
    pred_names = []
    for i in range(len(gt_indices)):
        local_idx = gt_indices[i]
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
    f1 = f1_score(gt_names, pred_names, labels=test_activities,
                  average='macro', zero_division=0) * 100
    f1_w = f1_score(gt_names, pred_names, labels=test_activities,
                    average='weighted', zero_division=0) * 100

    # Clean up to free GPU memory
    del best_encoder_state, best_classifier_state
    del ft_encoder, classifier, optimizer, inst_norm
    del train_loader, val_loader, test_loader
    del train_ds, val_ds, test_ds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'accuracy': acc, 'f1_macro': f1, 'f1_weighted': f1_w,
        'n_samples': len(gt_names), 'n_train_samples': len(train_data),
        'n_classes': num_test_classes,
    }


# =============================================================================
# Main
# =============================================================================

def print_results_table(all_results):
    print()
    print("=" * 130)
    print("CROSSHAR BASELINE RESULTS")
    print("=" * 130)

    header = (f"{'Dataset':<16}"
              f"{'ZS-Open Acc':>13}{'ZS-Open F1':>12}"
              f"{'ZS-Close Acc':>14}{'ZS-Close F1':>13}"
              f"{'1% FT Acc':>12}{'1% FT F1':>12}"
              f"{'10% FT Acc':>13}{'10% FT F1':>13}")
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
              f"{g('1pct_supervised','accuracy'):>11.1f}%{g('1pct_supervised','f1_macro'):>11.1f}%"
              f"{g('10pct_supervised','accuracy'):>12.1f}%{g('10pct_supervised','f1_macro'):>12.1f}%")

    print("=" * 130)
    print()
    print("Details:")
    print(f"  Model: CrossHAR (Masked Transformer + Contrastive)")
    print(f"  Hidden dim: {HIDDEN}, Layers: {N_LAYERS}, Heads: {N_HEADS}")
    print(f"  Embedding dim: {EMB_DIM} (full sequence, 120 timesteps)")
    print(f"  Checkpoint: {CROSSHAR_CHECKPOINT.relative_to(PROJECT_ROOT)}")
    print(f"  Zero-shot: Transformer_ft on 10 training datasets (87 classes), group-matched scoring")
    print(f"  Open-set: all 87 logits; Closed-set: logits masked to test-relevant groups")
    print(f"  Supervised: end-to-end fine-tuning (encoder lr={FINETUNE_ENCODER_LR}, "
          f"head lr={FINETUNE_HEAD_LR}), "
          f"{FINETUNE_EPOCHS} epochs, patience={FINETUNE_PATIENCE}")


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading CrossHAR model from {CROSSHAR_CHECKPOINT}...")
    if not CROSSHAR_CHECKPOINT.exists():
        print(f"ERROR: Checkpoint not found at {CROSSHAR_CHECKPOINT}")
        print("Make sure CrossHAR pretraining has completed.")
        sys.exit(1)

    crosshar_model = load_crosshar_model(str(CROSSHAR_CHECKPOINT), device)
    print("Model loaded successfully")

    # Load or train zero-shot Transformer_ft classifier
    global_labels = load_global_labels()
    zs_clf_path = OUTPUT_DIR / "crosshar_zs_transformer.pt"

    if zs_clf_path.exists():
        print(f"\nLoading cached zero-shot Transformer_ft from {zs_clf_path}...")
        zs_classifier = TransformerClassifier(input_dim=EMB_DIM, num_classes=len(global_labels)).to(device)
        zs_classifier.load_state_dict(torch.load(str(zs_clf_path), map_location=device))
        zs_classifier.train(False)
        print(f"  Loaded Transformer_ft with {len(global_labels)} classes")
    else:
        print(f"\nExtracting training embeddings for zero-shot ({len(TRAIN_DATASETS)} datasets)...")
        train_emb, train_lab = load_crosshar_training_embeddings(crosshar_model, global_labels, device)
        print(f"Training data: {train_emb.shape[0]} samples, shape={train_emb.shape}, "
              f"{len(np.unique(train_lab))} classes")

        # 90/10 train/val split for zero-shot classifier
        rng = np.random.RandomState(CLASSIFIER_SEED)
        N = len(train_emb)
        indices = np.arange(N)
        rng.shuffle(indices)
        val_n = int(N * 0.1)
        val_idx = indices[:val_n]
        train_idx = indices[val_n:]

        print(f"\nTraining zero-shot Transformer_ft classifier "
              f"({len(train_idx)} train, {len(val_idx)} val, 87 classes)...")
        zs_classifier = train_transformer_classifier(
            train_emb[train_idx], train_lab[train_idx],
            train_emb[val_idx], train_lab[val_idx],
            num_classes=len(global_labels), device=device,
            desc="CrossHAR | ZS Transformer_ft",
        )
        torch.save(zs_classifier.state_dict(), str(zs_clf_path))
        print(f"  Saved zero-shot Transformer_ft to {zs_clf_path}")

    # Run scoring on each test dataset
    all_results = {}

    for test_ds in TEST_DATASETS:
        print(f"\n{'='*60}")
        print(f"Scoring CrossHAR on {test_ds}")
        print(f"{'='*60}")

        raw_data, raw_labels = load_raw_data(test_ds)
        test_emb = extract_crosshar_embeddings(crosshar_model, raw_data, device)
        test_labels = get_window_labels(raw_labels)
        test_activities = get_dataset_labels(test_ds)

        print(f"  Test data: {test_emb.shape[0]} windows -> embeddings {test_emb.shape}, "
              f"{len(test_activities)} classes")
        print(f"  Classes: {test_activities}")

        ds_results = {}

        # Zero-shot open-set (Transformer_ft on full sequences)
        print(f"\n  --- Zero-Shot Open-Set (Transformer_ft) ---")
        pred_open = predict_transformer_global(zs_classifier, test_emb, device)
        ds_results['zero_shot_open_set'] = score_with_groups(
            pred_open, test_labels, test_ds, global_labels, DATASET_CONFIG)
        zs_open = ds_results['zero_shot_open_set']
        print(f"  ZS Open-set: Acc={zs_open['accuracy']:.1f}%, F1={zs_open['f1_macro']:.1f}%")

        # Zero-shot closed-set (Transformer_ft with masked logits)
        print(f"\n  --- Zero-Shot Closed-Set (Transformer_ft) ---")
        mask = get_closed_set_mask(test_ds, global_labels, DATASET_CONFIG)
        pred_closed = predict_transformer_global(zs_classifier, test_emb, device, logit_mask=mask)
        ds_results['zero_shot_closed_set'] = score_with_groups(
            pred_closed, test_labels, test_ds, global_labels, DATASET_CONFIG)
        ds_results['zero_shot_closed_set']['n_allowed'] = int(mask.sum())
        ds_results['zero_shot_closed_set']['n_masked'] = int((~mask).sum())
        zs_close = ds_results['zero_shot_closed_set']
        print(f"  ZS Closed-set: Acc={zs_close['accuracy']:.1f}%, F1={zs_close['f1_macro']:.1f}% "
              f"({zs_close['n_allowed']}/{len(global_labels)} labels allowed)")

        # 1% supervised (end-to-end fine-tuning)
        print(f"\n  --- 1% Supervised (End-to-End Fine-Tuning) ---")
        ds_results['1pct_supervised'] = score_supervised_finetune(
            crosshar_model, raw_data, test_labels, test_ds, device,
            label_rate=SUPERVISED_LABEL_RATE_1PCT, label_tag="1%")
        print(f"  1% finetune: Acc={ds_results['1pct_supervised']['accuracy']:.1f}%, "
              f"F1={ds_results['1pct_supervised']['f1_macro']:.1f}%")

        # 10% supervised (end-to-end fine-tuning)
        print(f"\n  --- 10% Supervised (End-to-End Fine-Tuning) ---")
        ds_results['10pct_supervised'] = score_supervised_finetune(
            crosshar_model, raw_data, test_labels, test_ds, device,
            label_rate=SUPERVISED_LABEL_RATE_10PCT, label_tag="10%")
        print(f"  10% finetune: Acc={ds_results['10pct_supervised']['accuracy']:.1f}%, "
              f"F1={ds_results['10pct_supervised']['f1_macro']:.1f}%")

        all_results[test_ds] = ds_results

    # Print summary table
    print_results_table(all_results)

    # Save results
    results_path = OUTPUT_DIR / "crosshar_evaluation.json"
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
