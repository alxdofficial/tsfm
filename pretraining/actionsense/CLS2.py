# pretrain_actionsense_cls.py  (visualization-only; loads latest checkpoint automatically)
import os
import re
import glob
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Data & Collate
from datasets.converters.ActionSenseConverter import ActionSenseConverter
from datasets.BaseDataset import tsfm_collate
from datasets.ActionSensePretrainingDatasets import ActionSenseActivityClsDataset

# Encoder & Head
from encoder.TSFMEncoder import TSFMEncoder
from pretraining.actionsense.cls_head import ActivityCLSHead  # expects (long_tokens, key_padding_mask) -> logits

# Non-learnable feature processors
from encoder.processors.CorrelationSummaryProcessor import CorrelationSummaryProcessor
from encoder.processors.FrequencyFeatureProcessor import FrequencyFeatureProcessor
from encoder.processors.HistogramFeatureProcessor import HistogramFeatureProcessor
from encoder.processors.StatisticalFeatureProcessor import StatisticalFeatureProcessor

# Debug
from pretraining.actionsense.debug_stats import DebugStats
dbg = DebugStats(out_dir="debug")

# Training utils (only used to pick device/amp)
from training_utils import (
    configure_device_and_amp,
    count_params,
)

# ------------------- Config -------------------
class Config:
    # Context & model dims
    context_size = 20
    llama_dim = 512   # semantic dim (F)

    # Dataloader (we'll slice to B=1 downstream if needed)
    batch_size = 8
    num_workers = 8

    # Encoder arch
    dropout = 0.1
    nhead = 8
    num_layers = 4
    mlp_ratio = 4.0

    # Checkpoint directory/pattern
    ckpt_dir = "checkpoints"
    ckpt_prefix = "cls_epoch_25"  # files like "cls_epoch_25.pt"

CFG = Config()

# ------------------- Builders -------------------
def build_dataloader(device: torch.device):
    converter = ActionSenseConverter()  # default patch_size=96
    episodes, metadata = converter.convert()

    dataset = ActionSenseActivityClsDataset(
        episodes, metadata,
        context_size=CFG.context_size,
        debug=True,
        store_episode_stats=False,
        allow_unlabeled=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=tsfm_collate,
    )
    return dataloader, metadata, dataset


def build_processors():
    return [
        CorrelationSummaryProcessor(),
        FrequencyFeatureProcessor(),
        HistogramFeatureProcessor(),
        StatisticalFeatureProcessor()
    ]


def build_encoder(processors, device: torch.device) -> nn.Module:
    encoder = TSFMEncoder(
        processors=processors,
        feature_dim=CFG.llama_dim,
        encoding_dim=CFG.llama_dim,
        num_layers=CFG.num_layers,
        nhead=CFG.nhead,
        dropout=CFG.dropout,
        mlp_ratio=CFG.mlp_ratio,
        pretraining_args=None,   # not used in classification
        recon_head=None,
    ).to(device)
    print(f"[DEBUG] Encoder params: {count_params(encoder):.2f}M")
    return encoder


def build_activity_head(num_classes: int, device: torch.device) -> nn.Module:
    head = ActivityCLSHead(
        d_model=CFG.llama_dim,
        nhead=CFG.nhead,
        num_classes=num_classes,
        dropout=CFG.dropout,
        mlp_hidden_ratio=4.0,
    ).to(device)
    print(f"[INIT] ActivityCLSHead(C={num_classes}) params={count_params(head):.2f}M")
    return head


# ------------------- Checkpoint helpers -------------------
def _find_latest_ckpt(ckpt_dir: str, prefix: str) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    paths = glob.glob(os.path.join(ckpt_dir, f"{prefix}*.pt"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir} matching {prefix}*.pt")

    # Prefer the highest epoch parsed from filename; fallback to mtime
    def parse_epoch(p):
        m = re.search(rf"{re.escape(prefix)}(\d+)\.pt$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    paths_with_epoch = [(p, parse_epoch(p)) for p in paths]
    paths_with_epoch.sort(key=lambda x: (x[1], os.path.getmtime(x[0])))
    latest = paths_with_epoch[-1][0]
    print(f"[CKPT] Using checkpoint: {latest}")
    return latest


def _apply_cfg_from_ckpt(ckpt_cfg: dict):
    if not isinstance(ckpt_cfg, dict):
        return
    # Only override fields that exist in CFG
    overridable = ["llama_dim", "dropout", "nhead", "num_layers", "mlp_ratio", "context_size"]
    for k in overridable:
        if k in ckpt_cfg:
            setattr(CFG, k, ckpt_cfg[k])
    # (batch_size/num_workers often runtime-specific; we keep current values)


def _infer_num_classes_from_head_state(head_state: dict) -> int | None:
    # Our classifier is Sequential with final Linear at index 4 -> keys 'classifier.4.weight'/'bias'
    # Fall back to search for the last '.weight' whose shape[0] is likely num_classes.
    for key in ["classifier.4.bias", "classifier.4.weight"]:
        if key in head_state:
            t = head_state[key]
            try:
                return int(t.shape[0])
            except Exception:
                pass
    # Generic fallback: find any 'classifier.*.bias' with 1D shape
    cand = [k for k in head_state.keys() if k.startswith("classifier.") and k.endswith(".bias")]
    for k in sorted(cand):
        t = head_state[k]
        if t.ndim == 1:
            return int(t.shape[0])
    return None


# ------------------- Visualization helpers -------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _first_valid_patch_indices(pad_mask_row: torch.Tensor, k: int) -> np.ndarray:
    valid_idx = pad_mask_row.nonzero(as_tuple=False).squeeze(-1).cpu().numpy()
    if valid_idx.size == 0:
        return np.array([], dtype=np.int64)
    return valid_idx[:min(k, valid_idx.size)]

def _uniform_sample_valid_indices(pad_mask_row: torch.Tensor, k: int) -> np.ndarray:
    """Uniformly sample up to k valid patch indices across the sequence."""
    valid_idx = pad_mask_row.nonzero(as_tuple=False).squeeze(-1).cpu().numpy()
    if valid_idx.size == 0:
        return np.array([], dtype=np.int64)
    if valid_idx.size <= k:
        return valid_idx
    lin = np.linspace(0, valid_idx.size - 1, num=k, dtype=np.int64)
    return valid_idx[lin]

def _plot_raw_timeseries(patches_1: torch.Tensor, sel_idx: np.ndarray, out_dir: str):
    _ensure_dir(out_dir)
    P, T, D = patches_1.shape
    n = len(sel_idx)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5*n), sharex=True)
    if n == 1: axes = [axes]
    time_axis = np.arange(T)
    for ax, p in zip(axes, sel_idx):
        X = patches_1[p].cpu().numpy()  # (T,D)
        ax.plot(time_axis, X)  # D lines
        ax.set_title(f"Raw time-series — patch {int(p)} (D={D})")
        ax.set_ylabel("value")
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[-1].set_xlabel("t (within patch)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "raw_timeseries_first_patches.png"))
    plt.close(fig)

def _plot_feature_heatmaps(encoder: TSFMEncoder, patches_1: torch.Tensor, sel_idx: np.ndarray, out_dir: str, apply_layernorm: bool = True):
    """
    Plot per-channel features (D x K) for selected patches.
    If apply_layernorm=True, apply the exact per-token LayerNorm over K that encode_batch() uses.
    """
    _ensure_dir(out_dir)
    with torch.no_grad():
        P, T, D = patches_1.shape
        feats = encoder._extract_features(patches_1.unsqueeze(0))  # (1,P,D,K)
        feats = feats.squeeze(0)  # (P,D,K)

        if apply_layernorm:
            K = feats.size(-1)
            feats = F.layer_norm(feats, normalized_shape=(K,))  # (P,D,K)

        K = feats.shape[-1]
        for p in sel_idx:
            mat = feats[p].cpu().numpy()  # (D,K)
            fig = plt.figure(figsize=(10, 4))
            plt.imshow(mat, aspect="auto")
            plt.title(f"Feature heatmap — patch {int(p)} (D={D}, K={K})" + (" [LayerNorm]" if apply_layernorm else " [raw]"))
            plt.xlabel("feature dim (K)")
            plt.ylabel("channel (D)")
            plt.colorbar()
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, f"features_patch_{int(p)}.png"))
            plt.close(fig)

def _plot_token_heatmaps(tokens_1: torch.Tensor, sel_idx: np.ndarray, out_dir: str):
    _ensure_dir(out_dir)
    P, D, Fdim = tokens_1.shape
    for p in sel_idx:
        mat = tokens_1[p].detach().cpu().numpy()  # (D,F)
        fig = plt.figure(figsize=(10, 4))
        plt.imshow(mat, aspect="auto")
        plt.title(f"Token heatmap — patch {int(p)} (D={D}, F={Fdim})")
        plt.xlabel("hidden dim (F)")
        plt.ylabel("channel (D)")
        plt.colorbar()
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"tokens_patch_{int(p)}.png"))
        plt.close(fig)

def _logits_bar_fallback(logits: torch.Tensor, targets: torch.Tensor, class_names=None, save_path="debug/vis_logits_bar.png"):
    _ensure_dir(os.path.dirname(save_path))
    logits_0 = logits[0].detach().cpu().numpy()
    t0 = int(targets[0].item())
    C = logits_0.shape[0]
    xs = np.arange(C)
    fig = plt.figure(figsize=(10, 3))
    plt.bar(xs, logits_0)
    if class_names is not None and len(class_names) == C:
        plt.xticks(xs, class_names, rotation=45, ha="right")
    else:
        plt.xticks(xs, [str(i) for i in xs])
    title_gt = class_names[t0] if class_names and (0 <= t0 < len(class_names)) else str(t0)
    plt.title(f"Logits (b=0), GT={title_gt}")
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

# ------------------- Visualization run -------------------
def visualize_one_batch():
    device, amp_ctx, scaler = configure_device_and_amp()

    # ---- Load checkpoint first (to configure model dims properly) ----
    ckpt_path = _find_latest_ckpt(CFG.ckpt_dir, CFG.ckpt_prefix)
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_cfg = ckpt.get("cfg", {})
    _apply_cfg_from_ckpt(ckpt_cfg)

    # ---- Build data and models consistent with ckpt ----
    dataloader, _, dataset = build_dataloader(device)
    processors = build_processors()

    # Infer num_classes from checkpoint head if possible
    num_classes_ckpt = None
    head_state = ckpt.get("head", {})
    if isinstance(head_state, dict) and head_state:
        num_classes_ckpt = _infer_num_classes_from_head_state(head_state)
    num_classes = num_classes_ckpt if (num_classes_ckpt is not None) else dataset.num_classes

    encoder = build_encoder(processors, device)
    head = build_activity_head(num_classes=num_classes, device=device)

    # Load weights (strict=False for robustness across minor code diffs)
    missing, unexpected = encoder.load_state_dict(ckpt.get("encoder", {}), strict=False)
    if missing or unexpected:
        print(f"[CKPT][encoder] missing={len(missing)} unexpected={len(unexpected)}")

    missing, unexpected = head.load_state_dict(head_state, strict=False)
    if missing or unexpected:
        print(f"[CKPT][head] missing={len(missing)} unexpected={len(unexpected)}")

    encoder.eval(); head.eval()

    # === pull one batch ===
    batch = next(iter(dataloader))
    # Slice to B=1 (left-pad friendly)
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor) and v.dim() > 0 and v.size(0) > 1:
            batch[k] = v[:1]
        elif isinstance(v, dict):
            for kk, vv in list(v.items()):
                if isinstance(vv, torch.Tensor) and vv.dim() > 0 and vv.size(0) > 1:
                    v[kk] = vv[:1]
            batch[k] = v
        elif isinstance(v, list):
            batch[k] = v[:1]

    # Move tensors to device
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    # === decide which patches to visualize (8 uniformly sampled valid) ===
    P_total = batch["patches"].shape[1]
    D = batch["patches"].shape[3]
    pad_mask_row = batch.get("pad_mask", torch.ones(1, P_total, dtype=torch.bool, device=device))[0]
    sel_idx = _uniform_sample_valid_indices(pad_mask_row, k=8)
    if sel_idx.size == 0:
        print("[VIS] No valid patches to visualize.")
        return

    # === 1) Raw time-series (all channels) for selected patches ===
    raw_dir = "debug/vis_raw"
    _plot_raw_timeseries(batch["patches"][0].detach().cpu(), sel_idx, raw_dir)

    # === 2) Feature heatmaps (D x K) for those patches — with LayerNorm applied ===
    feat_dir = "debug/vis_features"
    _plot_feature_heatmaps(encoder, batch["patches"][0], sel_idx, feat_dir, apply_layernorm=True)

    # === 3) Run encoder to get tokens and plot token heatmaps (D x F) ===
    with torch.no_grad():
        out = encoder.encode_batch(batch)     # adds "features" and "tokens" into batch
        tokens = out["tokens"]                # (B,P,D,F)
    tok_dir = "debug/vis_tokens"
    _plot_token_heatmaps(tokens[0], sel_idx, tok_dir)

    # === 4) Compute logits and plot as bar chart (b=0) ===
    flattened_key_padding_mask = None
    if "pad_mask" in batch and batch["pad_mask"] is not None:
        Bsz, P = batch["pad_mask"].shape
        Dch = tokens.shape[2]
        valid_flat = batch["pad_mask"].unsqueeze(-1).expand(Bsz, P, Dch).reshape(Bsz, P * Dch)
        flattened_key_padding_mask = ~valid_flat

    with torch.no_grad():
        logits = head(tokens, flattened_key_padding_mask)  # (B,C)

    cls_dir = "debug"
    if hasattr(head, "debug_logits_bar"):
        head.debug_logits_bar(
            logits, batch["activity_id"], b_idx=0,
            class_names=getattr(dataset, "id_to_activity", None),
            save_path=os.path.join(cls_dir, "vis_logits_bar.png"),
            annotate_values=False
        )
    else:
        _logits_bar_fallback(
            logits, batch["activity_id"],
            class_names=getattr(dataset, "id_to_activity", None),
            save_path=os.path.join(cls_dir, "vis_logits_bar.png"),
        )

    print("[VIS] Saved:")
    print(f"  - raw timeseries  : {raw_dir}/raw_timeseries_first_patches.png")
    print(f"  - feature heatmaps: {feat_dir}/features_patch_*.png")
    print(f"  - token heatmaps  : {tok_dir}/tokens_patch_*.png")
    print(f"  - logits bar      : {cls_dir}/vis_logits_bar.png")

# ------------------- Main ----------------
if __name__ == "__main__":
    # Visualization-only run; always loads the latest checkpoint
    visualize_one_batch()
