# eval_actionsense_cls.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

# ------------------- Config (HARD-CODED to match training) -------------------
class Config:
    # Context & model dims
    context_size = 20
    llama_dim = 512   # semantic dim (F)

    # Training (used here for eval dataloader parity)
    batch_size = 8
    num_workers = 8
    epochs = 50
    grad_clip = None
    lr = 2e-4
    weight_decay = 0.05
    dropout = 0.1
    nhead = 8
    num_layers = 4
    mlp_ratio = 4.0

    # Plot cadence (not relevant for eval)
    LOSS_PLOT_EVERY = 10

CFG = Config()

# --------- Eval-only constants (also hard-coded) ---------
CKPT_PATH = "checkpoints/best/cls_epoch_30.pt"  # change if evaluating a different checkpoint
OUT_DIR = os.path.join("debug", "eval", "actionsense_cls")
VAL_RATIO = 0.20
SPLIT_SEED = 42
VISUALIZE_BATCHES = 20                      # visualize first N val batches
MAX_PATCHES_PER_VIS = 8                    # uniformly sample this many patches for heatmaps
ANNOTATE_LOGITS = False

# ------------------- Project imports -------------------
from datasets.converters.ActionSenseConverter import ActionSenseConverter
from datasets.BaseDataset import tsfm_collate
from datasets.ActionSensePretrainingDatasets import ActionSenseActivityClsDataset

from encoder.TSFMEncoder import TSFMEncoder
from pretraining.actionsense.heads import ActivityCLSHead
from training_utils import configure_device_and_amp, count_params

# ------------------- Helpers -------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def uniform_indices(n: int, k: int):
    if n <= 0:
        return []
    k = max(1, min(k, n))
    return np.linspace(0, n - 1, num=k, dtype=int).tolist()

def tensor_to_numpy(x: torch.Tensor):
    return x.detach().float().cpu().numpy()

def add_patch_separators(ax, patch_rows):
    for r in patch_rows:
        ax.axhline(r - 0.5, linewidth=0.6, color="white", alpha=0.8)

def _heatmap(matrix: np.ndarray, title: str, out_path: str,
             xlabel="F (feature dim)", ylabel="Tokens (P*D)", separators=None):
    ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=(10, 6))
    im = plt.imshow(matrix, aspect="auto", interpolation="nearest")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if separators:
        add_patch_separators(plt.gca(), separators)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_feature_heatmap(features_b: torch.Tensor, out_path: str,
                         max_patches: int = MAX_PATCHES_PER_VIS, do_layernorm_for_plot: bool = True):
    """
    features_b: (P, D, F)
    """
    P, D, Fdim = features_b.shape
    sel_p = uniform_indices(P, max_patches)
    x = features_b[sel_p]  # (K, D, F)
    if do_layernorm_for_plot:
        x = F.layer_norm(x, normalized_shape=(Fdim,))
    x2 = x.reshape(-1, Fdim)  # (K*D, F)
    separators = [i * D for i in range(1, len(sel_p))]
    _heatmap(tensor_to_numpy(x2), f"Feature Heatmap (P={P},D={D},F={Fdim}) | {len(sel_p)} patches",
             out_path, separators=separators)

def plot_token_heatmap(tokens_b: torch.Tensor, out_path: str,
                       max_patches: int = MAX_PATCHES_PER_VIS, do_layernorm_for_plot: bool = True):
    """
    tokens_b: (P, D, F)
    """
    P, D, Fdim = tokens_b.shape
    sel_p = uniform_indices(P, max_patches)
    x = tokens_b[sel_p]
    if do_layernorm_for_plot:
        x = F.layer_norm(x, normalized_shape=(Fdim,))
    x2 = x.reshape(-1, Fdim)
    separators = [i * D for i in range(1, len(sel_p))]
    _heatmap(tensor_to_numpy(x2), f"Token Heatmap (P={P},D={D},F={Fdim}) | {len(sel_p)} patches",
             out_path, separators=separators)

# ------------------- Data builders -------------------
def build_dataloaders_for_eval(device: torch.device):
    """
    Build consolidated episodes once, then create train/val datasets (episode-level split).
    We'll use the checkpoint's class list as canonical; train_ds is still useful for debugging.
    """
    converter = ActionSenseConverter()
    episodes, metadata = converter.convert()

    train_ds = ActionSenseActivityClsDataset(
        episodes, metadata,
        context_size=CFG.context_size,
        debug=True,
        split="train", val_ratio=VAL_RATIO, split_seed=SPLIT_SEED,
        store_episode_stats=False,
        allow_unlabeled=False,
    )
    val_ds = ActionSenseActivityClsDataset(
        episodes, metadata,
        context_size=CFG.context_size,
        debug=True,
        split="val", val_ratio=VAL_RATIO, split_seed=SPLIT_SEED,
        store_episode_stats=False,
        allow_unlabeled=False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=tsfm_collate,
    )
    return val_loader, train_ds, val_ds

# ------------------- Model builders (checkpoint-aligned head) -------------------
def _infer_head_shape_from_state(head_state: dict, d_model_expected: int):
    """
    Infer hidden dim and num_classes from a saved ActivityCLSHead state_dict.
    We expect keys like:
      classifier.1.weight -> (hidden, d_model)
      classifier.4.weight -> (num_classes, hidden)
    """
    # Find the first Linear after LayerNorm in the classifier (usually index 1)
    h_w = None
    c_w = None
    for k, v in head_state.items():
        if k.endswith("classifier.1.weight"):
            h_w = v  # [hidden, d_model]
        elif k.endswith("classifier.4.weight"):
            c_w = v  # [num_classes, hidden]
    if h_w is None or c_w is None:
        # Fall back to scanning any linear weights
        for k, v in head_state.items():
            if k.startswith("classifier.") and k.endswith(".weight") and v.ndim == 2:
                out_f, in_f = v.shape
                # Heuristic: the one whose in_features == d_model is the hidden layer
                if in_f == d_model_expected and h_w is None:
                    h_w = v
                # The one whose in_features == hidden will be class layer, grab last seen
                if c_w is None and h_w is not None and in_f == h_w.shape[0]:
                    c_w = v
    if h_w is None or c_w is None:
        raise RuntimeError("Unable to infer head architecture from checkpoint (missing classifier weights).")
    hidden = int(h_w.shape[0])
    d_model_in = int(h_w.shape[1])
    num_classes = int(c_w.shape[0])
    if d_model_in != d_model_expected:
        # Warn but proceed—encoder proj size must equal checkpoint's classifier in_features
        print(f"[WARN] Head d_model_in ({d_model_in}) != expected ({d_model_expected}). Using {d_model_in}.")
    mlp_hidden_ratio = hidden / float(d_model_in)
    return hidden, num_classes, d_model_in, mlp_hidden_ratio

def build_models_from_ckpt(device: torch.device):
    """
    Rebuild encoder from hard-coded CFG, and rebuild head to EXACTLY match the checkpoint
    (hidden size and num_classes), then load weights strict=True.
    Also returns any checkpoint class names if available.
    """
    print(f"[LOAD] {CKPT_PATH}")
    state = torch.load(CKPT_PATH, map_location=device)

    # ---- Processors (match training stack) ----
    from encoder.processors.CorrelationSummaryProcessor import CorrelationSummaryProcessor
    from encoder.processors.FrequencyFeatureProcessor import FrequencyFeatureProcessor
    from encoder.processors.HistogramFeatureProcessor import HistogramFeatureProcessor
    from encoder.processors.StatisticalFeatureProcessor import StatisticalFeatureProcessor
    processors = [
        CorrelationSummaryProcessor(),
        FrequencyFeatureProcessor(),
        HistogramFeatureProcessor(),
        StatisticalFeatureProcessor()
    ]

    # ---- Encoder (hard-coded dims from CFG) ----
    encoder = TSFMEncoder(
        processors=processors,
        feature_dim=CFG.llama_dim,
        encoding_dim=CFG.llama_dim,
        num_layers=CFG.num_layers,
        nhead=CFG.nhead,
        dropout=CFG.dropout,
        mlp_ratio=CFG.mlp_ratio,
        pretraining_args=None,
        recon_head=None,
    ).to(device)
    print(f"[DEBUG] Encoder params: {count_params(encoder):.2f}M")
    encoder.load_state_dict(state["encoder"], strict=True)

    # ---- Head: reconstruct to match checkpoint ----
    head_state = state["head"]
    hidden, num_classes, d_model_in_ckpt, mlp_hidden_ratio = _infer_head_shape_from_state(
        head_state, CFG.llama_dim
    )
    # Build head that matches checkpoint exactly
    head = ActivityCLSHead(
        d_model=d_model_in_ckpt,
        nhead=CFG.nhead,
        num_classes=num_classes,
        dropout=CFG.dropout,
        mlp_hidden_ratio=mlp_hidden_ratio,
    ).to(device)
    head.load_state_dict(head_state, strict=True)

    # Optional: class names saved in training checkpoints (your trainer adds these on periodic saves)
    ckpt_id_to_activity = state.get("id_to_activity", None)  # list[str] if present
    ckpt_activity_to_id = state.get("activity_to_id", None)  # dict[str,int] if present

    return encoder, head, state, ckpt_id_to_activity, ckpt_activity_to_id

# ------------------- Evaluation Core -------------------
def print_class_index_legend(class_names: list[str] | None):
    """Print index -> label mapping once, in the same order as logits/classes."""
    if not class_names:
        print("[LEGEND] No class names available.")
        return
    print("\n[LEGEND] Class index → label (checkpoint order)")
    for i, name in enumerate(class_names):
        print(f"  {i:3d} → {name}")
    print("")

def print_sample_topk(logits_row: torch.Tensor, class_names: list[str] | None,
                      gt_idx: int | None = None, topk: int = 5):
    """Pretty-print GT and top-k predictions with probabilities for a single sample."""
    probs = torch.softmax(logits_row.float(), dim=-1)
    k = min(int(topk), probs.numel())
    vals, idxs = torch.topk(probs, k=k, dim=-1)
    def name_of(i: int) -> str:
        if class_names and 0 <= i < len(class_names):
            return class_names[i]
        return str(i)
    if gt_idx is not None and gt_idx >= 0:
        print(f"    GT: {gt_idx} :: {name_of(gt_idx)}")
    print("    Top-k predictions:")
    for rank in range(k):
        i = int(idxs[rank].item())
        p = float(vals[rank].item())
        print(f"      #{rank+1}: {i:3d} :: {name_of(i)}  (p={p:.3f})")

@torch.no_grad()
def run_validation_with_visuals(
    encoder: TSFMEncoder,
    head: ActivityCLSHead,
    val_loader: torch.utils.data.DataLoader,
    val_ds: ActionSenseActivityClsDataset,
    device: torch.device,
    out_dir: str,
    ckpt_id_to_activity=None,
    visualize_batches: int = VISUALIZE_BATCHES,
    max_patches_per_vis: int = MAX_PATCHES_PER_VIS,
    annotate_logits: bool = ANNOTATE_LOGITS
):
    """
    Use CKPT class list as canonical for metrics *and* plotting if available.
    Remap: val label ids -> ckpt label ids (by name).
    Also prints a legend (index->label) once and, for each visualized sample,
    prints GT and top-k predictions to the terminal.
    """
    encoder.eval(); head.eval()
    ce = nn.CrossEntropyLoss()

    # --- Build val->ckpt remap by name ---
    if ckpt_id_to_activity is not None and len(ckpt_id_to_activity) > 0:
        ckpt_name_to_id = {name: i for i, name in enumerate(ckpt_id_to_activity)}
        val_id_to_name = {i: name for i, name in enumerate(val_ds.id_to_activity)}
        remap_vec = torch.full((len(val_id_to_name),), -1, dtype=torch.long, device=device)
        for vid, vname in val_id_to_name.items():
            if vname in ckpt_name_to_id:
                remap_vec[vid] = ckpt_name_to_id[vname]
        class_names_for_plot = ckpt_id_to_activity
    else:
        print("[WARN] No class names found in checkpoint; assuming identical label mapping.")
        remap_vec = None
        class_names_for_plot = getattr(val_ds, "id_to_activity", None)

    # ---- Print the legend once so long labels are visible in terminal ----
    print_class_index_legend(class_names_for_plot)

    # Output dirs
    dir_logits   = ensure_dir(os.path.join(out_dir, "logits"))
    dir_features = ensure_dir(os.path.join(out_dir, "features"))
    dir_tokens   = ensure_dir(os.path.join(out_dir, "tokens"))

    total_loss = 0.0
    total_n = 0
    correct = 0
    rows = []

    with tqdm(total=len(val_loader), desc="[EVAL] Validation", dynamic_ncols=True) as pbar:
        for b_idx, batch in enumerate(val_loader):
            if len(batch) == 0:
                pbar.update(1); continue

            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}

            out = encoder.encode_batch(batch)     # adds "features"; returns "tokens"
            long_tokens = out["tokens"]           # (B,P,D,F)
            features    = out["features"]         # (B,P,D,F)

            # Build flattened key_padding_mask (B,P*D), True=pad
            flattened_key_padding_mask = None
            if "pad_mask" in batch and batch["pad_mask"] is not None:
                Bsz, P = batch["pad_mask"].shape
                D = long_tokens.shape[2]
                valid_flat = batch["pad_mask"].unsqueeze(-1).expand(Bsz, P, D).reshape(Bsz, P * D)
                flattened_key_padding_mask = ~valid_flat

            logits = head(long_tokens, flattened_key_padding_mask)  # (B,C_ckpt)
            targets_val = batch.get("activity_id", None)
            if targets_val is None:
                targets_val = torch.full((logits.size(0),), -1, dtype=torch.long, device=device)

            # ----- Metrics (use remapped labels if available) -----
            if remap_vec is not None:
                mapped = remap_vec[targets_val.clamp(min=0)]   # (B,), -1 for unmapped
                valid_mask = mapped.ge(0)
                if valid_mask.any():
                    sel_logits = logits[valid_mask]
                    sel_targets = mapped[valid_mask]
                    loss = ce(sel_logits, sel_targets)
                    total_loss += loss.item() * sel_targets.numel()
                    total_n += sel_targets.numel()
                    pred = sel_logits.argmax(dim=-1)
                    correct += (pred == sel_targets).sum().item()
            else:
                loss = ce(logits, targets_val)
                total_loss += loss.item() * targets_val.numel()
                total_n += targets_val.numel()
                pred = logits.argmax(dim=-1)
                correct += (pred == targets_val).sum().item()

            # ----- Visualizations (pick a *valid* sample and plot with mapped GT) -----
            if b_idx < visualize_batches:
                if remap_vec is not None:
                    if 'valid_mask' in locals() and valid_mask.any():
                        s = int(valid_mask.nonzero(as_tuple=True)[0][0])
                        gt_for_plot = int(mapped[s].item())
                    else:
                        s = 0
                        gt_for_plot = int(mapped[0].clamp(min=0).item())
                    targets_for_plot = mapped.clamp(min=0)
                else:
                    s = 0
                    gt_for_plot = int(targets_val[s].clamp(min=0).item())
                    targets_for_plot = targets_val.clamp(min=0)

                # ---- Terminal print: GT + top-k predictions (labels fully printed here) ----
                print(f"\n[PLOT] batch={b_idx} sample={s}")
                print_sample_topk(
                    logits_row=logits[s],
                    class_names=class_names_for_plot,
                    gt_idx=gt_for_plot,
                    topk=5
                )

                # ---- Plot images as before ----
                head.debug_logits_bar(
                    logits, targets_for_plot,
                    b_idx=s, class_names=class_names_for_plot,
                    save_path=os.path.join(dir_logits, f"val_b{b_idx:04d}_s{s}.png"),
                    annotate_values=annotate_logits
                )
                plot_feature_heatmap(
                    features_b=features[s],  # (P,D,F)
                    out_path=os.path.join(dir_features, f"val_b{b_idx:04d}_s{s}.png"),
                    max_patches=max_patches_per_vis,
                    do_layernorm_for_plot=True
                )
                plot_token_heatmap(
                    tokens_b=long_tokens[s],  # (P,D,F)
                    out_path=os.path.join(dir_tokens, f"val_b{b_idx:04d}_s{s}.png"),
                    max_patches=max_patches_per_vis,
                    do_layernorm_for_plot=True
                )

            rows.append({
                "batch_idx": b_idx,
                "num_items": logits.size(0),
                "used_for_metrics": int(total_n),  # cumulative count
            })

            pbar.update(1)

    avg_loss = total_loss / max(1, total_n)
    acc = correct / max(1, total_n)
    print(f"\n[VAL] avg_loss={avg_loss:.6f}  acc={acc:.4f}  (N={total_n} remappable items)")

    # Save metrics CSV
    df = pd.DataFrame(rows)
    df["final_avg_loss"] = avg_loss
    df["final_acc"] = acc
    df_path = os.path.join(out_dir, "val_summary.csv")
    df.to_csv(df_path, index=False)
    print(f"[OUT] Wrote metrics → {df_path}")

    return {"val_loss": avg_loss, "val_acc": acc, "N": total_n}



# ------------------- Main -------------------
if __name__ == "__main__":
    device, _, _ = configure_device_and_amp()  # same device logic as training

    # Data
    val_loader, train_ds, val_ds = build_dataloaders_for_eval(device)

    # Models (head reconstructed to match checkpoint EXACTLY)
    encoder, head, state, ckpt_id_to_activity, ckpt_activity_to_id = build_models_from_ckpt(device)

    ensure_dir(OUT_DIR)
    _ = run_validation_with_visuals(
        encoder=encoder,
        head=head,
        val_loader=val_loader,
        val_ds=val_ds,
        device=device,
        out_dir=OUT_DIR,
        ckpt_id_to_activity=ckpt_id_to_activity,
        visualize_batches=VISUALIZE_BATCHES,
        max_patches_per_vis=MAX_PATCHES_PER_VIS,
        annotate_logits=ANNOTATE_LOGITS
    )
