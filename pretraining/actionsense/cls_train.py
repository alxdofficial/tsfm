# pretrain_actionsense_cls.py
import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Optional, Tuple, Dict, Any, List

# Data & Collate
from datasets.converters.ActionSenseConverter import ActionSenseConverter
from datasets.BaseDataset import tsfm_collate
from datasets.ActionSensePretrainingDatasets import ActionSenseActivityClsDataset

# Encoder & Head
from encoder.TSFMEncoder import TSFMEncoder
from pretraining.actionsense.cls_head import ActivityCLSHead  # expects (long_tokens, key_padding_mask) -> logits

# Debug
from pretraining.actionsense.debug_stats import DebugStats
dbg = DebugStats(out_dir="debug")

# Training utils
from training_utils import (
    configure_device_and_amp,
    build_optimizer,
    build_warmup_cosine_scheduler,
    sanity_check_optimizer,
    count_params,
)

# ------------------- Config -------------------
class Config:
    # Context & model dims
    context_size = 20
    llama_dim = 512   # semantic dim (F)

    # Training
    batch_size = 32
    num_workers = 8
    epochs = 50
    grad_clip = None
    lr = 2e-4
    weight_decay = 0.05
    dropout = 0.1
    nhead = 8
    num_layers = 4
    mlp_ratio = 4.0

    # Plot cadence
    LOSS_PLOT_EVERY = 10

CFG = Config()

# ------------------- Builders (train + val) -------------------
def build_processors():
    from encoder.processors.CorrelationSummaryProcessor import CorrelationSummaryProcessor
    from encoder.processors.FrequencyFeatureProcessor import FrequencyFeatureProcessor
    from encoder.processors.HistogramFeatureProcessor import HistogramFeatureProcessor
    from encoder.processors.StatisticalFeatureProcessor import StatisticalFeatureProcessor
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


def build_dataloaders(device: torch.device):
    """
    Build train/val datasets from the same converted episode list using an episode-level split.
    We build TRAIN first (its label mapping is canonical), then VAL. At eval time we remap
    val ids to train ids by name to avoid label-mapping bugs.
    """
    converter = ActionSenseConverter()  # default patch_size=96
    episodes, metadata = converter.convert()

    train_ds = ActionSenseActivityClsDataset(
        episodes, metadata,
        context_size=CFG.context_size,
        debug=True,
        split="train", val_ratio=0.20, split_seed=42,
        store_episode_stats=False,
        allow_unlabeled=False,
    )
    val_ds = ActionSenseActivityClsDataset(
        episodes, metadata,
        context_size=CFG.context_size,
        debug=True,
        split="val", val_ratio=0.20, split_seed=42,
        store_episode_stats=False,
        allow_unlabeled=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=tsfm_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=tsfm_collate,
    )
    return train_loader, val_loader, metadata, train_ds, val_ds

# ------------------- Plotting -------------------
def plot_training_loss(loss_history, out_path="checkpoints/train_loss_cls.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(loss_history, label="Batch Loss")
    plt.xlabel("Batch #")
    plt.ylabel("Loss")
    plt.title("Training Loss (per batch) — Activity CLS")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_val_history(val_hist: List[Dict[str, float]], out_path="checkpoints/val_metrics_cls.png"):
    """
    val_hist: list of dicts with keys {"epoch", "loss", "acc"}
    """
    if not val_hist:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    epochs = [e["epoch"] for e in val_hist]
    losses = [e["loss"] for e in val_hist]
    accs   = [e["acc"]  for e in val_hist]

    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    l1, = ax1.plot(epochs, losses, marker="o", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2 = ax1.twinx()
    l2, = ax2.plot(epochs, accs, marker="s", linestyle="--", label="Val Acc", color="tab:green")
    ax2.set_ylabel("Accuracy")

    lines = [l1, l2]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="best")

    plt.title("Validation over Epochs — Activity CLS")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ------------------- Checkpoint -------------------
def save_ckpt(encoder, head, optimizer, scheduler, scaler, epoch, extra=None, out_dir="checkpoints"):
    os.makedirs(out_dir, exist_ok=True)
    state = {
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "head": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "cfg": vars(CFG),
    }
    if extra:
        state.update(extra)
    path = os.path.join(out_dir, f"cls_epoch_{epoch}.pt")
    torch.save(state, path)
    print(f"[CKPT] Saved checkpoint → {path}")

# ------------------- Evaluation (no-grad) -------------------
@torch.no_grad()
def evaluate(encoder: nn.Module, head: nn.Module, dataloader: DataLoader,
             ce: nn.Module, device: torch.device,
             id_remap: Optional[torch.Tensor]) -> Tuple[float, float]:
    """
    Returns: (avg_loss, accuracy) over valid (remappable) examples.

    id_remap: Tensor of shape (val_num_classes,), mapping val ids -> train ids.
              Unmapped classes have -1 and are ignored in metrics.
    """
    encoder.eval(); head.eval()
    total_loss = 0.0
    total_n = 0
    correct = 0

    for batch in dataloader:
        if len(batch) == 0:
            continue
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}

        if "activity_id" not in batch:
            continue

        out = encoder.encode_batch(batch)  # tokens: (B,P,D,F)
        long_tokens = out["tokens"]

        # Build flattened key_padding_mask (B,P*D), True=pad
        flattened_key_padding_mask = None
        if "pad_mask" in batch and batch["pad_mask"] is not None:
            Bsz, P = batch["pad_mask"].shape
            D = long_tokens.shape[2]
            valid_flat = batch["pad_mask"].unsqueeze(-1).expand(Bsz, P, D).reshape(Bsz, P * D)
            flattened_key_padding_mask = ~valid_flat

        logits = head(long_tokens, flattened_key_padding_mask)  # (B,C_train)
        targets_val_ids = batch["activity_id"].long()           # (B,)

        if id_remap is not None:
            vm = id_remap.clamp(min=-1)                         # safety
            mapped = vm[targets_val_ids]                        # (B,) in [-1..C_train-1]
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
                continue
        else:
            loss = ce(logits, targets_val_ids)
            total_loss += loss.item() * targets_val_ids.numel()
            total_n += targets_val_ids.numel()
            pred = logits.argmax(dim=-1)
            correct += (pred == targets_val_ids).sum().item()

    avg_loss = (total_loss / max(1, total_n))
    acc = (correct / max(1, total_n))
    return avg_loss, acc

# ------------------- Train -------------------
def train():
    device, amp_ctx, scaler = configure_device_and_amp()

    train_loader, val_loader, _, train_ds, val_ds = build_dataloaders(device)
    processors = build_processors()

    encoder = build_encoder(processors, device)
    head = build_activity_head(num_classes=train_ds.num_classes, device=device)

    params = list(encoder.parameters()) + list(head.parameters())

    def param_groups(model, wd):
        no_decay = []
        decay = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or n.endswith(".bias") or "norm" in n.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    head_lr = 1e-3
    enc_lr  = 2e-4
    optimizer = torch.optim.AdamW(
        param_groups(encoder, CFG.weight_decay) +
        [{"params": [p for g in param_groups(head, CFG.weight_decay) for p in g["params"]],
          "weight_decay": 0.0, "lr": head_lr}],
        lr=enc_lr, weight_decay=CFG.weight_decay
    )

    scheduler = build_warmup_cosine_scheduler(optimizer, epochs=CFG.epochs, steps_per_epoch=len(train_loader))
    sanity_check_optimizer(
        list(encoder.named_parameters()) + [("head."+n, p) for n, p in head.named_parameters()], optimizer
    )

    # --------- Build VAL->TRAIN id remap (by class name) ---------
    train_name_to_id = {name: i for name, i in train_ds.activity_to_id.items()}
    val_id_to_name = {i: name for i, name in enumerate(val_ds.id_to_activity)}
    remap_vec = torch.full((len(val_id_to_name),), -1, dtype=torch.long)
    for vid, vname in val_id_to_name.items():
        if vname in train_name_to_id:
            remap_vec[vid] = train_name_to_id[vname]
    id_remap_val2train = remap_vec.to(device)

    encoder.train(); head.train()
    global_step = 0
    loss_history: List[float] = []
    val_history: List[Dict[str, float]] = []
    ce = nn.CrossEntropyLoss()
    best_val_acc = -1.0

    for epoch in range(1, CFG.epochs + 1):
        epoch_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()
        steps_this_epoch = len(train_loader)
        print(f"\n[TRAIN-CLS] Epoch {epoch}/{CFG.epochs} - steps: {steps_this_epoch}")

        encoder.train(); head.train()

        with tqdm(total=steps_this_epoch, desc=f"Epoch {epoch}/{CFG.epochs}", dynamic_ncols=True) as pbar:
            for step, batch in enumerate(train_loader, start=1):
                batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}

                with amp_ctx:
                    out = encoder.encode_batch(batch)          # tokens: (B,P,D,F)
                    long_tokens = out["tokens"]

                    # Build flattened key_padding_mask (B,P*D), True=pad
                    flattened_key_padding_mask = None
                    if "pad_mask" in batch and batch["pad_mask"] is not None:
                        Bsz, P = batch["pad_mask"].shape
                        D = long_tokens.shape[2]
                        valid_flat = batch["pad_mask"].unsqueeze(-1).expand(Bsz, P, D).reshape(Bsz, P * D)
                        flattened_key_padding_mask = ~valid_flat

                    logits = head(long_tokens, flattened_key_padding_mask)  # (B,C)
                    targets = batch["activity_id"].long()
                    loss = ce(logits, targets)

                    # --- per-sample raw logits plot (only for b_idx=0) ---
                    head.debug_logits_bar(
                        logits, targets, b_idx=0,
                        class_names=getattr(train_ds, "id_to_activity", None),
                        save_path="debug/logits_bar.png",
                        annotate_values=False
                    )

                if not torch.isfinite(loss):
                    print("[WARN] Non-finite loss detected, skipping step.")
                    optimizer.zero_grad(set_to_none=True); scheduler.step(); pbar.update(1); continue

                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()

                    # ---- gradient debug (log AFTER unscale, BEFORE step) ----
                    scaler.unscale_(optimizer)
                    try:
                        dbg.set_step(global_step)
                        dbg.log_grads(encoder, encoder.grad_groups())
                        dbg.log_grads(head, head.grad_groups())
                    except Exception:
                        pass

                    if CFG.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(params, CFG.grad_clip)
                    scaler.step(optimizer); scaler.update()
                else:
                    loss.backward()
                    try:
                        dbg.set_step(global_step)
                        dbg.log_grads(encoder, encoder.grad_groups())
                        dbg.log_grads(head, head.grad_groups())
                    except Exception:
                        pass

                    if CFG.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(params, CFG.grad_clip)
                    optimizer.step()

                scheduler.step()
                global_step += 1
                epoch_loss += loss.item()
                loss_history.append(loss.item())

                with torch.no_grad():
                    pred = logits.argmax(dim=-1)
                    correct += (pred == targets).sum().item()
                    total += targets.numel()
                    acc = correct / max(1, total)

                if (len(loss_history) % CFG.LOSS_PLOT_EVERY) == 0:
                    plot_training_loss(loss_history, out_path="checkpoints/train_loss_cls.png")
                    try:
                        dbg.save_plots()
                    except Exception:
                        pass

                P = batch["patches"].shape[1]
                D = batch["patches"].shape[2]
                lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else CFG.lr

                pbar.set_postfix(loss=f"{loss.item():.6f}", acc=f"{acc:.3f}", P=P, D=D, lr=f"{lr_now:.2e}")
                pbar.update(1)

                if device.type == "cuda":
                    torch.cuda.empty_cache()

        dur = time.time() - t0
        avg_train = epoch_loss / max(1, steps_this_epoch)
        acc_epoch = correct / max(1, total)
        print(f"[EPOCH-CLS] {epoch} train_loss={avg_train:.6f} train_acc={acc_epoch:.3f} time={dur:.1f}s")

        # ---------- Validation every 5 epochs (and always at the very end) ----------
        if len(val_loader) > 0 and (epoch % 2 == 0 or epoch == CFG.epochs):
            val_loss, val_acc = evaluate(encoder, head, val_loader, ce, device, id_remap_val2train)
            val_history.append({"epoch": float(epoch), "loss": float(val_loss), "acc": float(val_acc)})
            print(f"[VAL-CLS]   {epoch}   val_loss={val_loss:.6f}   val_acc={val_acc:.3f}")
            plot_val_history(val_history, out_path="checkpoints/val_metrics_cls.png")

            # Keep best-by-accuracy checkpoint (optional but safe)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_ckpt(
                    encoder, head, optimizer, scheduler, scaler, epoch,
                    extra={"best_by": "val_acc", "val_acc": float(best_val_acc),
                           "id_to_activity": getattr(train_ds, "id_to_activity", None),
                           "activity_to_id": getattr(train_ds, "activity_to_id", None)},
                    out_dir="checkpoints/best"
                )

        # ---------- Periodic checkpoints ----------
        if (epoch % 10 == 0) or (epoch == CFG.epochs):
            extra = {}
            try:
                extra["id_to_activity"] = getattr(train_ds, "id_to_activity", None)
                extra["activity_to_id"] = getattr(train_ds, "activity_to_id", None)
            except Exception:
                pass
            save_ckpt(encoder, head, optimizer, scheduler, scaler, epoch, extra=extra, out_dir="checkpoints")


if __name__ == "__main__":
    train()
