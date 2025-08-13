# pretrain_actionsense_cls.py
import os
import time
import torch
import torch.nn as nn
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


# ------------------- Train -------------------
def train():
    device, amp_ctx, scaler = configure_device_and_amp()

    dataloader, _, dataset = build_dataloader(device)
    processors = build_processors()

    encoder = build_encoder(processors, device)
    head = build_activity_head(num_classes=dataset.num_classes, device=device)

    params = list(encoder.parameters()) + list(head.parameters())
    def param_groups(model, wd):
        no_decay = []
        decay = []
        for n,p in model.named_parameters():
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

    scheduler = build_warmup_cosine_scheduler(optimizer, epochs=CFG.epochs, steps_per_epoch=len(dataloader))
    sanity_check_optimizer(
        list(encoder.named_parameters()) + [("head."+n, p) for n, p in head.named_parameters()], optimizer
    )

    encoder.train(); head.train()
    global_step = 0
    loss_history = []
    ce = nn.CrossEntropyLoss()

    for epoch in range(1, CFG.epochs + 1):
        epoch_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()
        steps_this_epoch = len(dataloader)
        print(f"\n[TRAIN-CLS] Epoch {epoch}/{CFG.epochs} - steps: {steps_this_epoch}")

        with tqdm(total=steps_this_epoch, desc=f"Epoch {epoch}/{CFG.epochs}", dynamic_ncols=True) as pbar:
            for step, batch in enumerate(dataloader, start=1):
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
                        class_names=getattr(dataset, "id_to_activity", None),
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
        avg = epoch_loss / max(1, steps_this_epoch)
        acc_epoch = correct / max(1, total)
        print(f"[EPOCH-CLS] {epoch} avg_loss={avg:.6f} acc={acc_epoch:.3f} time={dur:.1f}s")

        # ---------- Save checkpoints every 25 epochs and at the end ----------
        if (epoch % 25 == 0) or (epoch == CFG.epochs):
            # include class mapping for convenience if present
            extra = {}
            try:
                extra["id_to_activity"] = getattr(dataset, "id_to_activity", None)
                extra["activity_to_id"] = getattr(dataset, "activity_to_id", None)
            except Exception:
                pass
            save_ckpt(encoder, head, optimizer, scheduler, scaler, epoch, extra=extra, out_dir="checkpoints")


if __name__ == "__main__":
    train()
