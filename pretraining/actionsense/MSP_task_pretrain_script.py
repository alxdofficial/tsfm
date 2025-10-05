# pretrain_actionsense_msp.py
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
from datasets.ActionSensePretrainingDatasets import ActionSenseMSPDataset

# Encoder & Recon Head
from encoder.TSFMEncoder import TSFMEncoder
from pretraining.actionsense.reconstruction_head import SmallRecon

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
    context_size = 100
    llama_dim = 512   # semantic dim (F)

    # Training
    batch_size = 1
    num_workers = 8
    epochs = 50
    grad_clip = 1.0
    lr = 2e-4
    weight_decay = 0.05
    dropout = 0.1
    nhead = 8
    num_layers = 4
    mlp_ratio = 4.0

    # Masking config for MSP
    mask_cfg = {
        "ratio_patch": 0.5,
        "keep_patch_ratio": 0.00,
    }

    # Recon head hardcoded dims per dataset
    D_channels = 66
    K_small = 44

    # Plot cadence
    LOSS_PLOT_EVERY = 10

CFG = Config()

# ------------------- Builders -------------------
def build_dataloader(device: torch.device):
    converter = ActionSenseConverter()  # default patch_size=96
    episodes, metadata = converter.convert()

    dataset = ActionSenseMSPDataset(
        episodes, metadata,
        context_size=CFG.context_size,
        debug=True,
        split="train", val_ratio=0.30, split_seed=42,
        store_episode_stats=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=tsfm_collate,
    )
    return dataloader, metadata


def build_processors():
    return [
        CorrelationSummaryProcessor(),
        FrequencyFeatureProcessor(),
        HistogramFeatureProcessor(),
        StatisticalFeatureProcessor()
    ]


def build_encoder_and_head(processors, device: torch.device) -> nn.Module:
    # Recon head first
    recon_hidden = min(1024, max(256, CFG.llama_dim // 4))
    recon_head = SmallRecon(
        semantic_dim=CFG.llama_dim,
        num_channels=CFG.D_channels,
        small_feature_dim=CFG.K_small,
        hidden=recon_hidden,
    ).to(device)
    print(f"[INIT] SmallRecon: D={recon_head.D} K={recon_head.K} hidden={recon_hidden} "
          f"params={count_params(recon_head):.2f}M")

    # Encoder (pass head in)
    encoder = TSFMEncoder(
        processors=processors,
        feature_dim=CFG.llama_dim,
        encoding_dim=CFG.llama_dim,
        num_layers=CFG.num_layers,
        nhead=CFG.nhead,
        dropout=CFG.dropout,
        mlp_ratio=CFG.mlp_ratio,
        pretraining_args=CFG.mask_cfg,
        recon_head=recon_head,
    ).to(device)
    print(f"[DEBUG] Encoder params (incl. head): {count_params(encoder):.2f}M")
    return encoder


def plot_training_loss(loss_history, out_path="checkpoints/train_loss_msp.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(loss_history, label="Batch Loss")
    plt.xlabel("Batch #")
    plt.ylabel("Loss")
    plt.title("Training Loss (per batch) — MSP")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ------------------- Train -------------------
def train():
    device, amp_ctx, scaler = configure_device_and_amp()

    dataloader, _ = build_dataloader(device)
    processors = build_processors()
    encoder = build_encoder_and_head(processors, device)

    # Optimizer / Scheduler
    optimizer = build_optimizer(encoder.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = build_warmup_cosine_scheduler(optimizer, epochs=CFG.epochs, steps_per_epoch=len(dataloader))
    sanity_check_optimizer(encoder.named_parameters(), optimizer)

    encoder.train()
    global_step = 0
    loss_history = []

    for epoch in range(1, CFG.epochs + 1):
        epoch_loss = 0.0
        t0 = time.time()
        steps_this_epoch = len(dataloader)
        print(f"\n[TRAIN-MSP] Epoch {epoch}/{CFG.epochs} - steps: {steps_this_epoch}")

        with tqdm(total=steps_this_epoch, desc=f"Epoch {epoch}/{CFG.epochs}", dynamic_ncols=True) as pbar:
            for step, batch in enumerate(dataloader, start=1):
                batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}

                with amp_ctx:
                    loss, aux = encoder.MSP_pretraining_step(batch)

                if not torch.isfinite(loss):
                    print("[WARN] Non-finite loss detected, skipping step.")
                    optimizer.zero_grad(set_to_none=True); scheduler.step(); pbar.update(1); continue

                # Optional recon debug image dump
                if global_step % 10 == 0:
                    encoder.debug_plot_reconstruction(
                        aux["targets_small"], aux["recon_small"], aux["token_mask"], b_idx=0, p_idx=None
                    )

                dbg.set_step(global_step)

                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if CFG.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(encoder.parameters(), CFG.grad_clip)
                    scaler.step(optimizer); scaler.update()
                else:
                    loss.backward()
                    if CFG.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(encoder.parameters(), CFG.grad_clip)
                    optimizer.step()

                scheduler.step()
                global_step += 1
                epoch_loss += loss.item()
                loss_history.append(loss.item())

                if (len(loss_history) % CFG.LOSS_PLOT_EVERY) == 0:
                    plot_training_loss(loss_history, out_path="checkpoints/train_loss_msp.png")

                P = batch["patches"].shape[1]
                D = batch["patches"].shape[2]
                masked_tokens = int(aux["token_mask"].sum().item())
                lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else CFG.lr

                pbar.set_postfix(
                    loss=f"{loss.item():.6f}",
                    masked_tokens=masked_tokens,
                    P=P, D=D, lr=f"{lr_now:.2e}"
                )
                pbar.update(1)

                if device.type == "cuda":
                    torch.cuda.empty_cache()

        dur = time.time() - t0
        avg = epoch_loss / max(1, steps_this_epoch)
        print(f"[EPOCH-MSP] {epoch} avg_loss={avg:.6f} time={dur:.1f}s")


if __name__ == "__main__":
    train()
