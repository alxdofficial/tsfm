# pretrain.py
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
# NEW: base + task-specific datasets and collate
from datasets.BaseDataset import tsfm_collate
from datasets.ActionSensePretrainingDatasets import ActionSenseMSPDataset

# Encoder & Recon Head
from encoder.TSFMEncoder import TSFMEncoder
from pretraining.actionsense.heads import SmallRecon

# Non-learnable feature processors
from encoder.processors.CorrelationSummaryProcessor import CorrelationSummaryProcessor
from encoder.processors.FrequencyFeatureProcessor import FrequencyFeatureProcessor
from encoder.processors.HistogramFeatureProcessor import HistogramFeatureProcessor
from encoder.processors.StatisticalFeatureProcessor import StatisticalFeatureProcessor

# Debug
from pretraining.actionsense.debug_stats import DebugStats

DEBUG_DIR = os.path.join("debug", "pretraining", "example")
dbg = DebugStats(out_dir=os.path.join(DEBUG_DIR, "stats"))

# Training utils
from training_utils import (
    configure_device_and_amp,
    build_optimizer,
    build_warmup_cosine_scheduler,
    sanity_check_optimizer,
    count_params,
    save_checkpoint,
)

# ------------------- Config -------------------
class Config:
    # Context & model dims
    context_size = 20
    llama_dim = 2048   # semantic dim (F)

    # Training
    batch_size = 10     # lower if OOM
    num_workers = 8
    epochs = 100
    grad_clip = 1.0
    lr = 2e-4
    weight_decay = 0.05

    # Masking config for MSP
    mask_cfg = {
        "ratio_patch": 0.4,
        "keep_patch_ratio": 0.00,
    }

    # Recon head hardcoded dims per dataset (you said these are known)
    D_channels = 66
    K_small = 44

    # Loss plot cadence (kept default = every batch to preserve behavior)
    LOSS_PLOT_EVERY = 1  # set to >1 to reduce I/O without changing logic

CFG = Config()


# ------------------- Builders -------------------
def build_dataloader(device: torch.device) -> tuple[DataLoader, dict]:
    # ---------------- Data ---------------------
    converter = ActionSenseConverter()  # uses default patch size = 96
    episodes, metadata = converter.convert()

    # Use the new MSPDataset (same behavior, no targets) + padding collate
    dataset = ActionSenseMSPDataset(
        episodes,
        metadata,
        context_size=CFG.context_size,
        debug=True,
        store_episode_stats=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=tsfm_collate,  # NEW: builds pad_mask and left-pads varying P
    )
    return dataloader, metadata


def build_processors():
    # ---------------- Processors ------------------
    processors = [
        CorrelationSummaryProcessor(),
        FrequencyFeatureProcessor(),
        HistogramFeatureProcessor(),
        StatisticalFeatureProcessor()
    ]
    return processors


def build_model(processors, device: torch.device) -> tuple[nn.Module, nn.Module]:
    # Peek at one real batch to get number of channels D (hardcoded here)
    D_channels = CFG.D_channels
    K_small = CFG.K_small
    llama_dim = CFG.llama_dim

    print(f"[SHAPES] D={D_channels}  K_small={K_small}  semantic_dim(F)={llama_dim}")

    # Build recon head *before* encoder/optimizer, and pass it into the encoder
    recon_hidden = min(1024, max(256, llama_dim // 4))
    recon_head = SmallRecon(
        semantic_dim=llama_dim,
        num_channels=D_channels,
        small_feature_dim=K_small,
        hidden=recon_hidden,
    ).to(device)

    print(f"[INIT] SmallRecon: D={recon_head.D} K={recon_head.K} hidden={recon_hidden} "
          f"params={count_params(recon_head):.2f}M")

    # ---------------- Encoder (with pre-initialized head) ---------------------
    encoder = TSFMEncoder(
        processors=processors,
        feature_dim=llama_dim,
        encoding_dim=llama_dim,
        pretraining_args=CFG.mask_cfg,
        recon_head=recon_head,     # << pass it in; no lazy init inside encoder
    ).to(device)

    print(f"[DEBUG] Encoder params (incl. head): {count_params(encoder):.2f}M")
    return encoder, recon_head


def plot_training_loss(loss_history, out_path=os.path.join(DEBUG_DIR, "train_loss.png")):
    """
    Save loss plot. Kept identical visuals; you can throttle via CFG.LOSS_PLOT_EVERY.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(loss_history, label="Batch Loss")
    plt.xlabel("Batch #")
    plt.ylabel("Loss")
    plt.title("Training Loss (per batch)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ------------------- Training Loop ----------------
def train():
    # FP16 autocast + GradScaler returned here
    device, amp_ctx, scaler = configure_device_and_amp()

    # Data & processors
    dataloader, _ = build_dataloader(device)
    processors = build_processors()

    # Model
    encoder, _ = build_model(processors, device)

    # Optimizer & Scheduler (Warmup → Cosine)
    optimizer = build_optimizer(encoder.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = build_warmup_cosine_scheduler(optimizer, epochs=CFG.epochs, steps_per_epoch=len(dataloader))

    # Sanity check: nothing missing from optimizer
    sanity_check_optimizer(encoder.named_parameters(), optimizer)

    # State
    encoder.train()
    global_step = 0
    last_num_channels = CFG.D_channels
    loss_history = []       # all batch losses
    loss_labels = []        # "eX-bY" labels

    for epoch in range(1, CFG.epochs + 1):
        epoch_loss = 0.0
        t0 = time.time()
        steps_this_epoch = len(dataloader)
        print(f"\n[TRAIN] Epoch {epoch}/{CFG.epochs} - steps: {steps_this_epoch}")

        with tqdm(total=steps_this_epoch, desc=f"Epoch {epoch}/{CFG.epochs}", dynamic_ncols=True) as pbar:
            for step, batch in enumerate(dataloader, start=1):
                batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}

                # ---- Forward & Loss (AMP fp16) ----
                with amp_ctx:
                    loss, aux = encoder.MSP_pretraining_step(batch)

                # quick NaN/Inf guard
                if not torch.isfinite(loss):
                    print("[WARN] Non-finite loss detected, skipping step.")
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    pbar.update(1)
                    continue

                # pick a stable b_idx (0) and let it auto-choose a masked patch
                if global_step % 10 == 0:
                    batch_size = batch["patches"].shape[0]
                    for dbg_b in range(batch_size):
                        encoder.debug_plot_reconstruction(
                            aux["targets_small"],
                            aux["recon_small"],
                            aux["token_mask"],
                            b_idx=dbg_b,
                            p_idx=None,
                            batch=batch,
                        )

                # debug only
                dbg.set_step(global_step)  # 1) step

                # ---- Backward + Optim Step (with GradScaler) ----
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    # Unscale before clipping so clip grad works on real values
                    scaler.unscale_(optimizer)
                    if CFG.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(encoder.parameters(), CFG.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if CFG.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(encoder.parameters(), CFG.grad_clip)
                    optimizer.step()

                scheduler.step()

                global_step += 1
                epoch_loss += loss.item()

                # --- Track loss for plotting ---
                loss_history.append(loss.item())
                loss_labels.append(f"e{epoch}-b{step}")

                # --- Save loss plot after each batch (same as original; throttle via CFG.LOSS_PLOT_EVERY) ---
                if (len(loss_history) % CFG.LOSS_PLOT_EVERY) == 0:
                    plot_training_loss(loss_history)

                P = batch["patches"].shape[1]
                D = batch["patches"].shape[2]
                masked_tokens = int(aux["token_mask"].sum().item())
                lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else CFG.lr

                pbar.set_postfix(
                    loss=f"{loss.item():.6f}",
                    masked_tokens=masked_tokens,
                    P=P,
                    D=D,
                    lr=f"{lr_now:.2e}"
                )
                pbar.update(1)

                if device.type == "cuda":
                    torch.cuda.empty_cache()

        dur = time.time() - t0
        avg = epoch_loss / max(1, steps_this_epoch)
        print(f"[EPOCH] {epoch} avg_loss={avg:.6f}  time={dur:.1f}s")

        # # Save checkpoints at the end of each epoch
        # if (epoch + 1) % 10 == 0 or epoch == CFG.epochs:
        #     save_checkpoint(encoder, epoch=epoch, num_channels=last_num_channels, out_dir="checkpoints")


if __name__ == "__main__":
    train()
