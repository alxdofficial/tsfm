import os
import math
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm 

from data_utils.collate import pad_collate
from data_utils.converters.Sensor2TextConverter import Sensor2TextConverter
from data_utils.TSFMPretrainingDataset import TSFMPretrainingDataset
from encoder.TSFMEncoder import TSFMEncoder

# Non-learnable feature processors
from encoder.processors.CorrelationSummaryProcessor import CorrelationSummaryProcessor
from encoder.processors.FrequencyFeatureProcessor import FrequencyFeatureProcessor
from encoder.processors.HistogramFeatureProcessor import HistogramFeatureProcessor
from encoder.processors.StatisticalFeatureProcessor import StatisticalFeatureProcessor
from encoder.debug_stats import DebugStats
dbg = DebugStats(out_dir="debug_stats")

# ------------------- Config -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

context_size = 16
llama_dim = 2048  # LLaMA tokenizer dim (F)
batch_size = 18    # lower if OOM
num_workers = 4
epochs = 100
grad_clip = 1.0
lr = 2e-4
weight_decay = 0.05
mask_cfg = {
    "ratio_feature": 0.20,  # mask 20% of (P,D,F)
    "ratio_patch":   0.10,  # mask 10% patches fully
}

# ------------------- Data ---------------------
converter = Sensor2TextConverter()  # uses default patch size = 96
episodes, metadata = converter.convert()

dataset = TSFMPretrainingDataset(episodes, metadata, context_size=context_size)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
    collate_fn=pad_collate
)

# ---------------- Processors ------------------
processors = [
    CorrelationSummaryProcessor(),
    FrequencyFeatureProcessor(),
    HistogramFeatureProcessor(),
    StatisticalFeatureProcessor()
]

# ---------------- Encoder ---------------------
encoder = TSFMEncoder(
    processors=processors,
    feature_dim=llama_dim,        # F
    encoding_dim=llama_dim,       # E (per sinusoid family before proj)
    pretraining_args=mask_cfg
).to(device)

print(f"[DEBUG] Encoder params: {sum(p.numel() for p in encoder.parameters())/1e6:.2f}M")

# ---------------- Optimizer/Sched -------------
optimizer = AdamW(encoder.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs * len(dataloader)))
scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

# ---------------- Saving Utils ----------------
def save_checkpoint(encoder: nn.Module, epoch: int, num_channels: int | None, out_dir: str = "checkpoints"):
    os.makedirs(out_dir, exist_ok=True)
    backbone_state = {k: v for k, v in encoder.state_dict().items() if not k.startswith("recon_head.")}
    backbone_path = os.path.join(out_dir, f"encoder_backbone_e{epoch}.pt")
    torch.save(backbone_state, backbone_path)
    print(f"[SAVE] Saved backbone to: {backbone_path}")

    if encoder.recon_head is not None and num_channels is not None:
        head_path = os.path.join(out_dir, f"msp_head_D{num_channels}_e{epoch}.pt")
        torch.save({
            "head": encoder.recon_head.state_dict(),
            "meta": {"D": num_channels, "feature_dim": encoder.feature_dim}
        }, head_path)
        print(f"[SAVE] Saved recon head to: {head_path}")

# ---------------- Training Loop ----------------
encoder.train()
global_step = 0
last_num_channels = None
loss_history = []       # all batch losses
loss_labels = []        # "eX-bY" labels

global_step = 0

for epoch in range(1, epochs + 1):
    epoch_loss = 0.0
    t0 = time.time()
    steps_this_epoch = len(dataloader)
    print(f"\n[TRAIN] Epoch {epoch}/{epochs} - steps: {steps_this_epoch}")

    with tqdm(total=steps_this_epoch, desc=f"Epoch {epoch}/{epochs}", dynamic_ncols=True) as pbar:
        for step, batch in enumerate(dataloader, start=1):
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}

            if last_num_channels is None and isinstance(batch.get("patches"), torch.Tensor):
                last_num_channels = batch["patches"].shape[2]

            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                loss, aux = encoder.masked_self_prediction(batch)

            # debug only
            dbg.set_step(global_step)  # 1) step
            dbg.log_scalar("loss/batch", float(loss.item()))
            lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else lr
            dbg.log_scalar("opt/lr", float(lr_now))


            scaler.scale(loss).backward()


            # debug only
            dbg.log_grads(encoder, groups=encoder.grad_groups())
            dbg.save_plots()


            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()

            # --- Track loss for plotting ---
            loss_history.append(loss.item())
            loss_labels.append(f"e{epoch}-b{step}")

            # --- Save loss plot after each batch ---
            plt.figure(figsize=(10, 4))
            plt.plot(loss_history, label="Batch Loss")
            plt.xlabel("Batch #")
            plt.ylabel("Loss")
            plt.title("Training Loss (per batch)")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            os.makedirs("checkpoints", exist_ok=True)
            plt.tight_layout()
            plt.savefig("checkpoints/train_loss.png")
            plt.close()

            P = batch["patches"].shape[1]
            D = batch["patches"].shape[2]
            masked_tokens = int(aux["token_mask"].sum().item())
            lr_now = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else lr

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

    # Save checkpoints at the end of each epoch
    if (epoch + 1) % 10 == 0 or epoch == epochs:
        save_checkpoint(encoder, epoch=epoch, num_channels=last_num_channels, out_dir="checkpoints")