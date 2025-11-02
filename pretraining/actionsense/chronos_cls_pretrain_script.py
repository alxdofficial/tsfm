"""
Chronos-2 Activity Classification Pretraining Script.

Architecture:
    Chronos2Encoder → Chronos2CLSHead → Activity logits

Key features:
- Uses continuous time series (not patches)
- Uses ActionSenseChronos2CLS dataset (18 arm joint channels)
- Native 60Hz sampling by default (configurable downsampling)
- Dynamic padding in collate function
- Cross-entropy loss with accuracy metrics
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Dict

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from datasets.ActionSenseChronos2CLS import (
    ActionSenseChronos2CLS,
    chronos2_cls_collate,
)
from encoders.chronos import Chronos2Encoder
from pretraining.actionsense.heads.chronos2_cls import Chronos2CLSHead


class Chronos2CLSModel(nn.Module):
    """
    End-to-end classification model using Chronos-2 encoder.

    Pipeline:
        continuous_stream (B, D, T) → Chronos2Encoder → (B, num_groups, D, 2048)
        → Chronos2CLSHead (attention pooling) → (B, num_classes) logits
    """

    def __init__(
        self,
        num_classes: int,
        freeze_chronos: bool = False,
        device: str = "cuda",
    ):
        super().__init__()

        self.device = device

        # 1. Chronos-2 encoder
        self.encoder = Chronos2Encoder(
            output_dim=2048,
            freeze_chronos=freeze_chronos,
            device=device,
        )

        # 2. Classification head with attention pooling
        self.cls_head = Chronos2CLSHead(
            d_model=2048,
            nhead=8,
            num_classes=num_classes,
            dropout=0.1,
            mlp_hidden_ratio=4.0,
        )

        print(f"[Chronos2CLSModel] Model initialized with {num_classes} classes")

    def forward(self, batch):
        """
        Forward pass.

        Args:
            batch: Dict with continuous_stream

        Returns:
            logits: (B, num_classes)
        """
        # 1. Encode continuous stream
        encoder_output = self.encoder(batch)
        # encoder_output["embeddings"]: (B, num_groups, D, 2048)
        # encoder_output["pad_mask"]: (B, num_groups)

        # 2. Pass through CLS head
        logits = self.cls_head(
            chronos_embeddings=encoder_output["embeddings"],
            pad_mask=encoder_output["pad_mask"],
        )  # (B, num_classes)

        return logits


def train_epoch(
    model, dataloader, optimizer, criterion, device, epoch, batch_loss_history, debug_dir, class_names
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        batch["continuous_stream"] = batch["continuous_stream"].to(device)
        targets = batch["activity_ids"].to(device)

        # Forward
        logits = model(batch)
        loss = criterion(logits, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        loss_val = loss.item()
        total_loss += loss_val
        num_batches += 1
        batch_loss_history.append(loss_val)

        pred = logits.argmax(dim=-1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)

        acc = correct / total
        pbar.set_postfix(loss=f"{loss_val:.4f}", acc=f"{acc:.3f}")

        # Debug: Save logits visualization every 50 batches
        if batch_idx % 50 == 0:
            model.cls_head.debug_logits_bar(
                logits,
                targets,
                b_idx=0,
                class_names=class_names,
                save_path=os.path.join(debug_dir, "logits", f"epoch{epoch}_batch{batch_idx}.png"),
            )

        # Memory management
        if batch_idx % 5 == 0:
            torch.cuda.empty_cache()

    avg_loss = total_loss / num_batches
    avg_acc = correct / total
    return avg_loss, avg_acc


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """
    Validate model.

    Returns:
        (avg_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Validation"):
        batch["continuous_stream"] = batch["continuous_stream"].to(device)
        targets = batch["activity_ids"].to(device)

        logits = model(batch)
        loss = criterion(logits, targets)

        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / len(dataloader)
    avg_acc = correct / total
    return avg_loss, avg_acc


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, out_dir):
    """Plot training and validation curves."""
    os.makedirs(out_dir, exist_ok=True)

    epochs = list(range(1, len(train_losses) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, train_losses, label="Train Loss", marker="o")
    ax1.plot(epochs, val_losses, label="Val Loss", marker="s")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Chronos-2 CLS: Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Accuracy plot
    ax2.plot(epochs, train_accs, label="Train Acc", marker="o")
    ax2.plot(epochs, val_accs, label="Val Acc", marker="s")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Chronos-2 CLS: Accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"[PLOT] Saved training curves: {out_dir}/training_curves.png")


def plot_batch_loss(batch_losses, out_dir):
    """Plot batch-level training loss."""
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(batch_losses, alpha=0.7, linewidth=0.8)
    plt.xlabel("Batch Number")
    plt.ylabel("Loss")
    plt.title("Chronos-2 CLS: Batch-Level Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "batch_loss.png"), dpi=150)
    plt.close()


def main():
    """Main training loop."""
    # -------------------- Configuration --------------------
    BASE_DIR = "data/actionsenseqa_native/data"
    MANIFEST_CSV = "data/actionsenseqa_native/data/manifest.csv"

    # Sampling (NATIVE by default - no downsampling)
    TARGET_FPS = None  # None means native 60Hz; set to 30 or 6 if OOM
    WINDOW_SECONDS = 10.0  # 10s at 60Hz means 600 timesteps
    RANDOM_WINDOW = True  # Data augmentation for training

    # Training
    BATCH_SIZE = 2  # Start small to test memory
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    FREEZE_CHRONOS = False

    # Validation
    VAL_RATIO = 0.2
    SPLIT_SEED = 42

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_DIR = "checkpoints/chronos_cls"
    DEBUG_DIR = "debug/pretraining/chronos_cls"
    SAVE_EVERY = 5

    print("=" * 80)
    print("Chronos-2 Activity Classification Pretraining")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Sampling: {'NATIVE 60Hz' if TARGET_FPS is None else f'{TARGET_FPS}Hz'}")
    effective_fps = TARGET_FPS if TARGET_FPS is not None else 60
    print(f"Window: {WINDOW_SECONDS}s ({int(WINDOW_SECONDS * effective_fps)} timesteps)")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Freeze Chronos: {FREEZE_CHRONOS}")
    print("=" * 80)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(os.path.join(DEBUG_DIR, "logits"), exist_ok=True)

    # -------------------- 1. Create datasets --------------------
    print("\n[1/4] Loading datasets...")
    train_dataset = ActionSenseChronos2CLS(
        base_dir=BASE_DIR,
        manifest_csv_path=MANIFEST_CSV,
        split="train",
        val_ratio=VAL_RATIO,
        split_seed=SPLIT_SEED,
        target_fps=TARGET_FPS,
        window_seconds=WINDOW_SECONDS,
        random_window=RANDOM_WINDOW,
        log_mode="info",
    )

    val_dataset = ActionSenseChronos2CLS(
        base_dir=BASE_DIR,
        manifest_csv_path=MANIFEST_CSV,
        split="val",
        val_ratio=VAL_RATIO,
        split_seed=SPLIT_SEED,
        target_fps=TARGET_FPS,
        window_seconds=WINDOW_SECONDS,
        random_window=False,  # Deterministic for validation
        log_mode="info",
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Num classes: {train_dataset.num_classes}")
    print(f"Classes: {train_dataset.id_to_activity}")

    # -------------------- 2. Create dataloaders --------------------
    print("\n[2/4] Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=chronos2_cls_collate,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=chronos2_cls_collate,
        num_workers=0,
    )

    # -------------------- 3. Create model --------------------
    print("\n[3/4] Initializing model...")
    model = Chronos2CLSModel(
        num_classes=train_dataset.num_classes,
        freeze_chronos=FREEZE_CHRONOS,
        device=DEVICE,
    ).to(DEVICE)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"\nTrainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # -------------------- 4. Training loop --------------------
    print("\n[4/4] Starting training...")
    print("=" * 80)

    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    batch_loss_history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print("-" * 80)

        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            DEVICE,
            epoch,
            batch_loss_history,
            DEBUG_DIR,
            train_dataset.id_to_activity,
        )
        train_time = time.time() - t0

        print(f"Train: loss={train_loss:.4f}, acc={train_acc:.3f}, time={train_time:.1f}s")

        # Validation
        t0 = time.time()
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        val_time = time.time() - t0

        print(f"Val:   loss={val_loss:.4f}, acc={val_acc:.3f}, time={val_time:.1f}s")

        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Plot curves
        plot_training_curves(train_losses, val_losses, train_accs, val_accs, DEBUG_DIR)
        plot_batch_loss(batch_loss_history, DEBUG_DIR)

        # Save checkpoints
        if epoch % SAVE_EVERY == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"chronos_cls_epoch{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "num_classes": train_dataset.num_classes,
                    "class_names": train_dataset.id_to_activity,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(CHECKPOINT_DIR, "chronos_cls_best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "num_classes": train_dataset.num_classes,
                    "class_names": train_dataset.id_to_activity,
                },
                best_path,
            )
            print(f"New best model saved: {best_path} (val_acc={val_acc:.3f})")

    # -------------------- Final summary --------------------
    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Final train loss: {train_losses[-1]:.4f}, acc: {train_accs[-1]:.3f}")
    print(f"Final val loss: {val_losses[-1]:.4f}, acc: {val_accs[-1]:.3f}")
    print(f"\nPlots saved to: {DEBUG_DIR}")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
