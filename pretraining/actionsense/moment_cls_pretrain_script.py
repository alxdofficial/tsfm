"""
MOMENT Activity Classification Pretraining Script.

Architecture:
    MOMENTEncoder -> MOMENTCLSHead -> Activity logits

Key features:
- Fixed 512 timesteps (MOMENT requirement)
- Uses ActionSenseMOMENTCLS dataset (18 arm joint channels)
- Adaptive sampling: pad/uniform/windowing based on session length
- Per-patch cross-channel attention + temporal pooling
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

from datasets.ActionSenseMOMENTCLS import (
    ActionSenseMOMENTCLS,
    moment_cls_collate,
)
from encoders.moment.encoder import MOMENTEncoder
from pretraining.actionsense.heads.moment_cls import MOMENTCLSHead


class MOMENTCLSModel(nn.Module):
    """
    End-to-end classification model using MOMENT encoder.

    Pipeline:
        continuous_stream (B, D, 512) -> MOMENTEncoder -> (B, D, 64, F)
        -> MOMENTCLSHead (two-stage attention) -> (B, num_classes) logits
    """

    def __init__(
        self,
        num_classes: int,
        moment_size: str = "small",
        freeze_moment: bool = True,
        output_dim: int = None,
        dropout: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()

        self.device = device

        # 1. MOMENT encoder
        self.encoder = MOMENTEncoder(
            model_size=moment_size,
            freeze_moment=freeze_moment,
            output_dim=output_dim,
            device=device,
        )

        # Get actual output dimension from encoder
        encoder_output_dim = self.encoder.output_dim

        # 2. Classification head with two-stage attention pooling
        self.cls_head = MOMENTCLSHead(
            d_model=encoder_output_dim,
            nhead=8,
            num_classes=num_classes,
            dropout=dropout,
            mlp_hidden_ratio=4.0,
        )

        print(f"[MOMENTCLSModel] Model initialized with {num_classes} classes")
        print(f"[MOMENTCLSModel] Dropout: {dropout}, Freeze MOMENT: {freeze_moment}")

    def forward(self, batch):
        """
        Forward pass.

        Args:
            batch: Dict with continuous_stream (B, D, 512)

        Returns:
            logits: (B, num_classes)
        """
        # 1. Encode continuous stream
        encoder_output = self.encoder(batch)

        # 2. Pass through CLS head
        logits = self.cls_head(
            moment_embeddings=encoder_output["embeddings"],
            pad_mask=encoder_output["pad_mask"],
        )

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
                save_path=os.path.join(debug_dir, "logits", f"epoch{epoch:02d}_batch{batch_idx:05d}.png"),
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
    ax1.set_title("MOMENT CLS: Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Accuracy plot
    ax2.plot(epochs, train_accs, label="Train Acc", marker="o")
    ax2.plot(epochs, val_accs, label="Val Acc", marker="s")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("MOMENT CLS: Accuracy")
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
    plt.title("MOMENT CLS: Batch-Level Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "batch_loss.png"), dpi=150)
    plt.close()




def main():
    """Main training loop."""
    # Configuration
    BASE_DIR = "data/actionsenseqa_native/data"
    MANIFEST_CSV = "data/actionsenseqa_native/data/manifest.csv"

    # MOMENT configuration
    MOMENT_SIZE = "base"  # 'small' (40M), 'base' (125M), or 'large' (385M)
    FREEZE_MOMENT = False  # Fine-tune MOMENT end-to-end
    OUTPUT_DIM = None      # Use MOMENT's native hidden_dim (512 for small)

    # Sampling configuration
    RANDOM_WINDOW = True   # Random window for data augmentation

    # Training (adjusted for unfrozen MOMENT)
    BATCH_SIZE = 196        # Reduced due to memory + backprop through encoder
    NUM_EPOCHS = 500
    LEARNING_RATE = 1e-5   # Much lower LR for fine-tuning pretrained model
    WEIGHT_DECAY = 1e-3    # Stronger regularization to prevent overfitting
    DROPOUT = 0.3          # Increased dropout in CLS head

    # Validation
    VAL_RATIO = 0.2
    SPLIT_SEED = 42

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_DIR = "checkpoints/moment_cls"
    DEBUG_DIR = "debug/pretraining/moment_cls"
    SAVE_EVERY = 5

    print("=" * 80)
    print("MOMENT Activity Classification Pretraining")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"MOMENT size: {MOMENT_SIZE} (freeze={FREEZE_MOMENT})")
    print(f"Adaptive sampling to 512 timesteps:")
    print(f"  - Short sessions (T<512): use all + pad")
    print(f"  - Long sessions (equiv>=10Hz): uniform sample")
    print(f"  - Very long (equiv<10Hz): window at 10Hz cap (random={RANDOM_WINDOW})")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("=" * 80)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(os.path.join(DEBUG_DIR, "logits"), exist_ok=True)

    # Create datasets
    print("\n[1/4] Loading datasets...")
    train_dataset = ActionSenseMOMENTCLS(
        base_dir=BASE_DIR,
        manifest_csv_path=MANIFEST_CSV,
        split="train",
        val_ratio=VAL_RATIO,
        split_seed=SPLIT_SEED,
        random_window=RANDOM_WINDOW,
        log_mode="info",
    )

    val_dataset = ActionSenseMOMENTCLS(
        base_dir=BASE_DIR,
        manifest_csv_path=MANIFEST_CSV,
        split="val",
        val_ratio=VAL_RATIO,
        split_seed=SPLIT_SEED,
        random_window=False,
        log_mode="info",
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Num classes: {train_dataset.num_classes}")

    # Create dataloaders
    print("\n[2/4] Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=moment_cls_collate,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=moment_cls_collate,
        num_workers=0,
    )

    # Create model
    print("\n[3/4] Initializing model...")
    model = MOMENTCLSModel(
        num_classes=train_dataset.num_classes,
        moment_size=MOMENT_SIZE,
        freeze_moment=FREEZE_MOMENT,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
        device=DEVICE,
    ).to(DEVICE)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"\nTrainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print("\n[4/4] Starting training...")
    print("=" * 80)

    best_val_acc = 0.0
    best_epoch = 0
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

        t0 = time.time()
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        val_time = time.time() - t0

        print(f"Val:   loss={val_loss:.4f}, acc={val_acc:.3f}, time={val_time:.1f}s")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        plot_training_curves(train_losses, val_losses, train_accs, val_accs, DEBUG_DIR)
        plot_batch_loss(batch_loss_history, DEBUG_DIR)

        if epoch % SAVE_EVERY == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"moment_cls_epoch{epoch}.pt")
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

        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_path = os.path.join(CHECKPOINT_DIR, "moment_cls_best.pt")
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
        else:
            print(f"Best so far: {best_val_acc:.3f} at epoch {best_epoch}")

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
