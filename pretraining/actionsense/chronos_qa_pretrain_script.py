"""
Chronos-2 QA pretraining script for ActionSense dataset.

Architecture:
    Chronos2Encoder → QA Head → LLaMA decoder

Key differences from TSFM pretraining:
    - Uses continuous time series (not patches)
    - Uses ActionSenseChronos2QA dataset (18 arm joint channels, up to 2016 timesteps)
    - Chronos-2 handles tokenization and encoding
    - Simpler pipeline: no separate tokenizer needed
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import List
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from datasets.ActionSenseChronos2QA import (
    ActionSenseChronos2QA,
    chronos2_qa_collate,
)
from encoders.chronos import Chronos2Encoder
from pretraining.actionsense.heads import SensorQALLMHead


class Chronos2QAModel(nn.Module):
    """
    End-to-end QA model using Chronos-2 encoder.

    Pipeline:
        continuous_stream (B, D, T) → Chronos2Encoder → (B, num_patches, D, 2048)
        → SensorQALLMHead (channel fusion) → LLaMA decoder → answer

    Key: Patches are temporally aligned for proper channel fusion
    """

    def __init__(
        self,
        freeze_chronos=False,
        llama_model_name="meta-llama/Llama-3.2-1B-Instruct",
        lora_rank=16,
        lora_alpha=32,
        use_lora=True,
        device="cuda",
    ):
        super().__init__()

        self.device = device

        # 1. Chronos-2 encoder (outputs temporally-aligned patches)
        self.encoder = Chronos2Encoder(
            output_dim=2048,  # Output feature dimension
            freeze_chronos=freeze_chronos,
            device=device,
        )

        # 2. QA head with channel fusion (same as TSFM pretraining)
        self.qa_head = SensorQALLMHead(
            llama_model_name=llama_model_name,
            feature_dim=2048,
            attn_heads=8,
            attn_dropout=0.1,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            use_lora=use_lora,
            log_mode="info",
        )

        print("[Chronos2QAModel] Model initialized")

    def forward(self, batch):
        """
        Forward pass.

        Args:
            batch: Dict with continuous_stream, questions, answers

        Returns:
            Dict with loss and logits
        """
        # 1. Encode continuous stream
        encoder_output = self.encoder(batch)
        # encoder_output["embeddings"]: (B, num_patches, D, 2048)
        # encoder_output["pad_mask"]: (B, num_patches)

        # 2. Pass through QA head with channel fusion
        loss, info = self.qa_head(
            tokens=encoder_output["embeddings"],  # (B, num_patches, D, 2048)
            pad_mask=encoder_output["pad_mask"],
            questions=batch["questions"],
            answers=batch["answers"],
        )

        return {"loss": loss, **info}


def train_epoch(model, dataloader, optimizer, device, epoch, batch_loss_history, debug_dir, plot_every=10, gen_samples_every=50, teacher_forcing_ratio=0.0, use_amp=False):
    """
    Train for one epoch with curriculum learning (teacher forcing warmup).

    Args:
        teacher_forcing_ratio: Fraction of batches to use teacher forcing (0.0 = none, 0.2 = first 20%)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Calculate warmup threshold
    warmup_batches = int(len(dataloader) * teacher_forcing_ratio)
    transition_logged = False

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        batch["continuous_stream"] = batch["continuous_stream"].to(device)

        # Curriculum learning: Teacher forcing for warmup, then autoregressive
        use_teacher_forcing = batch_idx < warmup_batches

        # Encode sensor data (needed for both modes)
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
            encoder_output = model.encoder(batch)

        if use_teacher_forcing:
            # Teacher forcing: Standard forward pass
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                output = model.qa_head(
                    tokens=encoder_output["embeddings"],
                    pad_mask=encoder_output["pad_mask"],
                    questions=batch["questions"],
                    answers=batch["answers"],
                )
                loss = output[0]  # qa_head.forward returns (loss, info)
        else:
            # Log transition once
            if not transition_logged and teacher_forcing_ratio > 0:
                print(f"\n[INFO] Switching to autoregressive training at batch {batch_idx}/{len(dataloader)}")
                transition_logged = True

            # Autoregressive: Model uses own predictions
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                loss = model.qa_head.forward_autoregressive(
                    tokens=encoder_output["embeddings"],
                    pad_mask=encoder_output["pad_mask"],
                    questions=batch["questions"],
                    answers=batch["answers"],
                )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val
        num_batches += 1

        # Record batch loss for plotting
        batch_loss_history.append(loss_val)

        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        # Plot batch loss every N batches
        if batch_idx > 0 and batch_idx % plot_every == 0:
            plot_batch_loss(batch_loss_history, debug_dir)

        # Generate and visualize samples every N batches (only during autoregressive phase)
        if not use_teacher_forcing and batch_idx > 0 and batch_idx % gen_samples_every == 0:
            with torch.no_grad():
                # Get predictions from forward_autoregressive
                output = model.qa_head.forward_autoregressive(
                    tokens=encoder_output["embeddings"],
                    pad_mask=encoder_output["pad_mask"],
                    questions=batch["questions"],
                    answers=batch["answers"],
                    return_predictions=True,
                )
                predictions = output["predictions"]

                # Prepare sample data for visualization
                samples_data = []
                for i in range(min(2, len(predictions))):  # Max 2 samples
                    samples_data.append({
                        "question": batch["questions"][i],
                        "generated": predictions[i],
                        "ground_truth": batch["answers"][i],
                        "continuous_stream": batch["continuous_stream"][i],
                        "metadata": {
                            "subject": batch["metadata"]["subject"][i] if "subject" in batch["metadata"] else "N/A",
                            "activity_names": batch["metadata"]["activity_names"][i] if "activity_names" in batch["metadata"] else [],
                        },
                    })

                # Save visualization
                save_generation_samples(
                    epoch=epoch,
                    samples_data=samples_data,
                    out_dir=debug_dir,
                    max_samples=2,
                    batch_num=batch_idx,
                )

        # Aggressive cache clearing for autoregressive training
        if batch_idx % 5 == 0:
            torch.cuda.empty_cache()

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, device, use_mixed_precision=False):
    """
    Validate model using autoregressive loss and exact match accuracy.

    Args:
        model: Model to validate
        dataloader: Validation dataloader
        device: Device to run on
        use_mixed_precision: If True, use automatic mixed precision

    Returns:
        dict: {"loss": avg_loss, "exact_match": exact_match_accuracy}
    """
    model.train(False)  # Set to inference mode
    total_loss = 0.0
    num_batches = 0
    exact_matches = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            batch["continuous_stream"] = batch["continuous_stream"].to(device)

            # Compute autoregressive loss and get predictions
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_mixed_precision):
                encoder_output = model.encoder(batch)
                output = model.qa_head.forward_autoregressive(
                    tokens=encoder_output["embeddings"],
                    pad_mask=encoder_output["pad_mask"],
                    questions=batch["questions"],
                    answers=batch["answers"],
                    return_predictions=True,
                )
                loss = output["loss"]
                predictions = output["predictions"]

            total_loss += loss.item()
            num_batches += 1

            # Compute exact match
            for pred, gt in zip(predictions, batch["answers"]):
                # Normalize for comparison (strip whitespace)
                if pred.strip().lower() == gt.strip().lower():
                    exact_matches += 1
                total_samples += 1

    avg_loss = total_loss / num_batches
    exact_match_acc = exact_matches / total_samples if total_samples > 0 else 0.0

    return {"loss": avg_loss, "exact_match": exact_match_acc}


def plot_loss_curves(train_losses, val_losses, out_dir):
    """Plot train vs validation loss curves."""
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    epochs = list(range(1, len(train_losses) + 1))

    plt.plot(epochs, train_losses, label="Train Loss", marker="o", linewidth=2)
    plt.plot(epochs, val_losses, label="Val Loss", marker="s", linewidth=2)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Chronos-2 QA Training: Train vs Validation Loss", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "train_val_loss.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved loss curve: {out_path}")


def plot_batch_loss(batch_losses, out_dir):
    """Plot batch-level training loss."""
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(batch_losses, alpha=0.7, linewidth=0.8)

    plt.xlabel("Batch Number", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Chronos-2 QA Training: Batch-Level Loss", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "train_batch_loss.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    # print(f"[PLOT] Saved batch loss: {out_path}")


def plot_exact_match(val_em_scores, out_dir):
    """Plot validation exact match accuracy over epochs."""
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    epochs = list(range(1, len(val_em_scores) + 1))

    plt.plot(epochs, val_em_scores, label="Val Exact Match", marker="o", linewidth=2, color="green")

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Exact Match Accuracy", fontsize=12)
    plt.title("Chronos-2 QA Validation: Exact Match Accuracy (Autoregressive Generation)", fontsize=14)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "val_exact_match.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved exact match: {out_path}")


def generate_training_samples(model, batch, max_samples=2):
    """
    Generate answers for a training batch and collect samples.

    Args:
        model: The model
        batch: Training batch
        max_samples: Max samples to generate

    Returns:
        Tuple of (samples_data, exact_match_rate)
    """
    model.train(False)  # Temporarily set to inference mode
    samples_data = []
    exact_matches = 0

    with torch.no_grad():
        encoder_output = model.encoder(batch)
        generated_answers = model.qa_head.generate(
            tokens=encoder_output["embeddings"],
            pad_mask=encoder_output["pad_mask"],
            questions=batch["questions"],
            max_new_tokens=32,
            do_sample=False,
        )

        # Compute exact match for all samples in batch
        for idx, (gen, gt) in enumerate(zip(generated_answers, batch["answers"])):
            pred_norm = gen.lower().strip()
            gt_norm = gt.lower().strip()
            if pred_norm == gt_norm:
                exact_matches += 1

            # Collect visualization samples
            if idx < max_samples:
                samples_data.append({
                    "question": batch["questions"][idx],
                    "generated": gen,
                    "ground_truth": gt,
                    "continuous_stream": batch["continuous_stream"][idx].cpu(),
                    "metadata": {
                        "subject": batch["metadata"]["subject"][idx],
                        "activity_names": batch["metadata"]["activity_names"][idx],
                    },
                })

    model.train(True)  # Set back to training mode
    exact_match_rate = exact_matches / len(generated_answers) if len(generated_answers) > 0 else 0.0
    return samples_data, exact_match_rate


def save_generation_samples(
    epoch,
    samples_data,
    out_dir,
    max_samples=4,
    batch_num=None,
):
    """
    Save visualization of generated vs ground truth answers with sensor data.

    Args:
        epoch: Current epoch number
        samples_data: List of dicts with:
            - question: str
            - generated: str
            - ground_truth: str
            - continuous_stream: (D, T) tensor
            - metadata: dict with subject, activities, etc.
        out_dir: Output directory
        max_samples: Maximum number of samples to visualize
        batch_num: Optional batch number (for training samples)
    """
    gen_dir = os.path.join(out_dir, "generations")
    os.makedirs(gen_dir, exist_ok=True)

    for idx, sample in enumerate(samples_data[:max_samples]):
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        # Top panel: Sensor signals (first 10 channels)
        stream = sample["continuous_stream"].detach().cpu().numpy()  # (D, T)
        D, T = stream.shape
        num_channels_to_plot = min(10, D)

        for ch_idx in range(num_channels_to_plot):
            axes[0].plot(stream[ch_idx, :], alpha=0.7, linewidth=0.8, label=f"Ch {ch_idx+1}")

        axes[0].set_xlabel("Timestep", fontsize=11)
        axes[0].set_ylabel("Value", fontsize=11)
        axes[0].set_title(f"Sensor Data: {D} channels × {T} timesteps (showing first {num_channels_to_plot} channels)", fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=8, ncol=5)
        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))

        # Bottom panel: Q&A text
        axes[1].axis("off")

        # Get metadata
        metadata = sample.get("metadata", {})
        subject = metadata.get("subject", "N/A")
        activities = metadata.get("activity_names", [])
        activity_str = ", ".join(activities) if activities else "N/A"

        # Format text
        question = sample["question"]
        generated = sample["generated"]
        ground_truth = sample["ground_truth"]

        # Determine if correct
        is_correct = generated.lower().strip() == ground_truth.lower().strip()
        result_color = "green" if is_correct else "red"
        result_text = "✓ CORRECT" if is_correct else "✗ INCORRECT"

        text_content = (
            f"METADATA:\n"
            f"  Subject: {subject}\n"
            f"  Activities: {activity_str}\n"
            f"  Channels: {D}, Timesteps: {T}\n\n"
            f"QUESTION:\n  {question}\n\n"
            f"GROUND TRUTH:\n  {ground_truth}\n\n"
            f"GENERATED:\n  {generated}\n\n"
            f"RESULT: {result_text}"
        )

        axes[1].text(
            0.05, 0.95,
            text_content,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        # Add result badge
        axes[1].text(
            0.95, 0.05,
            result_text,
            fontsize=14,
            fontweight="bold",
            color=result_color,
            verticalalignment="bottom",
            horizontalalignment="right",
        )

        # Create title based on whether this is validation or training
        if batch_num is not None:
            title = f"Training Sample - Epoch {epoch}, Batch {batch_num}, Sample {idx+1}"
            filename = f"train_epoch{epoch:03d}_batch{batch_num:05d}_sample{idx:02d}.png"
        else:
            title = f"Validation Sample - Epoch {epoch}, Sample {idx+1}"
            filename = f"val_epoch{epoch:03d}_sample{idx:02d}.png"

        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        out_path = os.path.join(gen_dir, filename)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"[PLOT] Saved {len(samples_data[:max_samples])} generation samples to {gen_dir}")


def main():
    """Main training loop."""
    # Configuration
    BASE_DIR = "data/actionsenseqa_native/data"
    QA_JSONL_PATH = "data/actionsenseqa_native/data/qa_pairs_templated.jsonl"
    MANIFEST_CSV_PATH = "data/actionsenseqa_native/data/manifest.csv"

    FREEZE_CHRONOS = False
    LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
    LORA_RANK = 16
    LORA_ALPHA = 32
    USE_LORA = True

    BATCH_SIZE = 2
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    VAL_RATIO = 0.2
    SPLIT_SEED = 42
    USE_MIXED_PRECISION = False  # Enable mixed precision training for memory efficiency
    TEACHER_FORCING_RATIO = 0.2  # Use teacher forcing for first 20% of batches per epoch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_DIR = "checkpoints/chronos_qa"
    DEBUG_DIR = "debug/pretraining/chronos_qa"
    SAVE_EVERY = 1
    PLOT_EVERY = 10  # Plot batch loss every N batches
    GEN_SAMPLES_EVERY = 2  # Save generation samples every N epochs (validation)
    GEN_SAMPLES_EVERY_BATCH = 50  # Generate samples every N training batches
    MAX_GEN_SAMPLES = 4  # Maximum samples to visualize per epoch

    print("=" * 80)
    print("Chronos-2 QA Pretraining")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Mixed precision: {USE_MIXED_PRECISION}")
    print(f"Freeze Chronos: {FREEZE_CHRONOS}")
    print(f"Use LoRA: {USE_LORA}")
    if USE_LORA:
        print(f"LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}")
    print("=" * 80)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    # 1. Create datasets
    print("\n[1/4] Loading datasets...")
    train_dataset = ActionSenseChronos2QA(
        base_dir=BASE_DIR,
        qa_jsonl_path=QA_JSONL_PATH,
        manifest_csv_path=MANIFEST_CSV_PATH,
        split="train",
        val_ratio=VAL_RATIO,
        split_seed=SPLIT_SEED,
        target_fps=6,         # Downsample to 30Hz for memory efficiency
        window_seconds=5.0,   # 10 second windows
        random_window=True,    # Random window selection for data augmentation
        log_mode="info",
    )

    val_dataset = ActionSenseChronos2QA(
        base_dir=BASE_DIR,
        qa_jsonl_path=QA_JSONL_PATH,
        manifest_csv_path=MANIFEST_CSV_PATH,
        split="val",
        val_ratio=VAL_RATIO,
        split_seed=SPLIT_SEED,
        target_fps=6,         # Downsample to 30Hz
        window_seconds=5.0,   # 10 second windows
        random_window=True,   # Deterministic for validation
        log_mode="info",
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # 2. Create dataloaders
    print("\n[2/4] Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=chronos2_qa_collate,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=chronos2_qa_collate,
        num_workers=0,
    )

    # 3. Create model
    print("\n[3/4] Initializing model...")
    model = Chronos2QAModel(
        freeze_chronos=FREEZE_CHRONOS,
        llama_model_name=LLAMA_MODEL_NAME,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        use_lora=USE_LORA,
        device=DEVICE,
    ).to(DEVICE)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"\nTrainable parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)

    # Mixed precision with bfloat16 (no GradScaler needed)
    if USE_MIXED_PRECISION:
        print("\n[INFO] Using automatic mixed precision training (bfloat16)")
        print("[INFO] Note: bfloat16 doesn't require GradScaler due to larger dynamic range")

    # 4. Training loop
    print("\n[4/4] Starting training...")
    print("=" * 80)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    batch_loss_history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print("-" * 80)

        train_loss = train_epoch(
            model, train_loader, optimizer, DEVICE, epoch,
            batch_loss_history, DEBUG_DIR, PLOT_EVERY, GEN_SAMPLES_EVERY_BATCH,
            teacher_forcing_ratio=TEACHER_FORCING_RATIO, use_amp=USE_MIXED_PRECISION
        )
        print(f"Train loss: {train_loss:.4f}")

        # Validation
        val_results = validate(
            model, val_loader, DEVICE,
            use_mixed_precision=USE_MIXED_PRECISION
        )
        val_loss = val_results["loss"]
        val_exact_match = val_results["exact_match"]
        print(f"Val loss: {val_loss:.4f} | Val exact match: {val_exact_match:.4f}")

        # Record epoch metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Plot loss curves after each epoch
        plot_loss_curves(train_losses, val_losses, DEBUG_DIR)

        if epoch % SAVE_EVERY == 0:
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                f"chronos_qa_epoch{epoch}.pt",
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint: {checkpoint_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(CHECKPOINT_DIR, "chronos_qa_best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                best_path,
            )
            print(f"New best model saved: {best_path} (val_loss: {val_loss:.4f})")

    # Final plots
    print("\n[INFO] Generating final plots...")
    plot_loss_curves(train_losses, val_losses, DEBUG_DIR)
    plot_batch_loss(batch_loss_history, DEBUG_DIR)

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss: {val_losses[-1]:.4f}")
    print(f"\nPlots saved to: {DEBUG_DIR}")
    print(f"  - train_val_loss.png (train & val loss curves)")
    print(f"  - train_batch_loss.png (batch-level training loss)")
    print(f"\nCheckpoints saved to: {CHECKPOINT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
