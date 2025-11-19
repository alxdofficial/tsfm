"""
Semantic alignment training for IMU activity recognition.

Stage 2 of training: Align pretrained encoder outputs with text embeddings.
"""

import sys
from pathlib import Path

# Add parent directories to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "tools" / "models"))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from datetime import datetime
from tqdm import tqdm
import json

from datasets.imu_pretraining_dataset.multi_dataset_loader import create_dataloaders
from imu_activity_recognition_encoder.encoder import IMUActivityRecognitionEncoder
from imu_activity_recognition_encoder.semantic_alignment import SemanticAlignmentHead
from training_scripts.imu_tool_pretraining.label_bank import LabelBank
from training_scripts.imu_tool_pretraining.prototype_manager import PrototypeManager
from training_scripts.imu_tool_pretraining.semantic_loss import SemanticAlignmentLoss, compute_retrieval_metrics
from training_scripts.imu_tool_pretraining.plot_utils import TrainingPlotter
from training_scripts.imu_tool_pretraining.memory_bank import MemoryBank


# ======================== HYPERPARAMETERS ========================

# Data configuration
DATA_ROOT = "/home/alex/code/tsfm/data"
DATASETS = ['uci_har', 'mhealth', 'pamap2', 'wisdm']

PATCH_SIZE_PER_DATASET = {
    'uci_har': 2.56,
    'mhealth': 4.0,
    'pamap2': 5.0,
    'wisdm': 10.0
}

MAX_PATCHES_PER_SAMPLE = 48

# Encoder configuration (must match pretrained model)
D_MODEL = 384
NUM_HEADS = 8
NUM_TEMPORAL_LAYERS = 4
DIM_FEEDFORWARD = 1536
DROPOUT = 0.1
USE_CROSS_CHANNEL = True
CNN_CHANNELS = [64, 128]
CNN_KERNEL_SIZES = [3, 5, 7]

# Semantic alignment configuration
D_MODEL_FUSED = 384  # Dimension after cross-channel fusion (match D_MODEL to avoid bottleneck)
SEMANTIC_DIM = 384  # Final embedding dimension (must match SentenceBERT)
NUM_BOTTLENECKS = 1  # Start with single bottleneck per patch
NUM_SEMANTIC_TEMPORAL_LAYERS = 2  # Temporal attention layers in semantic head

# Text encoder configuration
SENTENCE_BERT_MODEL = 'all-MiniLM-L6-v2'  # 384-dim embeddings, fast

# Training configuration
OUTPUT_DIR = "training_output/semantic_alignment"  # Note: plots go to semantic_alignment/<timestamp>/plots/
CHECKPOINT_DIR = None  # Will be set in main()
PRETRAINED_ENCODER_PATH = None  # Path to pretrained encoder checkpoint
SAVE_EVERY = 5
SEED = 42

# End-to-end training (encoder + semantic head)
EPOCHS = 30
BATCH_SIZE = 16  # Reduced from 32 due to training both encoder + head
LEARNING_RATE = 1e-4  # Conservative LR for stable training from scratch
WARMUP_EPOCHS = 3

# Shared parameters
NUM_WORKERS = 4
WEIGHT_DECAY = 1e-5
TEMPERATURE = 0.3  # Higher temperature for more stable training from scratch (was 0.1)

# Soft targets configuration
USE_SOFT_TARGETS = True  # Learn from semantic similarities between labels
SOFT_TARGET_TEMPERATURE = 0.5  # Temperature for soft target distribution (higher = smoother)
SOFT_TARGET_WEIGHT = 0.5  # Balance: 0=hard, 1=pure soft, 0.5=balanced

# Prototype soft targets configuration (DISABLED BY DEFAULT)
USE_PROTOTYPES = False  # Use cluster-based prototypes for soft targets
NUM_PROTOTYPE_CLUSTERS = 5  # Number of semantic clusters (e.g., locomotion, stationary, etc.)

# Memory bank configuration (MoCo-style queue for more negatives)
USE_MEMORY_BANK = True  # Enable memory bank for 4096+ negatives
MEMORY_BANK_SIZE = 4096  # Queue size (provides 4096 additional negatives)
PROTOTYPE_UPDATE_INTERVAL = 100  # Update prototypes every N batches
PROTOTYPE_WEIGHT = 0.5  # Balance between pairwise and prototype (0=pairwise, 1=prototype)

# Plotting
PLOT_EVERY_N_BATCHES = 10

# =================================================================


class SemanticAlignmentModel(nn.Module):
    """Complete model for semantic alignment combining encoder + semantic head."""

    def __init__(self, encoder: IMUActivityRecognitionEncoder, semantic_head: SemanticAlignmentHead):
        super().__init__()
        self.encoder = encoder
        self.semantic_head = semantic_head

    def forward(self, data, channel_descriptions, channel_mask, sampling_rates, patch_sizes):
        batch_size = data.shape[0]
        encoder_outputs, patch_masks = [], []

        for i in range(batch_size):
            with torch.no_grad():
                patches, _ = self.encoder.preprocess(
                    data[i], sampling_rate_hz=sampling_rates[i], patch_size_sec=patch_sizes[i]
                )
            if patches is None or len(patches) == 0:
                encoder_outputs.append(None)
                patch_masks.append(None)
                continue

            if len(patches) > MAX_PATCHES_PER_SAMPLE:
                patches = patches[:MAX_PATCHES_PER_SAMPLE]

            # Pad channel descriptions to match data's channel count (same as pretrain.py)
            num_channels = data[i].shape[1]  # Total channels including padding
            channel_descs = channel_descriptions[i]
            padded_descs = channel_descs + ["[PAD]"] * (num_channels - len(channel_descs))

            patches_tensor = patches.unsqueeze(0).to(data.device)  # Add batch dimension: (1, patches, 96, channels)
            encoded = self.encoder(patches_tensor, padded_descs)  # (1, patches, channels, d_model)
            encoder_outputs.append(encoded.squeeze(0))  # Remove batch dim: (patches, channels, d_model)
            patch_masks.append(torch.ones(len(patches), dtype=torch.bool, device=data.device))

        # Pad
        max_patches = max(len(p) for p in patch_masks if p is not None)
        max_channels = data.shape[2]
        padded_encoder_output = torch.zeros(batch_size, max_patches, max_channels, D_MODEL, device=data.device)
        padded_patch_mask = torch.zeros(batch_size, max_patches, dtype=torch.bool, device=data.device)

        for i, (enc_out, p_mask) in enumerate(zip(encoder_outputs, patch_masks)):
            if enc_out is not None:
                num_patches, num_channels, _ = enc_out.shape
                padded_encoder_output[i, :num_patches, :num_channels, :] = enc_out
                padded_patch_mask[i, :num_patches] = p_mask

        return self.semantic_head(padded_encoder_output, channel_mask=channel_mask, 
                                   patch_mask=padded_patch_mask, normalize=True)


def train_epoch(model, label_bank, dataloader, criterion, optimizer, device, epoch, scaler, prototype_manager=None, plotter=None, stage="stage1", memory_bank=None):
    """Train for one epoch."""
    model.train()

    # Track all metrics
    total_loss = 0.0
    total_acc_i2t = 0.0
    total_acc_t2i = 0.0
    total_pos_sim = 0.0
    total_neg_sim = 0.0
    total_sim_gap = 0.0

    pbar = tqdm(dataloader, desc=f"[{stage}] Epoch {epoch} Training")

    for batch_idx, batch in enumerate(pbar):
        data = batch['data'].to(device)
        channel_mask = batch['channel_mask'].to(device)
        label_texts = batch['label_texts']
        metadata = batch['metadata']

        sampling_rates = [m['sampling_rate_hz'] for m in metadata]
        patch_sizes = [m['patch_size_sec'] for m in metadata]
        channel_descriptions = [m['channel_descriptions'] for m in metadata]

        with torch.no_grad():
            text_embeddings = label_bank.encode(label_texts, normalize=True)

            # Add labels to prototype manager if enabled
            if prototype_manager is not None:
                prototype_manager.add_labels(label_texts, text_embeddings)

        # Clear gradients BEFORE forward pass
        optimizer.zero_grad()

        # Get queue embeddings from memory bank if enabled
        if memory_bank is not None and USE_MEMORY_BANK:
            imu_queue, text_queue = memory_bank.get_queue_embeddings(label_bank, device)
        else:
            imu_queue, text_queue = None, None

        with autocast('cuda', enabled=device.type == 'cuda'):
            imu_embeddings = model(data, channel_descriptions, channel_mask, sampling_rates, patch_sizes)
            loss, metrics = criterion(imu_embeddings, text_embeddings, label_texts,
                                     return_metrics=True, imu_queue=imu_queue, text_queue=text_queue)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Update memory bank with current batch embeddings
        if memory_bank is not None and USE_MEMORY_BANK:
            memory_bank.update(imu_embeddings, label_texts)

        # Accumulate metrics
        total_loss += metrics['loss']
        total_acc_i2t += metrics['acc_imu_to_text']
        total_acc_t2i += metrics['acc_text_to_imu']
        total_pos_sim += metrics['positive_similarity']
        total_neg_sim += metrics['negative_similarity']
        total_sim_gap += metrics['similarity_gap']

        pbar.set_postfix({
            'loss': f"{metrics['loss']:.4f}",
            'acc': f"{metrics['acc_imu_to_text']:.3f}",
            'sim_gap': f"{metrics['similarity_gap']:.3f}"
        })

        # Batch-level plotting (every N batches)
        if plotter is not None and (batch_idx + 1) % PLOT_EVERY_N_BATCHES == 0:
            global_batch = (epoch - 1) * len(dataloader) + batch_idx
            plotter.add_scalar(f'batch/{stage}_loss', metrics['loss'], global_batch)
            plotter.add_scalar(f'batch/{stage}_acc_imu_to_text', metrics['acc_imu_to_text'], global_batch)
            plotter.add_scalar(f'batch/{stage}_acc_text_to_imu', metrics['acc_text_to_imu'], global_batch)
            plotter.add_scalar(f'batch/{stage}_positive_similarity', metrics['positive_similarity'], global_batch)
            plotter.add_scalar(f'batch/{stage}_negative_similarity', metrics['negative_similarity'], global_batch)
            plotter.add_scalar(f'batch/{stage}_similarity_gap', metrics['similarity_gap'], global_batch)
            plotter.plot_all()

    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'acc_imu_to_text': total_acc_i2t / num_batches,
        'acc_text_to_imu': total_acc_t2i / num_batches,
        'positive_similarity': total_pos_sim / num_batches,
        'negative_similarity': total_neg_sim / num_batches,
        'similarity_gap': total_sim_gap / num_batches
    }


def validate(model, label_bank, dataloader, criterion, device, epoch, stage="stage1"):
    """Validate for one epoch."""
    model.eval()
    total_loss, total_acc_i2t, total_acc_t2i = 0.0, 0.0, 0.0
    all_imu_embeddings, all_text_embeddings = [], []
    pbar = tqdm(dataloader, desc=f"[{stage}] Epoch {epoch} Validation")

    with torch.no_grad():
        for batch in pbar:
            data = batch['data'].to(device)
            channel_mask = batch['channel_mask'].to(device)
            label_texts = batch['label_texts']
            metadata = batch['metadata']

            sampling_rates = [m['sampling_rate_hz'] for m in metadata]
            patch_sizes = [m['patch_size_sec'] for m in metadata]
            channel_descriptions = [m['channel_descriptions'] for m in metadata]

            text_embeddings = label_bank.encode(label_texts, normalize=True)

            with autocast('cuda', enabled=device.type == 'cuda'):
                imu_embeddings = model(data, channel_descriptions, channel_mask, sampling_rates, patch_sizes)
                loss, metrics = criterion(imu_embeddings, text_embeddings, label_texts, return_metrics=True)

            total_loss += metrics['loss']
            total_acc_i2t += metrics['acc_imu_to_text']
            total_acc_t2i += metrics['acc_text_to_imu']
            all_imu_embeddings.append(imu_embeddings.cpu())
            all_text_embeddings.append(text_embeddings.cpu())

            pbar.set_postfix({'loss': f"{metrics['loss']:.4f}", 'acc_i2t': f"{metrics['acc_imu_to_text']:.3f}"})

    num_batches = len(dataloader)
    avg_metrics = {'loss': total_loss / num_batches, 'acc_imu_to_text': total_acc_i2t / num_batches,
                   'acc_text_to_imu': total_acc_t2i / num_batches}

    all_imu_embeddings = torch.cat(all_imu_embeddings, dim=0).to(device)
    all_text_embeddings = torch.cat(all_text_embeddings, dim=0).to(device)
    retrieval_metrics = compute_retrieval_metrics(all_imu_embeddings, all_text_embeddings, k_values=[1, 5, 10])
    avg_metrics.update(retrieval_metrics)

    return avg_metrics


def main():
    """Main training function."""
    global CHECKPOINT_DIR

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    CHECKPOINT_DIR = Path(OUTPUT_DIR) / timestamp
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    plot_dir = CHECKPOINT_DIR / "plots"
    plot_dir.mkdir(exist_ok=True)

    print(f"Output directory: {CHECKPOINT_DIR}")

    # Save hyperparameters
    hyperparams = {
        'encoder': {'d_model': D_MODEL, 'num_heads': NUM_HEADS},
        'semantic': {'d_model_fused': D_MODEL_FUSED, 'semantic_dim': SEMANTIC_DIM,
                     'sentence_bert_model': SENTENCE_BERT_MODEL, 'temperature': TEMPERATURE,
                     'use_soft_targets': USE_SOFT_TARGETS, 'soft_target_temperature': SOFT_TARGET_TEMPERATURE,
                     'soft_target_weight': SOFT_TARGET_WEIGHT,
                     'use_prototypes': USE_PROTOTYPES, 'num_prototype_clusters': NUM_PROTOTYPE_CLUSTERS,
                     'prototype_update_interval': PROTOTYPE_UPDATE_INTERVAL, 'prototype_weight': PROTOTYPE_WEIGHT,
                     'use_memory_bank': USE_MEMORY_BANK, 'memory_bank_size': MEMORY_BANK_SIZE},
        'training': {'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LEARNING_RATE, 'warmup_epochs': WARMUP_EPOCHS}
    }
    with open(CHECKPOINT_DIR / 'hyperparameters.json', 'w') as f:
        json.dump(hyperparams, f, indent=2)

    torch.manual_seed(SEED)

    # Initialize models
    print("\nInitializing encoder...")
    encoder = IMUActivityRecognitionEncoder(
        d_model=D_MODEL, num_heads=NUM_HEADS,
        num_temporal_layers=NUM_TEMPORAL_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT, use_cross_channel=USE_CROSS_CHANNEL,
        cnn_channels=CNN_CHANNELS, cnn_kernel_sizes=CNN_KERNEL_SIZES
    ).to(device)

    if PRETRAINED_ENCODER_PATH and Path(PRETRAINED_ENCODER_PATH).exists():
        print(f"Loading pretrained encoder from {PRETRAINED_ENCODER_PATH}")
        checkpoint = torch.load(PRETRAINED_ENCODER_PATH, map_location=device)
        encoder_state_dict = {k.replace('encoder.', ''): v for k, v in checkpoint['model_state_dict'].items() 
                              if k.startswith('encoder.')}
        encoder.load_state_dict(encoder_state_dict)
        print("✓ Pretrained encoder loaded")
    else:
        print("Warning: No pretrained encoder loaded")

    print("Initializing semantic alignment head...")
    semantic_head = SemanticAlignmentHead(
        d_model=D_MODEL, d_model_fused=D_MODEL_FUSED, output_dim=SEMANTIC_DIM,
        num_bottlenecks=NUM_BOTTLENECKS, num_temporal_layers=NUM_SEMANTIC_TEMPORAL_LAYERS,
        num_heads=NUM_HEADS, dim_feedforward=D_MODEL_FUSED * 4, dropout=DROPOUT
    ).to(device)

    model = SemanticAlignmentModel(encoder, semantic_head).to(device)

    print(f"Initializing label bank with {SENTENCE_BERT_MODEL}...")
    label_bank = LabelBank(model_name=SENTENCE_BERT_MODEL, device=device)
    print(f"✓ Label bank initialized (embedding_dim={label_bank.embedding_dim})")

    # Initialize prototype manager if enabled
    prototype_manager = None
    if USE_PROTOTYPES:
        print(f"Initializing prototype manager ({NUM_PROTOTYPE_CLUSTERS} clusters)...")
        prototype_manager = PrototypeManager(
            num_clusters=NUM_PROTOTYPE_CLUSTERS,
            update_interval=PROTOTYPE_UPDATE_INTERVAL,
            device=device
        )
        print("✓ Prototype manager initialized")

    criterion = SemanticAlignmentLoss(
        temperature=TEMPERATURE, use_soft_targets=USE_SOFT_TARGETS,
        soft_target_temperature=SOFT_TARGET_TEMPERATURE, soft_target_weight=SOFT_TARGET_WEIGHT,
        use_prototypes=USE_PROTOTYPES, prototype_weight=PROTOTYPE_WEIGHT,
        prototype_manager=prototype_manager
    )

    # Initialize memory bank if enabled
    memory_bank = None
    if USE_MEMORY_BANK:
        print(f"Initializing memory bank (queue_size={MEMORY_BANK_SIZE}, embedding_dim={SEMANTIC_DIM})...")
        memory_bank = MemoryBank(queue_size=MEMORY_BANK_SIZE, embedding_dim=SEMANTIC_DIM)
        print(f"✓ Memory bank initialized - provides {MEMORY_BANK_SIZE} additional negatives")

    plotter = TrainingPlotter(plot_dir)

    print("\n" + "="*70)
    print("End-to-End Training: Encoder + Semantic Head")
    if USE_SOFT_TARGETS:
        print(f"Using pairwise soft targets (temperature={SOFT_TARGET_TEMPERATURE}, weight={SOFT_TARGET_WEIGHT})")
    if USE_PROTOTYPES:
        print(f"Using prototype soft targets ({NUM_PROTOTYPE_CLUSTERS} clusters, weight={PROTOTYPE_WEIGHT})")
    if USE_MEMORY_BANK:
        print(f"Using memory bank with {MEMORY_BANK_SIZE} queue size (effective batch size: {BATCH_SIZE + MEMORY_BANK_SIZE})")
    print("="*70)

    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(
        data_root=DATA_ROOT, datasets=DATASETS, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, patch_size_per_dataset=PATCH_SIZE_PER_DATASET, seed=SEED
    )

    # Setup optimizer (train all parameters)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=LEARNING_RATE * 0.01)
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_epoch(model, label_bank, train_loader, criterion, optimizer,
                                      device, epoch, scaler, prototype_manager, plotter, "train", memory_bank)
        val_metrics = validate(model, label_bank, val_loader, criterion, device, epoch, "val")

        # Step scheduler after warmup
        if epoch > WARMUP_EPOCHS:
            scheduler.step()

        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        plotter.add_scalar('epoch/lr', current_lr, epoch)

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc(I2T): {train_metrics['acc_imu_to_text']:.4f}, Sim Gap: {train_metrics['similarity_gap']:.3f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Recall@1: {val_metrics['recall@1_avg']:.4f}, Recall@5: {val_metrics['recall@5_avg']:.4f}")

        # Log comprehensive metrics
        plotter.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
        plotter.add_scalar('epoch/train_acc_imu_to_text', train_metrics['acc_imu_to_text'], epoch)
        plotter.add_scalar('epoch/train_acc_text_to_imu', train_metrics['acc_text_to_imu'], epoch)
        plotter.add_scalar('epoch/train_positive_similarity', train_metrics['positive_similarity'], epoch)
        plotter.add_scalar('epoch/train_negative_similarity', train_metrics['negative_similarity'], epoch)
        plotter.add_scalar('epoch/train_similarity_gap', train_metrics['similarity_gap'], epoch)

        plotter.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
        plotter.add_scalar('epoch/val_acc_imu_to_text', val_metrics['acc_imu_to_text'], epoch)
        plotter.add_scalar('epoch/val_acc_text_to_imu', val_metrics['acc_text_to_imu'], epoch)
        plotter.add_scalar('epoch/val_recall@1', val_metrics['recall@1_avg'], epoch)
        plotter.add_scalar('epoch/val_recall@5', val_metrics['recall@5_avg'], epoch)
        plotter.add_scalar('epoch/val_recall@10', val_metrics['recall@10_avg'], epoch)

        plotter.plot_all()

        # Save checkpoints
        if epoch % SAVE_EVERY == 0 or val_metrics['loss'] < best_val_loss:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'hyperparameters': hyperparams
            }
            torch.save(checkpoint, CHECKPOINT_DIR / f'epoch_{epoch}.pt')

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save(checkpoint, CHECKPOINT_DIR / 'best.pt')
                print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")

    plotter.close()
    print("\n" + "="*70)
    print(f"Training complete! Checkpoints: {CHECKPOINT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
