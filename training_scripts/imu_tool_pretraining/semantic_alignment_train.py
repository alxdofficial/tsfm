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
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler
from datetime import datetime
from tqdm import tqdm
import json
import math

from datasets.imu_pretraining_dataset.multi_dataset_loader import create_dataloaders
from imu_activity_recognition_encoder.encoder import IMUActivityRecognitionEncoder
from imu_activity_recognition_encoder.semantic_alignment import SemanticAlignmentHead
from training_scripts.imu_tool_pretraining.label_bank import LabelBank
from training_scripts.imu_tool_pretraining.semantic_loss import SemanticAlignmentLoss, compute_retrieval_metrics
from training_scripts.imu_tool_pretraining.plot_utils import TrainingPlotter, EmbeddingVisualizer
from training_scripts.imu_tool_pretraining.memory_bank import MemoryBank


# ======================== HYPERPARAMETERS ========================

# Data configuration
DATA_ROOT = "/home/alex/code/tsfm/data"
DATASETS = ['uci_har', 'hhar', 'mhealth', 'pamap2', 'wisdm', 'unimib_shar']

PATCH_SIZE_PER_DATASET = {
    'uci_har': 1.0,       # 50 Hz, 2.56s windows → 50 samples/patch, ~2-3 patches/session
    'hhar': 1.0,          # 50 Hz, 2.56s windows → 50 samples/patch, ~2-3 patches/session
    'mhealth': 2.0,       # 50 Hz, health monitoring → 100 samples/patch
    'pamap2': 2.0,        # 100 Hz, complex activities → 200 samples/patch
    'wisdm': 2.0,         # 20 Hz, phone-based → 40 samples/patch
    'unimib_shar': 1.0,   # 50 Hz, ~3s segments → 50 samples/patch, ~3 patches/session
}

MAX_PATCHES_PER_SAMPLE = 48
MAX_SESSIONS_PER_DATASET = 10000  # Limit sessions per dataset for faster experimentation (None = all)

# Encoder configuration (must match pretrained model)
D_MODEL = 384
NUM_HEADS = 8
NUM_TEMPORAL_LAYERS = 4
DIM_FEEDFORWARD = 1536
DROPOUT = 0.1
USE_CROSS_CHANNEL = True
CNN_CHANNELS = [32, 64]  # MUST match checkpoint: good_20251124_193747
CNN_KERNEL_SIZES = [5]  # MUST match checkpoint: single kernel (not multi-scale)
TARGET_PATCH_SIZE = 64  # Fixed timesteps per patch after interpolation

# Semantic alignment configuration
D_MODEL_FUSED = 384  # Dimension after cross-channel fusion (match D_MODEL to avoid bottleneck)
SEMANTIC_DIM = 384  # Final embedding dimension (must match SentenceBERT)
NUM_BOTTLENECKS = 4  # Multiple bottlenecks to reduce information bottleneck (9:1 → 2.25:1 compression)
NUM_SEMANTIC_TEMPORAL_LAYERS = 2  # Temporal attention layers in semantic head

# Text encoder configuration
SENTENCE_BERT_MODEL = 'all-MiniLM-L6-v2'  # 384-dim embeddings, fast

# Training configuration
OUTPUT_DIR = "training_output/semantic_alignment"  # Note: plots go to semantic_alignment/<timestamp>/plots/
CHECKPOINT_DIR = None  # Will be set in main()
PRETRAINED_ENCODER_PATH = "training_output/imu_pretraining/20251201_201049/latest.pt"  # Pretrained tokenizer
FREEZE_ENCODER = False  # Unfreeze encoder to learn discriminative representations
SAVE_EVERY = 5

# Resume configuration - set to a folder path to resume training from that checkpoint
# Example: RESUME_FROM = "training_output/semantic_alignment/20251124_234942"
RESUME_FROM = None  # Set to folder path to resume, or None to start fresh
SEED = 42
MAX_GRAD_NORM = 1.0  # Gradient clipping threshold

# Training hyperparameters
EPOCHS = 100
BATCH_SIZE = 32  # Micro-batch size
ACCUMULATION_STEPS = 16  # Effective batch = 32 × 16 = 512
LEARNING_RATE = 1e-4  # Reduced from 5e-4 - 5e-4 too aggressive for frozen encoder with batch_size=256
WARMUP_EPOCHS = 3

# Shared parameters
NUM_WORKERS = 8  # Increased from 4 for better CPU parallelism with large batches
PREFETCH_FACTOR = 4  # Prefetch 4 batches per worker (32 total batches ahead)
PERSISTENT_WORKERS = True  # Keep workers alive between epochs
WEIGHT_DECAY = 1e-5  # Reduced from 0.01 - high weight decay causes representation collapse in contrastive learning (SimCLR uses 1e-6)
TEMPERATURE = 0.07  # CLIP default - sharper discrimination for better gradient signal

# Soft targets configuration
# CRITICAL: Soft targets are ESSENTIAL for label augmentation to prevent treating synonyms as negatives
# With augmentation, batch contains duplicates like "walking", "strolling", "person walking"
# Hard targets treat these as negatives (contradictory!) → Soft targets weight by semantic similarity (correct!)
USE_SOFT_TARGETS = True  # ENABLED: Essential for label augmentation (prevents collapse from contradictory gradients)
SOFT_TARGET_TEMPERATURE = 0.5  # Sharpen soft targets - gives ~7x more weight to exact match
SOFT_TARGET_WEIGHT = 1.0  # Pure soft targets with adaptive recalibration

# Memory bank configuration (MoCo-style queue for more negatives)
# With soft targets, queue items with same semantic labels share probability mass.
# This is actually CORRECT for foundation models: any "walking" IMU should match any "walking" text.
# Gradient direction is preserved (0.99 cosine similarity), magnitude slightly diluted.
USE_MEMORY_BANK = True
MEMORY_BANK_SIZE = 256  # Provides 32 + 256 = 288 negatives per step

# Plotting
PLOT_EVERY_N_BATCHES = 10

# Embedding visualization
VISUALIZE_EMBEDDINGS = True  # Generate UMAP plots showing IMU-text alignment
VISUALIZE_EVERY_N_EPOCHS = 1  # How often to generate embedding plots (1 = every epoch)
UMAP_N_NEIGHBORS = 15  # UMAP parameter: 5-50, higher = more global structure
UMAP_MIN_DIST = 0.1  # UMAP parameter: 0.0-0.99, lower = tighter clusters

# Debug configuration
DEBUG_METRIC_FREQUENCY = 10  # Compute expensive debug metrics every N batches (not every batch)

# =================================================================


class SemanticAlignmentModel(nn.Module):
    """Complete model for semantic alignment combining encoder + semantic head."""

    def __init__(self, encoder: IMUActivityRecognitionEncoder, semantic_head: SemanticAlignmentHead):
        super().__init__()
        self.encoder = encoder
        self.semantic_head = semantic_head

    def forward(self, data, channel_descriptions, channel_mask, sampling_rates, patch_sizes):
        batch_size = data.shape[0]
        device = data.device

        # Step 1: Preprocess all samples (must be per-sample due to different sampling rates)
        all_patches = []
        all_channel_descs = []
        valid_samples = []

        with torch.no_grad():
            for i in range(batch_size):
                patches, _ = self.encoder.preprocess(
                    data[i], sampling_rate_hz=sampling_rates[i], patch_size_sec=patch_sizes[i]
                )

                if patches is None or len(patches) == 0:
                    all_patches.append(None)
                    all_channel_descs.append(None)
                    valid_samples.append(False)
                    continue

                if len(patches) > MAX_PATCHES_PER_SAMPLE:
                    patches = patches[:MAX_PATCHES_PER_SAMPLE]

                # Pad channel descriptions to match data's channel count
                num_channels = data[i].shape[1]
                channel_descs = channel_descriptions[i]
                padded_descs = channel_descs + ["[PAD]"] * (num_channels - len(channel_descs))

                all_patches.append(patches)
                all_channel_descs.append(padded_descs)
                valid_samples.append(True)

        # Step 2: Batch all valid patches together
        valid_indices = [i for i, v in enumerate(valid_samples) if v]

        if len(valid_indices) == 0:
            # All samples are invalid - return empty output
            max_channels = data.shape[2]
            padded_encoder_output = torch.zeros(batch_size, 1, max_channels, D_MODEL, device=device)
            padded_patch_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
            return self.semantic_head(padded_encoder_output, channel_mask=channel_mask,
                                       patch_mask=padded_patch_mask, normalize=True)

        # Find max patches among valid samples
        max_patches_valid = max(len(all_patches[i]) for i in valid_indices)
        max_channels = data.shape[2]

        # Create batched tensor for valid samples only
        num_valid = len(valid_indices)
        batched_patches = torch.zeros(num_valid, max_patches_valid, TARGET_PATCH_SIZE, max_channels, device=device)
        batched_channel_descs = []

        for batch_idx, sample_idx in enumerate(valid_indices):
            patches = all_patches[sample_idx]
            num_patches = len(patches)
            batched_patches[batch_idx, :num_patches] = patches
            batched_channel_descs.append(all_channel_descs[sample_idx])

        # Step 3: Encode all valid samples
        # Wrap in no_grad when encoder is frozen (saves 30-40% training time!)
        if FREEZE_ENCODER:
            with torch.no_grad():
                encoded_batch = self.encoder(batched_patches, batched_channel_descs)
        else:
            encoded_batch = self.encoder(batched_patches, batched_channel_descs)
        # encoded_batch shape: (num_valid, max_patches_valid, max_channels, d_model)

        # Step 4: Map outputs back to original batch positions
        encoder_outputs = [None] * batch_size
        patch_masks = [None] * batch_size

        for batch_idx, sample_idx in enumerate(valid_indices):
            num_patches = len(all_patches[sample_idx])
            encoder_outputs[sample_idx] = encoded_batch[batch_idx, :num_patches]
            patch_masks[sample_idx] = torch.ones(num_patches, dtype=torch.bool, device=device)

        # Step 5: Final padding to uniform dimensions
        max_patches = max(len(p) for p in patch_masks if p is not None)
        padded_encoder_output = torch.zeros(batch_size, max_patches, max_channels, D_MODEL, device=device)
        padded_patch_mask = torch.zeros(batch_size, max_patches, dtype=torch.bool, device=device)

        for i, (enc_out, p_mask) in enumerate(zip(encoder_outputs, patch_masks)):
            if enc_out is not None:
                num_patches, num_channels, _ = enc_out.shape
                padded_encoder_output[i, :num_patches, :num_channels, :] = enc_out
                padded_patch_mask[i, :num_patches] = p_mask

        return self.semantic_head(padded_encoder_output, channel_mask=channel_mask,
                                   patch_mask=padded_patch_mask, normalize=True)

    def get_attention_stats(self, data, channel_descriptions, channel_mask, sampling_rates, patch_sizes):
        """Get attention statistics from cross-channel fusion (for debugging)."""
        # Re-run forward up to encoder output (without gradient)
        batch_size = data.shape[0]
        device = data.device

        with torch.no_grad():
            all_patches = []
            all_channel_descs = []
            valid_samples = []

            for i in range(batch_size):
                patches, _ = self.encoder.preprocess(
                    data[i], sampling_rate_hz=sampling_rates[i], patch_size_sec=patch_sizes[i]
                )
                if patches is None or len(patches) == 0:
                    valid_samples.append(False)
                    continue
                if len(patches) > MAX_PATCHES_PER_SAMPLE:
                    patches = patches[:MAX_PATCHES_PER_SAMPLE]
                num_channels = data[i].shape[1]
                channel_descs = channel_descriptions[i]
                padded_descs = channel_descs + ["[PAD]"] * (num_channels - len(channel_descs))
                all_patches.append(patches)
                all_channel_descs.append(padded_descs)
                valid_samples.append(True)

            valid_indices = [i for i, v in enumerate(valid_samples) if v]
            if len(valid_indices) == 0:
                return {}

            max_patches_valid = max(len(all_patches[i]) for i in valid_indices)
            max_channels = data.shape[2]
            num_valid = len(valid_indices)

            batched_patches = torch.zeros(num_valid, max_patches_valid, TARGET_PATCH_SIZE, max_channels, device=device)
            batched_channel_descs = []

            for batch_idx, sample_idx in enumerate(valid_indices):
                patches = all_patches[sample_idx]
                num_patches = len(patches)
                batched_patches[batch_idx, :num_patches] = patches
                batched_channel_descs.append(all_channel_descs[sample_idx])

            encoded_batch = self.encoder(batched_patches, batched_channel_descs)

            # Get attention stats from semantic head
            return self.semantic_head.get_attention_stats(encoded_batch, channel_mask)


def compute_debug_metrics(imu_embeddings, text_embeddings, imu_queue=None, text_queue=None):
    """
    Compute comprehensive debugging metrics for diagnosing training issues.

    Returns dict with:
    - Representation collapse indicators (std, diversity)
    - Similarity distributions (positive vs negative)
    - Memory bank quality (if queue provided)
    - Intra-batch similarity (collapse detection)
    - Hard negative analysis
    """
    # Convert to fp32 for all operations (mixed precision compatibility)
    imu_embeddings = imu_embeddings.float()
    text_embeddings = text_embeddings.float()
    if imu_queue is not None:
        imu_queue = imu_queue.float()
    if text_queue is not None:
        text_queue = text_queue.float()

    batch_size = imu_embeddings.shape[0]
    debug_metrics = {}

    # 1. Representation Collapse Detection
    # Check if embeddings are diverse or collapsed to similar vectors
    # For L2-normalized d=384 vectors, expected std ≈ 1/sqrt(384) ≈ 0.051
    debug_metrics['imu_std'] = imu_embeddings.std(dim=0).mean().item()
    debug_metrics['text_std'] = text_embeddings.std(dim=0).mean().item()

    # Pairwise distances (diversity) - for uniform on hypersphere, expected ~1.0-1.4
    if batch_size > 1:
        debug_metrics['imu_diversity'] = torch.pdist(imu_embeddings).mean().item()
        debug_metrics['text_diversity'] = torch.pdist(text_embeddings).mean().item()
    else:
        debug_metrics['imu_diversity'] = 0.0
        debug_metrics['text_diversity'] = 0.0

    # 2. Intra-batch similarity (collapse detection)
    # IMU embeddings should be diverse (low intra-batch similarity)
    # Text embeddings may have higher similarity (same labels)
    imu_sim_matrix = torch.matmul(imu_embeddings, imu_embeddings.T)
    if batch_size > 1:
        imu_intra_mask = ~torch.eye(batch_size, dtype=torch.bool, device=imu_sim_matrix.device)
        debug_metrics['imu_intra_batch_sim'] = imu_sim_matrix[imu_intra_mask].mean().item()
        # Collapse indicator: if all IMU embeddings become same vector, this approaches 1.0
        debug_metrics['imu_intra_batch_sim_max'] = imu_sim_matrix[imu_intra_mask].max().item()
    else:
        debug_metrics['imu_intra_batch_sim'] = 0.0
        debug_metrics['imu_intra_batch_sim_max'] = 0.0

    # 3. Similarity Distribution Analysis
    sim_matrix = torch.matmul(imu_embeddings, text_embeddings.T)  # (batch, batch)

    # Positive pairs (diagonal - matched IMU and text)
    pos_sims = torch.diagonal(sim_matrix)
    debug_metrics['pos_sim_mean'] = pos_sims.mean().item()
    debug_metrics['pos_sim_std'] = pos_sims.std().item()
    debug_metrics['pos_sim_min'] = pos_sims.min().item()
    debug_metrics['pos_sim_max'] = pos_sims.max().item()

    # Negative pairs (off-diagonal - mismatched IMU and text)
    if batch_size > 1:
        neg_mask = ~torch.eye(batch_size, dtype=torch.bool, device=sim_matrix.device)
        neg_sims = sim_matrix[neg_mask]
        debug_metrics['neg_sim_mean'] = neg_sims.mean().item()
        debug_metrics['neg_sim_std'] = neg_sims.std().item()
        debug_metrics['neg_sim_max'] = neg_sims.max().item()  # Hardest negative

        # Similarity gap (should be positive and growing during training)
        debug_metrics['sim_gap'] = debug_metrics['pos_sim_mean'] - debug_metrics['neg_sim_mean']

        # 4. Hard negative analysis
        # How many negatives have higher similarity than the positive?
        # This indicates samples where model is confused
        pos_expanded = pos_sims.unsqueeze(1)  # (batch, 1)
        hard_negatives_per_sample = (sim_matrix > pos_expanded).sum(dim=1).float()  # Exclude diagonal
        debug_metrics['hard_negative_count'] = hard_negatives_per_sample.mean().item()
        # Rank of positive pair (0 = correct, higher = worse)
        debug_metrics['positive_rank'] = hard_negatives_per_sample.mean().item()

        # Margin: how much does positive exceed the hardest negative
        # For each row, find max negative similarity
        neg_max_per_row = (sim_matrix * neg_mask.float()).max(dim=1)[0]
        margin = pos_sims - neg_max_per_row
        debug_metrics['margin_mean'] = margin.mean().item()
        debug_metrics['margin_min'] = margin.min().item()  # Worst case margin
    else:
        debug_metrics['neg_sim_mean'] = 0.0
        debug_metrics['neg_sim_std'] = 0.0
        debug_metrics['neg_sim_max'] = 0.0
        debug_metrics['sim_gap'] = 0.0
        debug_metrics['hard_negative_count'] = 0.0
        debug_metrics['positive_rank'] = 0.0
        debug_metrics['margin_mean'] = 0.0
        debug_metrics['margin_min'] = 0.0

    # 5. Memory Bank Quality (if queue provided)
    if imu_queue is not None and len(imu_queue) > 0:
        # Queue diversity
        queue_sample = imu_queue[:min(100, len(imu_queue))]  # Sample to avoid expensive computation
        debug_metrics['queue_diversity'] = torch.pdist(queue_sample).mean().item()

        # Queue staleness: how different is queue from current batch?
        queue_vs_current = torch.cdist(imu_embeddings, imu_queue[:100]).mean().item()
        debug_metrics['queue_staleness'] = queue_vs_current

        # Queue similarity to current batch (for hard negative mining from queue)
        queue_sim = torch.matmul(imu_embeddings, imu_queue.T)  # (batch, queue_size)
        debug_metrics['queue_sim_mean'] = queue_sim.mean().item()
        debug_metrics['queue_sim_max'] = queue_sim.max().item()  # Hardest negative from queue

    return debug_metrics


def warmup_memory_bank(model, label_bank, dataloader, memory_bank, device, num_batches=None):
    """
    Warmup memory bank by filling the queue with embeddings before training.
    
    This reduces volatility in early training by ensuring the queue has diverse
    negatives from the start, rather than being filled with zeros or early
    low-quality embeddings.
    
    Args:
        model: The semantic alignment model
        label_bank: LabelBank for encoding text labels
        dataloader: Training dataloader
        memory_bank: MemoryBank instance to fill
        device: Device to run on
        num_batches: Number of batches to use (None = fill entire queue)
    """
    if memory_bank is None:
        return
    
    print(f"\nWarming up memory bank (queue_size={memory_bank.queue_size})...")
    model.eval()  # Eval mode during warmup (no training)
    
    # Calculate how many batches needed to fill queue
    if num_batches is None:
        batch_size = dataloader.batch_size
        num_batches = math.ceil(memory_bank.queue_size / batch_size)
    
    num_batches = min(num_batches, len(dataloader))
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            data = batch['data'].to(device)
            channel_mask = batch['channel_mask'].to(device)
            label_texts = batch['label_texts']
            metadata = batch['metadata']
            
            sampling_rates = [m['sampling_rate_hz'] for m in metadata]
            patch_sizes = [m['patch_size_sec'] for m in metadata]
            channel_descriptions = [m['channel_descriptions'] for m in metadata]

            # Encode text embeddings
            text_embeddings = label_bank.encode(label_texts, normalize=True)

            # Forward pass (no gradients)
            imu_embeddings = model(data, channel_descriptions, channel_mask, sampling_rates, patch_sizes)

            # Update memory bank with both embeddings
            memory_bank.update(imu_embeddings, text_embeddings)
            
            # Print progress
            filled = min((batch_idx + 1) * len(label_texts), memory_bank.queue_size)
            print(f"  Batch {batch_idx + 1}/{num_batches}: Queue filled {filled}/{memory_bank.queue_size} "
                  f"({100 * filled / memory_bank.queue_size:.1f}%)")
    
    print(f"✓ Memory bank warmup complete ({memory_bank.ptr} embeddings)")
    
    # Return model to training mode if it wasn't frozen
    if not FREEZE_ENCODER:
        model.train()


def _compute_per_layer_grad_norms(model, criterion=None):
    """
    Compute gradient norms for each sub-component of the semantic head.

    SemanticAlignmentHead sub-components:
    - cross_channel_fusion: Perceiver-style bottleneck attention
    - temporal_attention: TransformerEncoder over patches
    - attention_pooling: CLS token attention pooling
    - projection_head: 3-layer MLP to semantic space

    Also tracks logit_scale gradient if criterion is provided.

    Returns:
        Dict mapping component name to gradient norm
    """
    grad_norms = {}

    # Define components to track (skip frozen encoder components)
    components = {
        'cross_channel_fusion': 'semantic_head.cross_channel_fusion',
        'temporal_attention': 'semantic_head.temporal_attention',
        'attention_pooling': 'semantic_head.attention_pooling',
        'projection_head': 'semantic_head.projection_head',
    }

    for name, prefix in components.items():
        grad_norm = sum(
            p.grad.norm().item() ** 2
            for n, p in model.named_parameters()
            if prefix in n and p.grad is not None
        ) ** 0.5
        grad_norms[name] = grad_norm

    # Track logit_scale gradient (learnable temperature)
    if criterion is not None and hasattr(criterion, 'infonce'):
        logit_scale = criterion.infonce.logit_scale
        if logit_scale.grad is not None:
            grad_norms['logit_scale'] = logit_scale.grad.abs().item()
        else:
            grad_norms['logit_scale'] = 0.0

    return grad_norms


def train_epoch(model, label_bank, dataloader, criterion, optimizer, device, epoch, scaler, plotter=None, stage="stage1", memory_bank=None):
    """Train for one epoch."""
    model.train()

    # Keep encoder in evaluation mode if frozen (prevents dropout and batch norm updates)
    if FREEZE_ENCODER and hasattr(model, 'encoder'):
        model.encoder.eval()

    # Track all metrics
    total_loss = 0.0
    total_acc_i2t = 0.0
    total_acc_t2i = 0.0
    total_pos_sim = 0.0
    total_neg_sim = 0.0
    total_sim_gap = 0.0
    total_grad_norm = 0.0

    # Track debug metrics
    total_imu_std = 0.0
    total_text_std = 0.0
    total_imu_diversity = 0.0
    total_debug_sim_gap = 0.0
    total_queue_diversity = 0.0

    # Track per-layer gradient norms (dict for each sub-component)
    total_grad_norms = {
        'cross_channel_fusion': 0.0,
        'temporal_attention': 0.0,
        'attention_pooling': 0.0,
        'projection_head': 0.0,
        'logit_scale': 0.0,  # Learnable temperature
    }

    pbar = tqdm(dataloader, desc=f"[{stage}] Epoch {epoch} Training")

    # Initialize gradients at start of epoch
    optimizer.zero_grad()

    # Track accumulation steps for proper gradient logging
    accum_step_count = 0

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

        # Get queue embeddings from memory bank if enabled (no gradients needed)
        if memory_bank is not None and USE_MEMORY_BANK:
            with torch.no_grad():
                imu_queue, text_queue = memory_bank.get_queue_embeddings(device)
        else:
            imu_queue, text_queue = None, None

        with autocast('cuda', enabled=device.type == 'cuda'):
            imu_embeddings = model(data, channel_descriptions, channel_mask, sampling_rates, patch_sizes)
            loss, metrics = criterion(imu_embeddings, text_embeddings, label_texts,
                                     return_metrics=True, imu_queue=imu_queue, text_queue=text_queue)

        # Scale loss for gradient accumulation
        scaled_loss = loss / ACCUMULATION_STEPS

        # Compute debug metrics periodically (not every batch - expensive)
        # Embeddings are already normalized by model and label_bank
        debug_metrics = {}
        if batch_idx % DEBUG_METRIC_FREQUENCY == 0:
            with torch.no_grad():
                debug_metrics = compute_debug_metrics(imu_embeddings, text_embeddings, imu_queue, text_queue)
                # Add attention stats from cross-channel fusion
                attn_stats = model.get_attention_stats(data, channel_descriptions, channel_mask, sampling_rates, patch_sizes)
                debug_metrics.update(attn_stats)

        # Backward pass (accumulate gradients)
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # Initialize grad_norm for this batch (will be set on optimizer step)
        grad_norm = torch.tensor(0.0)
        batch_grad_norms = {k: 0.0 for k in total_grad_norms}

        # Step optimizer every ACCUMULATION_STEPS or at end of epoch
        is_accumulation_step = (batch_idx + 1) % ACCUMULATION_STEPS == 0
        is_last_batch = (batch_idx + 1) == len(dataloader)

        if is_accumulation_step or is_last_batch:
            # Compute per-layer gradient norms periodically (based on accumulation steps, not batch idx)
            # This ensures we actually capture gradients when they exist (after accumulation)
            should_log_grads = accum_step_count % DEBUG_METRIC_FREQUENCY == 0

            if scaler is not None:
                scaler.unscale_(optimizer)

                # Per-layer gradient norms (expensive - only compute periodically)
                if should_log_grads:
                    batch_grad_norms = _compute_per_layer_grad_norms(model, criterion)

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Per-layer gradient norms (expensive - only compute periodically)
                if should_log_grads:
                    batch_grad_norms = _compute_per_layer_grad_norms(model, criterion)

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                optimizer.step()

            # Reset gradients for next accumulation window
            optimizer.zero_grad()

            # Increment accumulation step counter
            accum_step_count += 1

            # Log gradient norms immediately after optimizer step (when gradients exist)
            if plotter is not None:
                global_batch_for_grad = (epoch - 1) * len(dataloader) + batch_idx
                plotter.add_scalar(f'batch/{stage}_grad_norm', grad_norm.item(), global_batch_for_grad)
                # Per-layer gradient norms (only when actually computed, not zeros)
                if should_log_grads:
                    for comp_name, comp_grad_norm in batch_grad_norms.items():
                        plotter.add_scalar(f'batch/debug_{stage}_{comp_name}_grad_norm', comp_grad_norm, global_batch_for_grad)

        # Update memory bank with current batch embeddings (no gradients needed)
        if memory_bank is not None and USE_MEMORY_BANK:
            with torch.no_grad():
                memory_bank.update(imu_embeddings.detach(), text_embeddings.detach())

        # Accumulate metrics
        total_loss += metrics['loss']
        total_acc_i2t += metrics['acc_imu_to_text']
        total_acc_t2i += metrics['acc_text_to_imu']
        total_pos_sim += metrics['positive_similarity']
        total_neg_sim += metrics['negative_similarity']
        total_sim_gap += metrics['similarity_gap']
        total_grad_norm += grad_norm.item()

        # Accumulate debug metrics (only when computed)
        if debug_metrics:
            total_imu_std += debug_metrics['imu_std']
            total_text_std += debug_metrics['text_std']
            total_imu_diversity += debug_metrics['imu_diversity']
            total_debug_sim_gap += debug_metrics['sim_gap']
            if 'queue_diversity' in debug_metrics:
                total_queue_diversity += debug_metrics['queue_diversity']

        # Accumulate per-layer gradient norms
        for k in total_grad_norms:
            total_grad_norms[k] += batch_grad_norms[k]

        pbar.set_postfix({
            'loss': f"{metrics['loss']:.4f}",
            'acc': f"{metrics['acc_imu_to_text']:.3f}",
            'sim_gap': f"{metrics['similarity_gap']:.3f}",
            'grad': f"{grad_norm.item():.2f}",
            'imu_std': f"{debug_metrics.get('imu_std', 0.0):.3f}"
        })

        # Batch-level plotting (every N batches)
        if plotter is not None and batch_idx % PLOT_EVERY_N_BATCHES == 0:
            global_batch = (epoch - 1) * len(dataloader) + batch_idx
            plotter.add_scalar(f'batch/{stage}_loss', metrics['loss'], global_batch)
            plotter.add_scalar(f'batch/{stage}_acc_imu_to_text', metrics['acc_imu_to_text'], global_batch)
            plotter.add_scalar(f'batch/{stage}_acc_text_to_imu', metrics['acc_text_to_imu'], global_batch)
            plotter.add_scalar(f'batch/{stage}_positive_similarity', metrics['positive_similarity'], global_batch)
            plotter.add_scalar(f'batch/{stage}_negative_similarity', metrics['negative_similarity'], global_batch)
            plotter.add_scalar(f'batch/{stage}_similarity_gap', metrics['similarity_gap'], global_batch)
            # Note: grad_norm is logged separately in the accumulation step block (where gradients actually exist)

            # Loss component metrics
            if 'loss_imu_to_text' in metrics:
                plotter.add_scalar(f'batch/{stage}_loss_imu_to_text', metrics['loss_imu_to_text'], global_batch)
                plotter.add_scalar(f'batch/{stage}_loss_text_to_imu', metrics['loss_text_to_imu'], global_batch)
            if 'logit_scale' in metrics:
                plotter.add_scalar(f'batch/{stage}_logit_scale', metrics['logit_scale'], global_batch)
            if 'logits_std' in metrics:
                plotter.add_scalar(f'batch/{stage}_logits_mean', metrics['logits_mean'], global_batch)
                plotter.add_scalar(f'batch/{stage}_logits_std', metrics['logits_std'], global_batch)

            # Soft target diagnostics
            if 'soft_target_entropy_ratio' in metrics:
                plotter.add_scalar(f'batch/{stage}_soft_target_entropy_ratio', metrics['soft_target_entropy_ratio'], global_batch)
                plotter.add_scalar(f'batch/{stage}_true_positive_target_prob', metrics['true_positive_target_prob'], global_batch)

            # Debug metrics (only plot when computed)
            if debug_metrics:
                # Collapse detection metrics
                plotter.add_scalar(f'batch/debug_{stage}_imu_std', debug_metrics['imu_std'], global_batch)
                plotter.add_scalar(f'batch/debug_{stage}_text_std', debug_metrics['text_std'], global_batch)
                plotter.add_scalar(f'batch/debug_{stage}_imu_diversity', debug_metrics['imu_diversity'], global_batch)
                plotter.add_scalar(f'batch/debug_{stage}_imu_intra_batch_sim', debug_metrics['imu_intra_batch_sim'], global_batch)
                plotter.add_scalar(f'batch/debug_{stage}_imu_intra_batch_sim_max', debug_metrics['imu_intra_batch_sim_max'], global_batch)

                # Similarity distribution metrics
                plotter.add_scalar(f'batch/debug_{stage}_pos_sim_mean', debug_metrics['pos_sim_mean'], global_batch)
                plotter.add_scalar(f'batch/debug_{stage}_neg_sim_mean', debug_metrics['neg_sim_mean'], global_batch)
                plotter.add_scalar(f'batch/debug_{stage}_sim_gap', debug_metrics['sim_gap'], global_batch)

                # Hard negative analysis
                plotter.add_scalar(f'batch/debug_{stage}_hard_negative_count', debug_metrics['hard_negative_count'], global_batch)
                plotter.add_scalar(f'batch/debug_{stage}_positive_rank', debug_metrics['positive_rank'], global_batch)
                plotter.add_scalar(f'batch/debug_{stage}_margin_mean', debug_metrics['margin_mean'], global_batch)
                plotter.add_scalar(f'batch/debug_{stage}_margin_min', debug_metrics['margin_min'], global_batch)

                # Memory bank metrics
                if 'queue_diversity' in debug_metrics:
                    plotter.add_scalar(f'batch/debug_{stage}_queue_diversity', debug_metrics['queue_diversity'], global_batch)
                    plotter.add_scalar(f'batch/debug_{stage}_queue_staleness', debug_metrics['queue_staleness'], global_batch)
                    plotter.add_scalar(f'batch/debug_{stage}_queue_sim_mean', debug_metrics['queue_sim_mean'], global_batch)
                    plotter.add_scalar(f'batch/debug_{stage}_queue_sim_max', debug_metrics['queue_sim_max'], global_batch)

                # Cross-channel attention stats
                if 'cross_channel_attn_entropy_ratio' in debug_metrics:
                    plotter.add_scalar(f'batch/debug_{stage}_attn_entropy_ratio', debug_metrics['cross_channel_attn_entropy_ratio'], global_batch)
                    plotter.add_scalar(f'batch/debug_{stage}_attn_max', debug_metrics['cross_channel_attn_max'], global_batch)
                    plotter.add_scalar(f'batch/debug_{stage}_attn_std', debug_metrics['cross_channel_attn_std'], global_batch)

                # Note: Per-layer gradient norms are logged in the accumulation step block

            plotter.plot_all()

    num_batches = len(dataloader)
    # Debug metrics are only computed every DEBUG_METRIC_FREQUENCY batches
    debug_count = (num_batches + DEBUG_METRIC_FREQUENCY - 1) // DEBUG_METRIC_FREQUENCY  # Ceiling division
    debug_count = max(debug_count, 1)  # Avoid division by zero

    return {
        'loss': total_loss / num_batches,
        'acc_imu_to_text': total_acc_i2t / num_batches,
        'acc_text_to_imu': total_acc_t2i / num_batches,
        'positive_similarity': total_pos_sim / num_batches,
        'negative_similarity': total_neg_sim / num_batches,
        'similarity_gap': total_sim_gap / num_batches,
        'grad_norm': total_grad_norm / num_batches,
        # Debug metrics (computed every DEBUG_METRIC_FREQUENCY batches)
        'imu_std': total_imu_std / debug_count,
        'text_std': total_text_std / debug_count,
        'imu_diversity': total_imu_diversity / debug_count,
        'debug_sim_gap': total_debug_sim_gap / debug_count,
        'queue_diversity': total_queue_diversity / debug_count if USE_MEMORY_BANK else 0.0,
        # Per-component gradient norms
        **{f'{k}_grad_norm': v / debug_count for k, v in total_grad_norms.items()}
    }


def validate(model, label_bank, dataloader, criterion, device, epoch, stage="stage1", plot_dir=None):
    """Validate for one epoch."""
    model.eval()
    total_loss, total_acc_i2t, total_acc_t2i = 0.0, 0.0, 0.0
    all_imu_embeddings, all_text_embeddings = [], []
    all_labels = []
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
                _, metrics = criterion(imu_embeddings, text_embeddings, label_texts, return_metrics=True)

            total_loss += metrics['loss']
            total_acc_i2t += metrics['acc_imu_to_text']
            total_acc_t2i += metrics['acc_text_to_imu']
            # Keep embeddings on GPU for faster concatenation
            all_imu_embeddings.append(imu_embeddings)
            all_text_embeddings.append(text_embeddings)
            all_labels.extend(label_texts)

            pbar.set_postfix({'loss': f"{metrics['loss']:.4f}", 'acc_i2t': f"{metrics['acc_imu_to_text']:.3f}"})

    num_batches = len(dataloader)
    avg_metrics = {'loss': total_loss / num_batches, 'acc_imu_to_text': total_acc_i2t / num_batches,
                   'acc_text_to_imu': total_acc_t2i / num_batches}

    # Concatenate on GPU (faster than CPU→GPU transfer)
    all_imu_embeddings = torch.cat(all_imu_embeddings, dim=0)
    all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
    retrieval_metrics = compute_retrieval_metrics(all_imu_embeddings, all_text_embeddings, k_values=[1, 5, 10])
    avg_metrics.update(retrieval_metrics)

    # Generate embedding visualization if enabled
    if VISUALIZE_EMBEDDINGS and plot_dir is not None and (epoch % VISUALIZE_EVERY_N_EPOCHS == 0):
        print(f"\n[Epoch {epoch}] Generating embedding visualization...")
        visualizer = EmbeddingVisualizer(
            output_dir=plot_dir,
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST
        )
        visualizer.plot_embedding_alignment_2d(
            imu_embeddings=all_imu_embeddings,
            text_embeddings=all_text_embeddings,
            labels=all_labels,
            epoch=epoch,
            metrics={
                'alignment_score': avg_metrics.get('acc_imu_to_text', 0.0),
                'gap': avg_metrics.get('recall@1_avg', 0.0)
            }
        )

    return avg_metrics


def main():
    """Main training function."""
    global CHECKPOINT_DIR

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Handle resume vs fresh start
    start_epoch = 1
    resume_checkpoint = None

    if RESUME_FROM and Path(RESUME_FROM).exists():
        # Resume from existing folder
        CHECKPOINT_DIR = Path(RESUME_FROM)
        plot_dir = CHECKPOINT_DIR / "plots"
        print(f"Resuming training from: {CHECKPOINT_DIR}")

        # Find the latest checkpoint
        checkpoint_files = list(CHECKPOINT_DIR.glob("epoch_*.pt"))
        if checkpoint_files:
            # Sort by epoch number and get the latest
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[1]))
            resume_checkpoint = torch.load(latest_checkpoint, map_location=device)
            start_epoch = resume_checkpoint['epoch'] + 1
            print(f"✓ Found checkpoint: {latest_checkpoint.name} (epoch {resume_checkpoint['epoch']})")
            print(f"  Will resume from epoch {start_epoch}")
        else:
            print("Warning: No checkpoint files found, starting from epoch 1")
    else:
        # Fresh start with new timestamp folder
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
                     'use_memory_bank': USE_MEMORY_BANK, 'memory_bank_size': MEMORY_BANK_SIZE},
        'training': {'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'accumulation_steps': ACCUMULATION_STEPS,
                     'effective_batch_size': BATCH_SIZE * ACCUMULATION_STEPS,
                     'lr': LEARNING_RATE, 'warmup_epochs': WARMUP_EPOCHS, 'max_grad_norm': MAX_GRAD_NORM}
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
        # Use strict=False to ignore channel encoding model weights (loaded dynamically when needed)
        _, unexpected_keys = encoder.load_state_dict(encoder_state_dict, strict=False)
        if unexpected_keys:
            print(f"  Note: Ignoring {len(unexpected_keys)} channel encoding weights (will be loaded dynamically)")
        print("✓ Pretrained encoder loaded")

        # Freeze encoder if requested (only train semantic head)
        if FREEZE_ENCODER:
            for param in encoder.parameters():
                param.requires_grad = False
            # Set encoder to evaluation mode (disables dropout, freezes batch/group norm stats)
            encoder.eval()
            print("✓ Encoder frozen and set to evaluation mode (only training semantic head)")
    else:
        print("Warning: No pretrained encoder loaded")

    print("Initializing semantic alignment head...")
    semantic_head = SemanticAlignmentHead(
        d_model=D_MODEL, d_model_fused=D_MODEL_FUSED, output_dim=SEMANTIC_DIM,
        num_bottlenecks=NUM_BOTTLENECKS, num_temporal_layers=NUM_SEMANTIC_TEMPORAL_LAYERS,
        num_heads=NUM_HEADS, dim_feedforward=D_MODEL_FUSED * 4, dropout=DROPOUT
    ).to(device)

    model = SemanticAlignmentModel(encoder, semantic_head).to(device)

    # Load model state if resuming
    if resume_checkpoint is not None:
        # Use strict=False to handle dynamically-loaded channel encoding weights
        # The checkpoint may contain encoder.positional_encoding.channel_encoding weights
        # that are loaded lazily and won't exist in a freshly initialized model
        missing_keys, unexpected_keys = model.load_state_dict(resume_checkpoint['model_state_dict'], strict=False)
        if unexpected_keys:
            # Filter out expected channel encoding keys
            channel_encoding_keys = [k for k in unexpected_keys if 'channel_encoding' in k]
            other_unexpected = [k for k in unexpected_keys if 'channel_encoding' not in k]
            if channel_encoding_keys:
                print(f"  Note: Ignoring {len(channel_encoding_keys)} channel encoding weights (loaded dynamically)")
            if other_unexpected:
                print(f"  Warning: Unexpected keys in checkpoint: {other_unexpected}")
        if missing_keys:
            print(f"  Warning: Missing keys in checkpoint: {missing_keys}")
        print(f"✓ Loaded model state from checkpoint")

    print(f"Initializing label bank with {SENTENCE_BERT_MODEL}...")
    label_bank = LabelBank(model_name=SENTENCE_BERT_MODEL, device=device)
    print(f"✓ Label bank initialized (embedding_dim={label_bank.embedding_dim})")

    criterion = SemanticAlignmentLoss(
        temperature=TEMPERATURE, use_soft_targets=USE_SOFT_TARGETS,
        soft_target_temperature=SOFT_TARGET_TEMPERATURE, soft_target_weight=SOFT_TARGET_WEIGHT
    )

    # Initialize memory bank if enabled
    memory_bank = None
    if USE_MEMORY_BANK:
        print(f"Initializing memory bank (queue_size={MEMORY_BANK_SIZE}, embedding_dim={SEMANTIC_DIM})...")
        memory_bank = MemoryBank(queue_size=MEMORY_BANK_SIZE, embedding_dim=SEMANTIC_DIM)
        print(f"✓ Memory bank initialized - provides {MEMORY_BANK_SIZE} additional negatives")

    plotter = TrainingPlotter(plot_dir)

    # Load existing metrics if resuming
    if resume_checkpoint is not None:
        plotter.load_metrics()

    print("\n" + "="*70)
    print("End-to-End Training: Encoder + Semantic Head")
    effective_batch = BATCH_SIZE * ACCUMULATION_STEPS
    print(f"Gradient accumulation: {ACCUMULATION_STEPS} steps (micro-batch={BATCH_SIZE}, effective={effective_batch})")
    if USE_SOFT_TARGETS:
        print(f"Using pairwise soft targets (temperature={SOFT_TARGET_TEMPERATURE}, weight={SOFT_TARGET_WEIGHT})")
    if USE_MEMORY_BANK:
        print(f"Using memory bank with {MEMORY_BANK_SIZE} queue size (negatives per step: {BATCH_SIZE + MEMORY_BANK_SIZE})")
    print("="*70)

    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(
        data_root=DATA_ROOT, datasets=DATASETS, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=PERSISTENT_WORKERS, patch_size_per_dataset=PATCH_SIZE_PER_DATASET, seed=SEED,
        max_sessions_per_dataset=MAX_SESSIONS_PER_DATASET
    )

    # Setup optimizer (trains semantic head only if encoder is frozen, otherwise all parameters)
    # IMPORTANT: Include criterion.parameters() for learnable temperature (logit_scale)
    all_params = list(filter(lambda p: p.requires_grad, model.parameters())) + list(criterion.parameters())
    optimizer = AdamW(all_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Setup scheduler with proper warmup + cosine decay
    def warmup_cosine_schedule(epoch):
        """Linear warmup followed by cosine decay."""
        if epoch < WARMUP_EPOCHS:
            # Linear warmup: 0 -> 1 over WARMUP_EPOCHS
            return (epoch + 1) / WARMUP_EPOCHS
        else:
            # Cosine decay after warmup
            progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
            return 0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    best_val_loss = float('inf')

    # Load optimizer/scheduler states if resuming
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
        # Get best_val_loss from checkpoint's val_metrics
        if 'val_metrics' in resume_checkpoint and resume_checkpoint['val_metrics']:
            best_val_loss = resume_checkpoint['val_metrics'].get('loss', float('inf'))
        print(f"✓ Loaded optimizer and scheduler states")
        print(f"  Best val loss so far: {best_val_loss:.4f}")

        # Load criterion state if available (for learnable temperature)
        if 'criterion_state_dict' in resume_checkpoint:
            criterion.load_state_dict(resume_checkpoint['criterion_state_dict'])
            print(f"✓ Loaded criterion state (logit_scale={criterion.infonce.logit_scale.exp().item():.2f})")

        # Load memory bank state if available
        if memory_bank is not None and 'memory_bank_state_dict' in resume_checkpoint:
            if resume_checkpoint['memory_bank_state_dict'] is not None:
                memory_bank.load_state_dict(resume_checkpoint['memory_bank_state_dict'])
                print(f"✓ Loaded memory bank state (size={len(memory_bank)}, ptr={memory_bank.ptr})")

    # Warmup memory bank if enabled (reduces early training volatility)
    # Skip if resuming since memory bank is restored from checkpoint
    if memory_bank is not None and USE_MEMORY_BANK and resume_checkpoint is None:
        warmup_memory_bank(model, label_bank, train_loader, memory_bank, device)

    # Training loop
    for epoch in range(start_epoch, EPOCHS + 1):
        train_metrics = train_epoch(model, label_bank, train_loader, criterion, optimizer,
                                      device, epoch, scaler, plotter, "train", memory_bank)
        val_metrics = validate(model, label_bank, val_loader, criterion, device, epoch, "val", plot_dir)

        # Step scheduler every epoch (warmup is built into the schedule)
        scheduler.step()

        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        plotter.add_scalar('epoch/lr', current_lr, epoch)

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc(I2T): {train_metrics['acc_imu_to_text']:.4f}, Sim Gap: {train_metrics['similarity_gap']:.3f}, Grad: {train_metrics['grad_norm']:.2f}")
        print(f"  Debug - IMU std: {train_metrics['imu_std']:.3f}, Diversity: {train_metrics['imu_diversity']:.3f}, Proj grad: {train_metrics['projection_head_grad_norm']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Recall@1: {val_metrics['recall@1_avg']:.4f}, Recall@5: {val_metrics['recall@5_avg']:.4f}")

        # Log comprehensive metrics
        plotter.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
        plotter.add_scalar('epoch/train_acc_imu_to_text', train_metrics['acc_imu_to_text'], epoch)
        plotter.add_scalar('epoch/train_acc_text_to_imu', train_metrics['acc_text_to_imu'], epoch)
        plotter.add_scalar('epoch/train_positive_similarity', train_metrics['positive_similarity'], epoch)
        plotter.add_scalar('epoch/train_negative_similarity', train_metrics['negative_similarity'], epoch)
        plotter.add_scalar('epoch/train_similarity_gap', train_metrics['similarity_gap'], epoch)
        plotter.add_scalar('epoch/train_grad_norm', train_metrics['grad_norm'], epoch)

        # Log debug metrics
        plotter.add_scalar('epoch/debug_imu_std', train_metrics['imu_std'], epoch)
        plotter.add_scalar('epoch/debug_text_std', train_metrics['text_std'], epoch)
        plotter.add_scalar('epoch/debug_imu_diversity', train_metrics['imu_diversity'], epoch)
        plotter.add_scalar('epoch/debug_sim_gap', train_metrics['debug_sim_gap'], epoch)
        plotter.add_scalar('epoch/debug_queue_diversity', train_metrics['queue_diversity'], epoch)
        # Log per-component gradient norms
        for comp_name in ['cross_channel_fusion', 'temporal_attention', 'attention_pooling', 'projection_head']:
            plotter.add_scalar(f'epoch/debug_{comp_name}_grad_norm', train_metrics[f'{comp_name}_grad_norm'], epoch)

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
                'criterion_state_dict': criterion.state_dict(),  # Save learnable temperature
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'memory_bank_state_dict': memory_bank.state_dict() if memory_bank else None,
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
