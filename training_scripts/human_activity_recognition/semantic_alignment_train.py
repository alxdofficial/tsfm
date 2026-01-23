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

from datasets.imu_pretraining_dataset.multi_dataset_loader import create_dataloaders, IMUPretrainingDataset, worker_init_fn
from torch.utils.data import DataLoader, WeightedRandomSampler
from imu_activity_recognition_encoder.encoder import IMUActivityRecognitionEncoder
from imu_activity_recognition_encoder.semantic_alignment import SemanticAlignmentHead
from imu_activity_recognition_encoder.token_text_encoder import (
    TokenTextEncoder, ChannelTextFusion, LearnableLabelBank
)
from training_scripts.human_activity_recognition.semantic_loss import SemanticAlignmentLoss
from val_scripts.human_activity_recognition.plot_utils import TrainingPlotter, EmbeddingVisualizer
from training_scripts.human_activity_recognition.memory_bank import MemoryBank
from val_scripts.human_activity_recognition.evaluation_metrics import compute_group_accuracy
import random
from typing import Tuple, Optional, List

# ======================== HYPERPARAMETERS ========================

# Data configuration
DATA_ROOT = "/home/alex/code/tsfm/data"
DATASETS = ['uci_har', 'hhar', 'mhealth', 'pamap2', 'wisdm', 'unimib_shar', 'dsads', 'hapt', 'kuhar', 'vtt_coniot', 'recgym']
random.seed(42)
PATCH_SIZE_PER_DATASET = {
    'uci_har': 1.0,       # 50 Hz, 2.56s sessions → 50 samples/patch, 2 patches/session
    'hhar': 1.0,          # 50 Hz, 2.56s sessions → 50 samples/patch, 2 patches/session
    'mhealth': 2.0,       # 50 Hz, variable sessions (2-20s)
    'pamap2': 2.0,        # 100 Hz, variable sessions (2-60s)
    'wisdm': 2.0,         # 20 Hz, variable sessions (2-60s)
    'unimib_shar': 1.0,   # 50 Hz, 3.02s sessions → 50 samples/patch, 3 patches/session
    # New datasets
    'dsads': 2.0,         # 25 Hz, variable sessions → 50 samples/patch
    'mobiact': 1.5,       # 50 Hz, short fall events → 75 samples/patch
    'realworld': 2.0,     # 50 Hz, variable sessions → 100 samples/patch
    'vtt_coniot': 2.0,    # 50 Hz, variable sessions → 100 samples/patch
    'recgym': 2.5,        # 20 Hz, gym exercises → 50 samples/patch
    'hapt': 1.5,          # 50 Hz, transitions (2-8s) → 75 samples/patch
    'kuhar': 1.5,         # 100 Hz, 18 activities → 150 samples/patch
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
NUM_SEMANTIC_TEMPORAL_LAYERS = 2  # Temporal attention layers in semantic head

# Multi-query fusion/pooling configuration (symmetric architecture)
NUM_FUSION_QUERIES = 4  # Query tokens for channel fusion (channels → 1 vector per patch)
USE_FUSION_SELF_ATTENTION = True  # Fusion queries coordinate via self-attention
NUM_POOL_QUERIES = 4  # Query tokens for temporal pooling (patches → 1 vector)
USE_POOL_SELF_ATTENTION = True  # Pooling queries coordinate via self-attention

# Text encoder configuration
SENTENCE_BERT_MODEL = 'all-MiniLM-L6-v2'  # 384-dim embeddings, fast

# Training configuration
OUTPUT_DIR = "training_output/semantic_alignment"  # Note: plots go to semantic_alignment/<timestamp>/plots/
CHECKPOINT_DIR = None  # Will be set in main()
PRETRAINED_ENCODER_PATH = "training_output/imu_pretraining/20260104_085259/latest.pt"  # Pretrained tokenizer
FREEZE_ENCODER = False  # Unfreeze encoder to learn discriminative representations
SAVE_EVERY = 5

# Resume configuration - set to a folder path to resume training from that checkpoint
# Example: RESUME_FROM = "training_output/semantic_alignment/20251124_234942"
RESUME_FROM = None  # Set to folder path to resume, or None to start fresh
SEED = 42
MAX_GRAD_NORM = 1.0  # Gradient clipping threshold

# Training hyperparameters
EPOCHS = 60
BATCH_SIZE = 16  # Micro-batch size (reduced for 48-channel datasets like PAMAP2)
ACCUMULATION_STEPS = 32  # Effective batch = 16 × 32 = 512
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
MEMORY_BANK_SIZE = 256  # Provides 16 + 256 = 272 negatives per step

# NOTE: Channel augmentation (random subsampling/shuffling) is DISABLED.
# Experiments showed better zero-shot generalization with consistent channel order.

# NOTE: Channel projection (learnable MLP after SentenceBERT) is ENABLED.
# This allows task-specific adaptation of frozen semantic embeddings.

# Plotting
PLOT_EVERY_N_BATCHES = 10

# Embedding visualization
VISUALIZE_EMBEDDINGS = True  # Generate UMAP plots showing IMU-text alignment
VISUALIZE_EVERY_N_EPOCHS = 1  # How often to generate embedding plots (1 = every epoch)

# Classification metrics during training
COMPUTE_CLASSIFICATION_METRICS = True  # Compute group-aware accuracy during validation
UNSEEN_DATASET = 'motionsense'  # Dataset for zero-shot evaluation (not in training)
EVAL_UNSEEN_EVERY = 5  # Evaluate on unseen dataset every N epochs
UMAP_N_NEIGHBORS = 15  # UMAP parameter: 5-50, higher = more global structure
UMAP_MIN_DIST = 0.1  # UMAP parameter: 0.0-0.99, lower = tighter clusters

# Debug configuration
DEBUG_METRIC_FREQUENCY = 10  # Compute expensive debug metrics every N batches (not every batch)

# Token-level text encoding configuration
# Uses cross-attention between sensor tokens and channel description tokens,
# plus learnable attention pooling for label refinement.
TOKEN_TEXT_NUM_HEADS = 4     # Attention heads for text fusion/pooling
TOKEN_TEXT_NUM_QUERIES = 4   # Learnable query tokens for label pooling

# Ablation: use mean pooling instead of learnable attention pooling
# When True: uses SentenceBERT's default mean pooling (no learnable params)
# When False: uses learnable attention pooling (LabelAttentionPooling)
USE_MEAN_POOLING = False  # Default: learnable attention pooling (baseline)

# Ablation: freeze label bank to test contribution of learnable text encoding
# When True: LabelAttentionPooling weights are frozen (uses random init queries)
# When False: LabelAttentionPooling is trainable (baseline)
FREEZE_LABEL_BANK = False  # Default: trainable (baseline)

# Class balancing configuration
# Uses LABEL_GROUPS to balance sampling at the semantic group level
# (e.g., "jogging" and "running" are same group, balanced together)
USE_GROUP_BALANCED_SAMPLING = True  # Default: enable group-balanced sampling

# Patch size augmentation configuration
# During training, randomly sample patch sizes from valid ranges per dataset
# Ranges are constrained by session duration (need ≥2 patches per session)
USE_PATCH_SIZE_AUGMENTATION = True  # Enable patch size augmentation for better generalization
MIN_PATCHES_PER_SAMPLE = 1  # Minimum patches required per sample

# Valid patch size ranges per dataset: (min_sec, max_sec, step_sec)
PATCH_SIZE_RANGE_PER_DATASET = {
    # Fixed-length sessions (~2.5-3s) - tighter range around 1.0s
    'uci_har':      (0.75, 1.25, 0.25),  # 50 Hz, 2.56s fixed → [0.75, 1.0, 1.25]
    'hhar':         (0.75, 1.25, 0.25),  # 50 Hz, 2.56s fixed → [0.75, 1.0, 1.25]
    'unimib_shar':  (0.75, 1.25, 0.25),  # 50 Hz, 3.02s fixed → [0.75, 1.0, 1.25]
    # Variable-length sessions (2-60s, median ~10s) - wider range centered on 1.5s
    'mhealth':      (1.0, 2.0, 0.5),     # 50 Hz, 2-20s → [1.0, 1.5, 2.0]
    'pamap2':       (1.0, 2.0, 0.5),     # 100 Hz, 2-60s → [1.0, 1.5, 2.0]
    'wisdm':        (1.0, 2.0, 0.5),     # 20 Hz, 2-60s → [1.0, 1.5, 2.0]
    # New datasets
    'dsads':        (1.5, 2.5, 0.5),     # 25 Hz, variable → [1.5, 2.0, 2.5]
    'mobiact':      (1.0, 2.0, 0.5),     # 50 Hz, short falls → [1.0, 1.5, 2.0]
    'realworld':    (1.5, 2.5, 0.5),     # 50 Hz, variable → [1.5, 2.0, 2.5]
    'vtt_coniot':   (1.5, 2.5, 0.5),     # 50 Hz, variable → [1.5, 2.0, 2.5]
    'recgym':       (2.0, 3.0, 0.5),     # 20 Hz, gym exercises → [2.0, 2.5, 3.0]
}

# =================================================================


class SemanticAlignmentModel(nn.Module):
    """
    Semantic alignment model with token-level text encoding.

    Features:
    - ChannelTextFusion: cross-attention between sensor tokens and channel description tokens
    - Works with LearnableLabelBank: learnable attention pooling for label refinement
    - Patch size augmentation: randomly sample patch sizes during training
    """

    def __init__(
        self,
        encoder: IMUActivityRecognitionEncoder,
        semantic_head: SemanticAlignmentHead,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_patch_augmentation: bool = False,
        min_patches_per_sample: int = 2
    ):
        super().__init__()
        self.encoder = encoder
        self.semantic_head = semantic_head
        self.use_patch_augmentation = use_patch_augmentation
        self.min_patches_per_sample = min_patches_per_sample

        # Token-level text encoder (frozen backbone)
        self.text_encoder = TokenTextEncoder()

        # Channel text fusion (learnable cross-attention)
        self.channel_fusion = ChannelTextFusion(
            d_model=D_MODEL,
            num_heads=num_heads,
            dropout=dropout
        )

    def _get_valid_patch_size(
        self,
        session_duration: float,
        patch_range: Tuple[float, float, float],
        default_patch_size: float
    ) -> float:
        """
        Sample a valid patch size from the range, respecting session duration constraint.

        Args:
            session_duration: Duration of the session in seconds
            patch_range: (min_sec, max_sec, step_sec) tuple, or None for no augmentation
            default_patch_size: Default patch size to use if range is None

        Returns:
            Sampled patch size in seconds
        """
        if patch_range is None:
            return default_patch_size

        min_sec, max_sec, step_sec = patch_range

        # Cap max by session duration (need at least min_patches_per_sample patches)
        max_feasible = session_duration / self.min_patches_per_sample
        actual_max = min(max_sec, max_feasible)

        # If session too short for ≥2 patches with min patch size, use full session as one patch
        if actual_max < min_sec:
            return session_duration  # One patch covering entire session

        # Generate valid sizes within feasible range
        num_steps = int((actual_max - min_sec) / step_sec) + 1
        valid_sizes = [min_sec + i * step_sec for i in range(num_steps)]

        return random.choice(valid_sizes)

    def forward(self, data, channel_descriptions, channel_mask, sampling_rates, patch_sizes,
                patch_ranges: Optional[List] = None):
        """
        Forward pass with optional patch size augmentation.

        Args:
            data: (batch, timesteps, channels) padded IMU data
            channel_descriptions: List of channel description lists
            channel_mask: (batch, channels) boolean mask
            sampling_rates: List of sampling rates per sample
            patch_sizes: List of default patch sizes per sample
            patch_ranges: Optional list of (min, max, step) tuples for patch augmentation
        """
        batch_size = data.shape[0]
        device = data.device

        # Step 1: Preprocess all samples
        all_patches = []
        all_channel_descs = []
        valid_samples = []

        with torch.no_grad():
            for i in range(batch_size):
                # Determine patch size (augmented during training, fixed during eval)
                if self.training and self.use_patch_augmentation and patch_ranges is not None:
                    # Compute session duration from data
                    # attention_mask not passed here, so estimate from non-zero timesteps
                    session_timesteps = data[i].shape[0]
                    session_duration = session_timesteps / sampling_rates[i]
                    patch_range = patch_ranges[i] if i < len(patch_ranges) else None
                    actual_patch_size = self._get_valid_patch_size(
                        session_duration, patch_range, patch_sizes[i]
                    )
                else:
                    actual_patch_size = patch_sizes[i]

                patches, _ = self.encoder.preprocess(
                    data[i], sampling_rate_hz=sampling_rates[i], patch_size_sec=actual_patch_size
                )

                if patches is None or len(patches) == 0:
                    all_patches.append(None)
                    all_channel_descs.append(None)
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

        # Step 2: Batch valid patches
        valid_indices = [i for i, v in enumerate(valid_samples) if v]

        if len(valid_indices) == 0:
            max_channels = data.shape[2]
            padded_encoder_output = torch.zeros(batch_size, 1, max_channels, D_MODEL, device=device)
            padded_patch_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
            return self.semantic_head(padded_encoder_output, channel_mask=channel_mask,
                                       patch_mask=padded_patch_mask, normalize=True)

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

        # Step 3: Encode with frozen/unfrozen encoder
        if FREEZE_ENCODER:
            with torch.no_grad():
                encoded_batch = self.encoder(batched_patches, batched_channel_descs)
        else:
            encoded_batch = self.encoder(batched_patches, batched_channel_descs)

        # Step 4: Apply channel text fusion per sample (each sample has different channel descriptions)
        # Text encoding is cached, so per-sample loop has minimal overhead
        fused_samples = []
        for i in range(num_valid):
            sample_channel_descs = batched_channel_descs[i][:max_channels]
            channel_tokens, channel_mask_text = self.text_encoder.encode(sample_channel_descs, device)
            # Cross-attention between this sample's sensor tokens and its channel text tokens
            fused_sample = self.channel_fusion(encoded_batch[i:i+1], channel_tokens, channel_mask_text)
            fused_samples.append(fused_sample)
        encoded_batch = torch.cat(fused_samples, dim=0)

        # Step 5: Map outputs back to original batch positions
        encoder_outputs = [None] * batch_size
        patch_masks = [None] * batch_size

        for batch_idx, sample_idx in enumerate(valid_indices):
            num_patches = len(all_patches[sample_idx])
            encoder_outputs[sample_idx] = encoded_batch[batch_idx, :num_patches]
            patch_masks[sample_idx] = torch.ones(num_patches, dtype=torch.bool, device=device)

        # Step 6: Final padding
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
    Compute debug metrics for diagnosing training issues.

    Returns dict with:
    - Representation collapse indicators (std, diversity)
    - Memory bank quality (if queue provided)

    NOTE: Similarity metrics (pos_sim, neg_sim, sim_gap) are computed in the loss
    function with proper label-aware masking. This function only computes
    collapse/health indicators that don't depend on labels.
    """
    # Convert to fp32 for all operations (mixed precision compatibility)
    imu_embeddings = imu_embeddings.float()
    text_embeddings = text_embeddings.float()
    if imu_queue is not None:
        imu_queue = imu_queue.float()

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
    else:
        debug_metrics['imu_diversity'] = 0.0

    # 2. Memory Bank Quality (if queue provided)
    if imu_queue is not None and len(imu_queue) > 0:
        # Queue diversity
        queue_sample = imu_queue[:min(100, len(imu_queue))]
        debug_metrics['queue_diversity'] = torch.pdist(queue_sample).mean().item()

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
    label_bank.eval()
    
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
    
    print(f"✓ Memory bank warmup complete ({len(memory_bank)} embeddings)")
    
    # Return model to training mode if it wasn't frozen
    if not FREEZE_ENCODER:
        model.train()


def _compute_per_layer_grad_norms(model, criterion=None, label_bank=None):
    """
    Compute gradient norms for each sub-component of the semantic head and label_bank.

    SemanticAlignmentHead sub-components:
    - cross_channel_fusion: Perceiver-style bottleneck attention
    - temporal_attention: TransformerEncoder over patches
    - attention_pooling: CLS token attention pooling
    - projection_head: 3-layer MLP to semantic space

    LearnableLabelBank sub-components (if provided):
    - label_pooling: LabelAttentionPooling with learnable queries

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

    # Track label_bank gradients (LearnableLabelBank with LabelAttentionPooling)
    if label_bank is not None:
        label_grad_norm = sum(
            p.grad.norm().item() ** 2
            for p in label_bank.parameters()
            if p.grad is not None
        ) ** 0.5
        grad_norms['label_pooling'] = label_grad_norm

    return grad_norms


def train_epoch(model, label_bank, dataloader, criterion, optimizer, device, epoch, scaler, plotter=None, stage="stage1", memory_bank=None):
    """Train for one epoch."""
    model.train()
    label_bank.train()  # Enable dropout in learnable attention pooling

    # Keep encoder in evaluation mode if frozen (prevents dropout and batch norm updates)
    if FREEZE_ENCODER and hasattr(model, 'encoder'):
        model.encoder.eval()

    # Track all metrics
    total_loss = 0.0
    total_pos_sim = 0.0
    total_neg_sim = 0.0
    total_sim_gap = 0.0
    total_grad_norm = 0.0

    # Track debug metrics (collapse detection only - similarity is in loss metrics)
    total_imu_std = 0.0
    total_text_std = 0.0
    total_imu_diversity = 0.0
    total_queue_diversity = 0.0

    # Track per-layer gradient norms (dict for each sub-component)
    total_grad_norms = {
        'cross_channel_fusion': 0.0,
        'temporal_attention': 0.0,
        'attention_pooling': 0.0,
        'projection_head': 0.0,
        'logit_scale': 0.0,  # Learnable temperature
        'label_pooling': 0.0,  # LearnableLabelBank attention pooling
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
        patch_ranges = [m.get('patch_size_range', None) for m in metadata]
        channel_descriptions = [m['channel_descriptions'] for m in metadata]

        # NOTE: label_bank.encode() needs gradients for learnable pooling!
        # The frozen SentenceBERT tokens are cached (no grad), but LabelAttentionPooling is trainable
        text_embeddings = label_bank.encode(label_texts, normalize=True)

        # Get queue embeddings from memory bank if enabled (no gradients needed)
        if memory_bank is not None and USE_MEMORY_BANK:
            with torch.no_grad():
                imu_queue, text_queue = memory_bank.get_queue_embeddings(device)
        else:
            imu_queue, text_queue = None, None

        with autocast('cuda', enabled=device.type == 'cuda'):
            imu_embeddings = model(data, channel_descriptions, channel_mask, sampling_rates, patch_sizes,
                                   patch_ranges=patch_ranges)
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

            # Collect all trainable parameters for gradient clipping
            all_trainable_params = (
                list(filter(lambda p: p.requires_grad, model.parameters())) +
                list(criterion.parameters()) +
                list(label_bank.parameters())
            )

            if scaler is not None:
                scaler.unscale_(optimizer)

                # Per-layer gradient norms (expensive - only compute periodically)
                if should_log_grads:
                    batch_grad_norms = _compute_per_layer_grad_norms(model, criterion, label_bank)

                grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable_params, max_norm=MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Per-layer gradient norms (expensive - only compute periodically)
                if should_log_grads:
                    batch_grad_norms = _compute_per_layer_grad_norms(model, criterion, label_bank)

                grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable_params, max_norm=MAX_GRAD_NORM)
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
        total_pos_sim += metrics['positive_similarity']
        total_neg_sim += metrics['negative_similarity']
        total_sim_gap += metrics['similarity_gap']
        total_grad_norm += grad_norm.item()

        # Accumulate debug metrics (only when computed)
        if debug_metrics:
            total_imu_std += debug_metrics['imu_std']
            total_text_std += debug_metrics['text_std']
            total_imu_diversity += debug_metrics['imu_diversity']
            if 'queue_diversity' in debug_metrics:
                total_queue_diversity += debug_metrics['queue_diversity']

        # Accumulate per-layer gradient norms
        for k in total_grad_norms:
            total_grad_norms[k] += batch_grad_norms[k]

        pbar.set_postfix({
            'loss': f"{metrics['loss']:.4f}",
            'sim_gap': f"{metrics['similarity_gap']:.3f}",
            'pos_sim': f"{metrics['positive_similarity']:.3f}",
            'grad': f"{grad_norm.item():.2f}"
        })

        # Batch-level plotting (every N batches)
        if plotter is not None and batch_idx % PLOT_EVERY_N_BATCHES == 0:
            global_batch = (epoch - 1) * len(dataloader) + batch_idx
            plotter.add_scalar(f'batch/{stage}_loss', metrics['loss'], global_batch)
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
            # NOTE: Similarity metrics are computed in the loss function with label-aware masking
            # Debug metrics only track collapse indicators (representation health)
            if debug_metrics:
                plotter.add_scalar(f'batch/debug_{stage}_imu_std', debug_metrics['imu_std'], global_batch)
                plotter.add_scalar(f'batch/debug_{stage}_text_std', debug_metrics['text_std'], global_batch)
                plotter.add_scalar(f'batch/debug_{stage}_imu_diversity', debug_metrics['imu_diversity'], global_batch)
                if 'queue_diversity' in debug_metrics:
                    plotter.add_scalar(f'batch/debug_{stage}_queue_diversity', debug_metrics['queue_diversity'], global_batch)
                # Cross-channel attention stats (from SemanticAlignmentHead)
                for attn_key in ['cross_channel_attn_entropy', 'cross_channel_attn_entropy_ratio',
                                 'cross_channel_attn_max', 'cross_channel_attn_std']:
                    if attn_key in debug_metrics:
                        plotter.add_scalar(f'batch/debug_{stage}_{attn_key}', debug_metrics[attn_key], global_batch)

            plotter.plot_all()

    num_batches = len(dataloader)
    # Debug metrics are only computed every DEBUG_METRIC_FREQUENCY batches
    debug_count = (num_batches + DEBUG_METRIC_FREQUENCY - 1) // DEBUG_METRIC_FREQUENCY  # Ceiling division
    debug_count = max(debug_count, 1)  # Avoid division by zero

    return {
        'loss': total_loss / num_batches,
        'positive_similarity': total_pos_sim / num_batches,
        'negative_similarity': total_neg_sim / num_batches,
        'similarity_gap': total_sim_gap / num_batches,
        'grad_norm': total_grad_norm / num_batches,
        # Debug metrics (computed every DEBUG_METRIC_FREQUENCY batches)
        'imu_std': total_imu_std / debug_count,
        'text_std': total_text_std / debug_count,
        'imu_diversity': total_imu_diversity / debug_count,
        'queue_diversity': total_queue_diversity / debug_count if USE_MEMORY_BANK else 0.0,
        # Per-component gradient norms
        **{f'{k}_grad_norm': v / debug_count for k, v in total_grad_norms.items()}
    }


def validate(model, label_bank, dataloader, criterion, device, epoch, stage="stage1", plot_dir=None,
             compute_classification=COMPUTE_CLASSIFICATION_METRICS):
    """Validate for one epoch and optionally compute classification metrics."""
    model.eval()
    label_bank.eval()  # Disable dropout in learnable attention pooling
    total_loss = 0.0
    total_pos_sim = 0.0
    total_neg_sim = 0.0
    total_sim_gap = 0.0
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
                # No patch augmentation during validation (patch_ranges=None)
                imu_embeddings = model(data, channel_descriptions, channel_mask, sampling_rates, patch_sizes,
                                       patch_ranges=None)
                _, metrics = criterion(imu_embeddings, text_embeddings, label_texts, return_metrics=True)

            total_loss += metrics['loss']
            total_pos_sim += metrics['positive_similarity']
            total_neg_sim += metrics['negative_similarity']
            total_sim_gap += metrics['similarity_gap']
            # Keep embeddings on GPU for faster concatenation
            all_imu_embeddings.append(imu_embeddings)
            all_text_embeddings.append(text_embeddings)
            all_labels.extend(label_texts)

            pbar.set_postfix({'loss': f"{metrics['loss']:.4f}", 'sim_gap': f"{metrics['similarity_gap']:.3f}"})

    num_batches = len(dataloader)
    avg_metrics = {
        'loss': total_loss / num_batches,
        'positive_similarity': total_pos_sim / num_batches,
        'negative_similarity': total_neg_sim / num_batches,
        'similarity_gap': total_sim_gap / num_batches
    }

    # Compute classification metrics and/or visualization if enabled
    need_concat = compute_classification or (VISUALIZE_EMBEDDINGS and epoch % VISUALIZE_EVERY_N_EPOCHS == 0)

    if need_concat and len(all_imu_embeddings) > 0:
        # Concatenate on GPU (faster than CPU→GPU transfer)
        all_imu_cat = torch.cat(all_imu_embeddings, dim=0)
        all_text_cat = torch.cat(all_text_embeddings, dim=0)

        # Compute group-aware classification accuracy
        if compute_classification:
            with torch.no_grad():
                class_metrics = compute_group_accuracy(
                    all_imu_cat, label_bank, all_labels, return_mrr=True
                )
            avg_metrics.update(class_metrics)

        # Generate embedding visualization if enabled
        if VISUALIZE_EMBEDDINGS and plot_dir is not None and (epoch % VISUALIZE_EVERY_N_EPOCHS == 0):
            print(f"\n[Epoch {epoch}] Generating embedding visualization...")
            visualizer = EmbeddingVisualizer(
                output_dir=plot_dir,
                n_neighbors=UMAP_N_NEIGHBORS,
                min_dist=UMAP_MIN_DIST
            )
            visualizer.plot_embedding_alignment_2d(
                imu_embeddings=all_imu_cat,
                text_embeddings=all_text_cat,
                labels=all_labels,
                epoch=epoch,
                metrics={
                    'pos_sim': avg_metrics['positive_similarity'],
                    'sim_gap': avg_metrics['similarity_gap']
                }
            )

    return avg_metrics


def evaluate_unseen(model, label_bank, dataloader, device, epoch):
    """
    Evaluate on unseen dataset for zero-shot performance.

    Returns classification accuracy and MRR.
    """
    model.eval()
    label_bank.eval()
    all_imu_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"[Epoch {epoch}] Unseen eval", leave=False):
            data = batch['data'].to(device)
            channel_mask = batch['channel_mask'].to(device)
            label_texts = batch['label_texts']
            metadata = batch['metadata']

            sampling_rates = [m['sampling_rate_hz'] for m in metadata]
            patch_sizes = [m['patch_size_sec'] for m in metadata]
            channel_descriptions = [m['channel_descriptions'] for m in metadata]

            with autocast('cuda', enabled=device.type == 'cuda'):
                imu_embeddings = model(data, channel_descriptions, channel_mask, sampling_rates, patch_sizes)

            all_imu_embeddings.append(imu_embeddings)
            all_labels.extend(label_texts)

    if len(all_imu_embeddings) > 0:
        all_imu_cat = torch.cat(all_imu_embeddings, dim=0)
        metrics = compute_group_accuracy(all_imu_cat, label_bank, all_labels, return_mrr=True)
        return metrics

    return {'accuracy': 0.0, 'mrr': 0.0}


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
        'encoder': {
            'd_model': D_MODEL, 'num_heads': NUM_HEADS, 'num_temporal_layers': NUM_TEMPORAL_LAYERS,
            'dim_feedforward': DIM_FEEDFORWARD, 'dropout': DROPOUT, 'use_cross_channel': USE_CROSS_CHANNEL,
            'cnn_channels': CNN_CHANNELS, 'cnn_kernel_sizes': CNN_KERNEL_SIZES,
            'target_patch_size': TARGET_PATCH_SIZE, 'use_channel_encoding': False
        },
        'semantic': {'d_model_fused': D_MODEL_FUSED, 'semantic_dim': SEMANTIC_DIM,
                     'sentence_bert_model': SENTENCE_BERT_MODEL, 'temperature': TEMPERATURE,
                     'use_soft_targets': USE_SOFT_TARGETS, 'soft_target_temperature': SOFT_TARGET_TEMPERATURE,
                     'soft_target_weight': SOFT_TARGET_WEIGHT,
                     'use_memory_bank': USE_MEMORY_BANK, 'memory_bank_size': MEMORY_BANK_SIZE},
        'training': {'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'accumulation_steps': ACCUMULATION_STEPS,
                     'effective_batch_size': BATCH_SIZE * ACCUMULATION_STEPS,
                     'lr': LEARNING_RATE, 'warmup_epochs': WARMUP_EPOCHS, 'max_grad_norm': MAX_GRAD_NORM},
        'data': {'channel_augmentation': False, 'use_channel_encoding': False},  # ChannelTextFusion handles channel semantics
        'channel_projection': {'enabled': True, 'hidden_dim': None},  # Hardcoded: projection enabled
        'token_level_text': {'num_heads': TOKEN_TEXT_NUM_HEADS, 'num_queries': TOKEN_TEXT_NUM_QUERIES,
                             'use_mean_pooling': USE_MEAN_POOLING, 'freeze_label_bank': FREEZE_LABEL_BANK},
        'semantic_head': {'num_temporal_layers': NUM_SEMANTIC_TEMPORAL_LAYERS,
                          'num_fusion_queries': NUM_FUSION_QUERIES, 'use_fusion_self_attention': USE_FUSION_SELF_ATTENTION,
                          'num_pool_queries': NUM_POOL_QUERIES, 'use_pool_self_attention': USE_POOL_SELF_ATTENTION}
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
        cnn_channels=CNN_CHANNELS, cnn_kernel_sizes=CNN_KERNEL_SIZES,
        use_channel_encoding=False,  # Disabled: ChannelTextFusion handles channel semantics
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
        num_temporal_layers=NUM_SEMANTIC_TEMPORAL_LAYERS,
        num_heads=NUM_HEADS, dim_feedforward=D_MODEL_FUSED * 4, dropout=DROPOUT,
        num_fusion_queries=NUM_FUSION_QUERIES, use_fusion_self_attention=USE_FUSION_SELF_ATTENTION,
        num_pool_queries=NUM_POOL_QUERIES, use_pool_self_attention=USE_POOL_SELF_ATTENTION
    ).to(device)

    print("Using token-level text encoding with cross-attention")
    model = SemanticAlignmentModel(
        encoder, semantic_head,
        num_heads=TOKEN_TEXT_NUM_HEADS,
        dropout=DROPOUT,
        use_patch_augmentation=USE_PATCH_SIZE_AUGMENTATION,
        min_patches_per_sample=MIN_PATCHES_PER_SAMPLE
    ).to(device)

    if USE_PATCH_SIZE_AUGMENTATION:
        print(f"\n=== Patch Size Augmentation Enabled ===")
        print(f"Min patches per sample: {MIN_PATCHES_PER_SAMPLE}")
        for ds, (min_s, max_s, step_s) in PATCH_SIZE_RANGE_PER_DATASET.items():
            valid_sizes = [min_s + i * step_s for i in range(int((max_s - min_s) / step_s) + 1)]
            print(f"  {ds}: {valid_sizes} sec")

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
        print("✓ Loaded model state from checkpoint")

    # Initialize learnable label bank (token-level attention pooling or mean pooling for ablation)
    pooling_type = "MEAN POOLING (ablation)" if USE_MEAN_POOLING else "learnable attention pooling"
    print(f"Initializing label bank ({pooling_type})...")
    label_bank = LearnableLabelBank(
        device=device,
        num_heads=TOKEN_TEXT_NUM_HEADS,
        num_queries=TOKEN_TEXT_NUM_QUERIES,
        dropout=DROPOUT,
        use_mean_pooling=USE_MEAN_POOLING
    )
    print(f"✓ Label bank initialized (embedding_dim={label_bank.embedding_dim}, pooling={pooling_type})")

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

    # Create datasets
    # Pass patch_size_range for training (augmentation), but not for validation (fixed sizes)
    train_dataset = IMUPretrainingDataset(
        data_root=DATA_ROOT,
        datasets=DATASETS,
        split='train',
        patch_size_per_dataset=PATCH_SIZE_PER_DATASET,
        patch_size_range_per_dataset=PATCH_SIZE_RANGE_PER_DATASET if USE_PATCH_SIZE_AUGMENTATION else None,
        max_sessions_per_dataset=MAX_SESSIONS_PER_DATASET,
        seed=SEED
    )
    val_dataset = IMUPretrainingDataset(
        data_root=DATA_ROOT,
        datasets=DATASETS,
        split='val',
        patch_size_per_dataset=PATCH_SIZE_PER_DATASET,
        patch_size_range_per_dataset=None,  # No augmentation for validation
        max_sessions_per_dataset=MAX_SESSIONS_PER_DATASET,
        seed=SEED
    )

    # Create train dataloader with optional group-balanced sampling
    if USE_GROUP_BALANCED_SAMPLING:
        # Compute weights for group-balanced sampling
        sample_weights = train_dataset.compute_group_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True  # Allow resampling rare groups
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,  # Replaces shuffle=True
            num_workers=NUM_WORKERS,
            prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=PERSISTENT_WORKERS,
            collate_fn=IMUPretrainingDataset.collate_fn,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )

        # Log group distribution
        group_dist = train_dataset.get_group_distribution()
        print(f"\n=== Group-Balanced Sampling Enabled ===")
        print(f"Groups: {len(group_dist)}")
        most_common = max(group_dist.items(), key=lambda x: x[1])
        least_common = min(group_dist.items(), key=lambda x: x[1])
        print(f"Most common: {most_common[0]} ({most_common[1]} samples)")
        print(f"Least common: {least_common[0]} ({least_common[1]} samples)")
        print(f"Imbalance ratio: {most_common[1] / least_common[1]:.1f}x → balanced to 1.0x")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=PERSISTENT_WORKERS,
            collate_fn=IMUPretrainingDataset.collate_fn,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )

    # Create val dataloader (no balancing needed)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=PERSISTENT_WORKERS,
        collate_fn=IMUPretrainingDataset.collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    # Create unseen dataset loader for zero-shot evaluation
    unseen_loader = None
    if UNSEEN_DATASET:
        try:
            unseen_dataset = IMUPretrainingDataset(
                data_root=DATA_ROOT,
                datasets=[UNSEEN_DATASET],
                split='val',
                max_sessions_per_dataset=MAX_SESSIONS_PER_DATASET,
                seed=SEED
            )
            unseen_loader = DataLoader(
                unseen_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                collate_fn=IMUPretrainingDataset.collate_fn,
                pin_memory=True
            )
            print(f"Loaded unseen dataset '{UNSEEN_DATASET}' with {len(unseen_dataset)} samples")
        except Exception as e:
            print(f"Warning: Could not load unseen dataset '{UNSEEN_DATASET}': {e}")

    # Setup optimizer (trains semantic head only if encoder is frozen, otherwise all parameters)
    # IMPORTANT: Include criterion.parameters() for learnable temperature (logit_scale)
    # and label_bank.parameters() for learnable attention pooling (if not using mean pooling)
    all_params = list(filter(lambda p: p.requires_grad, model.parameters())) + list(criterion.parameters())

    # Add label_bank parameters if using learnable attention pooling
    label_bank_params = list(label_bank.parameters())
    if USE_MEAN_POOLING:
        # Mean pooling has no learnable parameters
        print(f"✓ Using mean pooling - no label bank parameters to train")
    elif FREEZE_LABEL_BANK:
        # Freeze label bank for ablation (test contribution of learnable text encoding)
        for p in label_bank.parameters():
            p.requires_grad = False
        print(f"✓ Label bank FROZEN for ablation ({sum(p.numel() for p in label_bank_params)} params frozen)")
    elif len(label_bank_params) > 0:
        all_params += label_bank_params
        print(f"✓ Label bank parameters added to optimizer ({sum(p.numel() for p in label_bank_params)} params)")

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
        print("✓ Loaded optimizer and scheduler states")
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

        # Load label_bank state if available (learnable attention pooling weights)
        if 'label_bank_state_dict' in resume_checkpoint:
            label_bank.load_state_dict(resume_checkpoint['label_bank_state_dict'])
            print("✓ Loaded label_bank state (learnable attention pooling)")

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
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Sim Gap: {train_metrics['similarity_gap']:.3f}, Pos Sim: {train_metrics['positive_similarity']:.3f}, Grad: {train_metrics['grad_norm']:.2f}")
        print(f"  Debug - IMU std: {train_metrics['imu_std']:.3f}, Diversity: {train_metrics['imu_diversity']:.3f}, Proj grad: {train_metrics['projection_head_grad_norm']:.4f}")
        val_acc_str = f", Acc: {val_metrics['accuracy']:.1%}" if 'accuracy' in val_metrics else ""
        val_mrr_str = f", MRR: {val_metrics['mrr']:.3f}" if 'mrr' in val_metrics else ""
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Sim Gap: {val_metrics['similarity_gap']:.3f}, Pos Sim: {val_metrics['positive_similarity']:.3f}{val_acc_str}{val_mrr_str}")

        # Evaluate on unseen dataset periodically
        unseen_metrics = None
        if unseen_loader is not None and epoch % EVAL_UNSEEN_EVERY == 0:
            unseen_metrics = evaluate_unseen(model, label_bank, unseen_loader, device, epoch)
            print(f"  Unseen ({UNSEEN_DATASET}) - Acc: {unseen_metrics['accuracy']:.1%}, MRR: {unseen_metrics['mrr']:.3f}")

        # Log comprehensive metrics
        plotter.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
        plotter.add_scalar('epoch/train_positive_similarity', train_metrics['positive_similarity'], epoch)
        plotter.add_scalar('epoch/train_negative_similarity', train_metrics['negative_similarity'], epoch)
        plotter.add_scalar('epoch/train_similarity_gap', train_metrics['similarity_gap'], epoch)
        plotter.add_scalar('epoch/train_grad_norm', train_metrics['grad_norm'], epoch)

        # Log debug metrics (collapse detection only - similarity_gap is from loss function above)
        plotter.add_scalar('epoch/debug_imu_std', train_metrics['imu_std'], epoch)
        plotter.add_scalar('epoch/debug_text_std', train_metrics['text_std'], epoch)
        plotter.add_scalar('epoch/debug_imu_diversity', train_metrics['imu_diversity'], epoch)
        plotter.add_scalar('epoch/debug_queue_diversity', train_metrics['queue_diversity'], epoch)
        # Log per-component gradient norms (including label_pooling for LearnableLabelBank)
        for comp_name in ['cross_channel_fusion', 'temporal_attention', 'attention_pooling', 'projection_head', 'label_pooling']:
            plotter.add_scalar(f'epoch/debug_{comp_name}_grad_norm', train_metrics[f'{comp_name}_grad_norm'], epoch)

        plotter.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
        plotter.add_scalar('epoch/val_positive_similarity', val_metrics['positive_similarity'], epoch)
        plotter.add_scalar('epoch/val_negative_similarity', val_metrics['negative_similarity'], epoch)
        plotter.add_scalar('epoch/val_similarity_gap', val_metrics['similarity_gap'], epoch)
        if 'accuracy' in val_metrics:
            plotter.add_scalar('epoch/val_accuracy', val_metrics['accuracy'], epoch)
        if 'mrr' in val_metrics:
            plotter.add_scalar('epoch/val_mrr', val_metrics['mrr'], epoch)

        # Log unseen dataset metrics
        if unseen_metrics is not None:
            plotter.add_scalar('epoch/unseen_accuracy', unseen_metrics['accuracy'], epoch)
            plotter.add_scalar('epoch/unseen_mrr', unseen_metrics['mrr'], epoch)

        plotter.plot_all()

        # Save checkpoints
        if epoch % SAVE_EVERY == 0 or val_metrics['loss'] < best_val_loss:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'label_bank_state_dict': label_bank.state_dict(),  # Save learnable label embeddings
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
