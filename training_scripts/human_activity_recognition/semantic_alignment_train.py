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

from datasets.imu_pretraining_dataset.multi_dataset_loader import IMUPretrainingDataset, worker_init_fn
from torch.utils.data import DataLoader
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

class ChannelBucketBatchSampler:
    """
    Batch sampler that groups samples by channel count to minimize padding waste.

    Maintains class-balanced weighted sampling within each bucket.
    Replaces WeightedRandomSampler + default BatchSampler.

    Channel count distribution across datasets:
        3: unimib_shar | 6: hhar, hapt, kuhar, recgym | 9: uci_har, dsads
        12: wisdm | 21: mhealth | 52: pamap2

    Without bucketing: joint patch×channel utilization ~11% (89% wasted compute).
    With bucketing: channel utilization ~80-100% within each bucket.
    """

    def __init__(self, channel_counts, sample_weights, batch_size, num_samples=None):
        self.batch_size = batch_size

        # Group indices by channel count
        self.buckets = {}
        for idx, ch_count in enumerate(channel_counts):
            if ch_count not in self.buckets:
                self.buckets[ch_count] = []
            self.buckets[ch_count].append(idx)

        # Store weights per bucket for weighted sampling
        self.bucket_weights = {}
        for ch_count, indices in self.buckets.items():
            idx_tensor = torch.tensor(indices, dtype=torch.long)
            self.bucket_weights[ch_count] = sample_weights[idx_tensor]

        # Total batches per epoch
        total_samples = num_samples if num_samples is not None else len(channel_counts)
        self.num_batches = total_samples // batch_size if total_samples >= batch_size else 0

        # Bucket selection probabilities (proportional to sum of sample weights,
        # not bucket size, to preserve global group-balanced weighting intent)
        self.bucket_keys = list(self.buckets.keys())
        bucket_weight_sums = torch.tensor(
            [self.bucket_weights[k].sum().item() for k in self.bucket_keys],
            dtype=torch.float32
        )
        self.bucket_probs = bucket_weight_sums / bucket_weight_sums.sum()

        # Log bucket info
        for k in sorted(self.bucket_keys):
            print(f"  Channel bucket {k:2d}ch: {len(self.buckets[k]):5d} samples")

    def __iter__(self):
        if self.num_batches == 0 or len(self.bucket_keys) == 0:
            return
        for _ in range(self.num_batches):
            # Select bucket proportional to size
            bucket_idx = torch.multinomial(self.bucket_probs, 1).item()
            ch_count = self.bucket_keys[bucket_idx]
            indices = self.buckets[ch_count]
            weights = self.bucket_weights[ch_count]

            # Weighted sample within bucket (replacement=True for class balance)
            sampled = torch.multinomial(weights, self.batch_size, replacement=True)
            yield [indices[i] for i in sampled.tolist()]

    def __len__(self):
        return self.num_batches


# ======================== HYPERPARAMETERS ========================

# Data configuration
DATA_ROOT = "/home/alex/code/tsfm/data"
# Training datasets (10 diverse HAR datasets)
# Zero-shot test datasets are EXCLUDED: motionsense, realworld, mobiact, vtt_coniot
# Also excluded for GOAT comparison: opportunity, realdisp, daphnet_fog
DATASETS = ['uci_har', 'hhar', 'mhealth', 'pamap2', 'wisdm', 'unimib_shar', 'dsads', 'hapt', 'kuhar', 'recgym']
random.seed(42)
PATCH_SIZE_PER_DATASET = {
    # Fixed-length sessions (2.56s) - use 1.0s patches for 2 patches/session
    'uci_har': 1.0,       # 50 Hz, 2.56s fixed sessions
    'hhar': 1.0,          # 50 Hz, 2.56s fixed sessions
    # Variable-length sessions - max patch < min session duration
    'mhealth': 1.5,       # 50 Hz, min_session=2.0s → use 1.5s (was 2.0s)
    'pamap2': 2.0,        # 9 Hz, min_session=22.2s → plenty of room
    'wisdm': 1.5,         # 20 Hz, min_session=2.0s → use 1.5s (was 2.0s)
    'unimib_shar': 1.0,   # 50 Hz, 3.02s fixed sessions
    # New datasets
    'dsads': 2.0,         # 25 Hz, min_session=5.0s → use 2.0s
    'mobiact': 1.5,       # 50 Hz, min_session=2.0s → use 1.5s
    'realworld': 1.5,     # 50 Hz, min_session=2.0s → use 1.5s (was 2.0s)
    'vtt_coniot': 2.0,    # 50 Hz, min_session=60s → plenty of room
    'recgym': 1.5,        # 20 Hz, min_session=2.0s → use 1.5s (was 2.5s)
    'hapt': 1.25,         # 50 Hz, min_session=1.48s → use 1.25s (was 1.5s)
    'kuhar': 1.5,         # 100 Hz, min_session=2.0s → use 1.5s
    # Zero-shot datasets (NOT trained on, only for evaluation)
    'opportunity': 1.5,   # 30 Hz — zero-shot (GOAT baseline comparison)
    'realdisp': 1.5,      # 50 Hz — zero-shot (GOAT baseline comparison)
    'daphnet_fog': 1.5,   # 64 Hz — zero-shot (GOAT baseline comparison)
    'shoaib': 1.5,        # 50 Hz — zero-shot (LanHAR/CrossHAR baseline comparison)
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
EPOCHS = 100
BATCH_SIZE = 32  # Micro-batch size (BS=64 OOMs on 24GB RTX 4090, BS=32 fits)
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
MEMORY_BANK_SIZE = 256  # Provides batch + 256 negatives per step

# Class balancing configuration
# Caps oversampling to prevent rare labels from dominating training
# E.g., if walking has 50K samples and stand_to_sit has 200 samples,
# without cap: stand_to_sit sampled 250x more often (overfitting risk)
# with cap=20: stand_to_sit sampled only 20x more often
MAX_OVERSAMPLE_RATIO = 20.0  # Max oversampling factor for rare groups
SAMPLING_TEMPERATURE = 0.0  # Temperature for sampling: 0.0=balanced, 0.5=sqrt, 1.0=uniform

# NOTE: Channel augmentation (random subsampling/shuffling) is DISABLED.
# Experiments showed better zero-shot generalization with consistent channel order.

# NOTE: Channel projection (learnable MLP after SentenceBERT) is ENABLED.
# This allows task-specific adaptation of frozen semantic embeddings.

# Plotting
PLOT_EVERY_N_BATCHES = 50

# Embedding visualization
VISUALIZE_EMBEDDINGS = True  # Generate UMAP plots showing IMU-text alignment
VISUALIZE_EVERY_N_EPOCHS = 10  # How often to generate embedding plots

# Classification metrics during training
COMPUTE_CLASSIFICATION_METRICS = True  # Compute group-aware accuracy during validation
UNSEEN_DATASET = 'motionsense'  # Dataset for zero-shot evaluation (not in training)
EVAL_UNSEEN_EVERY = 5  # Evaluate on unseen dataset every N epochs
UMAP_N_NEIGHBORS = 15  # UMAP parameter: 5-50, higher = more global structure
UMAP_MIN_DIST = 0.1  # UMAP parameter: 0.0-0.99, lower = tighter clusters

# Debug configuration
DEBUG_METRIC_FREQUENCY = 50  # Compute expensive debug metrics every N batches (not every batch)

# Token-level text encoding configuration
# Uses cross-attention between sensor tokens and channel description tokens,
# plus learnable attention pooling for label refinement.
TOKEN_TEXT_NUM_HEADS = 4     # Attention heads for text fusion/pooling
TOKEN_TEXT_NUM_QUERIES = 4   # Learnable query tokens for label pooling
NUM_PROTOTYPES = 1           # Number of prototype embeddings per label (K=1 for single, K=3 for multi)

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
USE_ROTATION_AUGMENTATION = False  # Apply SO(3) rotation to sensor triads for orientation invariance
MIN_PATCHES_PER_SAMPLE = 1  # Minimum patches required per sample

# Valid patch size ranges per dataset: (min_sec, max_sec, step_sec)
# IMPORTANT: max_sec must be < min_session_duration for each dataset to guarantee valid patches
PATCH_SIZE_RANGE_PER_DATASET = {
    # Fixed-length sessions (~2.5-3s) - tighter range around 1.0s
    'uci_har':      (0.75, 1.25, 0.25),  # min_session=2.56s → [0.75, 1.0, 1.25]
    'hhar':         (0.75, 1.25, 0.25),  # min_session=2.56s → [0.75, 1.0, 1.25]
    'unimib_shar':  (0.75, 1.25, 0.25),  # min_session=3.02s → [0.75, 1.0, 1.25]
    # Variable-length sessions - max < min_session to avoid NaN
    'mhealth':      (1.0, 1.75, 0.25),   # min_session=2.0s → [1.0, 1.25, 1.5, 1.75]
    'pamap2':       (1.0, 2.0, 0.5),     # min_session=22.2s → [1.0, 1.5, 2.0]
    'wisdm':        (1.0, 1.75, 0.25),   # min_session=2.0s → [1.0, 1.25, 1.5, 1.75]
    # New datasets
    'dsads':        (1.5, 2.5, 0.5),     # min_session=5.0s → [1.5, 2.0, 2.5]
    'mobiact':      (1.0, 1.75, 0.25),   # min_session=2.0s → [1.0, 1.25, 1.5, 1.75]
    'realworld':    (1.0, 1.75, 0.25),   # min_session=2.0s → [1.0, 1.25, 1.5, 1.75]
    'vtt_coniot':   (1.5, 2.5, 0.5),     # min_session=60s → [1.5, 2.0, 2.5]
    'recgym':       (1.0, 1.75, 0.25),   # min_session=2.0s → [1.0, 1.25, 1.5, 1.75]
    'hapt':         (0.75, 1.25, 0.25),  # min_session=1.48s → [0.75, 1.0, 1.25]
    'kuhar':        (1.0, 1.75, 0.25),   # min_session=2.0s → [1.0, 1.25, 1.5, 1.75]
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
        min_patches_per_sample: int = 2,
        text_encoder: Optional['TokenTextEncoder'] = None
    ):
        super().__init__()
        self.encoder = encoder
        self.semantic_head = semantic_head
        self.use_patch_augmentation = use_patch_augmentation
        self.min_patches_per_sample = min_patches_per_sample

        # Token-level text encoder (frozen backbone) — shared with label_bank to save ~100MB
        self.text_encoder = text_encoder if text_encoder is not None else TokenTextEncoder()

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

    def forward(self, patches, channel_descriptions, channel_mask, patch_mask):
        """
        Forward pass with pre-patched data from DataLoader.

        Args:
            patches: (batch, max_patches, target_patch_size, max_channels) padded patches
            channel_descriptions: List of channel description lists per sample
            channel_mask: (batch, max_channels) boolean mask for valid channels
            patch_mask: (batch, max_patches) boolean mask for valid patches
        """
        batch_size = patches.shape[0]
        max_channels = patches.shape[3]
        device = patches.device

        if len(channel_descriptions) != batch_size:
            raise ValueError(
                f"Expected {batch_size} channel description lists, got {len(channel_descriptions)}"
            )
        for i, descs in enumerate(channel_descriptions):
            if len(descs) > max_channels:
                raise ValueError(
                    f"Sample {i} has {len(descs)} channel descriptions, exceeds max_channels={max_channels}"
                )

        # Pad channel descriptions to max_channels
        batched_channel_descs = [
            descs + ["[PAD]"] * (max_channels - len(descs))
            for descs in channel_descriptions
        ]

        # Encode with frozen/unfrozen encoder (already batched — patches from DataLoader)
        if FREEZE_ENCODER:
            with torch.no_grad():
                encoded_batch = self.encoder(patches, batched_channel_descs, channel_mask=channel_mask,
                                              patch_attention_mask=patch_mask)
        else:
            encoded_batch = self.encoder(patches, batched_channel_descs, channel_mask=channel_mask,
                                          patch_attention_mask=patch_mask)

        # Batch-encode ALL channel descriptions at once (per-label cache handles dedup)
        # Flatten: B lists of max_channels strings → B*max_channels strings
        all_descs = [desc for descs in batched_channel_descs for desc in descs]
        all_tokens, all_masks = self.text_encoder.encode(all_descs, device)

        # Reshape to (B, C, seq_len, D) for batched fusion
        seq_len = all_tokens.shape[1]
        text_tokens = all_tokens.reshape(batch_size, max_channels, seq_len, -1)
        text_masks = all_masks.reshape(batch_size, max_channels, seq_len)

        # Batched channel text fusion — single call replaces per-sample loop
        encoded_batch = self.channel_fusion(encoded_batch, text_tokens, text_masks)

        return self.semantic_head(encoded_batch, channel_mask=channel_mask,
                                   patch_mask=patch_mask, normalize=True)

    def _preprocess_raw_batch(self, data, channel_descriptions, channel_mask, sampling_rates, patch_sizes,
                              attention_mask=None):
        """
        Preprocess a batch of raw sensor data into padded patches for forward().

        Args:
            data: (batch, max_timesteps, max_channels) raw sensor data
            channel_descriptions: List of channel description lists per sample
            channel_mask: (batch, max_channels) boolean mask for valid channels
            sampling_rates: List of sampling rates per sample (Hz)
            patch_sizes: List of patch sizes per sample (seconds)
            attention_mask: Optional (batch, max_timesteps) boolean mask for valid timesteps

        Returns:
            Tuple of (batched_patches, patch_mask, batched_channel_mask, batched_channel_descs, valid_indices)
            or None if no valid samples.
        """
        batch_size = data.shape[0]
        device = data.device

        all_patches = []
        all_channel_descs = []
        valid_samples = []

        for i in range(batch_size):
            # Trim to valid timesteps
            if attention_mask is not None:
                valid_len = int(attention_mask[i].sum().item())
                sample_data = data[i, :valid_len]
            else:
                sample_data = data[i]

            patches, _ = self.encoder.preprocess(
                sample_data, sampling_rate_hz=sampling_rates[i], patch_size_sec=patch_sizes[i]
            )
            if patches is None or len(patches) == 0:
                all_patches.append(None)
                all_channel_descs.append(None)
                valid_samples.append(False)
                continue
            if len(patches) > MAX_PATCHES_PER_SAMPLE:
                patches = patches[:MAX_PATCHES_PER_SAMPLE]
            channel_descs = channel_descriptions[i]
            all_patches.append(patches)
            all_channel_descs.append(channel_descs)
            valid_samples.append(True)

        valid_indices = [i for i, v in enumerate(valid_samples) if v]
        if len(valid_indices) == 0:
            return None

        max_patches_valid = max(len(all_patches[i]) for i in valid_indices)
        max_channels = data.shape[2]
        num_valid = len(valid_indices)

        batched_patches = torch.zeros(num_valid, max_patches_valid, TARGET_PATCH_SIZE, max_channels, device=device)
        patch_mask = torch.zeros(num_valid, max_patches_valid, dtype=torch.bool, device=device)
        batched_channel_descs = []

        for batch_idx, sample_idx in enumerate(valid_indices):
            patches = all_patches[sample_idx]
            num_patches = len(patches)
            batched_patches[batch_idx, :num_patches] = patches
            patch_mask[batch_idx, :num_patches] = True
            batched_channel_descs.append(all_channel_descs[sample_idx])

        # Subset channel_mask to valid samples only
        valid_indices_t = torch.tensor(valid_indices, device=device)
        batched_channel_mask = channel_mask[valid_indices_t]

        return batched_patches, patch_mask, batched_channel_mask, batched_channel_descs, valid_indices

    def forward_from_raw(self, data, channel_descriptions, channel_mask, sampling_rates, patch_sizes,
                         attention_mask=None):
        """
        Forward pass from raw sensor data (for inference/evaluation).

        Preprocesses raw data per-sample (trim, patch, interpolate to TARGET_PATCH_SIZE),
        pads to uniform batch, then calls the efficient batched forward().

        Args:
            data: (batch, max_timesteps, max_channels) raw sensor data
            channel_descriptions: List of channel description lists per sample
            channel_mask: (batch, max_channels) boolean mask for valid channels
            sampling_rates: List of sampling rates per sample (Hz)
            patch_sizes: List of patch sizes per sample (seconds)
            attention_mask: Optional (batch, max_timesteps) boolean mask for valid timesteps

        Returns:
            (num_valid, semantic_dim) L2-normalized embeddings for valid samples.
            Samples that produce no valid patches are silently dropped.
        """
        result = self._preprocess_raw_batch(
            data, channel_descriptions, channel_mask, sampling_rates, patch_sizes, attention_mask
        )
        if result is None:
            # No valid samples — return empty embeddings
            return torch.zeros(0, SEMANTIC_DIM, device=data.device)

        batched_patches, patch_mask, batched_channel_mask, batched_channel_descs, valid_indices = result
        return self.forward(batched_patches, batched_channel_descs, batched_channel_mask, patch_mask)

    def get_attention_stats(self, data, channel_descriptions, channel_mask, sampling_rates, patch_sizes,
                            attention_mask=None):
        """Get attention statistics from cross-channel fusion (for debugging)."""
        with torch.no_grad():
            result = self._preprocess_raw_batch(
                data, channel_descriptions, channel_mask, sampling_rates, patch_sizes, attention_mask
            )
            if result is None:
                return {}

            batched_patches, patch_mask, batched_channel_mask, batched_channel_descs, valid_indices = result

            encoded_batch = self.encoder(batched_patches, batched_channel_descs, channel_mask=batched_channel_mask)

            # Get attention stats from semantic head (use valid-only channel_mask)
            return self.semantic_head.get_attention_stats(encoded_batch, batched_channel_mask)


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
        batch_size = dataloader.batch_size or BATCH_SIZE  # batch_size is None with batch_sampler
        num_batches = math.ceil(memory_bank.queue_size / batch_size)
    
    num_batches = min(num_batches, len(dataloader))
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            patches = batch['patches'].to(device)
            channel_mask = batch['channel_mask'].to(device)
            patch_mask = batch['patch_mask'].to(device)
            label_texts = batch['label_texts']
            metadata = batch['metadata']

            channel_descriptions = [m['channel_descriptions'] for m in metadata]

            # Encode text embeddings
            text_embeddings = label_bank.encode(label_texts, normalize=True)

            # Forward pass (no gradients)
            imu_embeddings = model(patches, channel_descriptions, channel_mask, patch_mask)

            # For multi-prototype: store the winning prototype (nearest to IMU)
            text_for_queue = text_embeddings
            if text_for_queue.dim() == 3:
                sims = torch.einsum('bd,bkd->bk', imu_embeddings, text_for_queue)
                best_idx = sims.argmax(dim=1)
                text_for_queue = text_for_queue[torch.arange(text_for_queue.shape[0]), best_idx]

            # Update memory bank with both embeddings
            memory_bank.update(imu_embeddings, text_for_queue)
            
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
    num_optimizer_steps = 0  # Count actual optimizer steps (not all batches)

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
    num_grad_computations = 0  # Count how many times per-layer grads were computed

    pbar = tqdm(dataloader, desc=f"[{stage}] Epoch {epoch} Training")

    # Initialize gradients at start of epoch
    optimizer.zero_grad()

    # Track accumulation steps for proper gradient logging
    accum_step_count = 0

    for batch_idx, batch in enumerate(pbar):
        patches = batch['patches'].to(device, non_blocking=True)
        channel_mask = batch['channel_mask'].to(device, non_blocking=True)
        patch_mask = batch['patch_mask'].to(device, non_blocking=True)
        label_texts = batch['label_texts']
        metadata = batch['metadata']

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
            imu_embeddings = model(patches, channel_descriptions, channel_mask, patch_mask)
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

            # Collect trainable parameters for gradient clipping
            # Clip model/label_bank params separately from logit_scale to prevent
            # temperature gradient from dominating and starving model updates
            model_and_bank_params = (
                list(filter(lambda p: p.requires_grad, model.parameters())) +
                list(label_bank.parameters())
            )
            logit_scale_params = list(criterion.parameters())

            if scaler is not None:
                scaler.unscale_(optimizer)

                # Per-layer gradient norms (expensive - only compute periodically)
                if should_log_grads:
                    batch_grad_norms = _compute_per_layer_grad_norms(model, criterion, label_bank)

                # Separate clipping: model/label_bank get their own budget,
                # logit_scale can't steal it even if its gradient spikes
                grad_norm = torch.nn.utils.clip_grad_norm_(model_and_bank_params, max_norm=MAX_GRAD_NORM)
                torch.nn.utils.clip_grad_norm_(logit_scale_params, max_norm=MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Per-layer gradient norms (expensive - only compute periodically)
                if should_log_grads:
                    batch_grad_norms = _compute_per_layer_grad_norms(model, criterion, label_bank)

                grad_norm = torch.nn.utils.clip_grad_norm_(model_and_bank_params, max_norm=MAX_GRAD_NORM)
                torch.nn.utils.clip_grad_norm_(logit_scale_params, max_norm=MAX_GRAD_NORM)
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
                # For multi-prototype: store the winning prototype (nearest to IMU)
                text_for_queue = text_embeddings.detach()
                if text_for_queue.dim() == 3:
                    # (B, K, D) -> pick best prototype per sample
                    sims = torch.einsum('bd,bkd->bk', imu_embeddings.detach(), text_for_queue)
                    best_idx = sims.argmax(dim=1)
                    text_for_queue = text_for_queue[torch.arange(text_for_queue.shape[0]), best_idx]
                memory_bank.update(imu_embeddings.detach(), text_for_queue)

        # Accumulate metrics (with NaN detection)
        batch_loss = metrics['loss']
        if math.isnan(batch_loss):
            # Log which batch/dataset caused NaN
            datasets_in_batch = [m.get('dataset', 'unknown') for m in metadata]
            dataset_counts = {}
            for ds in datasets_in_batch:
                dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
            print(f"\n[NaN DETECTED] Batch {batch_idx}, Epoch {epoch}")
            print(f"  Datasets in batch: {dataset_counts}")
            print(f"  Labels: {label_texts[:5]}...")
            # Skip this batch's contribution to avoid NaN propagation
            # (the loss still backprops, but we don't let it corrupt epoch metrics)
            batch_loss = 0.0  # Replace NaN with 0 for epoch averaging

        total_loss += batch_loss
        total_pos_sim += metrics['positive_similarity']
        total_neg_sim += metrics['negative_similarity']
        total_sim_gap += metrics['similarity_gap']

        # Only accumulate grad_norm on optimizer steps (non-step batches have grad_norm=0)
        if is_accumulation_step or is_last_batch:
            total_grad_norm += grad_norm.item()
            num_optimizer_steps += 1

        # Accumulate debug metrics (only when computed)
        if debug_metrics:
            total_imu_std += debug_metrics['imu_std']
            total_text_std += debug_metrics['text_std']
            total_imu_diversity += debug_metrics['imu_diversity']
            if 'queue_diversity' in debug_metrics:
                total_queue_diversity += debug_metrics['queue_diversity']

        # Accumulate per-layer gradient norms (only when actually computed, not zeros)
        if is_accumulation_step or is_last_batch:
            should_log_grads_here = (accum_step_count - 1) % DEBUG_METRIC_FREQUENCY == 0
            if should_log_grads_here:
                for k in total_grad_norms:
                    total_grad_norms[k] += batch_grad_norms[k]
                num_grad_computations += 1
            else:
                pass  # batch_grad_norms is all zeros, skip
        # (non-optimizer-step batches always have batch_grad_norms=0, skip)

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
    # Debug metrics (imu_std etc.) are computed every DEBUG_METRIC_FREQUENCY batches
    debug_count = (num_batches + DEBUG_METRIC_FREQUENCY - 1) // DEBUG_METRIC_FREQUENCY  # Ceiling division
    debug_count = max(debug_count, 1)  # Avoid division by zero

    return {
        'loss': total_loss / num_batches,
        'positive_similarity': total_pos_sim / num_batches,
        'negative_similarity': total_neg_sim / num_batches,
        'similarity_gap': total_sim_gap / num_batches,
        'grad_norm': total_grad_norm / max(num_optimizer_steps, 1),
        # Debug metrics (computed every DEBUG_METRIC_FREQUENCY batches)
        'imu_std': total_imu_std / debug_count,
        'text_std': total_text_std / debug_count,
        'imu_diversity': total_imu_diversity / debug_count,
        'queue_diversity': total_queue_diversity / debug_count if USE_MEMORY_BANK else 0.0,
        # Per-component gradient norms (computed every DEBUG_METRIC_FREQUENCY optimizer steps)
        **{f'{k}_grad_norm': v / max(num_grad_computations, 1) for k, v in total_grad_norms.items()}
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
            patches = batch['patches'].to(device, non_blocking=True)
            channel_mask = batch['channel_mask'].to(device, non_blocking=True)
            patch_mask = batch['patch_mask'].to(device, non_blocking=True)
            label_texts = batch['label_texts']
            metadata = batch['metadata']

            channel_descriptions = [m['channel_descriptions'] for m in metadata]

            text_embeddings = label_bank.encode(label_texts, normalize=True)

            with autocast('cuda', enabled=device.type == 'cuda'):
                imu_embeddings = model(patches, channel_descriptions, channel_mask, patch_mask)
                _, metrics = criterion(imu_embeddings, text_embeddings, label_texts, return_metrics=True)

            # NaN detection for validation
            batch_loss = metrics['loss']
            if math.isnan(batch_loss):
                datasets_in_batch = [m.get('dataset', 'unknown') for m in metadata]
                print(f"\n[NaN DETECTED in VAL] Labels: {label_texts[:5]}..., Datasets: {set(datasets_in_batch)}")
                batch_loss = 0.0

            total_loss += batch_loss
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
            patches = batch['patches'].to(device)
            channel_mask = batch['channel_mask'].to(device)
            patch_mask = batch['patch_mask'].to(device)
            label_texts = batch['label_texts']
            metadata = batch['metadata']

            channel_descriptions = [m['channel_descriptions'] for m in metadata]

            with autocast('cuda', enabled=device.type == 'cuda'):
                imu_embeddings = model(patches, channel_descriptions, channel_mask, patch_mask)

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
                             'use_mean_pooling': USE_MEAN_POOLING, 'freeze_label_bank': FREEZE_LABEL_BANK,
                             'num_prototypes': NUM_PROTOTYPES},
        'semantic_head': {'num_temporal_layers': NUM_SEMANTIC_TEMPORAL_LAYERS,
                          'num_fusion_queries': NUM_FUSION_QUERIES, 'use_fusion_self_attention': USE_FUSION_SELF_ATTENTION,
                          'num_pool_queries': NUM_POOL_QUERIES, 'use_pool_self_attention': USE_POOL_SELF_ATTENTION}
    }
    with open(CHECKPOINT_DIR / 'hyperparameters.json', 'w') as f:
        json.dump(hyperparams, f, indent=2)

    torch.manual_seed(SEED)

    # Enable cuDNN benchmark and TF32 for faster matmul on RTX 4090
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ Enabled cuDNN benchmark + TF32")

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

    # Create shared text encoder (one MiniLM instance for model + label_bank, saves ~100MB GPU)
    shared_text_encoder = TokenTextEncoder()

    print("Using token-level text encoding with cross-attention")
    model = SemanticAlignmentModel(
        encoder, semantic_head,
        num_heads=TOKEN_TEXT_NUM_HEADS,
        dropout=DROPOUT,
        use_patch_augmentation=USE_PATCH_SIZE_AUGMENTATION,
        min_patches_per_sample=MIN_PATCHES_PER_SAMPLE,
        text_encoder=shared_text_encoder
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
        num_prototypes=NUM_PROTOTYPES,
        dropout=0.0,  # Zero dropout: contrastive text targets must be deterministic
        use_mean_pooling=USE_MEAN_POOLING,
        text_encoder=shared_text_encoder  # Share MiniLM with model
    )
    print(f"✓ Label bank initialized (embedding_dim={label_bank.embedding_dim}, pooling={pooling_type})")

    criterion = SemanticAlignmentLoss(
        temperature=TEMPERATURE, use_soft_targets=USE_SOFT_TARGETS,
        soft_target_temperature=SOFT_TARGET_TEMPERATURE, soft_target_weight=SOFT_TARGET_WEIGHT
    ).to(device)

    # Initialize memory bank if enabled
    memory_bank = None
    if USE_MEMORY_BANK:
        print(f"Initializing memory bank (queue_size={MEMORY_BANK_SIZE}, embedding_dim={SEMANTIC_DIM}, device={device})...")
        memory_bank = MemoryBank(queue_size=MEMORY_BANK_SIZE, embedding_dim=SEMANTIC_DIM, device=device)
        print(f"✓ Memory bank initialized on {device} - provides {MEMORY_BANK_SIZE} additional negatives (no CPU round-trips)")

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
        seed=SEED,
        target_patch_size=TARGET_PATCH_SIZE,
        max_patches_per_sample=MAX_PATCHES_PER_SAMPLE,
        use_rotation_augmentation=USE_ROTATION_AUGMENTATION,
    )
    val_dataset = IMUPretrainingDataset(
        data_root=DATA_ROOT,
        datasets=DATASETS,
        split='val',
        patch_size_per_dataset=PATCH_SIZE_PER_DATASET,
        patch_size_range_per_dataset=None,  # No augmentation for validation
        max_sessions_per_dataset=MAX_SESSIONS_PER_DATASET,
        seed=SEED,
        target_patch_size=TARGET_PATCH_SIZE,
        max_patches_per_sample=MAX_PATCHES_PER_SAMPLE,
    )

    # Create train dataloader with channel-bucketed + group-balanced sampling
    if USE_GROUP_BALANCED_SAMPLING:
        # Compute weights for group-balanced sampling (with capped oversampling)
        sample_weights = train_dataset.compute_group_weights(max_oversample_ratio=MAX_OVERSAMPLE_RATIO, sampling_temperature=SAMPLING_TEMPERATURE)

        # Channel-bucketed batch sampler: groups same-channel-count samples
        # to minimize padding waste (joint util ~11% → ~80-100%)
        channel_counts = train_dataset.get_channel_counts()
        print(f"\n=== Channel-Bucketed + Group-Balanced Sampling ===")
        bucket_sampler = ChannelBucketBatchSampler(
            channel_counts=channel_counts,
            sample_weights=sample_weights,
            batch_size=BATCH_SIZE,
            num_samples=len(train_dataset)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=bucket_sampler,  # Replaces batch_size + sampler
            num_workers=NUM_WORKERS,
            prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=PERSISTENT_WORKERS,
            collate_fn=IMUPretrainingDataset.collate_patches_fn,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )

        # Log group distribution
        group_dist = train_dataset.get_group_distribution()
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
            collate_fn=IMUPretrainingDataset.collate_patches_fn,
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
        collate_fn=IMUPretrainingDataset.collate_patches_fn,
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
                seed=SEED,
                target_patch_size=TARGET_PATCH_SIZE,
                max_patches_per_sample=MAX_PATCHES_PER_SAMPLE,
            )
            unseen_loader = DataLoader(
                unseen_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                collate_fn=IMUPretrainingDataset.collate_patches_fn,
                pin_memory=True
            )
            print(f"Loaded unseen dataset '{UNSEEN_DATASET}' with {len(unseen_dataset)} samples")
        except Exception as e:
            print(f"Warning: Could not load unseen dataset '{UNSEEN_DATASET}': {e}")

    # Setup optimizer (trains semantic head only if encoder is frozen, otherwise all parameters)
    # IMPORTANT: Include criterion.parameters() for learnable temperature (logit_scale)
    # and label_bank.parameters() for learnable attention pooling (if not using mean pooling)

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
        print(f"✓ Label bank parameters added to optimizer ({sum(p.numel() for p in label_bank_params)} params)")

    # Collect all named parameters for decay/no-decay grouping
    # Standard practice: exempt biases, norms, and learnable scalars/embeddings from weight decay
    no_decay_keywords = {"bias", "norm", "LayerNorm", "GroupNorm", "logit_scale",
                         "mask_token", "pad_token", "queries"}
    decay_params = []
    no_decay_params = []

    # Model parameters (encoder + semantic head + channel fusion)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(kw in name for kw in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # Criterion parameters (logit_scale) — no decay
    for name, param in criterion.named_parameters():
        if param.requires_grad:
            no_decay_params.append(param)

    # Label bank parameters — no decay for queries/norms, decay for projections
    if not USE_MEAN_POOLING and not FREEZE_LABEL_BANK and len(label_bank_params) > 0:
        for name, param in label_bank.named_parameters():
            if not param.requires_grad:
                continue
            if any(kw in name for kw in no_decay_keywords):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    print(f"✓ Parameter groups: {len(decay_params)} decay, {len(no_decay_params)} no-decay "
          f"({sum(p.numel() for p in decay_params)} + {sum(p.numel() for p in no_decay_params)} params)")

    optimizer = AdamW([
        {'params': decay_params, 'weight_decay': WEIGHT_DECAY},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=LEARNING_RATE)

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

        # Load GradScaler state if available (prevents FP16 overflow on resume)
        if scaler is not None and 'scaler_state_dict' in resume_checkpoint:
            if resume_checkpoint['scaler_state_dict'] is not None:
                scaler.load_state_dict(resume_checkpoint['scaler_state_dict'])
                print(f"✓ Loaded GradScaler state (scale={scaler.get_scale():.1f})")

    # Apply torch.compile for JIT optimization (after all state loading)
    # NOTE: Cannot compile the full encoder because its positional_encoding path uses
    # SentenceTransformer which is incompatible with torch.dynamo tracing.
    # Instead, compile the heavy sub-modules: encoder's transformer, channel_fusion, semantic_head.
    if False and device.type == 'cuda' and hasattr(torch, 'compile'):  # Disabled: dynamic shape assertions fail with BS=64
        compile_backend = None
        _test_model = None
        try:
            _test_model = torch.compile(torch.nn.Linear(8, 8).to(device), dynamic=True)
            _test_model(torch.randn(2, 8, device=device))
            compile_backend = 'inductor'
        except Exception:
            try:
                _test_model = torch.compile(torch.nn.Linear(8, 8).to(device), backend='aot_eager', dynamic=True)
                _test_model(torch.randn(2, 8, device=device))
                compile_backend = 'aot_eager'
            except Exception:
                pass
        del _test_model

        if compile_backend:
            compile_kwargs = {'dynamic': True}
            if compile_backend != 'inductor':
                compile_kwargs['backend'] = compile_backend
            print(f"\nApplying torch.compile (backend={compile_backend}, dynamic=True)...")
            # Compile encoder's transformer (the heavy part) — skip positional_encoding (SentenceTransformer)
            model.encoder.transformer = torch.compile(model.encoder.transformer, **compile_kwargs)
            model.encoder.feature_extractor = torch.compile(model.encoder.feature_extractor, **compile_kwargs)
            model.channel_fusion = torch.compile(model.channel_fusion, **compile_kwargs)
            model.semantic_head = torch.compile(model.semantic_head, **compile_kwargs)
            print("✓ torch.compile applied to transformer, feature_extractor, channel_fusion, semantic_head")
        else:
            print("\nWarning: torch.compile not available, using eager mode")

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
            # Strip _orig_mod. prefix from compiled modules so checkpoints
            # are compatible with both compiled and uncompiled models
            raw_state = model.state_dict()
            clean_state = {k.replace('_orig_mod.', ''): v for k, v in raw_state.items()}
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': clean_state,
                'label_bank_state_dict': label_bank.state_dict(),  # Save learnable label embeddings
                'criterion_state_dict': criterion.state_dict(),  # Save learnable temperature
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'memory_bank_state_dict': memory_bank.state_dict() if memory_bank else None,
                'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
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
