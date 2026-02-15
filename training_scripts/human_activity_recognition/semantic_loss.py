"""
Semantic alignment loss functions.

Implements InfoNCE contrastive loss for aligning IMU embeddings with text embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional

# Debug flag - set to True to enable NaN debugging
DEBUG_NAN = False  # Set to True to enable verbose NaN debugging
_nan_debug_count = 0  # Track how many times we've printed debug info


def _check_tensor_health(tensor: torch.Tensor, name: str) -> dict:
    """Check a tensor for NaN, Inf, and other issues."""
    with torch.no_grad():
        info = {
            'name': name,
            'shape': tuple(tensor.shape),
            'dtype': str(tensor.dtype),
            'has_nan': torch.isnan(tensor).any().item(),
            'has_inf': torch.isinf(tensor).any().item(),
            'min': tensor.min().item() if tensor.numel() > 0 else None,
            'max': tensor.max().item() if tensor.numel() > 0 else None,
            'mean': tensor.mean().item() if tensor.numel() > 0 else None,
            'std': tensor.std().item() if tensor.numel() > 0 and tensor.numel() > 1 else None,
        }
        # Check for zeros (potential underflow)
        if tensor.numel() > 0:
            info['num_zeros'] = (tensor == 0).sum().item()
            info['pct_zeros'] = info['num_zeros'] / tensor.numel() * 100
        return info


def _debug_nan_loss(context: dict):
    """Print detailed debug info when NaN loss is detected."""
    global _nan_debug_count
    _nan_debug_count += 1

    # Only print first 5 occurrences to avoid spam
    if _nan_debug_count > 5:
        if _nan_debug_count == 6:
            print(f"\n[NaN DEBUG] Suppressing further NaN debug output (already printed 5 times)")
        return

    print(f"\n{'='*60}")
    print(f"[NaN DEBUG] NaN detected in loss computation (occurrence #{_nan_debug_count})")
    print(f"{'='*60}")

    for key, value in context.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")

    print(f"{'='*60}\n")


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss for semantic alignment.

    Aligns IMU embeddings with text embeddings in a shared semantic space.
    Supports:
    - Hard targets (one-hot)
    - Pairwise soft targets (semantic similarity)
    """

    def __init__(
        self,
        temperature: float = 0.1,
        use_soft_targets: bool = True,
        soft_target_temperature: float = 0.5,
        soft_target_weight: float = 0.5
    ):
        """
        Args:
            temperature: Temperature parameter for scaling similarities
            use_soft_targets: Whether to use soft targets based on label similarity
            soft_target_temperature: Temperature for computing soft target distribution
            soft_target_weight: Weight for soft targets (1-weight goes to hard targets)
        """
        super().__init__()
        # Learnable temperature (CLIP-style): initialized to 1/0.07 ≈ 14.3
        # Temperature is learned as exp(logit_scale) to ensure positivity
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/temperature))
        self.use_soft_targets = use_soft_targets
        self.soft_target_temperature = soft_target_temperature
        self.soft_target_weight = soft_target_weight

    def forward(
        self,
        imu_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        label_texts: Optional[list] = None,
        return_metrics: bool = True,
        imu_queue: Optional[torch.Tensor] = None,
        text_queue: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        Compute InfoNCE loss with optional soft targets and memory bank queue.

        Args:
            imu_embeddings: IMU embeddings (batch_size, embedding_dim), L2-normalized
            text_embeddings: Text embeddings (batch_size, embedding_dim), L2-normalized
            label_texts: Optional list of label text strings (unused, kept for API compatibility)
            return_metrics: Whether to return additional metrics
            imu_queue: Optional queue of past IMU embeddings (queue_size, embedding_dim)
            text_queue: Optional queue of past text embeddings (queue_size, embedding_dim)

        Returns:
            loss: Scalar loss value
            metrics: Optional dict with additional metrics (accuracy, etc.)
        """
        batch_size = imu_embeddings.shape[0]

        # Embeddings are now normalized in the model's forward pass (ProjectionHead)
        # This avoids double normalization and keeps gradient flow clean
        # Text embeddings are already normalized by label_bank.encode()
        # Queue embeddings were normalized when they were created
        # So we DON'T normalize again here

        # Expand embeddings with queue if provided (for more negatives)
        if imu_queue is not None and text_queue is not None and len(imu_queue) > 0:
            all_imu = torch.cat([imu_embeddings, imu_queue.detach()], dim=0)
            all_text = torch.cat([text_embeddings, text_queue.detach()], dim=0)
        else:
            all_imu = imu_embeddings
            all_text = text_embeddings

        # Compute cosine similarity matrix
        # Current batch (queries) vs all embeddings (keys: current + queue)
        # Both embeddings should already be L2-normalized
        # Scale by learnable temperature (clamped to prevent explosion)
        logits = torch.matmul(imu_embeddings, all_text.T) * self.logit_scale.exp().clamp(1, 50)  # (batch, batch+queue)

        # Compute soft targets over ENTIRE dimension (batch + queue) for proper normalization
        sim_mean_for_metrics = None  # Track for monitoring
        if self.use_soft_targets:
            # Compute text similarity over full dimension (batch + queue)
            # Note: Queue text embeddings have minor staleness (pooling layer evolves),
            # but with small queue (256) and gradual updates this is acceptable (standard MoCo practice)
            text_similarity_full = torch.matmul(text_embeddings, all_text.T)  # (batch, batch+queue)

            # Adaptive soft targets: normalize similarities to z-scores
            # Problem: SentenceBERT gives 0.4-0.9 for ALL human activities → weak discrimination
            # Solution: Convert to z-scores so differences are amplified
            # - True synonyms (walking/strolling): z ≈ +2 → high weight
            # - Different activities (walking/sitting): z ≈ -1 → low weight
            sim_mean = text_similarity_full.mean()
            sim_std = text_similarity_full.std().clamp(min=0.1)
            sim_mean_for_metrics = sim_mean.item()
            text_similarity_full = (text_similarity_full - sim_mean) / sim_std / self.soft_target_temperature

            # Apply softmax over FULL dimension to get proper probability distribution
            soft_targets_full = F.softmax(text_similarity_full, dim=1)  # Sums to 1.0 per row ✓

            # Create hard targets (identity for batch, zeros for queue)
            queue_size = all_text.shape[0] - batch_size
            hard_targets_batch = torch.eye(batch_size, device=imu_embeddings.device)
            hard_targets_queue = torch.zeros(batch_size, queue_size, device=imu_embeddings.device)
            hard_targets_full = torch.cat([hard_targets_batch, hard_targets_queue], dim=1)

            # Blend: (1-w) * hard + w * soft (still sums to 1.0 per row)
            targets = (1 - self.soft_target_weight) * hard_targets_full + \
                     self.soft_target_weight * soft_targets_full
        else:
            # Hard targets only: identity for batch, zeros for queue
            queue_size = all_text.shape[0] - batch_size
            hard_targets_batch = torch.eye(batch_size, device=imu_embeddings.device)
            if queue_size > 0:
                hard_targets_queue = torch.zeros(batch_size, queue_size, device=imu_embeddings.device)
                targets = torch.cat([hard_targets_batch, hard_targets_queue], dim=1)
            else:
                targets = hard_targets_batch

        # Compute loss
        # Text→IMU direction (text queries vs IMU keys)
        logits_t2i = torch.matmul(text_embeddings, all_imu.T) * self.logit_scale.exp().clamp(1, 50)  # (batch, batch+queue)

        if self.use_soft_targets:
            # Soft targets: use KL divergence
            log_probs = F.log_softmax(logits, dim=1)
            loss_imu_to_text = -(targets * log_probs).sum(dim=1).mean()

            log_probs_t2i = F.log_softmax(logits_t2i, dim=1)
            loss_text_to_imu = -(targets * log_probs_t2i).sum(dim=1).mean()
        else:
            # Hard targets: use cross-entropy
            labels = torch.arange(batch_size, device=imu_embeddings.device)
            loss_imu_to_text = F.cross_entropy(logits, labels)
            loss_text_to_imu = F.cross_entropy(logits_t2i, labels)

        loss = (loss_imu_to_text + loss_text_to_imu) / 2.0

        # === NaN DEBUGGING ===
        if DEBUG_NAN and (torch.isnan(loss) or torch.isnan(loss_text_to_imu) or torch.isnan(loss_imu_to_text)):
            with torch.no_grad():
                # Compute per-sample losses to find which samples cause NaN
                if self.use_soft_targets:
                    per_sample_loss_i2t = -(targets * log_probs).sum(dim=1)  # (batch,)
                    per_sample_loss_t2i = -(targets * log_probs_t2i).sum(dim=1)  # (batch,)
                else:
                    per_sample_loss_i2t = F.cross_entropy(logits, labels, reduction='none')
                    per_sample_loss_t2i = F.cross_entropy(logits_t2i, labels, reduction='none')

                nan_samples_i2t = torch.where(torch.isnan(per_sample_loss_i2t))[0].tolist()
                nan_samples_t2i = torch.where(torch.isnan(per_sample_loss_t2i))[0].tolist()

                # Check for -inf in log_probs (indicates softmax output was 0)
                inf_in_log_probs = torch.isinf(log_probs).any().item()
                inf_in_log_probs_t2i = torch.isinf(log_probs_t2i).any().item()

                # Check for zeros in targets (would cause 0 * -inf = NaN)
                zeros_in_targets = (targets == 0).sum().item()
                min_target = targets.min().item()

                # Check queue health
                queue_size = all_imu.shape[0] - batch_size

                context = {
                    'loss_values': {
                        'loss': loss.item(),
                        'loss_imu_to_text': loss_imu_to_text.item(),
                        'loss_text_to_imu': loss_text_to_imu.item(),
                    },
                    'nan_sample_indices': {
                        'i2t': nan_samples_i2t,
                        't2i': nan_samples_t2i,
                    },
                    'log_probs_health': {
                        'has_inf_i2t': inf_in_log_probs,
                        'has_inf_t2i': inf_in_log_probs_t2i,
                        'log_probs_min': log_probs.min().item(),
                        'log_probs_t2i_min': log_probs_t2i.min().item(),
                    },
                    'targets_health': {
                        'zeros_count': zeros_in_targets,
                        'min_value': min_target,
                        'shape': tuple(targets.shape),
                    },
                    'imu_embeddings': _check_tensor_health(imu_embeddings, 'imu_embeddings'),
                    'text_embeddings': _check_tensor_health(text_embeddings, 'text_embeddings'),
                    'all_imu': _check_tensor_health(all_imu, 'all_imu'),
                    'all_text': _check_tensor_health(all_text, 'all_text'),
                    'logits': _check_tensor_health(logits, 'logits'),
                    'logits_t2i': _check_tensor_health(logits_t2i, 'logits_t2i'),
                    'queue_info': {
                        'queue_size': queue_size,
                        'imu_queue_provided': imu_queue is not None,
                        'text_queue_provided': text_queue is not None,
                    },
                }

                # If queue is provided, check its health
                if imu_queue is not None and len(imu_queue) > 0:
                    context['imu_queue'] = _check_tensor_health(imu_queue, 'imu_queue')
                if text_queue is not None and len(text_queue) > 0:
                    context['text_queue'] = _check_tensor_health(text_queue, 'text_queue')

                # Check specific problematic samples
                if nan_samples_t2i:
                    sample_idx = nan_samples_t2i[0]
                    context['problematic_sample'] = {
                        'sample_idx': sample_idx,
                        'label': label_texts[sample_idx] if label_texts else 'N/A',
                        'imu_emb_norm': torch.norm(imu_embeddings[sample_idx]).item(),
                        'text_emb_norm': torch.norm(text_embeddings[sample_idx]).item(),
                        'logits_t2i_row': logits_t2i[sample_idx].tolist()[:10],  # First 10 values
                        'log_probs_t2i_row_min': log_probs_t2i[sample_idx].min().item(),
                        'targets_row_min': targets[sample_idx].min().item(),
                        'targets_row_sum': targets[sample_idx].sum().item(),
                    }

                _debug_nan_loss(context)

        # Compute metrics
        metrics = None
        if return_metrics:
            with torch.no_grad():
                # Raw similarity matrix (unscaled) for metrics
                raw_sim = torch.matmul(imu_embeddings, text_embeddings.T)  # (batch, batch)

                # Positive similarity: diagonal pairs (matched IMU-text)
                positive_sim = torch.diagonal(raw_sim).mean().item()

                # Label-aware negative similarity:
                # Use actual label strings (mapped to groups) to identify same-activity pairs
                # This handles synonyms like "walking"/"nordic_walking" → same group
                if label_texts is not None and len(label_texts) == batch_size:
                    # Map labels to groups (synonyms → same group)
                    from val_scripts.human_activity_recognition.evaluation_metrics import get_label_to_group_mapping
                    label_to_group = get_label_to_group_mapping()
                    label_groups = [label_to_group.get(lbl, lbl) for lbl in label_texts]

                    # Build same-group mask
                    # same_label_mask[i,j] = True if labels belong to same group
                    same_label_mask = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=raw_sim.device)
                    for i in range(batch_size):
                        for j in range(batch_size):
                            same_label_mask[i, j] = (label_groups[i] == label_groups[j])
                else:
                    # Fallback: only exclude diagonal (self-pairs)
                    same_label_mask = torch.eye(batch_size, dtype=torch.bool, device=raw_sim.device)

                # True negatives: different labels
                diff_label_mask = ~same_label_mask
                if diff_label_mask.any():
                    true_neg_sim = raw_sim[diff_label_mask].mean().item()
                else:
                    # Fallback if all pairs are same-label (unlikely)
                    true_neg_sim = 0.0

                # Similarity gap: positive vs TRUE negatives (label-aware)
                similarity_gap = positive_sim - true_neg_sim

                metrics = {
                    'loss': loss.item(),
                    'loss_imu_to_text': loss_imu_to_text.item(),
                    'loss_text_to_imu': loss_text_to_imu.item(),
                    'positive_similarity': positive_sim,
                    'negative_similarity': true_neg_sim,
                    'similarity_gap': similarity_gap
                }

                # Add soft target metrics if used
                if self.use_soft_targets:
                    # Compute average semantic similarity of labels in batch
                    text_similarity = torch.matmul(text_embeddings, text_embeddings.T)
                    batch_mask = torch.eye(batch_size, device=logits.device, dtype=torch.bool)
                    label_similarity = text_similarity[~batch_mask].mean().item()
                    metrics['label_semantic_similarity'] = label_similarity
                    # Track the mean/std similarity used for adaptive normalization
                    if sim_mean_for_metrics is not None:
                        metrics['soft_target_sim_mean'] = sim_mean_for_metrics

                # Track learnable temperature (logit_scale)
                metrics['logit_scale'] = self.logit_scale.exp().item()

                # Track soft target distribution stats (how concentrated vs uniform)
                if self.use_soft_targets:
                    # Entropy of soft targets (higher = more uniform, lower = more concentrated)
                    # Max entropy for N classes = log(N)
                    soft_target_entropy = -(targets * torch.log(targets + 1e-10)).sum(dim=1).mean().item()
                    max_entropy = np.log(targets.shape[1])
                    metrics['soft_target_entropy'] = soft_target_entropy
                    metrics['soft_target_entropy_ratio'] = soft_target_entropy / max_entropy  # 1.0 = uniform

                    # How much probability mass is on the true positive?
                    true_positive_prob = torch.diagonal(targets[:, :batch_size]).mean().item()
                    metrics['true_positive_target_prob'] = true_positive_prob

                # Track logit statistics (before softmax)
                metrics['logits_mean'] = logits.mean().item()
                metrics['logits_std'] = logits.std().item()
                metrics['logits_max'] = logits.max().item()
                metrics['logits_min'] = logits.min().item()

        return loss, metrics


class SemanticAlignmentLoss(nn.Module):
    """
    Combined loss for semantic alignment training.

    Supports multiple loss components that can be weighted.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        use_soft_targets: bool = True,
        soft_target_temperature: float = 0.5,
        soft_target_weight: float = 0.5
    ):
        """
        Args:
            temperature: Temperature for InfoNCE
            use_soft_targets: Whether to use soft targets based on label similarity
            soft_target_temperature: Temperature for computing soft target distribution
            soft_target_weight: Weight for soft targets (0=hard, 1=pure soft, 0.5=balanced)
        """
        super().__init__()
        self.infonce = InfoNCELoss(
            temperature=temperature,
            use_soft_targets=use_soft_targets,
            soft_target_temperature=soft_target_temperature,
            soft_target_weight=soft_target_weight
        )

    def forward(
        self,
        imu_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        label_texts: Optional[list] = None,
        return_metrics: bool = True,
        imu_queue: Optional[torch.Tensor] = None,
        text_queue: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        Compute semantic alignment loss.

        Args:
            imu_embeddings: IMU embeddings (batch_size, embedding_dim)
            text_embeddings: Text embeddings (batch_size, embedding_dim)
            label_texts: Optional list of label texts (needed for prototypes)
            return_metrics: Whether to return additional metrics
            imu_queue: Optional queue of past IMU embeddings (queue_size, embedding_dim)
            text_queue: Optional queue of past text embeddings (queue_size, embedding_dim)

        Returns:
            loss: Scalar loss value
            metrics: Optional dict with metrics
        """
        return self.infonce(imu_embeddings, text_embeddings, label_texts, return_metrics,
                           imu_queue, text_queue)


