"""
Semantic alignment loss functions.

Implements InfoNCE contrastive loss for aligning IMU embeddings with text embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


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
        logits = torch.matmul(imu_embeddings, all_text.T) * self.logit_scale.exp().clamp(0, 100)  # (batch, batch+queue)

        # Compute soft targets over ENTIRE dimension (batch + queue) for proper normalization
        sim_mean_for_metrics = None  # Track for monitoring
        if self.use_soft_targets:
            # Compute text similarity over full dimension (batch + queue)
            # Key insight: Queue text embeddings don't go stale (from frozen SentenceBERT)
            text_similarity_full = torch.matmul(text_embeddings, all_text.T)  # (batch, batch+queue)

            # Adaptive soft targets: normalize similarities to z-scores
            # Problem: SentenceBERT gives 0.4-0.9 for ALL human activities → weak discrimination
            # Solution: Convert to z-scores so differences are amplified
            # - True synonyms (walking/strolling): z ≈ +2 → high weight
            # - Different activities (walking/sitting): z ≈ -1 → low weight
            sim_mean = text_similarity_full.mean()
            sim_std = text_similarity_full.std() + 1e-6
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
        logits_t2i = torch.matmul(text_embeddings, all_imu.T) * self.logit_scale.exp().clamp(0, 100)  # (batch, batch+queue)

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

        # Compute metrics
        metrics = None
        if return_metrics:
            with torch.no_grad():
                # Accuracy: how often is the correct match the highest similarity?
                labels = torch.arange(batch_size, device=imu_embeddings.device)
                imu_to_text_acc = (logits.argmax(dim=1) == labels).float().mean().item()
                text_to_imu_acc = (logits_t2i.argmax(dim=1) == labels).float().mean().item()

                # Average similarity of positive pairs (unscale by dividing by learned temperature)
                positive_sim = torch.diagonal(logits).mean().item() / self.logit_scale.exp().item()

                # Average similarity of negative pairs
                # Create mask for positives (only diagonal of current batch, queue is all negatives)
                logits_mask = torch.zeros_like(logits, dtype=torch.bool)
                logits_mask[:, :batch_size] = torch.eye(batch_size, device=logits.device, dtype=torch.bool)
                negative_sim = logits[~logits_mask].mean().item() / self.logit_scale.exp().item()

                metrics = {
                    'loss': loss.item(),
                    'loss_imu_to_text': loss_imu_to_text.item(),
                    'loss_text_to_imu': loss_text_to_imu.item(),
                    'acc_imu_to_text': imu_to_text_acc,
                    'acc_text_to_imu': text_to_imu_acc,
                    'positive_similarity': positive_sim,
                    'negative_similarity': negative_sim,
                    'similarity_gap': positive_sim - negative_sim
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


def compute_retrieval_metrics(
    imu_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    k_values: list[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute retrieval metrics (recall@k) for evaluation.

    Args:
        imu_embeddings: IMU embeddings (N, embedding_dim), L2-normalized
        text_embeddings: Text embeddings (N, embedding_dim), L2-normalized
        k_values: List of k values for recall@k

    Returns:
        Dictionary with retrieval metrics
    """
    # Convert to fp32 for mixed precision compatibility
    imu_embeddings = imu_embeddings.float()
    text_embeddings = text_embeddings.float()

    batch_size = imu_embeddings.shape[0]

    # Compute similarity matrix
    similarities = torch.matmul(imu_embeddings, text_embeddings.T)

    metrics = {}

    # IMU-to-text retrieval
    for k in k_values:
        if k <= batch_size:
            # Get top-k indices
            _, top_k_indices = torch.topk(similarities, k=k, dim=1)
            # Check if correct index is in top-k
            correct = torch.arange(batch_size, device=similarities.device).unsqueeze(1)
            recall_at_k = (top_k_indices == correct).any(dim=1).float().mean().item()
            metrics[f'recall@{k}_imu_to_text'] = recall_at_k

    # Text-to-IMU retrieval
    similarities_T = similarities.T
    for k in k_values:
        if k <= batch_size:
            _, top_k_indices = torch.topk(similarities_T, k=k, dim=1)
            correct = torch.arange(batch_size, device=similarities.device).unsqueeze(1)
            recall_at_k = (top_k_indices == correct).any(dim=1).float().mean().item()
            metrics[f'recall@{k}_text_to_imu'] = recall_at_k

    # Average recall across both directions
    for k in k_values:
        if k <= batch_size:
            imu_to_text = metrics[f'recall@{k}_imu_to_text']
            text_to_imu = metrics[f'recall@{k}_text_to_imu']
            metrics[f'recall@{k}_avg'] = (imu_to_text + text_to_imu) / 2.0

    return metrics
