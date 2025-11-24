"""
Semantic alignment loss functions.

Implements InfoNCE contrastive loss for aligning IMU embeddings with text embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss for semantic alignment.

    Aligns IMU embeddings with text embeddings in a shared semantic space.
    Supports:
    - Hard targets (one-hot)
    - Pairwise soft targets (semantic similarity)
    - Prototype soft targets (cluster-based)
    - Hybrid (blend pairwise + prototype)
    """

    def __init__(
        self,
        temperature: float = 0.1,
        use_soft_targets: bool = True,
        soft_target_temperature: float = 0.5,
        soft_target_weight: float = 0.5,
        use_prototypes: bool = False,
        prototype_weight: float = 0.5,
        prototype_manager=None
    ):
        """
        Args:
            temperature: Temperature parameter for scaling similarities
            use_soft_targets: Whether to use soft targets based on label similarity
            soft_target_temperature: Temperature for computing soft target distribution
            soft_target_weight: Weight for soft targets (1-weight goes to hard targets)
            use_prototypes: Whether to use prototype-based soft targets
            prototype_weight: When both soft targets and prototypes enabled:
                             0.0 = pure pairwise, 1.0 = pure prototypes, 0.5 = balanced
            prototype_manager: PrototypeManager instance (required if use_prototypes=True)
        """
        super().__init__()
        self.temperature = temperature
        self.use_soft_targets = use_soft_targets
        self.soft_target_temperature = soft_target_temperature
        self.soft_target_weight = soft_target_weight
        self.use_prototypes = use_prototypes
        self.prototype_weight = prototype_weight
        self.prototype_manager = prototype_manager

        if self.use_prototypes and self.prototype_manager is None:
            raise ValueError("prototype_manager must be provided when use_prototypes=True")

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
            label_texts: Optional list of label text strings (needed for prototypes)
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
        logits = torch.matmul(imu_embeddings, all_text.T) / self.temperature  # (batch, batch+queue)

        # Start with hard targets (one-hot)
        hard_targets = torch.eye(batch_size, device=imu_embeddings.device)
        targets = hard_targets

        # Pairwise soft targets
        soft_targets_pairwise = None
        if self.use_soft_targets:
            # Compute semantic similarity between text labels
            # This captures relationships like "walking" is similar to "running"
            text_similarity = torch.matmul(text_embeddings, text_embeddings.T)
            text_similarity = text_similarity / self.soft_target_temperature

            # Convert to probability distribution (softmax over each row)
            soft_targets_pairwise = F.softmax(text_similarity, dim=1)

        # Prototype soft targets
        soft_targets_prototype = None
        if self.use_prototypes and label_texts is not None:
            # Get prototype-based soft targets
            soft_targets_prototype = self.prototype_manager.get_prototype_targets(
                label_texts,
                text_embeddings,
                temperature=self.soft_target_temperature
            )

        # Blend targets based on configuration
        if self.use_soft_targets and self.use_prototypes:
            # Hybrid: blend pairwise and prototype soft targets
            blended_soft = (1 - self.prototype_weight) * soft_targets_pairwise + \
                          self.prototype_weight * soft_targets_prototype
            targets = (1 - self.soft_target_weight) * hard_targets + \
                     self.soft_target_weight * blended_soft
        elif self.use_soft_targets:
            # Pairwise only
            targets = (1 - self.soft_target_weight) * hard_targets + \
                     self.soft_target_weight * soft_targets_pairwise
        elif self.use_prototypes:
            # Prototype only
            targets = (1 - self.soft_target_weight) * hard_targets + \
                     self.soft_target_weight * soft_targets_prototype

        # Expand targets to include queue dimension (if queue is used)
        # Queue embeddings are all negatives, so pad with zeros
        if imu_queue is not None and text_queue is not None and len(imu_queue) > 0:
            queue_size = imu_queue.shape[0]
            targets = torch.cat([targets, torch.zeros(batch_size, queue_size, device=targets.device)], dim=1)

        # Compute loss
        # Text→IMU direction (text queries vs IMU keys)
        logits_t2i = torch.matmul(text_embeddings, all_imu.T) / self.temperature  # (batch, batch+queue)

        if self.use_soft_targets or self.use_prototypes:
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

                # Average similarity of positive pairs
                positive_sim = torch.diagonal(logits).mean().item() * self.temperature

                # Average similarity of negative pairs
                # Create mask for positives (only diagonal of current batch, queue is all negatives)
                logits_mask = torch.zeros_like(logits, dtype=torch.bool)
                logits_mask[:, :batch_size] = torch.eye(batch_size, device=logits.device, dtype=torch.bool)
                negative_sim = logits[~logits_mask].mean().item() * self.temperature

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
        soft_target_weight: float = 0.5,
        use_prototypes: bool = False,
        prototype_weight: float = 0.5,
        prototype_manager=None
    ):
        """
        Args:
            temperature: Temperature for InfoNCE
            use_soft_targets: Whether to use soft targets based on label similarity
            soft_target_temperature: Temperature for computing soft target distribution
            soft_target_weight: Weight for soft targets (0=hard, 1=pure soft, 0.5=balanced)
            use_prototypes: Whether to use prototype-based soft targets
            prototype_weight: Balance between pairwise and prototype (0=pairwise, 1=prototype)
            prototype_manager: PrototypeManager instance
        """
        super().__init__()
        self.infonce = InfoNCELoss(
            temperature=temperature,
            use_soft_targets=use_soft_targets,
            soft_target_temperature=soft_target_temperature,
            soft_target_weight=soft_target_weight,
            use_prototypes=use_prototypes,
            prototype_weight=prototype_weight,
            prototype_manager=prototype_manager
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
