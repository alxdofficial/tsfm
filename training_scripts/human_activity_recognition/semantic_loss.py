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

    def _get_logit_scale(self):
        """Get clamped logit scale."""
        return self.logit_scale.exp().clamp(1, 50)

    def _has_queue(self, imu_queue, text_queue):
        """Check if a non-empty queue is provided."""
        return imu_queue is not None and text_queue is not None and len(imu_queue) > 0

    def _forward_single_prototype(self, imu_embeddings, text_embeddings, imu_queue, text_queue):
        """Compute loss for single-prototype (2D) text embeddings.

        Returns: (loss, logits, logits_t2i, targets, sim_mean_for_metrics)
        """
        batch_size = imu_embeddings.shape[0]
        logit_scale = self._get_logit_scale()

        # Expand embeddings with queue if provided (for more negatives)
        if self._has_queue(imu_queue, text_queue):
            all_imu = torch.cat([imu_embeddings, imu_queue.detach()], dim=0)
            all_text = torch.cat([text_embeddings, text_queue.detach()], dim=0)
        else:
            all_imu = imu_embeddings
            all_text = text_embeddings

        logits = torch.matmul(imu_embeddings, all_text.T) * logit_scale
        logits_t2i = torch.matmul(text_embeddings, all_imu.T) * logit_scale

        # Compute targets
        sim_mean_for_metrics = None
        if self.use_soft_targets:
            text_similarity_full = torch.matmul(text_embeddings, all_text.T)
            sim_mean = text_similarity_full.mean()
            sim_std = text_similarity_full.std().clamp(min=0.1)
            sim_mean_for_metrics = sim_mean.item()
            text_similarity_full = (text_similarity_full - sim_mean) / sim_std / self.soft_target_temperature
            soft_targets_full = F.softmax(text_similarity_full, dim=1)

            queue_size = all_text.shape[0] - batch_size
            hard_targets_full = torch.cat([
                torch.eye(batch_size, device=imu_embeddings.device),
                torch.zeros(batch_size, queue_size, device=imu_embeddings.device),
            ], dim=1)

            targets = (1 - self.soft_target_weight) * hard_targets_full + \
                     self.soft_target_weight * soft_targets_full

            log_probs = F.log_softmax(logits, dim=1)
            loss_imu_to_text = -(targets * log_probs).sum(dim=1).mean()
            log_probs_t2i = F.log_softmax(logits_t2i, dim=1)
            loss_text_to_imu = -(targets * log_probs_t2i).sum(dim=1).mean()
        else:
            queue_size = all_text.shape[0] - batch_size
            hard_targets_batch = torch.eye(batch_size, device=imu_embeddings.device)
            if queue_size > 0:
                targets = torch.cat([
                    hard_targets_batch,
                    torch.zeros(batch_size, queue_size, device=imu_embeddings.device),
                ], dim=1)
            else:
                targets = hard_targets_batch

            labels = torch.arange(batch_size, device=imu_embeddings.device)
            loss_imu_to_text = F.cross_entropy(logits, labels)
            loss_text_to_imu = F.cross_entropy(logits_t2i, labels)

        loss = (loss_imu_to_text + loss_text_to_imu) / 2.0
        return loss, logits, logits_t2i, targets, sim_mean_for_metrics

    def _forward_multi_prototype(self, imu_embeddings, text_embeddings, imu_queue, text_queue):
        """Compute loss for multi-prototype (3D) text embeddings.

        Returns: (loss, logits, logits_t2i, targets, best_text, sim_mean_for_metrics)
        """
        batch_size = imu_embeddings.shape[0]
        K = text_embeddings.shape[1]
        D = text_embeddings.shape[2]
        logit_scale = self._get_logit_scale()

        # Select best prototype per sample
        pos_sims = torch.einsum('bd,bkd->bk', imu_embeddings, text_embeddings)
        best_proto_idx = pos_sims.argmax(dim=1)
        best_text = text_embeddings[torch.arange(batch_size), best_proto_idx]

        # Flatten all prototypes for negatives
        all_text_flat = text_embeddings.reshape(batch_size * K, D)

        if self._has_queue(imu_queue, text_queue):
            all_imu = torch.cat([imu_embeddings, imu_queue.detach()], dim=0)
            all_text_for_logits = torch.cat([all_text_flat, text_queue.detach()], dim=0)
        else:
            all_imu = imu_embeddings
            all_text_for_logits = all_text_flat

        logits = torch.matmul(imu_embeddings, all_text_for_logits.T) * logit_scale
        total_keys = all_text_for_logits.shape[0]

        # Build prototype-weighted targets
        proto_weights = F.softmax(pos_sims / 0.1, dim=1)
        targets = torch.zeros(batch_size, total_keys, device=imu_embeddings.device)
        for i in range(batch_size):
            for k in range(K):
                targets[i, i * K + k] = proto_weights[i, k]

        # Soft targets from label similarity (optional)
        sim_mean_for_metrics = None
        if self.use_soft_targets:
            pairwise_proto_sim = torch.einsum('bkd,cld->bckl', text_embeddings, text_embeddings)
            label_sim = pairwise_proto_sim.max(dim=-1).values.max(dim=-1).values

            sim_mean = label_sim.mean()
            sim_std = label_sim.std().clamp(min=0.1)
            sim_mean_for_metrics = sim_mean.item()
            label_sim_norm = (label_sim - sim_mean) / sim_std / self.soft_target_temperature
            soft_label_weights = F.softmax(label_sim_norm, dim=1)

            soft_targets = torch.zeros(batch_size, total_keys, device=imu_embeddings.device)
            for i in range(batch_size):
                for j in range(batch_size):
                    for k in range(K):
                        soft_targets[i, j * K + k] = soft_label_weights[i, j] / K

            targets = (1 - self.soft_target_weight) * targets + self.soft_target_weight * soft_targets

        # IMU→Text loss
        log_probs = F.log_softmax(logits, dim=1)
        loss_imu_to_text = -(targets * log_probs).sum(dim=1).mean()

        # Text→IMU loss (best prototype → matched IMU)
        logits_t2i = torch.matmul(best_text, all_imu.T) * logit_scale
        if self._has_queue(imu_queue, text_queue):
            queue_size = all_imu.shape[0] - batch_size
            t2i_targets = torch.cat([
                torch.eye(batch_size, device=imu_embeddings.device),
                torch.zeros(batch_size, queue_size, device=imu_embeddings.device),
            ], dim=1)
        else:
            t2i_targets = torch.eye(batch_size, device=imu_embeddings.device)

        log_probs_t2i = F.log_softmax(logits_t2i, dim=1)
        loss_text_to_imu = -(t2i_targets * log_probs_t2i).sum(dim=1).mean()

        loss = (loss_imu_to_text + loss_text_to_imu) / 2.0
        return loss, logits, logits_t2i, targets, best_text, sim_mean_for_metrics

    def _compute_metrics(self, loss, logits, logits_t2i, targets,
                         imu_embeddings, text_for_metrics, text_embeddings,
                         label_texts, sim_mean_for_metrics, is_multi_prototype):
        """Compute training metrics dict."""
        batch_size = imu_embeddings.shape[0]
        loss_i2t = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        loss_t2i_val = loss.item() * 2 - loss_i2t.item()  # Derived from loss = (i2t + t2i) / 2

        raw_sim = torch.matmul(imu_embeddings, text_for_metrics.T)
        positive_sim = torch.diagonal(raw_sim).mean().item()

        # Label-aware negative similarity
        if label_texts is not None and len(label_texts) == batch_size:
            from val_scripts.human_activity_recognition.evaluation_metrics import get_label_to_group_mapping
            label_to_group = get_label_to_group_mapping()
            label_groups = [label_to_group.get(lbl, lbl) for lbl in label_texts]
            same_label_mask = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=raw_sim.device)
            for i in range(batch_size):
                for j in range(batch_size):
                    same_label_mask[i, j] = (label_groups[i] == label_groups[j])
        else:
            same_label_mask = torch.eye(batch_size, dtype=torch.bool, device=raw_sim.device)

        diff_label_mask = ~same_label_mask
        true_neg_sim = raw_sim[diff_label_mask].mean().item() if diff_label_mask.any() else 0.0

        metrics = {
            'loss': loss.item(),
            'loss_imu_to_text': loss_i2t.item(),
            'loss_text_to_imu': loss.item() * 2 - loss_i2t.item(),
            'positive_similarity': positive_sim,
            'negative_similarity': true_neg_sim,
            'similarity_gap': positive_sim - true_neg_sim,
            'logit_scale': self.logit_scale.exp().item(),
            'logits_mean': logits.mean().item(),
            'logits_std': logits.std().item(),
            'logits_max': logits.max().item(),
            'logits_min': logits.min().item(),
        }

        if self.use_soft_targets:
            text_similarity = torch.matmul(text_for_metrics, text_for_metrics.T)
            batch_mask = torch.eye(batch_size, device=imu_embeddings.device, dtype=torch.bool)
            metrics['label_semantic_similarity'] = text_similarity[~batch_mask].mean().item()
            if sim_mean_for_metrics is not None:
                metrics['soft_target_sim_mean'] = sim_mean_for_metrics

            soft_target_entropy = -(targets * torch.log(targets + 1e-10)).sum(dim=1).mean().item()
            max_entropy = np.log(targets.shape[1])
            metrics['soft_target_entropy'] = soft_target_entropy
            metrics['soft_target_entropy_ratio'] = soft_target_entropy / max_entropy

            if is_multi_prototype:
                K = text_embeddings.shape[1]
                true_pos_mass = sum(
                    targets[i, i*K:(i+1)*K].sum().item() for i in range(batch_size)
                )
                metrics['true_positive_target_prob'] = true_pos_mass / batch_size
            else:
                metrics['true_positive_target_prob'] = torch.diagonal(targets[:, :batch_size]).mean().item()

        return metrics

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
            text_embeddings: Text embeddings - (batch_size, D) for single prototype,
                             (batch_size, K, D) for multi-prototype
            label_texts: Optional list of label text strings
            return_metrics: Whether to return additional metrics
            imu_queue: Optional queue of past IMU embeddings (queue_size, D)
            text_queue: Optional queue of past text embeddings (queue_size, D)

        Returns:
            loss: Scalar loss value
            metrics: Optional dict with additional metrics
        """
        is_multi_prototype = text_embeddings.dim() == 3

        if is_multi_prototype:
            loss, logits, logits_t2i, targets, best_text, sim_mean = \
                self._forward_multi_prototype(imu_embeddings, text_embeddings, imu_queue, text_queue)
            text_for_metrics = best_text
        else:
            loss, logits, logits_t2i, targets, sim_mean = \
                self._forward_single_prototype(imu_embeddings, text_embeddings, imu_queue, text_queue)
            text_for_metrics = text_embeddings

        # NaN debugging
        if DEBUG_NAN and torch.isnan(loss):
            with torch.no_grad():
                context = {
                    'mode': 'multi_prototype' if is_multi_prototype else 'single_prototype',
                    'loss': loss.item(),
                    'imu_embeddings': _check_tensor_health(imu_embeddings, 'imu_embeddings'),
                    'logits': _check_tensor_health(logits, 'logits'),
                    'targets': _check_tensor_health(targets, 'targets'),
                }
                _debug_nan_loss(context)

        # Compute metrics
        metrics = None
        if return_metrics:
            with torch.no_grad():
                metrics = self._compute_metrics(
                    loss, logits, logits_t2i, targets,
                    imu_embeddings, text_for_metrics, text_embeddings,
                    label_texts, sim_mean, is_multi_prototype,
                )

        return loss, metrics


class SigLIPLoss(nn.Module):
    """
    SigLIP (Sigmoid Loss for Language-Image Pre-training) contrastive loss.

    Uses independent sigmoid per pair instead of softmax over all negatives.
    Works well even at small batch sizes because each pair provides an
    independent gradient signal, unlike InfoNCE which needs many negatives
    for stable softmax gradients.

    Reference: Zhai et al., "Sigmoid Loss for Language Image Pre-Training" (2023)
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        # Learnable temperature and bias (SigLIP-style)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/temperature))
        self.logit_bias = nn.Parameter(torch.zeros([]))

    def _get_logit_scale(self):
        """Get clamped logit scale."""
        return self.logit_scale.exp().clamp(1, 100)

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
        Compute SigLIP loss.

        Works with both single-prototype (2D) and multi-prototype (3D) text embeddings.
        Queue embeddings are ignored (SigLIP doesn't need them — each pair is independent).
        """
        batch_size = imu_embeddings.shape[0]
        logit_scale = self._get_logit_scale()

        # Handle multi-prototype: select best prototype per sample
        if text_embeddings.dim() == 3:
            pos_sims = torch.einsum('bd,bkd->bk', imu_embeddings, text_embeddings)
            best_proto_idx = pos_sims.argmax(dim=1)
            text_for_loss = text_embeddings[torch.arange(batch_size), best_proto_idx]
        else:
            text_for_loss = text_embeddings

        # Pairwise logits: (B, B) — all IMU vs all text
        logits = torch.matmul(imu_embeddings, text_for_loss.T) * logit_scale + self.logit_bias

        # Labels: +1 for positive pairs (diagonal), -1 for negative pairs
        labels = 2 * torch.eye(batch_size, device=imu_embeddings.device) - 1

        # SigLIP loss: -log_sigmoid(labels * logits)
        # = log(1 + exp(-labels * logits))
        loss = -F.logsigmoid(labels * logits).mean()

        metrics = None
        if return_metrics:
            with torch.no_grad():
                pos_logits = torch.diagonal(logits)
                neg_mask = ~torch.eye(batch_size, dtype=torch.bool, device=imu_embeddings.device)
                neg_logits = logits[neg_mask]

                raw_sim = torch.matmul(imu_embeddings, text_for_loss.T)
                positive_sim = torch.diagonal(raw_sim).mean().item()

                # Label-aware negative similarity
                if label_texts is not None and len(label_texts) == batch_size:
                    from val_scripts.human_activity_recognition.evaluation_metrics import get_label_to_group_mapping
                    label_to_group = get_label_to_group_mapping()
                    label_groups = [label_to_group.get(lbl, lbl) for lbl in label_texts]
                    same_label_mask = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=raw_sim.device)
                    for i in range(batch_size):
                        for j in range(batch_size):
                            same_label_mask[i, j] = (label_groups[i] == label_groups[j])
                else:
                    same_label_mask = torch.eye(batch_size, dtype=torch.bool, device=raw_sim.device)

                diff_label_mask = ~same_label_mask
                true_neg_sim = raw_sim[diff_label_mask].mean().item() if diff_label_mask.any() else 0.0

                metrics = {
                    'loss': loss.item(),
                    'loss_imu_to_text': loss.item(),  # SigLIP is symmetric
                    'loss_text_to_imu': loss.item(),
                    'positive_similarity': positive_sim,
                    'negative_similarity': true_neg_sim,
                    'similarity_gap': positive_sim - true_neg_sim,
                    'logit_scale': logit_scale.item(),
                    'logits_mean': logits.mean().item(),
                    'logits_std': logits.std().item(),
                    'logits_max': logits.max().item(),
                    'logits_min': logits.min().item(),
                }

        return loss, metrics


class SemanticAlignmentLoss(nn.Module):
    """
    Combined loss for semantic alignment training.

    Supports multiple loss components that can be weighted.
    Supports both InfoNCE and SigLIP loss functions.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        use_soft_targets: bool = True,
        soft_target_temperature: float = 0.5,
        soft_target_weight: float = 0.5,
        loss_type: str = "infonce"
    ):
        """
        Args:
            temperature: Temperature for contrastive loss
            use_soft_targets: Whether to use soft targets based on label similarity (InfoNCE only)
            soft_target_temperature: Temperature for computing soft target distribution
            soft_target_weight: Weight for soft targets (0=hard, 1=pure soft, 0.5=balanced)
            loss_type: "infonce" or "siglip"
        """
        super().__init__()
        self.loss_type = loss_type
        if loss_type == "siglip":
            self.loss_fn = SigLIPLoss(temperature=temperature)
        else:
            self.loss_fn = InfoNCELoss(
                temperature=temperature,
                use_soft_targets=use_soft_targets,
                soft_target_temperature=soft_target_temperature,
                soft_target_weight=soft_target_weight
            )

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with backward compatibility for old 'infonce.*' keys."""
        # Remap old 'infonce.*' keys to new 'loss_fn.*' keys
        remapped = {}
        for key, value in state_dict.items():
            if key.startswith('infonce.'):
                new_key = 'loss_fn.' + key[len('infonce.'):]
                remapped[new_key] = value
            else:
                remapped[key] = value
        return super().load_state_dict(remapped, strict=strict)

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
        return self.loss_fn(imu_embeddings, text_embeddings, label_texts, return_metrics,
                           imu_queue, text_queue)

    def forward_cached(
        self,
        all_imu_embeddings: torch.Tensor,
        all_text_embeddings: torch.Tensor,
        all_label_texts: Optional[list] = None,
        return_metrics: bool = True,
    ) -> tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        Compute loss over pre-gathered (cached) embeddings from all micro-batches.

        Used by GradCache: all micro-batch embeddings are concatenated into one large
        batch and loss is computed over the full set, giving N-1 fresh in-batch negatives
        instead of micro_batch_size-1.

        No memory bank queue is used — with GradCache providing hundreds of fresh
        negatives, stale queue embeddings are unnecessary.

        Args:
            all_imu_embeddings: Concatenated IMU embeddings from all micro-batches (N, D)
            all_text_embeddings: Concatenated text embeddings (N, D) or (N, K, D)
            all_label_texts: Optional concatenated label texts
            return_metrics: Whether to return metrics

        Returns:
            loss: Scalar loss value
            metrics: Optional dict with metrics
        """
        # Compute loss without queue (all negatives are fresh in-batch)
        return self.loss_fn(
            all_imu_embeddings, all_text_embeddings,
            all_label_texts, return_metrics,
            imu_queue=None, text_queue=None
        )


