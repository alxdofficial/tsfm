"""
Evaluation metrics for semantic alignment model.

Includes:
- Label groups for synonym handling (imported from shared location)
- Label-based semantic recall (retrieves from unique labels, not positions)
- Embedding space quality metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
from sklearn.metrics import f1_score

# Import label groups from shared location (used by both training and evaluation)
from datasets.imu_pretraining_dataset.label_groups import (
    LABEL_GROUPS,
    LABEL_GROUPS_SIMPLE,
    ACTIVE_LABEL_GROUPS,
    get_label_to_group_mapping,
    get_group_for_label,
    get_group_members,
)


# =============================================================================
# Multi-prototype similarity helper
# =============================================================================

def compute_similarity(imu_embeddings: torch.Tensor, label_embeddings: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity, handling multi-prototype labels.

    Args:
        imu_embeddings: (N, D) L2-normalized
        label_embeddings: (L, D) or (L, K, D) L2-normalized

    Returns:
        similarity: (N, L)
    """
    if label_embeddings.dim() == 3:
        # Multi-prototype: max similarity over K prototypes
        return torch.einsum('nd,lkd->nlk', imu_embeddings, label_embeddings).max(dim=-1).values
    else:
        return imu_embeddings @ label_embeddings.T


# =============================================================================
# Semantic Recall Metrics
# =============================================================================



# =============================================================================
# Embedding Space Quality Metrics
# =============================================================================



# =============================================================================
# Classification Accuracy Metrics
# =============================================================================

def compute_group_accuracy(
    imu_embeddings: torch.Tensor,
    label_bank,
    query_labels: List[str],
    return_mrr: bool = True,
    use_simple_groups: bool = False
) -> Dict[str, float]:
    """
    Compute group-aware classification accuracy.

    For each IMU embedding, finds the most similar label embedding and checks
    if the predicted label's group matches the ground truth label's group.
    Synonyms (e.g., "walking" and "nordic_walking") are treated as equivalent.

    Args:
        imu_embeddings: Query embeddings (N, D), L2-normalized
        label_bank: LabelBank or LearnableLabelBank to encode labels
        query_labels: Ground truth labels for each sample (N,)
        return_mrr: If True, also compute Mean Reciprocal Rank
        use_simple_groups: If True, use LABEL_GROUPS_SIMPLE (coarser grouping)

    Returns:
        Dict with 'accuracy' and optionally 'mrr'
    """
    imu_embeddings = imu_embeddings.float()
    device = imu_embeddings.device

    # Get unique labels from the query set
    unique_labels = sorted(set(query_labels))

    # Encode unique labels
    label_embeddings = label_bank.encode(unique_labels, normalize=True)  # (L, D) or (L, K, D)
    label_embeddings = label_embeddings.to(device)

    # Build label-to-group mapping
    label_to_group = get_label_to_group_mapping(use_simple=use_simple_groups)

    # Get group for each unique label
    unique_groups = [label_to_group.get(lbl, lbl) for lbl in unique_labels]

    # Compute similarity matrix: (N, L)
    similarities = compute_similarity(imu_embeddings, label_embeddings)

    # Get predictions (top-1)
    _, top1_indices = similarities.max(dim=1)  # (N,)
    top1_indices = top1_indices.cpu().numpy()

    # Compute accuracy
    correct = 0
    for i, gt_label in enumerate(query_labels):
        pred_idx = top1_indices[i]
        pred_group = unique_groups[pred_idx]
        gt_group = label_to_group.get(gt_label, gt_label)

        if pred_group == gt_group:
            correct += 1

    accuracy = correct / len(query_labels)
    metrics = {'accuracy': accuracy}

    # Compute F1 scores (for comparison with baselines that report F1)
    pred_groups_list = []
    gt_groups_list = []
    for i, gt_label in enumerate(query_labels):
        pred_idx = top1_indices[i]
        pred_groups_list.append(unique_groups[pred_idx])
        gt_groups_list.append(label_to_group.get(gt_label, gt_label))

    metrics['f1_macro'] = f1_score(gt_groups_list, pred_groups_list, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(gt_groups_list, pred_groups_list, average='weighted', zero_division=0)

    # Compute MRR if requested
    if return_mrr:
        # Sort by similarity (descending)
        _, sorted_indices = similarities.sort(dim=1, descending=True)
        sorted_indices = sorted_indices.cpu().numpy()

        reciprocal_ranks = []
        for i, gt_label in enumerate(query_labels):
            gt_group = label_to_group.get(gt_label, gt_label)

            # Find rank of first correct prediction
            for rank, idx in enumerate(sorted_indices[i], start=1):
                pred_group = unique_groups[idx]
                if pred_group == gt_group:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)

        metrics['mrr'] = sum(reciprocal_ranks) / len(reciprocal_ranks)

    return metrics


def compute_group_accuracy_majority_vote(
    patch_embeddings: torch.Tensor,
    patch_masks: torch.Tensor,
    label_bank,
    query_labels: List[str],
    return_mrr: bool = True,
    use_simple_groups: bool = False,
) -> Dict[str, float]:
    """
    Compute group-aware classification accuracy using per-patch majority voting.

    Each patch independently votes for a label. The session prediction is the
    label (group) with the most votes across valid patches.

    Args:
        patch_embeddings: Per-patch embeddings (N, P, D), L2-normalized
        patch_masks: Valid patch masks (N, P), True=valid
        label_bank: LabelBank or LearnableLabelBank to encode labels
        query_labels: Ground truth labels for each session (N,)
        return_mrr: If True, also compute Mean Reciprocal Rank
        use_simple_groups: If True, use LABEL_GROUPS_SIMPLE (coarser grouping)

    Returns:
        Dict with 'accuracy' and optionally 'mrr', 'f1_macro', 'f1_weighted'
    """
    patch_embeddings = patch_embeddings.float()
    device = patch_embeddings.device
    N, P, D = patch_embeddings.shape

    # Get unique labels and encode them
    unique_labels = sorted(set(query_labels))
    label_embeddings = label_bank.encode(unique_labels, normalize=True)  # (L, D) or (L, K, D)
    label_embeddings = label_embeddings.to(device)
    L = len(unique_labels)

    # Build label-to-group mapping
    label_to_group = get_label_to_group_mapping(use_simple=use_simple_groups)
    unique_groups = [label_to_group.get(lbl, lbl) for lbl in unique_labels]

    correct = 0
    pred_groups_list = []
    gt_groups_list = []
    reciprocal_ranks = []

    for i in range(N):
        valid_mask = patch_masks[i].bool()
        patches_i = patch_embeddings[i, valid_mask]  # (P_valid, D)

        if patches_i.shape[0] == 0:
            # No valid patches — skip (shouldn't happen with data validation)
            pred_groups_list.append("__none__")
            gt_groups_list.append(label_to_group.get(query_labels[i], query_labels[i]))
            if return_mrr:
                reciprocal_ranks.append(0.0)
            continue

        # Compute similarity to all labels: (P_valid, L)
        sims = compute_similarity(patches_i, label_embeddings)

        # Per-patch vote: each patch picks its most similar label
        votes = sims.argmax(dim=1)  # (P_valid,)

        # Majority vote: count votes per label, pick the label with most votes
        vote_counts = torch.zeros(L, device=device)
        for v in votes:
            vote_counts[v] += 1

        pred_idx = vote_counts.argmax().item()
        pred_group = unique_groups[pred_idx]
        gt_group = label_to_group.get(query_labels[i], query_labels[i])

        pred_groups_list.append(pred_group)
        gt_groups_list.append(gt_group)

        if pred_group == gt_group:
            correct += 1

        # MRR: rank labels by vote count, find rank of first correct group
        if return_mrr:
            _, sorted_indices = vote_counts.sort(descending=True)
            for rank, idx in enumerate(sorted_indices, start=1):
                if unique_groups[idx.item()] == gt_group:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)

    metrics = {'accuracy': correct / N if N > 0 else 0.0}

    # F1 scores
    metrics['f1_macro'] = f1_score(gt_groups_list, pred_groups_list, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(gt_groups_list, pred_groups_list, average='weighted', zero_division=0)

    if return_mrr:
        metrics['mrr'] = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    return metrics
