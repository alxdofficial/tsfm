"""
Evaluation metrics for semantic alignment model.

Includes:
- Label groups for synonym handling (imported from shared location)
- Label-based semantic recall (retrieves from unique labels, not positions)
- Embedding space quality metrics
"""

import torch
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

def compute_semantic_recall(
    imu_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    query_labels: List[str],
    corpus_labels: List[str],
    k_values: List[int] = [1, 5, 10],
    use_groups: bool = True,
    use_simple_groups: bool = False
) -> Dict[str, float]:
    """
    Compute recall@k with semantic grouping.

    IMPORTANT: For correct evaluation, pass:
    - imu_embeddings: (N, D) - N sample embeddings
    - text_embeddings: (L, D) - L unique label embeddings (NOT N per-sample embeddings!)
    - query_labels: (N,) - label for each sample
    - corpus_labels: (L,) - the L unique labels

    This gives credit if the retrieved item has the same label (or synonym if use_groups=True).

    Args:
        imu_embeddings: Query embeddings (N, D), L2-normalized
        text_embeddings: Corpus embeddings (L, D), L2-normalized - should be UNIQUE labels
        query_labels: Labels for each query (N,)
        corpus_labels: Labels for each corpus item (L,) - should be UNIQUE labels
        k_values: List of k values for recall@k
        use_groups: If True, treat synonym groups as equivalent
        use_simple_groups: If True, use LABEL_GROUPS_SIMPLE (coarser grouping)

    Returns:
        Dict with recall metrics
    """
    imu_embeddings = imu_embeddings.float()
    text_embeddings = text_embeddings.float()

    N = len(imu_embeddings)
    M = len(text_embeddings)

    # Build label matching logic
    if use_groups:
        label_to_group = get_label_to_group_mapping(use_simple=use_simple_groups)
        query_groups = [label_to_group.get(l, l) for l in query_labels]
        corpus_groups = [label_to_group.get(l, l) for l in corpus_labels]
    else:
        query_groups = query_labels
        corpus_groups = corpus_labels

    # Compute similarity matrix
    similarities = torch.matmul(imu_embeddings, text_embeddings.T)  # (N, M)

    # Get top-k indices for each query
    max_k = max(k_values)
    _, top_k_indices = torch.topk(similarities, k=min(max_k, M), dim=1)  # (N, k)
    top_k_indices = top_k_indices.cpu().numpy()

    metrics = {}

    for k in k_values:
        if k > M:
            continue

        correct = 0
        for i in range(N):
            query_group = query_groups[i]
            retrieved_indices = top_k_indices[i, :k]

            # Check if any retrieved item has matching group
            for idx in retrieved_indices:
                if corpus_groups[idx] == query_group:
                    correct += 1
                    break

        metrics[f'semantic_recall@{k}'] = correct / N

    return metrics


# =============================================================================
# Embedding Space Quality Metrics
# =============================================================================

def compute_embedding_quality_metrics(
    imu_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    labels: List[str],
    use_groups: bool = True,
    use_simple_groups: bool = False
) -> Dict[str, float]:
    """
    Compute metrics measuring the quality of the learned embedding space.

    Args:
        imu_embeddings: IMU embeddings (N, D), L2-normalized
        text_embeddings: Text embeddings (N, D), L2-normalized (per-sample)
        labels: Labels for each sample (N,)
        use_groups: If True, use synonym groups for clustering
        use_simple_groups: If True, use LABEL_GROUPS_SIMPLE (coarser grouping)

    Returns:
        Dict with embedding quality metrics
    """
    imu_embeddings = imu_embeddings.float().cpu().numpy()
    text_embeddings = text_embeddings.float().cpu().numpy()

    # Get class labels (groups or exact)
    if use_groups:
        label_to_group = get_label_to_group_mapping(use_simple=use_simple_groups)
        class_labels = [label_to_group.get(l, l) for l in labels]
    else:
        class_labels = labels

    unique_classes = sorted(set(class_labels))

    metrics = {}

    # 1. IMU-Text Alignment: avg cosine similarity between paired IMU and text
    pairwise_sim = np.sum(imu_embeddings * text_embeddings, axis=1)
    metrics['imu_text_alignment'] = float(np.mean(pairwise_sim))
    metrics['imu_text_alignment_std'] = float(np.std(pairwise_sim))

    # 2. Per-class centroids
    imu_centroids = {}
    text_centroids = {}

    for cls in unique_classes:
        mask = np.array(class_labels) == cls
        if mask.sum() > 0:
            imu_centroids[cls] = imu_embeddings[mask].mean(axis=0)
            text_centroids[cls] = text_embeddings[mask].mean(axis=0)

    # 3. Cross-modal gap: distance between IMU and text centroids per class
    cross_modal_gaps = []
    for cls in unique_classes:
        if cls in imu_centroids and cls in text_centroids:
            # Cosine distance (1 - cosine similarity)
            sim = np.dot(imu_centroids[cls], text_centroids[cls])
            sim /= (np.linalg.norm(imu_centroids[cls]) * np.linalg.norm(text_centroids[cls]) + 1e-8)
            cross_modal_gaps.append(1 - sim)

    metrics['cross_modal_gap'] = float(np.mean(cross_modal_gaps))
    metrics['cross_modal_gap_std'] = float(np.std(cross_modal_gaps))

    # 4. Intra-class distance (tightness of clusters)
    intra_distances = []
    for cls in unique_classes:
        mask = np.array(class_labels) == cls
        if mask.sum() > 1:
            cls_embeddings = imu_embeddings[mask]
            # Pairwise cosine similarities within class
            sims = np.dot(cls_embeddings, cls_embeddings.T)
            # Get upper triangle (excluding diagonal)
            n = len(cls_embeddings)
            upper_tri = sims[np.triu_indices(n, k=1)]
            if len(upper_tri) > 0:
                intra_distances.append(1 - np.mean(upper_tri))  # Convert to distance

    metrics['intra_class_distance'] = float(np.mean(intra_distances)) if intra_distances else 0.0

    # 5. Inter-class distance (separation between clusters)
    inter_distances = []
    centroid_list = list(imu_centroids.values())
    for i in range(len(centroid_list)):
        for j in range(i + 1, len(centroid_list)):
            sim = np.dot(centroid_list[i], centroid_list[j])
            sim /= (np.linalg.norm(centroid_list[i]) * np.linalg.norm(centroid_list[j]) + 1e-8)
            inter_distances.append(1 - sim)

    metrics['inter_class_distance'] = float(np.mean(inter_distances)) if inter_distances else 0.0

    # 6. Class separability ratio (higher is better)
    if metrics['intra_class_distance'] > 0:
        metrics['class_separability'] = metrics['inter_class_distance'] / metrics['intra_class_distance']
    else:
        metrics['class_separability'] = 0.0

    return metrics


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
