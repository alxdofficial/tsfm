"""
Comprehensive evaluation metrics for semantic alignment model.

Includes:
- Group-based semantic recall (credit for synonyms)
- Embedding space quality metrics
- Per-class breakdown
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# =============================================================================
# Label Groups - Synonyms that should be treated as equivalent
# =============================================================================

LABEL_GROUPS = {
    # Walking variants
    'walking': ['walking', 'nordic_walking'],

    # Stairs - ascending
    'ascending_stairs': ['ascending_stairs', 'climbing_stairs', 'going_up_stairs', 'walking_upstairs', 'stairs'],

    # Stairs - descending
    'descending_stairs': ['descending_stairs', 'going_down_stairs', 'walking_downstairs', 'stairs'],

    # Running/jogging
    'running': ['running', 'jogging'],

    # Lying/laying
    'lying': ['lying', 'laying'],

    # Sitting variants
    'sitting': ['sitting', 'sitting_down'],

    # Standing variants
    'standing': ['standing', 'standing_up_from_laying', 'standing_up_from_sitting'],

    # Falling variants (all falls are similar motion patterns)
    'falling': ['falling_backward', 'falling_backward_sitting', 'falling_forward',
                'falling_hitting_obstacle', 'falling_left', 'falling_right',
                'falling_with_protection', 'syncope'],

    # Jumping variants
    'jumping': ['jumping', 'jump_front_back', 'rope_jumping'],

    # Eating variants
    'eating': ['eating_chips', 'eating_pasta', 'eating_sandwich', 'eating_soup'],
}


def get_label_to_group_mapping() -> Dict[str, str]:
    """
    Create reverse mapping from individual labels to their group name.
    Labels not in any group map to themselves.
    """
    label_to_group = {}

    # Map grouped labels to their group name
    for group_name, labels in LABEL_GROUPS.items():
        for label in labels:
            # Note: some labels like 'stairs' may belong to multiple groups
            # We take the first assignment (ascending_stairs comes before descending_stairs)
            if label not in label_to_group:
                label_to_group[label] = group_name

    return label_to_group


def get_group_members(label: str, label_to_group: Dict[str, str]) -> List[str]:
    """
    Get all labels that are synonyms of the given label.
    Returns list including the label itself.
    """
    group = label_to_group.get(label, label)

    if group in LABEL_GROUPS:
        return LABEL_GROUPS[group]
    else:
        return [label]  # Ungrouped label - only matches itself


# =============================================================================
# Semantic Recall Metrics
# =============================================================================

def compute_semantic_recall(
    imu_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    query_labels: List[str],
    corpus_labels: List[str],
    k_values: List[int] = [1, 5, 10],
    use_groups: bool = True
) -> Dict[str, float]:
    """
    Compute recall@k with semantic grouping.

    Unlike standard recall which requires exact index match, this gives credit
    if the retrieved item has the same label (or a synonym label if use_groups=True).

    Args:
        imu_embeddings: Query embeddings (N, D), L2-normalized
        text_embeddings: Corpus embeddings (M, D), L2-normalized
        query_labels: Labels for each query (N,)
        corpus_labels: Labels for each corpus item (M,)
        k_values: List of k values for recall@k
        use_groups: If True, treat synonym groups as equivalent

    Returns:
        Dict with recall metrics
    """
    imu_embeddings = imu_embeddings.float()
    text_embeddings = text_embeddings.float()

    N = len(imu_embeddings)
    M = len(text_embeddings)

    # Build label matching logic
    if use_groups:
        label_to_group = get_label_to_group_mapping()
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


def compute_exact_label_recall(
    imu_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    query_labels: List[str],
    corpus_labels: List[str],
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute recall@k where credit is given if retrieved item has exact same label.

    This is stricter than group-based recall but more lenient than index-based recall.
    """
    return compute_semantic_recall(
        imu_embeddings, text_embeddings,
        query_labels, corpus_labels,
        k_values, use_groups=False
    )


# =============================================================================
# Embedding Space Quality Metrics
# =============================================================================

def compute_embedding_quality_metrics(
    imu_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    labels: List[str],
    use_groups: bool = True
) -> Dict[str, float]:
    """
    Compute metrics measuring the quality of the learned embedding space.

    Args:
        imu_embeddings: IMU embeddings (N, D), L2-normalized
        text_embeddings: Text embeddings (N, D), L2-normalized
        labels: Labels for each sample (N,)
        use_groups: If True, use synonym groups for clustering

    Returns:
        Dict with embedding quality metrics
    """
    imu_embeddings = imu_embeddings.float().cpu().numpy()
    text_embeddings = text_embeddings.float().cpu().numpy()

    N = len(labels)

    # Get class labels (groups or exact)
    if use_groups:
        label_to_group = get_label_to_group_mapping()
        class_labels = [label_to_group.get(l, l) for l in labels]
    else:
        class_labels = labels

    unique_classes = sorted(set(class_labels))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    class_indices = np.array([class_to_idx[c] for c in class_labels])

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
# Per-Class Breakdown
# =============================================================================

def compute_per_class_recall(
    imu_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    query_labels: List[str],
    corpus_labels: List[str],
    k: int = 1,
    use_groups: bool = True
) -> Dict[str, float]:
    """
    Compute recall@k broken down by class.

    Returns dict mapping class name to its recall@k.
    """
    imu_embeddings = imu_embeddings.float()
    text_embeddings = text_embeddings.float()

    N = len(imu_embeddings)
    M = len(text_embeddings)

    if use_groups:
        label_to_group = get_label_to_group_mapping()
        query_groups = [label_to_group.get(l, l) for l in query_labels]
        corpus_groups = [label_to_group.get(l, l) for l in corpus_labels]
    else:
        query_groups = query_labels
        corpus_groups = corpus_labels

    # Compute similarity and get top-k
    similarities = torch.matmul(imu_embeddings, text_embeddings.T)
    _, top_k_indices = torch.topk(similarities, k=min(k, M), dim=1)
    top_k_indices = top_k_indices.cpu().numpy()

    # Count correct per class
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for i in range(N):
        query_group = query_groups[i]
        class_total[query_group] += 1

        retrieved_indices = top_k_indices[i, :k]
        for idx in retrieved_indices:
            if corpus_groups[idx] == query_group:
                class_correct[query_group] += 1
                break

    # Compute per-class recall
    per_class_recall = {}
    for cls in class_total:
        per_class_recall[cls] = class_correct[cls] / class_total[cls]

    return per_class_recall


def compute_confusion_matrix(
    imu_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    query_labels: List[str],
    corpus_labels: List[str],
    use_groups: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute confusion matrix for top-1 retrieval.

    Returns:
        confusion_matrix: (num_classes, num_classes) array
        class_names: List of class names in order
    """
    imu_embeddings = imu_embeddings.float()
    text_embeddings = text_embeddings.float()

    N = len(imu_embeddings)

    if use_groups:
        label_to_group = get_label_to_group_mapping()
        query_groups = [label_to_group.get(l, l) for l in query_labels]
        corpus_groups = [label_to_group.get(l, l) for l in corpus_labels]
    else:
        query_groups = query_labels
        corpus_groups = corpus_labels

    unique_classes = sorted(set(query_groups))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    num_classes = len(unique_classes)

    # Get top-1 predictions
    similarities = torch.matmul(imu_embeddings, text_embeddings.T)
    top_1_indices = similarities.argmax(dim=1).cpu().numpy()

    # Build confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=np.int32)

    for i in range(N):
        true_class = class_to_idx[query_groups[i]]
        pred_class = class_to_idx[corpus_groups[top_1_indices[i]]]
        confusion[true_class, pred_class] += 1

    return confusion, unique_classes


# =============================================================================
# Full Evaluation Function
# =============================================================================

def evaluate_semantic_alignment(
    imu_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    labels: List[str],
    k_values: List[int] = [1, 5, 10, 20, 50],
    verbose: bool = True
) -> Dict:
    """
    Run full evaluation suite on semantic alignment model.

    Args:
        imu_embeddings: IMU embeddings (N, D), L2-normalized
        text_embeddings: Text embeddings (N, D), L2-normalized
        labels: Labels for each sample (N,)
        k_values: k values for recall computation
        verbose: Print results

    Returns:
        Dict with all metrics
    """
    results = {}

    # 1. Standard index-based recall (for comparison)
    if verbose:
        print("=" * 60)
        print("SEMANTIC ALIGNMENT EVALUATION")
        print("=" * 60)

    # 2. Group-based semantic recall
    semantic_recall = compute_semantic_recall(
        imu_embeddings, text_embeddings,
        labels, labels, k_values, use_groups=True
    )
    results.update(semantic_recall)

    if verbose:
        print("\n📊 Semantic Recall (with synonym groups):")
        for k in k_values:
            key = f'semantic_recall@{k}'
            if key in semantic_recall:
                print(f"  recall@{k}: {semantic_recall[key]*100:.2f}%")

    # 3. Exact label recall
    exact_recall = compute_exact_label_recall(
        imu_embeddings, text_embeddings,
        labels, labels, k_values
    )
    results.update({f'exact_{k}': v for k, v in exact_recall.items()})

    if verbose:
        print("\n📊 Exact Label Recall (no synonym grouping):")
        for k in k_values:
            key = f'semantic_recall@{k}'
            if key in exact_recall:
                print(f"  recall@{k}: {exact_recall[key]*100:.2f}%")

    # 4. Embedding quality metrics
    quality = compute_embedding_quality_metrics(
        imu_embeddings, text_embeddings, labels, use_groups=True
    )
    results.update(quality)

    if verbose:
        print("\n📊 Embedding Space Quality:")
        print(f"  IMU-Text alignment: {quality['imu_text_alignment']:.4f} ± {quality['imu_text_alignment_std']:.4f}")
        print(f"  Cross-modal gap: {quality['cross_modal_gap']:.4f} ± {quality['cross_modal_gap_std']:.4f}")
        print(f"  Intra-class distance: {quality['intra_class_distance']:.4f}")
        print(f"  Inter-class distance: {quality['inter_class_distance']:.4f}")
        print(f"  Class separability: {quality['class_separability']:.4f}")

    # 5. Per-class recall
    per_class = compute_per_class_recall(
        imu_embeddings, text_embeddings,
        labels, labels, k=1, use_groups=True
    )
    results['per_class_recall@1'] = per_class

    if verbose:
        print("\n📊 Per-Class Recall@1 (top 5 and bottom 5):")
        sorted_classes = sorted(per_class.items(), key=lambda x: x[1], reverse=True)
        print("  Best:")
        for cls, recall in sorted_classes[:5]:
            print(f"    {cls}: {recall*100:.1f}%")
        print("  Worst:")
        for cls, recall in sorted_classes[-5:]:
            print(f"    {cls}: {recall*100:.1f}%")

    # 6. Confusion matrix
    confusion, class_names = compute_confusion_matrix(
        imu_embeddings, text_embeddings,
        labels, labels, use_groups=True
    )
    results['confusion_matrix'] = confusion
    results['class_names'] = class_names

    if verbose:
        print("\n" + "=" * 60)

    return results
