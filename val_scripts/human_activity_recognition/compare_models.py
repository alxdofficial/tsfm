"""
Compare multiple semantic alignment models on validation/test sets.

Supports:
- Loading N models from checkpoint paths
- Comparing performance metrics side by side
- Testing on unseen datasets for zero-shot generalization

Usage:
    # Edit CHECKPOINT_PATHS below, then run:
    python val_scripts/human_activity_recognition/compare_models.py
"""

import torch
from torch.amp import autocast
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Add project root to path (val_scripts -> tsfm)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'tools' / 'models'))

from torch.utils.data import DataLoader

from datasets.imu_pretraining_dataset.multi_dataset_loader import create_dataloaders, IMUPretrainingDataset
from imu_activity_recognition_encoder.token_text_encoder import LearnableLabelBank
from training_scripts.human_activity_recognition.semantic_alignment_train import SemanticAlignmentModel
from val_scripts.human_activity_recognition.model_loading import load_model, load_label_bank
from val_scripts.human_activity_recognition.evaluation_metrics import get_label_to_group_mapping, compute_similarity
from datasets.imu_pretraining_dataset.label_augmentation import DATASET_CONFIGS
from datasets.imu_pretraining_dataset.label_groups import LABEL_GROUPS, get_group_for_label

# =============================================================================
# CONFIGURATION - Edit these values instead of using CLI args
# =============================================================================

# Checkpoint paths to compare - add as many as you want
# Format: {"display_name": "path/to/checkpoint.pt"}
CHECKPOINT_PATHS = {
    "11_datasets": "training_output/semantic_alignment/20260124_033735/best.pt",
}

# Datasets for evaluation (training datasets)
from val_scripts.human_activity_recognition.eval_config import (
    PATCH_SIZE_PER_DATASET, TRAINING_DATASETS, UNSEEN_DATASETS,
)
EVAL_DATASETS = TRAINING_DATASETS

# Patch sizes for unseen datasets
# Test 4 values covering the training augmentation range
UNSEEN_PATCH_SIZES = [1.0, 1.25, 1.5, 1.75]

# Channel subsets for zero-shot evaluation
# None = use all available IMU channels (no filtering needed)
UNSEEN_CHANNEL_SUBSETS = {
    'motionsense': [None],   # 6ch: acc + gyro
    'realworld': [None],     # 3ch: acc only
    'mobiact': [None],       # 6ch: acc + gyro
    'shoaib': [None],        # 45ch: 5 positions x acc + gyro + mag
    'opportunity': [None],   # 30ch: 5 IMUs x acc + gyro
    'realdisp': [None],      # 81ch: 9 sensors x acc + gyro + mag
    'daphnet_fog': [None],   # 9ch: 3 accelerometers
}

# Output directory
OUTPUT_DIR = "test_output/model_comparison"

# Evaluation settings
BATCH_SIZE = 32
MAX_SESSIONS_PER_DATASET = 10000  # Set to None for all sessions
EVAL_ON_TRAINING_DATASETS = False  # Set False to only do zero-shot eval
USE_SIMPLE_GROUPS = False  # True = coarse grouping (~12 groups), False = fine-grained (~25 groups)

# =============================================================================
# Label Coverage Analysis
# =============================================================================


def get_covered_groups(training_labels: set) -> Tuple[set, set]:
    """
    Determine which label groups have at least one member in training.

    Returns:
        covered_groups: Set of group names that have training coverage
        novel_groups: Set of group names with no training coverage
    """
    covered_groups = set()
    all_groups = set(LABEL_GROUPS.keys())

    for group_name, group_labels in LABEL_GROUPS.items():
        # Check if any label in this group is in training
        if any(label in training_labels for label in group_labels):
            covered_groups.add(group_name)

    novel_groups = all_groups - covered_groups
    return covered_groups, novel_groups


def categorize_label(label: str, training_labels: set, covered_groups: set) -> str:
    """
    Categorize a label as 'expected' or 'novel'.

    Expected: The label's group has at least one member in training
    Novel: The label's group has no members in training (completely new concept)

    Returns:
        'expected' or 'novel'
    """
    group = get_group_for_label(label)

    # If the label itself is in training, it's expected
    if label in training_labels:
        return 'expected'

    # If the label's group is covered by training, it's expected
    if group in covered_groups:
        return 'expected'

    # Check if this is a singleton (not in any group) but exists in training
    if group == label and label in training_labels:
        return 'expected'

    return 'novel'


def analyze_label_coverage(
    training_labels: set,
    zeroshot_labels: set,
    gt_labels: List[str],
    pred_labels: List[str]
) -> Dict:
    """
    Analyze label coverage and compute separate metrics for expected vs novel labels.

    Args:
        training_labels: Set of all labels seen during training
        zeroshot_labels: Set of all labels in zero-shot datasets
        gt_labels: List of ground truth labels (per sample)
        pred_labels: List of predicted labels (per sample)

    Returns:
        Dict with coverage analysis and per-category metrics
    """
    covered_groups, novel_groups = get_covered_groups(training_labels)

    # Categorize each zero-shot label
    expected_labels = set()
    novel_labels = set()

    for label in zeroshot_labels:
        category = categorize_label(label, training_labels, covered_groups)
        if category == 'expected':
            expected_labels.add(label)
        else:
            novel_labels.add(label)

    # Build detailed coverage info
    coverage_info = {
        'expected_labels': {},  # group -> {training: [...], zeroshot: [...]}
        'novel_labels': {},     # label -> dataset source (if known)
    }

    # Map expected labels to their groups
    for label in expected_labels:
        group = get_group_for_label(label)
        if group not in coverage_info['expected_labels']:
            # Find training labels in this group
            training_in_group = [l for l in LABEL_GROUPS.get(group, [label]) if l in training_labels]
            coverage_info['expected_labels'][group] = {
                'training': training_in_group,
                'zeroshot': []
            }
        if label not in training_labels:
            coverage_info['expected_labels'][group]['zeroshot'].append(label)

    # Track novel labels
    for label in novel_labels:
        coverage_info['novel_labels'][label] = get_group_for_label(label)

    # Separate predictions by category
    expected_indices = []
    novel_indices = []

    for i, gt in enumerate(gt_labels):
        category = categorize_label(gt, training_labels, covered_groups)
        if category == 'expected':
            expected_indices.append(i)
        else:
            novel_indices.append(i)

    # Compute metrics for each category
    def compute_category_metrics(indices):
        if not indices:
            return {'accuracy': 0, 'count': 0, 'correct': 0}

        correct = sum(1 for i in indices if get_group_for_label(gt_labels[i]) == get_group_for_label(pred_labels[i]))
        return {
            'accuracy': correct / len(indices),
            'count': len(indices),
            'correct': correct
        }

    expected_metrics = compute_category_metrics(expected_indices)
    novel_metrics = compute_category_metrics(novel_indices)

    # For novel labels, track what the model predicts most often
    novel_predictions = defaultdict(list)
    for i in novel_indices:
        gt = gt_labels[i]
        pred_group = get_group_for_label(pred_labels[i])
        novel_predictions[gt].append(pred_group)

    # Get most common prediction for each novel label
    novel_prediction_summary = {}
    for label, preds in novel_predictions.items():
        counter = Counter(preds)
        most_common = counter.most_common(3)  # Top 3 predictions
        novel_prediction_summary[label] = {
            'total_samples': len(preds),
            'top_predictions': [(pred, count, count/len(preds)*100) for pred, count in most_common]
        }

    return {
        'covered_groups': sorted(covered_groups),
        'novel_groups': sorted(novel_groups),
        'expected_labels': sorted(expected_labels),
        'novel_labels': sorted(novel_labels),
        'coverage_info': coverage_info,
        'expected_metrics': expected_metrics,
        'novel_metrics': novel_metrics,
        'novel_prediction_summary': novel_prediction_summary,
        'total_samples': len(gt_labels),
        'expected_sample_count': len(expected_indices),
        'novel_sample_count': len(novel_indices),
    }


def print_label_coverage_analysis(analysis: Dict, model_name: str):
    """Print formatted label coverage analysis."""
    print(f"\n{'='*70}")
    print(f"LABEL COVERAGE ANALYSIS - {model_name}")
    print(f"{'='*70}")

    # Expected labels section
    print(f"\nEXPECTED LABELS (have training equivalents):")
    print("-" * 50)
    for group, info in sorted(analysis['coverage_info']['expected_labels'].items()):
        print(f"  Group: {group}")
        print(f"    Training: {', '.join(info['training'][:5])}" +
              (f" (+{len(info['training'])-5} more)" if len(info['training']) > 5 else ""))
        if info['zeroshot']:
            print(f"    Zero-shot: {', '.join(info['zeroshot'])}")

    # Novel labels section
    print(f"\nNOVEL LABELS (no training equivalent):")
    print("-" * 50)
    for label, group in sorted(analysis['coverage_info']['novel_labels'].items()):
        print(f"  - {label}" + (f" (group: {group})" if group != label else ""))

    # Metrics comparison
    print(f"\n{'='*70}")
    print(f"METRICS BY LABEL COVERAGE - {model_name}")
    print(f"{'='*70}")
    print(f"{'Category':<20} {'Samples':>12} {'Correct':>12} {'Accuracy':>12}")
    print("-" * 56)

    exp = analysis['expected_metrics']
    nov = analysis['novel_metrics']
    total = analysis['total_samples']

    print(f"{'Expected':<20} {exp['count']:>12} ({exp['count']/total*100:>5.1f}%) {exp['correct']:>12} {exp['accuracy']*100:>11.2f}%")
    print(f"{'Novel':<20} {nov['count']:>12} ({nov['count']/total*100:>5.1f}%) {nov['correct']:>12} {nov['accuracy']*100:>11.2f}%")
    print(f"{'Overall':<20} {total:>12} {'':>8} {exp['correct']+nov['correct']:>12} {(exp['correct']+nov['correct'])/total*100:>11.2f}%")

    # Novel label predictions
    if analysis['novel_prediction_summary']:
        print(f"\n{'='*70}")
        print(f"NOVEL LABEL PREDICTIONS (what the model thinks they are)")
        print(f"{'='*70}")
        for label, info in sorted(analysis['novel_prediction_summary'].items()):
            print(f"\n  {label} ({info['total_samples']} samples):")
            for pred, count, pct in info['top_predictions']:
                print(f"    → {pred}: {count} ({pct:.1f}%)")


# =============================================================================
# Model Loading
# =============================================================================






# =============================================================================
# Metrics Computation (matches training metrics)
# =============================================================================


def compute_metrics(
    model: SemanticAlignmentModel,
    label_bank: LearnableLabelBank,
    dataloader: DataLoader,
    device: torch.device,
    all_unique_labels: List[str]
) -> Tuple[Dict[str, float], List[str], List[str], List[str], List[str]]:
    """
    Compute evaluation metrics matching training validation metrics.

    Metrics computed:
    - accuracy: Group-aware top-1 accuracy (same as training)
    - mrr: Mean Reciprocal Rank with group awareness (same as training)
    - positive_similarity: Mean cosine similarity of matched pairs
    - negative_similarity: Mean cosine similarity of non-matched pairs
    - similarity_gap: positive - negative similarity
    - recall@k: Exact label recall at k
    - group_recall@k: Group-aware recall at k
    - Per-dataset accuracy breakdown

    Returns:
        metrics: Dict of metric values
        gt_groups: List of ground truth group names
        pred_groups: List of predicted group names
        gt_labels: List of raw ground truth labels
        pred_labels: List of raw predicted labels
    """
    label_to_group = get_label_to_group_mapping(use_simple=USE_SIMPLE_GROUPS)

    all_imu_embeddings = []
    all_text_embeddings = []
    all_gt_labels = []
    all_datasets = []

    model.eval()
    label_bank.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing embeddings", leave=False):
            data = batch['data'].to(device)
            channel_mask = batch['channel_mask'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label_texts = batch['label_texts']
            metadata = batch['metadata']

            sampling_rates = [m['sampling_rate_hz'] for m in metadata]
            patch_sizes = [m['patch_size_sec'] for m in metadata]
            channel_descriptions = [m['channel_descriptions'] for m in metadata]
            datasets = [m['dataset'] for m in metadata]

            # Get IMU embeddings (model already normalizes)
            # Use autocast to match training precision (fp16 on CUDA)
            with autocast('cuda', enabled=device.type == 'cuda'):
                imu_emb = model.forward_from_raw(data, channel_descriptions, channel_mask, sampling_rates, patch_sizes,
                                                attention_mask=attention_mask)

            # Get text embeddings for this batch's labels
            text_emb = label_bank.encode(label_texts, normalize=True)

            all_imu_embeddings.append(imu_emb.cpu())
            all_text_embeddings.append(text_emb.cpu())
            all_gt_labels.extend(label_texts)
            all_datasets.extend(datasets)

    # Concatenate all embeddings
    imu_embeddings = torch.cat(all_imu_embeddings, dim=0)  # (N, D)
    text_embeddings = torch.cat(all_text_embeddings, dim=0)  # (N, D) or (N, K, D) with multi-prototype

    # Encode all unique labels for retrieval
    all_label_embeddings = label_bank.encode(all_unique_labels, normalize=True).cpu()  # (L, D) or (L, K, D)

    # Compute similarity matrix for retrieval: (N, L)
    similarity_matrix = compute_similarity(imu_embeddings, all_label_embeddings)

    metrics = {}
    N = len(all_gt_labels)
    L = len(all_unique_labels)

    # === Metrics matching training validation ===

    # 1. Group-aware accuracy (matches compute_group_accuracy in training)
    top1_indices = similarity_matrix.argmax(dim=1)
    correct_group = 0
    gt_groups = []
    pred_groups = []
    pred_labels = []  # Raw predicted labels (before group mapping)
    for i, gt_label in enumerate(all_gt_labels):
        gt_group = label_to_group.get(gt_label, gt_label)
        pred_label = all_unique_labels[top1_indices[i]]
        pred_group = label_to_group.get(pred_label, pred_label)
        gt_groups.append(gt_group)
        pred_groups.append(pred_group)
        pred_labels.append(pred_label)
        if gt_group == pred_group:
            correct_group += 1
    metrics['accuracy'] = correct_group / N

    # F1 scores (for comparison with NLS-HAR which reports F1, not accuracy)
    metrics['f1_macro'] = f1_score(gt_groups, pred_groups, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(gt_groups, pred_groups, average='weighted', zero_division=0)
    metrics['precision_macro'] = precision_score(gt_groups, pred_groups, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(gt_groups, pred_groups, average='macro', zero_division=0)

    # 2. Mean Reciprocal Rank (group-aware, matches training)
    mrr_group = 0
    for i, gt_label in enumerate(all_gt_labels):
        gt_group = label_to_group.get(gt_label, gt_label)
        sorted_indices = similarity_matrix[i].argsort(descending=True)
        for rank, idx in enumerate(sorted_indices, 1):
            pred_group = label_to_group.get(all_unique_labels[idx], all_unique_labels[idx])
            if pred_group == gt_group:
                mrr_group += 1 / rank
                break
    metrics['mrr'] = mrr_group / N

    # 3. Positive/Negative similarity (matches training loss metrics)
    # Positive: diagonal of pairwise IMU-text similarity
    if text_embeddings.dim() == 3:
        # Multi-prototype: best prototype per sample
        positive_sims = torch.einsum('nd,nkd->nk', imu_embeddings, text_embeddings).max(dim=-1).values
    else:
        positive_sims = (imu_embeddings * text_embeddings).sum(dim=1)
    metrics['positive_similarity'] = positive_sims.mean().item()

    # Negative: off-diagonal, excluding same-group pairs
    if text_embeddings.dim() == 3:
        raw_sim = torch.einsum('nd,mkd->nmk', imu_embeddings, text_embeddings).max(dim=-1).values  # (N, N)
    else:
        raw_sim = imu_embeddings @ text_embeddings.T  # (N, N)
    same_label_mask = torch.zeros(N, N, dtype=torch.bool)
    for i in range(N):
        for j in range(N):
            gi = label_to_group.get(all_gt_labels[i], all_gt_labels[i])
            gj = label_to_group.get(all_gt_labels[j], all_gt_labels[j])
            same_label_mask[i, j] = (gi == gj)

    diff_label_mask = ~same_label_mask
    if diff_label_mask.any():
        metrics['negative_similarity'] = raw_sim[diff_label_mask].mean().item()
    else:
        metrics['negative_similarity'] = 0.0

    metrics['similarity_gap'] = metrics['positive_similarity'] - metrics['negative_similarity']

    # === Additional retrieval metrics ===

    # Recall@K (exact and group-aware)
    for k in [1, 5, 10]:
        if k > L:
            continue
        correct_exact = 0
        correct_group = 0
        top_k_indices = similarity_matrix.topk(k, dim=1).indices

        for i, gt_label in enumerate(all_gt_labels):
            gt_group = label_to_group.get(gt_label, gt_label)
            predicted_labels = [all_unique_labels[idx] for idx in top_k_indices[i]]

            if gt_label in predicted_labels:
                correct_exact += 1

            predicted_groups = [label_to_group.get(lbl, lbl) for lbl in predicted_labels]
            if gt_group in predicted_groups:
                correct_group += 1

        metrics[f'recall@{k}'] = correct_exact / N
        metrics[f'group_recall@{k}'] = correct_group / N

    # === Per-dataset breakdown ===
    dataset_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
    for i, (gt_label, dataset) in enumerate(zip(all_gt_labels, all_datasets)):
        gt_group = label_to_group.get(gt_label, gt_label)
        pred_label = all_unique_labels[top1_indices[i]]
        pred_group = label_to_group.get(pred_label, pred_label)

        dataset_metrics[dataset]['total'] += 1
        if gt_group == pred_group:
            dataset_metrics[dataset]['correct'] += 1

    for dataset, counts in dataset_metrics.items():
        if counts['total'] > 0:
            metrics[f'{dataset}_accuracy'] = counts['correct'] / counts['total']

    # Per-dataset F1 scores
    dataset_predictions = defaultdict(lambda: {'gt': [], 'pred': []})
    for i, (gt_label, dataset) in enumerate(zip(all_gt_labels, all_datasets)):
        gt_group = label_to_group.get(gt_label, gt_label)
        pred_group = pred_groups[i]
        dataset_predictions[dataset]['gt'].append(gt_group)
        dataset_predictions[dataset]['pred'].append(pred_group)

    for dataset, preds in dataset_predictions.items():
        if len(preds['gt']) > 0:
            metrics[f'{dataset}_f1_macro'] = f1_score(preds['gt'], preds['pred'], average='macro', zero_division=0)

    return metrics, gt_groups, pred_groups, all_gt_labels, pred_labels


def get_unique_labels_from_loader(dataloader: DataLoader) -> List[str]:
    """Get all unique labels from a dataloader."""
    labels = set()
    for batch in dataloader:
        labels.update(batch['label_texts'])
    return sorted(list(labels))


def get_raw_labels_for_dataset(dataset_name: str) -> List[str]:
    """Get canonical/raw activity labels for a dataset."""
    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower in DATASET_CONFIGS:
        return sorted(list(DATASET_CONFIGS[dataset_name_lower]['synonyms'].keys()))
    else:
        print(f"Warning: No config found for {dataset_name}")
        return ['unknown']


def compute_closed_set_metrics(
    model: SemanticAlignmentModel,
    label_bank: LearnableLabelBank,
    dataloader: DataLoader,
    device: torch.device,
    dataset_labels: List[str]
) -> Dict[str, float]:
    """
    Compute closed-set metrics matching NLS-HAR protocol.

    This restricts predictions to ONLY the target dataset's labels,
    which is the protocol used by NLS-HAR (AAAI 2025).

    Key differences from open-set:
    - Open-set: Predicts from ALL training labels (~100), then maps to groups
    - Closed-set: Predicts from only this dataset's C labels (no group mapping)

    Args:
        model: The trained SemanticAlignmentModel
        label_bank: LearnableLabelBank for encoding labels
        dataloader: DataLoader for the target dataset
        device: torch device
        dataset_labels: List of ONLY this dataset's labels (e.g., 6 labels for MotionSense)

    Returns:
        Dict with closed-set metrics:
        - f1_closed_set: Macro F1 over dataset's C classes
        - accuracy_closed_set: Accuracy over dataset's C classes
    """
    model.eval()
    label_bank.eval()

    # Encode ONLY the dataset's labels (closed-set)
    with torch.no_grad():
        label_embeddings = label_bank.encode(dataset_labels, normalize=True)  # (C, D) or (C, K, D)
        label_embeddings = label_embeddings.to(device)

    all_gt_labels = []
    all_pred_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing closed-set metrics", leave=False):
            data = batch['data'].to(device)
            channel_mask = batch['channel_mask'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label_texts = batch['label_texts']
            metadata = batch['metadata']

            sampling_rates = [m['sampling_rate_hz'] for m in metadata]
            patch_sizes = [m['patch_size_sec'] for m in metadata]
            channel_descriptions = [m['channel_descriptions'] for m in metadata]

            # Get IMU embeddings
            with autocast('cuda', enabled=device.type == 'cuda'):
                imu_emb = model.forward_from_raw(data, channel_descriptions, channel_mask, sampling_rates, patch_sizes,
                                                attention_mask=attention_mask)

            # Compute similarity only against dataset_labels (closed-set)
            similarity = compute_similarity(imu_emb, label_embeddings)  # (batch, C)

            # Predict = argmax over C classes
            pred_indices = similarity.argmax(dim=1).cpu().numpy()
            pred_labels = [dataset_labels[i] for i in pred_indices]

            all_gt_labels.extend(label_texts)
            all_pred_labels.extend(pred_labels)

    # Compute closed-set metrics (no group mapping - raw labels)
    return {
        'f1_closed_set': f1_score(all_gt_labels, all_pred_labels, average='macro', zero_division=0),
        'accuracy_closed_set': accuracy_score(all_gt_labels, all_pred_labels),
    }


# =============================================================================
# Visualization Functions
# =============================================================================


def plot_group_histogram(gt_groups: List[str], output_path: Path, title: str = "Label Group Distribution"):
    """Plot histogram of label group frequencies."""
    from collections import Counter

    group_counts = Counter(gt_groups)
    groups = sorted(group_counts.keys())
    counts = [group_counts[g] for g in groups]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(range(len(groups)), counts, color='steelblue', edgecolor='black', alpha=0.8)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(f'{count}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xlabel('Activity Group', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Histogram saved to {output_path}")
    plt.close()


def plot_confusion_matrix(
    gt_groups: List[str],
    pred_groups: List[str],
    output_path: Path,
    title: str = "Confusion Matrix",
    only_gt_labels: bool = False
):
    """
    Plot confusion matrix for group predictions.

    Args:
        gt_groups: Ground truth group labels
        pred_groups: Predicted group labels
        output_path: Path to save the plot
        title: Plot title
        only_gt_labels: If True, only show labels that appear in ground truth
                        (useful for unseen datasets where we don't want to show
                        all possible prediction targets)
    """
    from collections import Counter

    # Get unique groups (sorted for consistent ordering)
    if only_gt_labels:
        # Only show groups that exist in ground truth
        all_groups = sorted(set(gt_groups))
    else:
        # Show all groups (GT + predictions)
        all_groups = sorted(set(gt_groups) | set(pred_groups))
    n_groups = len(all_groups)
    group_to_idx = {g: i for i, g in enumerate(all_groups)}

    # Build confusion matrix
    conf_matrix = np.zeros((n_groups, n_groups), dtype=int)
    other_predictions = 0  # Track predictions outside GT labels
    for gt, pred in zip(gt_groups, pred_groups):
        if pred in group_to_idx:
            conf_matrix[group_to_idx[gt], group_to_idx[pred]] += 1
        else:
            # Prediction is outside the displayed labels (only happens with only_gt_labels=True)
            other_predictions += 1

    if other_predictions > 0:
        print(f"  Note: {other_predictions} predictions fell outside GT labels (not shown in matrix)")

    # Normalize by row (recall per class)
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    conf_matrix_norm = np.divide(conf_matrix, row_sums, where=row_sums != 0)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(conf_matrix_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Recall (row-normalized)', rotation=-90, va="bottom", fontsize=11)

    # Set ticks
    ax.set_xticks(range(n_groups))
    ax.set_yticks(range(n_groups))
    ax.set_xticklabels(all_groups, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(all_groups, fontsize=9)

    # Add text annotations
    for i in range(n_groups):
        for j in range(n_groups):
            count = conf_matrix[i, j]
            pct = conf_matrix_norm[i, j]
            if count > 0:
                # Show count and percentage
                text_color = 'white' if pct > 0.5 else 'black'
                ax.text(j, i, f'{count}\n({pct:.0%})',
                        ha='center', va='center', color=text_color, fontsize=8)

    ax.set_xlabel('Predicted Group', fontsize=12)
    ax.set_ylabel('True Group', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {output_path}")
    plt.close()


# =============================================================================
# Main Comparison Logic
# =============================================================================


def run_comparison():
    """Run comparison across all configured checkpoints."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Label grouping: {'SIMPLE (~12 groups)' if USE_SIMPLE_GROUPS else 'FINE-GRAINED (~25 groups)'}")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate checkpoint paths
    print("\n" + "=" * 70)
    print("LOADING MODELS")
    print("=" * 70)

    loaded_models = {}
    for name, path in CHECKPOINT_PATHS.items():
        print(f"\n{name}:")
        print(f"  Path: {path}")
        try:
            model, checkpoint, hyperparams_path = load_model(path, device)
            label_bank = load_label_bank(checkpoint, device, hyperparams_path)
            info = {'epoch': checkpoint.get('epoch', 'unknown'), 'checkpoint_path': str(path)}
            loaded_models[name] = {
                'model': model,
                'label_bank': label_bank,
                'info': info
            }
            print(f"  Loaded (epoch {info['epoch']})")
        except Exception as e:
            print(f"  ✗ Failed to load: {e}")
            continue

    if not loaded_models:
        print("\nNo models loaded successfully. Exiting.")
        return

    model_names = list(loaded_models.keys())
    all_results = {}
    all_predictions = {}  # For histogram/confusion matrix
    unseen_predictions = {}  # For zero-shot histogram/confusion matrix

    # === Evaluation on training datasets ===
    if EVAL_ON_TRAINING_DATASETS and EVAL_DATASETS:
        print("\n" + "=" * 70)
        print("EVALUATING ON TRAINING DATASETS")
        print("=" * 70)
        print(f"Datasets: {EVAL_DATASETS}")

        # Create validation dataloader
        _, val_loader, _ = create_dataloaders(
            data_root='data',
            datasets=EVAL_DATASETS,
            batch_size=BATCH_SIZE,
            patch_size_per_dataset=PATCH_SIZE_PER_DATASET,
            max_sessions_per_dataset=MAX_SESSIONS_PER_DATASET,
            num_workers=0,
            seed=42
        )

        # Get unique labels
        print("Collecting unique labels...")
        all_labels = get_unique_labels_from_loader(val_loader)
        print(f"Found {len(all_labels)} unique labels")

        all_results['training_datasets'] = {}
        all_predictions = {}  # Store predictions for plotting

        for name in model_names:
            print(f"\nEvaluating {name}...")

            # Recreate dataloader (consumed by previous iteration)
            _, val_loader, _ = create_dataloaders(
                data_root='data',
                datasets=EVAL_DATASETS,
                batch_size=BATCH_SIZE,
                patch_size_per_dataset=PATCH_SIZE_PER_DATASET,
                max_sessions_per_dataset=MAX_SESSIONS_PER_DATASET,
                num_workers=0,
                seed=42
            )

            metrics, gt_groups, pred_groups, gt_labels, pred_labels = compute_metrics(
                loaded_models[name]['model'],
                loaded_models[name]['label_bank'],
                val_loader,
                device,
                all_labels
            )
            all_results['training_datasets'][name] = metrics
            all_predictions[name] = {'gt_groups': gt_groups, 'pred_groups': pred_groups,
                                     'gt_labels': gt_labels, 'pred_labels': pred_labels}

    # === Zero-shot evaluation on unseen datasets ===
    if UNSEEN_DATASETS:
        print("\n" + "=" * 70)
        print("ZERO-SHOT EVALUATION ON UNSEEN DATASETS")
        print("=" * 70)
        print(f"Unseen datasets: {UNSEEN_DATASETS}")
        print(f"Patch sizes to try: {UNSEEN_PATCH_SIZES}")

        # For zero-shot: use ALL training labels as retrieval set (more challenging & realistic)
        # This tests if the model can find the correct activity among ALL known activities
        # Group-aware metrics give credit for synonyms (e.g., "jogging" matches "running" group)
        print("\nBuilding retrieval set from ALL training dataset labels...")
        all_training_labels = set()
        for ds_name in EVAL_DATASETS:
            ds_labels = get_raw_labels_for_dataset(ds_name)
            all_training_labels.update(ds_labels)
            print(f"  {ds_name}: {len(ds_labels)} labels")

        # Save a copy of training-only labels for coverage analysis
        training_only_labels = all_training_labels.copy()

        # Also add unseen dataset labels (in case they have unique labels)
        print(f"Adding unseen dataset labels:")
        all_retrieval_labels = all_training_labels.copy()  # Start with training labels
        for ds_name in UNSEEN_DATASETS:
            ds_labels = get_raw_labels_for_dataset(ds_name)
            new_labels = set(ds_labels) - training_only_labels
            if new_labels:
                print(f"  {ds_name}: {len(new_labels)} NEW labels not in training: {sorted(new_labels)}")
            all_retrieval_labels.update(ds_labels)

        combined_labels = sorted(all_retrieval_labels)
        print(f"Total retrieval set: {len(combined_labels)} unique labels")

        # Store results for each configuration
        all_results['unseen_datasets'] = {}
        all_results['unseen_per_dataset'] = {}  # Per-dataset results with optimal settings
        all_results['unseen_channel_ablation'] = {}  # Channel subset ablation results
        unseen_predictions = {}  # Store predictions for plotting (best config)

        for name in model_names:
            print(f"\nEvaluating {name} (zero-shot) per dataset with multiple configurations...")

            # Track per-dataset optimal results
            per_dataset_best = {}
            channel_ablation = {}  # Track results for each channel subset

            for ds_name in UNSEEN_DATASETS:
                print(f"\n  Dataset: {ds_name}")

                # Get channel subsets to try for this dataset
                channel_subsets = UNSEEN_CHANNEL_SUBSETS.get(ds_name, [None])

                best_accuracy = -1
                best_patch_size = None
                best_channel_filter = None
                best_metrics = None
                dataset_ablation = []

                for channel_filter in channel_subsets:
                    filter_name = 'all_channels' if channel_filter is None else '+'.join(channel_filter)
                    print(f"\n    Channel filter: {filter_name}")

                    filter_best_acc = -1
                    filter_best_patch = None
                    filter_best_metrics = None

                    for patch_size in UNSEEN_PATCH_SIZES:
                        eval_patch_sizes = {ds_name: patch_size}

                        try:
                            unseen_dataset = IMUPretrainingDataset(
                                data_root='data',
                                datasets=[ds_name],
                                split='val',
                                patch_size_per_dataset=eval_patch_sizes,
                                max_sessions_per_dataset=MAX_SESSIONS_PER_DATASET,
                                channel_filter=channel_filter,
                                seed=42
                            )
                        except ValueError as e:
                            print(f"      patch={patch_size}s: SKIPPED ({e})")
                            continue

                        unseen_loader = DataLoader(
                            unseen_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=IMUPretrainingDataset.collate_fn
                        )

                        metrics, _, _, _, _ = compute_metrics(
                            loaded_models[name]['model'],
                            loaded_models[name]['label_bank'],
                            unseen_loader,
                            device,
                            combined_labels
                        )

                        print(f"      patch={patch_size}s: accuracy={metrics['accuracy']*100:.2f}%")

                        # Track best for this filter
                        if metrics['accuracy'] > filter_best_acc:
                            filter_best_acc = metrics['accuracy']
                            filter_best_patch = patch_size
                            filter_best_metrics = metrics

                        # Track overall best
                        if metrics['accuracy'] > best_accuracy:
                            best_accuracy = metrics['accuracy']
                            best_patch_size = patch_size
                            best_channel_filter = channel_filter
                            best_metrics = metrics

                    # Record ablation result for this filter
                    if filter_best_metrics is not None:
                        dataset_ablation.append({
                            'channel_filter': filter_name,
                            'best_patch_size': filter_best_patch,
                            'accuracy': filter_best_acc,
                        })
                        print(f"    → {filter_name} BEST: patch={filter_best_patch}s, acc={filter_best_acc*100:.2f}%")

                channel_ablation[ds_name] = dataset_ablation

                # Compute closed-set metrics (NLS-HAR style) using best config
                # This uses ONLY the target dataset's labels for prediction
                dataset_labels = get_raw_labels_for_dataset(ds_name)
                closed_set_loader = DataLoader(
                    IMUPretrainingDataset(
                        data_root='data',
                        datasets=[ds_name],
                        split='val',
                        patch_size_per_dataset={ds_name: best_patch_size},
                        max_sessions_per_dataset=MAX_SESSIONS_PER_DATASET,
                        channel_filter=best_channel_filter,
                        seed=42
                    ),
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=IMUPretrainingDataset.collate_fn
                )
                closed_metrics = compute_closed_set_metrics(
                    loaded_models[name]['model'],
                    loaded_models[name]['label_bank'],
                    closed_set_loader,
                    device,
                    dataset_labels
                )

                filter_name = 'all_channels' if best_channel_filter is None else '+'.join(best_channel_filter)
                per_dataset_best[ds_name] = {
                    'best_patch_size': best_patch_size,
                    'best_channel_filter': best_channel_filter,
                    'channel_filter_name': filter_name,
                    'accuracy': best_accuracy,
                    'f1_closed_set': closed_metrics['f1_closed_set'],
                    'accuracy_closed_set': closed_metrics['accuracy_closed_set'],
                    'metrics': best_metrics
                }
                print(f"  ★ {ds_name} OVERALL BEST: {filter_name}, patch={best_patch_size}s, acc={best_accuracy*100:.2f}%, F1_closed={closed_metrics['f1_closed_set']*100:.2f}%")

            all_results['unseen_per_dataset'][name] = per_dataset_best
            all_results['unseen_channel_ablation'][name] = channel_ablation

            # Compute combined metrics using per-dataset optimal settings
            print(f"\n  Computing combined metrics with optimal settings per dataset...")

            # For combined evaluation, we need to evaluate each dataset separately with its optimal settings
            # then aggregate the results (since channel_filter is per-dataset)
            all_gt_groups = []
            all_pred_groups = []
            all_gt_labels = []
            all_pred_labels = []
            combined_correct = 0
            combined_total = 0

            for ds_name in UNSEEN_DATASETS:
                ds_config = per_dataset_best[ds_name]
                optimal_patch = ds_config['best_patch_size']
                optimal_filter = ds_config['best_channel_filter']

                unseen_dataset = IMUPretrainingDataset(
                    data_root='data',
                    datasets=[ds_name],
                    split='val',
                    patch_size_per_dataset={ds_name: optimal_patch},
                    max_sessions_per_dataset=MAX_SESSIONS_PER_DATASET,
                    channel_filter=optimal_filter,
                    seed=42
                )
                unseen_loader = DataLoader(
                    unseen_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=IMUPretrainingDataset.collate_fn
                )

                metrics, gt_groups, pred_groups, gt_labels, pred_labels = compute_metrics(
                    loaded_models[name]['model'],
                    loaded_models[name]['label_bank'],
                    unseen_loader,
                    device,
                    combined_labels
                )

                all_gt_groups.extend(gt_groups)
                all_pred_groups.extend(pred_groups)
                all_gt_labels.extend(gt_labels)
                all_pred_labels.extend(pred_labels)
                combined_correct += int(metrics['accuracy'] * len(gt_labels))
                combined_total += len(gt_labels)

            combined_accuracy = combined_correct / combined_total if combined_total > 0 else 0

            # Store combined results
            optimal_configs = {ds: f"{per_dataset_best[ds]['channel_filter_name']}@{per_dataset_best[ds]['best_patch_size']}s"
                              for ds in UNSEEN_DATASETS}
            all_results['unseen_datasets'][name] = {
                'accuracy': combined_accuracy,
                'total_samples': combined_total,
                'optimal_configs': optimal_configs,
            }
            unseen_predictions[name] = {
                'gt_groups': all_gt_groups,
                'pred_groups': all_pred_groups,
                'gt_labels': all_gt_labels,
                'pred_labels': all_pred_labels
            }
            print(f"  COMBINED (optimal settings): accuracy={combined_accuracy*100:.2f}% ({combined_total} samples)")

        # === Label Coverage Analysis ===
        print("\n" + "=" * 70)
        print("LABEL COVERAGE ANALYSIS")
        print("=" * 70)

        # Get zero-shot dataset labels
        zeroshot_labels = set()
        for ds_name in UNSEEN_DATASETS:
            zeroshot_labels.update(get_raw_labels_for_dataset(ds_name))

        # Perform coverage analysis for each model
        all_results['coverage_analysis'] = {}
        for name in model_names:
            if name in unseen_predictions:
                preds = unseen_predictions[name]
                coverage_analysis = analyze_label_coverage(
                    training_labels=training_only_labels,
                    zeroshot_labels=zeroshot_labels,
                    gt_labels=preds['gt_labels'],
                    pred_labels=preds['pred_labels']
                )
                all_results['coverage_analysis'][name] = coverage_analysis
                print_label_coverage_analysis(coverage_analysis, name)

    # === Print Results ===
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    def print_metrics_table(results: Dict, title: str):
        if not results:
            return

        print(f"\n{title}")
        print("-" * 70)

        # Header
        print(f"{'Metric':<25}", end="")
        for name in model_names:
            if name in results:
                print(f"{name:>15}", end="")
        print()
        print("-" * (25 + 15 * len([n for n in model_names if n in results])))

        # Key metrics (same order as training, plus F1 scores for baseline comparison)
        key_metrics = [
            'accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro',
            'mrr', 'positive_similarity', 'negative_similarity',
            'similarity_gap', 'recall@1', 'recall@5', 'group_recall@1', 'group_recall@5'
        ]

        for metric in key_metrics:
            if not any(metric in results.get(name, {}) for name in model_names):
                continue
            print(f"{metric:<25}", end="")
            for name in model_names:
                if name not in results:
                    continue
                val = results[name].get(metric, None)
                if val is None:
                    print(f"{'N/A':>15}", end="")
                elif 'similarity' in metric or 'gap' in metric:
                    print(f"{val:>15.4f}", end="")
                else:
                    print(f"{val*100:>14.2f}%", end="")
            print()

        # Per-dataset accuracy and F1
        dataset_keys = [k for k in results[model_names[0]].keys() if k.endswith('_accuracy')]
        if dataset_keys:
            print(f"\n{'Per-dataset metrics:':<25}")
            for key in sorted(dataset_keys):
                dataset_name = key.replace('_accuracy', '')
                print(f"  {dataset_name:<23}", end="")
                for name in model_names:
                    if name not in results:
                        continue
                    acc = results[name].get(key, 0)
                    f1_key = f'{dataset_name}_f1_macro'
                    f1 = results[name].get(f1_key, None)
                    if f1 is not None:
                        print(f"{acc*100:>9.2f}% (F1:{f1*100:>5.1f}%)", end="")
                    else:
                        print(f"{acc*100:>14.2f}%", end="")
                print()

    if 'training_datasets' in all_results:
        print_metrics_table(all_results['training_datasets'], "Training Datasets")

    if 'unseen_datasets' in all_results:
        print_metrics_table(all_results['unseen_datasets'], "Zero-Shot (Unseen Datasets) - Optimal Settings")

        # Print per-dataset breakdown with optimal settings (channel filter + patch size)
        # Shows both open-set F1 (group-level) and closed-set F1 (NLS-HAR comparable)
        if 'unseen_per_dataset' in all_results:
            print(f"\n{'Per-Dataset Results with Optimal Settings:':<60}")
            print("-" * 140)
            print(f"{'Dataset':<15}", end="")
            for name in model_names:
                print(f"{'Acc(open)':>12}{'F1(open)':>10}{'F1(closed)':>12}{'Channels':>15}{'Patch':>8}", end="")
            print()
            print("-" * 140)
            for ds_name in UNSEEN_DATASETS:
                print(f"{ds_name:<15}", end="")
                for name in model_names:
                    if name in all_results['unseen_per_dataset']:
                        ds_results = all_results['unseen_per_dataset'][name].get(ds_name, {})
                        acc = ds_results.get('accuracy', 0)
                        ds_metrics = ds_results.get('metrics', {})
                        f1_open = ds_metrics.get(f'{ds_name}_f1_macro', ds_metrics.get('f1_macro', 0))
                        f1_closed = ds_results.get('f1_closed_set', 0)
                        ch_filter = ds_results.get('channel_filter_name', 'all')
                        patch = ds_results.get('best_patch_size', 'N/A')
                        print(f"{acc*100:>11.2f}%{f1_open*100:>9.2f}%{f1_closed*100:>11.2f}%{ch_filter:>15}{patch:>7}s", end="")
                    else:
                        print(f"{'N/A':>12}{'N/A':>10}{'N/A':>12}{'N/A':>15}{'N/A':>8}", end="")
                print()
            print("-" * 140)
            print("  Note: F1(open) = open-set group-level, F1(closed) = closed-set raw labels (NLS-HAR comparable)")

        # Print channel ablation results if multiple subsets were tested
        if 'unseen_channel_ablation' in all_results:
            has_ablation = any(
                len(ds_results) > 1
                for model_results in all_results['unseen_channel_ablation'].values()
                for ds_results in model_results.values()
            )
            if has_ablation:
                print(f"\n{'Channel Ablation Results (datasets with multiple channel configs):':<60}")
                print("-" * 80)
                for name in model_names:
                    if name not in all_results['unseen_channel_ablation']:
                        continue
                    print(f"\n  {name}:")
                    for ds_name, ablation_results in all_results['unseen_channel_ablation'][name].items():
                        if len(ablation_results) <= 1:
                            continue
                        print(f"    {ds_name}:")
                        for result in ablation_results:
                            ch = result['channel_filter']
                            acc = result['accuracy']
                            patch = result['best_patch_size']
                            marker = '★' if result == max(ablation_results, key=lambda x: x['accuracy']) else ' '
                            print(f"      {marker} {ch:<20} acc={acc*100:>6.2f}%  patch={patch}s")

    # Save results
    results_path = output_dir / 'comparison_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")

    # Create plots
    if 'training_datasets' in all_results and len(model_names) > 0:
        create_comparison_plot(all_results['training_datasets'], model_names, output_dir, "Training Datasets")

        # Create histogram and confusion matrix for each model
        for name in model_names:
            if name in all_predictions:
                preds = all_predictions[name]
                # Histogram of ground truth groups
                plot_group_histogram(
                    preds['gt_groups'],
                    output_dir / f'histogram_{name}_training.png',
                    title=f"Label Group Distribution - {name} (Training)"
                )
                # Confusion matrix
                plot_confusion_matrix(
                    preds['gt_groups'],
                    preds['pred_groups'],
                    output_dir / f'confusion_matrix_{name}_training.png',
                    title=f"Confusion Matrix - {name} (Training)"
                )

    if 'unseen_datasets' in all_results and len(model_names) > 0:
        create_comparison_plot(all_results['unseen_datasets'], model_names, output_dir, "Zero-Shot")

        # Create histogram and confusion matrix for each model (zero-shot)
        for name in model_names:
            if name in unseen_predictions:
                preds = unseen_predictions[name]
                plot_group_histogram(
                    preds['gt_groups'],
                    output_dir / f'histogram_{name}_zeroshot.png',
                    title=f"Label Group Distribution - {name} (Zero-Shot)"
                )
                plot_confusion_matrix(
                    preds['gt_groups'],
                    preds['pred_groups'],
                    output_dir / f'confusion_matrix_{name}_zeroshot.png',
                    title=f"Confusion Matrix - {name} (Zero-Shot)",
                    only_gt_labels=True  # Only show labels present in unseen dataset
                )


def create_comparison_plot(results: Dict, model_names: List[str], output_dir: Path, title_suffix: str):
    """Create bar chart comparing models."""
    metrics_to_plot = ['accuracy', 'mrr', 'positive_similarity', 'similarity_gap']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    x = np.arange(len(model_names))
    width = 0.6
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

    for ax_idx, metric in enumerate(metrics_to_plot):
        ax = axes[ax_idx]
        values = []
        for name in model_names:
            val = results.get(name, {}).get(metric, 0)
            if 'similarity' not in metric and 'gap' not in metric:
                val = val * 100  # Convert to percentage
            values.append(val)

        bars = ax.bar(x, values, width, color=colors)

        for bar, val in zip(bars, values):
            height = bar.get_height()
            fmt = f'{val:.1f}%' if 'similarity' not in metric and 'gap' not in metric else f'{val:.3f}'
            ax.annotate(fmt, xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)

        ax.set_ylabel('%' if 'similarity' not in metric and 'gap' not in metric else 'Cosine Sim')
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle(f'Model Comparison - {title_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    suffix = title_suffix.lower().replace(' ', '_').replace('(', '').replace(')', '')
    plot_path = output_dir / f'comparison_{suffix}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to {plot_path}")
    plt.close()


if __name__ == '__main__':
    run_comparison()
