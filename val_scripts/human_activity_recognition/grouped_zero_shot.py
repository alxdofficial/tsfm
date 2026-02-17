"""
Shared utilities for zero-shot evaluation of non-text-aligned models.

Provides label mapping, closed-set masking, and group-based scoring functions
used by individual model evaluation scripts (LiMU-BERT, MOMENT, CrossHAR).
Each model trains its own native classifier for zero-shot evaluation.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from datasets.imu_pretraining_dataset.label_groups import (
    LABEL_GROUPS,
    get_label_to_group_mapping,
)

# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "benchmark_data"
LIMUBERT_DATA_DIR = BENCHMARK_DIR / "processed" / "limubert"
DATASET_CONFIG_PATH = BENCHMARK_DIR / "dataset_config.json"
GLOBAL_LABEL_PATH = LIMUBERT_DATA_DIR / "global_label_mapping.json"


# =============================================================================
# Label Utilities
# =============================================================================

def load_global_labels() -> List[str]:
    """Load the 87 global training labels (sorted)."""
    with open(GLOBAL_LABEL_PATH) as f:
        return json.load(f)["labels"]


def load_dataset_config() -> dict:
    """Load the dataset configuration."""
    with open(DATASET_CONFIG_PATH) as f:
        return json.load(f)


def map_local_to_global_labels(
    local_labels: np.ndarray,
    dataset_name: str,
    dataset_config: dict,
    global_labels: List[str],
) -> np.ndarray:
    """Convert per-dataset local label indices to global label indices (0..86).

    Args:
        local_labels: (N,) local indices 0..num_classes-1
        dataset_name: name of the dataset in dataset_config
        dataset_config: loaded dataset_config.json
        global_labels: list of 87 global label strings

    Returns:
        (N,) global indices
    """
    sorted_activities = sorted(dataset_config["datasets"][dataset_name]["activities"])
    global_label_to_idx = {label: i for i, label in enumerate(global_labels)}

    global_indices = np.empty(len(local_labels), dtype=np.int64)
    for i, local_idx in enumerate(local_labels):
        activity_name = sorted_activities[local_idx]
        global_indices[i] = global_label_to_idx[activity_name]

    return global_indices


def get_mappable_info(
    test_dataset: str,
    global_labels: List[str],
    dataset_config: dict,
) -> Tuple[Set[str], float]:
    """Determine which test labels are mappable through synonym groups.

    A test label is mappable if its group contains at least one training label.

    Returns:
        mappable_test_labels: set of test label names that have training synonyms
        coverage: fraction of test dataset's labels that are mappable
    """
    label_to_group = get_label_to_group_mapping()
    training_label_set = set(global_labels)
    test_activities = sorted(dataset_config["datasets"][test_dataset]["activities"])

    mappable = set()
    for label in test_activities:
        group_name = label_to_group.get(label, label)
        if group_name in LABEL_GROUPS:
            # Check if any member of this group is in training
            if any(member in training_label_set for member in LABEL_GROUPS[group_name]):
                mappable.add(label)
        elif label in training_label_set:
            # Singleton: directly in training
            mappable.add(label)

    coverage = len(mappable) / len(test_activities) if test_activities else 0.0
    return mappable, coverage


# =============================================================================
# Closed-Set Mask
# =============================================================================

def get_closed_set_mask(
    test_dataset: str,
    global_labels: List[str],
    dataset_config: dict,
) -> np.ndarray:
    """Build a boolean mask over the 87 global labels for closed-set scoring.

    A training label is allowed (True) if its synonym group is represented
    among the test dataset's activities. Labels whose group has NO test
    activities are masked out (False).

    Returns:
        mask: (87,) bool array — True for allowed training labels
    """
    label_to_group = get_label_to_group_mapping()
    test_activities = sorted(dataset_config["datasets"][test_dataset]["activities"])

    # Collect groups present in the test dataset
    test_groups = set()
    for label in test_activities:
        group = label_to_group.get(label, label)
        test_groups.add(group)

    # Allow training labels whose group is in test_groups
    mask = np.zeros(len(global_labels), dtype=bool)
    for i, label in enumerate(global_labels):
        group = label_to_group.get(label, label)
        if group in test_groups:
            mask[i] = True

    return mask


# =============================================================================
# Scoring Functions
# =============================================================================

def score_with_groups(
    pred_global_indices: np.ndarray,
    test_labels: np.ndarray,
    test_dataset: str,
    global_labels: List[str],
    dataset_config: dict,
) -> Dict[str, float]:
    """Score predictions using label group matching on all samples.

    Maps both predictions and ground truth through label groups, then
    computes accuracy and F1. Unmappable test labels (whose group has
    no training members) will always be wrong — this is a genuine
    limitation vs text-aligned models and is included in the score.

    Returns dict with: accuracy, f1_macro, n_samples
    """
    label_to_group = get_label_to_group_mapping()
    test_activities = sorted(dataset_config["datasets"][test_dataset]["activities"])

    pred_groups = []
    gt_groups = []

    for i in range(len(test_labels)):
        local_idx = test_labels[i]
        if local_idx >= len(test_activities):
            continue

        gt_name = test_activities[local_idx]
        gt_group = label_to_group.get(gt_name, gt_name)

        pred_idx = pred_global_indices[i]
        pred_name = global_labels[pred_idx] if pred_idx < len(global_labels) else "unknown"
        pred_group = label_to_group.get(pred_name, pred_name)

        gt_groups.append(gt_group)
        pred_groups.append(pred_group)

    n_samples = len(gt_groups)
    if n_samples == 0:
        return {'accuracy': 0.0, 'f1_macro': 0.0, 'n_samples': 0}

    acc = accuracy_score(gt_groups, pred_groups) * 100
    f1 = f1_score(gt_groups, pred_groups, average='macro', zero_division=0) * 100
    f1_w = f1_score(gt_groups, pred_groups, average='weighted', zero_division=0) * 100

    return {'accuracy': acc, 'f1_macro': f1, 'f1_weighted': f1_w, 'n_samples': n_samples}
