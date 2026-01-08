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
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

# Add project root to path (val_scripts -> tsfm)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'tools' / 'models'))

from torch.utils.data import DataLoader

from imu_activity_recognition_encoder.encoder import IMUActivityRecognitionEncoder
from imu_activity_recognition_encoder.semantic_alignment import SemanticAlignmentHead
from training_scripts.human_activity_recognition.semantic_alignment_train import SemanticAlignmentModel
from datasets.imu_pretraining_dataset.multi_dataset_loader import create_dataloaders, IMUPretrainingDataset
from imu_activity_recognition_encoder.token_text_encoder import LearnableLabelBank
from val_scripts.human_activity_recognition.evaluation_metrics import get_label_to_group_mapping
from datasets.imu_pretraining_dataset.label_augmentation import DATASET_CONFIGS

# =============================================================================
# CONFIGURATION - Edit these values instead of using CLI args
# =============================================================================

# Checkpoint paths to compare - add as many as you want
# Format: {"display_name": "path/to/checkpoint.pt"}
CHECKPOINT_PATHS = {
    "current": "training_output/semantic_alignment/20260107_102025/epoch_60.pt",
    # Add more checkpoints here:
    # "previous": "training_output/semantic_alignment/20260105_123456/best.pt",
}

# Datasets for evaluation (training datasets)
EVAL_DATASETS = ['uci_har', 'hhar', 'mhealth', 'pamap2', 'wisdm', 'unimib_shar']

# Unseen datasets for zero-shot evaluation (empty list to skip)
UNSEEN_DATASETS = ['motionsense']

# Output directory
OUTPUT_DIR = "test_output/model_comparison"

# Evaluation settings
BATCH_SIZE = 32
MAX_SESSIONS_PER_DATASET = 10000  # Set to None for all sessions
EVAL_ON_TRAINING_DATASETS = True  # Set False to only do zero-shot eval
USE_SIMPLE_GROUPS = False  # True = coarse grouping (~12 groups), False = fine-grained (~25 groups)

# =============================================================================
# Model Loading
# =============================================================================


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[SemanticAlignmentModel, dict, dict]:
    """
    Load model from checkpoint with architecture from hyperparameters.json.

    Returns:
        model: The loaded SemanticAlignmentModel
        model_info: Dict with epoch and checkpoint_path
        checkpoint: The full checkpoint dict (for loading label_bank state)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    epoch = checkpoint.get('epoch', 'unknown')

    # Load hyperparameters from checkpoint directory
    hyperparams_path = checkpoint_path.parent / 'hyperparameters.json'
    if hyperparams_path.exists():
        with open(hyperparams_path) as f:
            hyperparams = json.load(f)
        enc_cfg = hyperparams.get('encoder', {})
        head_cfg = hyperparams.get('semantic_head', {})
        token_cfg = hyperparams.get('token_level_text', {})
    else:
        raise FileNotFoundError(
            f"hyperparameters.json not found at {hyperparams_path}. "
            "This checkpoint may be from an older incompatible version."
        )

    # Create encoder
    encoder = IMUActivityRecognitionEncoder(
        d_model=enc_cfg.get('d_model', 384),
        num_heads=enc_cfg.get('num_heads', 8),
        num_temporal_layers=enc_cfg.get('num_temporal_layers', 4),
        dim_feedforward=enc_cfg.get('dim_feedforward', 1536),
        dropout=enc_cfg.get('dropout', 0.1),
        use_cross_channel=enc_cfg.get('use_cross_channel', True),
        cnn_channels=enc_cfg.get('cnn_channels', [32, 64]),
        cnn_kernel_sizes=enc_cfg.get('cnn_kernel_sizes', [5]),
        target_patch_size=enc_cfg.get('target_patch_size', 64),
        use_channel_encoding=enc_cfg.get('use_channel_encoding', False)
    )

    # Create semantic head
    semantic_head = SemanticAlignmentHead(
        d_model=enc_cfg.get('d_model', 384),
        d_model_fused=384,
        output_dim=384,
        num_temporal_layers=head_cfg.get('num_temporal_layers', 2),
        num_heads=enc_cfg.get('num_heads', 8),
        dim_feedforward=enc_cfg.get('dim_feedforward', 1536),
        dropout=enc_cfg.get('dropout', 0.1),
        num_fusion_queries=head_cfg.get('num_fusion_queries', 4),
        use_fusion_self_attention=head_cfg.get('use_fusion_self_attention', True),
        num_pool_queries=head_cfg.get('num_pool_queries', 4),
        use_pool_self_attention=head_cfg.get('use_pool_self_attention', True)
    )

    # Create full model with token-level text encoding
    model = SemanticAlignmentModel(
        encoder,
        semantic_head,
        num_heads=token_cfg.get('num_heads', 4),
        dropout=enc_cfg.get('dropout', 0.1)
    )

    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint['model_state_dict'], strict=False
    )
    if unexpected_keys:
        # Filter out expected channel encoding keys
        other_unexpected = [k for k in unexpected_keys if 'channel_encoding' not in k]
        if other_unexpected:
            print(f"  Warning: Unexpected keys: {other_unexpected[:5]}...")
    if missing_keys:
        print(f"  Warning: Missing keys: {missing_keys[:5]}...")

    model.eval()
    model = model.to(device)

    model_info = {
        'epoch': epoch,
        'checkpoint_path': str(checkpoint_path)
    }

    return model, model_info, checkpoint


def load_label_bank(checkpoint: dict, device: torch.device, hyperparams_path: Path) -> LearnableLabelBank:
    """Load LearnableLabelBank with trained state from checkpoint."""
    # Get config from hyperparameters
    if hyperparams_path.exists():
        with open(hyperparams_path) as f:
            hyperparams = json.load(f)
        token_cfg = hyperparams.get('token_level_text', {})
    else:
        token_cfg = {}

    label_bank = LearnableLabelBank(
        device=device,
        num_heads=token_cfg.get('num_heads', 4),
        num_queries=token_cfg.get('num_queries', 4),
        dropout=0.1
    )

    # Load trained weights if available
    if 'label_bank_state_dict' in checkpoint:
        label_bank.load_state_dict(checkpoint['label_bank_state_dict'])
        print("  ✓ Loaded trained LearnableLabelBank state")
    else:
        print("  ⚠ No label_bank_state_dict in checkpoint, using untrained weights")

    label_bank.eval()
    return label_bank


# =============================================================================
# Metrics Computation (matches training metrics)
# =============================================================================


def compute_metrics(
    model: SemanticAlignmentModel,
    label_bank: LearnableLabelBank,
    dataloader: DataLoader,
    device: torch.device,
    all_unique_labels: List[str]
) -> Dict[str, float]:
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
            label_texts = batch['label_texts']
            metadata = batch['metadata']

            sampling_rates = [m['sampling_rate_hz'] for m in metadata]
            patch_sizes = [m['patch_size_sec'] for m in metadata]
            channel_descriptions = [m['channel_descriptions'] for m in metadata]
            datasets = [m['dataset'] for m in metadata]

            # Get IMU embeddings (model already normalizes)
            imu_emb = model(data, channel_descriptions, channel_mask, sampling_rates, patch_sizes)

            # Get text embeddings for this batch's labels
            text_emb = label_bank.encode(label_texts, normalize=True)

            all_imu_embeddings.append(imu_emb.cpu())
            all_text_embeddings.append(text_emb.cpu())
            all_gt_labels.extend(label_texts)
            all_datasets.extend(datasets)

    # Concatenate all embeddings
    imu_embeddings = torch.cat(all_imu_embeddings, dim=0)  # (N, D)
    text_embeddings = torch.cat(all_text_embeddings, dim=0)  # (N, D) - per-sample text

    # Encode all unique labels for retrieval
    all_label_embeddings = label_bank.encode(all_unique_labels, normalize=True).cpu()  # (L, D)

    # Compute similarity matrix for retrieval: (N, L)
    similarity_matrix = imu_embeddings @ all_label_embeddings.T

    metrics = {}
    N = len(all_gt_labels)
    L = len(all_unique_labels)

    # === Metrics matching training validation ===

    # 1. Group-aware accuracy (matches compute_group_accuracy in training)
    top1_indices = similarity_matrix.argmax(dim=1)
    correct_group = 0
    for i, gt_label in enumerate(all_gt_labels):
        gt_group = label_to_group.get(gt_label, gt_label)
        pred_label = all_unique_labels[top1_indices[i]]
        pred_group = label_to_group.get(pred_label, pred_label)
        if gt_group == pred_group:
            correct_group += 1
    metrics['accuracy'] = correct_group / N

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
    positive_sims = (imu_embeddings * text_embeddings).sum(dim=1)
    metrics['positive_similarity'] = positive_sims.mean().item()

    # Negative: off-diagonal, excluding same-group pairs
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

    return metrics


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
            model, info, checkpoint = load_model(path, device)
            hyperparams_path = Path(path).parent / 'hyperparameters.json'
            label_bank = load_label_bank(checkpoint, device, hyperparams_path)
            loaded_models[name] = {
                'model': model,
                'label_bank': label_bank,
                'info': info
            }
            print(f"  ✓ Loaded (epoch {info['epoch']})")
        except Exception as e:
            print(f"  ✗ Failed to load: {e}")
            continue

    if not loaded_models:
        print("\nNo models loaded successfully. Exiting.")
        return

    model_names = list(loaded_models.keys())
    all_results = {}

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
            max_sessions_per_dataset=MAX_SESSIONS_PER_DATASET,
            num_workers=0,
            seed=42
        )

        # Get unique labels
        print("Collecting unique labels...")
        all_labels = get_unique_labels_from_loader(val_loader)
        print(f"Found {len(all_labels)} unique labels")

        all_results['training_datasets'] = {}

        for name in model_names:
            print(f"\nEvaluating {name}...")

            # Recreate dataloader (consumed by previous iteration)
            _, val_loader, _ = create_dataloaders(
                data_root='data',
                datasets=EVAL_DATASETS,
                batch_size=BATCH_SIZE,
                max_sessions_per_dataset=MAX_SESSIONS_PER_DATASET,
                num_workers=0,
                seed=42
            )

            metrics = compute_metrics(
                loaded_models[name]['model'],
                loaded_models[name]['label_bank'],
                val_loader,
                device,
                all_labels
            )
            all_results['training_datasets'][name] = metrics

    # === Zero-shot evaluation on unseen datasets ===
    if UNSEEN_DATASETS:
        print("\n" + "=" * 70)
        print("ZERO-SHOT EVALUATION ON UNSEEN DATASETS")
        print("=" * 70)
        print(f"Unseen datasets: {UNSEEN_DATASETS}")

        # For zero-shot: use ALL training labels as retrieval set (more challenging & realistic)
        # This tests if the model can find the correct activity among ALL known activities
        # Group-aware metrics give credit for synonyms (e.g., "jogging" matches "running" group)
        print("\nBuilding retrieval set from ALL training dataset labels...")
        all_training_labels = set()
        for ds_name in EVAL_DATASETS:
            ds_labels = get_raw_labels_for_dataset(ds_name)
            all_training_labels.update(ds_labels)
            print(f"  {ds_name}: {len(ds_labels)} labels")

        # Also add unseen dataset labels (in case they have unique labels)
        print(f"Adding unseen dataset labels:")
        for ds_name in UNSEEN_DATASETS:
            ds_labels = get_raw_labels_for_dataset(ds_name)
            new_labels = set(ds_labels) - all_training_labels
            if new_labels:
                print(f"  {ds_name}: {len(new_labels)} NEW labels not in training: {sorted(new_labels)}")
            all_training_labels.update(ds_labels)

        combined_labels = sorted(all_training_labels)
        print(f"Total retrieval set: {len(combined_labels)} unique labels")

        all_results['unseen_datasets'] = {}

        for name in model_names:
            print(f"\nEvaluating {name} (zero-shot)...")

            unseen_dataset = IMUPretrainingDataset(
                data_root='data',
                datasets=UNSEEN_DATASETS,
                split='val',
                max_sessions_per_dataset=MAX_SESSIONS_PER_DATASET,
                seed=42
            )
            unseen_loader = DataLoader(
                unseen_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                collate_fn=IMUPretrainingDataset.collate_fn
            )

            metrics = compute_metrics(
                loaded_models[name]['model'],
                loaded_models[name]['label_bank'],
                unseen_loader,
                device,
                combined_labels
            )
            all_results['unseen_datasets'][name] = metrics

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

        # Key metrics (same order as training)
        key_metrics = [
            'accuracy', 'mrr', 'positive_similarity', 'negative_similarity',
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

        # Per-dataset accuracy
        dataset_keys = [k for k in results[model_names[0]].keys() if k.endswith('_accuracy')]
        if dataset_keys:
            print(f"\n{'Per-dataset accuracy:':<25}")
            for key in sorted(dataset_keys):
                dataset_name = key.replace('_accuracy', '')
                print(f"  {dataset_name:<23}", end="")
                for name in model_names:
                    if name not in results:
                        continue
                    val = results[name].get(key, 0)
                    print(f"{val*100:>14.2f}%", end="")
                print()

    if 'training_datasets' in all_results:
        print_metrics_table(all_results['training_datasets'], "Training Datasets")

    if 'unseen_datasets' in all_results:
        print_metrics_table(all_results['unseen_datasets'], "Zero-Shot (Unseen Datasets)")

    # Save results
    results_path = output_dir / 'comparison_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")

    # Create plots
    if 'training_datasets' in all_results and len(model_names) > 0:
        create_comparison_plot(all_results['training_datasets'], model_names, output_dir, "Training Datasets")

    if 'unseen_datasets' in all_results and len(model_names) > 0:
        create_comparison_plot(all_results['unseen_datasets'], model_names, output_dir, "Zero-Shot")


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
