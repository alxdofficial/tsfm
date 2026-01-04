"""
Compare multiple semantic alignment models on validation/test sets.

Supports:
- Loading N models from checkpoint paths
- Testing with shuffled vs unshuffled channels
- Comparing performance metrics side by side
- Testing on unseen datasets for zero-shot generalization

Usage:
    # Compare on training datasets
    python training_scripts/imu_tool_pretraining/compare_models.py \
        --models original=training_output/semantic_alignment/original/best.pt \
                 shuffling_off=training_output/semantic_alignment/shufllingoff/best.pt \
        --channel_modes unshuffled \
        --output_dir training_output/model_comparison

    # Test zero-shot on unseen MotionSense dataset
    python training_scripts/imu_tool_pretraining/compare_models.py \
        --models original=training_output/semantic_alignment/original/best.pt \
                 shuffling_off=training_output/semantic_alignment/shufllingoff/best.pt \
        --unseen_datasets motionsense \
        --output_dir training_output/model_comparison_zeroshot
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'tools' / 'models'))

from torch.utils.data import DataLoader

from imu_activity_recognition_encoder.encoder import IMUActivityRecognitionEncoder
from imu_activity_recognition_encoder.semantic_alignment import SemanticAlignmentHead
from training_scripts.imu_tool_pretraining.semantic_alignment_train import SemanticAlignmentModel
from datasets.imu_pretraining_dataset.multi_dataset_loader import create_dataloaders, IMUPretrainingDataset
from training_scripts.imu_tool_pretraining.label_bank import LabelBank
from training_scripts.imu_tool_pretraining.evaluation_metrics import (
    LABEL_GROUPS, get_label_to_group_mapping, get_group_members
)
from datasets.imu_pretraining_dataset.label_augmentation import DATASET_CONFIGS


def create_eval_dataloader(
    data_root: str,
    datasets: List[str],
    batch_size: int,
    channel_augmentation: bool,
    max_sessions_per_dataset: Optional[int] = None,
    seed: int = 42
) -> DataLoader:
    """
    Create a dataloader for evaluation with explicit channel_augmentation control.

    Unlike create_dataloaders(), this function respects the channel_augmentation
    parameter for evaluation (val/test) data.
    """
    # Use 'val' split for evaluation
    dataset = IMUPretrainingDataset(
        data_root=data_root,
        datasets=datasets,
        split='val',
        max_sessions_per_dataset=max_sessions_per_dataset,
        channel_augmentation=channel_augmentation,
        seed=seed
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=IMUPretrainingDataset.collate_fn,
        pin_memory=True
    )

    return loader


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[SemanticAlignmentModel, dict]:
    """Load model from checkpoint, auto-detecting architecture from hyperparameters."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    epoch = checkpoint.get('epoch', 'unknown')

    # Read hyperparameters to get model config
    hyperparams_path = checkpoint_path.parent / 'hyperparameters.json'
    channel_projection = False
    channel_projection_hidden_dim = None
    channel_augmentation = True  # Default to True (original behavior)

    if hyperparams_path.exists():
        with open(hyperparams_path) as f:
            hyperparams = json.load(f)
        if 'channel_projection' in hyperparams:
            channel_projection = hyperparams['channel_projection'].get('enabled', False)
            channel_projection_hidden_dim = hyperparams['channel_projection'].get('hidden_dim', None)
        if 'data' in hyperparams:
            channel_augmentation = hyperparams['data'].get('channel_augmentation', True)

    # Create model
    encoder = IMUActivityRecognitionEncoder(
        d_model=384, num_heads=8, num_temporal_layers=4, dim_feedforward=1536,
        dropout=0.1, use_cross_channel=True, cnn_channels=[32, 64], cnn_kernel_sizes=[5],
        target_patch_size=64,
        channel_projection=channel_projection,
        channel_projection_hidden_dim=channel_projection_hidden_dim
    )
    semantic_head = SemanticAlignmentHead(
        d_model=384, d_model_fused=384, output_dim=384, num_bottlenecks=4,
        num_temporal_layers=2, num_heads=8, dim_feedforward=1536, dropout=0.1
    )
    model = SemanticAlignmentModel(encoder, semantic_head)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.train(False)
    model = model.to(device)

    model_info = {
        'epoch': epoch,
        'channel_projection': channel_projection,
        'channel_augmentation': channel_augmentation,
        'checkpoint_path': str(checkpoint_path)
    }

    return model, model_info


def compute_metrics(
    model: SemanticAlignmentModel,
    dataloader,
    label_bank: LabelBank,
    device: torch.device,
    all_labels: List[str]
) -> Dict[str, float]:
    """Compute metrics for a model on a dataset."""

    label_to_group = get_label_to_group_mapping()

    all_imu_embeddings = []
    all_text_embeddings = []
    all_gt_labels = []
    all_datasets = []

    model.train(False)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches", leave=False):
            data = batch['data'].to(device)
            channel_mask = batch['channel_mask'].to(device)
            label_texts = batch['label_texts']
            metadata = batch['metadata']

            sampling_rates = [m['sampling_rate_hz'] for m in metadata]
            patch_sizes = [m['patch_size_sec'] for m in metadata]
            channel_descriptions = [m['channel_descriptions'] for m in metadata]
            datasets = [m['dataset'] for m in metadata]

            # Get IMU embeddings
            imu_emb = model(data, channel_descriptions, channel_mask, sampling_rates, patch_sizes)
            imu_emb = F.normalize(imu_emb, dim=-1)

            # Get text embeddings
            text_emb = label_bank.encode(label_texts, normalize=True)

            all_imu_embeddings.append(imu_emb.cpu())
            all_text_embeddings.append(text_emb.cpu())
            all_gt_labels.extend(label_texts)
            all_datasets.extend(datasets)

    # Concatenate all embeddings
    imu_embeddings = torch.cat(all_imu_embeddings, dim=0)  # (N, D)
    text_embeddings = torch.cat(all_text_embeddings, dim=0)  # (N, D)

    # Encode all unique labels
    all_label_embeddings = label_bank.encode(all_labels, normalize=True).cpu()  # (L, D)

    # Compute similarity matrix: (N, L)
    similarity_matrix = imu_embeddings @ all_label_embeddings.T

    # Compute metrics
    metrics = {}
    num_labels = len(all_labels)

    # Recall@K
    for k in [1, 5, 10, 20]:
        if k > num_labels:
            continue  # Skip if k exceeds number of labels
        correct = 0
        correct_group = 0

        top_k_indices = similarity_matrix.topk(k, dim=1).indices

        for i, gt_label in enumerate(all_gt_labels):
            gt_group = label_to_group.get(gt_label, gt_label)
            predicted_labels = [all_labels[idx] for idx in top_k_indices[i]]

            # Exact match
            if gt_label in predicted_labels:
                correct += 1

            # Group match (synonyms)
            predicted_groups = [label_to_group.get(lbl, lbl) for lbl in predicted_labels]
            if gt_group in predicted_groups:
                correct_group += 1

        metrics[f'recall@{k}'] = correct / len(all_gt_labels)
        metrics[f'group_recall@{k}'] = correct_group / len(all_gt_labels)

    # Mean Reciprocal Rank
    mrr = 0
    mrr_group = 0
    for i, gt_label in enumerate(all_gt_labels):
        gt_group = label_to_group.get(gt_label, gt_label)
        sorted_indices = similarity_matrix[i].argsort(descending=True)

        # Exact MRR
        for rank, idx in enumerate(sorted_indices, 1):
            if all_labels[idx] == gt_label:
                mrr += 1 / rank
                break

        # Group MRR
        for rank, idx in enumerate(sorted_indices, 1):
            pred_group = label_to_group.get(all_labels[idx], all_labels[idx])
            if pred_group == gt_group:
                mrr_group += 1 / rank
                break

    metrics['mrr'] = mrr / len(all_gt_labels)
    metrics['group_mrr'] = mrr_group / len(all_gt_labels)

    # Per-dataset metrics
    dataset_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
    top_1_indices = similarity_matrix.argmax(dim=1)

    for i, (gt_label, dataset) in enumerate(zip(all_gt_labels, all_datasets)):
        gt_group = label_to_group.get(gt_label, gt_label)
        pred_label = all_labels[top_1_indices[i]]
        pred_group = label_to_group.get(pred_label, pred_label)

        dataset_metrics[dataset]['total'] += 1
        if gt_group == pred_group:
            dataset_metrics[dataset]['correct'] += 1

    for dataset, counts in dataset_metrics.items():
        metrics[f'{dataset}_group_acc'] = counts['correct'] / counts['total'] if counts['total'] > 0 else 0

    # IMU-text alignment (positive pair similarity)
    positive_sims = (imu_embeddings * text_embeddings).sum(dim=1)
    metrics['mean_positive_sim'] = positive_sims.mean().item()
    metrics['std_positive_sim'] = positive_sims.std().item()

    return metrics


def get_all_unique_labels(dataloader) -> List[str]:
    """Get all unique labels from a dataloader."""
    labels = set()
    for batch in dataloader:
        labels.update(batch['label_texts'])
    return sorted(list(labels))


def get_raw_labels_for_dataset(dataset_name: str) -> List[str]:
    """
    Get the canonical/raw activity labels for a dataset.

    For zero-shot evaluation, we use raw labels (not augmented variations)
    to ensure consistent comparison.
    """
    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower in DATASET_CONFIGS:
        # Return the keys from the synonyms dict (these are the canonical labels)
        return sorted(list(DATASET_CONFIGS[dataset_name_lower]['synonyms'].keys()))
    else:
        print(f"Warning: No augmentation config found for {dataset_name}, using 'unknown'")
        return ['unknown']


def run_comparison(
    models: Dict[str, str],
    channel_modes: List[str],
    datasets: List[str],
    output_dir: Path,
    batch_size: int = 32,
    unseen_only: bool = False,
    max_sessions: int = None,
    unseen_datasets: List[str] = None
):
    """Run comparison across models and channel modes."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all models
    print("\n" + "=" * 70)
    print("LOADING MODELS")
    print("=" * 70)

    loaded_models = {}
    for name, checkpoint_path in models.items():
        print(f"\nLoading {name}...")
        model, info = load_model(checkpoint_path, device)
        loaded_models[name] = {'model': model, 'info': info}
        print(f"  Epoch: {info['epoch']}")
        print(f"  Channel projection: {info['channel_projection']}")

    # Initialize label bank
    label_bank = LabelBank(model_name='all-MiniLM-L6-v2', device=device)

    # Results storage
    all_results = {}

    # Skip training dataset evaluation if unseen_only
    if unseen_only:
        print("\nSkipping training dataset evaluation (--unseen_only flag set)")
    else:
        for channel_mode in channel_modes:
            print("\n" + "=" * 70)
            print(f"TESTING WITH CHANNEL MODE: {channel_mode.upper()}")
            print("=" * 70)

            channel_augmentation = (channel_mode == 'shuffled')

            # Create dataloader
            _, val_loader, _ = create_dataloaders(
                data_root='data',
                datasets=datasets,
                batch_size=batch_size,
                max_sessions_per_dataset=max_sessions,
                channel_augmentation=channel_augmentation,
                num_workers=0,
                seed=42  # Fixed seed for reproducibility
            )

            # Get all unique labels
            print("Collecting unique labels...")
            all_labels = get_all_unique_labels(val_loader)
            print(f"Found {len(all_labels)} unique labels")

            # Recreate dataloader (consumed by label collection)
            _, val_loader, _ = create_dataloaders(
                data_root='data',
                datasets=datasets,
                batch_size=batch_size,
                max_sessions_per_dataset=max_sessions,
                channel_augmentation=channel_augmentation,
                num_workers=0,
                seed=42
            )

            all_results[channel_mode] = {}

            for model_name, model_data in loaded_models.items():
                print(f"\nTesting {model_name}...")
                metrics = compute_metrics(
                    model_data['model'],
                    val_loader,
                    label_bank,
                    device,
                    all_labels
                )
                all_results[channel_mode][model_name] = metrics

                # Recreate dataloader for next model
                _, val_loader, _ = create_dataloaders(
                    data_root='data',
                    datasets=datasets,
                    batch_size=batch_size,
                    max_sessions_per_dataset=max_sessions,
                    channel_augmentation=channel_augmentation,
                    num_workers=0,
                    seed=42
                )

    # Evaluate on unseen datasets for zero-shot generalization
    unseen_results = {}
    if unseen_datasets:
        print("\n" + "=" * 70)
        print("ZERO-SHOT GENERALIZATION ON UNSEEN DATASETS")
        print("=" * 70)
        print(f"Unseen datasets: {unseen_datasets}")
        print("\nEach model tested with its OWN training channel mode:")
        for model_name, model_data in loaded_models.items():
            ch_aug = model_data['info']['channel_augmentation']
            print(f"  {model_name}: channel_augmentation={ch_aug} ({'shuffled' if ch_aug else 'unshuffled'})")

        # For zero-shot evaluation, use RAW/canonical labels (not augmented)
        # This ensures fair evaluation against the same labels the model was trained with
        print("\nUsing raw/canonical labels for zero-shot evaluation...")
        combined_labels = []
        for ds_name in unseen_datasets:
            raw_labels = get_raw_labels_for_dataset(ds_name)
            combined_labels.extend(raw_labels)
            print(f"  {ds_name}: {len(raw_labels)} labels - {raw_labels}")
        combined_labels = sorted(set(combined_labels))
        print(f"Testing with {len(combined_labels)} total raw labels in retrieval set")

        for model_name, model_data in loaded_models.items():
            # Use this model's training channel_augmentation setting
            model_channel_aug = model_data['info']['channel_augmentation']
            mode_str = "shuffled" if model_channel_aug else "unshuffled"
            print(f"\nTesting {model_name} on unseen datasets (channels: {mode_str})...")

            # Create dataloader with this model's channel mode
            # NOTE: Using create_eval_dataloader instead of create_dataloaders
            # because create_dataloaders hardcodes channel_augmentation=False for val/test
            unseen_loader = create_eval_dataloader(
                data_root='data',
                datasets=unseen_datasets,
                batch_size=batch_size,
                channel_augmentation=model_channel_aug,
                max_sessions_per_dataset=max_sessions,
                seed=42
            )

            metrics = compute_metrics(
                model_data['model'],
                unseen_loader,
                label_bank,
                device,
                combined_labels
            )
            unseen_results[model_name] = metrics

    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    model_names = list(models.keys())

    # Only print training dataset results if we have them
    if all_results:
        for channel_mode in channel_modes:
            print(f"\n--- Channel Mode: {channel_mode.upper()} ---")
            print(f"{'Metric':<25}", end="")
            for name in model_names:
                print(f"{name:>15}", end="")
            print()
            print("-" * (25 + 15 * len(model_names)))

            # Key metrics to display
            key_metrics = ['recall@1', 'recall@5', 'recall@10', 'group_recall@1',
                           'group_recall@5', 'mrr', 'group_mrr', 'mean_positive_sim']

            for metric in key_metrics:
                print(f"{metric:<25}", end="")
                for name in model_names:
                    val = all_results[channel_mode][name].get(metric, 0)
                    if 'sim' in metric:
                        print(f"{val:>15.4f}", end="")
                    else:
                        print(f"{val*100:>14.2f}%", end="")
                print()

            # Per-dataset metrics
            print(f"\n{'Per-dataset group_acc:':<25}")
            dataset_metrics = [k for k in all_results[channel_mode][model_names[0]].keys()
                             if k.endswith('_group_acc')]
            for metric in sorted(dataset_metrics):
                dataset_name = metric.replace('_group_acc', '')
                print(f"  {dataset_name:<23}", end="")
                for name in model_names:
                    val = all_results[channel_mode][name].get(metric, 0)
                    print(f"{val*100:>14.2f}%", end="")
                print()

    # Print unseen dataset results (zero-shot generalization)
    if unseen_results:
        print("\n" + "=" * 70)
        print("ZERO-SHOT GENERALIZATION RESULTS")
        print("=" * 70)
        print(f"(Each model tested with its own training channel mode)")

        print(f"\n{'Metric':<25}", end="")
        for name in model_names:
            print(f"{name:>15}", end="")
        print()
        print("-" * (25 + 15 * len(model_names)))

        # Key metrics for zero-shot
        key_metrics = ['recall@1', 'recall@5', 'recall@10', 'group_recall@1',
                       'group_recall@5', 'mrr', 'group_mrr', 'mean_positive_sim']

        for metric in key_metrics:
            # Skip metrics that don't exist for any model
            if all(metric not in unseen_results[name] for name in model_names):
                continue
            print(f"{metric:<25}", end="")
            for name in model_names:
                val = unseen_results[name].get(metric, None)
                if val is None:
                    print(f"{'N/A':>15}", end="")
                elif 'sim' in metric:
                    print(f"{val:>15.4f}", end="")
                else:
                    print(f"{val*100:>14.2f}%", end="")
            print()

        # Per unseen dataset metrics
        print(f"\n{'Per-dataset group_acc:':<25}")
        dataset_metrics = [k for k in unseen_results[model_names[0]].keys()
                         if k.endswith('_group_acc')]
        for metric in sorted(dataset_metrics):
            dataset_name = metric.replace('_group_acc', '')
            print(f"  {dataset_name:<23}", end="")
            for name in model_names:
                val = unseen_results[name].get(metric, 0)
                print(f"{val*100:>14.2f}%", end="")
            print()

    # Save results to JSON
    combined_results = {
        'training_datasets': all_results,
        'unseen_datasets': unseen_results if unseen_results else {}
    }
    results_path = output_dir / 'comparison_results.json'
    with open(results_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Create comparison plot only if we have training dataset results
    if all_results:
        create_comparison_plot(all_results, model_names, channel_modes, output_dir)

    # Create zero-shot comparison plot if we have unseen results
    if unseen_results:
        create_zeroshot_plot(unseen_results, model_names, unseen_datasets, output_dir)

    return combined_results


def create_comparison_plot(
    results: Dict,
    model_names: List[str],
    channel_modes: List[str],
    output_dir: Path
):
    """Create bar chart comparing models."""

    metrics_to_plot = ['group_recall@1', 'group_recall@5', 'group_mrr', 'mean_positive_sim']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    x = np.arange(len(channel_modes))
    width = 0.8 / len(model_names)

    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

    for ax_idx, metric in enumerate(metrics_to_plot):
        ax = axes[ax_idx]

        for i, model_name in enumerate(model_names):
            values = [results[mode][model_name].get(metric, 0) for mode in channel_modes]
            if 'sim' not in metric:
                values = [v * 100 for v in values]  # Convert to percentage

            offset = (i - len(model_names) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=model_name, color=colors[i])

            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.1f}' if 'sim' not in metric else f'{val:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Channel Mode')
        ax.set_ylabel('Percentage (%)' if 'sim' not in metric else 'Similarity')
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(channel_modes)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Model Comparison Across Channel Modes', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = output_dir / 'comparison_plot.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to {plot_path}")
    plt.close()


def create_zeroshot_plot(
    results: Dict,
    model_names: List[str],
    unseen_datasets: List[str],
    output_dir: Path
):
    """Create bar chart comparing zero-shot performance on unseen datasets."""

    metrics_to_plot = ['group_recall@1', 'group_recall@5', 'group_mrr', 'mean_positive_sim']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    x = np.arange(len(model_names))
    width = 0.6

    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

    for ax_idx, metric in enumerate(metrics_to_plot):
        ax = axes[ax_idx]

        values = [results[model_name].get(metric, 0) for model_name in model_names]
        if 'sim' not in metric:
            values = [v * 100 for v in values]

        bars = ax.bar(x, values, width, color=colors)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}' if 'sim' not in metric else f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel('Percentage (%)' if 'sim' not in metric else 'Similarity')
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.grid(axis='y', alpha=0.3)

    dataset_str = ', '.join(unseen_datasets)
    plt.suptitle(f'Zero-Shot Generalization on Unseen Datasets\n({dataset_str})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = output_dir / 'zeroshot_comparison_plot.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Zero-shot comparison plot saved to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare multiple semantic alignment models')
    parser.add_argument('--models', nargs='+', required=True,
                        help='Model specifications as name=checkpoint_path pairs')
    parser.add_argument('--channel_modes', nargs='+', default=['unshuffled', 'shuffled'],
                        choices=['unshuffled', 'shuffled'],
                        help='Channel modes to test')
    parser.add_argument('--datasets', nargs='+',
                        default=['uci_har', 'hhar', 'mhealth', 'pamap2', 'wisdm', 'unimib_shar'],
                        help='Datasets to test on')
    parser.add_argument('--output_dir', type=str, default='training_output/model_comparison',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--max_sessions', type=int, default=None,
                        help='Max sessions per dataset (None = all)')
    parser.add_argument('--unseen_datasets', nargs='+', default=[],
                        help='Unseen datasets for zero-shot generalization testing (e.g., motionsense)')
    parser.add_argument('--unseen_only', action='store_true',
                        help='Skip evaluation on training datasets, only test on unseen datasets')

    args = parser.parse_args()

    # Parse model specifications
    models = {}
    for spec in args.models:
        if '=' not in spec:
            raise ValueError(f"Invalid model spec '{spec}'. Use format: name=checkpoint_path")
        name, path = spec.split('=', 1)
        models[name] = path

    print("Models to compare:")
    for name, path in models.items():
        print(f"  {name}: {path}")

    run_comparison(
        models=models,
        channel_modes=args.channel_modes,
        datasets=args.datasets,
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        max_sessions=args.max_sessions,
        unseen_datasets=args.unseen_datasets,
        unseen_only=args.unseen_only
    )


if __name__ == '__main__':
    main()
