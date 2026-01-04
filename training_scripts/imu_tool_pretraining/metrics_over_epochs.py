"""
Compute semantic alignment metrics across multiple epochs.

Usage:
    python training_scripts/imu_tool_pretraining/metrics_over_epochs.py \
        --run_dir training_output/semantic_alignment/20251202_105310 \
        --epochs 1 10 30 60
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'tools' / 'models'))

from evaluation_metrics import (
    compute_semantic_recall,
    compute_embedding_quality_metrics
)


def load_checkpoint_and_compute_metrics(
    checkpoint_path: str,
    val_loader,
    label_bank,
    device: torch.device,
    k_values: list = [1, 5, 10, 20, 50]
) -> dict:
    """Load a checkpoint and compute all metrics."""
    from imu_activity_recognition_encoder.encoder import IMUActivityRecognitionEncoder
    from imu_activity_recognition_encoder.semantic_alignment import SemanticAlignmentHead
    from training_scripts.imu_tool_pretraining.semantic_alignment_train import SemanticAlignmentModel

    # Load checkpoint
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"\n{'='*60}")
    print(f"Computing metrics for epoch {epoch}")
    print(f"{'='*60}")

    # Read hyperparameters to get model config (like compare_models.py)
    hyperparams_path = checkpoint_path.parent / 'hyperparameters.json'
    channel_projection = False  # Default
    channel_projection_hidden_dim = None

    if hyperparams_path.exists():
        with open(hyperparams_path) as f:
            hyperparams = json.load(f)
        if 'channel_projection' in hyperparams:
            channel_projection = hyperparams['channel_projection'].get('enabled', False)
            channel_projection_hidden_dim = hyperparams['channel_projection'].get('hidden_dim', None)
        print(f"  Loaded hyperparameters: channel_projection={channel_projection}")
    else:
        print(f"  Warning: No hyperparameters.json found, using defaults")

    # Create model with correct architecture
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

    # Collect embeddings
    all_imu_embeddings = []
    all_text_embeddings = []
    all_labels = []

    total_batches = len(val_loader)
    print(f"Collecting embeddings from {total_batches} batches...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            data = batch['data'].to(device)
            channel_mask = batch['channel_mask'].to(device)
            label_texts = batch['label_texts']
            metadata = batch['metadata']

            sampling_rates = [m['sampling_rate_hz'] for m in metadata]
            patch_sizes = [m['patch_size_sec'] for m in metadata]
            channel_descriptions = [m['channel_descriptions'] for m in metadata]

            imu_emb = model(data, channel_descriptions, channel_mask, sampling_rates, patch_sizes)
            text_emb = label_bank.encode(label_texts, normalize=True)

            all_imu_embeddings.append(imu_emb.cpu())
            all_text_embeddings.append(text_emb.cpu())
            all_labels.extend(label_texts)

            # Progress indicator
            if (batch_idx + 1) % 20 == 0 or batch_idx == total_batches - 1:
                pct = (batch_idx + 1) / total_batches * 100
                print(f"  [{batch_idx + 1}/{total_batches}] {pct:.0f}% complete")

    all_imu_embeddings = torch.cat(all_imu_embeddings, dim=0)
    all_text_embeddings = torch.cat(all_text_embeddings, dim=0)

    print(f"Collected {len(all_labels)} samples")

    # Get unique labels for classification-style evaluation
    # (retrieve against L unique labels, not N sample embeddings)
    unique_labels = sorted(set(all_labels))
    unique_label_embeddings = label_bank.encode(unique_labels, normalize=True).cpu()
    print(f"Using {len(unique_labels)} unique labels for retrieval")

    # Compute metrics
    print("Computing metrics...")
    results = {'epoch': epoch}

    # Semantic recall with groups (classification-style: retrieve from unique labels)
    print("  - Semantic recall (with groups)...")
    semantic_recall = compute_semantic_recall(
        all_imu_embeddings, unique_label_embeddings,
        all_labels, unique_labels, k_values, use_groups=True
    )
    results.update(semantic_recall)

    # Exact label recall
    print("  - Exact label recall...")
    exact_recall = compute_semantic_recall(
        all_imu_embeddings, unique_label_embeddings,
        all_labels, unique_labels, k_values, use_groups=False
    )
    results.update({f'exact_{k}': v for k, v in exact_recall.items()})

    # Embedding quality
    print("  - Embedding quality metrics...")
    quality = compute_embedding_quality_metrics(
        all_imu_embeddings, all_text_embeddings, all_labels, use_groups=True
    )
    results.update(quality)

    # Print summary
    print(f"\nSemantic Recall@1: {results.get('semantic_recall@1', 0)*100:.2f}%")
    print(f"Semantic Recall@5: {results.get('semantic_recall@5', 0)*100:.2f}%")
    print(f"Exact Recall@1: {results.get('exact_semantic_recall@1', 0)*100:.2f}%")
    print(f"IMU-Text Alignment: {results.get('imu_text_alignment', 0):.4f}")
    print(f"Class Separability: {results.get('class_separability', 0):.4f}")

    return results


def plot_metrics_over_epochs(epochs_results: dict, output_path: str):
    """Create plots showing metrics evolution over epochs."""
    epochs = sorted(epochs_results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Semantic Recall @1
    ax = axes[0, 0]
    key = 'semantic_recall@1'
    values = [epochs_results[e].get(key, 0) * 100 for e in epochs]
    ax.plot(epochs, values, 'o-', label='Semantic R@1', linewidth=2, markersize=8, color='tab:blue')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Semantic Recall@1 (%)', fontsize=12)
    ax.set_title('Semantic Recall@1 (with synonym groups)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)

    # 2. Exact Recall @1
    ax = axes[0, 1]
    key = 'exact_semantic_recall@1'
    values = [epochs_results[e].get(key, 0) * 100 for e in epochs]
    ax.plot(epochs, values, 'o-', label='Exact R@1', linewidth=2, markersize=8, color='tab:orange')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Exact Recall@1 (%)', fontsize=12)
    ax.set_title('Exact Label Recall@1 (no grouping)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)

    # 3. Embedding quality metrics
    ax = axes[1, 0]
    metrics_to_plot = [
        ('imu_text_alignment', 'IMU-Text Alignment', 'tab:blue'),
        ('class_separability', 'Class Separability', 'tab:orange'),
    ]
    for metric, label, color in metrics_to_plot:
        values = [epochs_results[e].get(metric, 0) for e in epochs]
        ax.plot(epochs, values, 'o-', label=label, linewidth=2, markersize=8, color=color)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Embedding Quality Metrics', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)

    # 4. Distance metrics
    ax = axes[1, 1]
    metrics_to_plot = [
        ('cross_modal_gap', 'Cross-modal Gap', 'tab:red'),
        ('intra_class_distance', 'Intra-class Dist', 'tab:green'),
        ('inter_class_distance', 'Inter-class Dist', 'tab:purple'),
    ]
    for metric, label, color in metrics_to_plot:
        values = [epochs_results[e].get(metric, 0) for e in epochs]
        ax.plot(epochs, values, 'o-', label=label, linewidth=2, markersize=8, color=color)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.set_title('Distance Metrics', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)

    plt.suptitle('Semantic Alignment Metrics Over Training', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved metrics plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compute metrics across epochs')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Training run directory containing checkpoints')
    parser.add_argument('--epochs', type=int, nargs='+', default=[1, 10, 30, 60],
                        help='Epochs to process')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (defaults to run_dir/plots)')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load validation data loader
    from datasets.imu_pretraining_dataset.multi_dataset_loader import create_dataloaders
    from training_scripts.imu_tool_pretraining.label_bank import LabelBank

    _, val_loader, _ = create_dataloaders(
        data_root='data',
        datasets=['uci_har', 'hhar', 'mhealth', 'pamap2', 'wisdm', 'unimib_shar'],
        batch_size=64,
        max_sessions_per_dataset=10000,
        num_workers=4
    )

    label_bank = LabelBank(model_name='all-MiniLM-L6-v2', device=device)

    # Process each epoch
    epochs_results = {}
    valid_epochs = [e for e in args.epochs if (run_dir / f'epoch_{e}.pt').exists()]
    total_epochs = len(valid_epochs)

    print(f"\nWill process {total_epochs} epochs: {valid_epochs}")

    for epoch_idx, epoch in enumerate(args.epochs):
        checkpoint_path = run_dir / f'epoch_{epoch}.pt'
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint for epoch {epoch} not found at {checkpoint_path}")
            continue

        print(f"\n[Epoch {epoch_idx + 1}/{total_epochs}]")
        results = load_checkpoint_and_compute_metrics(
            str(checkpoint_path), val_loader, label_bank, device
        )
        # Keep only serializable items for JSON
        serializable_results = {k: v for k, v in results.items()
                               if not isinstance(v, (np.ndarray, dict))}
        epochs_results[epoch] = serializable_results

    if not epochs_results:
        print("Error: No checkpoints found!")
        return

    # Save results to JSON
    results_path = output_dir / 'epoch_metrics.json'
    with open(results_path, 'w') as f:
        json.dump(epochs_results, f, indent=2)
    print(f"\nSaved metrics to {results_path}")

    # Plot metrics
    plot_metrics_over_epochs(epochs_results, str(output_dir / 'metrics_over_epochs.png'))

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Epoch':>6} | {'Sem R@1':>8} | {'Sem R@5':>8} | {'Exact R@1':>9} | {'Alignment':>10} | {'Separability':>12}")
    print("-"*80)
    for epoch in sorted(epochs_results.keys()):
        r = epochs_results[epoch]
        print(f"{epoch:>6} | {r.get('semantic_recall@1', 0)*100:>7.2f}% | "
              f"{r.get('semantic_recall@5', 0)*100:>7.2f}% | "
              f"{r.get('exact_semantic_recall@1', 0)*100:>8.2f}% | "
              f"{r.get('imu_text_alignment', 0):>10.4f} | "
              f"{r.get('class_separability', 0):>12.4f}")
    print("="*80)


if __name__ == '__main__':
    main()
