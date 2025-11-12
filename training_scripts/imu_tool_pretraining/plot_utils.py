"""
Plotting utilities for training metrics.

Replaces TensorBoard with local PNG plots saved to disk.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


class TrainingPlotter:
    """Collects training metrics and generates PNG plots."""

    def __init__(self, output_dir: Path):
        """
        Args:
            output_dir: Directory to save plots and metrics
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage for metrics
        self.metrics = {
            'epoch': {
                'train_loss': [],
                'val_loss': [],
                'train_mae_loss': [],
                'val_mae_loss': [],
                'train_contrastive_loss': [],
                'val_contrastive_loss': [],
                'lr': [],
            },
            'train_per_dataset': {},  # {dataset: {'loss': [], 'mae': [], 'contrastive': []}}
            'val_per_dataset': {},
        }

        # Track current epoch
        self.current_epoch = 0

    def add_scalar(self, tag: str, value: float, step: int):
        """
        Add a scalar metric (mimics TensorBoard API).

        Args:
            tag: Metric name (e.g., 'epoch/train_loss', 'train_per_dataset/uci_har/loss')
            value: Metric value
            step: Step number (epoch or batch)
        """
        # Parse tag
        parts = tag.split('/')

        if parts[0] == 'epoch':
            # Epoch-level metrics
            metric_name = parts[1]
            if metric_name in self.metrics['epoch']:
                self.metrics['epoch'][metric_name].append((step, value))

        elif parts[0] == 'train_per_dataset':
            # Per-dataset training metrics
            dataset_name = parts[1]
            metric_name = parts[2]

            if dataset_name not in self.metrics['train_per_dataset']:
                self.metrics['train_per_dataset'][dataset_name] = {
                    'loss': [],
                    'mae_loss': [],
                    'contrastive_loss': []
                }

            self.metrics['train_per_dataset'][dataset_name][metric_name].append((step, value))

        elif parts[0] == 'val_per_dataset':
            # Per-dataset validation metrics
            dataset_name = parts[1]
            metric_name = parts[2]

            if dataset_name not in self.metrics['val_per_dataset']:
                self.metrics['val_per_dataset'][dataset_name] = {
                    'loss': [],
                    'mae_loss': [],
                    'contrastive_loss': []
                }

            self.metrics['val_per_dataset'][dataset_name][metric_name].append((step, value))

    def save_metrics(self):
        """Save metrics to JSON file."""
        metrics_file = self.output_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def plot_all(self):
        """Generate all plots and save as PNG files."""
        self._plot_overall_loss()
        self._plot_loss_components()
        self._plot_per_dataset_losses()
        self._plot_learning_rate()
        self.save_metrics()

    def _plot_overall_loss(self):
        """Plot overall train/val loss over epochs."""
        # Skip if no data yet
        if not self.metrics['epoch']['train_loss'] and not self.metrics['epoch']['val_loss']:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract data
        if self.metrics['epoch']['train_loss']:
            epochs_train, train_loss = zip(*self.metrics['epoch']['train_loss'])
            ax.plot(epochs_train, train_loss, label='Train Loss', marker='o', linewidth=2)

        if self.metrics['epoch']['val_loss']:
            epochs_val, val_loss = zip(*self.metrics['epoch']['val_loss'])
            ax.plot(epochs_val, val_loss, label='Val Loss', marker='s', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        if self.metrics['epoch']['train_loss'] or self.metrics['epoch']['val_loss']:
            ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'overall_loss.png', dpi=150)
        plt.close(fig)

    def _plot_loss_components(self):
        """Plot MAE loss vs Contrastive loss over epochs."""
        # Skip if no data yet
        has_train_data = self.metrics['epoch']['train_mae_loss'] or self.metrics['epoch']['train_contrastive_loss']
        has_val_data = self.metrics['epoch']['val_mae_loss'] or self.metrics['epoch']['val_contrastive_loss']
        if not has_train_data and not has_val_data:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Train losses
        if self.metrics['epoch']['train_mae_loss']:
            epochs, mae_loss = zip(*self.metrics['epoch']['train_mae_loss'])
            ax1.plot(epochs, mae_loss, label='MAE Loss', marker='o', linewidth=2, color='tab:blue')

        if self.metrics['epoch']['train_contrastive_loss']:
            epochs, contrast_loss = zip(*self.metrics['epoch']['train_contrastive_loss'])
            ax1.plot(epochs, contrast_loss, label='Contrastive Loss', marker='s', linewidth=2, color='tab:orange')

        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Loss Components', fontsize=14, fontweight='bold')
        if has_train_data:
            ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Validation losses
        if self.metrics['epoch']['val_mae_loss']:
            epochs, mae_loss = zip(*self.metrics['epoch']['val_mae_loss'])
            ax2.plot(epochs, mae_loss, label='MAE Loss', marker='o', linewidth=2, color='tab:blue')

        if self.metrics['epoch']['val_contrastive_loss']:
            epochs, contrast_loss = zip(*self.metrics['epoch']['val_contrastive_loss'])
            ax2.plot(epochs, contrast_loss, label='Contrastive Loss', marker='s', linewidth=2, color='tab:orange')

        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Validation Loss Components', fontsize=14, fontweight='bold')
        if has_val_data:
            ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'loss_components.png', dpi=150)
        plt.close(fig)

    def _plot_per_dataset_losses(self):
        """Plot per-dataset losses over epochs."""
        # Get all datasets
        datasets_train = list(self.metrics['train_per_dataset'].keys())
        datasets_val = list(self.metrics['val_per_dataset'].keys())
        all_datasets = sorted(set(datasets_train + datasets_val))

        if not all_datasets:
            return

        # Create subplots: one row for train, one for val
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Train per-dataset losses
        ax_train = axes[0]
        for dataset in all_datasets:
            if dataset in self.metrics['train_per_dataset']:
                data = self.metrics['train_per_dataset'][dataset]['loss']
                if data:
                    epochs, losses = zip(*data)
                    ax_train.plot(epochs, losses, label=dataset, marker='o', linewidth=2)

        ax_train.set_xlabel('Epoch', fontsize=12)
        ax_train.set_ylabel('Loss', fontsize=12)
        ax_train.set_title('Training Loss per Dataset', fontsize=14, fontweight='bold')
        ax_train.legend(fontsize=10)
        ax_train.grid(True, alpha=0.3)

        # Validation per-dataset losses
        ax_val = axes[1]
        for dataset in all_datasets:
            if dataset in self.metrics['val_per_dataset']:
                data = self.metrics['val_per_dataset'][dataset]['loss']
                if data:
                    epochs, losses = zip(*data)
                    ax_val.plot(epochs, losses, label=dataset, marker='s', linewidth=2)

        ax_val.set_xlabel('Epoch', fontsize=12)
        ax_val.set_ylabel('Loss', fontsize=12)
        ax_val.set_title('Validation Loss per Dataset', fontsize=14, fontweight='bold')
        ax_val.legend(fontsize=10)
        ax_val.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'per_dataset_losses.png', dpi=150)
        plt.close(fig)

        # Also create detailed per-dataset plots (MAE + Contrastive)
        for dataset in all_datasets:
            self._plot_dataset_detail(dataset)

    def _plot_dataset_detail(self, dataset: str):
        """Plot detailed metrics for a specific dataset."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Train MAE
        if dataset in self.metrics['train_per_dataset']:
            data = self.metrics['train_per_dataset'][dataset].get('mae_loss', [])
            if data:
                epochs, losses = zip(*data)
                axes[0, 0].plot(epochs, losses, marker='o', linewidth=2, color='tab:blue')
        axes[0, 0].set_title(f'{dataset} - Train MAE Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('MAE Loss')
        axes[0, 0].grid(True, alpha=0.3)

        # Train Contrastive
        if dataset in self.metrics['train_per_dataset']:
            data = self.metrics['train_per_dataset'][dataset].get('contrastive_loss', [])
            if data:
                epochs, losses = zip(*data)
                axes[0, 1].plot(epochs, losses, marker='o', linewidth=2, color='tab:orange')
        axes[0, 1].set_title(f'{dataset} - Train Contrastive Loss', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Contrastive Loss')
        axes[0, 1].grid(True, alpha=0.3)

        # Val MAE
        if dataset in self.metrics['val_per_dataset']:
            data = self.metrics['val_per_dataset'][dataset].get('mae_loss', [])
            if data:
                epochs, losses = zip(*data)
                axes[1, 0].plot(epochs, losses, marker='s', linewidth=2, color='tab:blue')
        axes[1, 0].set_title(f'{dataset} - Val MAE Loss', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE Loss')
        axes[1, 0].grid(True, alpha=0.3)

        # Val Contrastive
        if dataset in self.metrics['val_per_dataset']:
            data = self.metrics['val_per_dataset'][dataset].get('contrastive_loss', [])
            if data:
                epochs, losses = zip(*data)
                axes[1, 1].plot(epochs, losses, marker='s', linewidth=2, color='tab:orange')
        axes[1, 1].set_title(f'{dataset} - Val Contrastive Loss', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Contrastive Loss')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'dataset_{dataset}_detail.png', dpi=150)
        plt.close(fig)

    def _plot_learning_rate(self):
        """Plot learning rate schedule."""
        if not self.metrics['epoch']['lr']:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        epochs, lrs = zip(*self.metrics['epoch']['lr'])
        ax.plot(epochs, lrs, linewidth=2, color='tab:green')

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale for better visualization

        plt.tight_layout()
        plt.savefig(self.output_dir / 'learning_rate.png', dpi=150)
        plt.close(fig)

    def close(self):
        """Finalize plotting (mimics TensorBoard API)."""
        self.plot_all()
        print(f"\n✓ Plots saved to {self.output_dir}")
        print(f"  - overall_loss.png")
        print(f"  - loss_components.png")
        print(f"  - per_dataset_losses.png")
        print(f"  - learning_rate.png")
        print(f"  - dataset_*_detail.png (per dataset)")
        print(f"  - metrics.json (raw data)")
