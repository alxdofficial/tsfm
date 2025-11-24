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

        # Storage for metrics - dynamically populated
        self.metrics = {
            'epoch': {},  # {metric_name: [(step, value), ...]}
            'batch': {},  # {metric_name: [(step, value), ...]}
            'train_per_dataset': {},  # {dataset: {metric: [(step, value), ...]}}
            'val_per_dataset': {},
        }

        # Track current epoch and batch
        self.current_epoch = 0
        self.global_batch = 0  # Global batch counter across all epochs

    def add_scalar(self, tag: str, value: float, step: int):
        """
        Add a scalar metric (mimics TensorBoard API).

        Args:
            tag: Metric name (e.g., 'epoch/train_loss', 'batch/stage1_loss', 'train_per_dataset/uci_har/loss')
            value: Metric value
            step: Step number (epoch or batch)
        """
        # Parse tag
        parts = tag.split('/')

        if parts[0] == 'epoch':
            # Epoch-level metrics - dynamically add if not exists
            metric_name = parts[1]
            if metric_name not in self.metrics['epoch']:
                self.metrics['epoch'][metric_name] = []
            self.metrics['epoch'][metric_name].append((step, value))

        elif parts[0] == 'batch':
            # Batch-level training metrics - dynamically add if not exists
            metric_name = parts[1]
            if metric_name not in self.metrics['batch']:
                self.metrics['batch'][metric_name] = []
            self.metrics['batch'][metric_name].append((step, value))

        elif parts[0] == 'train_per_dataset':
            # Per-dataset training metrics - dynamically add dataset and metric
            dataset_name = parts[1]
            metric_name = parts[2]

            if dataset_name not in self.metrics['train_per_dataset']:
                self.metrics['train_per_dataset'][dataset_name] = {}

            if metric_name not in self.metrics['train_per_dataset'][dataset_name]:
                self.metrics['train_per_dataset'][dataset_name][metric_name] = []

            self.metrics['train_per_dataset'][dataset_name][metric_name].append((step, value))

        elif parts[0] == 'val_per_dataset':
            # Per-dataset validation metrics - dynamically add dataset and metric
            dataset_name = parts[1]
            metric_name = parts[2]

            if dataset_name not in self.metrics['val_per_dataset']:
                self.metrics['val_per_dataset'][dataset_name] = {}

            if metric_name not in self.metrics['val_per_dataset'][dataset_name]:
                self.metrics['val_per_dataset'][dataset_name][metric_name] = []

            self.metrics['val_per_dataset'][dataset_name][metric_name].append((step, value))

    def save_metrics(self):
        """Save metrics to JSON file."""
        metrics_file = self.output_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def plot_all(self):
        """Generate all plots and save as PNG files."""
        self._plot_batch_training_losses()  # Real-time training convergence
        self._plot_debug_metrics()  # Real-time debug metrics (gradients, collapse indicators)
        self._plot_overall_loss()  # Epoch-level train vs val
        self._plot_loss_components()  # Epoch-level MAE vs Contrastive
        self._plot_per_dataset_losses()
        self._plot_learning_rate()
        self.save_metrics()

    def _plot_batch_training_losses(self):
        """Plot batch-level training losses for real-time convergence monitoring."""
        # Find any loss metric in batch
        loss_metrics = [k for k in self.metrics['batch'].keys() if 'loss' in k.lower()]
        if not loss_metrics:
            return

        # Plot the first loss metric as main plot
        main_loss = loss_metrics[0]
        batches, losses = zip(*self.metrics['batch'][main_loss])

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(batches, losses, linewidth=1, alpha=0.7, label=main_loss)

        # Plot other loss metrics if they exist
        for metric in loss_metrics[1:]:
            if self.metrics['batch'][metric]:
                b, v = zip(*self.metrics['batch'][metric])
                ax.plot(b, v, linewidth=1, alpha=0.7, label=metric)

        ax.set_xlabel('Batch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss (Batch-level)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'batch_training_losses.png', dpi=150)
        plt.close(fig)

    def _plot_debug_metrics(self):
        """Plot batch-level debug metrics for monitoring gradient flow and representation collapse."""
        # Find debug metrics in batch (look for metrics starting with 'debug_')
        debug_metrics = [k for k in self.metrics['batch'].keys() if k.startswith('debug_')]
        if not debug_metrics:
            return

        # Group metrics by category
        collapse_metrics = [k for k in debug_metrics if 'std' in k or 'diversity' in k]
        gradient_metrics = [k for k in debug_metrics if 'grad_norm' in k]
        similarity_metrics = [k for k in debug_metrics if 'sim' in k]

        # Create 3 subplots: collapse indicators, gradient norms, similarity
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # Plot 1: Representation collapse indicators (std, diversity)
        ax1 = axes[0]
        for metric in collapse_metrics:
            if self.metrics['batch'][metric]:
                batches, values = zip(*self.metrics['batch'][metric])
                label = metric.replace('debug_train_', '').replace('debug_', '')
                ax1.plot(batches, values, linewidth=1.5, alpha=0.8, label=label, marker='o', markersize=2)

        ax1.set_xlabel('Batch', fontsize=11)
        ax1.set_ylabel('Value', fontsize=11)
        ax1.set_title('Representation Collapse Indicators (Higher = Better)', fontsize=13, fontweight='bold')
        ax1.axhline(y=0.1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Collapse threshold (0.1)')
        if collapse_metrics:
            ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Gradient norms
        ax2 = axes[1]
        for metric in gradient_metrics:
            if self.metrics['batch'][metric]:
                batches, values = zip(*self.metrics['batch'][metric])
                label = metric.replace('debug_train_', '').replace('_grad_norm', '').replace('debug_', '')
                ax2.plot(batches, values, linewidth=1.5, alpha=0.8, label=label, marker='o', markersize=2)

        ax2.set_xlabel('Batch', fontsize=11)
        ax2.set_ylabel('Gradient Norm', fontsize=11)
        ax2.set_title('Gradient Flow (Higher = Better)', fontsize=13, fontweight='bold')
        ax2.axhline(y=0.01, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Vanishing threshold (0.01)')
        if gradient_metrics:
            ax2.legend(fontsize=9, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')  # Log scale for gradient magnitudes

        # Plot 3: Similarity metrics
        ax3 = axes[2]
        for metric in similarity_metrics:
            if self.metrics['batch'][metric]:
                batches, values = zip(*self.metrics['batch'][metric])
                label = metric.replace('debug_train_', '').replace('debug_', '')
                ax3.plot(batches, values, linewidth=1.5, alpha=0.8, label=label, marker='o', markersize=2)

        ax3.set_xlabel('Batch', fontsize=11)
        ax3.set_ylabel('Similarity', fontsize=11)
        ax3.set_title('Contrastive Similarity Metrics', fontsize=13, fontweight='bold')
        if similarity_metrics:
            ax3.legend(fontsize=9, loc='best')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'debug_metrics.png', dpi=150)
        plt.close(fig)

    def _plot_overall_loss(self):
        """Plot overall train/val loss over epochs."""
        # Find loss metrics
        train_loss = self.metrics['epoch'].get('train_loss')
        val_loss = self.metrics['epoch'].get('val_loss')

        if not train_loss and not val_loss:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract and plot data
        if train_loss:
            epochs_train, losses_train = zip(*train_loss)
            ax.plot(epochs_train, losses_train, label='Train Loss', marker='o', linewidth=2)

        if val_loss:
            epochs_val, losses_val = zip(*val_loss)
            ax.plot(epochs_val, losses_val, label='Val Loss', marker='s', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'overall_loss.png', dpi=150)
        plt.close(fig)

    def _plot_loss_components(self):
        """Plot all non-loss metrics (accuracy, similarity, etc.) over epochs."""
        # Get all metrics except main loss
        train_metrics = [k for k in self.metrics['epoch'].keys()
                        if k.startswith('train_') and k != 'train_loss']
        val_metrics = [k for k in self.metrics['epoch'].keys()
                      if k.startswith('val_') and k != 'val_loss']

        if not train_metrics and not val_metrics:
            return

        # Create plots for each metric type
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Plot training metrics
        for metric in train_metrics:
            if self.metrics['epoch'][metric]:
                epochs, values = zip(*self.metrics['epoch'][metric])
                axes[0].plot(epochs, values, label=metric.replace('train_', ''), marker='o', linewidth=2)

        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Value', fontsize=12)
        axes[0].set_title('Training Metrics', fontsize=14, fontweight='bold')
        if train_metrics:
            axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Plot validation metrics
        for metric in val_metrics:
            if self.metrics['epoch'][metric]:
                epochs, values = zip(*self.metrics['epoch'][metric])
                axes[1].plot(epochs, values, label=metric.replace('val_', ''), marker='s', linewidth=2)

        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Value', fontsize=12)
        axes[1].set_title('Validation Metrics', fontsize=14, fontweight='bold')
        if val_metrics:
            axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics.png', dpi=150)
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
        lr_data = self.metrics['epoch'].get('lr')
        if not lr_data:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        epochs, lrs = zip(*lr_data)
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
        print(f"  - batch_training_losses.png (real-time training convergence)")
        print(f"  - debug_metrics.png (gradients, collapse indicators, similarity)")
        print(f"  - overall_loss.png (epoch-level train vs val)")
        print(f"  - loss_components.png (epoch-level MAE vs contrastive)")
        print(f"  - per_dataset_losses.png")
        print(f"  - learning_rate.png")
        print(f"  - dataset_*_detail.png (per dataset)")
        print(f"  - metrics.json (raw data)")
