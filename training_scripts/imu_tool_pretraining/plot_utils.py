"""
Plotting utilities for training metrics.

Replaces TensorBoard with local PNG plots saved to disk.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np


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

    def load_metrics(self, metrics_file: Path = None):
        """Load metrics from JSON file to resume plotting.

        Args:
            metrics_file: Path to metrics.json. If None, uses output_dir/metrics.json
        """
        if metrics_file is None:
            metrics_file = self.output_dir / 'metrics.json'

        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                self.metrics = json.load(f)
            print(f"✓ Loaded existing metrics from {metrics_file}")

            # Print summary of loaded data
            n_epochs = len(self.metrics.get('epoch', {}).get('train_loss', []))
            n_batches = len(self.metrics.get('batch', {}).get('train_loss', []))
            print(f"  Resuming from: {n_epochs} epochs, {n_batches} batch records")
            return True
        return False

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

        # Group metrics by category (attention metrics first to exclude from other groups)
        attention_metrics = [k for k in debug_metrics if 'attn' in k]
        # Collapse includes std, diversity, and queue staleness (representation health)
        collapse_metrics = [k for k in debug_metrics if
                          ('std' in k or 'diversity' in k or 'staleness' in k)
                          and 'attn' not in k]
        gradient_metrics = [k for k in debug_metrics if 'grad_norm' in k]
        # Similarity includes: sim, margin, hard_negative, positive_rank
        similarity_metrics = [k for k in debug_metrics if
                             ('sim' in k or 'margin' in k or 'negative' in k or 'rank' in k)
                             and 'attn' not in k]

        # Create 4 subplots: collapse indicators, gradient norms, similarity, attention
        fig, axes = plt.subplots(4, 1, figsize=(14, 16))

        # Plot 1: Representation collapse indicators (std, diversity)
        ax1 = axes[0]
        for metric in collapse_metrics:
            if self.metrics['batch'][metric]:
                batches, values = zip(*self.metrics['batch'][metric])
                label = metric.replace('debug_train_', '').replace('debug_', '')
                ax1.plot(batches, values, linewidth=1.5, alpha=0.8, label=label, marker='o', markersize=2)

        ax1.set_xlabel('Batch', fontsize=11)
        ax1.set_ylabel('Value', fontsize=11)
        ax1.set_title('Representation Health (std/diversity: Higher=Better)', fontsize=13, fontweight='bold')
        ax1.axhline(y=0.1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Collapse threshold (0.1)')
        if collapse_metrics:
            ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Gradient norms
        ax2 = axes[1]
        for metric in gradient_metrics:
            if self.metrics['batch'][metric]:
                batches, values = zip(*self.metrics['batch'][metric])
                # Filter out zero values (not computed on that step)
                filtered = [(b, v) for b, v in zip(batches, values) if v > 0]
                if filtered:
                    batches, values = zip(*filtered)
                    label = metric.replace('debug_train_', '').replace('_grad_norm', '').replace('debug_', '')
                    # Use scatter for sparse gradient data (cleaner visualization)
                    ax2.scatter(batches, values, alpha=0.7, label=label, s=15)

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

        # Plot 4: Attention metrics (cross-channel attention stats)
        ax4 = axes[3]
        for metric in attention_metrics:
            if self.metrics['batch'][metric]:
                batches, values = zip(*self.metrics['batch'][metric])
                label = metric.replace('debug_train_', '').replace('debug_', '')
                ax4.plot(batches, values, linewidth=1.5, alpha=0.8, label=label, marker='o', markersize=2)

        ax4.set_xlabel('Batch', fontsize=11)
        ax4.set_ylabel('Value', fontsize=11)
        ax4.set_title('Cross-Channel Attention Stats', fontsize=13, fontweight='bold')
        if attention_metrics:
            ax4.legend(fontsize=9, loc='best')
        ax4.grid(True, alpha=0.3)

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
        print(f"  - debug_metrics.png (gradients, collapse, similarity, attention)")
        print(f"  - overall_loss.png (epoch-level train vs val)")
        print(f"  - loss_components.png (epoch-level MAE vs contrastive)")
        print(f"  - per_dataset_losses.png")
        print(f"  - learning_rate.png")
        print(f"  - dataset_*_detail.png (per dataset)")
        print(f"  - metrics.json (raw data)")


class EmbeddingVisualizer:
    """Visualizes high-dimensional embeddings using dimensionality reduction."""

    def __init__(self, output_dir: Path, n_neighbors: int = 15, min_dist: float = 0.1):
        """
        Args:
            output_dir: Directory to save visualization plots
            n_neighbors: UMAP n_neighbors parameter (5-50, higher = more global)
            min_dist: UMAP min_dist parameter (0.0-0.99, lower = tighter clusters)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist

    def plot_embedding_alignment_2d(
        self,
        imu_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        labels: List[str],
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        umap_metric: str = 'cosine'
    ):
        """
        Plot 2D UMAP projection showing alignment between IMU and text embeddings.

        Args:
            imu_embeddings: IMU embeddings (N, embedding_dim), should be L2-normalized
            text_embeddings: Text embeddings (N, embedding_dim), should be L2-normalized
            labels: Ground truth labels for each sample (N,)
            epoch: Current epoch number
            metrics: Optional dict of metrics to display (alignment_score, separation, gap, etc.)
            umap_metric: Distance metric for UMAP ('cosine' for normalized embeddings)
        """
        try:
            import umap
            from sklearn.metrics import silhouette_score
        except ImportError:
            print("Warning: umap-learn or scikit-learn not installed. Skipping embedding visualization.")
            return

        # Convert to numpy if tensors
        if hasattr(imu_embeddings, 'cpu'):
            imu_embeddings = imu_embeddings.cpu().numpy()
        if hasattr(text_embeddings, 'cpu'):
            text_embeddings = text_embeddings.cpu().numpy()

        # Create label mapping (text -> integer)
        unique_labels = sorted(set(labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        label_indices = np.array([label_to_idx[label] for label in labels])

        # Combine embeddings for joint UMAP projection
        all_embeddings = np.vstack([imu_embeddings, text_embeddings])

        # Fit UMAP
        print(f"  Computing UMAP projection (n_neighbors={self.n_neighbors}, min_dist={self.min_dist})...")
        reducer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_components=2,
            metric=umap_metric,
            random_state=42
        )
        embedding_2d = reducer.fit_transform(all_embeddings)

        # Split back into IMU and text
        imu_2d = embedding_2d[:len(imu_embeddings)]
        text_2d = embedding_2d[len(imu_embeddings):]

        # Compute additional metrics
        nn_accuracy = self._compute_nn_accuracy(imu_embeddings, text_embeddings)

        # Compute silhouette score (measure of cluster quality)
        try:
            silhouette = silhouette_score(imu_2d, label_indices, metric='euclidean')
        except:
            silhouette = 0.0

        # Create plot
        fig, ax = plt.subplots(figsize=(14, 10))

        # Choose colormap based on number of labels
        if len(unique_labels) <= 10:
            cmap = plt.cm.tab10
        else:
            cmap = plt.cm.tab20

        # Plot IMU embeddings (circles)
        for idx, label in enumerate(unique_labels):
            mask = label_indices == idx
            color = cmap(idx / len(unique_labels))

            # IMU points
            ax.scatter(
                imu_2d[mask, 0], imu_2d[mask, 1],
                c=[color], marker='o', s=30, alpha=0.6,
                label=f'{label} (IMU)', edgecolors='none'
            )

            # Text points (larger, more prominent)
            if mask.any():
                ax.scatter(
                    text_2d[mask, 0], text_2d[mask, 1],
                    c=[color], marker='^', s=80, alpha=0.9,
                    edgecolors='black', linewidths=0.5
                )

        # Add metrics text box
        if metrics is None:
            metrics = {}

        metrics_text = f'Epoch: {epoch}\n'
        metrics_text += f"NN Accuracy: {nn_accuracy:.2%}\n"
        metrics_text += f"Silhouette: {silhouette:.3f}\n"

        if 'alignment_score' in metrics:
            metrics_text += f"Alignment: {metrics['alignment_score']:.3f}\n"
        if 'separation' in metrics:
            metrics_text += f"Separation: {metrics['separation']:.3f}\n"
        if 'gap' in metrics:
            metrics_text += f"Gap: {metrics['gap']:.3f}"

        # Place text box in upper right
        ax.text(
            0.98, 0.98, metrics_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        # Styling
        ax.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12)
        ax.set_title(
            f'IMU-Text Embedding Alignment (Epoch {epoch})\n'
            f'Circles: IMU | Triangles: Text Labels',
            fontsize=14, fontweight='bold'
        )

        # Legend (only show first occurrence of each label)
        handles, labels_legend = ax.get_legend_handles_labels()
        # Keep only IMU labels (remove duplicates)
        unique_handles = []
        unique_labels = []
        seen = set()
        for h, l in zip(handles, labels_legend):
            label_name = l.replace(' (IMU)', '')
            if label_name not in seen:
                unique_handles.append(h)
                unique_labels.append(label_name)
                seen.add(label_name)

        ax.legend(
            unique_handles, unique_labels,
            fontsize=9, loc='upper left',
            framealpha=0.9, ncol=2 if len(unique_labels) > 8 else 1
        )
        ax.grid(True, alpha=0.2)

        # Save
        plt.tight_layout()
        output_path = self.output_dir / f'embedding_alignment_epoch_{epoch:03d}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  ✓ Saved embedding visualization: {output_path.name}")

    def _compute_nn_accuracy(
        self,
        imu_embeddings: np.ndarray,
        text_embeddings: np.ndarray
    ) -> float:
        """
        Compute nearest neighbor accuracy: % of IMU embeddings whose
        nearest neighbor is the correct text embedding.

        Args:
            imu_embeddings: IMU embeddings (N, embedding_dim)
            text_embeddings: Text embeddings (N, embedding_dim)

        Returns:
            Nearest neighbor accuracy (0-1)
        """
        # Compute cosine similarity (embeddings should already be normalized)
        similarities = np.matmul(imu_embeddings, text_embeddings.T)

        # For each IMU embedding, find nearest text embedding
        nearest_text_indices = np.argmax(similarities, axis=1)

        # Correct if index matches (diagonal)
        correct_indices = np.arange(len(imu_embeddings))
        correct = (nearest_text_indices == correct_indices).sum()

        return correct / len(imu_embeddings)
