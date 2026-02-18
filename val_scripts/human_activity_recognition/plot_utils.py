"""
Plotting utilities for training metrics.

Replaces TensorBoard with local PNG plots saved to disk.
"""

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (thread-safe with OO API)
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
        self._metrics_lock = threading.Lock()  # Guards self.metrics access

        # Track current epoch and batch
        self.current_epoch = 0
        self.global_batch = 0  # Global batch counter across all epochs

        # Background plotting thread (Agg backend + OO API is thread-safe)
        self._plot_thread = None

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

        with self._metrics_lock:
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
                loaded = json.load(f)
            with self._metrics_lock:
                self.metrics = loaded
            print(f"✓ Loaded existing metrics from {metrics_file}")

            # Print summary of loaded data
            n_epochs = len(self.metrics.get('epoch', {}).get('train_loss', []))
            n_batches = len(self.metrics.get('batch', {}).get('train_loss', []))
            print(f"  Resuming from: {n_epochs} epochs, {n_batches} batch records")
            return True
        return False

    def plot_all(self):
        """Generate all plots in a background thread.

        Skips if the previous plot is still running to avoid piling up.
        Uses Agg backend + OO matplotlib API which is thread-safe.
        """
        # Skip if previous plot still running
        if self._plot_thread is not None and self._plot_thread.is_alive():
            return

        self._plot_thread = threading.Thread(target=self._plot_all_sync, daemon=True)
        self._plot_thread.start()

    def _plot_all_sync(self):
        """Synchronous plot implementation (runs in background thread)."""
        import copy
        # Snapshot metrics under lock to avoid racing with add_scalar
        with self._metrics_lock:
            snapshot = copy.deepcopy(self.metrics)
        # Use snapshot for all plotting (avoids holding lock during slow I/O)
        real_metrics = self.metrics
        self.metrics = snapshot
        try:
            self._plot_batch_training_losses()
            self._plot_debug_metrics()
            self._plot_overall_loss()
            self._plot_loss_components()
            self._plot_per_dataset_losses()
            self._plot_learning_rate()
        except Exception as e:
            print(f"Warning: Background plotting failed: {e}")
        finally:
            self.metrics = real_metrics
        # Save real (up-to-date) metrics under lock
        with self._metrics_lock:
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
        # Also check for similarity metrics (logged without debug_ prefix)
        all_batch_keys = self.metrics['batch'].keys()
        has_similarity = any(('similarity' in k or 'sim_gap' in k) for k in all_batch_keys)
        if not debug_metrics and not has_similarity:
            return

        # Group metrics by category (attention metrics first to exclude from other groups)
        attention_metrics = [k for k in debug_metrics if 'attn' in k]
        # Collapse includes std, diversity, and queue staleness (representation health)
        collapse_metrics = [k for k in debug_metrics if
                          ('std' in k or 'diversity' in k or 'staleness' in k)
                          and 'attn' not in k]
        gradient_metrics = [k for k in debug_metrics if 'grad_norm' in k]
        # Similarity metrics: include both debug_ prefixed AND regular batch similarity metrics
        # (loss function computes label-aware similarity metrics logged without debug_ prefix)
        all_batch_keys = self.metrics['batch'].keys()
        similarity_metrics = [k for k in all_batch_keys if
                             ('similarity' in k or 'sim_gap' in k)
                             and 'attn' not in k and 'loss' not in k]

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
        # Wait for any background plot to finish
        if self._plot_thread is not None and self._plot_thread.is_alive():
            self._plot_thread.join(timeout=30)
        # Final plot synchronously to ensure everything is saved
        self._plot_all_sync()
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

        # Handle multi-prototype text embeddings: (N, K, D) → (N, D) via mean
        if text_embeddings.ndim == 3:
            text_embeddings = text_embeddings.mean(axis=1)

        # Filter out invalid embeddings (NaN/Inf) - safety net for edge cases
        # Zero-norm vectors can become NaN after normalization if they slip through
        imu_valid = ~(np.isnan(imu_embeddings).any(axis=1) | np.isinf(imu_embeddings).any(axis=1))
        text_valid = ~(np.isnan(text_embeddings).any(axis=1) | np.isinf(text_embeddings).any(axis=1))
        valid_mask = imu_valid & text_valid

        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            print(f"  WARNING: Filtering {n_invalid} invalid embeddings (NaN/Inf)")
            imu_embeddings = imu_embeddings[valid_mask]
            text_embeddings = text_embeddings[valid_mask]
            labels = [labels[i] for i in range(len(labels)) if valid_mask[i]]

            if len(imu_embeddings) == 0:
                print("  ERROR: All embeddings invalid! Skipping visualization.")
                return

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
        nn_accuracy = self._compute_nn_accuracy(imu_embeddings, text_embeddings, labels)

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

        if 'pos_sim' in metrics:
            metrics_text += f"Pos Sim: {metrics['pos_sim']:.3f}\n"
        if 'sim_gap' in metrics:
            metrics_text += f"Sim Gap: {metrics['sim_gap']:.3f}"

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
        text_embeddings: np.ndarray,
        labels: List[str]
    ) -> float:
        """
        Compute nearest neighbor accuracy: % of IMU embeddings whose
        nearest text neighbor has the SAME LABEL.

        This is LABEL-BASED matching, not position-based. With label augmentation,
        multiple samples share the same label, so we check if the retrieved
        text embedding's label matches the query's label.

        Args:
            imu_embeddings: IMU embeddings (N, embedding_dim)
            text_embeddings: Text embeddings (N, embedding_dim)
            labels: Ground truth labels for each sample (N,)

        Returns:
            Nearest neighbor accuracy (0-1)
        """
        # Compute cosine similarity (embeddings should already be normalized)
        similarities = np.matmul(imu_embeddings, text_embeddings.T)

        # For each IMU embedding, find nearest text embedding
        nearest_text_indices = np.argmax(similarities, axis=1)

        # Correct if retrieved label matches query label (LABEL-BASED)
        correct = 0
        for i, nearest_idx in enumerate(nearest_text_indices):
            if labels[nearest_idx] == labels[i]:
                correct += 1

        return correct / len(imu_embeddings)

    def _get_group_colors(self, labels: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Return coarse group colorings for all labels.

        Maps each fine-grained label to one of ~16 coarse activity groups defined in
        LABEL_GROUPS_SIMPLE, then assigns a fixed hex color to each group. Both IMU and
        text markers for the same group always share the same color, and the assignment
        is stable across calls (fixed palette, alphabetically sorted groups).

        Args:
            labels: List of activity label strings present in the current batch
                (e.g. ['walking', 'running', 'sitting', ...]). Used to determine
                which groups are actually present so the palette isn't wasted on
                absent groups.

        Returns:
            label_to_group: Dict mapping each label string to its coarse group name.
                Labels not in any LABEL_GROUPS_SIMPLE entry map to themselves.
            group_to_color: Dict mapping each coarse group name to a hex color string
                drawn from a fixed 20-color palette.
        """
        from datasets.imu_pretraining_dataset.label_groups import get_label_to_group_mapping
        label_to_group = get_label_to_group_mapping(use_simple=True)
        # Find groups actually present in the data
        groups_present = sorted(set(label_to_group.get(l, l) for l in labels))
        # 20 distinguishable colors — enough for ~16 groups
        palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
            '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
        ]
        group_to_color = {g: palette[i % len(palette)] for i, g in enumerate(groups_present)}
        return label_to_group, group_to_color

    def _filter_embeddings(
        self,
        imu_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        labels: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Convert embeddings to numpy and remove rows with NaN or Inf values.

        Handles three input cases transparently:
          - PyTorch tensors: moved to CPU and converted via .numpy()
          - Multi-prototype text embeddings (N, K, D): averaged over prototype dim K
          - Pre-converted numpy arrays: passed through unchanged

        A sample is dropped if *either* its IMU or text embedding contains any
        NaN or Inf entry. This keeps the IMU and text arrays the same length so
        they remain paired for cosine similarity computation.

        Args:
            imu_embeddings: IMU embeddings, shape (N, D) or torch.Tensor.
            text_embeddings: Text embeddings, shape (N, D), (N, K, D), or
                torch.Tensor of either shape.
            labels: List of N activity label strings corresponding to each row.

        Returns:
            Tuple (imu_embeddings, text_embeddings, labels) with invalid rows
            removed. Arrays are always plain numpy float arrays on return.

        Raises:
            ValueError: If no valid embeddings remain after filtering (all rows
                contained NaN or Inf).
        """
        # Convert to numpy if tensors
        if hasattr(imu_embeddings, 'cpu'):
            imu_embeddings = imu_embeddings.cpu().numpy()
        if hasattr(text_embeddings, 'cpu'):
            text_embeddings = text_embeddings.cpu().numpy()

        # Handle multi-prototype text embeddings: (N, K, D) -> (N, D) via mean
        if text_embeddings.ndim == 3:
            text_embeddings = text_embeddings.mean(axis=1)

        # Filter out invalid embeddings (NaN/Inf)
        imu_valid = ~(np.isnan(imu_embeddings).any(axis=1) | np.isinf(imu_embeddings).any(axis=1))
        text_valid = ~(np.isnan(text_embeddings).any(axis=1) | np.isinf(text_embeddings).any(axis=1))
        valid_mask = imu_valid & text_valid

        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            print(f"  WARNING: Filtering {n_invalid} invalid embeddings (NaN/Inf)")
            imu_embeddings = imu_embeddings[valid_mask]
            text_embeddings = text_embeddings[valid_mask]
            labels = [labels[i] for i in range(len(labels)) if valid_mask[i]]

            if len(imu_embeddings) == 0:
                raise ValueError("All embeddings invalid!")

        return imu_embeddings, text_embeddings, labels

    def plot_interactive_3d(
        self,
        imu_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        labels: List[str],
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        umap_metric: str = 'cosine'
    ):
        """Interactive 3D UMAP visualization saved as a self-contained HTML file.

        Fits UMAP in 3D on the joint set of IMU and text embeddings (cosine distance
        by default). Produces one Plotly Scatter3d trace per (activity group, modality)
        pair so clicking a legend entry toggles the entire group. IMU samples are
        shown as small semi-transparent dots; text embeddings are deduplicated by label
        and shown as labelled diamonds.

        Built-in Plotly interactivity:
          - Rotate / zoom / pan the 3D scene
          - Hover markers for label and group details
          - Click legend entries to show / hide individual groups

        Args:
            imu_embeddings: IMU embeddings, shape (N, D).
            text_embeddings: Text embeddings, shape (N, D).
            labels: Activity label for each of the N samples.
            epoch: Training epoch number (shown in title and embedded in filename).
            metrics: Optional scalar metrics to display in the figure title,
                e.g. {'pos_sim': 0.82, 'sim_gap': 0.41}.
            umap_metric: Distance metric passed to umap.UMAP (default 'cosine').

        Outputs:
            ``embedding_3d_epoch_{epoch:03d}.html`` — written to ``self.output_dir``.
            The file is fully self-contained (plotly.js bundled) so it opens
            correctly without a server.
        """
        try:
            import umap
            import plotly.graph_objects as go
        except ImportError:
            print("Warning: umap-learn or plotly not installed. Skipping 3D visualization.")
            return

        try:
            imu_embeddings, text_embeddings, labels = self._filter_embeddings(
                imu_embeddings, text_embeddings, labels
            )
        except ValueError:
            print("  ERROR: All embeddings invalid! Skipping 3D visualization.")
            return

        label_to_group, group_to_color = self._get_group_colors(labels)

        # Assign group for each sample
        groups = [label_to_group.get(l, l) for l in labels]

        # Joint UMAP 3D projection
        all_embeddings = np.vstack([imu_embeddings, text_embeddings])
        print(f"  Computing 3D UMAP projection (n_neighbors={self.n_neighbors}, min_dist={self.min_dist})...")
        reducer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_components=3,
            metric=umap_metric,
            random_state=42,
        )
        embedding_3d = reducer.fit_transform(all_embeddings)
        imu_3d = embedding_3d[:len(imu_embeddings)]
        text_3d = embedding_3d[len(imu_embeddings):]

        # Build Plotly figure — one trace per (group, modality) for legend filtering
        fig = go.Figure()
        groups_sorted = sorted(group_to_color.keys())

        for group_name in groups_sorted:
            color = group_to_color[group_name]

            # IMU points for this group
            mask = np.array([g == group_name for g in groups])
            if mask.any():
                idx = np.where(mask)[0]
                hover = [f"Label: {labels[i]}<br>Group: {group_name}<br>Idx: {i}" for i in idx]
                fig.add_trace(go.Scatter3d(
                    x=imu_3d[mask, 0], y=imu_3d[mask, 1], z=imu_3d[mask, 2],
                    mode='markers',
                    marker=dict(size=3, color=color, opacity=0.5),
                    name=f'{group_name} (IMU)',
                    legendgroup=group_name,
                    hovertext=hover,
                    hoverinfo='text',
                ))

            # Text points for this group (deduplicate identical labels)
            text_mask = mask
            if text_mask.any():
                idx = np.where(text_mask)[0]
                # Deduplicate: keep first occurrence of each label within this group
                seen_labels = set()
                dedup_idx = []
                for i in idx:
                    if labels[i] not in seen_labels:
                        seen_labels.add(labels[i])
                        dedup_idx.append(i)
                dedup_idx = np.array(dedup_idx)
                hover = [f"Label: {labels[i]}<br>Group: {group_name}" for i in dedup_idx]
                text_labels = [labels[i] for i in dedup_idx]
                fig.add_trace(go.Scatter3d(
                    x=text_3d[dedup_idx, 0], y=text_3d[dedup_idx, 1], z=text_3d[dedup_idx, 2],
                    mode='markers+text',
                    marker=dict(size=8, color=color, opacity=1.0, symbol='diamond'),
                    text=text_labels,
                    textposition='top center',
                    textfont=dict(size=9, color=color),
                    name=f'{group_name} (Text)',
                    legendgroup=group_name,
                    hovertext=hover,
                    hoverinfo='text',
                ))

        # Layout
        title_parts = [f'IMU-Text Embedding Alignment — 3D UMAP (Epoch {epoch})']
        if metrics:
            metric_strs = [f'{k}: {v:.3f}' for k, v in metrics.items()]
            title_parts.append(' | '.join(metric_strs))
        fig.update_layout(
            title='<br>'.join(title_parts),
            scene=dict(
                xaxis_title='UMAP 1',
                yaxis_title='UMAP 2',
                zaxis_title='UMAP 3',
                bgcolor='rgb(20, 20, 30)',
            ),
            paper_bgcolor='rgb(20, 20, 30)',
            font=dict(color='white'),
            legend=dict(
                bgcolor='rgba(40,40,50,0.8)',
                font=dict(size=10),
                itemsizing='constant',
            ),
            margin=dict(l=0, r=0, t=60, b=0),
        )

        output_path = self.output_dir / f'embedding_3d_epoch_{epoch:03d}.html'
        fig.write_html(str(output_path), include_plotlyjs=True)
        print(f"  ✓ Saved interactive 3D visualization: {output_path.name}")

    def plot_density_contours(
        self,
        imu_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        labels: List[str],
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        umap_metric: str = 'cosine'
    ):
        """2D UMAP with per-group KDE density contours and annotated text markers.

        Fits a joint 2D UMAP on all IMU and text embeddings, then draws filled
        gaussian_kde contours for each activity group. Groups with ≥20 IMU points
        get three nested filled contour bands (30 / 60 / 90 % of peak density);
        groups with fewer points fall back to a plain scatter. Text embeddings are
        deduplicated by label and shown as star markers with label annotations;
        ``adjustText`` is used for non-overlapping placement when available.

        Args:
            imu_embeddings: IMU embeddings, shape (N, D).
            text_embeddings: Text embeddings, shape (N, D).
            labels: Activity label for each of the N samples.
            epoch: Training epoch number (shown in metrics box, title, filename).
            metrics: Optional dict with keys 'pos_sim' and/or 'sim_gap' to include
                in the annotation box (alongside NN accuracy and silhouette score).
            umap_metric: Distance metric passed to umap.UMAP (default 'cosine').

        Outputs:
            ``embedding_density_epoch_{epoch:03d}.png`` — 200 DPI PNG written to
            ``self.output_dir``.
        """
        try:
            import umap
            from scipy.stats import gaussian_kde
            from sklearn.metrics import silhouette_score
        except ImportError:
            print("Warning: umap-learn, scipy, or scikit-learn not installed. Skipping density contour plot.")
            return

        try:
            imu_embeddings, text_embeddings, labels = self._filter_embeddings(
                imu_embeddings, text_embeddings, labels
            )
        except ValueError:
            print("  ERROR: All embeddings invalid! Skipping density contour plot.")
            return

        label_to_group, group_to_color = self._get_group_colors(labels)
        groups = [label_to_group.get(l, l) for l in labels]
        groups_arr = np.array(groups)

        # Joint UMAP 2D projection
        all_embeddings = np.vstack([imu_embeddings, text_embeddings])
        print(f"  Computing 2D UMAP for density contours (n_neighbors={self.n_neighbors}, min_dist={self.min_dist})...")
        reducer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_components=2,
            metric=umap_metric,
            random_state=42,
        )
        embedding_2d = reducer.fit_transform(all_embeddings)
        imu_2d = embedding_2d[:len(imu_embeddings)]
        text_2d = embedding_2d[len(imu_embeddings):]

        # Compute metrics
        label_indices = np.array([sorted(set(labels)).index(l) for l in labels])
        nn_accuracy = self._compute_nn_accuracy(imu_embeddings, text_embeddings, labels)
        try:
            silhouette = silhouette_score(imu_2d, label_indices, metric='euclidean')
        except Exception:
            silhouette = 0.0

        # Create figure
        fig, ax = plt.subplots(figsize=(16, 12))

        groups_sorted = sorted(group_to_color.keys())
        contour_handles = []  # For legend

        for group_name in groups_sorted:
            color = group_to_color[group_name]
            mask = groups_arr == group_name
            pts = imu_2d[mask]

            if len(pts) >= 20:
                # KDE contours
                try:
                    kde = gaussian_kde(pts.T)
                    # Evaluate on a grid
                    x_min, x_max = pts[:, 0].min() - 1, pts[:, 0].max() + 1
                    y_min, y_max = pts[:, 1].min() - 1, pts[:, 1].max() + 1
                    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                    positions = np.vstack([xx.ravel(), yy.ravel()])
                    zz = kde(positions).reshape(xx.shape)

                    # Contour levels at 30%, 60%, 90% of peak density
                    peak = zz.max()
                    levels = [peak * 0.30, peak * 0.60, peak * 0.90, peak * 1.01]
                    alphas = [0.15, 0.25, 0.35]

                    from matplotlib.colors import to_rgba
                    for i in range(len(levels) - 1):
                        rgba = to_rgba(color, alpha=alphas[i])
                        ax.contourf(xx, yy, zz, levels=[levels[i], levels[i + 1]],
                                    colors=[rgba])

                    # Outline at the outermost level
                    ax.contour(xx, yy, zz, levels=[levels[0]], colors=[color],
                               linewidths=0.8, alpha=0.6)

                    # Legend proxy
                    import matplotlib.patches as mpatches
                    contour_handles.append(mpatches.Patch(color=color, alpha=0.4, label=group_name))
                except Exception:
                    # KDE failed (e.g., singular matrix) — fall back to scatter
                    ax.scatter(pts[:, 0], pts[:, 1], c=color, s=10, alpha=0.4, edgecolors='none')
                    import matplotlib.patches as mpatches
                    contour_handles.append(mpatches.Patch(color=color, alpha=0.4, label=group_name))
            elif len(pts) > 0:
                # Too few points for KDE — scatter
                ax.scatter(pts[:, 0], pts[:, 1], c=color, s=10, alpha=0.4, edgecolors='none')
                import matplotlib.patches as mpatches
                contour_handles.append(mpatches.Patch(color=color, alpha=0.4, label=group_name))

        # Plot text embedding positions — deduplicate labels
        seen_text = {}  # label -> (x, y, group)
        for i, label in enumerate(labels):
            if label not in seen_text:
                seen_text[label] = (text_2d[i, 0], text_2d[i, 1], label_to_group.get(label, label))

        text_xs, text_ys, text_colors, text_names = [], [], [], []
        for label, (x, y, grp) in seen_text.items():
            text_xs.append(x)
            text_ys.append(y)
            text_colors.append(group_to_color.get(grp, '#888888'))
            text_names.append(label)

        ax.scatter(text_xs, text_ys, c=text_colors, marker='*', s=120,
                   edgecolors='black', linewidths=0.5, zorder=5)

        # Annotate text labels
        texts = []
        for x, y, name in zip(text_xs, text_ys, text_names):
            t = ax.annotate(
                name, (x, y), fontsize=7, alpha=0.85,
                xytext=(4, 4), textcoords='offset points',
            )
            texts.append(t)

        # Try adjustText for non-overlapping labels (graceful fallback)
        try:
            from adjustText import adjust_text
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.4))
        except ImportError:
            pass  # Basic offset is fine

        # Metrics box
        if metrics is None:
            metrics = {}
        metrics_text = f'Epoch: {epoch}\n'
        metrics_text += f'NN Accuracy: {nn_accuracy:.2%}\n'
        metrics_text += f'Silhouette: {silhouette:.3f}\n'
        if 'pos_sim' in metrics:
            metrics_text += f"Pos Sim: {metrics['pos_sim']:.3f}\n"
        if 'sim_gap' in metrics:
            metrics_text += f"Sim Gap: {metrics['sim_gap']:.3f}"
        ax.text(
            0.98, 0.98, metrics_text,
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        )

        # Compact legend
        if contour_handles:
            ax.legend(handles=contour_handles, fontsize=9, loc='upper left',
                      framealpha=0.9, ncol=2 if len(contour_handles) > 8 else 1)

        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.set_title(
            f'IMU Density Contours + Text Labels (Epoch {epoch})\n'
            f'Stars: Text Embeddings | Contours: IMU Density per Group',
            fontsize=14, fontweight='bold',
        )
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        output_path = self.output_dir / f'embedding_density_epoch_{epoch:03d}.png'
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved density contour visualization: {output_path.name}")

    def plot_paper_figure(
        self,
        imu_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        labels: List[str],
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        umap_metric: str = 'cosine',
        n_neighbors: int = 10,
        min_dist: float = 0.05,
        exclude_groups: Tuple[str, ...] = ('postural_transition',),
    ):
        """Publication-quality 2D embedding figure using per-label prototype centroids.

        Projects per-label IMU centroids and per-label text centroids (2 × n_labels
        points total) rather than all individual samples. Averaging removes
        within-label noise, giving tight well-separated clusters; UMAP also produces
        cleaner layouts at this scale (~174 points vs. ~11 k). Text diamonds land
        inside or very close to their corresponding IMU cluster; thin lines connect
        each matched pair to directly visualise cross-modal alignment.

        Visual design:
          - Large circles (s=160, white edge) for IMU centroids drawn first.
          - Smaller diamonds (s=40, black edge) for text centroids on top — their
            edges always peek out from behind the overlapping diamond.
          - Bold group-name label at the centroid of each IMU group.
          - Fine per-label annotations on text centroids, placed by adjustText.
          - NN accuracy in the metrics box is computed on ALL individual sample
            embeddings (not centroids) so it reflects honest retrieval performance.
          - No convex-hull or fill shapes (too noisy at this scale).
          - Axis limits are set tightly around the data (8 % padding) so no blank
            UMAP space is wasted in the printed figure.

        Args:
            imu_embeddings: IMU embeddings, shape (N, D).
            text_embeddings: Text embeddings, shape (N, D).
            labels: Activity label for each of the N samples.
            epoch: Training epoch number (shown in metrics box, title, filename).
            metrics: Optional dict with keys 'pos_sim' and/or 'sim_gap'.
            umap_metric: Distance metric passed to umap.UMAP (default 'cosine').
            n_neighbors: UMAP n_neighbors. Default 10 is appropriate for ~174
                prototype points; would be too small for thousands of samples.
            min_dist: UMAP min_dist. Default 0.05 gives tight cluster packing at
                prototype scale.
            exclude_groups: Coarse group names (LABEL_GROUPS_SIMPLE keys) to drop
                before projecting. Defaults to ``('postural_transition',)`` which
                tends to be a distant outlier that wastes axis space.

        Outputs:
            ``embedding_paper_epoch_{epoch:03d}.png`` — 300 DPI PNG written to
            ``self.output_dir``. Does **not** overwrite
            ``embedding_alignment_epoch_*.png`` (the sample-level plots).
        """
        try:
            import umap
        except ImportError:
            print("Warning: umap-learn not installed. Skipping paper figure.")
            return

        try:
            imu_embeddings, text_embeddings, labels = self._filter_embeddings(
                imu_embeddings, text_embeddings, labels
            )
        except ValueError:
            print("  ERROR: All embeddings invalid! Skipping paper figure.")
            return

        label_to_group, group_to_color = self._get_group_colors(labels)

        # Drop samples whose coarse group is in exclude_groups
        if exclude_groups:
            keep = [label_to_group.get(l, l) not in exclude_groups for l in labels]
            n_dropped = len(labels) - sum(keep)
            if n_dropped:
                print(f"  Excluding {n_dropped} samples from groups: {exclude_groups}")
            imu_embeddings = imu_embeddings[keep]
            text_embeddings = text_embeddings[keep]
            labels = [l for l, k in zip(labels, keep) if k]
            # Recompute group colors with remaining labels (may remove a group entirely)
            label_to_group, group_to_color = self._get_group_colors(labels)

        labels_arr = np.array(labels)
        unique_labels = sorted(set(labels))

        # Build per-label centroids (average across all samples with that label)
        imu_centroids = np.zeros((len(unique_labels), imu_embeddings.shape[1]))
        text_centroids = np.zeros((len(unique_labels), text_embeddings.shape[1]))
        for i, lbl in enumerate(unique_labels):
            mask = labels_arr == lbl
            imu_centroids[i] = imu_embeddings[mask].mean(axis=0)
            text_centroids[i] = text_embeddings[mask].mean(axis=0)

        # Re-normalise after averaging (cosine metric expects unit vectors)
        imu_centroids /= np.maximum(np.linalg.norm(imu_centroids, axis=1, keepdims=True), 1e-8)
        text_centroids /= np.maximum(np.linalg.norm(text_centroids, axis=1, keepdims=True), 1e-8)

        centroid_groups = np.array([label_to_group.get(l, l) for l in unique_labels])

        nn_accuracy = self._compute_nn_accuracy(imu_embeddings, text_embeddings, labels)

        # Joint UMAP on 87+87=174 prototype points
        # Small N → well-spaced, no compression → clean cluster separation
        all_centroids = np.vstack([imu_centroids, text_centroids])
        n = len(imu_centroids)
        actual_neighbors = min(n_neighbors, len(all_centroids) - 1)
        print(f"  Computing paper UMAP on {len(all_centroids)} label centroids "
              f"(n_neighbors={actual_neighbors}, min_dist={min_dist})...")
        reducer = umap.UMAP(
            n_neighbors=actual_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric=umap_metric,
            random_state=42,
        )
        proj = reducer.fit_transform(all_centroids)
        imu_proj = proj[:n]   # (87, 2)
        text_proj = proj[n:]  # (87, 2)

        groups_sorted = sorted(group_to_color.keys())
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.set_facecolor('#fafafa')

        # Layer 1: thin lines connecting each IMU centroid to its paired text centroid
        for i in range(n):
            color = group_to_color.get(centroid_groups[i], '#888888')
            ax.plot(
                [imu_proj[i, 0], text_proj[i, 0]],
                [imu_proj[i, 1], text_proj[i, 1]],
                color=color, linewidth=0.6, alpha=0.35, zorder=2,
            )

        # Layer 3: IMU centroids (large circles — drawn first so diamonds sit on top
        #           but circles are bigger so their edges always peek out)
        for group_name in groups_sorted:
            color = group_to_color[group_name]
            mask = centroid_groups == group_name
            pts = imu_proj[mask]
            if len(pts) == 0:
                continue
            ax.scatter(pts[:, 0], pts[:, 1], c=color, marker='o', s=160,
                       alpha=0.85, edgecolors='white', linewidths=1.0, zorder=3)

        # Layer 4: text centroids (smaller diamonds on top)
        for group_name in groups_sorted:
            color = group_to_color[group_name]
            mask = centroid_groups == group_name
            pts = text_proj[mask]
            if len(pts) == 0:
                continue
            ax.scatter(pts[:, 0], pts[:, 1], c=color, marker='D', s=40,
                       alpha=0.95, edgecolors='black', linewidths=0.5, zorder=4)

        # Layer 5: label name annotations on text centroids
        annot_texts = []
        for i, lbl in enumerate(unique_labels):
            t = ax.annotate(
                lbl.replace('_', ' '), (text_proj[i, 0], text_proj[i, 1]),
                fontsize=5.5, alpha=0.8,
                xytext=(3, 3), textcoords='offset points', zorder=5,
            )
            annot_texts.append(t)

        try:
            from adjustText import adjust_text
            adjust_text(annot_texts, ax=ax,
                        arrowprops=dict(arrowstyle='-', color='gray', alpha=0.3, lw=0.5),
                        only_move={'points': 'y', 'text': 'xy'})
        except ImportError:
            pass

        # Layer 6: bold group name at IMU group centroid
        for group_name in groups_sorted:
            mask = centroid_groups == group_name
            pts = imu_proj[mask]
            if len(pts) == 0:
                continue
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            ax.text(cx, cy, group_name.replace('_', '\n'),
                    fontsize=7.5, fontweight='bold',
                    ha='center', va='center', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              alpha=0.80, edgecolor='none'))

        # Metrics box
        if metrics is None:
            metrics = {}
        metrics_lines = [f'Epoch: {epoch}', f'NN Acc: {nn_accuracy:.1%}']
        if 'pos_sim' in metrics:
            metrics_lines.append(f"Pos Sim: {metrics['pos_sim']:.3f}")
        if 'sim_gap' in metrics:
            metrics_lines.append(f"Sim Gap: {metrics['sim_gap']:.3f}")
        ax.text(0.02, 0.02, '\n'.join(metrics_lines),
                transform=ax.transAxes, fontsize=9, va='bottom', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#cccccc'))

        # Legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        group_handles = [
            Patch(facecolor=group_to_color[g], alpha=0.65, label=g.replace('_', ' '))
            for g in groups_sorted if np.any(centroid_groups == g)
        ]
        modality_handles = [
            Line2D([0], [0], marker='o', color='#555', ms=9, lw=0,
                   markeredgecolor='white', markeredgewidth=0.8, label='IMU centroid'),
            Line2D([0], [0], marker='D', color='#555', ms=5, lw=0,
                   markeredgecolor='black', markeredgewidth=0.5, label='Text centroid'),
        ]
        # Place legend outside the axes (right side) so it never overlaps data points.
        ax.legend(handles=modality_handles + group_handles,
                  fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0),
                  framealpha=0.9, ncol=1,
                  title='Activity Group', title_fontsize=9, borderpad=0.6)

        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.set_title(
            f'IMU–Text Embedding Alignment (Epoch {epoch})\n'
            f'Circles: IMU centroids  |  Diamonds: Text centroids  |  Lines: Matched pairs',
            fontsize=13, fontweight='bold',
        )
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Tight axis limits — clip to data extent so blank UMAP space is removed.
        # Compute bounds from the actual projected points (not annotations).
        all_pts = np.vstack([imu_proj, text_proj])
        xpad = (all_pts[:, 0].max() - all_pts[:, 0].min()) * 0.08
        ypad = (all_pts[:, 1].max() - all_pts[:, 1].min()) * 0.08
        ax.set_xlim(all_pts[:, 0].min() - xpad, all_pts[:, 0].max() + xpad)
        ax.set_ylim(all_pts[:, 1].min() - ypad, all_pts[:, 1].max() + ypad)

        plt.tight_layout()
        output_path = self.output_dir / f'embedding_paper_epoch_{epoch:03d}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved paper figure: {output_path.name}")
