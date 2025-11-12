"""
Prototype manager for prototypical soft targets.

Accumulates label embeddings and performs clustering to create prototype vectors
representing semantic categories (e.g., locomotion, stationary, activities).
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans
from collections import defaultdict


class PrototypeManager:
    """
    Manages prototype vectors for semantic categories.

    Accumulates label embeddings over time, clusters them to find semantic
    categories, and provides prototype-based soft targets for training.
    """

    def __init__(
        self,
        num_clusters: int = 5,
        update_interval: int = 100,
        min_samples_per_cluster: int = 3,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            num_clusters: Number of prototype clusters (semantic categories)
            update_interval: Number of add_labels calls between clustering updates
            min_samples_per_cluster: Minimum labels needed before clustering
            device: Device to store prototypes on
        """
        self.num_clusters = num_clusters
        self.update_interval = update_interval
        self.min_samples_per_cluster = min_samples_per_cluster
        self.device = device if device is not None else torch.device('cpu')

        # Label memory: accumulate embeddings for each unique label
        self.label_embeddings = defaultdict(list)  # {label_text: [embeddings]}

        # Clustering results
        self.prototypes = None  # (num_clusters, embedding_dim) tensor
        self.label_to_cluster = {}  # {label_text: cluster_id}
        self.cluster_labels = defaultdict(list)  # {cluster_id: [label_texts]}

        # Tracking
        self.num_labels_seen = 0
        self.num_updates = 0
        self.update_counter = 0
        self.is_initialized = False

    def add_labels(
        self,
        label_texts: List[str],
        text_embeddings: torch.Tensor
    ):
        """
        Add label embeddings to memory.

        Args:
            label_texts: List of label text strings
            text_embeddings: Corresponding embeddings (batch, embedding_dim)
        """
        # Store embeddings for each label
        for label, emb in zip(label_texts, text_embeddings):
            self.label_embeddings[label].append(emb.cpu().detach())

        self.update_counter += 1

        # Check if we should update prototypes
        if self.update_counter >= self.update_interval:
            total_samples = sum(len(embs) for embs in self.label_embeddings.values())
            if total_samples >= self.min_samples_per_cluster * self.num_clusters:
                self.update_prototypes()
            self.update_counter = 0

    def update_prototypes(self):
        """
        Perform clustering on accumulated label embeddings to compute prototypes.
        """
        if len(self.label_embeddings) == 0:
            return

        # Compute mean embedding for each label
        label_means = {}
        labels_list = []
        embeddings_list = []

        for label, emb_list in self.label_embeddings.items():
            # Average all embeddings for this label
            mean_emb = torch.stack(emb_list).mean(dim=0)
            label_means[label] = mean_emb
            labels_list.append(label)
            embeddings_list.append(mean_emb.numpy())

        if len(labels_list) < self.num_clusters:
            # Not enough unique labels yet
            return

        # Stack into matrix for clustering
        embeddings_matrix = np.stack(embeddings_list)  # (num_labels, embedding_dim)

        # Perform k-means clustering
        num_clusters_actual = min(self.num_clusters, len(labels_list))
        kmeans = KMeans(n_clusters=num_clusters_actual, random_state=42, n_init=10)
        cluster_assignments = kmeans.fit_predict(embeddings_matrix)

        # Store cluster centers as prototypes
        self.prototypes = torch.from_numpy(kmeans.cluster_centers_).float().to(self.device)

        # Update label to cluster mapping
        self.label_to_cluster = {}
        self.cluster_labels = defaultdict(list)

        for label, cluster_id in zip(labels_list, cluster_assignments):
            self.label_to_cluster[label] = int(cluster_id)
            self.cluster_labels[int(cluster_id)].append(label)

        self.num_updates += 1
        self.is_initialized = True
        self.num_labels_seen = len(labels_list)

        print(f"\n[PrototypeManager] Updated prototypes (update #{self.num_updates})")
        print(f"  Total unique labels: {self.num_labels_seen}")
        print(f"  Number of clusters: {num_clusters_actual}")
        for cluster_id, cluster_labels in self.cluster_labels.items():
            print(f"  Cluster {cluster_id} ({len(cluster_labels)} labels): {cluster_labels[:5]}{'...' if len(cluster_labels) > 5 else ''}")

    def get_prototype_targets(
        self,
        label_texts: List[str],
        text_embeddings: torch.Tensor,
        temperature: float = 0.5
    ) -> torch.Tensor:
        """
        Compute soft targets based on prototypes.

        Args:
            label_texts: List of labels in batch
            text_embeddings: Text embeddings for batch (batch, embedding_dim)
            temperature: Temperature for softmax over prototypes

        Returns:
            Soft targets (batch, batch) - distribution over batch labels based on prototype distances
        """
        if not self.is_initialized or self.prototypes is None:
            # Fallback to hard targets if prototypes not ready
            batch_size = len(label_texts)
            return torch.eye(batch_size, device=text_embeddings.device)

        batch_size = len(label_texts)
        prototypes = self.prototypes.to(text_embeddings.device)

        # For each label, find its cluster and prototype
        soft_targets = []

        for i, label in enumerate(label_texts):
            if label in self.label_to_cluster:
                # Get cluster ID for this label
                cluster_id = self.label_to_cluster[label]
                prototype = prototypes[cluster_id]  # (embedding_dim,)

                # Compute similarity of this prototype to all text embeddings in batch
                similarities = torch.matmul(prototype, text_embeddings.T)  # (batch,)
                similarities = similarities / temperature

                # Softmax to get distribution
                probs = torch.softmax(similarities, dim=0)
                soft_targets.append(probs)
            else:
                # Label not seen during clustering, use one-hot
                one_hot = torch.zeros(batch_size, device=text_embeddings.device)
                one_hot[i] = 1.0
                soft_targets.append(one_hot)

        return torch.stack(soft_targets)  # (batch, batch)

    def get_cluster_info(self, label_text: str) -> Optional[Dict]:
        """
        Get cluster information for a label.

        Args:
            label_text: Label to query

        Returns:
            Dict with cluster_id and cluster_labels, or None if not found
        """
        if label_text not in self.label_to_cluster:
            return None

        cluster_id = self.label_to_cluster[label_text]
        return {
            'cluster_id': cluster_id,
            'cluster_labels': self.cluster_labels[cluster_id],
            'num_labels_in_cluster': len(self.cluster_labels[cluster_id])
        }

    def get_statistics(self) -> Dict:
        """Get statistics about prototype manager state."""
        return {
            'is_initialized': self.is_initialized,
            'num_unique_labels': len(self.label_embeddings),
            'num_labels_seen': self.num_labels_seen,
            'num_clusters': len(self.cluster_labels) if self.is_initialized else 0,
            'num_updates': self.num_updates,
            'total_embeddings_accumulated': sum(len(embs) for embs in self.label_embeddings.values())
        }

    def save(self, path: str):
        """Save prototype manager state."""
        state = {
            'num_clusters': self.num_clusters,
            'prototypes': self.prototypes.cpu() if self.prototypes is not None else None,
            'label_to_cluster': self.label_to_cluster,
            'cluster_labels': dict(self.cluster_labels),
            'num_updates': self.num_updates,
            'is_initialized': self.is_initialized
        }
        torch.save(state, path)

    def load(self, path: str):
        """Load prototype manager state."""
        state = torch.load(path, map_location='cpu')
        self.num_clusters = state['num_clusters']
        self.prototypes = state['prototypes'].to(self.device) if state['prototypes'] is not None else None
        self.label_to_cluster = state['label_to_cluster']
        self.cluster_labels = defaultdict(list, state['cluster_labels'])
        self.num_updates = state['num_updates']
        self.is_initialized = state['is_initialized']
