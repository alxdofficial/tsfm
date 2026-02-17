"""Test 2: Similarity computation correctness.

Verifies compute_similarity handles both 2D and 3D label embeddings,
and that group accuracy / semantic recall return correct results.
"""

import sys
from pathlib import Path
import torch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from val_scripts.human_activity_recognition.evaluation_metrics import (
    compute_similarity,
    compute_semantic_recall,
    compute_group_accuracy,
)


class TestComputeSimilarity:
    """Test compute_similarity with 2D and 3D label embeddings."""

    def test_2d_labels_shape(self):
        """(N, D) IMU x (L, D) labels -> (N, L) similarity."""
        N, D, L = 10, 384, 5
        imu = torch.randn(N, D)
        imu = imu / imu.norm(dim=-1, keepdim=True)
        labels = torch.randn(L, D)
        labels = labels / labels.norm(dim=-1, keepdim=True)
        sim = compute_similarity(imu, labels)
        assert sim.shape == (N, L), f"Expected ({N}, {L}), got {sim.shape}"

    def test_3d_labels_shape(self):
        """(N, D) IMU x (L, K, D) labels -> (N, L) similarity (max over K)."""
        N, D, L, K = 10, 384, 5, 3
        imu = torch.randn(N, D)
        imu = imu / imu.norm(dim=-1, keepdim=True)
        labels = torch.randn(L, K, D)
        labels = labels / labels.norm(dim=-1, keepdim=True)
        sim = compute_similarity(imu, labels)
        assert sim.shape == (N, L), f"Expected ({N}, {L}), got {sim.shape}"

    def test_2d_cosine_similarity_values(self):
        """Verify similarity values are correct cosine similarities."""
        # Identity case: each IMU embedding is one of the label embeddings
        D = 8
        labels = torch.eye(D)[:3]  # 3 labels, each a one-hot
        imu = labels.clone()  # IMU matches labels exactly
        sim = compute_similarity(imu, labels)
        # Diagonal should be 1.0 (self-similarity)
        for i in range(3):
            assert abs(sim[i, i].item() - 1.0) < 1e-6
        # Off-diagonal should be 0.0 (orthogonal)
        assert abs(sim[0, 1].item()) < 1e-6

    def test_3d_max_over_prototypes(self):
        """With K prototypes, similarity should be max over K."""
        D = 8
        imu = torch.zeros(1, D)
        imu[0, 0] = 1.0  # unit vector along dim 0

        # Label has 2 prototypes: one aligned, one orthogonal
        labels = torch.zeros(1, 2, D)
        labels[0, 0, 0] = 1.0  # prototype 0: aligned -> sim = 1.0
        labels[0, 1, 1] = 1.0  # prototype 1: orthogonal -> sim = 0.0

        sim = compute_similarity(imu, labels)
        assert sim.shape == (1, 1)
        assert abs(sim[0, 0].item() - 1.0) < 1e-6, "Should pick max (aligned) prototype"

    def test_similarity_range(self):
        """Cosine similarity of normalized vectors should be in [-1, 1]."""
        N, D, L = 50, 384, 20
        imu = torch.randn(N, D)
        imu = imu / imu.norm(dim=-1, keepdim=True)
        labels = torch.randn(L, D)
        labels = labels / labels.norm(dim=-1, keepdim=True)
        sim = compute_similarity(imu, labels)
        assert sim.min() >= -1.0 - 1e-6
        assert sim.max() <= 1.0 + 1e-6


class TestComputeSemanticRecall:
    """Test semantic recall computation."""

    def test_perfect_recall(self):
        """When IMU embeddings match their labels exactly, recall@1 should be 1.0."""
        D = 384
        labels = ['walking', 'running', 'sitting']
        # Create distinct embeddings per label
        label_embeddings = torch.randn(3, D)
        label_embeddings = label_embeddings / label_embeddings.norm(dim=-1, keepdim=True)

        # IMU embeddings = exact label embeddings (perfect match)
        imu_embeddings = label_embeddings.clone()

        metrics = compute_semantic_recall(
            imu_embeddings, label_embeddings,
            query_labels=labels, corpus_labels=labels,
            k_values=[1, 3], use_groups=False
        )
        assert metrics['semantic_recall@1'] == 1.0

    def test_recall_at_k_monotonic(self):
        """recall@k should be monotonically non-decreasing with k."""
        D = 384
        N, L = 20, 10
        imu = torch.randn(N, D)
        imu = imu / imu.norm(dim=-1, keepdim=True)
        labels_emb = torch.randn(L, D)
        labels_emb = labels_emb / labels_emb.norm(dim=-1, keepdim=True)

        query_labels = [f'label_{i % L}' for i in range(N)]
        corpus_labels = [f'label_{i}' for i in range(L)]

        metrics = compute_semantic_recall(
            imu, labels_emb,
            query_labels=query_labels, corpus_labels=corpus_labels,
            k_values=[1, 3, 5], use_groups=False
        )
        assert metrics['semantic_recall@1'] <= metrics['semantic_recall@3']
        assert metrics['semantic_recall@3'] <= metrics['semantic_recall@5']


class TestComputeGroupAccuracy:
    """Test group-aware classification accuracy."""

    def test_perfect_classification(self):
        """When similarity perfectly separates labels, accuracy should be 1.0."""
        from model.token_text_encoder import LearnableLabelBank

        # Use real label bank for encoding
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        label_bank = LearnableLabelBank(
            num_heads=4, num_queries=4, num_prototypes=1, device=device
        )
        label_bank.train(False)

        # Encode labels to get their text embeddings
        test_labels = ['walking', 'sitting', 'running']
        with torch.no_grad():
            label_embs = label_bank.encode(test_labels, normalize=True)

        # Use the label embeddings as "perfect" IMU embeddings
        imu_embs = label_embs.clone()

        metrics = compute_group_accuracy(
            imu_embs, label_bank,
            query_labels=test_labels, return_mrr=True
        )
        assert metrics['accuracy'] == 1.0, f"Expected accuracy=1.0, got {metrics['accuracy']}"
        assert metrics['mrr'] == 1.0, f"Expected MRR=1.0, got {metrics['mrr']}"

    def test_accuracy_returns_f1(self):
        """Verify F1 scores are included in the output."""
        from model.token_text_encoder import LearnableLabelBank

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        label_bank = LearnableLabelBank(
            num_heads=4, num_queries=4, num_prototypes=1, device=device
        )
        label_bank.train(False)

        test_labels = ['walking', 'sitting']
        with torch.no_grad():
            label_embs = label_bank.encode(test_labels, normalize=True)

        metrics = compute_group_accuracy(
            label_embs, label_bank, query_labels=test_labels
        )
        assert 'f1_macro' in metrics
        assert 'f1_weighted' in metrics
