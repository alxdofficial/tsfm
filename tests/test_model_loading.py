"""Test 1: Model and label bank loading from checkpoint.

Verifies that model + label bank loading works correctly across
all the evaluation scripts that duplicate this logic.
"""

import sys
from pathlib import Path
import torch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.models.imu_activity_recognition_encoder.encoder import IMUActivityRecognitionEncoder
from tools.models.imu_activity_recognition_encoder.semantic_alignment import (
    SemanticAlignmentHead,
)
from tools.models.imu_activity_recognition_encoder.token_text_encoder import (
    LearnableLabelBank,
    LabelAttentionPooling,
    MultiPrototypeLabelPooling,
)


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def encoder_and_head(device):
    """Create a minimal encoder + semantic head for testing."""
    encoder = IMUActivityRecognitionEncoder(
        d_model=128,
        num_heads=8,
        num_temporal_layers=2,
        dim_feedforward=512,
        dropout=0.1,
    )
    head = SemanticAlignmentHead(
        d_model=128,
        d_model_fused=256,
        output_dim=384,
        num_temporal_layers=2,
        num_heads=8,
        dim_feedforward=1024,
        dropout=0.1,
    )
    return encoder.to(device), head.to(device)


class TestModelConstruction:
    """Test model construction and forward pass produce correct shapes."""

    def test_encoder_output_shape(self, encoder_and_head, device):
        encoder, _ = encoder_and_head
        B, P, T, C = 2, 5, 64, 9
        patches = torch.randn(B, P, T, C, device=device)
        output = encoder(patches)
        assert output.shape == (B, P, C, 128), f"Expected (2, 5, 9, 128), got {output.shape}"

    def test_semantic_head_output_shape(self, encoder_and_head, device):
        encoder, head = encoder_and_head
        B, P, T, C = 2, 5, 64, 9
        patches = torch.randn(B, P, T, C, device=device)
        encoder_out = encoder(patches)
        embedding = head(encoder_out)
        assert embedding.shape == (B, 384), f"Expected (2, 384), got {embedding.shape}"

    def test_embedding_is_l2_normalized(self, encoder_and_head, device):
        encoder, head = encoder_and_head
        patches = torch.randn(2, 5, 64, 9, device=device)
        encoder_out = encoder(patches)
        embedding = head(encoder_out)
        norms = embedding.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"Embeddings not L2-normalized: norms={norms}"

    def test_model_set_to_inference_mode(self, encoder_and_head):
        encoder, head = encoder_and_head
        # Switch to inference mode
        encoder.train(False)
        head.train(False)
        assert not encoder.training
        assert not head.training


class TestLabelBankLoading:
    """Test LearnableLabelBank creation and encoding."""

    def test_single_prototype_encode_shape(self, device):
        label_bank = LearnableLabelBank(
            num_heads=4,
            num_queries=4,
            num_prototypes=1,
            dropout=0.1,
            device=device,
        )
        label_bank.train(False)
        labels = ['walking', 'running', 'sitting']
        with torch.no_grad():
            embeddings = label_bank.encode(labels, normalize=True)
        assert embeddings.shape == (3, 384), f"Expected (3, 384), got {embeddings.shape}"

    def test_multi_prototype_encode_shape(self, device):
        K = 3
        label_bank = LearnableLabelBank(
            num_heads=4,
            num_queries=4,
            num_prototypes=K,
            dropout=0.1,
            device=device,
        )
        label_bank.train(False)
        labels = ['walking', 'running', 'sitting', 'standing']
        with torch.no_grad():
            embeddings = label_bank.encode(labels, normalize=True)
        assert embeddings.shape == (4, K, 384), f"Expected (4, 3, 384), got {embeddings.shape}"

    def test_label_bank_embeddings_normalized(self, device):
        label_bank = LearnableLabelBank(
            num_heads=4, num_queries=4, num_prototypes=1, device=device
        )
        label_bank.train(False)
        with torch.no_grad():
            embeddings = label_bank.encode(['walking', 'running'], normalize=True)
        norms = embeddings.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_multi_prototype_embeddings_normalized(self, device):
        label_bank = LearnableLabelBank(
            num_heads=4, num_queries=4, num_prototypes=3, device=device
        )
        label_bank.train(False)
        with torch.no_grad():
            embeddings = label_bank.encode(['walking', 'running'], normalize=True)
        # Shape: (2, 3, 384) -- norm over last dim
        norms = embeddings.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_label_bank_state_dict_roundtrip(self, device):
        """Verify state_dict save/load produces identical encodings."""
        label_bank = LearnableLabelBank(
            num_heads=4, num_queries=4, num_prototypes=1, device=device
        )
        label_bank.train(False)

        with torch.no_grad():
            emb_before = label_bank.encode(['walking', 'running'], normalize=True)

        state = label_bank.state_dict()

        # Create fresh bank and load state
        label_bank2 = LearnableLabelBank(
            num_heads=4, num_queries=4, num_prototypes=1, device=device
        )
        label_bank2.load_state_dict(state)
        label_bank2.train(False)

        with torch.no_grad():
            emb_after = label_bank2.encode(['walking', 'running'], normalize=True)

        assert torch.allclose(emb_before, emb_after, atol=1e-6), \
            "State dict roundtrip changed label bank output"


class TestCheckpointStateDict:
    """Test state_dict loading behavior with unexpected keys."""

    def test_strict_false_handles_extra_keys(self, encoder_and_head, device):
        """Verify strict=False allows loading checkpoints with extra keys."""
        encoder, head = encoder_and_head
        state = encoder.state_dict()
        # Add a fake extra key (simulating channel_encoding from old checkpoint)
        state['positional_encoding.channel_encoding.FAKE_KEY'] = torch.zeros(10)
        # Should not raise with strict=False
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        assert 'positional_encoding.channel_encoding.FAKE_KEY' in unexpected

    def test_model_produces_output_after_state_load(self, encoder_and_head, device):
        encoder, head = encoder_and_head
        # Save and reload
        enc_state = encoder.state_dict()
        head_state = head.state_dict()
        encoder.load_state_dict(enc_state)
        head.load_state_dict(head_state)
        encoder.train(False)
        head.train(False)

        patches = torch.randn(1, 3, 64, 6, device=device)
        with torch.no_grad():
            features = encoder(patches)
            emb = head(features)
        assert emb.shape == (1, 384)
        assert not torch.isnan(emb).any()
