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

from model.encoder import IMUActivityRecognitionEncoder
from model.semantic_alignment import (
    SemanticAlignmentHead,
)
from model.token_text_encoder import (
    LearnableLabelBank,
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
        feature_extractor_type='physical_filterbank',
        dft_size=64,  # patches in these tests are length 64
    )
    head = SemanticAlignmentHead(
        d_model=128,
        d_model_fused=256,
        output_dim=384,
        num_heads=8,
        dropout=0.1,
    )
    return encoder.to(device), head.to(device)


class TestModelConstruction:
    """Test model construction and forward pass produce correct shapes."""

    def test_encoder_output_shape(self, encoder_and_head, device):
        encoder, _ = encoder_and_head
        B, P, T, C = 2, 5, 64, 9
        patches = torch.randn(B, P, T, C, device=device)
        output = encoder(patches, sampling_rate_hz=50.0, patch_len_samples=64)
        assert output.shape == (B, P, C, 128), f"Expected (2, 5, 9, 128), got {output.shape}"

    def test_semantic_head_output_shape(self, encoder_and_head, device):
        encoder, head = encoder_and_head
        B, P, T, C = 2, 5, 64, 9
        patches = torch.randn(B, P, T, C, device=device)
        encoder_out = encoder(patches, sampling_rate_hz=50.0, patch_len_samples=64)
        embedding = head(encoder_out)
        # per-patch head: (batch, patches, output_dim)
        assert embedding.shape == (B, P, 384), f"Expected (2, 5, 384), got {embedding.shape}"

    def test_embedding_is_l2_normalized(self, encoder_and_head, device):
        encoder, head = encoder_and_head
        patches = torch.randn(2, 5, 64, 9, device=device)
        encoder_out = encoder(patches, sampling_rate_hz=50.0, patch_len_samples=64)
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

    def test_encode_shape(self, device):
        label_bank = LearnableLabelBank(device=device)
        label_bank.train(False)
        labels = ['walking', 'running', 'sitting']
        with torch.no_grad():
            embeddings = label_bank.encode(labels, normalize=True)
        assert embeddings.shape == (3, 384), f"Expected (3, 384), got {embeddings.shape}"

    def test_label_bank_embeddings_normalized(self, device):
        label_bank = LearnableLabelBank(device=device)
        label_bank.train(False)
        with torch.no_grad():
            embeddings = label_bank.encode(['walking', 'running'], normalize=True)
        norms = embeddings.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_encode_deterministic(self, device):
        """Frozen mean-pool: two fresh banks produce identical encodings."""
        with torch.no_grad():
            emb1 = LearnableLabelBank(device=device).encode(['walking', 'running'], normalize=True)
            emb2 = LearnableLabelBank(device=device).encode(['walking', 'running'], normalize=True)
        assert torch.allclose(emb1, emb2, atol=1e-6)


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
            features = encoder(patches, sampling_rate_hz=50.0, patch_len_samples=64)
            emb = head(features)
        # per-patch head: (batch, patches, output_dim)
        assert emb.shape == (1, 3, 384)
        assert not torch.isnan(emb).any()
