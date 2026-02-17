"""Test 6: End-to-end encoder forward pass.

Verifies encoder output shapes, semantic head output, mask behavior,
and multi-prototype label bank encoding.
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
    ProjectionHead,
)
from model.token_text_encoder import (
    LearnableLabelBank,
)


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def encoder(device):
    enc = IMUActivityRecognitionEncoder(
        d_model=128,
        num_heads=8,
        num_temporal_layers=2,
        dim_feedforward=512,
        dropout=0.0,  # No dropout for deterministic tests
        target_patch_size=64,
    ).to(device)
    enc.train(False)
    return enc


@pytest.fixture
def head(device):
    h = SemanticAlignmentHead(
        d_model=128,
        d_model_fused=256,
        output_dim=384,
        num_temporal_layers=2,
        num_heads=8,
        dim_feedforward=1024,
        dropout=0.0,
    ).to(device)
    h.train(False)
    return h


class TestEncoderForwardPass:
    """Test encoder produces correct output shapes."""

    def test_basic_output_shape(self, encoder, device):
        B, P, T, C = 2, 5, 64, 9
        patches = torch.randn(B, P, T, C, device=device)
        with torch.no_grad():
            output = encoder(patches)
        assert output.shape == (B, P, C, 128)

    def test_different_channel_counts(self, encoder, device):
        """Test with different numbers of channels."""
        for C in [3, 6, 9, 12]:
            patches = torch.randn(1, 3, 64, C, device=device)
            with torch.no_grad():
                output = encoder(patches)
            assert output.shape == (1, 3, C, 128), f"Failed for C={C}"

    def test_different_patch_counts(self, encoder, device):
        """Test with different numbers of patches."""
        for P in [1, 5, 10, 20]:
            patches = torch.randn(1, P, 64, 6, device=device)
            with torch.no_grad():
                output = encoder(patches)
            assert output.shape == (1, P, 6, 128), f"Failed for P={P}"

    def test_no_nan_in_output(self, encoder, device):
        patches = torch.randn(2, 5, 64, 9, device=device)
        with torch.no_grad():
            output = encoder(patches)
        assert not torch.isnan(output).any(), "Encoder output contains NaN"

    def test_gradient_flow(self, device):
        """Verify gradients flow from loss to input."""
        enc = IMUActivityRecognitionEncoder(d_model=128, num_temporal_layers=2).to(device)
        patches = torch.randn(2, 3, 64, 6, device=device, requires_grad=True)
        output = enc(patches)
        loss = output.sum()
        loss.backward()
        assert patches.grad is not None
        assert not torch.isnan(patches.grad).any()


class TestSemanticHeadOutput:
    """Test semantic head produces correct output."""

    def test_output_shape_384(self, encoder, head, device):
        patches = torch.randn(2, 5, 64, 9, device=device)
        with torch.no_grad():
            enc_out = encoder(patches)
            embedding = head(enc_out)
        assert embedding.shape == (2, 384)

    def test_output_is_normalized(self, encoder, head, device):
        patches = torch.randn(2, 5, 64, 9, device=device)
        with torch.no_grad():
            enc_out = encoder(patches)
            embedding = head(enc_out)
        norms = embedding.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestMaskBehavior:
    """Test that masks affect encoder output correctly."""

    def test_patch_attention_mask(self, encoder, device):
        """Padded patches should not use CNN features (replaced by pad_token before transformer)."""
        B, P, T, C = 2, 8, 64, 6
        patches = torch.randn(B, P, T, C, device=device)
        mask = torch.ones(B, P, dtype=torch.bool, device=device)
        mask[:, 5:] = False  # Last 3 patches are padding

        with torch.no_grad():
            output_masked = encoder(patches, patch_attention_mask=mask)
            output_no_mask = encoder(patches)

        # Valid patches (0-4) should differ between masked and unmasked runs
        # because the transformer attention changes when padding is introduced
        # Main check: no NaN in output
        assert not torch.isnan(output_masked).any(), "Masked output contains NaN"
        # Padded patch features should be finite
        for i in range(B):
            assert torch.isfinite(output_masked[i, 5:]).all(), \
                "Padded patch features should be finite"

    def test_mae_mask(self, encoder, device):
        """MAE-masked patches should use mask_token."""
        B, P, T, C = 2, 8, 64, 6
        patches = torch.randn(B, P, T, C, device=device)
        mae_mask = torch.zeros(B, P, dtype=torch.bool, device=device)
        mae_mask[:, :3] = True  # First 3 patches are MAE-masked

        with torch.no_grad():
            output_masked = encoder(patches, mae_mask=mae_mask)

        # Main check: no NaN
        assert not torch.isnan(output_masked).any()


class TestMultiPrototypeLabelBank:
    """Test multi-prototype label bank encoding."""

    def test_k1_shape(self, device):
        bank = LearnableLabelBank(
            num_heads=4, num_queries=4, num_prototypes=1, device=device
        )
        bank.train(False)
        with torch.no_grad():
            emb = bank.encode(['walking', 'running', 'sitting'])
        assert emb.shape == (3, 384)

    def test_k3_shape(self, device):
        bank = LearnableLabelBank(
            num_heads=4, num_queries=4, num_prototypes=3, device=device
        )
        bank.train(False)
        with torch.no_grad():
            emb = bank.encode(['walking', 'running', 'sitting'])
        assert emb.shape == (3, 3, 384)

    def test_k3_normalized(self, device):
        bank = LearnableLabelBank(
            num_heads=4, num_queries=4, num_prototypes=3, device=device
        )
        bank.train(False)
        with torch.no_grad():
            emb = bank.encode(['walking', 'running'], normalize=True)
        norms = emb.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_different_labels_produce_different_embeddings(self, device):
        bank = LearnableLabelBank(
            num_heads=4, num_queries=4, num_prototypes=1, device=device
        )
        bank.train(False)
        with torch.no_grad():
            emb = bank.encode(['walking', 'sitting'])
        # Different activities should have different embeddings
        cos_sim = torch.dot(emb[0], emb[1]).item()
        assert cos_sim < 0.99, f"Different labels too similar: cos_sim={cos_sim}"


class TestProjectionHeadNormalization:
    """Test ProjectionHead L2 normalization."""

    def test_manual_normalization(self, device):
        """Verify current manual norm.clamp implementation produces unit vectors."""
        proj = ProjectionHead(input_dim=256, hidden_dim=512, output_dim=384).to(device)
        x = torch.randn(4, 256, device=device)
        with torch.no_grad():
            out = proj(x, normalize=True)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"ProjectionHead output not normalized: norms={norms}"

    def test_without_normalization(self, device):
        proj = ProjectionHead(input_dim=256, hidden_dim=512, output_dim=384).to(device)
        x = torch.randn(4, 256, device=device)
        with torch.no_grad():
            out = proj(x, normalize=False)
        norms = out.norm(dim=-1)
        # Without normalization, norms should NOT be exactly 1
        assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)
