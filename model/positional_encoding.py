"""
Positional Encoding module for IMU Activity Recognition Encoder

Provides two types of positional information:
1. Temporal encoding: Position of each patch in the sequence
2. Channel semantic encoding: Semantic meaning of each channel (e.g., "accelerometer x-axis")
"""

import math

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict
import warnings


class TemporalPositionalEncoding(nn.Module):
    """
    Temporal positional encoding for patch sequences.

    Uses sinusoidal encoding similar to the original Transformer paper,
    which allows the model to understand the temporal order of patches.

    The encoding is added to patch features to inject temporal information.
    """

    def __init__(
        self,
        d_model: int,
        max_patches: int = 5000,
        init_scale: float = 0.1
    ):
        """
        Args:
            d_model: Feature dimension
            max_patches: Maximum number of patches to support
            init_scale: Initial scaling factor for positional encodings
        """
        super().__init__()

        self.d_model = d_model
        self.max_patches = max_patches

        # Create sinusoidal positional encoding
        pe = self._create_sinusoidal_encoding(max_patches, d_model)
        self.register_buffer('pe', pe)  # (max_patches, d_model)

        # Learnable scaling factor
        self.scale = nn.Parameter(torch.tensor(init_scale))

    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """
        Create sinusoidal positional encoding.

        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add temporal positional encoding to input features.

        Args:
            x: Input tensor of shape (batch_size, num_patches, num_channels, d_model)
               or (batch_size, num_patches, d_model)

        Returns:
            Same shape as input with positional encoding added
        """
        if x.dim() == 4:
            # Shape: (batch, patches, channels, d_model)
            batch_size, num_patches, num_channels, d_model = x.shape

            if num_patches > self.max_patches:
                raise ValueError(
                    f"Input has {num_patches} patches but max_patches is {self.max_patches}"
                )

            # Get positional encoding for this sequence length
            # Shape: (num_patches, d_model)
            pe = self.pe[:num_patches, :]

            # Reshape to broadcast: (1, num_patches, 1, d_model)
            pe = pe.unsqueeze(0).unsqueeze(2)

            # Add scaled positional encoding
            return x + self.scale * pe

        elif x.dim() == 3:
            # Shape: (batch, patches, d_model)
            batch_size, num_patches, d_model = x.shape

            if num_patches > self.max_patches:
                raise ValueError(
                    f"Input has {num_patches} patches but max_patches is {self.max_patches}"
                )

            # Get positional encoding
            pe = self.pe[:num_patches, :]

            # Reshape to broadcast: (1, num_patches, d_model)
            pe = pe.unsqueeze(0)

            # Add scaled positional encoding
            return x + self.scale * pe

        else:
            raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D")


class ChannelSemanticEncoding(nn.Module):
    """
    Channel semantic encoding using language embeddings.

    Each channel is encoded based on its semantic meaning (e.g., "accelerometer x-axis"),
    which helps the model understand what type of sensor data each channel represents.

    Uses Sentence-BERT to encode channel descriptions into dense vectors, with a learnable
    projection layer for task-specific adaptation.
    """

    def __init__(
        self,
        d_model: int,
        init_scale: float = 0.1,
        sentence_bert_model: str = 'all-MiniLM-L6-v2'
    ):
        """
        Args:
            d_model: Feature dimension
            init_scale: Initial scaling factor for channel encodings
            sentence_bert_model: Name of Sentence-BERT model to use
        """
        super().__init__()

        self.d_model = d_model
        self.sentence_bert_model = sentence_bert_model

        # Initialize Sentence-BERT lazily (only when needed)
        # _IMPORT_FAILED sentinel distinguishes "not tried" from "import failed"
        self._encoder = None
        self._sbert_dim = None
        self._import_failed = False
        self._fallback_cache: Dict[int, torch.Tensor] = {}  # num_channels -> stable fallback

        # Learnable scaling factor
        self.scale = nn.Parameter(torch.tensor(init_scale))

        # Learnable padding embedding for padded channels
        # Will match Sentence-BERT dimension (384 for all-MiniLM-L6-v2)
        self.pad_channel_embedding = nn.Parameter(torch.randn(d_model) / math.sqrt(d_model))

        # Learnable projection layer after frozen SentenceBERT
        # This allows task-specific adaptation while preserving pretrained semantics
        # Always enabled - experiments showed better zero-shot generalization with projection
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # Initialize projection close to identity (residual-style)
        nn.init.zeros_(self.projection[2].weight)
        nn.init.zeros_(self.projection[2].bias)

        # Cache for sentence BERT embeddings (stores frozen embeddings before projection)
        self._sbert_embedding_cache: Dict[str, torch.Tensor] = {}

    def _get_encoder(self):
        """Lazy initialization of Sentence-BERT encoder."""
        if self._import_failed:
            return None
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                # Use object.__setattr__ to avoid registering as nn.Module submodule
                # This prevents SentenceBERT weights from being saved in checkpoints
                encoder = SentenceTransformer(self.sentence_bert_model)
                object.__setattr__(self, '_encoder', encoder)
                self._sbert_dim = self._encoder.get_sentence_embedding_dimension()

                # Verify d_model matches Sentence-BERT dimension
                if self.d_model != self._sbert_dim:
                    raise ValueError(
                        f"d_model ({self.d_model}) must match Sentence-BERT dimension ({self._sbert_dim}). "
                        f"For {self.sentence_bert_model}, use d_model={self._sbert_dim}"
                    )

                # Freeze Sentence-BERT parameters - no need to fine-tune
                for param in self._encoder.parameters():
                    param.requires_grad = False
            except ImportError:
                warnings.warn(
                    "sentence-transformers not installed. Channel semantic encoding will be disabled. "
                    "Install with: pip install sentence-transformers"
                )
                self._import_failed = True
        return self._encoder

    def encode_channel_descriptions(
        self,
        channel_descriptions: List[str]
    ) -> torch.Tensor:
        """
        Encode channel descriptions into dense vectors.

        Args:
            channel_descriptions: List of channel descriptions
                e.g., ["accelerometer x-axis", "gyroscope y-axis", "[PAD]", ...]

        Returns:
            Channel encodings of shape (num_channels, d_model)
        """
        encoder = self._get_encoder()

        if encoder is None:
            # Fallback: Use stable random embeddings (seeded, cached by num_channels)
            num_channels = len(channel_descriptions)
            if num_channels not in self._fallback_cache:
                gen = torch.Generator().manual_seed(42)
                self._fallback_cache[num_channels] = torch.randn(
                    num_channels, self.d_model, generator=gen
                ) * 0.01
            return self._fallback_cache[num_channels]

        # Separate padded channels from valid channels
        embeddings_list = []
        non_pad_descriptions = []
        non_pad_indices = []

        for i, desc in enumerate(channel_descriptions):
            if desc == "[PAD]":
                embeddings_list.append(None)  # Placeholder
            else:
                non_pad_descriptions.append(desc)
                non_pad_indices.append(i)
                embeddings_list.append(None)  # Placeholder

        # Encode non-padded channels using Sentence-BERT (with per-string caching)
        if non_pad_descriptions:
            # Find which descriptions need encoding (not in cache)
            to_encode = []
            to_encode_indices = []
            for i, desc in enumerate(non_pad_descriptions):
                if desc not in self._sbert_embedding_cache:
                    to_encode.append(desc)
                    to_encode_indices.append(i)

            # Encode only new descriptions
            if to_encode:
                with torch.no_grad():
                    new_embeddings = encoder.encode(to_encode, convert_to_tensor=True)
                # Cache each individual description
                for desc, emb in zip(to_encode, new_embeddings):
                    self._sbert_embedding_cache[desc] = emb

            # Retrieve all embeddings from cache
            for i, idx in enumerate(non_pad_indices):
                desc = non_pad_descriptions[i]
                embeddings_list[idx] = self._sbert_embedding_cache[desc]

        # Fill in padded embeddings with learned pad token
        for i in range(len(embeddings_list)):
            if embeddings_list[i] is None:
                embeddings_list[i] = self.pad_channel_embedding

        # Stack all embeddings
        embeddings = torch.stack(embeddings_list)

        # Apply learnable projection to adapt frozen SentenceBERT embeddings for the task
        # Ensure embeddings are on same device as projection layer
        proj_device = next(self.projection.parameters()).device
        embeddings = embeddings.to(proj_device)
        # Residual connection: embeddings + projection(embeddings)
        # Initialized to identity, so starts with original embeddings
        embeddings = embeddings + self.projection(embeddings)

        return embeddings

    def forward(
        self,
        x: torch.Tensor,
        channel_descriptions: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Add channel semantic encoding to input features.

        Args:
            x: Input tensor of shape (batch_size, num_patches, num_channels, d_model)
            channel_descriptions: Optional list of channel descriptions.
                If None, uses generic channel encodings.

        Returns:
            Same shape as input with channel encoding added
        """
        batch_size, num_patches, num_channels, d_model = x.shape

        if channel_descriptions is not None and len(channel_descriptions) != num_channels:
            raise ValueError(
                f"Number of channel descriptions ({len(channel_descriptions)}) "
                f"does not match number of channels ({num_channels})"
            )

        # Get channel encodings
        if channel_descriptions is not None:
            channel_enc = self.encode_channel_descriptions(channel_descriptions)
        else:
            # Use zeros as fallback when no channel descriptions provided
            channel_enc = torch.zeros(num_channels, self.d_model)

        # Ensure channel encoding is on same device as input
        channel_enc = channel_enc.to(x.device)

        # Reshape to broadcast: (1, 1, num_channels, d_model)
        channel_enc = channel_enc.unsqueeze(0).unsqueeze(0)

        # Add scaled channel encoding
        return x + self.scale * channel_enc


class IMUPositionalEncoding(nn.Module):
    """
    Combined positional encoding for IMU data.

    Combines:
    1. Temporal encoding: Position of each patch in time
    2. Channel semantic encoding: Meaning of each channel

    Both encodings are learned to scale appropriately during training.
    """

    def __init__(
        self,
        d_model: int,
        max_patches: int = 5000,
        temporal_init_scale: float = 0.1,
        channel_init_scale: float = 0.1,
        sentence_bert_model: str = 'all-MiniLM-L6-v2',
        use_channel_encoding: bool = True
    ):
        """
        Args:
            d_model: Feature dimension
            max_patches: Maximum number of patches
            temporal_init_scale: Initial scale for temporal encoding
            channel_init_scale: Initial scale for channel encoding
            sentence_bert_model: Sentence-BERT model name
            use_channel_encoding: Whether to use channel semantic encoding
        """
        super().__init__()

        self.d_model = d_model
        self.use_channel_encoding = use_channel_encoding

        # Temporal encoding
        self.temporal_encoding = TemporalPositionalEncoding(
            d_model=d_model,
            max_patches=max_patches,
            init_scale=temporal_init_scale
        )

        # Channel semantic encoding (always with learnable projection for better generalization)
        if use_channel_encoding:
            self.channel_encoding = ChannelSemanticEncoding(
                d_model=d_model,
                init_scale=channel_init_scale,
                sentence_bert_model=sentence_bert_model
            )
        else:
            self.channel_encoding = None

    def forward(
        self,
        x: torch.Tensor,
        channel_descriptions: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Add positional encodings to input features.

        Args:
            x: Input tensor of shape (batch_size, num_patches, num_channels, d_model)
            channel_descriptions: Optional channel descriptions for semantic encoding

        Returns:
            Features with positional encodings added
        """
        # Add temporal encoding
        x = self.temporal_encoding(x)

        # Add channel semantic encoding
        if self.use_channel_encoding and self.channel_encoding is not None:
            x = self.channel_encoding(x, channel_descriptions)

        return x


def test_positional_encoding():
    """Test positional encoding modules."""
    print("Testing Positional Encoding...")

    # Test 1: Temporal encoding
    print("\n1. Testing temporal positional encoding...")
    d_model = 128
    temp_enc = TemporalPositionalEncoding(d_model=d_model, max_patches=100, init_scale=0.1)

    # 4D input: (batch, patches, channels, d_model)
    x_4d = torch.randn(2, 10, 9, d_model)
    out_4d = temp_enc(x_4d)
    assert out_4d.shape == x_4d.shape
    print(f"   ✓ 4D input shape: {x_4d.shape} -> {out_4d.shape}")

    # 3D input: (batch, patches, d_model)
    x_3d = torch.randn(2, 10, d_model)
    out_3d = temp_enc(x_3d)
    assert out_3d.shape == x_3d.shape
    print(f"   ✓ 3D input shape: {x_3d.shape} -> {out_3d.shape}")

    # Test that encoding changes with position
    x = torch.zeros(1, 5, d_model)
    out = temp_enc(x)
    # Different positions should have different encodings
    assert not torch.allclose(out[0, 0, :], out[0, 1, :])
    assert not torch.allclose(out[0, 0, :], out[0, 4, :])
    print(f"   ✓ Encoding varies by position")

    # Test 2: Channel semantic encoding (without Sentence-BERT)
    print("\n2. Testing channel semantic encoding (fallback mode)...")
    chan_enc = ChannelSemanticEncoding(d_model=d_model, init_scale=0.1)

    x = torch.randn(2, 10, 9, d_model)
    out = chan_enc(x, channel_descriptions=None)  # Use fallback
    assert out.shape == x.shape
    print(f"   ✓ Fallback mode works: {x.shape} -> {out.shape}")

    # Test 3: Combined encoding
    print("\n3. Testing combined IMU positional encoding...")
    pos_enc = IMUPositionalEncoding(
        d_model=d_model,
        max_patches=100,
        temporal_init_scale=0.1,
        channel_init_scale=0.1,
        use_channel_encoding=True
    )

    x = torch.randn(4, 15, 9, d_model)
    out = pos_enc(x, channel_descriptions=None)
    assert out.shape == x.shape
    print(f"   ✓ Combined encoding: {x.shape} -> {out.shape}")

    # Test 4: Temporal encoding only (no channel encoding)
    print("\n4. Testing temporal encoding only...")
    pos_enc_temp_only = IMUPositionalEncoding(
        d_model=d_model,
        use_channel_encoding=False
    )

    x = torch.randn(4, 15, 9, d_model)
    out = pos_enc_temp_only(x)
    assert out.shape == x.shape
    print(f"   ✓ Temporal only: {x.shape} -> {out.shape}")

    # Test 5: Learnable scaling
    print("\n5. Testing learnable scaling...")
    temp_enc = TemporalPositionalEncoding(d_model=d_model, init_scale=0.1)
    initial_scale = temp_enc.scale.item()
    assert abs(initial_scale - 0.1) < 1e-6
    print(f"   ✓ Initial scale: {initial_scale}")

    # Simulate training step
    x = torch.randn(2, 10, d_model, requires_grad=True)
    out = temp_enc(x)
    loss = out.sum()
    loss.backward()
    assert temp_enc.scale.grad is not None
    print(f"   ✓ Scale is learnable (has gradient)")

    # Test 6: Different patch lengths
    print("\n6. Testing variable patch lengths...")
    pos_enc = IMUPositionalEncoding(d_model=d_model, max_patches=100)

    for num_patches in [5, 10, 20, 50]:
        x = torch.randn(2, num_patches, 9, d_model)
        out = pos_enc(x)
        assert out.shape == x.shape
    print(f"   ✓ Tested patch lengths: 5, 10, 20, 50")

    # Test 7: Different channel counts
    print("\n7. Testing variable channel counts...")
    for num_channels in [6, 9, 23, 30, 40]:
        x = torch.randn(2, 10, num_channels, d_model)
        out = pos_enc(x)
        assert out.shape == x.shape
    print(f"   ✓ Tested channel counts: 6, 9, 23, 30, 40")

    print("\n" + "="*80)
    print("✓ ALL POSITIONAL ENCODING TESTS PASSED!")
    print("="*80)


if __name__ == "__main__":
    test_positional_encoding()
