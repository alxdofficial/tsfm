"""
Semantic alignment modules for IMU activity recognition.

Transforms patch-level encoder outputs into a single semantic representation
that can be aligned with text embeddings via contrastive learning.
"""

import torch
import torch.nn as nn
from typing import Optional


class CrossChannelFusion(nn.Module):
    """
    Perceiver-style cross-channel fusion using learnable bottleneck tokens.

    Reduces (batch, patches, channels, d_model) to (batch, patches, d_model_fused)
    by attending from bottleneck queries to all channel representations.
    """

    def __init__(
        self,
        d_model: int,
        d_model_fused: int,
        num_bottlenecks: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Input dimension per channel
            d_model_fused: Output fused dimension
            num_bottlenecks: Number of learnable bottleneck tokens per patch
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.num_bottlenecks = num_bottlenecks
        self.d_model_fused = d_model_fused

        # Learnable bottleneck queries (stronger initialization for learnable tokens)
        self.bottleneck_tokens = nn.Parameter(torch.randn(num_bottlenecks, d_model_fused) * 0.02)

        # Cross-attention: bottleneck queries attend to channel keys/values
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model_fused,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Input projection (project channel features to d_model_fused)
        self.input_proj = nn.Linear(d_model, d_model_fused)

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model_fused, d_model_fused * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model_fused * 4, d_model_fused),
            nn.Dropout(dropout)
        )

        # Layer normalization (reduced from 2 to 1 for better gradient flow)
        self.norm1 = nn.LayerNorm(d_model_fused)

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input features (batch, patches, channels, d_model)
            channel_mask: Optional mask for padding channels (batch, channels)
                         True = valid, False = padding

        Returns:
            Fused features (batch, patches, d_model_fused) if num_bottlenecks=1
            or (batch, patches, num_bottlenecks, d_model_fused) if num_bottlenecks>1
        """
        batch_size, num_patches, num_channels, d_model = x.shape

        # Project input to d_model_fused
        x_proj = self.input_proj(x)  # (batch, patches, channels, d_model_fused)

        # Reshape for cross-attention: flatten patches and channels
        kv = x_proj.reshape(batch_size, num_patches * num_channels, self.d_model_fused)

        # Prepare bottleneck queries (replicate for batch and patches)
        queries = self.bottleneck_tokens.unsqueeze(0).unsqueeze(0)  # (1, 1, num_bottlenecks, d_model_fused)
        queries = queries.expand(batch_size, num_patches, -1, -1)  # (batch, patches, num_bottlenecks, d_model_fused)
        queries = queries.reshape(batch_size * num_patches, self.num_bottlenecks, self.d_model_fused)

        # Reshape kv for attention
        kv = kv.reshape(batch_size * num_patches, num_channels, self.d_model_fused)

        # Prepare attention mask if provided
        # key_padding_mask should be (batch*patches, num_channels) regardless of num_bottlenecks
        key_padding_mask = None
        if channel_mask is not None:
            # channel_mask: (batch, channels) -> expand to (batch*patches, channels)
            key_padding_mask = channel_mask.unsqueeze(1).expand(-1, num_patches, -1)
            key_padding_mask = key_padding_mask.reshape(batch_size * num_patches, num_channels)
            # Convert to attention mask format (True = ignore)
            key_padding_mask = ~key_padding_mask

        # Cross-attention
        attended, _ = self.cross_attention(
            query=queries,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask
        )

        # Residual connection and norm (after attention)
        attended = self.norm1(queries + attended)

        # Feedforward with residual (no norm to reduce gradient suppression)
        output = attended + self.ffn(attended)

        # Reshape back
        output = output.reshape(batch_size, num_patches, self.num_bottlenecks, self.d_model_fused)

        # Handle multiple bottlenecks by flattening into patch dimension
        if self.num_bottlenecks == 1:
            output = output.squeeze(2)  # (batch, patches, d_model_fused)
        else:
            # Flatten bottlenecks into patch dimension for temporal processing
            # (batch, patches, num_bottlenecks, d_model_fused) → (batch, patches*num_bottlenecks, d_model_fused)
            # Example: (8, 10, 4, 384) → (8, 40, 384)
            output = output.reshape(batch_size, num_patches * self.num_bottlenecks, self.d_model_fused)

        return output


class TemporalAttention(nn.Module):
    """
    Temporal attention over patches using standard transformer encoder.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Feature dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
        """
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(
        self,
        x: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input features (batch, patches, d_model)
            patch_mask: Optional mask for padding patches (batch, patches)
                       True = valid, False = padding

        Returns:
            Attended features (batch, patches, d_model)
        """
        # Convert mask format if provided (True = ignore for transformer)
        src_key_padding_mask = ~patch_mask if patch_mask is not None else None

        return self.transformer(x, src_key_padding_mask=src_key_padding_mask)


class CLSAttentionPooling(nn.Module):
    """
    Attention pooling using a learnable CLS token.

    Prepends a CLS token to the sequence and extracts it after attention.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        # Learnable CLS token (stronger initialization for learnable tokens)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Single transformer layer for CLS attention
        self.attention = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

    def forward(
        self,
        x: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input features (batch, patches, d_model)
            patch_mask: Optional mask for padding patches (batch, patches)
                       True = valid, False = padding

        Returns:
            Pooled representation (batch, d_model)
        """
        batch_size = x.shape[0]

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        x_with_cls = torch.cat([cls_tokens, x], dim=1)  # (batch, 1+patches, d_model)

        # Prepare mask (CLS token is never masked)
        if patch_mask is not None:
            cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=x.device)
            mask_with_cls = torch.cat([cls_mask, patch_mask], dim=1)  # (batch, 1+patches)
            src_key_padding_mask = ~mask_with_cls
        else:
            src_key_padding_mask = None

        # Apply attention
        attended = self.attention(x_with_cls, src_key_padding_mask=src_key_padding_mask)

        # Extract CLS token
        cls_output = attended[:, 0, :]  # (batch, d_model)

        return cls_output


class ProjectionHead(nn.Module):
    """
    MLP projection head for mapping to shared semantic space.

    Uses 3-layer MLP architecture (standard in SimCLR, MoCo v3) for better
    semantic space transformation in contrastive learning.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super().__init__()

        # 3-layer MLP: input → hidden → hidden → output
        # No dropout: SimCLR/MoCo papers don't use dropout in projection head
        # Dropout can hurt contrastive learning by adding noise to similarity computation
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),  # Additional hidden layer
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Initialize weights with Xavier uniform for better gradient flow
        nn.init.xavier_uniform_(self.mlp[0].weight, gain=1.0)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.xavier_uniform_(self.mlp[2].weight, gain=1.0)
        nn.init.zeros_(self.mlp[2].bias)
        nn.init.xavier_uniform_(self.mlp[4].weight, gain=1.0)
        nn.init.zeros_(self.mlp[4].bias)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Args:
            x: Input features (batch, input_dim)
            normalize: Whether to L2-normalize output

        Returns:
            Projected embeddings (batch, output_dim)
        """
        import torch.nn.functional as F

        x = self.mlp(x)

        # L2 normalize embeddings to unit length (standard in contrastive learning)
        # This prevents magnitude explosion AND avoids gradient issues in the loss
        # By normalizing here instead of in the loss, gradients don't flow through normalization
        if normalize:
            x = F.normalize(x, p=2, dim=-1)

        return x


class SemanticAlignmentHead(nn.Module):
    """
    Complete semantic alignment pipeline.

    Takes encoder output (batch, patches, channels, d_model) and produces
    a single semantic embedding (batch, output_dim) aligned with text space.
    """

    def __init__(
        self,
        d_model: int,
        d_model_fused: int = 256,
        output_dim: int = 256,
        num_bottlenecks: int = 1,
        num_temporal_layers: int = 2,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Input dimension per channel from encoder
            d_model_fused: Dimension after cross-channel fusion
            output_dim: Final embedding dimension
            num_bottlenecks: Number of bottleneck tokens for fusion
            num_temporal_layers: Number of temporal attention layers
            num_heads: Number of attention heads
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.cross_channel_fusion = CrossChannelFusion(
            d_model=d_model,
            d_model_fused=d_model_fused,
            num_bottlenecks=num_bottlenecks,
            num_heads=num_heads,
            dropout=dropout
        )

        self.temporal_attention = TemporalAttention(
            d_model=d_model_fused,
            num_heads=num_heads,
            num_layers=num_temporal_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.attention_pooling = CLSAttentionPooling(
            d_model=d_model_fused,
            num_heads=num_heads,
            dropout=dropout
        )

        self.projection_head = ProjectionHead(
            input_dim=d_model_fused,
            hidden_dim=d_model_fused * 2,
            output_dim=output_dim,
            dropout=dropout
        )

    def forward(
        self,
        encoder_output: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Args:
            encoder_output: Encoder features (batch, patches, channels, d_model)
            channel_mask: Optional mask for padding channels (batch, channels)
            patch_mask: Optional mask for padding patches (batch, patches)
            normalize: Whether to L2-normalize final embedding

        Returns:
            Semantic embedding (batch, output_dim)
        """
        # Cross-channel fusion
        fused = self.cross_channel_fusion(encoder_output, channel_mask)  # (batch, patches, d_model_fused) or (batch, patches*num_bottlenecks, d_model_fused)

        # Expand patch_mask if using multiple bottlenecks
        if self.cross_channel_fusion.num_bottlenecks > 1 and patch_mask is not None:
            # Repeat mask for each bottleneck: (batch, patches) → (batch, patches*num_bottlenecks)
            batch_size, num_patches = patch_mask.shape
            patch_mask = patch_mask.unsqueeze(2).expand(
                -1, -1, self.cross_channel_fusion.num_bottlenecks
            )
            patch_mask = patch_mask.reshape(batch_size, -1)  # (batch, patches*num_bottlenecks)

        # Temporal attention (TransformerEncoder has internal residuals - no external residual needed)
        temporal = self.temporal_attention(fused, patch_mask)  # (batch, patches, d_model_fused)

        # Attention pooling (transformer already has LayerNorm internally)
        pooled = self.attention_pooling(temporal, patch_mask)  # (batch, d_model_fused)

        # Project to semantic space
        embedding = self.projection_head(pooled, normalize=normalize)  # (batch, output_dim)

        return embedding
