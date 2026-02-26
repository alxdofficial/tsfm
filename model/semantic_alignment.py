"""
Semantic alignment modules for IMU activity recognition.

Transforms patch-level encoder outputs into a single semantic representation
that can be aligned with text embeddings via contrastive learning.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiQueryAttention(nn.Module):
    """
    Multi-query attention for sequence-to-vector transformation.

    Uses multiple learnable query tokens that:
    1. Cross-attend to input sequence
    2. Self-attend among themselves (optional)
    3. Concatenate and project to single output vector

    This is the core building block for both channel fusion and temporal pooling.

    Reference: Set Transformer (ICML 2019), Perceiver, PMA (Pooling by Multihead Attention)
    """

    def __init__(
        self,
        d_model: int,
        num_queries: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_self_attention: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.num_queries = num_queries
        self.use_self_attention = use_self_attention

        # Learnable query tokens (small init prevents representation collapse)
        self.queries = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)

        # Cross-attention: queries attend to input sequence
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Self-attention: queries coordinate with each other
        if use_self_attention:
            self.self_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            self.norm2 = nn.LayerNorm(d_model)

        # Output projection: concat queries → single vector
        self.out_proj = nn.Sequential(
            nn.Linear(d_model * num_queries, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        # Small init for stable training while preserving gradient flow
        # NOTE: zeros init kills gradients since d_input = d_output @ W.T = 0
        nn.init.normal_(self.out_proj[2].weight, std=0.01)
        nn.init.zeros_(self.out_proj[2].bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input (batch, seq_len, d_model)
            mask: Valid positions (batch, seq_len), True=valid, False=padding
            need_weights: Return attention weights for visualization

        Returns:
            output: (batch, d_model)
            attn_weights: (batch, num_queries, seq_len) if need_weights else None
        """
        B = x.shape[0]
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)
        key_padding_mask = ~mask if mask is not None else None

        # Cross-attention
        attended, attn_weights = self.cross_attn(
            query=queries, key=x, value=x,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights, average_attn_weights=True
        )
        attended = self.norm1(queries + attended)

        # Self-attention
        if self.use_self_attention:
            self_out, _ = self.self_attn(query=attended, key=attended, value=attended)
            attended = self.norm2(attended + self_out)

        # Combine: concat + project + residual
        out = self.out_proj(attended.reshape(B, -1))
        out = out + attended.mean(dim=1)

        return out, attn_weights if need_weights else None


class CrossChannelFusion(nn.Module):
    """
    Fuses channels into a single vector per patch using multi-query attention.

    Reduces (batch, patches, channels, d_model) to (batch, patches, d_model_fused).
    """

    def __init__(
        self,
        d_model: int,
        d_model_fused: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_queries: int = 4,
        use_self_attention: bool = True
    ):
        super().__init__()
        self.d_model_fused = d_model_fused
        self.num_queries = num_queries

        # Project input channels to fused dimension
        self.input_proj = nn.Linear(d_model, d_model_fused)

        # Multi-query attention for channel fusion
        self.attention = MultiQueryAttention(
            d_model=d_model_fused,
            num_queries=num_queries,
            num_heads=num_heads,
            dropout=dropout,
            use_self_attention=use_self_attention
        )

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: Input (batch, patches, channels, d_model)
            channel_mask: Optional mask (batch, channels), True=valid

        Returns:
            Fused (batch, patches, d_model_fused)
        """
        B, P, C, D = x.shape

        # Project and reshape for per-patch processing
        x_proj = self.input_proj(x)  # (B, P, C, d_fused)
        x_flat = x_proj.reshape(B * P, C, self.d_model_fused)

        # Expand mask for all patches
        mask_flat = None
        if channel_mask is not None:
            mask_flat = channel_mask.unsqueeze(1).expand(-1, P, -1).reshape(B * P, C)

        # Apply multi-query attention
        out, attn_weights = self.attention(x_flat, mask_flat, need_weights=return_attention_weights)

        # Reshape back to per-patch
        out = out.reshape(B, P, self.d_model_fused)

        if return_attention_weights:
            return out, attn_weights
        return out


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


class MultiQueryPooling(nn.Module):
    """
    Pools a sequence to a single vector using multi-query attention.

    Thin wrapper around MultiQueryAttention for API compatibility.
    """

    def __init__(
        self,
        d_model: int,
        num_queries: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_self_attention: bool = True
    ):
        super().__init__()
        self.num_queries = num_queries

        self.attention = MultiQueryAttention(
            d_model=d_model,
            num_queries=num_queries,
            num_heads=num_heads,
            dropout=dropout,
            use_self_attention=use_self_attention
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input (batch, seq_len, d_model)
            mask: Optional mask (batch, seq_len), True=valid

        Returns:
            Pooled (batch, d_model)
        """
        out, _ = self.attention(x, mask)
        return out


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
        num_temporal_layers: int = 2,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_fusion_queries: int = 4,
        use_fusion_self_attention: bool = True,
        num_pool_queries: int = 4,
        use_pool_self_attention: bool = True
    ):
        """
        Args:
            d_model: Input dimension per channel from encoder
            d_model_fused: Dimension after cross-channel fusion
            output_dim: Final embedding dimension
            num_temporal_layers: Number of temporal attention layers
            num_heads: Number of attention heads
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            num_fusion_queries: Number of query tokens for channel fusion
            use_fusion_self_attention: Whether fusion queries coordinate via self-attention
            num_pool_queries: Number of query tokens for temporal pooling
            use_pool_self_attention: Whether pooling queries coordinate via self-attention
        """
        super().__init__()
        self.output_dim = output_dim

        # Multi-query channel fusion: fuses all channels into one vector per patch
        self.cross_channel_fusion = CrossChannelFusion(
            d_model=d_model,
            d_model_fused=d_model_fused,
            num_heads=num_heads,
            dropout=dropout,
            num_queries=num_fusion_queries,
            use_self_attention=use_fusion_self_attention
        )

        self.temporal_attention = TemporalAttention(
            d_model=d_model_fused,
            num_heads=num_heads,
            num_layers=num_temporal_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # Multi-query pooling: pools temporal sequence to single vector
        self.attention_pooling = MultiQueryPooling(
            d_model=d_model_fused,
            num_queries=num_pool_queries,
            num_heads=num_heads,
            dropout=dropout,
            use_self_attention=use_pool_self_attention
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
        # Validate: all samples must have at least one valid patch
        # If this fails, there's a bug in data loading or preprocessing
        if patch_mask is not None:
            invalid_samples = ~patch_mask.any(dim=1)
            if invalid_samples.any():
                invalid_indices = invalid_samples.nonzero(as_tuple=True)[0].tolist()
                raise ValueError(
                    f"Samples {invalid_indices} have no valid patches (all-False patch_mask). "
                    f"This indicates a bug in data loading or preprocessing - sessions with "
                    f"insufficient data for the patch size should be filtered out."
                )

        # All samples valid - standard path
        # Cross-channel fusion: (batch, patches, channels, d_model) -> (batch, patches, d_model_fused)
        fused = self.cross_channel_fusion(encoder_output, channel_mask)

        # Temporal attention across patches
        temporal = self.temporal_attention(fused, patch_mask)  # (batch, patches, d_model_fused)

        # Multi-query pooling to single vector
        pooled = self.attention_pooling(temporal, patch_mask)  # (batch, d_model_fused)

        # Project to semantic space
        embedding = self.projection_head(pooled, normalize=normalize)  # (batch, output_dim)

        return embedding

    def get_attention_stats(
        self,
        encoder_output: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Get attention statistics for debugging.

        Returns dict with:
            - cross_channel_attn_entropy: How uniform is attention over channels (higher = more uniform)
            - cross_channel_attn_max: Max attention weight (higher = more focused)
        """
        # Get attention weights from cross-channel fusion
        _, attn_weights = self.cross_channel_fusion(
            encoder_output, channel_mask, return_attention_weights=True
        )

        if attn_weights is None:
            return {}

        # attn_weights shape: (batch*patches, num_queries, num_channels)
        # Compute entropy of attention distribution
        # Entropy = -sum(p * log(p)), max entropy = log(num_channels)
        attn_entropy = -(attn_weights * torch.log(attn_weights + 1e-10)).sum(dim=-1).mean().item()
        num_channels = attn_weights.shape[-1]
        max_entropy = math.log(num_channels)

        return {
            'cross_channel_attn_entropy': attn_entropy,
            'cross_channel_attn_entropy_ratio': attn_entropy / max_entropy,  # 1.0 = uniform
            'cross_channel_attn_max': attn_weights.max(dim=-1)[0].mean().item(),  # Avg max weight
            'cross_channel_attn_std': attn_weights.std(dim=-1).mean().item(),  # How spread out
        }
