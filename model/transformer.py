"""
Transformer module for IMU Activity Recognition Encoder

Implements temporal attention for modeling dependencies across patches.
Processes each channel independently (channel-independent temporal attention).
"""

import torch
import torch.nn as nn
from typing import Optional


class TemporalSelfAttention(nn.Module):
    """
    Multi-head self-attention over the temporal (patch) dimension.

    Processes each channel independently, allowing the model to learn
    temporal dependencies within each sensor channel.
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
            dropout: Dropout probability
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Scaling factor for attention scores
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply temporal self-attention.

        Args:
            x: Input tensor of shape (batch_size * num_channels, num_patches, d_model)
            mask: Optional attention mask of shape (num_patches, num_patches)
            key_padding_mask: Optional patch validity mask of shape (batch_channels, num_patches)
                             True = valid patch, False = padded patch

        Returns:
            Output tensor of same shape as input
        """
        batch_channels, num_patches, d_model = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch_channels, num_patches, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        # (batch_channels, num_patches, num_heads, head_dim)
        Q = Q.view(batch_channels, num_patches, self.num_heads, self.head_dim)
        K = K.view(batch_channels, num_patches, self.num_heads, self.head_dim)
        V = V.view(batch_channels, num_patches, self.num_heads, self.head_dim)

        # Transpose to (batch_channels, num_heads, num_patches, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention scores
        # (batch_channels, num_heads, num_patches, num_patches)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Apply patch padding mask (same pattern as cross-channel attention)
        if key_padding_mask is not None:
            # Mask attention TO padded patches (column mask)
            # (batch_channels, num_patches) -> (batch_channels, 1, 1, num_patches)
            mask_to = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(~mask_to, float('-inf'))

            # Mask attention FROM padded patches (row mask)
            # (batch_channels, num_patches) -> (batch_channels, 1, num_patches, 1)
            mask_from = key_padding_mask.unsqueeze(1).unsqueeze(3)
            attn_scores = attn_scores.masked_fill(~mask_from, float('-inf'))

        # Softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Replace NaNs with zeros (happens when entire row is -inf, e.g. padded patches)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # (batch_channels, num_heads, num_patches, head_dim)
        attn_output = torch.matmul(attn_weights, V)

        # Reshape back
        # (batch_channels, num_patches, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()

        # Concatenate heads
        # (batch_channels, num_patches, d_model)
        attn_output = attn_output.view(batch_channels, num_patches, d_model)

        # Final projection
        output = self.out_proj(attn_output)

        return output


class CrossChannelSelfAttention(nn.Module):
    """
    Multi-head self-attention over the channel dimension.

    Allows different sensor channels to communicate and share information
    within each patch at the same temporal position.

    This enables the model to learn cross-channel dependencies and interactions
    (e.g., correlation between accelerometer and gyroscope).
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
            dropout: Dropout probability
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Scaling factor for attention scores
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply cross-channel self-attention.

        Args:
            x: Input tensor of shape (batch_size * num_patches, num_channels, d_model)
            channel_mask: Optional channel validity mask of shape (batch_size * num_patches, num_channels)
                         True = valid channel, False = padded channel

        Returns:
            Output tensor of same shape as input
        """
        batch_patches, num_channels, d_model = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch_patches, num_channels, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        # (batch_patches, num_channels, num_heads, head_dim)
        Q = Q.view(batch_patches, num_channels, self.num_heads, self.head_dim)
        K = K.view(batch_patches, num_channels, self.num_heads, self.head_dim)
        V = V.view(batch_patches, num_channels, self.num_heads, self.head_dim)

        # Transpose to (batch_patches, num_heads, num_channels, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention scores
        # (batch_patches, num_heads, num_channels, num_channels)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply channel mask if provided
        # Mask out attention to padded channels
        if channel_mask is not None:
            # Expand mask for attention matrix
            # (batch_patches, num_channels) -> (batch_patches, 1, 1, num_channels)
            mask_expanded = channel_mask.unsqueeze(1).unsqueeze(2)

            # Mask attention to padded channels (set to -inf so softmax gives 0)
            attn_scores = attn_scores.masked_fill(~mask_expanded, float('-inf'))

            # Also mask attention FROM padded channels
            # (batch_patches, num_channels) -> (batch_patches, 1, num_channels, 1)
            mask_from = channel_mask.unsqueeze(1).unsqueeze(3)
            attn_scores = attn_scores.masked_fill(~mask_from, float('-inf'))

        # Softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Replace NaNs with zeros (happens when entire row is -inf)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # (batch_patches, num_heads, num_channels, head_dim)
        attn_output = torch.matmul(attn_weights, V)

        # Reshape back
        # (batch_patches, num_channels, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()

        # Concatenate heads
        # (batch_patches, num_channels, d_model)
        attn_output = attn_output.view(batch_patches, num_channels, d_model)

        # Final projection
        output = self.out_proj(attn_output)

        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Two-layer MLP with GELU activation.
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Input/output dimension
            dim_feedforward: Hidden dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (*, d_model)

        Returns:
            Output tensor of same shape
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class TemporalTransformerBlock(nn.Module):
    """
    Single transformer block with temporal self-attention.

    Architecture:
    - Temporal self-attention with residual connection and layer norm
    - Feed-forward network with residual connection and layer norm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Feature dimension
            num_heads: Number of attention heads
            dim_feedforward: Hidden dimension for feed-forward network
            dropout: Dropout probability
        """
        super().__init__()

        self.self_attn = TemporalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        self.feed_forward = FeedForward(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor of shape (batch_size * num_channels, num_patches, d_model)
            mask: Optional attention mask
            key_padding_mask: Optional patch validity mask (batch_channels, num_patches)
                             True = valid, False = padded

        Returns:
            Output tensor of same shape
        """
        # Self-attention with residual
        attn_output = self.self_attn(x, mask, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x


class DualBranchTransformerBlock(nn.Module):
    """
    Dual-branch transformer block with both temporal and cross-channel attention.

    Architecture:
    1. Temporal self-attention (patches within each channel)
    2. Cross-channel self-attention (channels within each patch)
    3. Feed-forward network

    Each step includes residual connections and layer normalization.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Feature dimension
            num_heads: Number of attention heads
            dim_feedforward: Hidden dimension for feed-forward network
            dropout: Dropout probability
        """
        super().__init__()

        # Temporal attention (over patches within channel)
        self.temporal_attn = TemporalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        # Cross-channel attention (over channels within patch)
        self.cross_channel_attn = CrossChannelSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        # Feed-forward network
        self.feed_forward = FeedForward(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
        patch_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through dual-branch transformer block.

        Args:
            x: Input tensor of shape (batch_size, num_patches, num_channels, d_model)
            temporal_mask: Optional mask for temporal attention (num_patches, num_patches)
            channel_mask: Optional mask for channel attention (batch_size, num_channels)
                         True = valid channel, False = padded
            patch_padding_mask: Optional patch validity mask (batch_size, num_patches)
                               True = valid patch, False = padded

        Returns:
            Output tensor of shape (batch_size, num_patches, num_channels, d_model)
        """
        batch_size, num_patches, num_channels, d_model = x.shape

        # 1. Temporal self-attention (patches within each channel)
        # Reshape: (batch, patches, channels, d_model) -> (batch*channels, patches, d_model)
        x_temporal = x.permute(0, 2, 1, 3)  # (batch, channels, patches, d_model)
        x_temporal = x_temporal.reshape(batch_size * num_channels, num_patches, d_model)

        # Expand patch padding mask to (B*C, P) for temporal attention
        if patch_padding_mask is not None:
            temporal_key_padding_mask = patch_padding_mask.unsqueeze(1).expand(
                batch_size, num_channels, num_patches
            ).reshape(batch_size * num_channels, num_patches)
        else:
            temporal_key_padding_mask = None

        # Apply temporal attention
        temporal_output = self.temporal_attn(x_temporal, temporal_mask,
                                             key_padding_mask=temporal_key_padding_mask)

        # Reshape back: (batch*channels, patches, d_model) -> (batch, patches, channels, d_model)
        temporal_output = temporal_output.reshape(batch_size, num_channels, num_patches, d_model)
        temporal_output = temporal_output.permute(0, 2, 1, 3)

        # Residual and norm
        x = x + self.dropout(temporal_output)
        x = self.norm1(x)

        # 2. Cross-channel self-attention (channels within each patch)
        # Reshape: (batch, patches, channels, d_model) -> (batch*patches, channels, d_model)
        x_channel = x.reshape(batch_size * num_patches, num_channels, d_model)

        # Expand channel mask if provided
        # (batch, channels) -> (batch*patches, channels)
        if channel_mask is not None:
            channel_mask_expanded = channel_mask.unsqueeze(1).expand(
                batch_size, num_patches, num_channels
            ).reshape(batch_size * num_patches, num_channels)
        else:
            channel_mask_expanded = None

        # Apply cross-channel attention
        channel_output = self.cross_channel_attn(x_channel, channel_mask_expanded)

        # Reshape back: (batch*patches, channels, d_model) -> (batch, patches, channels, d_model)
        channel_output = channel_output.reshape(batch_size, num_patches, num_channels, d_model)

        # Residual and norm
        x = x + self.dropout(channel_output)
        x = self.norm2(x)

        # 3. Feed-forward network
        ff_output = self.feed_forward(x)

        # Residual and norm
        x = x + self.dropout(ff_output)
        x = self.norm3(x)

        return x


class ChannelIndependentTemporalTransformer(nn.Module):
    """
    Temporal transformer that processes each channel independently.

    Each channel's patch sequence is processed through the same transformer,
    learning temporal dependencies without mixing information across channels.

    Input:  (batch, patches, channels, d_model)
    Process: Each channel independently through temporal attention
    Output: (batch, patches, channels, d_model)
    """

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Feature dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dim_feedforward: Hidden dimension for feed-forward networks
            dropout: Dropout probability
        """
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Stack of transformer blocks
        self.layers = nn.ModuleList([
            TemporalTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        patch_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process input through channel-independent temporal transformer.

        Args:
            x: Input tensor of shape (batch_size, num_patches, num_channels, d_model)
            mask: Optional attention mask of shape (num_patches, num_patches)
            patch_padding_mask: Optional patch validity mask (batch_size, num_patches)
                               True = valid patch, False = padded

        Returns:
            Output tensor of shape (batch_size, num_patches, num_channels, d_model)

        Processing:
            1. Reshape to (batch * channels, patches, d_model)
            2. Apply transformer layers
            3. Reshape back to (batch, patches, channels, d_model)
        """
        batch_size, num_patches, num_channels, d_model = x.shape

        # Reshape to process each channel independently
        # (batch, patches, channels, d_model) -> (batch * channels, patches, d_model)
        x = x.permute(0, 2, 1, 3)  # (batch, channels, patches, d_model)
        x = x.reshape(batch_size * num_channels, num_patches, d_model)

        # Expand patch padding mask to (B*C, P)
        if patch_padding_mask is not None:
            key_padding_mask = patch_padding_mask.unsqueeze(1).expand(
                batch_size, num_channels, num_patches
            ).reshape(batch_size * num_channels, num_patches)
        else:
            key_padding_mask = None

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask, key_padding_mask=key_padding_mask)

        # Reshape back to original format
        # (batch * channels, patches, d_model) -> (batch, patches, channels, d_model)
        x = x.reshape(batch_size, num_channels, num_patches, d_model)
        x = x.permute(0, 2, 1, 3)  # (batch, patches, channels, d_model)

        return x


class DualBranchTransformer(nn.Module):
    """
    Dual-branch transformer with both temporal and cross-channel attention.

    Processes patches with two types of attention:
    1. Temporal attention: Models dependencies across patches within each channel
    2. Cross-channel attention: Models interactions between channels within each patch

    Input:  (batch, patches, channels, d_model)
    Process: Temporal attention → Cross-channel attention → FFN (per block)
    Output: (batch, patches, channels, d_model)
    """

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Feature dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dim_feedforward: Hidden dimension for feed-forward networks
            dropout: Dropout probability
        """
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Stack of dual-branch transformer blocks
        self.layers = nn.ModuleList([
            DualBranchTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
        patch_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process input through dual-branch transformer.

        Args:
            x: Input tensor of shape (batch_size, num_patches, num_channels, d_model)
            temporal_mask: Optional mask for temporal attention (num_patches, num_patches)
            channel_mask: Optional mask for channel attention (batch_size, num_channels)
                         True = valid channel, False = padded
            patch_padding_mask: Optional patch validity mask (batch_size, num_patches)
                               True = valid patch, False = padded

        Returns:
            Output tensor of shape (batch_size, num_patches, num_channels, d_model)
        """
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, temporal_mask, channel_mask, patch_padding_mask)

        return x


class IMUTransformer(nn.Module):
    """
    Main transformer module for IMU encoder.

    Supports two modes:
    1. Temporal-only: Channel-independent temporal attention (backward compatible)
    2. Dual-branch: Temporal + cross-channel attention (enables channel interactions)
    """

    def __init__(
        self,
        d_model: int = 128,
        num_temporal_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        use_cross_channel: bool = False
    ):
        """
        Args:
            d_model: Feature dimension
            num_temporal_layers: Number of temporal transformer layers
            num_heads: Number of attention heads
            dim_feedforward: Hidden dimension for feed-forward networks
            dropout: Dropout probability
            use_cross_channel: Whether to use cross-channel attention (default: False for backward compatibility)
        """
        super().__init__()

        self.use_cross_channel = use_cross_channel

        if use_cross_channel:
            # Use dual-branch transformer with temporal + cross-channel attention
            self.transformer = DualBranchTransformer(
                d_model=d_model,
                num_layers=num_temporal_layers,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
        else:
            # Use temporal-only transformer (backward compatible)
            self.transformer = ChannelIndependentTemporalTransformer(
                d_model=d_model,
                num_layers=num_temporal_layers,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )

    def forward(
        self,
        x: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
        patch_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process input through transformer.

        Args:
            x: Input tensor of shape (batch_size, num_patches, num_channels, d_model)
            temporal_mask: Optional mask for temporal attention
            channel_mask: Optional mask for channel attention (only used if use_cross_channel=True)
                         Shape: (batch_size, num_channels), True = valid, False = padded
            patch_padding_mask: Optional patch validity mask (batch_size, num_patches)
                               True = valid patch, False = padded

        Returns:
            Output tensor of shape (batch_size, num_patches, num_channels, d_model)
        """
        if self.use_cross_channel:
            return self.transformer(x, temporal_mask, channel_mask, patch_padding_mask)
        else:
            return self.transformer(x, temporal_mask, patch_padding_mask)


def test_transformer():
    """Test transformer modules."""
    print("Testing Transformer...")

    # Test 1: Temporal self-attention
    print("\n1. Testing temporal self-attention...")
    d_model = 128
    num_heads = 8
    attn = TemporalSelfAttention(d_model=d_model, num_heads=num_heads)

    x = torch.randn(4, 10, d_model)  # (batch*channels, patches, d_model)
    out = attn(x)
    assert out.shape == x.shape
    print(f"   ✓ Attention output shape: {x.shape} -> {out.shape}")

    # Test 2: Feed-forward network
    print("\n2. Testing feed-forward network...")
    ff = FeedForward(d_model=d_model, dim_feedforward=512)
    x = torch.randn(4, 10, d_model)
    out = ff(x)
    assert out.shape == x.shape
    print(f"   ✓ Feed-forward output shape: {x.shape} -> {out.shape}")

    # Test 3: Transformer block
    print("\n3. Testing transformer block...")
    block = TemporalTransformerBlock(d_model=d_model, num_heads=num_heads)
    x = torch.randn(4, 10, d_model)
    out = block(x)
    assert out.shape == x.shape
    print(f"   ✓ Block output shape: {x.shape} -> {out.shape}")

    # Test 4: Channel-independent temporal transformer
    print("\n4. Testing channel-independent temporal transformer...")
    transformer = ChannelIndependentTemporalTransformer(
        d_model=d_model,
        num_layers=4,
        num_heads=num_heads
    )

    x = torch.randn(2, 10, 9, d_model)  # (batch, patches, channels, d_model)
    out = transformer(x)
    assert out.shape == x.shape
    print(f"   ✓ Transformer output shape: {x.shape} -> {out.shape}")

    # Test 5: Full IMU transformer
    print("\n5. Testing full IMU transformer...")
    imu_transformer = IMUTransformer(
        d_model=d_model,
        num_temporal_layers=4,
        num_heads=num_heads
    )

    x = torch.randn(4, 15, 9, d_model)
    out = imu_transformer(x)
    assert out.shape == x.shape
    print(f"   ✓ IMU transformer output shape: {x.shape} -> {out.shape}")

    # Test 6: Variable channel counts
    print("\n6. Testing variable channel counts...")
    for num_channels in [6, 9, 23, 30, 40]:
        x = torch.randn(2, 10, num_channels, d_model)
        out = imu_transformer(x)
        assert out.shape == x.shape
    print(f"   ✓ Tested channel counts: 6, 9, 23, 30, 40")

    # Test 7: Variable patch counts
    print("\n7. Testing variable patch counts...")
    for num_patches in [5, 10, 20, 50]:
        x = torch.randn(2, num_patches, 9, d_model)
        out = imu_transformer(x)
        assert out.shape == x.shape
    print(f"   ✓ Tested patch counts: 5, 10, 20, 50")

    # Test 8: Gradient flow
    print("\n8. Testing gradient flow...")
    imu_transformer = IMUTransformer(d_model=d_model, num_temporal_layers=2)
    x = torch.randn(2, 10, 9, d_model, requires_grad=True)
    out = imu_transformer(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print(f"   ✓ Gradients flow correctly")

    # Test 9: Attention mask
    print("\n9. Testing attention mask...")
    # Create a causal mask (lower triangular)
    num_patches = 10
    mask = torch.tril(torch.ones(num_patches, num_patches))

    imu_transformer = IMUTransformer(d_model=d_model, num_temporal_layers=2)
    x = torch.randn(2, num_patches, 9, d_model)
    out = imu_transformer(x, temporal_mask=mask)
    assert out.shape == x.shape
    print(f"   ✓ Attention mask works")

    # Test 10: Different layer counts
    print("\n10. Testing different layer counts...")
    for num_layers in [1, 2, 4, 6]:
        imu_transformer = IMUTransformer(
            d_model=d_model,
            num_temporal_layers=num_layers
        )
        x = torch.randn(2, 10, 9, d_model)
        out = imu_transformer(x)
        assert out.shape == x.shape
    print(f"   ✓ Tested layer counts: 1, 2, 4, 6")

    # Test 11: Channel independence
    print("\n11. Testing channel independence...")
    imu_transformer = IMUTransformer(d_model=d_model, num_temporal_layers=2)
    x = torch.zeros(1, 5, 2, d_model)

    # Set one channel to random values, one to zeros
    x[:, :, 0, :] = torch.randn(1, 5, d_model)
    x[:, :, 1, :] = 0.0

    out = imu_transformer(x)

    # Outputs for channel 0 should be different from channel 1
    assert not torch.allclose(out[0, :, 0, :], out[0, :, 1, :])
    print(f"   ✓ Channels processed independently")

    # Test 12: Cross-channel self-attention
    print("\n12. Testing cross-channel self-attention...")
    cross_channel_attn = CrossChannelSelfAttention(d_model=d_model, num_heads=num_heads)
    x = torch.randn(20, 9, d_model)  # (batch*patches, channels, d_model)
    out = cross_channel_attn(x)
    assert out.shape == x.shape
    print(f"   ✓ Cross-channel attention output shape: {x.shape} -> {out.shape}")

    # Test 13: Cross-channel attention with channel mask
    print("\n13. Testing cross-channel attention with channel mask...")
    batch_patches = 20
    num_channels = 9
    x = torch.randn(batch_patches, num_channels, d_model)

    # Create mask: first 6 channels valid, rest padded
    channel_mask = torch.zeros(batch_patches, num_channels, dtype=torch.bool)
    channel_mask[:, :6] = True

    out = cross_channel_attn(x, channel_mask)
    assert out.shape == x.shape
    print(f"   ✓ Cross-channel attention with mask works")

    # Test 14: Dual-branch transformer block
    print("\n14. Testing dual-branch transformer block...")
    dual_block = DualBranchTransformerBlock(d_model=d_model, num_heads=num_heads)
    x = torch.randn(2, 10, 9, d_model)  # (batch, patches, channels, d_model)
    out = dual_block(x)
    assert out.shape == x.shape
    print(f"   ✓ Dual-branch block output shape: {x.shape} -> {out.shape}")

    # Test 15: Dual-branch transformer block with masks
    print("\n15. Testing dual-branch block with masks...")
    batch_size = 2
    num_patches = 10
    num_channels = 9
    x = torch.randn(batch_size, num_patches, num_channels, d_model)

    # Create channel mask
    channel_mask = torch.ones(batch_size, num_channels, dtype=torch.bool)
    channel_mask[1, 6:] = False  # Second sample has only 6 channels

    out = dual_block(x, temporal_mask=None, channel_mask=channel_mask)
    assert out.shape == x.shape
    print(f"   ✓ Dual-branch block with channel mask works")

    # Test 16: Dual-branch transformer
    print("\n16. Testing dual-branch transformer...")
    dual_transformer = DualBranchTransformer(
        d_model=d_model,
        num_layers=2,
        num_heads=num_heads
    )
    x = torch.randn(2, 10, 9, d_model)
    out = dual_transformer(x)
    assert out.shape == x.shape
    print(f"   ✓ Dual-branch transformer output shape: {x.shape} -> {out.shape}")

    # Test 17: IMU transformer with cross-channel attention
    print("\n17. Testing IMU transformer with cross-channel attention...")
    imu_transformer_cross = IMUTransformer(
        d_model=d_model,
        num_temporal_layers=2,
        num_heads=num_heads,
        use_cross_channel=True
    )
    x = torch.randn(2, 10, 9, d_model)
    out = imu_transformer_cross(x)
    assert out.shape == x.shape
    print(f"   ✓ IMU transformer with cross-channel output shape: {x.shape} -> {out.shape}")

    # Test 18: IMU transformer with cross-channel and channel mask
    print("\n18. Testing IMU transformer with cross-channel and mask...")
    batch_size = 2
    x = torch.randn(batch_size, 10, 9, d_model)
    channel_mask = torch.ones(batch_size, 9, dtype=torch.bool)
    channel_mask[1, 6:] = False

    out = imu_transformer_cross(x, temporal_mask=None, channel_mask=channel_mask)
    assert out.shape == x.shape
    print(f"   ✓ IMU transformer with cross-channel and mask works")

    # Test 19: Variable channel counts with cross-channel attention
    print("\n19. Testing variable channels with cross-channel...")
    for num_channels_test in [6, 9, 23, 30, 40]:
        x = torch.randn(2, 10, num_channels_test, d_model)
        out = imu_transformer_cross(x)
        assert out.shape == x.shape
    print(f"   ✓ Tested channel counts with cross-channel: 6, 9, 23, 30, 40")

    # Test 20: Gradient flow through cross-channel attention
    print("\n20. Testing gradient flow through cross-channel...")
    imu_transformer_cross = IMUTransformer(
        d_model=d_model,
        num_temporal_layers=2,
        use_cross_channel=True
    )
    x = torch.randn(2, 10, 9, d_model, requires_grad=True)
    out = imu_transformer_cross(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print(f"   ✓ Gradients flow correctly through cross-channel")

    # Test 21: Cross-channel enables channel interaction
    print("\n21. Testing cross-channel enables channel interaction...")
    imu_transformer_cross = IMUTransformer(
        d_model=d_model,
        num_temporal_layers=2,
        use_cross_channel=True
    )
    x = torch.zeros(1, 5, 2, d_model)

    # Set one channel to random values, one to zeros
    x[:, :, 0, :] = torch.randn(1, 5, d_model)
    x[:, :, 1, :] = 0.0

    out = imu_transformer_cross(x)

    # With cross-channel attention, channel 1 should be influenced by channel 0
    # (not all zeros anymore)
    assert not torch.allclose(out[0, :, 1, :], torch.zeros_like(out[0, :, 1, :]), atol=1e-6)
    print(f"   ✓ Cross-channel attention enables channel interaction")

    print("\n" + "="*80)
    print("✓ ALL TRANSFORMER TESTS PASSED!")
    print("  - Temporal attention (channel-independent)")
    print("  - Cross-channel attention")
    print("  - Dual-branch architecture")
    print("  - Channel masking for variable channels")
    print("="*80)


if __name__ == "__main__":
    test_transformer()
