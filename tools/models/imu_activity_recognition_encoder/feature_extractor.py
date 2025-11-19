"""
Feature Extractor module for IMU Activity Recognition Encoder

Fixed 1D CNN architecture for 96-timestep patches.
Uses multi-scale convolutions to capture patterns at different temporal scales.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class MultiScaleConv1D(nn.Module):
    """
    1D convolution block with optional parallel branches.

    By default uses a single kernel size (5) for simplicity. Can optionally use
    multiple kernel sizes to capture patterns at different scales:
    - Small kernels (3): Capture fine-grained, high-frequency patterns
    - Medium kernels (5): Capture mid-level temporal patterns
    - Large kernels (7): Capture longer-range dependencies

    When multiple kernels are used, outputs are concatenated for multi-scale representation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = [5],
        dropout: float = 0.1
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (per branch)
            kernel_sizes: List of kernel sizes for parallel branches
            dropout: Dropout probability
        """
        super().__init__()

        self.kernel_sizes = kernel_sizes
        self.num_branches = len(kernel_sizes)

        # Create parallel convolution branches
        self.branches = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2  # Same padding
            branch = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False
                ),
                nn.GroupNorm(num_groups=1, num_channels=out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
            self.branches.append(branch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-scale convolution.

        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_len)

        Returns:
            Concatenated output of shape (batch_size, out_channels * num_branches, seq_len)
        """
        # Process each branch
        branch_outputs = []
        for branch in self.branches:
            out = branch(x)
            branch_outputs.append(out)

        # Concatenate along channel dimension
        return torch.cat(branch_outputs, dim=1)


class ChannelIndependentCNN(nn.Module):
    """
    Channel-independent 1D CNN for fixed 96-timestep patches.

    This module processes each input channel independently through the same CNN,
    extracting temporal features without mixing information across channels.
    Cross-channel interactions are handled later by the transformer.

    Architecture:
    - Input: (batch, num_patches, 96, num_channels)
    - Process each channel independently
    - Convolutions with increasing depth (default kernel size 5)
    - Output: (batch, num_patches, num_channels, d_model)
    """

    def __init__(
        self,
        d_model: int = 128,
        cnn_channels: List[int] = [64, 128],
        kernel_sizes: List[int] = [5],
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Output feature dimension
            cnn_channels: Number of channels in each CNN layer
            kernel_sizes: Kernel sizes for convolutions (default [5] for simplicity)
            dropout: Dropout probability
        """
        super().__init__()

        self.d_model = d_model
        self.cnn_channels = cnn_channels
        self.num_scales = len(kernel_sizes)

        # Build CNN layers dynamically
        self.layers = nn.ModuleList()

        # First layer: 1 channel -> cnn_channels[0] * num_scales
        self.layers.append(MultiScaleConv1D(
            in_channels=1,
            out_channels=cnn_channels[0],
            kernel_sizes=kernel_sizes,
            dropout=dropout
        ))

        # Subsequent layers: cnn_channels[i-1] * num_scales -> cnn_channels[i] * num_scales
        for i in range(1, len(cnn_channels)):
            self.layers.append(MultiScaleConv1D(
                in_channels=cnn_channels[i-1] * self.num_scales,
                out_channels=cnn_channels[i],
                kernel_sizes=kernel_sizes,
                dropout=dropout
            ))

        # Calculate final CNN output channels
        final_cnn_channels = cnn_channels[-1] * self.num_scales

        # Adaptive pooling to reduce temporal dimension
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Projection to d_model
        self.projection = nn.Sequential(
            nn.Linear(final_cnn_channels, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through channel-independent CNN.

        Args:
            x: Input tensor of shape (batch_size, num_patches, 96, num_channels)

        Returns:
            Features of shape (batch_size, num_patches, num_channels, d_model)

        Processing:
            1. Reshape to process each (patch, channel) independently
            2. Apply CNN layers
            3. Pool temporal dimension
            4. Project to d_model
            5. Reshape back to (batch, patches, channels, features)
        """
        batch_size, num_patches, seq_len, num_channels = x.shape

        # Verify input is 96 timesteps
        assert seq_len == 96, f"Expected 96 timesteps, got {seq_len}"

        # Reshape to (batch * patches * channels, 1, 96) to process independently
        x = x.permute(0, 1, 3, 2)  # (batch, patches, channels, 96)
        x = x.reshape(batch_size * num_patches * num_channels, 1, seq_len)

        # Apply CNN layers sequentially
        for layer in self.layers:
            x = layer(x)

        # Global average pooling over temporal dimension
        x = self.adaptive_pool(x)  # (batch*patches*channels, final_cnn_channels, 1)
        x = x.squeeze(-1)  # (batch*patches*channels, final_cnn_channels)

        # Project to d_model
        x = self.projection(x)  # (batch*patches*channels, d_model)

        # Reshape back to (batch, patches, channels, d_model)
        x = x.reshape(batch_size, num_patches, num_channels, self.d_model)

        return x


class FixedPatchCNN(nn.Module):
    """
    Fixed CNN architecture for 96-timestep patches.

    This is the main feature extraction module that transforms raw sensor patches
    into learned feature representations.

    Key properties:
    - Fixed input size: 96 timesteps
    - Channel-independent processing
    - Temporal feature extraction with CNN (default kernel size 5)
    - Output: dense feature vectors per patch per channel
    """

    def __init__(
        self,
        d_model: int = 128,
        cnn_channels: List[int] = [64, 128],
        kernel_sizes: List[int] = [5],
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Output feature dimension
            cnn_channels: Channel progression through CNN layers (e.g., [64, 128])
            kernel_sizes: Kernel sizes for convolution (default [5] for simplicity)
            dropout: Dropout probability
        """
        super().__init__()

        self.d_model = d_model

        self.cnn = ChannelIndependentCNN(
            d_model=d_model,
            cnn_channels=cnn_channels,
            kernel_sizes=kernel_sizes,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from 96-timestep patches.

        Args:
            x: Input patches of shape (batch_size, num_patches, 96, num_channels)

        Returns:
            Features of shape (batch_size, num_patches, num_channels, d_model)

        Example:
            >>> cnn = FixedPatchCNN(d_model=128)
            >>> x = torch.randn(32, 10, 96, 9)  # 32 samples, 10 patches, 9 channels
            >>> features = cnn(x)
            >>> features.shape  # (32, 10, 9, 128)
        """
        return self.cnn(x)

    def get_output_dim(self) -> int:
        """Get the output feature dimension."""
        return self.d_model


def test_feature_extractor():
    """Test the feature extractor with various configurations."""
    print("Testing Feature Extractor...")

    # Test 1: Basic functionality
    print("\n1. Testing basic functionality...")
    batch_size = 4
    num_patches = 10
    num_channels = 9
    seq_len = 96

    cnn = FixedPatchCNN(d_model=128, cnn_channels=[64, 128], kernel_sizes=[3, 5, 7])
    x = torch.randn(batch_size, num_patches, seq_len, num_channels)
    features = cnn(x)

    assert features.shape == (batch_size, num_patches, num_channels, 128)
    print(f"   ✓ Input shape: {x.shape}")
    print(f"   ✓ Output shape: {features.shape}")

    # Test 2: Different channel counts
    print("\n2. Testing variable channel counts...")
    for nc in [6, 9, 23, 30, 40]:
        x = torch.randn(2, 5, 96, nc)
        features = cnn(x)
        assert features.shape == (2, 5, nc, 128)
    print(f"   ✓ Tested channel counts: 6, 9, 23, 30, 40")

    # Test 3: Different d_model sizes
    print("\n3. Testing different d_model sizes...")
    for d_model in [64, 128, 256]:
        cnn = FixedPatchCNN(d_model=d_model)
        x = torch.randn(2, 5, 96, 9)
        features = cnn(x)
        assert features.shape == (2, 5, 9, d_model)
    print(f"   ✓ Tested d_model sizes: 64, 128, 256")

    # Test 4: Single-layer CNN
    print("\n4. Testing single-layer CNN...")
    cnn = FixedPatchCNN(d_model=128, cnn_channels=[64])
    x = torch.randn(2, 5, 96, 9)
    features = cnn(x)
    assert features.shape == (2, 5, 9, 128)
    print(f"   ✓ Single-layer CNN works")

    # Test 5: Verify fixed 96 timesteps requirement
    print("\n5. Testing fixed 96 timesteps requirement...")
    cnn = FixedPatchCNN(d_model=128)
    try:
        x = torch.randn(2, 5, 100, 9)  # Wrong size
        features = cnn(x)
        assert False, "Should have raised assertion error"
    except AssertionError as e:
        assert "Expected 96 timesteps" in str(e)
    print(f"   ✓ Correctly enforces 96 timesteps")

    # Test 6: Channel independence
    print("\n6. Testing channel independence...")
    cnn = FixedPatchCNN(d_model=128)
    x = torch.randn(1, 1, 96, 2)

    # Set one channel to all zeros, one to random values
    x[:, :, :, 0] = torch.randn(1, 1, 96)
    x[:, :, :, 1] = 0.0

    features = cnn(x)

    # Features for channel 0 should be different from features for channel 1
    # (they're processed independently)
    assert not torch.allclose(features[0, 0, 0, :], features[0, 0, 1, :])
    print(f"   ✓ Channels processed independently")

    # Test 7: Gradient flow
    print("\n7. Testing gradient flow...")
    cnn = FixedPatchCNN(d_model=128)
    x = torch.randn(2, 5, 96, 9, requires_grad=True)
    features = cnn(x)
    loss = features.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    print(f"   ✓ Gradients flow correctly")

    print("\n" + "="*80)
    print("✓ ALL FEATURE EXTRACTOR TESTS PASSED!")
    print("="*80)


if __name__ == "__main__":
    test_feature_extractor()
