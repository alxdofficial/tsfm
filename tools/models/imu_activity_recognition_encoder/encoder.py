"""
Main encoder module for IMU Activity Recognition.

Assembles all components into a complete encoder:
- Preprocessing: Patching, interpolation, normalization
- Feature Extraction: Fixed 1D CNN for 64 timesteps
- Positional Encoding: Temporal + channel semantic
- Transformer: Channel-independent temporal attention
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple

try:
    from .preprocessing import preprocess_imu_data
    from .feature_extractor import FixedPatchCNN
    from .positional_encoding import IMUPositionalEncoding
    from .transformer import IMUTransformer
except ImportError:
    # For running as script
    from preprocessing import preprocess_imu_data
    from feature_extractor import FixedPatchCNN
    from positional_encoding import IMUPositionalEncoding
    from transformer import IMUTransformer


class IMUActivityRecognitionEncoder(nn.Module):
    """
    Complete encoder for IMU activity recognition.

    This encoder processes raw IMU sensor data through multiple stages:
    1. Preprocessing: Patches, interpolates to 64 timesteps, normalizes
    2. Feature Extraction: Multi-scale 1D CNN per channel
    3. Positional Encoding: Temporal position + channel semantics
    4. Transformer: Temporal attention for sequence modeling

    The encoder outputs rich representations that can be used for:
    - Activity classification (with a task head)
    - Masked autoencoding pretraining
    - Transfer learning to new activities

    Key features:
    - Variable channel support (6-40 channels)
    - Variable sampling rates (automatically handled)
    - Channel-independent processing (scales to many channels)
    - Fixed 64-timestep patches (consistent architecture)
    """

    def __init__(
        self,
        # Model architecture
        d_model: int = 128,
        num_heads: int = 8,
        num_temporal_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        use_cross_channel: bool = False,

        # CNN parameters
        cnn_channels: List[int] = [64, 128],
        cnn_kernel_sizes: List[int] = [3, 5, 7],
        patch_chunk_size: Optional[int] = None,

        # Preprocessing parameters
        target_patch_size: int = 64,
        normalization_method: str = 'zscore',
        interpolation_method: str = 'linear',

        # Positional encoding
        temporal_init_scale: float = 0.1,
        channel_init_scale: float = 0.1,
        use_channel_encoding: bool = True,
        sentence_bert_model: str = 'all-MiniLM-L6-v2',

        # Learnable token initialization
        mask_token_init_scale: float = 0.1,

        # Other
        max_patches: int = 5000
    ):
        """
        Args:
            d_model: Feature dimension throughout the model
            num_heads: Number of attention heads in transformer
            num_temporal_layers: Number of temporal transformer layers
            dim_feedforward: Hidden dimension in feed-forward networks
            dropout: Dropout probability
            use_cross_channel: Whether to use cross-channel attention (default: False for backward compatibility)

            cnn_channels: Channel progression in CNN (e.g., [64, 128])
            cnn_kernel_sizes: Kernel sizes for multi-scale CNN (e.g., [3, 5, 7])

            target_patch_size: Fixed size after interpolation (default: 64)
            normalization_method: 'zscore', 'minmax', or 'none'
            interpolation_method: 'linear', 'cubic', or 'nearest'

            temporal_init_scale: Initial scale for temporal positional encoding
            channel_init_scale: Initial scale for channel semantic encoding
            use_channel_encoding: Whether to use channel semantic encoding
            sentence_bert_model: Sentence-BERT model for channel encoding

            mask_token_init_scale: Initialization scale for mask/pad tokens (scales with sqrt(d_model))

            max_patches: Maximum number of patches to support
        """
        super().__init__()

        self.d_model = d_model
        self.target_patch_size = target_patch_size
        self.normalization_method = normalization_method
        self.interpolation_method = interpolation_method
        self.use_cross_channel = use_cross_channel

        # Feature extractor (CNN)
        self.feature_extractor = FixedPatchCNN(
            d_model=d_model,
            cnn_channels=cnn_channels,
            kernel_sizes=cnn_kernel_sizes,
            dropout=dropout,
            patch_chunk_size=patch_chunk_size
        )

        # Positional encoding (channel_projection=True always for better generalization)
        self.positional_encoding = IMUPositionalEncoding(
            d_model=d_model,
            max_patches=max_patches,
            temporal_init_scale=temporal_init_scale,
            channel_init_scale=channel_init_scale,
            sentence_bert_model=sentence_bert_model,
            use_channel_encoding=use_channel_encoding
        )

        # Transformer
        self.transformer = IMUTransformer(
            d_model=d_model,
            num_temporal_layers=num_temporal_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_cross_channel=use_cross_channel
        )

        # Learnable tokens for MAE and padding
        # Shape: (1, 1, 1, d_model) to broadcast to (batch, patches, channels, d_model)
        # Scale properly with d_model: init_scale / sqrt(d_model)
        token_init_std = mask_token_init_scale / (d_model ** 0.5)
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, d_model) * token_init_std)
        self.pad_token = nn.Parameter(torch.randn(1, 1, 1, d_model) * token_init_std)

    def preprocess(
        self,
        data: torch.Tensor,
        sampling_rate_hz: float,
        patch_size_sec: float,
        stride_sec: Optional[float] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Preprocess raw IMU data.

        Args:
            data: Raw sensor data of shape (num_timesteps, num_channels)
            sampling_rate_hz: Sampling rate in Hz
            patch_size_sec: Duration of each patch in seconds
            stride_sec: Stride between patches (default: patch_size_sec)

        Returns:
            Tuple of (preprocessed_patches, metadata)
            - preprocessed_patches: shape (num_patches, 96, num_channels)
            - metadata: dict with preprocessing statistics
        """
        return preprocess_imu_data(
            data=data,
            sampling_rate_hz=sampling_rate_hz,
            patch_size_sec=patch_size_sec,
            stride_sec=stride_sec,
            target_patch_size=self.target_patch_size,
            normalization_method=self.normalization_method,
            interpolation_method=self.interpolation_method
        )

    def forward(
        self,
        patches: torch.Tensor,
        channel_descriptions: Optional[List[str]] = None,
        temporal_mask: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
        mae_mask: Optional[torch.Tensor] = None,
        patch_attention_mask: Optional[torch.Tensor] = None,
        channel_dropout_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode preprocessed IMU patches.

        Args:
            patches: Preprocessed patches of shape (batch_size, num_patches, 96, num_channels)
            channel_descriptions: Optional channel descriptions for semantic encoding.
                                 Can be List[str] (same for all samples) or List[List[str]] (per sample)
            temporal_mask: Optional mask for temporal attention
            channel_mask: Optional mask for channel attention (batch_size, num_channels)
                         True = valid channel, False = padded. Only used if use_cross_channel=True
            mae_mask: Optional MAE mask (batch_size, num_patches) - True=masked, apply mask_token
            patch_attention_mask: Optional patch validity mask (batch_size, num_patches) - True=valid, False=padding
            channel_dropout_mask: Optional channel dropout mask (batch_size, num_channels) - True=dropped, apply mask_token

        Returns:
            Encoded features of shape (batch_size, num_patches, num_channels, d_model)

        Example:
            >>> encoder = IMUActivityRecognitionEncoder(d_model=128)
            >>> patches = torch.randn(8, 10, 64, 9)  # 8 samples, 10 patches, 64 timesteps, 9 channels
            >>> features = encoder(patches)
            >>> features.shape  # (8, 10, 9, 128)
        """
        # Verify input shape
        batch_size, num_patches, seq_len, num_channels = patches.shape
        assert seq_len == self.target_patch_size, \
            f"Expected patches with {self.target_patch_size} timesteps, got {seq_len}"

        # Extract features with CNN
        # (batch, patches, target_patch_size, channels) -> (batch, patches, channels, d_model)
        features = self.feature_extractor(patches)

        # Apply mask_token and pad_token at feature level (AFTER CNN, BEFORE positional encoding)
        if mae_mask is not None or patch_attention_mask is not None or channel_dropout_mask is not None:
            features = features.clone()  # Don't modify in-place

            # Replace MAE-masked patches with mask_token
            if mae_mask is not None:
                # Expand mask_token: (1, 1, 1, d_model) -> (batch, 1, channels, d_model)
                mask_token_expanded = self.mask_token.expand(-1, -1, num_channels, -1)

                for i in range(batch_size):
                    for p in range(num_patches):
                        if mae_mask[i, p]:
                            # Replace all channels at this patch position with mask_token
                            features[i, p] = mask_token_expanded[0, 0]  # (channels, d_model)

            # Replace padded patches with pad_token
            if patch_attention_mask is not None:
                # Expand pad_token: (1, 1, 1, d_model) -> (batch, 1, channels, d_model)
                pad_token_expanded = self.pad_token.expand(-1, -1, num_channels, -1)

                for i in range(batch_size):
                    for p in range(num_patches):
                        if not patch_attention_mask[i, p]:
                            # Replace all channels at this patch position with pad_token
                            features[i, p] = pad_token_expanded[0, 0]  # (channels, d_model)

            # Replace dropped channels with mask_token (channel dropout)
            if channel_dropout_mask is not None:
                # channel_dropout_mask: (batch, channels) - True = dropped
                for i in range(batch_size):
                    for c in range(num_channels):
                        if channel_dropout_mask[i, c]:
                            # Replace this channel across all patches with mask_token
                            features[i, :, c] = self.mask_token[0, 0, 0]  # (d_model,)

        # Add positional encodings
        # Handle both List[str] and List[List[str]] for channel_descriptions
        if channel_descriptions is not None and isinstance(channel_descriptions[0], list):
            # Per-sample channel descriptions: process each sample separately
            features_list = []
            for i in range(batch_size):
                sample_features = features[i:i+1]  # (1, patches, channels, d_model)
                sample_descs = channel_descriptions[i]
                encoded = self.positional_encoding(sample_features, sample_descs)
                features_list.append(encoded)
            features = torch.cat(features_list, dim=0)  # (batch, patches, channels, d_model)
        else:
            # Single list for all samples
            features = self.positional_encoding(features, channel_descriptions)

        # Process through transformer
        # (batch, patches, channels, d_model) -> (batch, patches, channels, d_model)
        encoded = self.transformer(features, temporal_mask, channel_mask)

        return encoded

    def encode_from_raw(
        self,
        data: torch.Tensor,
        sampling_rate_hz: float,
        patch_size_sec: float,
        stride_sec: Optional[float] = None,
        channel_descriptions: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        End-to-end encoding from raw sensor data.

        This is a convenience method that combines preprocessing and encoding.

        Args:
            data: Raw sensor data of shape (num_timesteps, num_channels)
                  Can also be batched: (batch_size, num_timesteps, num_channels)
            sampling_rate_hz: Sampling rate in Hz
            patch_size_sec: Duration of each patch in seconds
            stride_sec: Stride between patches (default: patch_size_sec)
            channel_descriptions: Optional channel descriptions

        Returns:
            Tuple of (encoded_features, metadata)
            - encoded_features: shape (batch_size, num_patches, num_channels, d_model)
            - metadata: dict with preprocessing statistics

        Example:
            >>> encoder = IMUActivityRecognitionEncoder()
            >>> data = torch.randn(1000, 9)  # 10 seconds at 100 Hz, 9 channels
            >>> features, metadata = encoder.encode_from_raw(
            ...     data, sampling_rate_hz=100.0, patch_size_sec=2.0
            ... )
            >>> features.shape  # (1, 5, 9, 128) - 1 sample, 5 patches, 9 channels, 128 features
        """
        # Handle both single and batched input
        if data.dim() == 2:
            # Single sample: (timesteps, channels)
            data = data.unsqueeze(0)  # (1, timesteps, channels)
            unbatch = True
        elif data.dim() == 3:
            # Batched: (batch, timesteps, channels)
            unbatch = False
        else:
            raise ValueError(f"Expected 2D or 3D input, got {data.dim()}D")

        batch_size = data.shape[0]

        # Preprocess each sample in the batch
        all_patches = []
        all_metadata = []

        for i in range(batch_size):
            patches, metadata = self.preprocess(
                data[i],
                sampling_rate_hz,
                patch_size_sec,
                stride_sec
            )
            all_patches.append(patches)
            all_metadata.append(metadata)

        # Stack patches: (batch, num_patches, 96, num_channels)
        batched_patches = torch.stack(all_patches, dim=0)

        # Encode
        encoded = self.forward(batched_patches, channel_descriptions)

        # Combine metadata
        combined_metadata = {
            'batch_size': batch_size,
            'num_patches': batched_patches.shape[1],
            'num_channels': batched_patches.shape[3],
            'sampling_rate_hz': sampling_rate_hz,
            'patch_size_sec': patch_size_sec,
            'per_sample_metadata': all_metadata
        }

        if unbatch:
            encoded = encoded.squeeze(0)  # Remove batch dimension

        return encoded, combined_metadata

    def get_config(self) -> Dict:
        """Get model configuration."""
        return {
            'd_model': self.d_model,
            'target_patch_size': self.target_patch_size,
            'normalization_method': self.normalization_method,
            'interpolation_method': self.interpolation_method,
        }


def test_encoder():
    """Test the complete encoder."""
    print("Testing IMU Activity Recognition Encoder...")

    # Test 1: Basic forward pass with preprocessed patches
    print("\n1. Testing basic forward pass...")
    encoder = IMUActivityRecognitionEncoder(
        d_model=128,
        num_temporal_layers=2,
        num_heads=8
    )

    patches = torch.randn(4, 10, 64, 9)  # 4 samples, 10 patches, 64 timesteps, 9 channels
    features = encoder(patches)

    assert features.shape == (4, 10, 9, 128)
    print(f"   ✓ Input shape: {patches.shape}")
    print(f"   ✓ Output shape: {features.shape}")

    # Test 2: End-to-end encoding from raw data
    print("\n2. Testing end-to-end encoding from raw data...")
    data = torch.randn(1000, 9)  # 10 seconds at 100 Hz, 9 channels
    features, metadata = encoder.encode_from_raw(
        data,
        sampling_rate_hz=100.0,
        patch_size_sec=2.0
    )

    assert features.shape[0] == 5  # 5 patches (10 seconds / 2 seconds per patch)
    assert features.shape[1] == 9  # 9 channels
    assert features.shape[2] == 128  # d_model
    print(f"   ✓ Raw data shape: {data.shape}")
    print(f"   ✓ Encoded shape: {features.shape}")
    print(f"   ✓ Metadata keys: {list(metadata.keys())}")

    # Test 3: Batched encoding from raw data
    print("\n3. Testing batched encoding from raw data...")
    batched_data = torch.randn(3, 1000, 9)  # 3 samples
    features, metadata = encoder.encode_from_raw(
        batched_data,
        sampling_rate_hz=100.0,
        patch_size_sec=2.0
    )

    assert features.shape == (3, 5, 9, 128)
    print(f"   ✓ Batched data shape: {batched_data.shape}")
    print(f"   ✓ Encoded shape: {features.shape}")

    # Test 4: Different datasets (different sampling rates and channels)
    print("\n4. Testing different datasets...")

    # UCI HAR: 50 Hz, 9 channels
    uci_data = torch.randn(500, 9)
    uci_features, _ = encoder.encode_from_raw(uci_data, 50.0, 2.0)
    assert uci_features.shape == (5, 9, 128)
    print(f"   ✓ UCI HAR (50 Hz, 9 ch): {uci_data.shape} -> {uci_features.shape}")

    # ActionSense: 200 Hz, 30 channels
    actionsense_data = torch.randn(2000, 30)
    actionsense_features, _ = encoder.encode_from_raw(actionsense_data, 200.0, 2.0)
    assert actionsense_features.shape == (5, 30, 128)
    print(f"   ✓ ActionSense (200 Hz, 30 ch): {actionsense_data.shape} -> {actionsense_features.shape}")

    # MHEALTH: 50 Hz, 23 channels
    mhealth_data = torch.randn(500, 23)
    mhealth_features, _ = encoder.encode_from_raw(mhealth_data, 50.0, 2.0)
    assert mhealth_features.shape == (5, 23, 128)
    print(f"   ✓ MHEALTH (50 Hz, 23 ch): {mhealth_data.shape} -> {mhealth_features.shape}")

    # PAMAP2: 100 Hz, 40 channels
    pamap2_data = torch.randn(1000, 40)
    pamap2_features, _ = encoder.encode_from_raw(pamap2_data, 100.0, 2.0)
    assert pamap2_features.shape == (5, 40, 128)
    print(f"   ✓ PAMAP2 (100 Hz, 40 ch): {pamap2_data.shape} -> {pamap2_features.shape}")

    # Test 5: Channel descriptions
    print("\n5. Testing channel semantic encoding...")
    channel_descs = [
        "accelerometer x-axis",
        "accelerometer y-axis",
        "accelerometer z-axis",
        "gyroscope x-axis",
        "gyroscope y-axis",
        "gyroscope z-axis",
        "magnetometer x-axis",
        "magnetometer y-axis",
        "magnetometer z-axis"
    ]

    patches = torch.randn(2, 10, 64, 9)  # Must match target_patch_size=64
    features = encoder(patches, channel_descriptions=channel_descs)
    assert features.shape == (2, 10, 9, 128)
    print(f"   ✓ Channel descriptions used successfully")

    # Test 6: Overlapping patches
    print("\n6. Testing overlapping patches...")
    data = torch.randn(1000, 9)
    features, metadata = encoder.encode_from_raw(
        data,
        sampling_rate_hz=100.0,
        patch_size_sec=2.0,
        stride_sec=1.0  # 50% overlap
    )
    assert features.shape[0] == 9  # More patches due to overlap
    print(f"   ✓ Overlapping patches: {features.shape[0]} patches with 50% overlap")

    # Test 7: Different model sizes
    print("\n7. Testing different model sizes...")
    for d_model in [64, 128, 256]:
        enc = IMUActivityRecognitionEncoder(d_model=d_model, num_temporal_layers=2)
        data = torch.randn(500, 9)
        features, _ = enc.encode_from_raw(data, 50.0, 2.0)
        assert features.shape[2] == d_model
    print(f"   ✓ Tested d_model sizes: 64, 128, 256")

    # Test 8: Gradient flow
    print("\n8. Testing gradient flow...")
    encoder = IMUActivityRecognitionEncoder(d_model=128, num_temporal_layers=2)
    patches = torch.randn(2, 10, 64, 9, requires_grad=True)  # Must use 64 timesteps
    features = encoder(patches)
    loss = features.sum()
    loss.backward()
    assert patches.grad is not None
    assert not torch.isnan(patches.grad).any()
    print(f"   ✓ Gradients flow correctly")

    # Test 9: Config retrieval
    print("\n9. Testing config retrieval...")
    config = encoder.get_config()
    assert 'd_model' in config
    assert config['target_patch_size'] == 64
    print(f"   ✓ Config: {config}")

    print("\n" + "="*80)
    print("✓ ALL ENCODER TESTS PASSED!")
    print("="*80)
    print("\nEncoder is ready for:")
    print("  - Pretraining with masked autoencoding")
    print("  - Fine-tuning on activity recognition tasks")
    print("  - Transfer learning to new datasets")


if __name__ == "__main__":
    test_encoder()
