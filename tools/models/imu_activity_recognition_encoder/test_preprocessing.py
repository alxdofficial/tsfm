"""
Unit tests for preprocessing module.

Tests patching, interpolation, and normalization operations.
"""

import torch
import numpy as np
from preprocessing import (
    create_patches,
    interpolate_patches,
    normalize_patches,
    preprocess_imu_data
)


class TestCreatePatches:
    """Tests for create_patches function."""

    def test_basic_patching(self):
        """Test basic patching with standard parameters."""
        # Create 10 seconds of data at 50 Hz, 9 channels
        data = torch.randn(500, 9)

        # Create 2-second patches
        patches = create_patches(data, sampling_rate_hz=50.0, patch_size_sec=2.0)

        # Should have 5 non-overlapping patches of 100 timesteps each
        assert patches.shape == (5, 100, 9)
        assert patches.dtype == torch.float32

    def test_different_sampling_rates(self):
        """Test patching works with different sampling rates."""
        # Test 50 Hz
        data_50hz = torch.randn(500, 9)  # 10 seconds at 50 Hz
        patches_50hz = create_patches(data_50hz, sampling_rate_hz=50.0, patch_size_sec=2.0)
        assert patches_50hz.shape == (5, 100, 9)

        # Test 100 Hz
        data_100hz = torch.randn(1000, 9)  # 10 seconds at 100 Hz
        patches_100hz = create_patches(data_100hz, sampling_rate_hz=100.0, patch_size_sec=2.0)
        assert patches_100hz.shape == (5, 200, 9)

        # Test 200 Hz
        data_200hz = torch.randn(2000, 6)  # 10 seconds at 200 Hz, 6 channels
        patches_200hz = create_patches(data_200hz, sampling_rate_hz=200.0, patch_size_sec=2.0)
        assert patches_200hz.shape == (5, 400, 6)

    def test_overlapping_patches(self):
        """Test patching with overlap (stride < patch_size)."""
        data = torch.randn(500, 9)  # 10 seconds at 50 Hz

        # 2-second patches with 1-second stride (50% overlap)
        patches = create_patches(
            data,
            sampling_rate_hz=50.0,
            patch_size_sec=2.0,
            stride_sec=1.0
        )

        # Should have 9 overlapping patches
        assert patches.shape == (9, 100, 9)

    def test_patch_data_integrity(self):
        """Test that patches contain correct data from original signal."""
        # Create simple test data with known pattern
        data = torch.arange(0, 100).unsqueeze(1).float()  # Single channel, values 0-99

        # Create 2 patches of 50 timesteps each
        patches = create_patches(data, sampling_rate_hz=50.0, patch_size_sec=1.0)

        assert patches.shape == (2, 50, 1)

        # First patch should be [0, 1, 2, ..., 49]
        assert torch.allclose(patches[0, :, 0], torch.arange(0, 50).float())

        # Second patch should be [50, 51, 52, ..., 99]
        assert torch.allclose(patches[1, :, 0], torch.arange(50, 100).float())

    def test_insufficient_data_error(self):
        """Test that error is raised when data is too short."""
        data = torch.randn(50, 9)  # Only 1 second at 50 Hz

        try:
            create_patches(data, sampling_rate_hz=50.0, patch_size_sec=2.0)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Patch size" in str(e) and "is larger than data length" in str(e)

    def test_numpy_input(self):
        """Test that numpy arrays are accepted."""
        data_np = np.random.randn(500, 9)
        patches = create_patches(data_np, sampling_rate_hz=50.0, patch_size_sec=2.0)

        assert patches.shape == (5, 100, 9)
        assert isinstance(patches, torch.Tensor)


class TestInterpolatePatches:
    """Tests for interpolate_patches function."""

    def test_downsampling(self):
        """Test downsampling from 200 to 96 timesteps."""
        patches = torch.randn(5, 200, 9)
        interpolated = interpolate_patches(patches, target_size=96)

        assert interpolated.shape == (5, 96, 9)
        assert interpolated.dtype == torch.float32

    def test_upsampling(self):
        """Test upsampling from 50 to 96 timesteps."""
        patches = torch.randn(5, 50, 9)
        interpolated = interpolate_patches(patches, target_size=96)

        assert interpolated.shape == (5, 96, 9)

    def test_no_change_needed(self):
        """Test that patches already at target size are returned unchanged."""
        patches = torch.randn(5, 96, 9)
        interpolated = interpolate_patches(patches, target_size=96)

        assert torch.allclose(patches, interpolated)

    def test_interpolation_preserves_signal_shape(self):
        """Test that interpolation preserves signal characteristics."""
        # Create a simple sinusoidal signal
        t = np.linspace(0, 2*np.pi, 200)
        signal = np.sin(t)
        patches = torch.from_numpy(signal).unsqueeze(0).unsqueeze(-1).float()  # (1, 200, 1)

        interpolated = interpolate_patches(patches, target_size=96)

        # Check shape
        assert interpolated.shape == (1, 96, 1)

        # Check that signal still looks sinusoidal (min and max preserved roughly)
        original_min = patches[0, :, 0].min()
        original_max = patches[0, :, 0].max()
        interp_min = interpolated[0, :, 0].min()
        interp_max = interpolated[0, :, 0].max()

        assert torch.abs(original_min - interp_min) < 0.1
        assert torch.abs(original_max - interp_max) < 0.1

    def test_different_methods(self):
        """Test different interpolation methods."""
        patches = torch.randn(3, 200, 9)

        # Linear (default)
        linear = interpolate_patches(patches, target_size=96, method='linear')
        assert linear.shape == (3, 96, 9)

        # Cubic
        cubic = interpolate_patches(patches, target_size=96, method='cubic')
        assert cubic.shape == (3, 96, 9)

        # Nearest
        nearest = interpolate_patches(patches, target_size=96, method='nearest')
        assert nearest.shape == (3, 96, 9)

        # Linear and cubic produce identical results (F.interpolate 1D uses linear for both)
        assert torch.allclose(linear, cubic)
        # Nearest should differ from linear
        assert not torch.allclose(linear, nearest)

    def test_numpy_input(self):
        """Test that numpy arrays are accepted."""
        patches_np = np.random.randn(5, 200, 9)
        interpolated = interpolate_patches(patches_np, target_size=96)

        assert interpolated.shape == (5, 96, 9)
        assert isinstance(interpolated, torch.Tensor)


class TestNormalizePatches:
    """Tests for normalize_patches function."""

    def test_zscore_normalization(self):
        """Test z-score normalization produces zero mean and unit variance."""
        # Create patches with known statistics
        patches = torch.randn(10, 96, 9) * 5.0 + 10.0  # Mean around 10, std around 5

        normalized, means, stds = normalize_patches(patches, method='zscore')

        # Check shapes
        assert normalized.shape == (10, 96, 9)
        assert means.shape == (10, 9)
        assert stds.shape == (10, 9)

        # Normalized patches should have approximately zero mean and unit variance
        # (per patch, per channel)
        for i in range(10):
            for j in range(9):
                patch_mean = normalized[i, :, j].mean()
                patch_std = normalized[i, :, j].std(unbiased=False)

                assert torch.abs(patch_mean) < 1e-5, f"Mean should be ~0, got {patch_mean}"
                assert torch.abs(patch_std - 1.0) < 1e-5, f"Std should be ~1, got {patch_std}"

    def test_minmax_normalization(self):
        """Test min-max normalization produces values in [0, 1] range."""
        patches = torch.randn(10, 96, 9) * 10.0 + 5.0

        normalized, mins, ranges = normalize_patches(patches, method='minmax')

        # Check shapes
        assert normalized.shape == (10, 96, 9)
        assert mins.shape == (10, 9)
        assert ranges.shape == (10, 9)

        # Normalized patches should be in [0, 1] range (per patch, per channel)
        for i in range(10):
            for j in range(9):
                patch_min = normalized[i, :, j].min()
                patch_max = normalized[i, :, j].max()

                assert patch_min >= -1e-5, f"Min should be ~0, got {patch_min}"
                assert patch_max <= 1.0 + 1e-5, f"Max should be ~1, got {patch_max}"

    def test_no_normalization(self):
        """Test that 'none' method returns unchanged patches."""
        patches = torch.randn(10, 96, 9)

        normalized, means, stds = normalize_patches(patches, method='none')

        # Patches should be unchanged
        assert torch.allclose(patches, normalized)

        # Means should be zeros, stds should be ones
        assert torch.allclose(means, torch.zeros(10, 9))
        assert torch.allclose(stds, torch.ones(10, 9))

    def test_per_patch_per_channel_independence(self):
        """Test that normalization is independent per patch and per channel."""
        # Create patches with different statistics per patch and channel
        patches = torch.zeros(3, 96, 2)

        # Patch 0, channel 0: mean=10, range [5, 15]
        patches[0, :, 0] = torch.linspace(5, 15, 96)

        # Patch 0, channel 1: mean=50, range [40, 60]
        patches[0, :, 1] = torch.linspace(40, 60, 96)

        # Patch 1, channel 0: mean=-5, range [-10, 0]
        patches[1, :, 0] = torch.linspace(-10, 0, 96)

        normalized, means, stds = normalize_patches(patches, method='zscore')

        # Each patch-channel combination should be independently normalized
        # Mean should be stored correctly
        assert torch.abs(means[0, 0] - 10.0) < 0.1
        assert torch.abs(means[0, 1] - 50.0) < 0.1
        assert torch.abs(means[1, 0] - (-5.0)) < 0.1

    def test_zero_std_handling(self):
        """Test that zero standard deviation is handled (constant signals)."""
        # Create patches with constant values (zero variance)
        patches = torch.ones(5, 96, 9) * 42.0

        normalized, means, stds = normalize_patches(patches, method='zscore')

        # Should not crash due to division by zero
        # Normalized values should be zero (or very close)
        assert not torch.isnan(normalized).any()
        assert not torch.isinf(normalized).any()


class TestPreprocessIMUData:
    """Tests for complete preprocessing pipeline."""

    def test_full_pipeline_50hz(self):
        """Test full pipeline with 50 Hz data."""
        # 10 seconds at 50 Hz, 9 channels
        data = torch.randn(500, 9)

        patches, metadata = preprocess_imu_data(
            data,
            sampling_rate_hz=50.0,
            patch_size_sec=2.0
        )

        # Should have 5 patches of 64 timesteps each (default target_patch_size)
        assert patches.shape == (5, 64, 9)

        # Check metadata
        assert metadata['original_patch_size'] == 100
        assert metadata['target_patch_size'] == 64
        assert metadata['sampling_rate_hz'] == 50.0
        assert metadata['patch_size_sec'] == 2.0
        assert metadata['num_channels'] == 9
        assert metadata['means'].shape == (5, 9)
        assert metadata['stds'].shape == (5, 9)

    def test_full_pipeline_100hz(self):
        """Test full pipeline with 100 Hz data."""
        # 10 seconds at 100 Hz, 6 channels
        data = torch.randn(1000, 6)

        patches, metadata = preprocess_imu_data(
            data,
            sampling_rate_hz=100.0,
            patch_size_sec=2.0
        )

        # Should have 5 patches of 64 timesteps each (default target_patch_size)
        assert patches.shape == (5, 64, 6)
        assert metadata['original_patch_size'] == 200
        assert metadata['num_channels'] == 6

    def test_full_pipeline_200hz(self):
        """Test full pipeline with 200 Hz data."""
        # 10 seconds at 200 Hz, 40 channels
        data = torch.randn(2000, 40)

        patches, metadata = preprocess_imu_data(
            data,
            sampling_rate_hz=200.0,
            patch_size_sec=2.0
        )

        # Should have 5 patches of 64 timesteps each (default target_patch_size)
        assert patches.shape == (5, 64, 40)
        assert metadata['original_patch_size'] == 400
        assert metadata['num_channels'] == 40

    def test_pipeline_normalized_output(self):
        """Test that pipeline output is properly normalized."""
        data = torch.randn(500, 9)

        patches, metadata = preprocess_imu_data(
            data,
            sampling_rate_hz=50.0,
            patch_size_sec=2.0,
            normalization_method='zscore'
        )

        # Each patch, each channel should have approximately zero mean and unit variance
        for i in range(patches.shape[0]):
            for j in range(patches.shape[2]):
                mean = patches[i, :, j].mean()
                std = patches[i, :, j].std(unbiased=False)

                assert torch.abs(mean) < 1e-4
                assert torch.abs(std - 1.0) < 1e-4

    def test_pipeline_with_overlap(self):
        """Test pipeline with overlapping patches."""
        data = torch.randn(500, 9)

        patches, metadata = preprocess_imu_data(
            data,
            sampling_rate_hz=50.0,
            patch_size_sec=2.0,
            stride_sec=1.0  # 50% overlap
        )

        # Should have 9 overlapping patches
        assert patches.shape == (9, 64, 9)


def test_integration_all_datasets():
    """Integration test simulating data from different datasets."""

    # UCI HAR: 50 Hz, 9 channels (3-axis accel + 3-axis gyro, body + total)
    uci_har_data = torch.randn(500, 9)
    uci_har_patches, _ = preprocess_imu_data(
        uci_har_data,
        sampling_rate_hz=50.0,
        patch_size_sec=2.0
    )
    assert uci_har_patches.shape == (5, 64, 9)

    # ActionSense: 200 Hz, variable channels (let's say 30)
    actionsense_data = torch.randn(2000, 30)
    actionsense_patches, _ = preprocess_imu_data(
        actionsense_data,
        sampling_rate_hz=200.0,
        patch_size_sec=2.0
    )
    assert actionsense_patches.shape == (5, 64, 30)

    # MHEALTH: 50 Hz, 23 channels
    mhealth_data = torch.randn(500, 23)
    mhealth_patches, _ = preprocess_imu_data(
        mhealth_data,
        sampling_rate_hz=50.0,
        patch_size_sec=2.0
    )
    assert mhealth_patches.shape == (5, 64, 23)

    # PAMAP2: 100 Hz, 40 channels
    pamap2_data = torch.randn(1000, 40)
    pamap2_patches, _ = preprocess_imu_data(
        pamap2_data,
        sampling_rate_hz=100.0,
        patch_size_sec=2.0
    )
    assert pamap2_patches.shape == (5, 64, 40)

    print("✓ All dataset integration tests passed!")


if __name__ == "__main__":
    # Run basic tests
    print("Running preprocessing unit tests...")

    # Test patching
    print("\n1. Testing create_patches...")
    test_patches = TestCreatePatches()
    test_patches.test_basic_patching()
    test_patches.test_different_sampling_rates()
    test_patches.test_overlapping_patches()
    test_patches.test_patch_data_integrity()
    test_patches.test_numpy_input()
    print("   ✓ All patching tests passed")

    # Test interpolation
    print("\n2. Testing interpolate_patches...")
    test_interp = TestInterpolatePatches()
    test_interp.test_downsampling()
    test_interp.test_upsampling()
    test_interp.test_no_change_needed()
    test_interp.test_interpolation_preserves_signal_shape()
    test_interp.test_different_methods()
    test_interp.test_numpy_input()
    print("   ✓ All interpolation tests passed")

    # Test normalization
    print("\n3. Testing normalize_patches...")
    test_norm = TestNormalizePatches()
    test_norm.test_zscore_normalization()
    test_norm.test_minmax_normalization()
    test_norm.test_no_normalization()
    test_norm.test_per_patch_per_channel_independence()
    test_norm.test_zero_std_handling()
    print("   ✓ All normalization tests passed")

    # Test full pipeline
    print("\n4. Testing preprocess_imu_data...")
    test_pipeline = TestPreprocessIMUData()
    test_pipeline.test_full_pipeline_50hz()
    test_pipeline.test_full_pipeline_100hz()
    test_pipeline.test_full_pipeline_200hz()
    test_pipeline.test_pipeline_normalized_output()
    test_pipeline.test_pipeline_with_overlap()
    print("   ✓ All pipeline tests passed")

    # Integration test
    print("\n5. Running integration tests...")
    test_integration_all_datasets()

    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED!")
    print("="*80)
