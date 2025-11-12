"""
Quick start example for IMU Activity Recognition Encoder.

This script demonstrates basic usage of the encoder with synthetic data.
"""

import torch
from encoder import IMUActivityRecognitionEncoder
from config import get_config


def example_1_basic_usage():
    """Example 1: Basic usage with raw sensor data."""
    print("="*80)
    print("Example 1: Basic Usage")
    print("="*80)

    # Create encoder with default configuration
    encoder = IMUActivityRecognitionEncoder(**get_config("default"))

    # Generate synthetic IMU data: 10 seconds at 100 Hz, 9 channels
    # In practice, this would be real sensor data from accelerometer + gyroscope
    data = torch.randn(1000, 9)

    # Encode the data with 2-second patches
    features, metadata = encoder.encode_from_raw(
        data,
        sampling_rate_hz=100.0,
        patch_size_sec=2.0
    )

    print(f"Input shape: {data.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Encoded as: {features.shape[0]} patches × {features.shape[1]} channels × {features.shape[2]} features")
    print()


def example_2_with_channel_descriptions():
    """Example 2: Using channel semantic descriptions."""
    print("="*80)
    print("Example 2: With Channel Descriptions")
    print("="*80)

    encoder = IMUActivityRecognitionEncoder(**get_config("default"))

    # UCI HAR dataset format
    data = torch.randn(500, 9)  # 10 seconds at 50 Hz

    # Define channel meanings
    channel_descriptions = [
        "body accelerometer x-axis",
        "body accelerometer y-axis",
        "body accelerometer z-axis",
        "body gyroscope x-axis",
        "body gyroscope y-axis",
        "body gyroscope z-axis",
        "total accelerometer x-axis",
        "total accelerometer y-axis",
        "total accelerometer z-axis"
    ]

    features, metadata = encoder.encode_from_raw(
        data,
        sampling_rate_hz=50.0,
        patch_size_sec=2.0,
        channel_descriptions=channel_descriptions
    )

    print(f"Used channel descriptions: {len(channel_descriptions)} channels")
    print(f"Output shape: {features.shape}")
    print()


def example_3_batched_processing():
    """Example 3: Processing multiple samples in a batch."""
    print("="*80)
    print("Example 3: Batched Processing")
    print("="*80)

    encoder = IMUActivityRecognitionEncoder(**get_config("small"))  # Use small for speed

    # Simulate a batch of 8 samples
    batch_data = torch.randn(8, 500, 9)  # 8 samples, 10 seconds each at 50 Hz

    features, metadata = encoder.encode_from_raw(
        batch_data,
        sampling_rate_hz=50.0,
        patch_size_sec=2.0
    )

    print(f"Batch size: {batch_data.shape[0]}")
    print(f"Output shape: {features.shape}")
    print(f"Features per sample: {features.shape[1]} patches × {features.shape[2]} channels × {features.shape[3]} features")
    print()


def example_4_different_datasets():
    """Example 4: Encoding data from different datasets."""
    print("="*80)
    print("Example 4: Different Datasets")
    print("="*80)

    encoder = IMUActivityRecognitionEncoder(**get_config("default"))

    datasets = [
        ("UCI HAR", 50, 9, 500),
        ("ActionSense", 200, 30, 2000),
        ("MHEALTH", 50, 23, 500),
        ("PAMAP2", 100, 40, 1000)
    ]

    for name, sampling_rate, num_channels, num_timesteps in datasets:
        data = torch.randn(num_timesteps, num_channels)
        features, _ = encoder.encode_from_raw(
            data,
            sampling_rate_hz=float(sampling_rate),
            patch_size_sec=2.0
        )
        print(f"{name:15s} ({sampling_rate:3d} Hz, {num_channels:2d} ch): {data.shape} → {features.shape}")

    print()


def example_5_overlapping_patches():
    """Example 5: Using overlapping patches for smoother representations."""
    print("="*80)
    print("Example 5: Overlapping Patches")
    print("="*80)

    encoder = IMUActivityRecognitionEncoder(**get_config("default"))

    data = torch.randn(1000, 9)

    # Non-overlapping patches
    features_no_overlap, _ = encoder.encode_from_raw(
        data,
        sampling_rate_hz=100.0,
        patch_size_sec=2.0
    )

    # 50% overlap
    features_50_overlap, _ = encoder.encode_from_raw(
        data,
        sampling_rate_hz=100.0,
        patch_size_sec=2.0,
        stride_sec=1.0
    )

    print(f"Non-overlapping: {features_no_overlap.shape[0]} patches")
    print(f"50% overlap: {features_50_overlap.shape[0]} patches")
    print(f"Overlap gives {features_50_overlap.shape[0] / features_no_overlap.shape[0]:.1f}× more patches")
    print()


def example_6_model_sizes():
    """Example 6: Comparing different model sizes."""
    print("="*80)
    print("Example 6: Model Size Comparison")
    print("="*80)

    data = torch.randn(500, 9)

    for size in ["small", "default", "large"]:
        config = get_config(size)
        encoder = IMUActivityRecognitionEncoder(**config)

        # Count parameters
        total_params = sum(p.numel() for p in encoder.parameters())

        features, _ = encoder.encode_from_raw(data, 50.0, 2.0)

        print(f"{size.capitalize():8s} - {total_params:,} params, d_model={config['d_model']:3d}, output: {features.shape}")

    print()


def example_7_training_ready():
    """Example 7: Preparing data for training."""
    print("="*80)
    print("Example 7: Training-Ready Output")
    print("="*80)

    encoder = IMUActivityRecognitionEncoder(**get_config("default"))

    # Simulate a training batch
    batch_size = 16
    batch_data = torch.randn(batch_size, 500, 9)
    labels = torch.randint(0, 6, (batch_size,))  # 6 activity classes

    # Encode
    features, _ = encoder.encode_from_raw(batch_data, 50.0, 2.0)

    # For classification, you might want to pool over patches and channels
    # This is just an example - actual task heads would be implemented separately
    pooled_features = features.mean(dim=(1, 2))  # Average over patches and channels

    print(f"Batch data shape: {batch_data.shape}")
    print(f"Encoded features shape: {features.shape}")
    print(f"Pooled features shape: {pooled_features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"\nThese features can now be passed to a classification head:")
    print(f"  - Input: {pooled_features.shape}")
    print(f"  - Output: (batch_size, num_classes)")
    print()


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("IMU ACTIVITY RECOGNITION ENCODER - USAGE EXAMPLES")
    print("="*80 + "\n")

    example_1_basic_usage()
    example_2_with_channel_descriptions()
    example_3_batched_processing()
    example_4_different_datasets()
    example_5_overlapping_patches()
    example_6_model_sizes()
    example_7_training_ready()

    print("="*80)
    print("All examples completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
