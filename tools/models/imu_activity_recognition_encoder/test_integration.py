"""
Integration test for IMU Activity Recognition Encoder.

Tests the complete pipeline end-to-end with realistic scenarios.
"""

import torch
from encoder import IMUActivityRecognitionEncoder
from config import get_config


def test_uci_har_dataset():
    """Test with UCI HAR dataset specifications."""
    print("\n" + "="*80)
    print("Testing UCI HAR Dataset (50 Hz, 9 channels)")
    print("="*80)

    encoder = IMUActivityRecognitionEncoder(**get_config("default"))

    # Simulate 10 seconds of data at 50 Hz
    data = torch.randn(500, 9)

    # Channel descriptions for UCI HAR
    channels = [
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

    # Encode with 2-second patches
    features, metadata = encoder.encode_from_raw(
        data,
        sampling_rate_hz=50.0,
        patch_size_sec=2.0,
        channel_descriptions=channels
    )

    print(f"✓ Raw data shape: {data.shape}")
    print(f"✓ Encoded shape: {features.shape}")
    print(f"✓ Number of patches: {features.shape[0]}")
    print(f"✓ Number of channels: {features.shape[1]}")
    print(f"✓ Feature dimension: {features.shape[2]}")

    # Verify shapes
    assert features.shape == (5, 9, 128), f"Expected (5, 9, 128), got {features.shape}"
    print("✓ UCI HAR test passed!")


def test_actionsense_dataset():
    """Test with ActionSense dataset specifications."""
    print("\n" + "="*80)
    print("Testing ActionSense Dataset (200 Hz, 30 channels)")
    print("="*80)

    encoder = IMUActivityRecognitionEncoder(**get_config("default"))

    # Simulate 10 seconds of data at 200 Hz
    data = torch.randn(2000, 30)

    # Encode with 2-second patches
    features, metadata = encoder.encode_from_raw(
        data,
        sampling_rate_hz=200.0,
        patch_size_sec=2.0
    )

    print(f"✓ Raw data shape: {data.shape}")
    print(f"✓ Encoded shape: {features.shape}")
    print(f"✓ Original patch size: {metadata['per_sample_metadata'][0]['original_patch_size']}")
    print(f"✓ Target patch size: {metadata['per_sample_metadata'][0]['target_patch_size']}")

    # Verify shapes
    assert features.shape == (5, 30, 128), f"Expected (5, 30, 128), got {features.shape}"

    # Verify interpolation worked (400 timesteps -> 64)
    assert metadata['per_sample_metadata'][0]['original_patch_size'] == 400
    assert metadata['per_sample_metadata'][0]['target_patch_size'] == 64

    print("✓ ActionSense test passed!")


def test_mhealth_dataset():
    """Test with MHEALTH dataset specifications."""
    print("\n" + "="*80)
    print("Testing MHEALTH Dataset (50 Hz, 23 channels)")
    print("="*80)

    encoder = IMUActivityRecognitionEncoder(**get_config("default"))

    # Simulate 10 seconds of data at 50 Hz
    data = torch.randn(500, 23)

    # Encode with 2-second patches
    features, metadata = encoder.encode_from_raw(
        data,
        sampling_rate_hz=50.0,
        patch_size_sec=2.0
    )

    print(f"✓ Raw data shape: {data.shape}")
    print(f"✓ Encoded shape: {features.shape}")

    # Verify shapes
    assert features.shape == (5, 23, 128), f"Expected (5, 23, 128), got {features.shape}"
    print("✓ MHEALTH test passed!")


def test_pamap2_dataset():
    """Test with PAMAP2 dataset specifications."""
    print("\n" + "="*80)
    print("Testing PAMAP2 Dataset (100 Hz, 40 channels)")
    print("="*80)

    encoder = IMUActivityRecognitionEncoder(**get_config("default"))

    # Simulate 10 seconds of data at 100 Hz
    data = torch.randn(1000, 40)

    # Encode with 2-second patches
    features, metadata = encoder.encode_from_raw(
        data,
        sampling_rate_hz=100.0,
        patch_size_sec=2.0
    )

    print(f"✓ Raw data shape: {data.shape}")
    print(f"✓ Encoded shape: {features.shape}")

    # Verify shapes
    assert features.shape == (5, 40, 128), f"Expected (5, 40, 128), got {features.shape}"
    print("✓ PAMAP2 test passed!")


def test_batched_processing():
    """Test batched processing across multiple samples."""
    print("\n" + "="*80)
    print("Testing Batched Processing")
    print("="*80)

    encoder = IMUActivityRecognitionEncoder(**get_config("small"))  # Use small for speed

    # Simulate a batch of 16 samples
    batch_size = 16
    batched_data = torch.randn(batch_size, 500, 9)

    # Encode batch
    features, metadata = encoder.encode_from_raw(
        batched_data,
        sampling_rate_hz=50.0,
        patch_size_sec=2.0
    )

    print(f"✓ Batch size: {batch_size}")
    print(f"✓ Encoded shape: {features.shape}")

    # Verify shapes
    assert features.shape == (batch_size, 5, 9, 64), f"Expected (16, 5, 9, 64), got {features.shape}"
    print("✓ Batched processing test passed!")


def test_overlapping_patches():
    """Test overlapping patch generation."""
    print("\n" + "="*80)
    print("Testing Overlapping Patches")
    print("="*80)

    encoder = IMUActivityRecognitionEncoder(**get_config("small"))

    data = torch.randn(1000, 9)  # 10 seconds at 100 Hz

    # Non-overlapping
    features_no_overlap, _ = encoder.encode_from_raw(
        data, sampling_rate_hz=100.0, patch_size_sec=2.0
    )

    # 50% overlap
    features_50_overlap, _ = encoder.encode_from_raw(
        data, sampling_rate_hz=100.0, patch_size_sec=2.0, stride_sec=1.0
    )

    print(f"✓ Non-overlapping patches: {features_no_overlap.shape[0]}")
    print(f"✓ 50% overlap patches: {features_50_overlap.shape[0]}")

    # 50% overlap should produce more patches
    assert features_50_overlap.shape[0] > features_no_overlap.shape[0]
    print("✓ Overlapping patches test passed!")


def test_different_model_sizes():
    """Test small, default, and large model configurations."""
    print("\n" + "="*80)
    print("Testing Different Model Sizes")
    print("="*80)

    data = torch.randn(500, 9)

    for size, expected_dim in [("small", 64), ("default", 128), ("large", 256)]:
        config = get_config(size)
        encoder = IMUActivityRecognitionEncoder(**config)

        features, _ = encoder.encode_from_raw(
            data, sampling_rate_hz=50.0, patch_size_sec=2.0
        )

        print(f"✓ {size.capitalize()} model - d_model: {expected_dim}, shape: {features.shape}")
        assert features.shape[2] == expected_dim

    print("✓ Different model sizes test passed!")


def test_gradient_flow():
    """Test that gradients flow correctly for training."""
    print("\n" + "="*80)
    print("Testing Gradient Flow")
    print("="*80)

    encoder = IMUActivityRecognitionEncoder(**get_config("small"))

    # Create patches with gradients enabled
    patches = torch.randn(2, 5, 96, 9, requires_grad=True)

    # Forward pass
    features = encoder(patches)

    # Dummy loss
    loss = features.sum()

    # Backward pass
    loss.backward()

    print(f"✓ Features shape: {features.shape}")
    print(f"✓ Loss value: {loss.item():.4f}")
    print(f"✓ Patches gradient shape: {patches.grad.shape}")
    print(f"✓ Gradients contain NaN: {torch.isnan(patches.grad).any().item()}")

    # Verify gradients exist and are valid
    assert patches.grad is not None
    assert not torch.isnan(patches.grad).any()
    print("✓ Gradient flow test passed!")


def test_model_parameter_count():
    """Test and report model parameter counts."""
    print("\n" + "="*80)
    print("Testing Model Parameter Counts")
    print("="*80)

    for size in ["small", "default", "large"]:
        config = get_config(size)
        encoder = IMUActivityRecognitionEncoder(**config)

        total_params = sum(p.numel() for p in encoder.parameters())
        trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

        print(f"✓ {size.capitalize()} model:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")

    print("✓ Parameter count test passed!")


def test_reproducibility():
    """Test that results are reproducible with same seed."""
    print("\n" + "="*80)
    print("Testing Reproducibility")
    print("="*80)

    # Set seed
    torch.manual_seed(42)
    encoder1 = IMUActivityRecognitionEncoder(**get_config("small"))
    encoder1.train(False)  # Disable dropout and set to inference mode
    data1 = torch.randn(500, 9)

    torch.manual_seed(42)
    encoder2 = IMUActivityRecognitionEncoder(**get_config("small"))
    encoder2.train(False)  # Disable dropout and set to inference mode
    data2 = torch.randn(500, 9)

    # Both should produce identical results
    with torch.no_grad():
        features1, _ = encoder1.encode_from_raw(data1, 50.0, 2.0)
        features2, _ = encoder2.encode_from_raw(data2, 50.0, 2.0)

    print(f"✓ Features 1 shape: {features1.shape}")
    print(f"✓ Features 2 shape: {features2.shape}")
    print(f"✓ Max difference: {(features1 - features2).abs().max().item():.10f}")

    assert torch.allclose(features1, features2, atol=1e-6)
    print("✓ Reproducibility test passed!")


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*80)
    print("IMU ACTIVITY RECOGNITION ENCODER - INTEGRATION TESTS")
    print("="*80)

    tests = [
        ("UCI HAR Dataset", test_uci_har_dataset),
        ("ActionSense Dataset", test_actionsense_dataset),
        ("MHEALTH Dataset", test_mhealth_dataset),
        ("PAMAP2 Dataset", test_pamap2_dataset),
        ("Batched Processing", test_batched_processing),
        ("Overlapping Patches", test_overlapping_patches),
        ("Different Model Sizes", test_different_model_sizes),
        ("Gradient Flow", test_gradient_flow),
        ("Model Parameter Count", test_model_parameter_count),
        ("Reproducibility", test_reproducibility),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            failed += 1

    print("\n" + "="*80)
    print("INTEGRATION TEST SUMMARY")
    print("="*80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n🎉 ALL INTEGRATION TESTS PASSED! 🎉")
        print("\nThe encoder is ready for:")
        print("  ✓ Pretraining with masked autoencoding")
        print("  ✓ Fine-tuning on activity recognition tasks")
        print("  ✓ Transfer learning to new datasets")
        print("  ✓ Production deployment")
    else:
        print(f"\n❌ {failed} test(s) failed")

    print("="*80)


if __name__ == "__main__":
    run_all_tests()
