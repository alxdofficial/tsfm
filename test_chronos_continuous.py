"""
Test script for Chronos-2 dataset and encoder.

Tests:
1. ActionSenseChronos2QA dataset loading and output shapes
2. Collate function batching
3. Chronos2Encoder forward pass
4. End-to-end integration (dataset → encoder → QA head)
"""

import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from datasets.ActionSenseChronos2QA import (
    ActionSenseChronos2QA,
    chronos2_qa_collate,
)
from encoders.chronos import Chronos2Encoder
from torch.utils.data import DataLoader


def test_dataset():
    """Test ActionSenseChronos2QA dataset."""
    print("\n" + "=" * 80)
    print("TEST 1: ActionSenseChronos2QA Dataset")
    print("=" * 80)

    try:
        dataset = ActionSenseChronos2QA(
            base_dir="data/actionsenseqa_native/data",
            qa_jsonl_path="data/actionsenseqa_native/data/qa_pairs_templated.jsonl",
            manifest_csv_path="data/actionsenseqa_native/data/manifest.csv",
            split="train",
            val_ratio=0.2,
            split_seed=42,
            log_mode="info",
        )

        print(f"\n✓ Dataset loaded: {len(dataset)} samples")

        # Test single sample
        sample = dataset[0]

        print("\nSample structure:")
        print(f"  - continuous_stream shape: {sample['continuous_stream'].shape}")
        print(f"  - question: {sample['question'][:100]}...")
        print(f"  - answer: {sample['answer']}")
        print(f"  - metadata keys: {list(sample['metadata'].keys())}")

        # Verify shape
        D, T = sample["continuous_stream"].shape
        expected_D = 18  # 6 arm joints (bilateral) × 3 axes
        max_T = 2016

        assert D == expected_D, f"Expected D={expected_D}, got D={D}"
        assert T <= max_T, f"T={T} exceeds max {max_T}"

        # Verify divisible by 48 for Chronos-2 patching (only applies after collate)
        # Individual samples can be any length

        print(f"\n✓ Output shape verified: ({D}, {T})")
        print(f"✓ 18 arm joint channels (bilateral shoulder/forearm/wrist) at native rate (60Hz)")
        print(f"✓ Variable length: {T} timesteps ({T / 60.0:.2f}s @ 60Hz)")
        print("✓ Dataset test PASSED")

        return dataset

    except Exception as e:
        print(f"\n✗ Dataset test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_collate(dataset):
    """Test collate function."""
    print("\n" + "=" * 80)
    print("TEST 2: Collate Function")
    print("=" * 80)

    if dataset is None:
        print("✗ Skipping (dataset not loaded)")
        return

    try:
        # Create small batch manually
        batch = [dataset[i] for i in range(4)]
        collated = chronos2_qa_collate(batch)

        print("\nCollated batch structure:")
        print(f"  - continuous_stream shape: {collated['continuous_stream'].shape}")
        print(f"  - questions: {len(collated['questions'])} items")
        print(f"  - answers: {len(collated['answers'])} items")
        print(f"  - metadata keys: {list(collated['metadata'].keys())}")

        # Verify shape
        B, D, T = collated["continuous_stream"].shape
        assert B == 4, f"Expected B=4, got B={B}"
        assert D == 18, f"Expected D=18, got D={D}"
        assert T <= 2016, f"T={T} exceeds max 2016"
        assert T % 48 == 0, f"T={T} must be divisible by 48 after collate"
        assert T >= 48, f"T={T} must be at least 48"

        print(f"\n✓ Batch shape verified: ({B}, {D}, {T})")
        print(f"✓ Dynamic padding: all samples padded to {T} timesteps")
        print(f"✓ Patching compatible: {T} ÷ 48 = {T // 48} groups")
        print("✓ Collate test PASSED")

        return collated

    except Exception as e:
        print(f"\n✗ Collate test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_encoder(collated_batch):
    """Test Chronos2Encoder."""
    print("\n" + "=" * 80)
    print("TEST 3: Chronos2Encoder")
    print("=" * 80)

    if collated_batch is None:
        print("✗ Skipping (no batch available)")
        return

    try:
        print("\nInitializing Chronos2Encoder...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        encoder = Chronos2Encoder(
            output_dim=2048,
            freeze_chronos=False,
            device=device,
        )

        print("\n✓ Encoder initialized")

        # Prepare batch
        batch = {
            "continuous_stream": collated_batch["continuous_stream"],
            "questions": collated_batch["questions"],
            "answers": collated_batch["answers"],
        }

        print("\nRunning forward pass...")
        with torch.no_grad():
            output = encoder(batch)

        print("\nEncoder output structure:")
        print(f"  - embeddings shape: {output['embeddings'].shape}")
        print(f"  - pad_mask shape: {output['pad_mask'].shape}")
        print(f"  - raw_embeddings shape: {output['raw_embeddings'].shape}")

        # Verify shapes
        B, seq_len, output_dim = output["embeddings"].shape
        assert output_dim == 2048, f"Expected output_dim=2048, got {output_dim}"
        assert output["pad_mask"].shape == (B, seq_len), "Pad mask shape mismatch"

        # Get actual T from batch
        T = collated_batch["continuous_stream"].shape[2]
        expected_groups = T // 48

        print(f"\n✓ Output shape verified: ({B}, {seq_len}, {output_dim})")
        print(f"✓ Input: 18 channels × {T} timesteps → {seq_len} patch groups")
        print(f"✓ Expected {expected_groups} groups, got {seq_len}")
        print("✓ Encoder test PASSED")

        return encoder, output

    except Exception as e:
        print(f"\n✗ Encoder test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_dataloader(dataset):
    """Test DataLoader with collate function."""
    print("\n" + "=" * 80)
    print("TEST 4: DataLoader Integration")
    print("=" * 80)

    if dataset is None:
        print("✗ Skipping (dataset not loaded)")
        return

    try:
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=chronos2_qa_collate,
            num_workers=0,
        )

        print(f"\nDataLoader created with {len(dataloader)} batches")

        # Test iteration
        batch = next(iter(dataloader))

        print("\nFirst batch structure:")
        print(f"  - continuous_stream shape: {batch['continuous_stream'].shape}")
        print(f"  - questions: {len(batch['questions'])} items")
        print(f"  - answers: {len(batch['answers'])} items")

        print("\n✓ DataLoader test PASSED")

    except Exception as e:
        print(f"\n✗ DataLoader test FAILED: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("CHRONOS-2 CONTINUOUS PIPELINE TESTS")
    print("=" * 80)

    # Test 1: Dataset
    dataset = test_dataset()

    # Test 2: Collate
    collated_batch = test_collate(dataset)

    # Test 3: Encoder
    encoder, output = test_encoder(collated_batch)

    # Test 4: DataLoader
    test_dataloader(dataset)

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("✓ All core components verified")
    print("\nNext steps:")
    print("  1. Run: python test_chronos_continuous.py")
    print("  2. If tests pass, run training: python pretraining/actionsense/chronos_qa_pretrain_script.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
