"""
Test Chronos-2 CLS integration.

Verifies:
1. Dataset loads correctly
2. Collate function works
3. Encoder outputs correct shape
4. CLS head accepts embeddings correctly
5. Forward pass works end-to-end
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from datasets.ActionSenseChronos2CLS import ActionSenseChronos2CLS, chronos2_cls_collate
from encoders.chronos import Chronos2Encoder
from pretraining.actionsense.heads.chronos2_cls import Chronos2CLSHead
from torch.utils.data import DataLoader

print("=" * 80)
print("Test: Chronos-2 CLS Integration")
print("=" * 80)

# Test parameters
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n[CONFIG]")
print(f"  Device: {device}")

# Step 1: Test Dataset
print(f"\n[STEP 1] Testing ActionSenseChronos2CLS dataset...")
try:
    dataset = ActionSenseChronos2CLS(
        base_dir="data/actionsenseqa_native/data",
        manifest_csv_path="data/actionsenseqa_native/data/manifest.csv",
        split="train",
        val_ratio=0.2,
        split_seed=42,
        target_fps=None,  # Native 60Hz
        window_seconds=5.0,  # Shorter window for testing
        random_window=False,  # Deterministic
        log_mode="info",
    )

    print(f"  ✓ Dataset loaded: {len(dataset)} samples")
    print(f"  ✓ Num classes: {dataset.num_classes}")
    print(f"  ✓ Classes: {dataset.id_to_activity}")

    # Test single sample
    sample = dataset[0]
    print(f"\n  Sample structure:")
    print(f"    - continuous_stream shape: {sample['continuous_stream'].shape}")
    print(f"    - activity_id: {sample['activity_id']}")
    print(f"    - activity_name: {sample['metadata']['activity_name']}")

    D, T = sample["continuous_stream"].shape
    assert D == 18, f"Expected D=18, got D={D}"
    print(f"  ✓ Output shape verified: ({D}, {T})")

except Exception as e:
    print(f"  ✗ Dataset test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Test Collate
print(f"\n[STEP 2] Testing collate function...")
try:
    batch = [dataset[i] for i in range(min(2, len(dataset)))]
    collated = chronos2_cls_collate(batch)

    print(f"\n  Collated batch structure:")
    print(f"    - continuous_stream shape: {collated['continuous_stream'].shape}")
    print(f"    - activity_ids shape: {collated['activity_ids'].shape}")

    B, D, T = collated["continuous_stream"].shape
    assert D == 18, f"Expected D=18, got D={D}"
    assert T % 16 == 0, f"T={T} must be divisible by 16"

    print(f"  ✓ Batch shape verified: ({B}, {D}, {T})")
    print(f"  ✓ Collate test PASSED")

except Exception as e:
    print(f"  ✗ Collate test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Test Encoder
print(f"\n[STEP 3] Testing Chronos2Encoder...")
try:
    encoder = Chronos2Encoder(
        output_dim=2048,
        freeze_chronos=False,
        device=device,
    )

    print(f"  ✓ Encoder initialized")

    # Prepare batch
    batch_input = {
        "continuous_stream": collated["continuous_stream"].to(device),
    }

    print(f"\n  Running forward pass...")
    with torch.no_grad():
        encoder_output = encoder(batch_input)

    print(f"\n  Encoder output structure:")
    print(f"    - embeddings shape: {encoder_output['embeddings'].shape}")
    print(f"    - pad_mask shape: {encoder_output['pad_mask'].shape}")

    B, num_groups, D_out, output_dim = encoder_output["embeddings"].shape
    assert output_dim == 2048, f"Expected output_dim=2048, got {output_dim}"
    assert D_out == 18, f"Expected D=18, got D={D_out}"

    print(f"  ✓ Output shape verified: ({B}, {num_groups}, {D_out}, {output_dim})")
    print(f"  ✓ Encoder test PASSED")

except Exception as e:
    print(f"  ✗ Encoder test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Test CLS Head
print(f"\n[STEP 4] Testing Chronos2CLSHead...")
try:
    cls_head = Chronos2CLSHead(
        d_model=2048,
        nhead=8,
        num_classes=dataset.num_classes,
        dropout=0.1,
    ).to(device)

    print(f"  ✓ CLS head initialized")

    # Forward pass
    with torch.no_grad():
        logits = cls_head(
            chronos_embeddings=encoder_output["embeddings"],
            pad_mask=encoder_output["pad_mask"],
        )

    print(f"\n  CLS head output:")
    print(f"    - logits shape: {logits.shape}")

    B_out, num_classes = logits.shape
    assert num_classes == dataset.num_classes, f"Expected {dataset.num_classes} classes, got {num_classes}"

    print(f"  ✓ Logits shape verified: ({B_out}, {num_classes})")
    print(f"  ✓ CLS head test PASSED")

except Exception as e:
    print(f"  ✗ CLS head test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Test DataLoader
print(f"\n[STEP 5] Testing DataLoader...")
try:
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=chronos2_cls_collate,
        num_workers=0,
    )

    batch = next(iter(dataloader))
    print(f"  ✓ DataLoader created: {len(dataloader)} batches")
    print(f"  ✓ First batch shape: {batch['continuous_stream'].shape}")
    print(f"  ✓ DataLoader test PASSED")

except Exception as e:
    print(f"  ✗ DataLoader test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n{'=' * 80}")
print(f"✓ All tests passed!")
print(f"{'=' * 80}")
print(f"\nYou can now run training with:")
print(f"  python pretraining/actionsense/chronos_cls_pretrain_script.py")
print(f"{'=' * 80}")
