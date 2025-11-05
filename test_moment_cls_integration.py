"""
Test MOMENT CLS integration.

Verifies:
1. Dataset loads correctly and produces (18, 512) tensors
2. Collate function works
3. MOMENT encoder outputs correct shape
4. CLS head accepts embeddings correctly
5. Forward pass works end-to-end
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from datasets.ActionSenseMOMENTCLS import ActionSenseMOMENTCLS, moment_cls_collate
from encoders.moment.encoder import MOMENTEncoder
from pretraining.actionsense.heads.moment_cls import MOMENTCLSHead
from torch.utils.data import DataLoader

print("=" * 80)
print("Test: MOMENT CLS Integration")
print("=" * 80)

# Test parameters
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n[CONFIG]")
print(f"  Device: {device}")

# Step 1: Test Dataset
print(f"\n[STEP 1] Testing ActionSenseMOMENTCLS dataset...")
try:
    dataset = ActionSenseMOMENTCLS(
        base_dir="data/actionsenseqa_native/data",
        manifest_csv_path="data/actionsenseqa_native/data/manifest.csv",
        split="train",
        val_ratio=0.2,
        split_seed=42,
        random_window=False,  # Deterministic for testing
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
    assert T == 512, f"Expected T=512, got T={T}"
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
    collated = moment_cls_collate(batch)

    print(f"\n  Collated batch structure:")
    print(f"    - continuous_stream shape: {collated['continuous_stream'].shape}")
    print(f"    - activity_ids shape: {collated['activity_ids'].shape}")

    B, D, T = collated["continuous_stream"].shape
    assert D == 18, f"Expected D=18, got D={D}"
    assert T == 512, f"Expected T=512, got T={T}"

    print(f"  ✓ Batch shape verified: ({B}, {D}, {T})")
    print(f"  ✓ Collate test PASSED")

except Exception as e:
    print(f"  ✗ Collate test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Test MOMENT Encoder
print(f"\n[STEP 3] Testing MOMENTEncoder...")
try:
    encoder = MOMENTEncoder(
        model_size="small",
        freeze_moment=True,
        output_dim=None,  # Use native 512
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
    print(f"    - pad_mask: {encoder_output['pad_mask']}")

    B, D_out, P, F = encoder_output["embeddings"].shape
    assert D_out == 18, f"Expected D=18, got D={D_out}"
    assert P == 64, f"Expected P=64 patches, got P={P}"
    assert F == encoder.output_dim, f"Expected F={encoder.output_dim}, got F={F}"

    print(f"  ✓ Output shape verified: ({B}, {D_out}, {P}, {F})")
    print(f"  ✓ Encoder test PASSED")

except Exception as e:
    print(f"  ✗ Encoder test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Test CLS Head
print(f"\n[STEP 4] Testing MOMENTCLSHead...")
try:
    cls_head = MOMENTCLSHead(
        d_model=encoder.output_dim,
        nhead=8,
        num_classes=dataset.num_classes,
        dropout=0.1,
    ).to(device)

    print(f"  ✓ CLS head initialized")

    # Forward pass
    with torch.no_grad():
        logits = cls_head(
            moment_embeddings=encoder_output["embeddings"],
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
        collate_fn=moment_cls_collate,
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

# Step 6: Test End-to-End (manual composition)
print(f"\n[STEP 6] Testing end-to-end forward pass...")
try:
    # Manually compose encoder + head (since training script doesn't exist yet)
    print(f"  ✓ Using manual composition: encoder + cls_head")

    batch_input = {
        "continuous_stream": batch["continuous_stream"].to(device),
    }

    with torch.no_grad():
        # Encoder
        encoder_output = encoder(batch_input)
        # CLS head
        output = cls_head(
            moment_embeddings=encoder_output["embeddings"],
            pad_mask=encoder_output["pad_mask"],
        )

    print(f"\n  End-to-end output:")
    print(f"    - logits shape: {output.shape}")

    B_final, num_classes_final = output.shape
    assert num_classes_final == dataset.num_classes, f"Expected {dataset.num_classes}, got {num_classes_final}"

    print(f"  ✓ End-to-end test PASSED")

except Exception as e:
    print(f"  ✗ End-to-end test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n{'=' * 80}")
print(f"✓ All tests passed!")
print(f"{'=' * 80}")
print(f"\nMOMENT CLS system ready for training!")
print(f"Note: Training script to be created separately")
print(f"{'=' * 80}")
