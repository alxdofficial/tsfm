"""
Debug script to verify MOMENT dataset is working correctly.
"""
import sys
sys.path.insert(0, '.')

from datasets.ActionSenseMOMENTCLS import ActionSenseMOMENTCLS, moment_cls_collate
from torch.utils.data import DataLoader

print("=" * 80)
print("MOMENT Dataset Debug")
print("=" * 80)

# Configuration matching training script
BASE_DIR = "data/actionsenseqa_native/data"
MANIFEST_CSV = "data/actionsenseqa_native/data/manifest.csv"
VAL_RATIO = 0.2
SPLIT_SEED = 42
BATCH_SIZE = 16

print("\n[1/3] Creating train dataset...")
train_dataset = ActionSenseMOMENTCLS(
    base_dir=BASE_DIR,
    manifest_csv_path=MANIFEST_CSV,
    split="train",
    val_ratio=VAL_RATIO,
    split_seed=SPLIT_SEED,
    random_window=True,
    log_mode="info",
)

print("\n[2/3] Creating val dataset...")
val_dataset = ActionSenseMOMENTCLS(
    base_dir=BASE_DIR,
    manifest_csv_path=MANIFEST_CSV,
    split="val",
    val_ratio=VAL_RATIO,
    split_seed=SPLIT_SEED,
    random_window=False,
    log_mode="info",
)

print("\n[3/3] Creating dataloaders...")
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=moment_cls_collate,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=moment_cls_collate,
    num_workers=0,
)

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Train dataset: {len(train_dataset)} samples")
print(f"Val dataset: {len(val_dataset)} samples")
print(f"Num classes: {train_dataset.num_classes}")
print(f"\nTrain dataloader: {len(train_loader)} batches")
print(f"Val dataloader: {len(val_loader)} batches")

# Test iteration
print("\n" + "=" * 80)
print("Testing iteration...")
print("=" * 80)

print("\nTrain loader:")
for i, batch in enumerate(train_loader):
    if i == 0:
        print(f"  Batch 0 shape: {batch['continuous_stream'].shape}")
        print(f"  Batch 0 activity_ids: {batch['activity_ids'].shape}")
    if i >= 2:
        print(f"  ... batch {i}")

print(f"  Total batches iterated: {i + 1}")

print("\nVal loader:")
for i, batch in enumerate(val_loader):
    if i == 0:
        print(f"  Batch 0 shape: {batch['continuous_stream'].shape}")
        print(f"  Batch 0 activity_ids: {batch['activity_ids'].shape}")
    if i >= 2:
        print(f"  ... batch {i}")

print(f"  Total batches iterated: {i + 1}")

print("\n" + "=" * 80)
print("✓ Dataset working correctly!")
print("=" * 80)
