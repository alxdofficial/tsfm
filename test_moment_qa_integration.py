"""
Test MOMENT QA integration.

Verifies:
1. Dataset loads correctly
2. Collate function works
3. Encoder outputs correct shape
4. QA head accepts embeddings correctly
5. Forward pass works end-to-end
6. Generate function works
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from datasets.ActionSenseMOMENTQA import ActionSenseMOMENTQA, moment_qa_collate
from encoders.moment import MOMENTEncoder
from pretraining.actionsense.heads.moment_qa import MOMENTQAHead
from torch.utils.data import DataLoader

print("=" * 80)
print("Test: MOMENT QA Integration")
print("=" * 80)

# Test parameters
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n[CONFIG]")
print(f"  Device: {device}")

# Step 1: Test Dataset
print(f"\n[STEP 1] Testing ActionSenseMOMENTQA dataset...")
try:
    dataset = ActionSenseMOMENTQA(
        base_dir="data/actionsenseqa_native/data",
        qa_jsonl_path="data/actionsenseqa_native/data/qa_pairs_templated.jsonl",
        manifest_csv_path="data/actionsenseqa_native/data/manifest.csv",
        split="train",
        val_ratio=0.2,
        split_seed=42,
        random_window=False,  # Deterministic for testing
        log_mode="info",
    )

    print(f"  ✓ Dataset loaded: {len(dataset)} samples")

    # Test single sample
    sample = dataset[0]
    print(f"\n  Sample structure:")
    print(f"    - continuous_stream shape: {sample['continuous_stream'].shape}")
    print(f"    - question: {sample['question'][:50]}...")
    print(f"    - answer: {sample['answer']}")
    print(f"    - metadata subject: {sample['metadata']['subject']}")

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
    collated = moment_qa_collate(batch)

    print(f"\n  Collated batch structure:")
    print(f"    - continuous_stream shape: {collated['continuous_stream'].shape}")
    print(f"    - questions: {len(collated['question'])} questions")
    print(f"    - answers: {len(collated['answer'])} answers")

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

# Step 3: Test Encoder
print(f"\n[STEP 3] Testing MOMENTEncoder...")
try:
    encoder = MOMENTEncoder(
        model_size="small",  # Use small for testing
        freeze_moment=False,
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
    assert F == 512, f"Expected F=512 (MOMENT-small), got F={F}"

    print(f"  ✓ Output shape verified: ({B}, {D_out}, {P}, {F})")
    print(f"  ✓ Encoder test PASSED")

except Exception as e:
    print(f"  ✗ Encoder test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Test QA Head
print(f"\n[STEP 4] Testing MOMENTQAHead...")
try:
    qa_head = MOMENTQAHead(
        moment_dim=512,  # MOMENT-small
        llama_model_name="meta-llama/Llama-3.2-1B-Instruct",
        nhead=8,
        dropout=0.1,
        lora_rank=16,
        use_lora=True,
    ).to(device)

    print(f"  ✓ QA head initialized")

    # Forward pass
    print(f"\n  Running forward pass...")
    with torch.no_grad():
        loss, info = qa_head(
            moment_embeddings=encoder_output["embeddings"],
            questions=collated["question"],
            answers=collated["answer"],
        )

    print(f"\n  QA head output:")
    print(f"    - loss: {loss.item():.4f}")
    print(f"    - info keys: {list(info.keys())}")

    print(f"  ✓ Forward pass successful")
    print(f"  ✓ QA head test PASSED")

except Exception as e:
    print(f"  ✗ QA head test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Test Generate
print(f"\n[STEP 5] Testing generate function...")
try:
    print(f"  Running generation...")
    with torch.no_grad():
        generated_answers = qa_head.generate(
            moment_embeddings=encoder_output["embeddings"],
            questions=collated["question"],
            max_new_tokens=32,
            do_sample=False,
        )

    print(f"\n  Generated answers:")
    for i, (q, gen, gt) in enumerate(zip(collated["question"], generated_answers, collated["answer"])):
        print(f"\n  Sample {i+1}:")
        print(f"    Question: {q[:80]}...")
        print(f"    Generated: {gen}")
        print(f"    Ground truth: {gt}")

    assert len(generated_answers) == B, f"Expected {B} answers, got {len(generated_answers)}"
    print(f"\n  ✓ Generated {len(generated_answers)} answers")
    print(f"  ✓ Generate test PASSED")

except Exception as e:
    print(f"  ✗ Generate test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Test DataLoader
print(f"\n[STEP 6] Testing DataLoader...")
try:
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=moment_qa_collate,
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
print(f"  python pretraining/actionsense/moment_qa_pretrain_script.py")
print(f"{'=' * 80}")
