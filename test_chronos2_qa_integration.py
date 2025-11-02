"""
Test updated Chronos-2 QA integration.

Verifies:
1. Encoder outputs correct shape (B, D*num_patches, 2048)
2. Chronos2QAHead accepts embeddings correctly
3. Forward pass works end-to-end
4. Generation works
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from encoders.chronos import Chronos2Encoder
from pretraining.actionsense.heads.chronos2_qa import Chronos2QAHead

print("=" * 80)
print("Test: Chronos-2 QA Integration")
print("=" * 80)

# Test parameters
B = 1  # Reduced from 2 to save memory
D = 18
T = 512  # Reduced from 2016 to test quickly (will give ~500 sensor embeddings)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n[CONFIG]")
print(f"  Batch size: {B}")
print(f"  Channels: {D}")
print(f"  Timesteps: {T}")
print(f"  Device: {device}")

# Step 1: Test Encoder
print(f"\n[STEP 1] Testing Chronos2Encoder...")
encoder = Chronos2Encoder(
    output_dim=2048,
    freeze_chronos=False,
    device=device,
)

batch = {
    "continuous_stream": torch.randn(B, D, T, device=device)
}

encoder_output = encoder(batch)
embeddings = encoder_output["embeddings"]

print(f"  Input shape: {batch['continuous_stream'].shape}")
print(f"  Output shape: {embeddings.shape}")
print(f"  Expected: (B={B}, D*num_patches=?, 2048)")

# Calculate expected shape
num_patches_per_channel = T // 16
expected_seq_len = D * num_patches_per_channel
print(f"  Calculated: (B={B}, seq_len={expected_seq_len}, 2048)")

assert embeddings.shape == (B, expected_seq_len, 2048), \
    f"Shape mismatch! Got {embeddings.shape}, expected ({B}, {expected_seq_len}, 2048)"

print(f"  ✓ Encoder output shape correct!")

# Free encoder to save GPU memory
del encoder
torch.cuda.empty_cache()

# Step 2: Test QA Head
print(f"\n[STEP 2] Testing Chronos2QAHead...")
qa_head = Chronos2QAHead(
    llama_model_name="meta-llama/Llama-3.2-1B-Instruct",
    chronos_dim=2048,
    lora_rank=16,
    lora_alpha=32,
    use_lora=True,
    log_mode="info",
    device=device,
)

questions = [
    "Did the person cook during this period?",
]
answers = [
    "Yes.",
]

print(f"  Questions: {len(questions)}")
print(f"  Embeddings shape: {embeddings.shape}")

# Test forward pass
print(f"\n[STEP 3] Testing forward pass...")
loss, info = qa_head(
    chronos_embeddings=embeddings,
    questions=questions,
    answers=answers,
)

print(f"  Loss: {loss.item():.4f}")
print(f"  Logits shape: {info['logits'].shape}")
print(f"  ✓ Forward pass successful!")

# Test generation
print(f"\n[STEP 4] Testing generation...")
generated = qa_head.generate(
    chronos_embeddings=embeddings,
    questions=questions,
    max_new_tokens=32,
    do_sample=False,
)

print(f"  Generated answers:")
for i, (q, gen, gt) in enumerate(zip(questions, generated, answers)):
    print(f"    Sample {i+1}:")
    print(f"      Q: {q}")
    print(f"      Generated: '{gen}'")
    print(f"      Ground truth: '{gt}'")
    match = gen.lower().strip() == gt.lower().strip()
    print(f"      Match: {'✓' if match else '✗'}")

print(f"\n{'=' * 80}")
print(f"✓ All tests passed!")
print(f"{'=' * 80}")
