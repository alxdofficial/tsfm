"""Test to debug why generation is producing empty strings."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
from datasets.ActionSenseChronos2QA import ActionSenseChronos2QA, chronos2_qa_collate
from encoders.chronos import Chronos2Encoder
from pretraining.actionsense.heads import SensorQALLMHead

print("Loading dataset...")
dataset = ActionSenseChronos2QA(
    base_dir="data/actionsenseqa_native/data",
    qa_jsonl_path="data/actionsenseqa_native/data/qa_pairs_templated.jsonl",
    manifest_csv_path="data/actionsenseqa_native/data/manifest.csv",
    split="val",
    val_ratio=0.2,
    split_seed=42,
    target_fps=10,
    window_seconds=5.0,
    random_window=False,
    log_mode="info",
)

# Get a small batch
samples = [dataset[i] for i in range(2)]
batch = chronos2_qa_collate(samples)

print(f"\nBatch shape: {batch['continuous_stream'].shape}")
print(f"Questions: {batch['questions']}")
print(f"Answers: {batch['answers']}")

# Initialize model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nInitializing model on {device}...")

encoder = Chronos2Encoder(
    output_dim=2048,
    freeze_chronos=False,
    device=device,
)

qa_head = SensorQALLMHead(
    llama_model_name="meta-llama/Llama-3.2-1B-Instruct",
    feature_dim=2048,
    attn_heads=8,
    attn_dropout=0.1,
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.05,
    use_lora=True,
    log_mode="debug",  # Enable debug logging
)

# Move batch to device
batch["continuous_stream"] = batch["continuous_stream"].to(device)

print("\nEncoding sensor data...")
with torch.no_grad():
    encoder_output = encoder({"continuous_stream": batch["continuous_stream"]})

    print(f"Encoder output shape: {encoder_output['embeddings'].shape}")

    print("\nGenerating answers...")
    generated = qa_head.generate(
        tokens=encoder_output["embeddings"],
        pad_mask=encoder_output["pad_mask"],
        questions=batch["questions"],
        max_new_tokens=32,
        do_sample=False,
    )

print(f"\n{'='*60}")
print("Generation Results:")
print(f"{'='*60}")
for i, (q, gt, gen) in enumerate(zip(batch["questions"], batch["answers"], generated)):
    print(f"\nSample {i+1}:")
    print(f"  Question: {q}")
    print(f"  Ground truth: {gt}")
    print(f"  Generated: '{gen}'")
    print(f"  Generated length: {len(gen)} chars")
    print(f"  Is empty: {len(gen.strip()) == 0}")
