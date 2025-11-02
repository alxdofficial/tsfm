"""Quick test to verify dataset downsampling is working."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from datasets.ActionSenseChronos2QA import ActionSenseChronos2QA

# Test with 10Hz, 10 second windows
dataset = ActionSenseChronos2QA(
    base_dir="data/actionsenseqa_native/data",
    qa_jsonl_path="data/actionsenseqa_native/data/qa_pairs_templated.jsonl",
    manifest_csv_path="data/actionsenseqa_native/data/manifest.csv",
    split="val",
    val_ratio=0.2,
    split_seed=42,
    target_fps=10,
    window_seconds=10.0,
    random_window=False,
    log_mode="info",
)

print(f"\n{'='*60}")
print("Dataset Configuration:")
print(f"{'='*60}")
print(f"Target FPS: {dataset.target_fps}")
print(f"Window seconds: {dataset.window_seconds}")
print(f"Window timesteps: {dataset.window_timesteps}")
print(f"Downsample factor: {dataset.downsample_factor}")
print(f"Random window: {dataset.random_window}")

# Get a sample
sample = dataset[0]
stream = sample["continuous_stream"]
D, T = stream.shape

print(f"\n{'='*60}")
print("Sample Data:")
print(f"{'='*60}")
print(f"Channels (D): {D}")
print(f"Timesteps (T): {T}")
print(f"Expected timesteps at 10Hz for 10s: {10 * 10} = 100")
print(f"Actual matches expected: {T == 100}")
print(f"\nQuestion: {sample['question']}")
print(f"Answer: {sample['answer']}")
print(f"{'='*60}")
