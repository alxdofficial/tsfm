# Exporting Threads for Fine-tuning

This document explains how to export your generated threads to JSONL training format compatible with Llama and other LLM fine-tuning frameworks.

## Important: Native OpenAI Format Storage

**As of v4, threads are stored directly in OpenAI format!** This means:
- No conversion needed - threads are already training-ready
- Export just copies filtered threads to a JSONL file
- Simplified workflow with better compatibility

## Quick Start

### From the Interactive Tool

```bash
python datascripts/generate_tool_chains/generate_simple.py

# In the main menu, select option 3
[3] Export to JSONL training format
```

This will:
1. Let you filter by finished threads or include all
2. Copy threads to `data/training_exports/training_data_openai_TIMESTAMP.jsonl`
3. Show export summary and preview

## Output Format

### OpenAI Messages Format

**File**: `data/training_exports/training_data_openai_YYYYMMDD_HHMMSS.jsonl`

This is the industry-standard format compatible with:
- ✅ Axolotl
- ✅ LLaMA-Factory
- ✅ TRL (HuggingFace)
- ✅ Most fine-tuning frameworks

**Structure:**
```json
{
  "id": "uci_har_train_16_3157_v4_1234",
  "metadata": {
    "dataset": "uci_har",
    "session": "train_16_3157",
    "ground_truth": "walking",
    "status": "finished",
    "num_turns": 3,
    "num_tool_calls": 2,
    "num_responds": 0,
    "is_correct": true,
    "confidence": "high",
    "user_query": "What activity is being performed?",
    "schema_version": "openai_v1",
    "created_at": "2025-11-08T...",
    "last_modified": "2025-11-08T..."
  },
  "messages": [
    {
      "role": "user",
      "content": "Dataset Information:\n{\n  \"dataset_name\": \"UCI HAR\",\n  \"description\": \"Human activity recognition from smartphone sensors...\",\n  \"channels\": [...]\n}\n\nUser Query: What activity is being performed?"
    },
    {
      "role": "assistant",
      "content": "I need to check what sensors are available first.",
      "tool_calls": [{
        "id": "call_uci_har_train_16_3157_v4_1234_turn_1",
        "type": "function",
        "function": {
          "name": "show_channel_stats",
          "arguments": "{\"dataset_name\": \"uci_har\"}"
        }
      }]
    },
    {
      "role": "tool",
      "tool_call_id": "call_uci_har_train_16_3157_v4_1234_turn_1",
      "name": "show_channel_stats",
      "content": "{\"channels\": [{\"name\": \"body_acc_x\", ...}], \"num_channels\": 9}"
    },
    {
      "role": "assistant",
      "content": "Based on the periodic acceleration patterns, this is walking.\n\nFinal Answer: walking",
      "tool_calls": null
    }
  ]
}
```

## File Format: JSONL

The output uses **JSONL** (JSON Lines):
- Each line is a complete JSON object (one training example)
- Easy to stream and process
- Standard for most training frameworks
- No conversion needed - threads are already in this format!

## Using with Fine-tuning Frameworks

### Axolotl

```yaml
# config.yml
datasets:
  - path: data/training_exports/training_data_openai_20250107_103000.jsonl
    type: sharegpt  # Axolotl auto-detects OpenAI format as sharegpt-compatible
    conversation: openai  # Specify OpenAI format
```

### LLaMA-Factory

```json
{
  "dataset_name": {
    "file_name": "data/training_exports/training_data_openai_20250107_103000.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages"
    }
  }
}
```

### TRL (Transformers)

```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files="data/training_exports/training_data_openai_20250107_103000.jsonl"
)

# TRL expects 'messages' column
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    formatting_func=lambda x: x["messages"]  # Already in correct format!
)
```

## Filtering and Quality Control

### Filter Options

When exporting from the interactive tool, you can choose:

1. **Finished threads only (recommended)**: Only exports threads with `status="finished"`
2. **All threads**: Includes active, finished, and max_turns_reached threads

### Manual Review

Before training, review the exported data:

```python
import json

with open("data/training_exports/training_data_openai_20251108_120000.jsonl") as f:
    for line in f:
        thread = json.loads(line)

        # Check correctness
        if thread["metadata"].get("is_correct") == False:
            print(f"Incorrect: {thread['id']}")

        # Check length
        if len(thread["messages"]) < 4:
            print(f"Too short: {thread['id']}")

        # Check tool usage
        if thread["metadata"]["num_tool_calls"] < 1:
            print(f"No tools used: {thread['id']}")
```

### Quality Metrics to Check

1. **Correctness**: `metadata.is_correct` should be `true`
2. **Length**: At least 1-2 tool calls per conversation
3. **Diversity**: Mix of datasets and activities
4. **Tool usage**: Good balance of tool calls vs. reasoning
5. **Completion**: `metadata.status` should be "finished"

## Advanced Usage

### Custom Filtering with Python

Since threads are already in OpenAI format, you can directly filter and copy them:

```python
from pathlib import Path
import json

threads_dir = Path("data/tool_chain_of_thoughts/threads")
output = []

for thread_file in threads_dir.glob("*.json"):
    with open(thread_file) as f:
        thread = json.load(f)

    # Custom filters
    if (thread["metadata"]["status"] == "finished" and
        thread["metadata"]["num_tool_calls"] >= 2 and
        thread["metadata"].get("is_correct") == True):
        output.append(thread)

# Save filtered
with open("data/training_exports/custom_filtered.jsonl", "w") as f:
    for thread in output:
        f.write(json.dumps(thread) + "\n")

print(f"Exported {len(output)} threads")
```

### Inspect Thread Files Directly

Threads are stored in human-readable JSON:

```bash
# View a thread
cat data/tool_chain_of_thoughts/threads/uci_har_train_16_3157_v4_1234.json | jq '.'

# Count finished threads
jq -s 'map(select(.metadata.status == "finished")) | length' data/tool_chain_of_thoughts/threads/*.json

# List all thread IDs with status
jq -r '"\(.id): \(.metadata.status)"' data/tool_chain_of_thoughts/threads/*.json
```

## Troubleshooting

### "No threads found matching the filter"
- Make sure you've generated some threads first
- Check `data/tool_chain_of_thoughts/threads/` exists and has JSON files
- If filtering by "finished only", make sure you have completed threads
- Try option 2 to include all threads

### Threads in old v3 format
- If you have threads from the old v3 format, they're incompatible with v4
- Old threads use `thread_id`, `conversation`, etc.
- New threads use `id`, `messages`, and nest everything in `metadata`
- Move old threads to a backup location or regenerate them

## Best Practices

1. **Generate diverse data**: Use all 4 datasets
2. **Complete threads**: Aim for 2-3 tool calls + final answer
3. **Quality over quantity**: 50 high-quality threads > 500 low-quality
4. **Review before training**: Spot-check the exported data
5. **Version your exports**: Timestamps in filename help track iterations

## Next Steps

After exporting:
1. Review the JSONL file
2. Split into train/val (e.g., 90/10)
3. Configure your training framework
4. Fine-tune!
5. Evaluate on held-out sessions

## What Changed in v4?

The v4 refactoring made the export process much simpler:

### Before (v3)
- Threads stored in custom format with `thread_id`, `conversation`, etc.
- Required conversion script (`export_training_data.py`)
- Two-step process: generate → convert
- Separate export utility with validation

### After (v4)
- **Native OpenAI format storage** - threads are training-ready immediately
- No conversion needed - just filter and copy to JSONL
- Simplified export: choose filter, get JSONL file
- Thread structure matches training format exactly

### Benefits
- ✅ Cleaner codebase - removed conversion logic
- ✅ Faster export - no conversion overhead
- ✅ Direct inspection - threads are readable as-is
- ✅ Better compatibility - industry-standard format
- ✅ Simpler workflow - one format throughout

### Migration from v3
If you have v3 threads, move them to a backup location:
```bash
mkdir -p data/tool_chain_of_thoughts/threads_v3_backup
mv data/tool_chain_of_thoughts/threads/*_v3_*.json data/tool_chain_of_thoughts/threads_v3_backup/
```

## Support

For issues or questions:
- Check thread files in `data/tool_chain_of_thoughts/threads/`
- Verify threads have `"schema_version": "openai_v1"`
- Ensure threads are properly finished (`status: "finished"`)
