# Tool Chain Generation Scripts

This folder contains all scripts for generating tool use chain-of-thought training data.

## Quick Start

### 1. Test the System
```bash
# Edit PROJECT_ID in test_simple.py (line 23)
python datascripts/generate_tool_chains/test_simple.py
```

### 2. Generate Chains (Simple, Hardcoded)
```bash
# Edit PROJECT_ID in generate_simple.py (line 24)
python datascripts/generate_tool_chains/generate_simple.py
```

### 3. Generate Chains (Full, with ArgParse)
```bash
python datascripts/generate_tool_chains/generate_chains_v2.py \
    --project YOUR_PROJECT_ID \
    --dataset uci_har \
    --num-samples 10 \
    --verbose \
    --interactive
```

## Scripts

### Simple Scripts (Hardcoded Config)

**`generate_simple.py`**
- Main generation script
- All config at top of file (no argparse)
- Edit PROJECT_ID, DATASET, NUM_SAMPLES, INTERACTIVE
- Best for quick testing and iteration

**`test_simple.py`**
- Tests both generation functions + full chain
- Hardcoded config
- Edit PROJECT_ID at top
- Run this first to verify everything works

### Full Scripts (ArgParse)

**`generate_chains_v2.py`**
- Full-featured generation script
- Command-line arguments for all options
- Interactive mode with manual approval
- More flexible for batch generation

**`test_generation_v2.py`**
- Comprehensive test suite
- Command-line arguments
- Tests individual functions and full chains

**`test_gemini_setup.py`**
- Verifies Vertex AI authentication
- Tests basic Gemini API calls
- Useful for debugging connection issues

## Configuration

### In `generate_simple.py`:
```python
PROJECT_ID = "your-project-id"  # ← CHANGE THIS
LOCATION = "us-central1"
MODEL = "gemini-2.5-flash"
DATASET = "uci_har"  # Options: uci_har, pamap2, mhealth, wisdm
NUM_SAMPLES = 5
QUERY_TEMP = 0.9
STEP_TEMP = 0.7
INTERACTIVE = True  # Set False for batch mode
```

### With `generate_chains_v2.py`:
```bash
--project YOUR_PROJECT_ID      # Required
--dataset uci_har              # Required
--location us-central1         # Default
--model gemini-2.5-flash       # Default
--num-samples 10               # Default: 5
--query-temp 0.9               # Default
--step-temp 0.7                # Default
--verbose                      # Show progress
--interactive                  # Manual approval
```

## Output

All chains saved to: `data/tool_chain_of_thoughts/`

Each chain includes:
- Generated user query
- Turn-by-turn tool use reasoning
- Tool execution results
- Final classification
- Correctness evaluation

## Dependencies

These scripts use:
- `generation.py` (this folder) - Core functions (generate_query, generate_next_step)
- `schemas.py` (this folder) - Pydantic models for structured outputs
- `prompts/` (this folder) - Example-driven prompts
  - `system_instructions.txt` - Agent role definition
  - `query_generation_examples.txt` - 30+ query patterns
  - `next_step_examples.txt` - 7+ reasoning patterns
- `../../tools/tool_executor.py` - Tool execution logic

## Interactive Mode

When `--interactive` is enabled (or `INTERACTIVE = True`):

**At each step you can:**
- **Y** - Accept and continue
- **n** - Reject and abort chain
- **r** - Regenerate this step
- **s** - Skip this turn (tool use only)

**You'll approve:**
1. Generated user query
2. Each tool use decision
3. Final classification
4. Whether to save the complete chain

## Examples

### Quick Test
```bash
python datascripts/generate_tool_chains/test_simple.py
```

### Generate 5 Chains Interactively
```bash
python datascripts/generate_tool_chains/generate_simple.py
# Set INTERACTIVE = True in file
```

### Generate 100 Chains in Batch
```bash
python datascripts/generate_tool_chains/generate_chains_v2.py \
    --project YOUR_PROJECT_ID \
    --dataset uci_har \
    --num-samples 100 \
    --model gemini-2.5-flash \
    --verbose
# Note: no --interactive flag = batch mode
```

### Generate Across All Datasets
```bash
for dataset in uci_har pamap2 mhealth wisdm; do
    python datascripts/generate_tool_chains/generate_chains_v2.py \
        --project YOUR_PROJECT_ID \
        --dataset $dataset \
        --num-samples 25 \
        --verbose
done
```

## Troubleshooting

**Import errors:**
- Scripts use `sys.path.insert` to find project root
- Should work from any directory

**Dataset not found:**
```bash
ls data/uci_har/manifest.json  # Should exist
```

**Authentication errors:**
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

**Test Vertex AI setup:**
```bash
python datascripts/generate_tool_chains/test_gemini_setup.py \
    --project YOUR_PROJECT_ID
```
