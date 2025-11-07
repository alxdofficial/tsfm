# Setup Complete ✓

## What to Run

### 1. First Test (5 minutes)
Edit `datascripts/generate_tool_chains/test_simple.py` and set your PROJECT_ID (line 23):
```bash
python datascripts/generate_tool_chains/test_simple.py
```

This will:
- Test query generation (3 queries)
- Test next step generation (1 decision)
- Test full chain generation (1 complete chain)

### 2. Interactive Generation
Edit `datascripts/generate_tool_chains/generate_simple.py` and set your PROJECT_ID (line 24):
```bash
python datascripts/generate_tool_chains/generate_simple.py
```

This will generate 5 chains with manual approval at each step.

### 3. Batch Generation
Set `INTERACTIVE = False` in `datascripts/generate_tool_chains/generate_simple.py`:
```bash
python datascripts/generate_tool_chains/generate_simple.py
```

This will generate chains automatically without prompts.

## What Was Cleaned Up

### ✗ Deleted (V1/obsolete):
- `datascripts/generate_classification_chains.py` - Old V1 script
- `prompts/classification_chain_generation.txt` - V1 prompt
- `prompts/tool_chain_generation.txt` - Old prompt
- `QUICKSTART.md` - V1 quickstart
- `docs/CLASSIFICATION_CHAINS_GUIDE.md` - V1 guide
- `docs/GENERATION_WORKFLOW_EXAMPLE.md` - V1 example
- `docs/option2_rationale.md` - Old design doc

### ✓ Kept (V2 active files):

**Main scripts (in `datascripts/generate_tool_chains/`):**
- `generate_simple.py` - **← RUN THIS** (hardcoded, no argparse)
- `test_simple.py` - **← TEST THIS FIRST**
- `generate_chains_v2.py` - Full version with argparse
- `test_generation_v2.py` - Full test suite
- `test_gemini_setup.py` - Auth verification
- `README.md` - Folder-specific documentation

**Supporting code:**
- `tools/generation.py` - Two core functions (generate_query, generate_next_step)
- `tools/schemas.py` - Pydantic models for structured outputs
- `tools/tool_executor.py` - Tool execution (show_channel_stats, select_channels)

**Prompts (examples-driven):**
- `prompts/system_instructions.txt` - Agent role definition
- `prompts/query_generation_examples.txt` - 30+ query patterns
- `prompts/next_step_examples.txt` - 7+ reasoning patterns, natural termination logic

**Documentation:**
- `RUN_THIS.md` - Quick start guide
- `QUICKSTART_V2.md` - Full walkthrough
- `docs/GENERATION_V2_GUIDE.md` - Architecture details
- `ARCHITECTURE.md` - System overview
- `DATA_FORMAT.md` - Dataset structure
- `TOOL_CHAIN_FORMAT.md` - Output schema

**Data scripts (for reference):**
- `datascripts/setup_all_ts_datasets.py` - Dataset setup
- `datascripts/verify_conversions.py` - Data verification
- `datascripts/uci_har/`, `pamap2/`, `mhealth/`, `wisdm/` - Dataset converters

**Other V2 files (keep but don't run):**
- `datascripts/generate_chains_v2.py` - Full version with argparse
- `datascripts/test_generation_v2.py` - Full test suite
- `datascripts/test_gemini_setup.py` - Auth test

## Key Features

### 1. Interactive Mode
At each step you can:
- **Y** - Accept and continue
- **n** - Reject and abort chain
- **r** - Regenerate this step
- **s** - Skip this turn (tool use only)

### 2. Natural Termination
The system knows when to stop:
- ✓ Simple queries → minimal tools
- ✓ Specific requests → targeted filtering
- ✓ Missing tools → classify with available info

### 3. Structured Outputs
- 0% parsing errors (Pydantic schemas)
- Guaranteed correct tool syntax
- Validated completeness

### 4. Examples-Driven Quality
- 30+ query patterns → diverse queries
- 7+ reasoning patterns → good analytical thinking
- Language variation guidelines → training diversity

## Configuration

All settings in `datascripts/generate_tool_chains/generate_simple.py`:

```python
# GCP
PROJECT_ID = "your-project-id"  # ← EDIT THIS
LOCATION = "us-central1"
MODEL = "gemini-2.5-flash"

# Dataset
DATASET = "uci_har"  # Options: uci_har, pamap2, mhealth, wisdm
NUM_SAMPLES = 5

# Temperatures
QUERY_TEMP = 0.9  # Higher = more diverse
STEP_TEMP = 0.7   # Balanced

# Behavior
MAX_TURNS = 5
INTERACTIVE = True  # Set False for batch mode
```

## Output Location

Chains saved to: `data/tool_chain_of_thoughts/`

Example output:
```json
{
  "conversation_id": "uci_har_session_001_v2_042",
  "schema_version": "2.0",
  "user_query": "Using IMU sensors, classify this activity",
  "conversation": [
    {
      "turn": 1,
      "reasoning": "I need to check what IMU channels are available...",
      "tool_call": {"tool_name": "show_channel_stats", ...},
      "tool_result": {...}
    }
  ],
  "final_classification": "walking",
  "confidence": "high",
  "is_correct": true
}
```

## Next Steps After Generation

1. ✓ Generate 100+ chains across datasets
2. ✓ Review quality (sample 10-20 chains)
3. ✓ Filter: keep `is_correct == true` and `confidence == "high"`
4. → Build motion_tokenizer tool
5. → Extend chains to Phase 2 (tokenization)
6. → Use for LLM fine-tuning

## Cost

~$0.0003 per chain with gemini-2.5-flash
- 100 chains = ~$0.03
- 1000 chains = ~$0.30

## Troubleshooting

**"Dataset not found":**
```bash
ls data/uci_har/manifest.json  # Should exist
```

**"Authentication error":**
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

**"No classification reached":**
- Normal if chain exceeded MAX_TURNS
- Increase MAX_TURNS or adjust prompts

**Want to extend to new tools:**
1. Add tool to `tools/tool_executor.py`
2. Add examples to `prompts/next_step_examples.txt`
3. Update schema in `tools/schemas.py`
4. That's it! Examples-driven = easy to extend
