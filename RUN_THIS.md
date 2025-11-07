# Quick Start - V2 Generation System

## Setup (One Time)

1. **Install dependencies:**
```bash
pip install google-genai pydantic numpy pandas matplotlib pyarrow
```

2. **Authenticate with GCP:**
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

3. **Edit configuration** in `datascripts/generate_tool_chains/generate_simple.py`:
```python
PROJECT_ID = "your-project-id"  # ← CHANGE THIS
DATASET = "uci_har"              # Options: uci_har, pamap2, mhealth, wisdm
NUM_SAMPLES = 5
INTERACTIVE = True               # Set False for batch mode
```

## Run

### Test the system:
```bash
# Edit PROJECT_ID in test_simple.py first (line 23)
python datascripts/generate_tool_chains/test_simple.py
```

### Generate chains interactively:
```bash
# Edit PROJECT_ID in generate_simple.py first (line 24)
python datascripts/generate_tool_chains/generate_simple.py
```

You'll be prompted to approve:
- Each generated query (Y/n/r to regenerate)
- Each tool decision (Y/n/s to skip/r to regenerate)
- Each final classification (Y/n/r to regenerate)
- Whether to save the chain (Y/n)

### Generate chains in batch (no prompts):
```bash
# Set INTERACTIVE = False in generate_simple.py
python generate_simple.py
```

## Output

Chains are saved to: `data/tool_chain_of_thoughts/`

Each chain is a JSON file with:
- User query
- Tool use reasoning
- Tool execution results
- Final classification
- Correctness evaluation

## Configuration Options

Edit at the top of `generate_simple.py`:

```python
# GCP Settings
PROJECT_ID = "your-project-id"
LOCATION = "us-central1"
MODEL = "gemini-2.5-flash"

# Dataset Settings
DATASET = "uci_har"
NUM_SAMPLES = 5

# Temperature Settings
QUERY_TEMP = 0.9  # Higher = more diverse queries
STEP_TEMP = 0.7   # Balanced creativity

# Behavior
MAX_TURNS = 5
INTERACTIVE = True
```

## Files You Need

**Core scripts:**
- `generate_simple.py` - Main generation (hardcoded config)
- `test_simple.py` - Test the system

**Supporting code:**
- `tools/generation.py` - Two core functions
- `tools/schemas.py` - Pydantic models
- `tools/tool_executor.py` - Tool execution

**Prompts:**
- `prompts/system_instructions.txt` - Agent role
- `prompts/query_generation_examples.txt` - 30+ query patterns
- `prompts/next_step_examples.txt` - 7+ reasoning patterns

**Documentation:**
- `QUICKSTART_V2.md` - Full guide
- `docs/GENERATION_V2_GUIDE.md` - Architecture details
- `ARCHITECTURE.md` - System overview

## Common Issues

**"Dataset not found":**
- Make sure dataset exists in `data/DATASET/manifest.json`

**"Failed to initialize client":**
- Run `gcloud auth application-default login`
- Check PROJECT_ID is correct

**"No final classification":**
- Normal - chain exceeded MAX_TURNS without classifying
- Will be skipped automatically

**Parse errors:**
- Should never happen with V2 (structured outputs!)
- If it does, report as bug
