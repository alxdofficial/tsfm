# START HERE

## 📁 New Organized Structure

All tool chain generation scripts are now in:
```
datascripts/generate_tool_chains/
```

This makes it clear which scripts are for generating training data vs. converting datasets.

## 🚀 Quick Start

### 1. Test the System (5 minutes)
```bash
# Edit PROJECT_ID on line 23
python datascripts/generate_tool_chains/test_simple.py
```

Expected output:
```
✓ Gemini client initialized
TEST 1: Query generation working!
TEST 2: Next step generation working!
TEST 3: Full chain generation working!
✓ All tests passed!
```

### 2. Generate Your First Chains
```bash
# Edit PROJECT_ID on line 24
python datascripts/generate_tool_chains/generate_simple.py
```

You'll interactively approve each step:
- Query generation
- Tool decisions
- Final classification
- Save confirmation

### 3. Batch Generate (No Prompts)
Edit `datascripts/generate_tool_chains/generate_simple.py`:
```python
INTERACTIVE = False  # Change from True
```

Then run:
```bash
python datascripts/generate_tool_chains/generate_simple.py
```

## 📂 Project Structure

```
tsfm/
├── datascripts/
│   ├── generate_tool_chains/     ← ALL CHAIN GENERATION CODE
│   │   ├── README.md             ← Full documentation
│   │   ├── generate_simple.py    ← RUN THIS (hardcoded config)
│   │   ├── test_simple.py        ← TEST THIS FIRST
│   │   ├── generate_chains_v2.py ← Full version with CLI args
│   │   ├── test_generation_v2.py ← Comprehensive tests
│   │   ├── test_gemini_setup.py  ← Auth verification
│   │   ├── generation.py         ← Core generation functions
│   │   ├── schemas.py            ← Pydantic models
│   │   └── prompts/              ← Example-driven prompts
│   │       ├── system_instructions.txt
│   │       ├── query_generation_examples.txt
│   │       └── next_step_examples.txt
│   │
│   ├── uci_har/                  ← Dataset converters
│   ├── pamap2/
│   ├── mhealth/
│   ├── wisdm/
│   ├── actionsense/
│   ├── shared/                   ← Shared utilities
│   ├── setup_all_ts_datasets.py  ← Dataset setup
│   └── verify_conversions.py     ← Data verification
│
├── tools/
│   └── tool_executor.py          ← Tool implementations
│
├── data/                         ← Datasets + generated chains
│   ├── uci_har/
│   ├── pamap2/
│   ├── ...
│   └── tool_chain_of_thoughts/   ← Generated chains (output)
│
└── Documentation:
    ├── START_HERE.md             ← This file
    ├── RUN_THIS.md               ← Quick instructions
    ├── SETUP_COMPLETE.md         ← Full setup guide
    ├── QUICKSTART_V2.md          ← Detailed walkthrough
    ├── ARCHITECTURE.md           ← System design
    ├── DATA_FORMAT.md            ← Dataset structure
    ├── TOOL_CHAIN_FORMAT.md      ← Output schema
    └── docs/GENERATION_V2_GUIDE.md  ← Architecture details
```

## 🎯 Which Script Should I Use?

### For Quick Testing:
**`test_simple.py`**
- Tests all functions
- Hardcoded config
- No command-line args

### For Simple Generation:
**`generate_simple.py`**
- All config at top of file
- No command-line args
- Easy to edit and run
- Best for: Quick testing, iteration, learning

### For Production/Batch:
**`generate_chains_v2.py`**
- Full CLI with argparse
- More options and flexibility
- Best for: Batch generation, automation, scripts

## 📝 Configuration

Edit at the top of `datascripts/generate_tool_chains/generate_simple.py`:

```python
# Line 24 - CHANGE THIS!
PROJECT_ID = "your-project-id"

# Other settings
LOCATION = "us-central1"
MODEL = "gemini-2.5-flash"
DATASET = "uci_har"              # Options: uci_har, pamap2, mhealth, wisdm
NUM_SAMPLES = 5
QUERY_TEMP = 0.9                 # Higher = more diverse queries
STEP_TEMP = 0.7                  # Balanced reasoning
INTERACTIVE = True               # Set False for batch mode
```

## 🔧 Authentication Setup

```bash
# One-time setup
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable aiplatform.googleapis.com

# Install dependencies
pip install google-genai pydantic numpy pandas matplotlib pyarrow
```

## 📤 Output

Chains are saved to: `data/tool_chain_of_thoughts/`

Each file is named: `{dataset}_{session}_v2_{random}.json`

Example structure:
```json
{
  "conversation_id": "uci_har_session_001_v2_042",
  "schema_version": "2.0",
  "user_query": "Using IMU sensors, classify this",
  "conversation": [
    {
      "turn": 1,
      "reasoning": "I need to check available channels...",
      "tool_call": {"tool_name": "show_channel_stats", ...},
      "tool_result": {...}
    }
  ],
  "final_classification": "walking",
  "confidence": "high",
  "is_correct": true
}
```

## 💰 Cost

With `gemini-2.5-flash`:
- ~$0.0003 per chain
- 100 chains = ~$0.03
- 1000 chains = ~$0.30

## ❓ Common Issues

### "Dataset not found"
```bash
# Check dataset exists
ls data/uci_har/manifest.json
```

### "Authentication error"
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### "Import errors"
All scripts automatically add project root to `sys.path`, should work from anywhere.

### "No classification reached"
Normal - chain exceeded max turns without classifying. Will be skipped.

## 📚 More Documentation

- **`datascripts/generate_tool_chains/README.md`** - Detailed folder docs
- **`RUN_THIS.md`** - Quick start guide
- **`SETUP_COMPLETE.md`** - Complete setup walkthrough
- **`QUICKSTART_V2.md`** - Full feature guide
- **`docs/GENERATION_V2_GUIDE.md`** - Architecture and design

## ✅ Next Steps

1. ✓ Test the system: `python datascripts/generate_tool_chains/test_simple.py`
2. ✓ Generate a few chains interactively
3. ✓ Review quality (check JSON output)
4. → Generate 100+ chains across datasets
5. → Filter for high quality (`is_correct == true`, `confidence == "high"`)
6. → Build motion_tokenizer for Phase 2
7. → Use for LLM fine-tuning
