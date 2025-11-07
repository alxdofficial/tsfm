# Quickstart: V2 Generation System

Generate high-quality classification training data using the two-function approach.

## What's Different in V2?

**V1 (Old):**
- Hardcoded query templates
- Regex parsing of Gemini responses
- ~10% parsing errors
- Manual language variation

**V2 (New):**
- ✅ Gemini generates diverse queries
- ✅ Structured outputs (Pydantic schemas)
- ✅ 0% parsing errors
- ✅ Examples-driven for quality
- ✅ Easy to extend with new tools

## Quick Start

### 1. Setup (same as before)

```bash
pip install --break-system-packages google-genai
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable aiplatform.googleapis.com
```

### 2. Test V2 Functions

```bash
python datascripts/test_generation_v2.py --project YOUR_PROJECT_ID
```

Expected output:
```
TEST 1: Query Generation
Query 1: "What activity is happening in this session?"
Query 2: "Using only accelerometer data, classify this"
Query 3: "Is the person walking or running?"

TEST 2: Next Step Generation
Action: use_tool
Reasoning: I need to understand what sensors are available...
Tool: show_channel_stats

TEST 3: Full Chain Generation
✓ Classification: walking
✓ Correct: True
```

### 3. Generate Chains

```bash
# Automated generation
python datascripts/generate_chains_v2.py \
    --project YOUR_PROJECT_ID \
    --dataset uci_har \
    --num-samples 10 \
    --verbose

# Interactive mode with manual approval at each step
python datascripts/generate_chains_v2.py \
    --project YOUR_PROJECT_ID \
    --dataset uci_har \
    --num-samples 5 \
    --verbose \
    --interactive
```

Interactive mode allows you to:
- Review and approve/reject/regenerate each query
- Review and approve/reject/regenerate/skip each tool decision
- Review the final classification before saving
- Manually approve each chain before it's saved to disk

## How It Works

### Function 1: Query Generation

```python
# Gemini generates: "Using IMU sensors, what activity is this?"
query = generate_query(client, manifest, temperature=0.9)
```

**Input:** Dataset manifest
**Output:** Diverse, realistic user query
**Examples:** 30+ query patterns in prompt

### Function 2: Next Step Generation

```python
# Gemini decides: Use show_channel_stats
decision = generate_next_step(
    client, dataset, session, query, history, temperature=0.7
)
```

**Input:** Conversation state
**Output:** Structured decision (tool use OR classification)
**Examples:** 7+ reasoning patterns in prompt

### Full Chain

```python
chain = generate_classification_chain(
    client, project, location, dataset, session, ground_truth
)
```

1. Generates query ← Function 1
2. Loop:
   - Generate next step ← Function 2
   - If tool → execute and continue
   - If classify → done
3. Save complete chain

## Output Structure (V2)

```json
{
  "conversation_id": "uci_har_session_001_v2_042",
  "schema_version": "2.0",
  "user_query": "Based on IMU data, classify this activity",
  "conversation": [
    {
      "turn": 1,
      "reasoning": "I need to see what IMU channels are available...",
      "tool_call": {
        "tool_name": "show_channel_stats",
        "parameters": {"dataset_name": "uci_har"}
      },
      "tool_result": {...}
    },
    {
      "turn": 2,
      "reasoning": "I'll select the 6 IMU channels (acc + gyro)...",
      "tool_call": {
        "tool_name": "select_channels",
        "parameters": {
          "dataset_name": "uci_har",
          "channel_names": ["body_acc_x", "body_acc_y", ...]
        }
      },
      "tool_result": {...}
    }
  ],
  "final_classification": "walking",
  "confidence": "high",
  "explanation": "Based on periodic acceleration at ~1.8 Hz...",
  "is_correct": true
}
```

## Batch Generation

Generate 100 diverse chains (~$0.03):

```bash
for dataset in uci_har pamap2 mhealth wisdm; do
    python datascripts/generate_chains_v2.py \
        --project YOUR_PROJECT_ID \
        --dataset $dataset \
        --num-samples 25 \
        --model gemini-1.5-flash \
        --query-temp 0.9 \
        --step-temp 0.7
done
```

## Query Diversity Examples

V2 generates varied queries:

- "What activity is happening in this session?"
- "Using only accelerometer data, what is this?"
- "Is the person walking or running?"
- "Based on the IMU sensors, classify this activity"
- "What does the motion signature indicate?"
- "Using phone and watch sensors, what's happening?"

## Reasoning Quality Examples

**V2 produces detailed reasoning:**

```
"The user specifically requested IMU data, which includes
accelerometer and gyroscope. Looking at the available channels,
I can see body_acc_x/y/z and body_gyro_x/y/z are the IMU sensors.
I'll select these 6 channels to focus on what the user asked for."
```

**Not this:**
```
"I'll select channels."
```

## Key Files

```
prompts/
├── system_instructions.txt          # Tell Gemini it's generating training data
├── query_generation_examples.txt    # 30+ query patterns
└── next_step_examples.txt           # 7+ reasoning patterns

tools/
├── schemas.py                       # Pydantic models (structured outputs)
└── generation.py                    # Two core functions

datascripts/
├── generate_chains_v2.py           # Main generation script
└── test_generation_v2.py           # Test individual functions
```

## Extending to Phase 2

When you add `motion_tokenizer`:

1. **Update system instructions:**
   Add "motion_tokenizer: Encode IMU data" to tools list

2. **Add examples:**
   Add reasoning example for when to use tokenizer

3. **Update schema:**
   ```python
   tool_name: Literal["...", "motion_tokenizer"]
   ```

4. **Extend existing chains:**
   Load V2 chains, continue with new tools available

**That's it!** The examples-driven approach makes it easy to scale.

## Troubleshooting

**Queries too similar:**
- Increase `--query-temp` to 0.95

**Reasoning too verbose:**
- Lower `--step-temp` to 0.6

**Wrong tool choices:**
- Add/improve examples in next_step_examples.txt

**Parse errors:**
- Should never happen with V2 (structured outputs!)

## Next Steps

1. ✅ Generate 100 chains across all datasets
2. ✅ Review quality (check 10-20 samples)
3. ✅ Filter: keep `is_correct == true` and `confidence == "high"`
4. ✅ Build motion_tokenizer
5. ✅ Extend chains to Phase 2 with tokenizer
6. ✅ Use for LLM fine-tuning

## Documentation

- Full guide: `docs/GENERATION_V2_GUIDE.md`
- Architecture: `ARCHITECTURE.md`
- Data format: `TOOL_CHAIN_FORMAT.md`
