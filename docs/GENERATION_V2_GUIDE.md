## Classification Chain Generation V2

The improved two-function approach for generating high-quality training data.

## Architecture

### Two Core Functions

**1. `generate_query(client, manifest)` → user_query**
- Takes: Dataset manifest
- Returns: A realistic user query
- Uses: Structured output (Pydantic schema)
- Temperature: 0.9 (high for diversity)
- Examples: Rich prompt with 30+ query patterns

**2. `generate_next_step(client, dataset, session, query, history)` → decision**
- Takes: Current conversation state
- Returns: Either tool use OR final classification
- Uses: Structured output (Pydantic schema)
- Temperature: 0.7 (balanced)
- Examples: 7+ detailed reasoning patterns

### System Instructions

Gemini is told that it's a **training data generation agent** creating examples for fine-tuning. This framing helps it understand:
- Generate plausible, realistic traces
- Vary language for diversity
- Follow good analytical patterns
- Current scope: activity classification from motion sensors

## Key Design Decisions

### 1. Structured Outputs

We use Pydantic models enforced via `response_schema`:

```python
class NextStepDecision(BaseModel):
    action: NextStepAction  # Enum: USE_TOOL or CLASSIFY
    reasoning: str

    # If USE_TOOL:
    tool_name: Optional[Literal["show_channel_stats", "select_channels"]]
    parameters: Optional[Dict[str, Any]]

    # If CLASSIFY:
    classification: Optional[str]
    confidence: Optional[Literal["high", "medium", "low"]]
    explanation: Optional[str]
```

**Why:** Guarantees correct tool use syntax, eliminates parsing errors, enforces structure.

### 2. Examples-Driven Prompting

Both functions use rich example libraries:

**Query Generation:**
- 6 query pattern categories
- 30+ example queries
- Dataset-specific examples (UCI HAR, PAMAP2, MHEALTH, WISDM)
- Language variation guidelines

**Next Step Generation:**
- 7 detailed reasoning examples
- Decision framework (flowchart logic)
- Language variation alternatives
- Domain knowledge demonstrations

**Why:** Gemini learns from examples, producing higher quality and more diverse outputs.

### 3. Temperature Strategy

| Function | Temperature | Rationale |
|----------|-------------|-----------|
| Query generation | 0.9 | High diversity in queries |
| Step generation | 0.7 | Balanced creativity + consistency |

**Why:** Queries should be very diverse (user language varies), but reasoning should be somewhat consistent (analytical patterns).

### 4. Universal Handles

Data and tokens referenced via handles:
- `ds_abc123` - Dataset handle from `select_channels`
- `tok_xyz789` - Token handle from `motion_tokenizer` (future)

**Why:** Prepares for Phase 2 where raw data isn't in prompts.

## Usage

### Test Individual Functions

```bash
# Test query generation
python datascripts/test_generation_v2.py --project YOUR_PROJECT_ID
```

### Generate Chains

```bash
# Generate 10 chains with diverse queries
python datascripts/generate_chains_v2.py \
    --project YOUR_PROJECT_ID \
    --dataset uci_har \
    --num-samples 10 \
    --query-temp 0.9 \
    --step-temp 0.7 \
    --verbose
```

### Batch Generation

```bash
# 25 chains per dataset = 100 total
for dataset in uci_har pamap2 mhealth wisdm; do
    python datascripts/generate_chains_v2.py \
        --project YOUR_PROJECT_ID \
        --dataset $dataset \
        --num-samples 25 \
        --model gemini-1.5-flash
done
```

## Output Format

### Schema Version 2.0

New fields compared to V1:
- `query_temperature` and `step_temperature`
- `generation_method: "two_function_structured_output"`
- More consistent tool call format (guaranteed by Pydantic)

Example:
```json
{
  "conversation_id": "uci_har_session_001_v2_042",
  "schema_version": "2.0",
  "generation_metadata": {
    "gemini_model": "gemini-1.5-flash",
    "query_temperature": 0.9,
    "step_temperature": 0.7,
    "generation_method": "two_function_structured_output"
  },
  "initial_context": {
    "user_query": "Based on IMU sensors, what activity is this?",
    ...
  },
  "conversation": [
    {
      "turn": 1,
      "reasoning": "I need to check what IMU channels are available...",
      "tool_call": {
        "tool_name": "show_channel_stats",
        "parameters": {"dataset_name": "uci_har"}
      },
      "tool_result": {...}
    }
  ],
  "final_classification": "walking",
  "confidence": "high",
  "is_correct": true
}
```

## Query Diversity Examples

The system generates varied queries like:

**Simple:**
- "What activity is happening?"
- "Classify this session"

**Specific activity:**
- "Is the person walking?"
- "Was running captured here?"

**Channel-specific:**
- "Using only accelerometer, what is this?"
- "Based on gyroscope data, classify the activity"

**Technical:**
- "Analyze the motion signature and classify"
- "What does the IMU pattern indicate?"

**Multi-sensor:**
- "Using phone and watch data, what's happening?"
- "Based on multi-modal sensors, classify this"

## Reasoning Quality

Example of good vs. poor reasoning:

**Good:**
```
"The user specifically requested IMU data. Looking at the available
channels from show_channel_stats, I can see body_acc_x/y/z and
body_gyro_x/y/z are the IMU sensors. I'll filter to just these 6
channels to focus on what the user requested."
```

**Poor:**
```
"I'll select some channels."
```

The V2 system produces good reasoning through:
- Example-driven prompting
- Explicit guidelines (reference previous results, explain why)
- Structured outputs (reasoning field is required)

## Extending to Phase 2

When motion_tokenizer is ready:

1. **Add to system instructions:**
   ```
   - motion_tokenizer: Encode IMU data to tokens
   - motion_classifier: Classify from tokens
   ```

2. **Add examples to next_step_examples.txt:**
   ```
   Example 8: Using Motion Tokenizer

   Situation: Selected IMU channels, user wants classification

   Good Reasoning: "I have the relevant IMU channels selected.
   To classify the activity, I'll first tokenize this motion data
   into compact representations using 1-second patches at 25% overlap.
   This will extract gait cycle features suitable for classification."

   Tool: motion_tokenizer
   ```

3. **Update Pydantic schema:**
   ```python
   tool_name: Literal[
       "show_channel_stats",
       "select_channels",
       "motion_tokenizer",
       "motion_classifier"
   ]
   ```

4. **Extend existing chains:**
   Load V2 chains, continue from where they ended with new tools available.

## Advantages Over V1

| Aspect | V1 | V2 |
|--------|----|----|
| Query generation | Manual templates | Gemini-generated, diverse |
| Tool syntax | Regex parsing | Structured output (Pydantic) |
| Reasoning quality | Variable | Consistent (examples-driven) |
| Language variety | Low | High (explicit variation) |
| Extensibility | Manual updates | Add examples + schemas |
| Error rate | ~10% parse failures | ~0% (structured outputs) |
| Scalability | Template maintenance | Example library growth |

## Cost

Same as V1 (~$0.0003 per chain with Flash), but:
- Higher quality outputs
- Less manual filtering needed
- Fewer regeneration attempts (no parse errors)

**Net effect: Lower cost per usable chain**

## Best Practices

1. **Start with high diversity:**
   - `query_temp=0.9` for initial generation
   - `step_temp=0.7` for balanced reasoning

2. **Review examples:**
   - Check first 10-20 chains
   - Adjust examples if patterns are off
   - Add new examples as you discover good patterns

3. **Filter by quality:**
   - Keep `is_correct == true`
   - Prefer `confidence == "high"`
   - Check reasoning length (>50 words is good)

4. **Grow examples library:**
   - Save exceptional chains
   - Extract their reasoning patterns
   - Add as examples to prompts

5. **Version your prompts:**
   - Track changes to system_instructions.txt
   - Track changes to example files
   - Record which prompts generated which chains
