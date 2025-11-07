# Architecture: Tool-Use Time Series Foundation Model (tool-use-om)

## Vision: Option 2 - Frozen Tokenizer + Lightweight Task Heads + LLM Orchestration

This project implements a **compositional foundation model system** for time series analysis, combining:

1. **Pretrained frozen tokenizers** (patch encoders)
2. **Lightweight task-specific heads** (trained in latent space)
3. **LLM orchestration layer** (tool-use reasoning for model/head selection)

This architecture (referred to as "Option 2" from design discussions) provides the best balance of:
- ✅ Zero/few-shot capability via frozen representations
- ✅ Task flexibility via interchangeable heads
- ✅ Dataset generality via standardized preprocessing
- ✅ Interpretability via LLM-driven reasoning

---

## Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: LLM Orchestration (Tool-Use Reasoning)            │
│  - Analyzes user query + dataset metadata                   │
│  - Selects tokenizer, configures parameters                 │
│  - Chooses task head, interprets results                    │
│  - Generates explanations and insights                      │
└────────────────────┬────────────────────────────────────────┘
                     │ Tool Calls
┌────────────────────▼────────────────────────────────────────┐
│  Layer 2: Task Heads (Lightweight, Interchangeable)         │
│  - Classification: prototypes, linear probe, kNN            │
│  - Forecasting: latent state-space + channel decoder        │
│  - QA/Retrieval: similarity search, motif finding           │
│  - All operate in frozen token space                        │
└────────────────────┬────────────────────────────────────────┘
                     │ Tokens (B, D, P)
┌────────────────────▼────────────────────────────────────────┐
│  Layer 1: Frozen Tokenizer (Pretrained Patch Encoder)       │
│  - Converts raw time series → latent tokens                 │
│  - Input: (B, C, T) time series + context features          │
│  - Output: (B, D, P) tokens where P = num patches           │
│  - Frozen after pretraining (no task-specific fine-tuning)  │
└────────────────────┬────────────────────────────────────────┘
                     │ Raw Data
┌────────────────────▼────────────────────────────────────────┐
│  Layer 0: Standardized Data Format                          │
│  - Session-based parquet files                              │
│  - Real-world time (seconds), never timesteps               │
│  - Minimal token-efficient manifests                        │
│  - Dataset-agnostic structure                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Two-Phase Implementation

We're building this system **incrementally in two phases** to validate the architecture and generate training data progressively.

### Phase 1: EDA Tool-Use Reasoning (Current)

**Goal:** Teach the LLM to reason about time series data exploration and preprocessing.

**Tools:**
- `show_session_stats` - Understand dataset size and temporal characteristics
- `show_channel_stats` - Analyze signal distributions and ranges
- `select_channels` - Filter to relevant modalities
- `filter_by_time` - Extract temporal windows

**Training Data:** Generated using large models (Gemini) with real dataset contexts.

**Output:** Fine-tuned Llama 3B that can:
- Analyze dataset metadata
- Reason about temporal structure
- Select appropriate preprocessing steps
- Explain decisions in natural language

**Status:** ✅ Infrastructure complete, ⏳ data generation in progress

---

### Phase 2: Model & Head Selection Reasoning (Future)

**Goal:** Teach the LLM to select and configure tokenizers and task heads.

**Tools:**
- `select_tokenizer` - Choose pretrained encoder based on data characteristics
- `configure_tokenizer` - Set patch size, stride, overlap, context features
- `select_channels_for_encoding` - Choose channel subset for tokenization
- `select_task_head` - Pick classification/forecasting/QA head
- `configure_head` - Set head-specific parameters
- `interpret_results` - Generate explanations from head outputs

**Decisions the LLM Makes:**

1. **Tokenizer Selection:**
   - Which pretrained model? (Model-A: good for short signals, Model-B: multi-rate, etc.)
   - Patch window size? (Balance between temporal resolution and context)
   - Stride/overlap? (Dense tokens vs. computational efficiency)
   - Which channels to encode? (Focus on task-relevant modalities)

2. **Task Head Selection:**
   - Classification: prototypes (few-shot), linear probe (more data), kNN (interpretable)
   - Forecasting: latent dynamics + optional channel decoder
   - QA/Retrieval: similarity search for exploratory analysis

3. **Calibration & Adaptation:**
   - Should we train the head? (yes if >20 labels, no for zero-shot)
   - Escalate to raw-space head? (if token-space performance below threshold)
   - Which metrics to use? (accuracy, AUROC, CRPS, etc.)

**Training Data:** Generated using large models with:
- Real dataset manifests
- Simulated tokenizer specs (Model-A, B, C with known properties)
- Task descriptions and constraints
- Ground truth performance (synthetic or from pilot experiments)

**Status:** 📋 Designed, not yet implemented

---

## Key Design Principles

### 1. **Frozen Tokenizer = Foundation**

Once pretrained, the tokenizer is **never fine-tuned** for specific tasks:
- Preserves zero-shot transfer capability
- Allows task heads to be trained independently
- Enables rapid experimentation with new heads
- Only exception: optional LoRA for extreme domain shift

### 2. **Lightweight Task Heads**

Heads are deliberately simple (linear layers, prototypes, small transformers):
- Fast to train (<5 minutes on CPU)
- Easy to interpret
- Low risk of overfitting
- Can be trained on as few as 5-20 examples

### 3. **Query-Conditional Decoding**

For forecasting, we **don't decode all channels**:
- LLM specifies which channels user cares about
- Train channel-specific decoders only for those
- Predict latent dynamics, decode on demand
- If user only needs scalars (e.g., "next heart rate"), skip full decoder

### 4. **Interpretability First**

Every prediction comes with explanations:
- Classification: top-k prototypes, salient channels, confidence
- Forecasting: trend summary, anomaly flags, uncertainty bounds
- QA: nearest neighbors, motif matches, distance metrics
- LLM narrates findings in natural language

### 5. **Real-World Time Reasoning**

All temporal reasoning uses **seconds**, never timesteps:
- Eliminates sampling rate confusion
- Makes multi-rate data handling straightforward
- Aligns with how humans think about time
- Simplifies cross-dataset comparisons

---

## Why Option 2? (vs. Alternatives)

### Option 1: Task-Specific Models on Raw Data
**Problem:** Combinatorial explosion of preprocessing, brittle pipelines, undermines foundation story.
**When to use:** Only as fallback when token-space performance is insufficient (auto-gated).

### Option 3: LLM Operates Directly on Tokens
**Problem:** Weak quantitative outputs, no calibrated predictions, limited to exploratory QA.
**When to use:** Keep as the EDA layer on top of Option 2 for open-ended analysis.

### Option 2: Frozen Tokenizer + Task Heads
**Advantages:**
- Modular: swap heads without retraining tokenizer
- Fast: train heads in minutes, not hours
- Zero-shot friendly: pretrained representations transfer
- Interpretable: heads are simple enough to explain
- Practical: fallback to Option 1 when needed, promote Option 3 for EDA

---

## Data Flow Example

```python
# User query
"Find walking activities and predict the next 5 seconds of accelerometer data"

# Layer 3: LLM orchestration
1. Analyzes query → identifies: classification + forecasting tasks
2. Calls show_session_stats → learns dataset has 545 sessions, 33s avg
3. Calls select_tokenizer(task="classification+forecasting", modalities=["accel"])
   → Chooses Model-B (good for forecasting) with 2s patches, 0.5s stride
4. Calls select_channels_for_encoding(["accel_x", "accel_y", "accel_z"])
5. Calls select_task_head(task="classification", method="prototypes")
   → Uses few-shot prototypes with 5 labeled walking examples

# Layer 2: Task head execution
6. Classification head: Identifies 45 walking segments (confidence: 0.92)
7. LLM calls select_task_head(task="forecasting", horizon=5.0)
   → Uses latent dynamics model + accel-specific decoder
8. Forecasting head: Predicts next 5s latent states, decodes to accel channels

# Layer 3: LLM interprets & explains
9. "Found 45 walking episodes (92% confidence). Forecasted next 5 seconds:
    accelerometer shows typical walking gait pattern with 1.2 Hz stride
    frequency. Prediction uncertainty: ±0.3 m/s². Salient channels: accel_y
    (vertical motion) explains 68% of variance."
```

---

## Implementation Roadmap

### ✅ Phase 0: Infrastructure (Complete)
- [x] Standardized data format (parquet + manifest + labels)
- [x] 5 dataset converters with debug visualizations
- [x] Real-world time semantics throughout
- [x] Dataset-agnostic loading and statistics

### ⏳ Phase 1: EDA Tool-Use (In Progress)
- [x] EDA tool schemas (4 tools)
- [x] Real tool executors (load parquet, compute stats)
- [x] Task templates for data generation (20 templates)
- [ ] Generate training examples with Gemini
- [ ] Fine-tune Llama 3B on tool-use
- [ ] Validate on held-out datasets

### 📋 Phase 2: Model Selection (Planned)
- [ ] Design tokenizer specs (3-5 hypothetical pretrained models)
- [ ] Define task head zoo (classification, forecasting, QA)
- [ ] Create Phase 2 tool schemas
- [ ] Generate model selection training data
- [ ] Fine-tune on top of Phase 1 model
- [ ] Validate with synthetic tokenizer outputs

### 🚀 Phase 3: Real Foundation Model (Future)
- [ ] Implement/adapt actual tokenizer (e.g., based on MOMENT)
- [ ] Pretrain on multi-dataset corpus
- [ ] Implement task heads (prototypes, linear, forecaster)
- [ ] Connect LLM orchestrator to real heads
- [ ] End-to-end benchmarking

---

## Success Metrics

### Phase 1 (EDA Tool-Use)
- **Accuracy:** LLM selects correct tools >90% of time on held-out queries
- **Reasoning:** Generated explanations are factually correct
- **Generalization:** Works on datasets not seen during training

### Phase 2 (Model Selection)
- **Appropriate Tokenizer:** Selects model with best inductive bias for task
- **Good Hyperparameters:** Patch size/stride within 20% of oracle choices
- **Head Selection:** Picks prototypes for few-shot, linear for more data, etc.

### Phase 3 (Real Foundation)
- **Zero-shot Classification:** >80% accuracy on unseen activity types
- **Few-shot Adaptation:** Match/exceed specialized models with 5-20 examples
- **Forecasting:** CRPS within 10% of oracle on held-out datasets
- **Interpretability:** Explanations validated by domain experts

---

## References

- **Option 2 Design Discussion:** See `docs/option2_rationale.md` for detailed analysis
- **Phase 2 Tool Specifications:** See `tools/PHASE2_DESIGN.md`
- **Training Data Generation:** See `docs/training_data_generation.md`
- **Dataset Pipeline:** See `datascripts/README.md`

---

## Questions & Future Work

**Q: Why not fine-tune the tokenizer for each task?**
A: Breaks zero-shot capability, increases compute, and recent evidence suggests frozen representations + lightweight heads work well (see CLIP, SimCLR literature).

**Q: What if token-space performance is bad?**
A: Auto-gate to Option 1: train slightly larger raw-space head while keeping tokenizer frozen. Still maintains foundation story.

**Q: How do you handle multi-rate data?**
A: Tokenizer can interpolate/resample to common rate, or process each modality separately and fuse tokens. LLM decides based on task.

**Q: What about very long sequences (hours)?**
A: Hierarchical tokenization: first-level patches → second-level "super-patches." Or sliding window with overlap-based fusion.

**Future Extensions:**
- Hierarchical tokenization for long sequences
- Multi-modal fusion in token space (align different sensors)
- Active learning: LLM requests labels for most informative samples
- Causal reasoning: "what would happen if we changed X?"
