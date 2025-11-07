# Tool-Use Time Series Foundation Model (tool-use-om)

**A compositional foundation model system combining frozen tokenizers, lightweight task heads, and LLM orchestration for dataset-agnostic time series analysis.**

---

## 🎯 What is This?

This project implements **Option 2** from our architecture discussions: a three-layer system where:

1. **Frozen pretrained tokenizers** convert raw time series → latent tokens
2. **Lightweight task heads** operate in token space (classification, forecasting, QA)
3. **LLM orchestration** reasons about which models/heads to use and interprets results

This approach provides:
- ✅ **Zero/few-shot** capability via frozen representations
- ✅ **Task flexibility** via interchangeable heads
- ✅ **Dataset generality** via standardized preprocessing
- ✅ **Interpretability** via LLM-driven reasoning and explanations

📖 **Full architecture details:** See [`ARCHITECTURE.md`](ARCHITECTURE.md)

---

## 🏗️ Two-Phase Implementation

We're building this system incrementally:

### Phase 1: EDA Tool-Use Reasoning (Current)

**Goal:** Teach LLM to reason about time series data exploration and preprocessing.

**Status:** ✅ Infrastructure complete, ⏳ Training data generation in progress

**What's Working:**
- Standardized dataset format (parquet + manifest + labels)
- 5 dataset converters (UCI HAR, PAMAP2, MHEALTH, WISDM, ActionSense)
- 4 EDA tools: session stats, channel stats, channel selection, time filtering
- Debug visualizations for data verification
- Real-world time semantics (seconds, not timesteps)

**Example Query:**
```
User: "How many sessions are longer than 30 seconds? What channels are available?"

LLM Tool Reasoning:
1. show_session_stats("actionsense")
   → 545 sessions, avg 33.1s, min 2.0s, max 178.9s
2. Interprets: ~470 sessions > 30s
3. show_channel_stats("actionsense")
   → 66 joint channels, all at 60Hz
4. Returns: "Approximately 470 sessions exceed 30 seconds. Dataset has 66
   joint motion capture channels sampled at 60Hz, covering full body kinematics."
```

### Phase 2: Model & Head Selection (Planned)

**Goal:** Teach LLM to select and configure tokenizers and task heads.

**Tools to Add:**
- `select_tokenizer` - Choose pretrained encoder
- `configure_tokenizer` - Set patch size, stride, channel subset
- `select_task_head` - Pick classification/forecasting/QA head
- `configure_head` - Set head-specific parameters

**Example Query:**
```
User: "Find walking activities and predict next 5 seconds of accelerometer."

LLM Tool Reasoning:
1. select_tokenizer(task="classification+forecasting", modalities=["accel"])
   → Chooses Model-B (good for forecasting), 2s patches, 0.5s stride
2. select_task_head(task="classification", method="prototypes", labels=5)
   → Few-shot prototype classifier
3. [Executes] → Finds 45 walking segments (92% confidence)
4. select_task_head(task="forecasting", horizon=5.0, channels=["accel_x/y/z"])
   → Latent dynamics + accel decoder
5. [Executes] → Predicts next 5s with ±0.3 m/s² uncertainty
```

📖 **Phase 2 design details:** See [`tools/PHASE2_DESIGN.md`](tools/PHASE2_DESIGN.md)

---

## 🗂️ Repository Structure

```
tsfm/
├── ARCHITECTURE.md              # Full architecture & rationale (READ THIS FIRST)
├── README.md                    # This file
│
├── data/                        # Standardized datasets
│   ├── actionsense/
│   │   ├── manifest.json        # Minimal metadata (channels, sampling rates)
│   │   ├── labels.json          # Session → activity labels
│   │   ├── sessions/            # Parquet files per session
│   │   └── debug_*.png          # Visualization for verification
│   ├── uci_har/
│   ├── pamap2/
│   ├── mhealth/
│   └── wisdm/
│
├── datascripts/                 # Dataset download & conversion
│   ├── README.md                # Dataset pipeline documentation
│   ├── download_all_datasets.py
│   ├── convert_*.py             # Per-dataset converters
│   ├── setup_all_datasets.py   # Master pipeline
│   └── visualization_utils.py  # Debug plotting
│
├── tools/                       # Tool definitions & executors
│   ├── schemas.json             # Phase 1 EDA tool schemas
│   ├── executor.py              # Phase 1 tool implementations
│   ├── __init__.py
│   └── PHASE2_DESIGN.md         # Phase 2 model selection tools (spec)
│
└── docs/                        # Design documents
    ├── option2_rationale.md     # Why Option 2? (vs. alternatives)
    └── training_data_generation.md  # How we generate examples
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install pandas pyarrow matplotlib numpy

# Optional: for data generation
pip install google-generativeai
```

### 2. Process Datasets

```bash
# Download and convert all datasets (takes ~15-20 minutes)
python datascripts/setup_all_datasets.py

# Or process one dataset
python datascripts/setup_all_datasets.py uci_har
```

This creates standardized parquet files + manifests + debug plots.

### 3. Test Tools

```bash
# Test EDA tools on ActionSense dataset
python tools/executor.py actionsense
```

### 4. Generate Training Data (WIP)

```bash
# Set up Google Gemini API
export GEMINI_API_KEY='your-key'
# or use: gcloud auth application-default login

# Generate tool-use examples
python datascripts/actionsense/generate_training_data.py
```

---

## 📊 Available Datasets

| Dataset | Sessions | Channels | Rate | Activities | Size |
|---------|----------|----------|------|------------|------|
| **ActionSense** | 545 | 66 (joints) | 60 Hz | 23 kitchen activities | 5.0h |
| **UCI HAR** | 10,299 | 9 (accel+gyro) | 50 Hz | 6 activities | ~7.3h |
| **PAMAP2** | ~200 | 40 (3 IMUs+HR) | 100 Hz | 18 activities | ~10h |
| **MHEALTH** | ~120 | 23 (3 IMUs+ECG) | 50 Hz | 12 activities | ~2h |
| **WISDM** | ~900 | 12 (phone+watch) | 20 Hz | 18 activities | ~76h |

All converted to uniform format: `data/{dataset}/sessions/session_XXX/data.parquet`

---

## 🎓 Design Philosophy

### Why "Option 2"?

We evaluated three approaches:

**Option 1: Task-Specific Models on Raw Data**
- ❌ Brittle preprocessing pipelines
- ❌ Combinatorial explosion per dataset
- ❌ Undermines zero-shot story
- ✅ Best task performance when you have lots of labels

**Option 2: Frozen Tokenizer + Task Heads** (⭐ This repo)
- ✅ Modular: swap heads without retraining encoder
- ✅ Fast: train heads in minutes, not hours
- ✅ Zero-shot friendly: representations transfer
- ✅ Interpretable: simple heads, LLM explanations
- ✅ Practical: fallback to Option 1 when needed

**Option 3: LLM Operates Directly on Tokens**
- ✅ Simplest UX for exploratory QA
- ❌ Weak quantitative outputs
- ❌ No calibrated predictions
- ✅ Great as EDA layer on top of Option 2

📖 **Detailed comparison:** See [`ARCHITECTURE.md`](ARCHITECTURE.md#why-option-2-vs-alternatives)

### Key Principles

1. **Real-World Time:** All reasoning uses seconds, never timesteps
2. **Frozen Foundation:** Tokenizer pretrained once, never fine-tuned per task
3. **Lightweight Heads:** Train in minutes, easy to interpret, low overfitting risk
4. **Query-Conditional:** Only decode channels user cares about (not all)
5. **Interpretable:** Every prediction comes with explanations

---

## 📈 Current Status

### ✅ Completed (Phase 1 Infrastructure)

- [x] Standardized data format design
- [x] 5 dataset converters with debug visualizations
- [x] EDA tool schemas (4 tools)
- [x] Real tool executors (load parquet, compute stats)
- [x] Task templates for data generation (20 templates)
- [x] Real-world time semantics throughout

### ⏳ In Progress (Phase 1 Training)

- [ ] Generate training examples with Gemini
- [ ] Fine-tune Llama 3B on EDA tool-use
- [ ] Validate on held-out datasets and queries

### 📋 Planned (Phase 2)

- [ ] Design tokenizer specs (3-5 hypothetical models)
- [ ] Define task head zoo (classification, forecasting, QA)
- [ ] Create Phase 2 tool schemas
- [ ] Generate model selection training data
- [ ] Fine-tune on top of Phase 1 model

### 🚀 Future (Phase 3)

- [ ] Implement/adapt actual tokenizer (e.g., MOMENT-based)
- [ ] Pretrain on multi-dataset corpus
- [ ] Implement real task heads
- [ ] End-to-end benchmarking

---

## 🔗 Key Documents

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Full system design, data flow, rationale
- **[datascripts/README.md](datascripts/README.md)** - Dataset pipeline documentation
- **[tools/PHASE2_DESIGN.md](tools/PHASE2_DESIGN.md)** - Phase 2 tool specifications
- **[docs/option2_rationale.md](docs/option2_rationale.md)** - Why this architecture?

---

## 🤝 Contributing / Using This Codebase

### For Code Agents (Claude, ChatGPT, etc.)

When working on this repository:

1. **Read [`ARCHITECTURE.md`](ARCHITECTURE.md) first** - understand the vision
2. **We're in Phase 1** - focus on EDA tool-use, not model implementation yet
3. **Real-world time always** - use seconds/Hz, never timesteps
4. **Dataset-agnostic** - code should work with any standardized dataset
5. **Verify with debug plots** - all converters generate visualizations

### For Developers

This is a research project exploring compositional foundation models. The architecture is intentionally modular to allow experimentation with:

- Different tokenizers (MOMENT, Chronos, custom)
- Different task heads (prototypes, linear, transformers)
- Different LLM orchestration strategies
- Different training data generation approaches

Feel free to adapt or extend any component while maintaining the core principles.

---

## 📝 Citation & References

This project builds on ideas from:
- **Foundation models:** CLIP, SimCLR (frozen encoders + task heads)
- **Time series models:** MOMENT, Chronos, TimesFM
- **LLM tool-use:** ReAct, Toolformer, function calling
- **Design discussions:** See `docs/option2_rationale.md` for detailed analysis

---

## 📜 License

[Add your license here]

---

## 🏷️ Branch Info

**Branch:** `tool-use-om` (tool-use omni model)
**Purpose:** Explore compositional tool-using agent architecture
**Status:** Active development (Phase 1)

**Other branches:**
- `master` - Previous work on direct end-to-end training with MOMENT/Chronos
