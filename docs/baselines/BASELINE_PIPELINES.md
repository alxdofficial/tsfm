# Baseline Pipelines: From Data to Metrics

What each model needs to train and evaluate, and why some break the simple
"preprocess → train → eval" mental model.

---

## The Simple Mental Model (and Where It Breaks)

The intuitive pipeline for a foundation model benchmark is:

```
1. Preprocess data  →  2. Train/download model  →  3. Freeze model  →  4. Evaluate
```

**3 out of 5 models follow this exactly**: LiMU-BERT, MOMENT, CrossHAR.
They have pretrained checkpoints, we freeze them, extract embeddings, and evaluate.

**2 models break the pattern**: LanHAR and TSFM (partially).

- **LanHAR** has no pretrained checkpoint at all. Its method IS training a sensor-text
  alignment from scratch. The "eval script" contains a full 60-epoch training run.
- **TSFM** has a pretrained encoder, but the semantic alignment head was trained
  separately. Both are loaded from checkpoint for evaluation.

---

## Model-by-Model Pipelines

### 1. LiMU-BERT — Simplest Pipeline

```
[Already done]  Pretrain encoder on 10 HAR datasets (masked reconstruction)
[Already done]  Extract embeddings for all 14 datasets → .npy files
                ↓
[Eval script]   Load pre-extracted embeddings (N, 120, 72)
                Split 80/10/10 → reshape to (M, 20, 72) sub-windows
                Train GRU classifier on frozen embeddings
                Report 1%, 10%, linear probe metrics
```

**What's pretrained**: Encoder (frozen, never touched during eval)
**What trains during eval**: Only the downstream GRU/linear classifier
**Checkpoint**: `auxiliary_repos/LIMU-BERT-Public/saved/.../pretrained_combined.pt`
**Embeddings**: `auxiliary_repos/LIMU-BERT-Public/embed/embed_pretrained_combined_{dataset}_20_120.npy`
**Special**: Sub-windows split from full windows FIRST (prevents data leakage), then filtered
**Zero-shot**: Not possible (not text-aligned)
**Time**: ~5 min per dataset

---

### 2. MOMENT — Download and Evaluate

```
[Already done]  Download MOMENT-1-large from HuggingFace
                ↓
[Eval script]   Load raw data (N, 120, 6)
                Left-pad to (N, 6, 512) with input mask
                Process each channel independently as (N, 1, 512)
                Concatenate per-channel embeddings → (N, 6144)
                Train SVM-RBF classifier (GridSearchCV)
                Report 1%, 10%, linear probe metrics
```

**What's pretrained**: Entire model (downloaded from HuggingFace, never trained by us)
**What trains during eval**: Only the downstream SVM/linear classifier
**Checkpoint**: `AutonLab/MOMENT-1-large` (auto-downloaded)
**Special**: Left-padding to 512 timesteps; per-channel concatenation (6 × 1024 = 6144-dim)
**Zero-shot**: Not possible (not text-aligned)
**Time**: ~15 min per dataset (SVM GridSearchCV is slow)

---

### 3. CrossHAR — Pretrained Encoder + Downstream Classifier

```
[Already done]  Pretrain encoder on 10 HAR datasets (masked + contrastive)
                ↓
[Eval script]   Load raw data (N, 120, 6)
                Apply InstanceNorm per sample
                Extract (N, 120, 72) sequence embeddings via frozen encoder
                Train Transformer_ft classifier on frozen embeddings
                Report 1%, 10%, linear probe metrics
```

**What's pretrained**: Encoder (frozen, never touched during eval)
**What trains during eval**: Only the downstream Transformer_ft/linear classifier
**Checkpoint**: `auxiliary_repos/CrossHAR/saved/.../model_masked_6_1.pt`
**Special**: InstanceNorm preprocessing (matches original paper's `IMUDataset`)
**Zero-shot**: Not possible (not text-aligned)
**Time**: ~8 min per dataset

---

### 4. LanHAR — TRAINS FROM SCRATCH DURING EVALUATION

This is the model that breaks the mental model. There is no "pretrained LanHAR
checkpoint." The evaluation script contains a complete 2-stage training pipeline.

```
[Eval script — Stage 1: 10 epochs]
    Load SciBERT from HuggingFace (general scientific text model)
    Fine-tune SciBERT on 87 activity label descriptions
    Losses: CLIP + cross-entropy + 2× triplet
    Output: Activity-aware text encoder
                ↓
[Eval script — Stage 2: 50 epochs]
    Initialize TimeSeriesTransformer from RANDOM WEIGHTS
    Load sensor data from 10 TRAINING datasets only (no test data)
    Apply gravity alignment to all sensor data
    Train sensor encoder to align with frozen SciBERT via CLIP loss
    Per-sample LLM descriptions used 70% of the time (optional)
    Output: Trained sensor encoder + text projections
                ↓
[Eval script — Downstream evaluation]
    Extract sensor embeddings via trained encoder
    Compute text embeddings for each label via trained SciBERT
    Zero-shot: cosine similarity between sensor and text embeddings
    Supervised: train linear classifier on sensor embeddings
    Report all 5 metrics (zero-shot + supervised)
```

**What's pretrained**: Only SciBERT (generic, from HuggingFace). Everything else
trains from scratch.
**What trains during eval**:
  - Stage 1: SciBERT fine-tuning (10 epochs)
  - Stage 2: Sensor encoder + projection heads (50 epochs from random init)
  - Downstream: Linear classifier
**Checkpoint**: None — training happens every time eval runs
**Special requirements**:
  - **Gravity alignment**: Lowpass filter → estimate gravity → Rodrigues rotation
    to align all sensor data to a gravity-aligned frame. Applied to all data.
  - **Per-sample LLM descriptions** (optional): Original paper uses GPT-4. We use
    local Llama/Qwen via `generate_lanhar_descriptions.py`. If descriptions are
    not available, the model still works using per-class descriptions only.
  - **Source-only training**: Stage 2 trains on the 10 training datasets only.
    The original paper combines source + target domains, but we exclude test data
    to ensure fair comparison with other baselines that never see test data.
**Zero-shot**: Yes (text-aligned after training)
**Time**: ~90 min total (Stage 1: ~5 min, Stage 2: ~60 min, Downstream: ~25 min)

### Why LanHAR trains from scratch

LanHAR's contribution is a METHOD for aligning sensor data with text, not a
pretrained model. The sensor encoder doesn't exist before training — it's created
by aligning with SciBERT via contrastive learning. This is fundamentally different
from LiMU-BERT/CrossHAR (which produce reusable frozen encoders) or MOMENT (which
is a general-purpose model). LanHAR is more like a recipe than a product.

The original paper combines source + target data during training (domain adaptation),
but we train on source only to keep the comparison fair with other baselines that
never see test data.

### LLM descriptions: GPT-4 vs local LLM

The original LanHAR paper uses GPT-4 to generate per-sample text descriptions
from signal processing features (EDA, gyroscope patterns, gait sync). These
descriptions are fed to SciBERT during Stage 2 training.

Our implementation uses a local LLM (Ollama with Qwen2.5:14B or similar) via
`generate_lanhar_descriptions.py`. The descriptions are pre-generated and stored
in `benchmark_data/processed/lanhar_descriptions/{dataset}_descriptions.csv`.

If descriptions are unavailable for a dataset, LanHAR falls back to per-class
template descriptions (e.g., "Activity=walking. Sensor pattern: periodic vertical
oscillation..."). This reduces quality but the model still functions.

---

### 5. TSFM (Our Model) — Pretrained Encoder + Semantic Head

```
[Already done]  Pretrain encoder on 10 HAR datasets (MAE + contrastive)
                   → training_output/imu_pretraining/.../latest.pt
[Already done]  Train semantic alignment head (encoder frozen)
                   → training_output/semantic_alignment/.../best.pt
                ↓
[Eval script]   Load encoder + semantic head + label bank from checkpoint
                For each test dataset:
                    Sweep patch sizes [1.0, 1.25, 1.5, 1.75, 2.0] sec
                    Extract (N, 384) L2-normalized embeddings
                Zero-shot: cosine similarity with 87 label bank embeddings
                Supervised: train linear classifier on sensor embeddings
                Report all 5 metrics
```

**What's pretrained**: Encoder (Stage 1) + Semantic head + Label bank (Stage 2)
**What trains during eval**: Only the downstream linear classifier
**Checkpoint**: `training_output/semantic_alignment/{run_id}/best.pt`
**Special**: Patch size sweep to find optimal temporal resolution per dataset
**Zero-shot**: Yes (text-aligned via learnable label bank, 384-dim)
**Time**: ~10 min per dataset (patch sweep adds overhead)
**NOTE**: Training in progress. Results pending completion.

---

## Summary Comparison

| | LiMU-BERT | MOMENT | CrossHAR | LanHAR | TSFM |
|---|---|---|---|---|---|
| **Pretrained checkpoint** | Yes | Yes (HF) | Yes | No | Yes |
| **Trains during eval** | Classifier only | Classifier only | Classifier only | **Full model** | Classifier only |
| **Training time in eval** | ~5 min | ~15 min | ~8 min | **~90 min** | ~10 min |
| **Embedding dim** | 72 | 6144 | 72 | 768 | 384 |
| **Text-aligned** | No | No | No | Yes | Yes |
| **Zero-shot capable** | No | No | No | Yes | Yes |
| **# metrics reported** | 3 | 3 | 3 | 5 | 5 |
| **Special preprocessing** | Sub-window split | Left-pad 512 | InstanceNorm | Gravity align | Patch sweep |
| **Classifier** | GRU | SVM-RBF | Transformer_ft | Linear | Linear |
| **Needs GPU** | Minimal | Yes (large model) | Minimal | Yes (training) | Yes |

---

## What "Evaluation" Actually Runs

To be completely explicit about what happens when you run each eval script:

| Script | What actually happens |
|--------|---------------------|
| `evaluate_limubert.py` | Loads .npy embeddings → trains GRU → reports metrics |
| `evaluate_moment.py` | Loads MOMENT → extracts embeddings → trains SVM → reports metrics |
| `evaluate_crosshar.py` | Loads encoder → extracts embeddings → trains Transformer_ft → reports metrics |
| `evaluate_lanhar.py` | **Trains SciBERT (10ep) → trains sensor encoder (50ep)** → extracts embeddings → trains linear → reports metrics |
| `evaluate_tsfm.py` | Loads encoder+head → extracts embeddings → trains linear → reports metrics |

The key insight: for LiMU-BERT/MOMENT/CrossHAR/TSFM, "evaluation" means
"load a frozen model, extract embeddings, train a small classifier."
For LanHAR, "evaluation" means "train the entire model from scratch, THEN
evaluate it."

---

## Model Capabilities

| | LiMU-BERT | MOMENT | CrossHAR | LanHAR | TSFM |
|---|---|---|---|---|---|
| **Sees label text?** | No | No | No | Yes (SciBERT) | Yes (label bank) |
| **Unseen labels?** | Cannot classify | Cannot classify | Cannot classify | Nearest known label | Nearest known label |
| **Zero-shot capable?** | No | No | No | Yes | Yes |
| **Foundation model?** | No | **Yes** | No | No | Yes (ours) |

### What "Foundation Model" Means for Each

- **LiMU-BERT** (SenSys 2021): **Self-supervised pretraining for HAR.** BERT-style
  masked reconstruction on IMU data. The paper calls it "representation learning" —
  it learns within-dataset features and requires supervised fine-tuning. Not a
  foundation model: single-domain, no generalization to unseen tasks.

- **MOMENT** (ICML 2024): **True time series foundation model.** Pretrained on "The
  Time Series Pile" — a large diverse corpus spanning healthcare, engineering, finance,
  and more. Handles classification, forecasting, anomaly detection, and imputation
  out-of-the-box. The only genuine foundation model in the comparison, but it is a
  general-purpose time series model, not HAR-specific, and has no text alignment.

- **CrossHAR** (IMWUT 2024): **Cross-dataset self-supervised pretraining for HAR.**
  Hierarchical masked reconstruction + contrastive pretraining. The paper itself
  explicitly states that building "foundation models for HAR" is **future work**,
  acknowledging CrossHAR is not one. Evaluates only 4 coarse activity classes in
  the original paper.

- **LanHAR** (IMWUT 2025): **LLM-guided semantic alignment for cross-dataset HAR.**
  Uses LLM-generated text descriptions to align sensor data with language via CLIP-style
  training. The paper explicitly contrasts its approach against building a foundation
  model, claiming lower cost and flexibility. Trains from scratch each time — there is
  no reusable pretrained encoder.

- **NLS-HAR** (AAAI 2025): **Empirical study of contrastive IMU-text alignment.**
  Investigates why CLIP-style natural language supervision fails for HAR and proposes
  mitigations. Single-dataset training, no code released. Frames foundation models
  for HAR as an aspirational future goal. (Included by citing paper results only.)

- **TSFM** (ours): **Text-aligned IMU foundation model.** Dual-branch Transformer
  encoder pretrained on 10 HAR datasets via MAE + contrastive learning, then aligned
  to text via semantic alignment head. Generalizes to unseen datasets and labels via
  learnable label bank.
