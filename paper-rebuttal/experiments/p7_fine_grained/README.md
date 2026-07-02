# EXP-P7 — Fine-grained ablations

**Branch:** `rebuttal/experiment/fine-grained-ablations`
**Addresses:** Reviewer **E4** ("ablations not fine-grained enough — adaptive pooling, spectral-temporal
tokenization, soft targets, memory queues, synonym augmentation, SBERT; adaptive-pool vs fixed
resampling, temporal-only vs spectral-temporal, channel-independent vs channel-specific"), **C2**, **D3**.
**Type:** TRAIN ×~8 on RunPod. Most variants are **pure config** via `TSFM_CONFIG_OVERRIDES`/env (Phase 0).

## What it tests
Isolates the contribution of each design choice by changing **one knob at a fixed model size**
(`small_deep`), so the delta vs the headline is attributable to that knob alone (not confounded by
size/depth, which is what naively switching `TSFM_MODEL_SIZE` would do).

## Variants (each one run, vs the headline baseline = P6 `headline` seed 42)
| variant | knob | env (set by the runner) |
|---|---|---|
| `temporal_only` | spectral-temporal → temporal-only tokenizer | `TSFM_CONFIG_OVERRIDES={"feature_extractor_type":"cnn"}` |
| `channel_indep` | cross-channel fusion → channel-independent | `{"use_cross_channel":false}` |
| `cnn_multi` | kernel `[5]` → `[3,5,7]` | `{"cnn_kernel_sizes":[3,5,7]}` |
| `spectral_half` | spectral ratio 0.25 → 0.5 | `{"spectral_ratio":0.5}` |
| `hard_targets` | soft targets → hard targets | `ABLATION_SOFT_TARGETS=0` |
| `tau_0p3` | soft-target temperature τ_s 0.5 → 0.3 | `TSFM_SOFT_TARGET_TEMP=0.3` |
| `soft_weight_0p5` | soft/hard blend 1.0 → 0.5 | `TSFM_SOFT_TARGET_WEIGHT=0.5` |
| `sbert_mpnet` | MiniLM-384 → MPNet-768 text encoder | `{"contrastive_text_model":"all-mpnet-base-v2","contrastive_text_dim":768,"semantic_dim":768}` |

**SBERT-swap footgun (handled):** the variant overrides `contrastive_text_model` (the LIVE encoder),
NOT `sentence_bert_model` (a dead path), and bumps `semantic_dim` to 768 in the same patch because the
label-bank pooling has no projection (`semantic_dim` must equal the SBERT dim). MPNet is ~109M params —
give this pod a bit more VRAM headroom (still fits 24 GB).

## Descoped (would need new code; not run unless the rebuttal demands them)
- **additive vs gated conditioning** — needs a `conditioning` flag in `ChannelTextFusion`
  (`token_text_encoder.py`); the active mechanism is the gated fusion, NOT the dead
  `use_channel_encoding` path. Moderate.
- **adaptive-pool vs fixed-resample** — needs threading `interpolation_method`/`tokenization_mode`
  through the encoder build (currently dropped) + `preprocessing.py`. Hard.
- `normalization_method` / init-scale overrides are **not exposed** (the encoder build ignores them,
  so a config override would be recorded-but-not-applied — a provenance trap). Left out deliberately.

## How to run / interpret
```bash
bash scripts/runpod_experiment.sh --exp P7 --variant temporal_only          # one pod per variant
# compare each variant's val_accuracy/unseen to the headline; promote meaningful ones to 3 seeds (P6).
```
Every variant's effective config is recorded in `hyperparameters.json` (verify the knob actually
changed there — that's the smoke test below). ~3 h/run, ~$0.75 each; ~8 runs ≈ $6.

## Smoke test (run locally before paying for pods)
For each config variant: `TSFM_EPOCHS=1 TSFM_CONFIG_OVERRIDES=... python ...semantic_alignment_train.py`,
then confirm `hyperparameters.json.config.<knob>` reflects the override and the module class changed
(e.g. `feature_extractor_type=="cnn"` → `FixedPatchCNN`). See `smoke_results.md`.
