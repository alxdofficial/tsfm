# HALO Repository Guide

This document tells AI agents (and humans) what this repository contains and how
to navigate it. It reflects the **V2** cleanup (see `docs/v2/REPO_CLEANUP_PLAN.md`);
the pre-cleanup state is preserved at the `v1-archive` git tag.

## Project summary

HALO (aka "TSFM" in code) is a language-aligned IMU foundation model for human
activity recognition. A channel-independent patch encoder produces per-patch
embeddings that are CLIP-style contrastively aligned to frozen-SentenceBERT text
label embeddings, with per-channel natural-language "channel descriptions"
(placement + sampling rate) injected via `ChannelTextFusion`. At inference,
activities are recognized zero-shot by cosine similarity to text label
embeddings — no per-dataset classifier.

The **headline model** is config `small_deep` (d=384, 8 dual-branch layers,
`per_patch_prediction=True`, ~26M active params). It is trained **from scratch**
during alignment (there is no separate self-supervised pretraining stage — the
legacy Stage-1 was removed in V2). Headline checkpoint:
`training_output/semantic_alignment/small_deep_v2_4b3fdd6/best.pt`.

## Directory structure

| Directory | Purpose | Key files |
|---|---|---|
| `model/` | Model architecture | `config.py` (MODEL_SIZE presets), `encoder.py` (dual-branch transformer), `feature_extractor.py` (spectral+temporal tokenizer), `positional_encoding.py`, `semantic_alignment.py` (per-patch head), `token_text_encoder.py` (`ChannelTextFusion`, `LearnableLabelBank`), `preprocessing.py` |
| `training_scripts/human_activity_recognition/` | Training | `semantic_alignment_train.py` (main; defines `SemanticAlignmentModel` + `ChannelBucketBatchSampler`), `semantic_loss.py` (symmetric InfoNCE + soft targets), `memory_bank.py` (MoCo queue) |
| `val_scripts/human_activity_recognition/` | **Evaluation (protocol v2)** | `eval_v2.py` (scoring core: ground truth, subject-disjoint splits, ConSE, metrics), `evaluate_tsfm_v2.py` (HALO evaluator), `eval_common.py` (shared embed/forward helpers), `model_loading.py`, `run_baselines_v2.py` (generic baseline driver), `baselines/` (adapter package), `assemble_v2_table.py`, `plot_utils.py` (training plots) |
| `datasets/imu_pretraining_dataset/` | Dataloader + labels | `multi_dataset_loader.py`, `label_groups.py` (semantic groups for training sampling), `label_augmentation.py`, `augmentations.py` |
| `datascripts/` | Dataset download + conversion | one folder per dataset (`convert.py`); `shared/` utilities; `setup_all_ts_datasets.py` |
| `benchmark_data/` | Eval data + config | `dataset_config.json` (train + zero-shot lists), `scripts/` (preprocessing), `eval_v2/labels/*.json` (pre-registered per-dataset label vocabularies) |
| `docs/` | Documentation | `baselines/EVALUATION_PROTOCOL_V2.md`, `baselines/RESULTS_V2.md`, `v2/` (redesign + cleanup plans), `ARCHITECTURE.md`, `DATA_FORMAT.md` |
| `auxiliary_repos/` | Vendored baseline repos (gitignored) | CrossHAR, LIMU-BERT-Public — rebuilt via `scripts/fetch_baselines.sh` (planned) |
| `tests/` | pytest | `test_eval_v2.py` (protocol), model/loader/loss/aug tests |

## Datasets (V2)

- **Train (11):** uci_har, hhar, pamap2, wisdm, dsads, kuhar, unimib_shar, hapt, mhealth, recgym, capture24.
- **Test (6, held out):** motionsense, realworld, mobiact, shoaib, harth, inclusivehar.
- **Appendix/retained conversion:** opportunity. Dropped from the primary benchmark: vtt_coniot + the "severe-OOD" tier; realdisp/daphnet_fog/usc_had/actionsense converters.

## Evaluation protocol v2 (the current protocol)

Single clean rule — see `docs/baselines/EVALUATION_PROTOCOL_V2.md`:
- **ZS-XD**: zero-shot vs each dataset's **own** label strings; exact match; **macro-F1 primary**.
- **Subject-disjoint** splits everywhere (few-shot); subject-stratified bootstrap CIs.
- **ConSE bridge** (Norouzi 2014) for closed-vocab baselines; **parity rows** (20 Hz + neutral text) isolate the architecture advantage.
- Ground truth via `eval_v2.window_ground_truth` (offset-free — never the v1 `get_window_labels`).

## Baselines (V2)

- **Integrated (6):** CrossHAR, LiMU-BERT, ssl-wearables/harnet5 (ConSE tier); UniMTS (cosine tier);
  NormWear (l1 tier); DeepConvLSTM (few-shot tier, via `run_fewshot_v2.py`). **Dropped:** MOMENT, LanHAR, LLaSA.
  See `docs/baselines/BASELINES_OVERVIEW.md` for how each works, param counts, and heterogeneity/open-set gotchas.
- Each baseline is a small adapter in `val_scripts/human_activity_recognition/baselines/` (`ConSEAdapter`,
  `CosineAdapter`, or the `l1`/`fewshot` tiers + `@register`). Adding one = drop a module; the generic
  `run_baselines_v2.py` driver picks it up automatically.
- Dataset catalog (metadata, plots, caveats): `docs/DATASOURCES.md`. Augmentations: `docs/AUGMENTATIONS.md`.

## Common tasks

- **Train:** `python training_scripts/human_activity_recognition/semantic_alignment_train.py` (env: `TSFM_*`, `MODEL_SIZE`).
- **Evaluate HALO:** `TSFM_CHECKPOINT=training_output/semantic_alignment/small_deep_v2_4b3fdd6/best.pt python val_scripts/human_activity_recognition/evaluate_tsfm_v2.py` (add `--zs-only`, `--channel-text neutral --eval-rate 20` for the parity row).
- **Run baselines:** `python val_scripts/human_activity_recognition/run_baselines_v2.py [--baselines crosshar limubert]`.
- **Assemble the results table:** `python val_scripts/human_activity_recognition/assemble_v2_table.py`.
- **Add a baseline:** create `baselines/<name>.py` subclassing `ConSEAdapter`/`CosineAdapter`, `@register` it, import it in `baselines/__init__.py`.
- **Tests:** `pytest tests/ -q`.

## Key conventions

1. Model dims come from `model/config.py` (`MODEL_SIZE` + `TSFM_*` env overrides) — never hardcode.
2. Checkpoint dirs contain `hyperparameters.json` beside `best.pt`; `model_loading.load_model` tolerates benign missing/unexpected keys (legacy channel-encoding + the gated session-level head for per-patch models).
3. Authoritative v2 numbers: `docs/baselines/RESULTS_V2.md`; results JSONs under `test_output/eval_v2/`.
4. `training_output/`, `test_output/{ablation,baseline}_evaluation/`, `auxiliary_repos/`, `data/**` are gitignored — do not commit checkpoints/large artifacts.

## Environment

- Python 3.11 in `.venv/`. GPU: RTX 4090 24GB locally.
