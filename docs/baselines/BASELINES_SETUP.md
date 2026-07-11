# Baseline Setup Guide (Protocol v2)

> ℹ️ Setup steps are largely current, but the baseline SET has grown since — see
> [`BASELINES_OVERVIEW.md`](BASELINES_OVERVIEW.md) for the full 6-baseline roster (adds
> ssl-wearables, UniMTS, NormWear, DeepConvLSTM) and `cloud/recipes.json` for per-baseline
> repos/weights/deps.

This guide covers the active V2 baseline path. The current primary baseline set is
CrossHAR and LiMU-BERT through the shared ConSE adapter driver. MOMENT, LanHAR,
and LLaSA are not part of the active V2 table.

## Required Repositories

```bash
mkdir -p auxiliary_repos

# CrossHAR
git clone https://github.com/kingdomrush2/CrossHAR.git auxiliary_repos/CrossHAR
cd auxiliary_repos/CrossHAR && git checkout 77b63d3 && cd ../..

# LiMU-BERT
git clone https://github.com/dapowan/LIMU-BERT-Public.git auxiliary_repos/LIMU-BERT-Public
cd auxiliary_repos/LIMU-BERT-Public && git checkout decffee && cd ../..
```

The checked-in adapter code expects these repos under `auxiliary_repos/`.

## Benchmark Data

Prepare the standardized sessions and evaluation arrays:

```bash
python datascripts/setup_all_ts_datasets.py
python benchmark_data/scripts/export_raw.py
python benchmark_data/scripts/preprocess_limubert.py
python benchmark_data/scripts/preprocess_tsfm_eval.py
python benchmark_data/scripts/generate_eval_v2_labels.py
```

Active V2 zero-shot datasets come from `benchmark_data/dataset_config.json`:
`motionsense`, `realworld`, `mobiact`, `shoaib`, `harth`, `inclusivehar`.

## Cached Baseline Heads

CrossHAR and LiMU-BERT are closed-vocabulary classifiers. Their cached heads and
`benchmark_data/processed/limubert/global_label_mapping.json` must be generated
together from the same training vocabulary. If Capture24 is included in the
baseline train set, rebuild both heads and the mapping; do not reuse the older
87-way heads.

Expected cache locations:

```text
test_output/baseline_evaluation/crosshar_zs_transformer.pt
test_output/baseline_evaluation/limubert_zs_gru.pt
benchmark_data/processed/limubert/global_label_mapping.json
```

The V2 driver now checks classifier output width against the mapping and checks
per-dataset `mapping.json` files against `eval_v2/labels/*.json` before scoring.

## Run Evaluations

```bash
# HALO native row
TSFM_CHECKPOINT=training_output/semantic_alignment/small_deep_v2_4b3fdd6/best.pt \
python val_scripts/human_activity_recognition/evaluate_tsfm_v2.py --zs-only

# HALO parity row
python val_scripts/human_activity_recognition/evaluate_tsfm_v2.py \
  --zs-only --channel-text neutral --eval-rate 20

# ConSE baselines
python val_scripts/human_activity_recognition/run_baselines_v2.py --baselines crosshar limubert

# Table
python val_scripts/human_activity_recognition/assemble_v2_table.py
```

Outputs are written to `test_output/eval_v2/*.json`.

## What Reviewers Should Inspect

- `docs/baselines/EVALUATION_PROTOCOL_V2.md`
- `docs/baselines/RESULTS_V2.md`
- `val_scripts/human_activity_recognition/run_baselines_v2.py`
- `val_scripts/human_activity_recognition/baselines/`
- `benchmark_data/scripts/preprocess_limubert.py`
- `benchmark_data/scripts/generate_eval_v2_labels.py`
