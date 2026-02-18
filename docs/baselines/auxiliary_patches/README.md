# Baseline Repository Patches

Patches applied to baseline repositories to support our 14-dataset benchmark.
The original repos only support 4 datasets with 4-class collapsed labels.

## How to Apply

```bash
cd auxiliary_repos/CrossHAR && git checkout 77b63d3
git apply ../../docs/baselines/auxiliary_patches/crosshar.patch
cp ../../docs/baselines/auxiliary_patches/crosshar_new_files/* .

cd ../LIMU-BERT-Public && git checkout decffee
git apply ../../docs/baselines/auxiliary_patches/limubert.patch
cp ../../docs/baselines/auxiliary_patches/limubert_new_files/* .
mkdir -p config
cp ../../docs/baselines/auxiliary_patches/limubert_new_files/*.json config/
```

LanHAR required no source code modifications.

## CrossHAR Changes

**Patch** (`crosshar.patch` — 3 files):
1. `augmentations.py`: Fix NumPy permutation bug (permute indices instead of array of arrays)
2. `dataset/data_config.json`: Expand from 4 to 15 datasets with full per-dataset activity labels
3. `utils.py`: Expand CLI dataset choices + add NaN safety guard

**New files** (`crosshar_new_files/`):
- `create_combined_dataset.py`: Build combined training set from all 10 training datasets
- `pretrain_combined.json`: Config for combined pretraining

## LIMU-BERT Changes

**Patch** (`limubert.patch` — 4 files):
1. `config.py`: Robust config parsing — filter unknown JSON keys (e.g., `has_gyro`) before NamedTuple construction
2. `dataset/data_config.json`: Expand from 4 to 15 datasets with `has_gyro` metadata per dataset
3. `train.py`: Re-enable periodic checkpoint saving (was commented out in original)
4. `utils.py`: Remove hardcoded dataset/version restrictions from CLI argument parsers

**New files** (`limubert_new_files/`):
- `create_combined_dataset.py`: Build combined training set
- `pretrain_100ep.json`: Config for 100-epoch pretraining
- `train_100ep.json`: Config for 100-epoch downstream training
- `run_all_training.sh`: Script to run full pretraining + embedding extraction pipeline
