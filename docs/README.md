# Documentation Index

This file defines where to look first and which files are authoritative.

## Single Source Of Truth

| Topic | Authoritative Source |
|------|-----------------------|
| Semantic alignment training behavior and active hyperparameters | `training_scripts/human_activity_recognition/semantic_alignment_train.py` |
| Evaluation behavior per model | `val_scripts/human_activity_recognition/evaluate_tsfm_v2.py` and `val_scripts/human_activity_recognition/run_baselines_v2.py` |
| Cross-model evaluation protocol and fairness rules | `docs/baselines/EVALUATION_PROTOCOL_V2.md` |
| Baseline implementation details and paper adaptations | `docs/baselines/BASELINE_IMPLEMENTATION_NOTES.md` |
| Reproducible result artifacts | `test_output/eval_v2/*.json` |
| Human-readable result summary | `docs/baselines/RESULTS_V2.md` |
| Data preprocessing pipeline | `datascripts/setup_all_ts_datasets.py` and `datascripts/README.md` |
| Benchmark preprocessing pipeline | `benchmark_data/scripts/*.py` and `benchmark_data/README.md` |
| Dataset catalog (roles, sensors, rates, sizes, caveats) | `docs/DATASOURCES.md` |
| Augmentation catalog (params + presets) | `docs/AUGMENTATIONS.md` |
| Baseline model catalog (how/why/size/heterogeneity/open-set) | `docs/baselines/BASELINES_OVERVIEW.md` |

## Supporting Documents

| Document | Role |
|---------|------|
| `README.md` | Project overview and quick-start commands |
| `docs/ARCHITECTURE.md` | Conceptual model design and component-level explanations |
| `docs/EXPERIMENTS.md` | Experiment notes and historical context (not the runtime source of truth for current constants) |
| `model/README.md` | Encoder/module API usage notes |
| `DATA_FORMAT.md` | Standardized dataset format contract |

## Navigation

- Baseline docs: `docs/baselines/`
- Training scripts: `training_scripts/human_activity_recognition/`
- Evaluation scripts: `val_scripts/human_activity_recognition/`
- Dataset conversion scripts: `datascripts/`
- Benchmark conversion scripts: `benchmark_data/scripts/`
