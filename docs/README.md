# Documentation Index

This file defines where to look first and which files are authoritative.

## Single Source Of Truth

| Topic | Authoritative Source |
|------|-----------------------|
| Semantic alignment training behavior and active hyperparameters | `training_scripts/human_activity_recognition/semantic_alignment_train.py` |
| Optional Stage-1 pretraining behavior and hyperparameters | `training_scripts/human_activity_recognition/pretrain.py` |
| Evaluation behavior per model | `val_scripts/human_activity_recognition/evaluate_*.py` |
| Cross-model evaluation protocol and fairness rules | `docs/baselines/EVALUATION_PROTOCOL.md` |
| Baseline implementation details and paper adaptations | `docs/baselines/BASELINE_IMPLEMENTATION_NOTES.md` |
| Reproducible result artifacts | `docs/baselines/results/*_evaluation.json` |
| Human-readable result summary | `docs/baselines/RESULTS.md` |
| Data preprocessing pipeline | `datascripts/setup_all_ts_datasets.py` and `datascripts/README.md` |
| Benchmark preprocessing pipeline | `benchmark_data/scripts/*.py` and `benchmark_data/README.md` |

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
