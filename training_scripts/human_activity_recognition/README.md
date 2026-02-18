# Human Activity Recognition Training

Current workflow focuses on semantic alignment training (`semantic_alignment_train.py`).

## Recommended Workflow (Current)

```bash
cd training_scripts/human_activity_recognition
python semantic_alignment_train.py
```

## Optional Legacy Workflow

Stage-1 pretraining (`pretrain.py`) still exists and can be run, but it is optional in the
current workflow.

```bash
cd training_scripts/human_activity_recognition
python pretrain.py
python pretrain.py --resume path/to/checkpoint.pt
```

## Single Source Of Truth

To avoid config drift, treat script constants as authoritative:

- `semantic_alignment_train.py`: active semantic-alignment training behavior
- `pretrain.py`: optional Stage-1 pretraining behavior
- `val_scripts/human_activity_recognition/model_loading.py`: checkpoint loading contract used by evaluation

This README is intentionally high-level and should not be used as the canonical source for exact
constant values.

## File Structure

```text
training_scripts/human_activity_recognition/
├── pretrain.py                  # Optional Stage-1 pretraining
├── semantic_alignment_train.py  # Main training script (current default)
├── losses.py                    # Stage-1 losses (MAE + contrastive)
├── semantic_loss.py             # Stage-2 losses (InfoNCE variants)
├── memory_bank.py               # MoCo-style memory bank
└── README.md                    # This file
```

## Outputs

### Semantic Alignment (Current)

```text
training_output/semantic_alignment/{timestamp}/
├── hyperparameters.json
├── epoch_*.pt
├── best.pt
└── plots/
```

### Optional Stage-1 Pretraining

```text
training_output/imu_pretraining/{timestamp}/
├── config.yaml
├── latest.pt
├── best.pt
├── checkpoint_epoch_*.pt
└── plots/
```

## Loading Trained Models

```python
from val_scripts.human_activity_recognition.model_loading import load_model, load_label_bank

model, checkpoint, hyperparams_path = load_model("path/to/best.pt", device)
label_bank = load_label_bank(checkpoint, device, hyperparams_path)
```

## Tests

```bash
pytest tests/ -v
pytest tests/test_losses.py -v
pytest tests/test_memory_bank.py -v
```
