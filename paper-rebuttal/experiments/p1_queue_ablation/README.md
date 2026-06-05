# EXP-P1 — Memory-queue ablation (none / hard_neg / semantic)

**Branch:** `rebuttal/experiment/queue-ablation`
**Addresses:** Reviewer **A2** — *"Soft targets are built for in-batch labels, but queued samples
appear to be hard negatives, which may push semantically similar labels apart… clarify whether
semantic similarity is computed over the queue, and add an ablation comparing no queue,
hard-negative queue, and semantic-aware queue."*
**Type:** TRAIN ×3 on RunPod. **The semantic arm is the only new ML code in the campaign.**

## The concern is real and live
A is correct. In the current default (`hard_neg`), queued entries receive target probability
**exactly 0** in the soft-target loss, so a synonym sitting in the queue ("walking" while the batch
has "strolling") is treated as a negative and pushed apart — contradicting the synonym-aware design.

## What each mode does
| mode | queue | queued-entry targets | env |
|---|---|---|---|
| `none` | off | — (no memory bank) | `TSFM_QUEUE_MODE=none` |
| `hard_neg` | on | **0** (hard negatives — current default) | `TSFM_QUEUE_MODE=hard_neg` |
| `semantic` | on | **soft**, by label-text similarity | `TSFM_QUEUE_MODE=semantic` |

**Mechanism (semantic):** `MemoryBank` caches the frozen SBERT mean-pool text of each queued entry
(`frozen_queue`, lock-step with the imu/text queues, checkpoint-persisted). The loss extends the
soft-target support to the queue — `soft_text_all = cat([frozen_batch, frozen_queue])` — so queued
synonyms get soft target mass instead of zero. The `targets` matrix is shared by the i2t and t2i
directions, so both are fixed at once. **All P1 runs force `TSFM_GRAD_CACHE=0`** (GradCache bypasses
the queue, which would make the three modes identical — guarded with a hard error).

## How to run (one pod per mode)
```bash
bash scripts/runpod_experiment.sh --exp P1 --variant none
bash scripts/runpod_experiment.sh --exp P1 --variant hard_neg
bash scripts/runpod_experiment.sh --exp P1 --variant semantic
```
~3 h/run on a 3090 (~$0.75); 3 runs ≈ $2.25 (`hard_neg` ≈ the deployed/headline recipe).

## How to interpret
Compare ZS-open / unseen accuracy across the 3 modes after scoring the checkpoints:
- **semantic ≥ hard_neg** → confirms the queue's hard-negative treatment was hurting synonyms; the
  fix helps (the honest, A-satisfying answer).
- **semantic ≈ hard_neg** → the queue effect is small at this scale; report it plainly and note the
  in-batch soft targets already carry most of the signal.

## Verification (done, before any pod spend)
Unit test (`code change verified locally`): the loss's queued-column target mass is **0.0000** in
`hard_neg` and **0.63** in `semantic` (rows still sum to 1); the three modes give distinct losses;
`frozen_queue` round-trips through the checkpoint. End-to-end: a `TSFM_QUEUE_MODE=semantic
TSFM_EPOCHS=1` run completes clean and records `queue_mode: semantic` in hyperparameters.json.
