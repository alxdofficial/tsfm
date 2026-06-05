# EXP-P6 — Multi-seed variance (headline + key ablations)

**Branch:** `rebuttal/experiment/multi-seed`
**Addresses:** Reviewer **E4** ("report variance over multiple seeds, especially because the ablation
runs use a shorter training schedule than the final model") and **A6**.
**Type:** TRAIN ×N on RunPod. **No code change** — `TSFM_SEED` is fully wired (Phase 0).

## What it tests
Whether the headline result and the key ablation effects are **robust to random seed**, not a
single-run artifact. We report **mean ± std** across ≥3 seeds so the rebuttal can put error bars on
the headline lead (+13.7 pp ZS-open over MOMENT) and on the ablation deltas.

## Variants × seeds
Run each config at **seeds 42, 43, 44**:

| variant | what it is | env (set by the runner) |
|---|---|---|
| `headline` | the full model (the deployed recipe) | `TSFM_MODEL_SIZE=small_deep` |
| `no_soft_targets` | ablate soft targets | `ABLATION_SOFT_TARGETS=0` |
| `no_queue` | ablate the memory queue | `TSFM_QUEUE_MODE=none TSFM_GRAD_CACHE=0` |

(Extend with whichever ablations from P1/P7 prove meaningful — promote only those to 3 seeds.)

## How to run (one pod per run)
```bash
# on each pod (3090 community is plenty — see paper-rebuttal/experiments/RUNPOD_PLAN.md)
bash scripts/runpod_experiment.sh --exp P6 --variant headline --seed 42
bash scripts/runpod_experiment.sh --exp P6 --variant headline --seed 43
bash scripts/runpod_experiment.sh --exp P6 --variant headline --seed 44
# ... repeat for no_soft_targets / no_queue
```
Each pod self-terminates on success; artifacts land on `/workspace/artifacts` + Drive.

## How to interpret
After pulling artifacts (`scripts/runpod_fetch.sh`):
```bash
python scripts/runpod_aggregate.py paper-rebuttal/experiments/runpod_artifacts/P6
```
→ prints `val_accuracy / val_mrr / val_loss` as **mean ± std** per config across seeds.
**Reporting:** if std is small relative to the ablation delta, the effect is real → state
"mean ± std over 3 seeds." Determinism note: with `TSFM_NO_COMPILE=1` the per-seed variance is
dominated by data-order/init, not kernel noise.

## Cost
~3 h/run on a 3090 (~$0.75). Minimum set = 3 configs × 3 seeds = 9 runs ≈ $7 (the headline seed-42
run doubles as the P1 `hard_neg`/P7 baseline).
