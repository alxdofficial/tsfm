# Prompt for an independent AI agent — review the HALO rebuttal code changes

*Copy everything below the line into a fresh agent session (with access to this repo).*

---

You are an **independent, skeptical code reviewer**. Audit a set of changes made for a MobiCom paper
rebuttal (HALO — an IMU↔text contrastive "foundation model" for activity recognition). The changes
(a) fix evaluation bugs and speed up eval, (b) add ablation-campaign infrastructure, (c) add a new
"semantic-aware memory queue" training feature, and (d) add RunPod orchestration that spends real GPU
money. These feed the paper, so **correctness and comparison-fairness matter more than style**. Do not
trust comments or commit messages — read the code and trace the real call paths.

## Setup
```bash
cd <repo-root>
git fetch
git checkout rebuttal/experiment/queue-ablation   # most complete branch: eval fixes + Phase-0 + P1 + scripts
git diff --stat master...HEAD                      # the full scope of changes
```
Other branches: `rebuttal/experiment/{multi-seed,fine-grained-ablations}` (same infra, different READMEs);
`rebuttal/fix/pre-ablation-bugs` (the eval fixes + Phase-0 base).

## What to scrutinize (by area)

### A. Evaluation correctness — the headline rebuttal fix
`val_scripts/human_activity_recognition/evaluate_{tsfm,moment,crosshar,lanhar,limubert,llasa}.py`,
`generate_lanhar_descriptions.py`, `datasets/imu_pretraining_dataset/multi_dataset_loader.py`
- **HARTH label-index fix**: `get_window_labels` subtracted the per-window min (HARTH activity codes are
  `[2..11]`, min>0) to 0-index, but did not restore the offset → every HARTH ground-truth was shifted by
  −2 groups. The fix re-adds `+ t`. **Verify:** applied in *every* evaluator; affects **only** HARTH
  (the sole dataset with min code >0 — so 5-main + VTT must be unchanged); no double-correction.
- **MOMENT SVM parallelization** (`evaluate_moment.py`): `predict`/`decision_function` chunked across
  threads (joblib). **Verify** the chunked+concatenated result is **bit-identical** to serial (order
  preserved, no off-by-one).
- **HALO eval speedups** (`evaluate_tsfm.py`): per-patch majority vote vectorized to a GPU `scatter_add`;
  extraction batch 32→256. **Verify** the vectorized vote equals the old per-window loop (tie-break =
  first index; padded patches masked out); batch size doesn't change embeddings beyond fp noise.
- **Cache/split fix** (`multi_dataset_loader.py`): seeded RNG for the dataset split. **Verify** train/val/
  test are disjoint and deterministic.

### B. Ablation infrastructure (Phase 0) — `training_scripts/human_activity_recognition/semantic_alignment_train.py`
- **`TSFM_CONFIG_OVERRIDES`** (JSON → `_cfg.update`): **verify it runs BEFORE every global unpack**,
  especially `CONTRASTIVE_TEXT_MODEL/DIM` and `SEMANTIC_DIM` (the SBERT-swap ablation depends on this);
  malformed JSON should fail loudly.
- **Queue-mode / GradCache guards**: **verify a normal medium/large run** (`TSFM_GRAD_CACHE=1`, no queue
  env) still trains with NO spurious `SystemExit`; the exit must fire only on an *explicit*
  `TSFM_QUEUE_MODE` request under GradCache.
- **`TSFM_EPOCHS`, soft-target env**: wired and recorded truthfully in `hyperparameters.json`.

### C. P1 semantic queue (new ML code) — `memory_bank.py`, `semantic_loss.py`, `semantic_alignment_train.py`
- **`MemoryBank.frozen_queue`**: **verify** it stays aligned 1:1 with the imu/text queues across the
  wraparound case AND the `batch_size >= queue_size` case; `state_dict`/`load_state_dict` round-trip.
- **`semantic_loss._forward_single_prototype` semantic branch**: **verify** `soft_text_all =
  cat([frozen_batch, frozen_queue])` yields exactly `B+Q` columns matching the logits; the `hard_neg`
  path is unchanged (no regression); BOTH i2t and t2i are fixed (they share the `targets` matrix);
  SigLIP and multi-prototype paths are unaffected.
- **Train plumbing**: `frozen_dim = CONTRASTIVE_TEXT_DIM` is the dim of `encode_frozen()` output;
  `flat_frozen` aligns with the queue flatten (`flat_imu_q`); frozen passed in BOTH per-patch and pooled
  criterion calls AND at the warmup enqueue.
- **Quick check:** in `hard_neg`, queued-column target mass should be `0`; in `semantic`, `>0`, with rows
  still summing to 1.

### D. RunPod orchestration — `scripts/runpod_experiment.sh`, `make_code_bundle.sh`, `runpod_fetch.sh`, `runpod_aggregate.py`
- **Auto-terminate safety**: **verify** the pod is NEVER terminated when training failed (`TRAIN_RC!=0`)
  or before artifacts are copied; artifacts are copied even on failure.
- **Artifact retrieval**: the `RUN_DIR` glob `*_ablation_${RUN_TAG}` must match the train script's output
  dir (it appends a microsecond timestamp); **`plots/metrics.json`** (not a run-root `metrics.json`) is
  the file copied; `best.pt` falls back to the latest `epoch_*.pt`.
- **Variant→env map** (the `resolve_env` case): cross-check every variant against the READMEs
  (`paper-rebuttal/experiments/p{1,6,7}_*/README.md`). A wrong/missing env = an invalid or wasted run.
  Especially: P1 variants must set `TSFM_GRAD_CACHE=0`; `sbert_mpnet` must set all three of
  `contrastive_text_model`/`contrastive_text_dim`/`semantic_dim`.
- **Robustness**: `set -uo pipefail` with possibly-unset `RUNPOD_POD_ID`/`RUNPOD_API_KEY`; idempotent
  setup; the `TSFM_CONFIG_OVERRIDES` JSON must survive the env array intact (quoting); graceful behavior
  if Drive/rclone/runpodctl are absent.

## Ground truth to check against
`docs/baselines/RESULTS.md` (corrected HARTH numbers), `paper-rebuttal/experiments/RUNPOD_PLAN.md`
(the design + the 22 issues the original automated sweep found — confirm the blocking ones are fixed),
and the per-experiment READMEs.

## Output format
For each issue: `` `file:line` `` · **severity** (critical / high / medium / low) · a **concrete failure
scenario** (how it bites) · the **fix**. Put a **"FIX BEFORE SPENDING GPU MONEY"** list first (anything
that would invalidate an ablation, waste a pod run, or lose a checkpoint), then lower-priority items.
End with a one-line **go / no-go** on launching the campaign. Don't pad — only real issues.
