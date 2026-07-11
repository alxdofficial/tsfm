# Baseline Training Readiness

**Status:** **NO-GO for final billable runs**, updated 2026-07-11 after a clean-pod,
data-contract, architecture, optimization, and publication sweep. The audited
training-code baseline is `78807b41d70f8213e0ff870734c35ac752c4a91c`.

All six active adapters now exist. "Implemented" is not the same as "ready":
three cannot start from a clean cloud checkout, and every row still has at least
one scientific or reporting gate. Per-model evidence, paper citations, and exact
deviations are canonical in
[`../baselines/BASELINE_IMPLEMENTATION_NOTES.md`](../baselines/BASELINE_IMPLEMENTATION_NOTES.md).
Shared fairness rules are canonical in
[`../baselines/EVALUATION_PROTOCOL_V2.md`](../baselines/EVALUATION_PROTOCOL_V2.md).

## Current Gate Matrix

| Model | Clean-pod execution | Scientific status | Final-run gate |
|---|---|---|---|
| CrossHAR | Passes | No-go | Retrain backbone on the frozen 10-source corpus; subject-disjoint/balanced/calibrated 86-way head; provenance |
| LiMU-BERT | Fails: missing `models` package | No-go | Package pinned source; correct units and transition labels; current-corpus paper-strength or explicitly compute-matched training |
| SSL-Wearables | Conditional network fetch | Conditional | Pin upstream revision/weight; use exact `harnet5 frozen-head ConSE` name or report a separately named strong fine-tune; calibrate head |
| UniMTS | Fails: missing `contrastive` package and checkpoint | No-go | Fix Hugging Face ID; package source/weight; verify placement, units, resampling, and 6-to-10-second adaptation |
| NormWear | Fails: missing `NormWear` package and checkpoints | No-go | Package strict weights; implement 65 Hz native preprocessing, real-channel input, and frozen natural label text |
| DeepConvLSTM | Passes | Conditional | Common validation metric; at least five seeds or subject folds; strict all-dataset completion and provenance |

No row in this table is currently approved as a final paper run. CrossHAR and
DeepConvLSTM are useful for cloud smoke tests only.

## 1. Shared Scientific Gates

### 1.1 Freeze The Final HALO And Data Contract

The existing HALO result files reference the historical
`small_deep_v2_4b3fdd6` checkpoint. Its saved configuration is
spectral-temporal/200 epochs, while current V2 forces the physical filterbank and
uses 100 epochs. It also predates the addition of Capture24. Newly trained
baselines cannot be compared with those JSONs as if they represented current V2.

Before training any final model, freeze and hash:

- the exact 10-source manifest and source caps (recgym dropped 2026-07-11 → global
  ConSE source vocab is now 86 labels, not 94; cached 94-way heads must be refit);
- all six target window indices, subject IDs, and one canonical window label;
- per-dataset units, gravity convention, real channels, and native rate;
- the current HALO architecture, loss, source sampler, epochs/steps, and seeds;
- the data bundle, label configs, and evaluation code commit.

The current 20 Hz and native-rate label tensors agree in window and subject order
but disagree on three RealWorld and one MobiAct majority labels. All scorers must
consume a single rate-independent ground-truth artifact before final runs.

### 1.2 Correct Model-Native Data

- LiMU-BERT preprocessing currently leaves UCI-HAR, HAPT, and UniMiB near g-scale
  and then divides them by 9.8 as if they were m/s^2. RecGym has no physical scale.
- NormWear currently receives 20 Hz x 120 samples even though its published
  pipeline is 65 Hz x 6 seconds with detrending and Gaussian smoothing. Upstream
  code does not automatically resample a supplied 20 Hz tensor.
- NormWear must receive only real channels; zero-padded gyroscope channels are an
  implementation artifact for fixed six-axis baselines, not observed sensors.
- UniMTS needs a frozen per-dataset placement, unit, gravity, resampling, and
  padding contract. Its one-core-stream parity row and any multi-placement native
  row must remain separate.
- SSL-Wearables must continue using dedicated 30 Hz, g-unit, gravity-present
  acceleration arrays rather than LiMU-BERT's 20 Hz m/s^2 copy.

### 1.3 Freeze Fair Training Policies

Do not equate models by epoch count alone. Current 100-epoch head schedules
correspond to roughly 23,000 CrossHAR, 138,000 LiMU-BERT, and 15,000
SSL-Wearables optimizer updates because LiMU-BERT creates six classifier samples
per parent window. Record examples and optimizer steps.

The final policy must pre-register:

- whether CrossHAR uses its official 1,600/800 schedule, the repository's custom
  200/100 schedule, or both as separately named paper-faithful/compute-matched rows;
- whether LiMU-BERT uses the paper's 3,200 pretraining and 700 classifier epochs
  or an explicitly named compute-matched deviation;
- a shared corpus-matched source class/dataset sampler or loss, plus separately
  named faithful unweighted runs where required;
- held-out source-subject validation and source-only temperature scaling for all
  ConSE heads;
- validation macro-F1 as the common supervised checkpoint-selection metric;
- at least five registered seeds or rotated subject folds for DeepConvLSTM and
  HALO few-shot rows.

### 1.4 Report Comparisons In The Correct Category

- **Corpus-matched:** HALO, CrossHAR, and LiMU-BERT after all three are trained on
  the same frozen sources.
- **Externally pretrained:** SSL-Wearables, UniMTS, and NormWear, with their
  external data and checkpoint variants visible in the table.
- **Supervised floor:** DeepConvLSTM in FS/full-shot only.

The categories must not be collapsed into a single claim of equal training data.
Current fixed windows expose about 218.1 hours to CrossHAR/LiMU-BERT and 116.5
center-cropped hours to the SSL head, while HALO's native-session configuration is
roughly 399 hours. SSL-Wearables additionally brings about 700,000 person-days of
external UK-Biobank pretraining. See the cited model notes for sources.

## 2. Clean-Pod And Artifact Gates

`auxiliary_repos/` is entirely Git-ignored. The data bundle contains processed
arrays and cached ConSE heads, not the LiMU-BERT, UniMTS, or NormWear source
packages. `cloud/apply_recipe.py` downloads only CrossHAR and LiMU-BERT backbone
files and does not clone or pin upstream repositories.

Clean-checkout simulation of the current recipe inputs produced:

```text
crosshar       setup/refit path: PASS
deepconvlstm   build/backward path: PASS
limubert       ModuleNotFoundError: models
unimts         ModuleNotFoundError: contrastive
normwear       ModuleNotFoundError: NormWear
```

The UniMTS recipe also uses the wrong released model account:
`xiyuanzh/UniMTS` instead of `xiyuanz/UniMTS`.

Before a job can be marked ready, preflight must verify:

- exact Git SHA is reachable, not merely repository `HEAD`;
- pinned upstream source commit is installed/importable;
- every required checkpoint exists and matches SHA-256;
- the exact recipe runs in a clean checkout with the current data bundle;
- all requested result files are newly created, parse, contain every requested
  dataset, and record provenance;
- a deliberately injected model/dataset failure produces a nonzero process exit,
  `FAILED`, and no accepted stale result.

The existing `cloud/preflight.py` currently prints `READY` for all six jobs even
though the three imports above fail. Its success is not approval to spend.

## 3. Runner Failure Semantics

**Partially fixed 2026-07-11.** `run_baselines_v2.py` and `run_fewshot_v2.py` now
stream per-dataset output to a `.partial.json` sidecar, drop any stale final file
before running, and only atomically promote the sidecar to the final
`baseline_v2_*.json`/`fewshot_v2_*.json` once **every requested dataset** succeeds;
any failure records `_status:"failed"` + `_failed_datasets` and exits the process
nonzero. The five dirty 2/6-dataset `baseline_v2_*.json` were purged. **Still open:**
UniMTS and NormWear evaluator modules have no command-line `main`, so the first
command in each cloud recipe is a no-op; the following caught setup error can still
lead to a zero exit and `DONE` with no result — this must be closed in the recipe
layer + preflight (§2), not just the runners.

The final harness must satisfy all of the following:

- any requested model or dataset failure makes the process nonzero;
- output is written to a new run-specific path, never a tracked/stale filename;
- a success sentinel is emitted only after schema, support, and hash validation;
- fleet exits nonzero if any job reports `FAILED`, `TIMEOUT`, `NO_GOOD_HOST`, or
  an incomplete result;
- result upload never treats a pre-existing repository JSON as current output.

## 4. Vast.ai Safety Gates

The current fleet defaults are unsafe for reruns and parallel processes:

- The default run ID is only `r<git-sha-prefix>`. Reusing the same SHA can expose
  a new pod to an old R2 `DONE`/`FAILED` sentinel.
- Separate fleet processes at the same default run ID share a teardown namespace;
  one process can reconcile and destroy another process's pod.
- `cloud/halo up` only recognizes `--vast`; other fleet flags are interpreted as
  job names. It also calls `python` and `vastai` from `PATH`, so this checkout
  requires activating `.venv` first.
- `cloud/halo nuke` uses an obsolete CLI path and lacks noninteractive `-y`.
- The on-pod `poweroff` watchdog is not an API-level Vast contract destruction and
  must not be the only billing backstop.

For any smoke run before these are fixed, use one unique timestamped run ID per
fleet process, verify the R2 prefix is empty, launch jobs sequentially, monitor
the instance list independently, and manually confirm destruction. Final runs
remain blocked until these safeguards are enforced by code.

## 5. Resource Gates

- CrossHAR and LiMU-BERT materialize large sequence embeddings and need at least
  32 GiB host RAM. The current offer query does not enforce host RAM even though
  it accepts a `min_gb` argument.
- A 24 GiB RTX 4090 is sufficient for the measured local forward/training smoke
  tests. NormWear batch 128 peaked below that capacity after model loading.
- GPU memory alone is not a complete offer requirement; record host RAM, disk,
  GPU model/VRAM, CUDA, PyTorch, and wall time for every run.

## 6. Acceptance Checklist

A final fleet launch is approved only when every item is checked:

- [ ] Current HALO configuration and 10-source data manifest frozen (recgym dropped)
- [ ] One canonical GT artifact used by HALO and every baseline
- [ ] LiMU-BERT source units and transition-label handling corrected
- [ ] NormWear 65 Hz native preprocessing and real-channel handling validated
- [ ] LiMU-BERT, UniMTS, and NormWear source/weights packaged and pinned
- [ ] CrossHAR and LiMU-BERT backbones retrained under named schedules
- [ ] ConSE source-subject validation, balance policy, and calibration frozen
- [ ] DeepConvLSTM/HALO few-shot folds or seeds frozen
- [ ] Parameter, corpus, context, and effective-hours disclosures emitted
- [~] Runner failures propagate nonzero and stale/partial outputs are rejected
      (runners done 2026-07-11; recipe no-op `main` + preflight validation still open)
- [ ] Unique run IDs and sentinel isolation enforced
- [ ] Preflight executes each exact clean-pod recipe and validates output schema
- [ ] Full repository and focused evaluation tests pass

## Verified So Far

- `pytest -q`: 170 passed, 25 return-value warnings in dataset conversion tests.
- `tests/test_eval_v2.py`: 29 passed.
- CrossHAR, LiMU-BERT, SSL-Wearables, UniMTS, and NormWear local forwards produced
  finite outputs when their local ignored dependencies were present.
- DeepConvLSTM completed a finite forward, loss, backward, and gradient smoke.
- No existing R2 sentinel was present under `runs/r78807b41` during the audit.

These checks establish numerical viability, not publication readiness. The gates
above remain authoritative.
