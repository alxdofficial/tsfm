# HALO Repo Cleanup Plan

**Date:** 2026-07-02 | **Branch:** V2 | **Status:** PLAN — nothing deleted yet; awaiting review.

Produced by a 7-subsystem staleness audit (model, training, dataloader, dataset
converters, eval, baseline repos, scripts/docs/outputs), each grepping for real
references, then synthesized. Ground-truth spot-checks confirmed: V2 = master +
additions (no deletions); `auxiliary_repos/` is gitignored (local-disk only, not
git bloat); `pretrain.py`/`losses.py` (Stage-1) are orphaned from the headline
path; 48 `data/**/debug_*.png` + 1 stray PDF + v1 result JSONs are git-tracked;
`evaluate_tsfm_v2` imports `evaluate_tsfm` (so the v1 file needs an `eval_common`
split before deletion). Execute on a clean-slate V2 branch cut from HEAD, with
`v1-archive` tagged first.

---

# HALO Repo Cleanup Plan

*Lead-engineer synthesis of seven subsystem audits. This is a PLAN — nothing is deleted yet. Every claim below traces to auditor evidence; risk labels are preserved and reconciled across auditors.*

---

## 0. Executive summary

### Headline numbers

| Category | Approx. removable | Where |
|---|---|---|
| **Dead/legacy source code** | **~10,000–12,000 LOC** | Stage-1 stack, dropped-baseline wrappers, v1 eval scoring, in-file `__main__` harnesses, dead aug/model branches |
| **Local disk (untracked, gitignored)** | **~4–5 GB reclaimable** | `auxiliary_repos/` sub-trees (3.8G), plus optional relocation of `paper-rebuttal/` (21M) and `.serena/` (11M) |
| **Git-tracked bloat to untrack** | **~10 MB + policy fixes** | 48 `data/*/debug_*.png`, one stray PDF, stale v1 JSONs, `data/vtt_coniot/` |
| **Whole directories to drop** | 5 dataset converters + 3 baseline vendored repos + dropped-baseline wrappers | — |

### The 3–5 biggest wins

1. **Retire the entire Stage-1 pretrain stack** (`pretrain.py` 1092 + `losses.py` 700 ≈ 1,800 LOC, ~36% of the training subsystem) — disconnected from the headline model (`PRETRAINED_ENCODER_PATH=None`). Also decouples the eval import graph from Stage-1 losses via `__init__.py`.
2. **Drop MOMENT / LanHAR / LLaSA baselines** — deletes `evaluate_lanhar.py` (1882), `evaluate_moment.py` (812), `evaluate_llasa.py` (441), `generate_lanhar_descriptions.py` (~440) ≈ **3,575 LOC**, frees **~742 MB** local (LanHAR 737M + LLaSA 5.5M), and removes ~40% of the v2 orchestrator's dispatch.
3. **Split `evaluate_tsfm.py` (1076 LOC)** into a shared `eval_common.py` + delete the v1 scoring/CLI half — unblocks retiring `grouped_zero_shot.py` group scorers, `evaluation_metrics.compute_similarity/compute_semantic_recall`, and `harth_analysis.py`.
4. **Modularize baselines behind a `BaselineAdapter` Protocol** — collapses five 440–1882-line monoliths into ~80-line adapters; adding UniMTS + ssl-wearables becomes a ~60–100-line file each instead of an 800-line script.
5. **Reclaim ~4–5 GB local disk** by pruning regenerable data inside `auxiliary_repos/` (`CrossHAR/dataset/` 927M, `LIMU-BERT-Public/embed/` 2.0G, nested `.git` 124M) and adding a `scripts/fetch_baselines.sh` for reproducibility.

### The split: shared-legacy (A) vs V2-only (B)

- **(A) Shared legacy — remove/clean on BOTH branches** because it is dead regardless of v1/v2: the Stage-1 stack, in-file `__main__` demos in `model/`, dead augmentation methods, dead `ChannelSemanticEncoding`, dropped-dataset converters and their config entries, the tracked `debug_*.png` and stray PDF, broken hardcoded paths (`/home/alex/code/tsfm`, `alxdofficial/tsfm@master`).
- **(B) Clean V2 repo — keep vs drop**: KEEP the v2 eval stack (`eval_v2.py`, `evaluate_tsfm_v2.py`, `evaluate_baselines_v2.py`→`run_baselines_v2.py`, `assemble_v2_table.py`, `model_loading.py`), the headline train path (`semantic_alignment_train.py` standard loop, `semantic_loss.py` InfoNCE, `memory_bank.py`), 16-dataset converters, and the `small_deep`/`small` presets. DROP the v1 eval scoring/CLI, v1 docs/runners, dropped baselines, and (pending human sign-off) the Stage-1 capability. The `RESULTS.md`↔`RESULTS_V2.md` and `PROTOCOL`↔`PROTOCOL_V2` duplication collapses to one v2 doc each.

**Critical "do NOT delete" caveats** (multiple auditors, high-consequence): several model variants *look* dead but are kept alive by ablation checkpoints — `ChannelIndependentTemporalTransformer`/`TemporalTransformerBlock`, `FixedPatchCNN`/`ChannelIndependentCNN`, the session-level `TemporalAttention`/`MultiQueryPooling` path, and `LearnableLabelBank._mean_pool`. The ABLATION_* env flags, VISUALIZE_EMBEDDINGS/UMAP scaffolding, `memory_bank.py`, `label_groups.py` (LABEL_GROUPS), and `plot_utils.py` are all LIVE. Keep them.

---

## 1. Delete / archive inventory (ranked, with risk)

### 1A. Dropped-baseline code (MOMENT / LanHAR / LLaSA) — V2-only, low-med risk

| Path | Why (evidence) | Risk | Action |
|---|---|---|---|
| `val_scripts/.../evaluate_lanhar.py` (1882 LOC) | LanHAR in drop set; only importer is `evaluate_baselines_v2.py` + `run_all_evaluations.sh`; carries ~450 LOC unused train_stage1/2 | med | delete |
| `val_scripts/.../evaluate_moment.py` (812) | MOMENT in drop set; only importer `evaluate_baselines_v2`; also drop `momentfm` from requirements if unused elsewhere | low | delete |
| `val_scripts/.../evaluate_llasa.py` (441) | LLaSA cut (7B, ZS-only); only importer `evaluate_baselines_v2` (gated `--include-llasa`) | low | delete |
| `val_scripts/.../generate_lanhar_descriptions.py` (~440) | Feeds only `evaluate_lanhar.load_per_sample_descriptions`; LanHAR-specific | low | delete |
| `auxiliary_repos/LanHAR/` (737M) | Vendored repo for dropped baseline; v2 never reads `uci_best_model.pth` | low | delete (local) |
| `auxiliary_repos/LLaSA/` (5.5M) | Dropped; only `evaluate_llasa.py` refs it | low | delete (local) |
| `evaluate_baselines_v2.py` moment/lanhar/llasa branches | `setup_moment/moment_probs`, `setup_lanhar/lanhar_predict`, `setup_llasa/llasa_predict`, GENERATIVE_BASELINES, `--include-llasa` | med | see §2/§4 (modularize, not wholesale delete) |

### 1B. v1-only-superseded eval — V2-only, low-med risk

| Path | Why (evidence) | Risk | Action |
|---|---|---|---|
| `val_scripts/.../eval_config.py` (47) | ZERO importers; all 5 named consumers deleted; stale datasets | low | delete |
| `val_scripts/.../harth_analysis.py` (700) | ZERO importers; wired to v1 `compute_similarity`; breaks when v1 scoring retired | med | archive/delete |
| `evaluate_tsfm.py` v1 scoring block (~500) | `evaluate_zero_shot_{open,closed,majority_vote}`, `evaluate_supervised_finetune`, `print_results_table`, `main`, v1 split helpers — NOT imported by v2 | med | delete after `eval_common` split (§2, §3) |
| `grouped_zero_shot.py` 5 dead fns + v1 group scorers | `load_dataset_config/get_mappable_info/score_exact/score_with_groups_from_names/aggregate_logits_to_test_labels` = zero callers; group scorers used only by v1 wrappers | med | reduce to `load_global_labels` (§2) |
| `scripts/run_all_evaluations.sh`, `run_baselines_only.sh`, `auto_eval_after_training.sh`, `generate_results_table.py` | v1 6-model runners; broken `/home/alex/code/tsfm` paths; dropped baselines; superseded by v2 stack | low | delete |
| `scripts/eval_ablations.sh`, `generate_ablation_results.py` | v1 metric keys + VTT-ConIoT | low | port to v2 or delete |
| `scripts/sweep_patch_sizes.py`, `benchmark_scaled_model.py` | v1 internals / non-deployed configs | low | archive |
| `docs/baselines/RESULTS.md`, `EVALUATION_PROTOCOL.md`, `docs/ablation_results.md`, `docs/baselines/results/*.json`, `SCALING.md` | superseded by `*_V2.md`; dropped baselines/VTT tier | low | archive |

### 1C. Stage-1 pretrain — shared-legacy, med risk (NEEDS-HUMAN, see confirm-first)

| Path | Why | Risk | Action |
|---|---|---|---|
| `training_scripts/.../pretrain.py` (1092) | ZERO runtime importers; `PRETRAINED_ENCODER_PATH=None`; README "Optional Legacy Workflow" | med | archive (confirm first) |
| `training_scripts/.../losses.py` (700) | Only used by `pretrain.py` + `__init__.py` + `tests/test_losses.py` | med | archive with pretrain |
| `datasets/.../augmentations.py` get_weak/strong/mixed/create_positive_pair + channel_shuffle/time_shift/time_warp/magnitude_warp/resample (~300+) | Stage-1 contrastive scaffolding; dead for headline | med | delete with pretrain decision |
| `multi_dataset_loader.py` raw branch + `collate_fn` + `create_dataloaders` + `_create_splits` 'test' branch | Coupled only to `pretrain.py` + `__main__` | med | delete with pretrain decision |

### 1D. Unused-dataset converters — shared-legacy, low risk

| Path | Why | Risk | Action |
|---|---|---|---|
| `datascripts/actionsense/` (whole dir incl. `generate_templated_qa.py`, `generate_manifest.py`, `task_templates.json`) | Absent from `dataset_config.json`; no data materialized; abandoned QA-templating mechanism referenced nowhere | low | delete |
| `datascripts/vtt_coniot/` | v1 data produced but dropped from v2; referenced in config/export_raw/preprocess_limubert (clean alongside) | low | archive |
| `datascripts/realdisp/`, `daphnet_fog/`, `usc_had/` | In `zero_shot_datasets` but NO materialized data; not in v2 EVALUATED_DATASETS | low | archive |
| `data/vtt_coniot/` (tracked labels/manifest + 4 debug PNGs) | Dropped dataset; only tracked `data/` dir outside the 16 | low | delete (untrack) |

### 1E. Bloat / artifacts — mostly local disk, low risk

| Path | Why | Risk | Action |
|---|---|---|---|
| `auxiliary_repos/CrossHAR/dataset/` (927M, incl. `vtt_coniot/` 8.5M) | CrossHAR's own training corpus; wrappers read only `saved/...model_masked_6_1.pt` | low | delete (local, regenerable) |
| `auxiliary_repos/LIMU-BERT-Public/embed/` (2.0G) | Only read by v1 `evaluate_limubert` finetune; v2 recomputes | low | delete (local) |
| `auxiliary_repos/*/.git` (LIMU 108M, LanHAR 14M, …) | Vendored VCS history | low | strip via shallow-clone fetch script |
| `auxiliary_repos/papers/` (9.3M) | Reference PDFs; no code path | low | move to `docs/reference/` or leave |
| `scripts/run_wave2_eval.sh`, `scripts/debug_eval_smoke.py` | Untracked one-offs on v1 harness | low | delete (local) |
| 48 × `data/*/debug_*.png` (~10M tracked) | Dev artifacts committed to git | low | untrack + gitignore |
| `figures/fig_dataset_discrepancy.pdf` (151K) | Violates repo's own `*.pdf` ignore rule; `.py` regenerates | low | untrack |
| `test_output/` tracked v1 JSONs (`{moment,llasa,lanhar}_evaluation.json`, `ablation_evaluation/*/tsfm_evaluation.json`, `harth_analysis/*.png`) | v1 metric schema, dropped baselines | med | delete tracked v1, keep `eval_v2/*` |
| in-file `__main__` harnesses in `model/` (~700 LOC across 5 files; `encoder.py` refs dropped ActionSense) | Not collected by pytest; real tests in `tests/` | low | move valuable bits to `tests/`, delete blocks |
| `datascripts/debug_motionsense.py`, `datascripts/shared/validate_dataset.py` | Ad-hoc debug / unreferenced validator (duplicates `verify_conversions.py`) | low | archive / merge |

### ⚠️ Confirm-first (HIGH-consequence / NEEDS-HUMAN)

- **Stage-1 pretrain retirement** (`pretrain.py`, `losses.py`, coupled aug/loader raw path, `tests/test_losses.py` Stage-1 half): auditors flag *archive not delete* — human may want the capability for paper ablations. If kept, isolate; if dropped, remove as one unit.
- **`_train_epoch_gradcache` + `forward_cached`** (~310 LOC): intended memory-saving path for medium/large (>48GB GPU). Drop only if V2 ships `small_deep` only; otherwise gate behind a clearly-named optional module.
- **`multi_dataset_loader.channel_filter` hook**: wired but never activated — may be a latent zero-shot channel-subset eval hook v2 should adopt. Do not auto-delete.
- **`MultiPrototypeLabelPooling` + `_forward_multi_prototype`**: dead across all configs (`num_prototypes=1`) but a designed capability with test coverage. Keep only if multi-prototype ablations planned.
- **`dataset_config.json` zero_shot pruning** (drop 4 stale entries): silently widens every default pipeline run, but confirm no v1-comparison run still needs them.
- **`paper-rebuttal/` relocation**: `RESUBMISSION_CLEAN_SLATE.md` is active '27 planning — confirm before moving to a separate writing repo.
- **`benchmark_scaled_model.py` / `docs/baselines/SCALING.md`**: keep only if scaling numbers are still cited in the resubmission.

---

## 2. In-file dead code (keep the file, cut the code)

**`model/semantic_alignment.py`**
- Gate `self.temporal_attention` + `self.attention_pooling` construction (lines 397–412) on `not per_patch_prediction`. Headline `small_deep` carries ~9.3M trained-but-never-executed params into every `best.pt`. **CANNOT delete** — the `small` preset checkpoint (per_patch=False) uses this session-level path; build conditionally. Changes state_dict keys → needs a checkpoint-compat shim or documented break.
- Delete `get_attention_stats` (466–498) — its only caller (`SemanticAlignmentModel.get_attention_stats`, train:609) has zero callers.

**`model/positional_encoding.py`** — delete `ChannelSemanticEncoding` (120–336, ~217 LOC) and the `use_channel_encoding` branch; superseded by `ChannelTextFusion`. Loader already tolerates absence (`strict=False`, model_loading.py:218).

**`model/encoder.py`** — after the above, collapse `forward` lines 280–297 to a single `features = self.positional_encoding.temporal_encoding(features)`. Drop `encode_from_raw` (306–386) + `preprocess()` wrapper (170–199) — used only by this file's own test (confirm no notebook depends first).

**`model/config.py`** — trim to `{small, small_deep}`; archive `TINY/MEDIUM/LARGE_CONFIG`. Remove the always-overridden `use_channel_encoding` key (present in all 5 configs, forced False everywhere). Note `pretrain.py:940` calls `get_encoder_config('default')` which raises ValueError → pretrain is itself non-runnable.

**`model/token_text_encoder.py`** — delete `LearnableLabelEncoder` + `LabelAttentionPooling` thin wrapper (referenced only by own `__main__`); superseded by `LearnableLabelBank`. Consider `MultiPrototypeLabelPooling` (216–320) → confirm-first.

**`model/feature_extractor.py`** — `MultiScaleConv1D` multi-branch machinery never selected (`cnn_kernel_sizes=[5]` everywhere); keep single-kernel path (live), optionally simplify. Keep `FixedPatchCNN`/`ChannelIndependentCNN` (live via ablations).

**`training_scripts/.../semantic_alignment_train.py`**
- Delete `SemanticAlignmentModel.get_attention_stats` (609) — no callers.
- Extract/gate `_train_epoch_gradcache` (1061–1370) — confirm-first.
- Reconcile persisted `use_channel_encoding=True` (line 1784) vs constructed False (1865); prune unused `feature_extractor_type`/`spectral_ratio` keys for the deployed encoder.
- Split the 2409-line file: `SemanticAlignmentModel` → its own module (imported by eval), training-loop functions → another, so eval doesn't drag the whole train file.

**`training_scripts/.../__init__.py`** — drop the eager `from .losses import ...` (line 8) so the eval/train import graph stops transitively loading Stage-1 losses.

**`training_scripts/.../semantic_loss.py`** — remove `SigLIPLoss` (409–503, `LOSS_TYPE` hardcoded 'infonce', no env override) and `_forward_multi_prototype` (211–286) → confirm-first with multi-prototype decision. `forward_cached` dies with GradCache.

**`datasets/imu_pretraining_dataset/`**
- `label_augmentation.py`: delete `batch_augment_labels` + `get_augmentation_stats` (0 refs); keep `augment_label` (live). Prune per-dataset configs for `vtt_coniot/realdisp/daphnet_fog/usc_had`.
- `label_groups.py`: **KEEP** (LABEL_GROUPS is training-live via `compute_group_weights` + `semantic_loss.py:23`). Prune dropped-dataset label vocabulary (VTT-ConIoT/REALDISP/USC-HAD/Daphnet strings). `LABEL_GROUPS_SIMPLE` is figures-only → keep only if regenerating UMAP figures.
- `multi_dataset_loader.py`: simplify `select_channel_groups` (75–117) to "all groups, sorted"; drop inert `min/max_channel_groups` ctor args; remove ~140-line `__main__` harness.

**`val_scripts/.../evaluation_metrics.py`** — **KEEP the file** (training needs `compute_group_accuracy` + `compute_group_accuracy_majority_vote` + re-exported `get_label_to_group_mapping`). Delete `compute_embedding_quality_metrics` (133, zero callers). Retire `compute_similarity`/`compute_semantic_recall` with the v1 scoring block.

**`val_scripts/.../grouped_zero_shot.py`** — reduce to `load_global_labels` (only v2 dependency, `evaluate_baselines_v2:45`); extract to a labels util; drop the 5 dead fns + v1 group scorers.

**`val_scripts/.../plot_utils.py`** — **KEEP** (training infra). Delete `plot_paper_figure` (1251, zero callers).

**`benchmark_data/scripts/`** — `export_raw.extract_subject()`: drop `usc_had/realdisp/daphnet_fog/vtt_coniot` branches + fix '14 datasets' docstring. `preprocess_limubert.ACC_IN_G_UNITS`: remove stale `'vtt_coniot'`.

---

## 3. Proposed clean directory structure

```
HALO/code/
├── model/                              # core model — build modes conditionally
│   ├── config.py                       # {small, small_deep} only; drop use_channel_encoding
│   ├── preprocessing.py                # live
│   ├── feature_extractor.py            # spectral_temporal + cnn (ablation-live)
│   ├── positional_encoding.py          # temporal PE only (ChannelSemanticEncoding removed)
│   ├── encoder.py                      # simplified forward, encode_from_raw dropped
│   ├── transformer.py                  # __main__ demo moved to tests/
│   ├── semantic_alignment.py           # per_patch head + conditional session head
│   └── token_text_encoder.py           # ChannelTextFusion + LearnableLabelBank
│
├── datasets/imu_pretraining_dataset/
│   ├── multi_dataset_loader.py         # patch path only (raw/collate_fn removed w/ Stage-1)
│   ├── label_groups.py                 # LABEL_GROUPS (16-dataset vocab)
│   ├── label_augmentation.py           # augment_label + 10 train configs
│   └── augmentations.py                # jitter/scale (+ rotation as documented ablation)
│
├── training/                           # was training_scripts/human_activity_recognition/
│   ├── model_def.py                    # SemanticAlignmentModel (split out for eval import)
│   ├── train.py                        # main() + standard loop
│   ├── semantic_loss.py                # InfoNCE (SigLIP removed)
│   ├── memory_bank.py                  # live
│   ├── plot_utils.py                   # training plotter/UMAP (moved from val_scripts)
│   └── __init__.py                     # no eager Stage-1 import
│
├── eval/                               # was val_scripts/human_activity_recognition/
│   ├── eval_v2.py                      # scoring core
│   ├── evaluate_tsfm_v2.py             # deployed-model evaluator
│   ├── eval_common.py                  # NEW: shared IMU-load/embed/forward helpers
│   ├── model_loading.py                # checkpoint loaders
│   ├── evaluation_metrics.py           # trimmed to training-used group metrics
│   ├── labels_util.py                  # NEW: load_global_labels
│   ├── run_baselines_v2.py             # renamed driver
│   ├── assemble_v2_table.py
│   └── baselines/                      # NEW adapter package (see §4)
│       ├── __init__.py                 # REGISTRY
│       ├── base.py                     # BaselineAdapter Protocol + ConSE/Cosine mixins
│       ├── crosshar.py                 # ~80 LOC
│       ├── limubert.py                 # ~80 LOC
│       ├── unimts.py                   # NEW
│       └── ssl_wearables.py            # NEW
│
├── datascripts/                        # 16 converters, one folder layout each
│   ├── motionsense/convert.py          # moved from loose process_motionsense.py
│   ├── <15 other dataset dirs>/
│   ├── shared/{windowing,visualization_utils,download_all_datasets}.py
│   ├── verify_conversions.py           # config-driven, iterates all 16
│   └── setup_all_ts_datasets.py        # 16 datasets incl. motionsense
│
├── benchmark_data/
│   ├── dataset_config.json             # zero_shot pruned to 6 v2 test sets
│   └── scripts/{export_raw, preprocess_limubert, preprocess_tsfm_eval,
│                preprocess_tsfm, generate_eval_v2_labels}.py
│
├── scripts/
│   ├── run_eval_v2.sh                  # single v2 runner (replaces v1 pair)
│   ├── run_ablations.sh               # training-side, kept
│   ├── fetch_baselines.sh             # NEW: clone upstreams + fetch checkpoints
│   └── setup_runpod.sh                # URL/branch updated to HALO@V2
│
├── docs/
│   ├── README.md                       # points at v2 sources
│   ├── ARCHITECTURE.md                 # rewritten for per-patch small_deep path
│   ├── EVALUATION_PROTOCOL_V2.md       # (v1 archived)
│   ├── RESULTS_V2.md                   # (v1 archived)
│   ├── ablation_results.md            # regenerated under eval-v2
│   ├── baselines/BASELINE_ADAPTERS.md # one adapter-interface doc (replaces 5)
│   └── reference/                      # baseline PDFs (moved out of code tree)
│
├── tests/                              # incl. relocated model __main__ smoke tests
├── figures/                            # PDFs untracked (regenerable)
├── archive/                            # optional: legacy_stage1/, v1_docs/, v1_eval/
├── auxiliary_repos/                    # gitignored; rebuilt by fetch_baselines.sh
├── CLAUDE.md · README.md               # rewritten for V2 (6-test, new baselines)
```

*Moves:* `process_motionsense.py` → `datascripts/motionsense/convert.py` (+ register in setup); `plot_utils.py` conceptually under `training/`; `paper-rebuttal/` → separate writing repo; baseline PDFs → `docs/reference/`.

---

## 4. Baseline modularization design

**Problem today:** five `evaluate_*.py` monoliths (5,332 LOC) mix model definition, checkpoint loading, embedding/prob extraction, and ~60% dead v1 scoring. `evaluate_baselines_v2.py` is *already* a partial adapter layer (per-baseline `setup_X` + `X_probs`/`X_predict` + dispatch dicts CONSE_PROBS/SETUPS/*_BASELINES), but logic lives as loose functions cherry-picking ~5 symbols each. Adding a baseline = edit 3 dicts + if/elif tiers + write an 800-line script.

**Protocol (`baselines/base.py`):**

```python
class BaselineAdapter(Protocol):
    name: str
    tier: Literal["conse", "cosine"]          # generative tier retired with LLaSA
    def setup(self, device) -> "State": ...    # load model + artifacts once
    # ConSE tier:
    def window_probs(self, ds, state, device) -> np.ndarray: ...      # (N,87) softmax over global labels
    # Cosine tier:
    def window_embeddings(self, ds, state, device) -> np.ndarray: ... # (N,D) L2-normed
    def encode_labels(self, L_D, state, device) -> np.ndarray: ...    # (L,D) text prototypes
```

Provide `ConSEAdapter` / `CosineAdapter` base classes so a concrete adapter overrides only `setup()` + its one tier method. A `@register` decorator appends to `REGISTRY` — **adding a baseline = drop a file in `baselines/` and import it**, no dict/driver edits.

**Example adapter (`baselines/crosshar.py`, ~80 LOC):**

```python
@register
class CrossHARAdapter(CosineAdapter):
    name, tier = "crosshar", "cosine"
    def setup(self, device):
        model = load_crosshar_model(CROSSHAR_CHECKPOINT, device)   # reused primitive
        return State(model=model, emb_dim=EMB_DIM)
    def window_embeddings(self, ds, state, device):
        raw = load_limubert_raw(ds)                                # BENCH_LIMU path
        return l2norm(extract_crosshar_embeddings(state.model, raw, device))
    def encode_labels(self, L_D, state, device):
        return get_sbert_encoder(device).encode(L_D)              # eval_v2 helper
```

**Generic driver (`run_baselines_v2.py`)** — one loop, all scoring from `eval_v2`:

```python
for name in args.baselines:
    A = REGISTRY[name]; state = A.setup(device)
    for ds in datasets:
        _, L_D, gt, subjects, keep = eval_v2.window_ground_truth(ds)
        if A.tier == "conse":
            probs = A.window_probs(ds, state, device)[keep]
            preds, info = eval_v2.conse_predict(probs, GLOBAL, L_D, encode=sbert)
        else:
            emb = A.window_embeddings(ds, state, device)[keep]
            preds = eval_v2.predict_from_similarity(emb @ A.encode_labels(L_D,state,device).T, L_D)
        results[ds] = score(gt, preds, subjects)
```

**Migration is near-mechanical:** `setup_*`→`adapter.setup()`, `*_probs`→`window_probs()`, `lanhar_predict`→`window_embeddings()+encode_labels()`. Extract CrossHAR's 5 reusable symbols (`load_crosshar_model`, `extract_crosshar_embeddings`, `TransformerClassifier`, `EMB_DIM`, `CROSSHAR_CHECKPOINT`) and LiMU-BERT's 5 (`load_limubert_model`, `GRUClassifier`, `EMB_DIM`, `normalize_for_limubert`, `reshape_and_merge`) into adapters; delete the ~700-LOC v1 remainder of each. **TSFM registers as a cosine adapter** over `evaluate_tsfm_v2`'s per-patch embeddings, unifying model + baselines under one driver.

**UniMTS + ssl-wearables** each slot in as a ~60–100-LOC adapter reusing upstream model code from its fetched clone — pick the tier, override `setup()` + one signal method.

**`auxiliary_repos/` handling:** keep gitignored (correct today). Add `scripts/fetch_baselines.sh` that:
- `git clone --depth 1` each upstream (strips the ~124M nested `.git` bloat),
- downloads only the needed checkpoints (`CrossHAR/saved/...model_masked_6_1.pt` 532K; LiMU-BERT `saved/` 18M) — NOT the regenerable `CrossHAR/dataset/` (927M) or `LIMU-BERT-Public/embed/` (2.0G),
- co-locates checkpoint-path constants inside each adapter so "where is the weight" lives with the loader.

This makes the "clean V2 repo" reproducible (currently impossible — no committed fetch script, only prose in `BASELINES_SETUP.md`).

---

## 5. .gitignore / repo-hygiene fixes

**Already correct (no action):** `training_output/**` (11G, 0 tracked), `test_output/` `.pt`/`.log`, `auxiliary_repos/`, `.serena/`, `paper-rebuttal/`, `*.pdf` global rule, `benchmark_data/processed/tsfm/` symlinks.

**Add to `.gitignore` + untrack existing:**
```
data/**/debug_*.png          # 48 files, ~10M tracked dev artifacts
```
- `git rm --cached figures/fig_dataset_discrepancy.pdf` — violates the repo's own `*.pdf` rule (only tracked PDF); the `.py` generator reproduces it.
- `git rm --cached` the 48 `data/*/debug_*.png` (keep `labels.json`/`manifest.json`).
- `git rm --cached` v1 tracked `test_output/` artifacts (`baseline_evaluation/{moment,llasa,lanhar}_evaluation.json`, `ablation_evaluation/*/tsfm_evaluation.json`, `harth_analysis/*.png`); keep `eval_v2/*`.
- `git rm --cached -r data/vtt_coniot/` (dropped dataset metadata + 4 debug PNGs).
- Remove `docs/baselines/results/*.json` (stale v1 duplicates, diverged from `test_output/` copies).

**Local disk de-bloat (not git, but reclaims ~4–5G):** prune `auxiliary_repos/CrossHAR/dataset/` (927M), `LIMU-BERT-Public/embed/` (2.0G), nested `.git` dirs (~124M) — all regenerable via `fetch_baselines.sh`.

**Broken hardcoded references to fix before shipping** (grep before release):
- `grep -rn 'code/tsfm'` → `run_baselines_only.sh`, `auto_eval_after_training.sh`.
- `grep -rn 'alxdofficial/tsfm'` → `setup_runpod.sh` (old name + `master`, should be `HALO`/`V2`).

Since `auxiliary_repos/`, `training_output/`, and `paper-rebuttal/` are already untracked, git history is NOT bloated by the multi-GB artifacts — no `filter-repo`/BFG surgery needed. Only the ~10M of tracked PNGs/PDF/JSONs warrant untracking, and those are small enough that a plain `git rm --cached` + commit suffices (history rewrite optional).

---

## 6. Sequenced cleanup plan

*Nothing is deleted yet — this sequences the actual work. Do it on a fresh `V2` clean-slate branch cut from current HEAD.*

**Phase 0 — Branch + safety net**
- Cut `V2` branch. Tag current HEAD as `v1-archive` so all v1 code/results remain retrievable without living in the tree.
- Run the full test suite to capture a green baseline before any change.

**Phase 1 — Zero-risk deletes (no import graph impact)**
- Untrack `debug_*.png`, the stray PDF, `data/vtt_coniot/`, stale v1 `test_output/` JSONs, `docs/baselines/results/*.json` (§5).
- Delete untracked one-offs: `scripts/run_wave2_eval.sh`, `scripts/debug_eval_smoke.py`.
- Delete fully-orphaned files: `val_scripts/.../eval_config.py`, `harth_analysis.py`, `datascripts/shared/validate_dataset.py`, `datascripts/debug_motionsense.py`.
- Move `model/` `__main__` demos into `tests/` (drop the ActionSense reference); delete the in-file blocks.
- Archive v1 docs (`RESULTS.md`, `EVALUATION_PROTOCOL.md`, `ablation_results.md`, `SCALING.md`) under `archive/`; add `fetch_baselines.sh`; fix hardcoded paths/URLs in `setup_runpod.sh`/`package_checkpoints.sh`.

**Phase 2 — Dataset-set pruning (mechanical, config-driven)**
- Prune `dataset_config.json` `zero_shot_datasets` to the 6 v2 test sets *(confirm-first)*; trim `export_raw.extract_subject()` branches, `preprocess_limubert.ACC_IN_G_UNITS`, docstrings.
- Delete/archive the 5 dropped converters (`actionsense/`, `vtt_coniot/`, `realdisp/`, `daphnet_fog/`, `usc_had/`).
- Move `process_motionsense.py` → `datascripts/motionsense/convert.py`; register in `setup_all_ts_datasets.py`. Consolidate validators into config-driven `verify_conversions.py`.
- Prune dropped-dataset label vocab in `label_groups.py` / `label_augmentation.py`.

**Phase 3 — Drop baselines + modularize (biggest LOC win)**
- Delete `evaluate_moment.py`, `evaluate_lanhar.py`, `evaluate_llasa.py`, `generate_lanhar_descriptions.py`.
- Build `baselines/` package (`base.py` Protocol + `crosshar.py` + `limubert.py`); refactor `evaluate_baselines_v2.py` → `run_baselines_v2.py` generic driver (§4). Register TSFM as a cosine adapter.
- Delete `auxiliary_repos/LanHAR/`, `LLaSA/` locally; prune `CrossHAR/dataset/`, `LIMU-BERT-Public/embed/`, nested `.git`.
- Add UniMTS + ssl-wearables adapters.

**Phase 4 — v1 eval scoring retirement (ordered — has a dependency chain)**
- Split `evaluate_tsfm.py`: extract `eval_common.py` (`extract_tsfm_embeddings`, `extract_tsfm_per_patch_embeddings`, `_forward_batch`, `compute_cosine_accuracy`, `load_raw_data`, `get_dataset_metadata` + FINETUNE_* constants) imported by `evaluate_tsfm_v2`, `figures/fig_embedding_umap.py`, `scripts/sweep_patch_sizes.py`. **Then** delete the v1 scoring/CLI block.
- Reduce `grouped_zero_shot.py` to `load_global_labels` (→ `labels_util.py`); delete its 5 dead fns + v1 group scorers.
- Trim `evaluation_metrics.py`: delete `compute_embedding_quality_metrics`; retire `compute_similarity`/`compute_semantic_recall` (**keep** `compute_group_accuracy*` — training needs them). Split `tests/test_similarity_computation.py` accordingly.
- Delete v1 runners (`run_all_evaluations.sh`, `run_baselines_only.sh`, `auto_eval_after_training.sh`, `generate_results_table.py`); replace with one `run_eval_v2.sh`. Port or delete `eval_ablations.sh` + `generate_ablation_results.py`.

**Phase 5 — In-file model/train dead code**
- `positional_encoding.py`: delete `ChannelSemanticEncoding` + `use_channel_encoding` plumbing → collapse `encoder.forward` 280–297.
- `config.py`: trim to `{small, small_deep}`, drop `use_channel_encoding`.
- `token_text_encoder.py`: delete `LearnableLabelEncoder`/`LabelAttentionPooling`.
- `semantic_alignment.py`: delete `get_attention_stats`; gate session-level head on `not per_patch_prediction` **behind a checkpoint-compat shim** (state_dict key change — validate `small` + all 18 headline checkpoints still load).
- `train.py`/`semantic_loss.py`: delete `get_attention_stats` wrapper; remove `SigLIPLoss`; reconcile `use_channel_encoding` metadata. Split `SemanticAlignmentModel` into `model_def.py`; empty the eager Stage-1 import in `__init__.py`.

**Phase 6 — Human-gated decisions (do NOT auto-execute)**
- **Stage-1**: archive vs delete `pretrain.py` + `losses.py` + coupled raw-loader/aug paths + Stage-1 half of `test_losses.py`.
- **GradCache**: drop vs gate `_train_epoch_gradcache` + `forward_cached` (depends on whether medium/large ship).
- **Multi-prototype**: keep vs delete `MultiPrototypeLabelPooling` + `_forward_multi_prototype`.
- **`channel_filter`** hook: adopt for v2 channel-subset eval vs delete.
- **`paper-rebuttal/`** relocation; **`benchmark_scaled_model.py`/`SCALING.md`** retention.

**Phase 7 — Docs + index refresh (lockstep, or agents get misrouted)**
- Rewrite `CLAUDE.md`, `README.md`, `docs/README.md`, `docs/ARCHITECTURE.md`, `docs/EXPERIMENTS.md` for V2: per-patch `small_deep` (8-layer, trained from scratch, label bank demoted), 16-dataset / 6-test set, new baseline set, v2 protocol/results as single source of truth. Remove `tools/` and `docs/ablations.md` misreferences. Collapse `MODEL_COMPARISON.md` + 4 `BASELINE_*.md` into one `BASELINE_ADAPTERS.md`.

**Validation gate after each phase:** re-run `tests/`, plus a `small_deep` checkpoint-load smoke test (§5 shim risk) and one end-to-end `evaluate_tsfm_v2` + `run_baselines_v2` dry run to confirm the eval import graph and adapters still resolve.