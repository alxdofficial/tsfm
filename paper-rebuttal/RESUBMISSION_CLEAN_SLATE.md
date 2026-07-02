# HALO → MobiCom'27 — Clean-Slate Resubmission Brief

*Assembled 2026-06-11 from a full audit of the rebuttal campaign (debug sweep, codebase audit, perf analysis, corrected re-runs, all 13 rebuttal branches, and the submitted paper text). This is the ground-truth handoff for restarting clean. The MobiCom'26 #1698 submission was **not accepted**; we resubmit to MobiCom'27 with **new experiments now permitted** (the CFP constraints that gagged the rebuttal no longer apply).*

> **The one-line story:** the single weakness all knowledgeable reviewers (A, C, D, E) led with — *"the model collapses on severe-OOD (HARTH/VTT), so the foundation-model claim is overstated"* — was, for HARTH, **a label-indexing bug in shared evaluation code**, not a real collapse. Corrected, HARTH is graceful degradation (~29%), not <3%. Fixing this honestly defuses the #1 reject driver — but it also means we must drop the "universal collapse = sensor-distribution shift" narrative and report some numbers where HALO is competitive rather than winning.

---

## PART A — What is actually wrong in the submitted paper

### A-tier: defects that change PUBLISHED numbers (must fix before resubmission)

| # | Defect | Affected published claim | Status in code |
|---|---|---|---|
| **A1** | **HARTH label-offset bug** in `get_window_labels` (`t=min(codes); labels-=t` never restored). HARTH is the *only* dataset with min code >0 (codes `{2..8,11}`, min=2), so every HARTH GT was shifted −2 and scored against the wrong activity name. Bug is in **shared GT code, copy-pasted into all 6 evaluators** → every model's HARTH column is wrong. | Table 5 (severe-OOD), HARTH columns in Tables 3/4, per-activity/confusion analysis, **and the entire EXP-P3 "gravity→alignment-inversion" story (retracted — it explained a non-existent collapse)** | **FIXED** `e9ffeb1` (`+t` restored in all 6 evaluators). Verified no-op on the 16 min=0 datasets. `figures/fig_embedding_umap.py:112` copy still buggy (figure only). |
| **A2** | **Cache-dependent split → train/val leakage.** Global-RNG state differed on cache-miss vs cache-hit; train and val built as separate instances → **~73% val/train session overlap**. | The reported **in-distribution val_acc ≈85%** is leakage-inflated and has not been re-derived. (Held-out zero-shot headline is unaffected — different split.) Also: split is random-session, **not subject-disjoint**. | **FIXED** (dedicated seeded `random.Random` per split, cache-independent) in `multi_dataset_loader.py`. Honest val number still needs regenerating. |
| **A3** | **LLaSA accelerometer not ÷9.8** — raw m/s² fed ~9.8× too large into a LiMU-BERT encoder → every LLaSA embedding OOD → LLaSA unfairly crippled, inflating HALO's relative lead. | LLaSA's column across the comparison tables. | **FIXED** (`/= 9.8`). ⚠️ See D1 — may now be over-applied to already-g-scaled hapt/harth. |
| **A4** | **LanHAR HARTH label shift un-restored** (same −t never added back; LLaSA's path *had* added it). | LanHAR's HARTH column only. | **FIXED** (`return window_labels + t`). |
| **A5** | **Table 7 native-rate "+11.4 pp" is STALE** — computed on an old 4-layer checkpoint; does **not** replicate on the deployed model. Corrected decomposition (EXP-F): native-rate **+2.56**, metadata **−3.27** (rich text *hurts*), combined **+0.41 pp**. | Table 7 fairness claim. | **OPEN** (paper fix). Safe replacement below. |
| **A6** | **Scaling table 46.0 vs deployed 42.0** — `docs/baselines/SCALING.md` / Table 8 Small-35M = 46.0, but no results JSON backs 46.0 (deployed JSON = 42.0). Table 8 is also a *different* aggregation (3 datasets @20 Hz) than Tables 3/4 (5 datasets, native), so the whole scaling table's provenance is unverified. | Table 8 (model-scaling), the "−8.2 pp Medium regression / −13.2 pp Tiny" deltas are computed off the unbacked 46.0. | **OPEN** — regenerate Table 8 from JSON-backed runs; standardize on 42.0. |
| **A7** | **Stale headline JSON hazard** — `test_output/baseline_evaluation/tsfm_evaluation.json` is actually the *no_text_aug ablation* (6–14 pp off). The real headline is `tsfm_evaluation_small_deep_v2.json`. | Tables 3/4 if the wrong file is cited. | **OPEN/advisory** — always cite the `small_deep_v2` JSON. |
| **A8** | **"42.0%" is the mean of 5 main datasets only**; all-7 mean ≈ 34.3% (post-HARTH-fix). Reviewer-derivable. | Headline number framing (D2/C4). | **OPEN** (state the convention explicitly). |
| **A9** | **Figure 2 plots 4 datasets, not 7** — text implies 7; Reviewer D is factually right. | Figure 2. | **OPEN** (regenerate with all 7). |

### The HARTH narrative reframe (the strategic core)

The submitted story — *"all embedding models collapse to <3% on HARTH ⇒ genuine sensor-distribution shift, not a HALO defect"* — is **not supported by corrected data**. Corrected HARTH ZS-open: HALO 29.1, CrossHAR 30.3, MOMENT 17.5, LanHAR 9.3 (LiMU-BERT run crashed). This is **graceful degradation, and HALO is ~#2 on HARTH accuracy** (CrossHAR edges it; HALO leads on F1, 12.4 vs 6.0). The honest, still-strong framing for the resubmission:
- **Two distinct severe-OOD modes:** HARTH = sensor-configuration shift → *graceful degradation for everyone* (9–30%); VTT-ConIoT = 50% novel labels → *genuine zero-shot floor for everyone* (HALO 1.3, all ≤8.3). The "collapse" claim is true **only for VTT**, and it is not HALO-specific.
- HALO **recovers on HARTH to 76.8% @10% labels** — the real deployment story.
- This *removes* the #1 reviewer attack at the cost of conceding HALO doesn't win HARTH zero-shot. Net positive.

---

## PART B — Corrected, trustworthy numbers

**Provenance:** HALO = checkpoint `small_deep_v2_4b3fdd6/best.pt` (epoch 186). Corrected runs 2026-06-05; scoring/fairness/OOD 2026-06-04. `verify_reproduction` confirms the pipeline reproduces the reference JSON to **|Δ| = 0.000 pp**.

### B1 — Severe-OOD, zero-shot open-set (%), CORRECTED (group-matched, 87-label)

| Dataset | HALO | MOMENT | CrossHAR | LanHAR | LiMU-BERT | Reading |
|---|---|---|---|---|---|---|
| **HARTH** (sensor shift) | 29.1 (F1 12.4) | 17.5 | **30.3** | 9.3 | ⚠️ crashed | graceful degradation; HALO #2 acc / #1 F1 |
| **VTT-ConIoT** (50% novel labels) | 1.3 | 1.6 | 0.7 | 8.3 | 3.4 | genuine floor — *for everyone*, not HALO-specific |

HALO corrected HARTH other cells: ZS-closed 30.4, 1%FT 62.0, **10%FT 76.8**.

### B2 — Exact-match vs group-match scoring, 5-main ZS averages (%)  *(EXP-X / EXP-P5, complete, PASS)*

| Model | open_exact | open_group | closed (exact=group) |
|---|---|---|---|
| **HALO** | **38.63** | **41.97** | 53.11 |
| MOMENT | 20.68 | 28.28 | 44.70 |
| LiMU-BERT | 17.43 | 21.69 | 33.14 |
| CrossHAR | 18.02 | 21.16 | 38.21 |
| LanHAR | 8.73 | 15.82 | 28.38 |

**Key:** HALO's *exact-match* open-set (38.63) beats every baseline's *group-match* (≤28.28). The lead **widens** under strict scoring (group +13.7 → exact +17.95 pp), and HALO ranks #1 in all 4 scoring cells (ranking is protocol-invariant). This is the honest answer to "group scoring inflates HALO" — and it lets us **correct the paper's false scoring sentence** (line 780: *"text-aligned models use exact string match"* — actually HALO's open-set is group-matched too; the corrected description is safe because HALO wins either way).

### B3 — Fairness 2×2, HALO ZS-open 5-main (%)  *(EXP-F, complete, PASS)*

| Condition | ZS-open |
|---|---|
| A — native rate + rich text (deployed) | 41.97 |
| B — 20 Hz + generic text (**baseline-equivalent handicap**) | **41.56** |
| C — 20 Hz + rich text | 39.41 |
| D — native rate + generic text | 45.24 |

Effects: native-rate **+2.56**, metadata **−3.27** (rich text slightly *hurts*), combined **+0.41**. **Safe story:** even fully handicapped (B = 41.56), HALO beats every baseline (next best MOMENT 28.28) → the win is architectural, not an input-format artifact. **Replaces the stale Table 7 +11.4 pp claim (A5).**

### B4 — Numbers we DON'T have (gaps to fill)

- **LiMU-BERT corrected HARTH** — run crashed (`ModuleNotFoundError: No module named 'models'`); the JSON entry is stale/malformed. Re-run needed.
- **TSFM-Tiny / TSFM-Medium corrected HARTH** — pending (needed if Table 8 scaling is kept).
- **Full per-dataset corrected baseline tables** (non-HARTH) — only the 5-main *averages* survived; `corrected_baselines.log` truncated. Per-dataset baseline JSONs exist in `test_output/baseline_evaluation/` but were not re-verified.
- **Multi-seed error bars** for the +13.7 pp lead — not produced (P6 not run).

---

## PART C — Experiments: done vs still-to-run, mapped to reviewers

*Now that new experiments are allowed, the rebuttal-era work becomes real paper content. Critically, the biggest-ticket reviewer asks were the ones **never actually run** — only design READMEs exist.*

| Reviewer ask | Experiment | State | Action for '27 |
|---|---|---|---|
| Severe-OOD collapse overclaimed (A1/C/E3) | HARTH label-bug fix + re-run | ✅ done (5 of 6 models) | re-run LiMU-BERT + Tiny/Medium; rewrite §6.2.3 |
| Group scoring inflates lead (E/C2) | EXP-X exact vs group | ✅ done | add exact-match column; correct scoring text |
| Ranking protocol-dependent (E) | EXP-P5 scoring sensitivity | ✅ done | report ranking-invariance |
| Unfair comparison: native rate + metadata (C/E1) | EXP-F fairness 2×2 | ✅ done (LanHAR-native unconfirmed) | replace Table 7 |
| **Queue conflicts w/ synonym objective (A2)** | **EXP-P1 queue ablation (none/hard_neg/semantic)** | **❌ NOT RUN — README only** | **run ×3 — A's exact ask; verify semantic mode stores group ids (D3)** |
| **No multi-seed / variance (E4)** | **EXP-P6 ≥3 seeds** | **❌ NOT RUN — README only** | **run — mean±std on the +13.7 pp lead** |
| **Ablations not fine-grained (C/D/E4)** | **EXP-P7 ~8 single-knob arms** | **❌ NOT RUN — README only** | **run — pooling, tokenizer, channel-indep, soft/hard, τ, SBERT** |
| **Open-vocab under-evidenced (E/D)** | **open-vocab eval (distractors/paraphrases/fine-grained)** | **❌ NOT RUN — only `__pycache__`** | **run — the brief's "98–133% / ≤0.2 pp" numbers have no saved artifacts** |
| Placement-unknown datasets (D) | train/test with vs without UCI-HAR | ❌ not covered | small run or written justification |
| Param tradeoff / more params (D) | scaling check | partial (Table 8 unverified, A6) | regenerate Table 8 from JSON |
| Novelty / "what's new" (B/C/D) | *framing, not an experiment* | see `NOVELTY_REVISED.md` | the dominant accept/reject axis — rewrite §1 |

> ⚠️ **Discrepancy to resolve:** `SUPERVISOR_MEETING_BRIEF.md` lists open-vocab (EXP-P2) as "✅ done" with specific numbers, but **no open-vocab artifacts exist in-repo** (only compiled pycache). Either results were lost (RunPod, never pulled) or recovered-from-Notion claims. Treat open-vocab as **must-rerun** until artifacts resurface.

---

## PART D — Latent paper-text-vs-code discrepancies (camera-ready honesty; none fixed)

These only surface on code/artifact release, but the resubmission's text should match the released model:
- **Stage-1 SSL pretraining is non-functional/unused** (`pretrain.py` calls a config that doesn't exist) — the deployed checkpoint is **alignment-only, trained from scratch**, not the "two-stage foundation model" the paper describes. (Reviewers already smelled this; reframe Stage-1 as optional/auxiliary.)
- Open-set is **group-scored for HALO too** (paper line 780 says exact) — corrected in B2.
- Conditioning is **gated `ChannelTextFusion`**, not additive `ChannelSemanticEncoding` (the additive path is dead, built with `=False`).
- Tokenizer kernel is **[5]**, not {3,5,7}; mask ratio **0.3**, not 0.5.
- Synonyms: **3.56 avg, hand-authored**, not "2.3 WordNet."
- Param count: paper "∼35M" vs actual ~29.3M (Medium 63–66.6M) — reconcile.
- Channel range stated as **3–45** (Tables/Fig 2) but text also says **51** (illustrative) — `NOVELTY_REVISED.md` uses 3–51; pick one.

---

## PART E — Clean codebase: branch + folder/disk plan

### Branches — the keeper and the deletable

**THE clean base = `rebuttal/fix/pre-ablation-bugs`.** It is the *only* origin of every genuine core fix (HARTH, split-leakage, LLaSA, LanHAR-HARTH, numpy-import, DataLoader hang, checkpoint corruption) plus two math-preserving perf wins. Fixes flowed one-way from here into the big experiment branches; nothing needs gathering from scattered branches.

**Safe to delete now** (0 commits ahead of master — fully contained in `master`):
- `alex2`, `tool-use-om`, `tool-use-om2`, `cleanup/codebase-refactor`, `eval-small-deep-v2`
- `rebuttal/experiment/verify-reproduction` (identical to master, 0/0)

**Keep until consolidated** (hold unique committed work the resubmission needs):
- `rebuttal/fix/pre-ablation-bugs` — **the base; keep.**
- `rebuttal/experiment/{exact-match-scoring, fairness-native-rate, scoring-sensitivity, open-vocab-eval}` — hold the committed corrected-result scripts/logs (100% paper-rebuttal/, no core code). **Consolidate their results onto the base before deleting.**
- `rebuttal/experiment/ood-failure-analysis` — built on **buggy** HARTH GT; its analysis is retracted. Keep only as provenance, then delete.
- `rebuttal/experiment/queue-ablation` — the **only** branch with a genuine new core feature: the semantic-aware memory queue (`b6190a5`, `memory_bank.py`/`semantic_loss.py`). **Cherry-pick `b6190a5` onto the base if we keep the queue ablation** (we should — it's Reviewer A's ask).
- `rebuttal/experiment/{multi-seed, fine-grained-ablations}` — fixes + ablation scaffolding only (no results). Keep scaffolding, low priority.
- `pod/*` (3) — pure code-only public mirrors, regenerable. Delete after the resubmission base is set.

> ⚠️ **Before deleting anything:** several corrected results/logs live as **untracked** working-tree files or on experiment branches only. Commit/consolidate the corrected JSONs, logs, and scripts onto the resubmission base first, or they're lost.

**Recommended consolidation:** branch `resubmit/base` off `rebuttal/fix/pre-ablation-bugs`; cherry-pick `b6190a5` (semantic queue); copy the corrected result logs from the experiment branches into `paper-rebuttal/experiments/`; commit; verify; *then* prune.

### Folders / disk

Master's **tracked** folders are all live (model/, training_scripts/, val_scripts/, datasets/, datascripts/, benchmark_data/, docs/, figures/, scripts/, tests/, data/ manifests, test_output/). None are dead code — there's no master-folder to delete. The real cleanup is **disk** (all gitignored):

| Folder | Size | Action |
|---|---|---|
| `training_output/` | **56 GB** | biggest win — prune old run dirs; **keep `small_deep_v2_4b3fdd6/best.pt`** (deployed) + any checkpoint needed for re-runs |
| `data/` | 12 GB | keep — needed for re-runs |
| `auxiliary_repos/` | 3.7 GB | re-clonable baseline repos; keep only if re-running baselines locally |
| `benchmark_data/` | 2.6 GB | keep — needed for HARTH re-runs |
| `test_output/` | 931 MB | keep corrected `*_small_deep_v2.json` + corrected baseline JSONs; prune stale |

---

## PART F — Recommended "start clean" sequence

1. **Lock ground truth.** Adopt this brief's corrected numbers (Part B) as canonical. Confirm `docs/baselines/RESULTS.md` (updated in `64ae0b5`) matches.
2. **Consolidate the base.** Create `resubmit/base` off `rebuttal/fix/pre-ablation-bugs`; cherry-pick the semantic-queue feature; pull in the corrected result logs; commit so nothing important is untracked.
3. **Close the result gaps (re-runs):** LiMU-BERT corrected HARTH; TSFM-Tiny/Medium HARTH; regenerate Table 8 from JSON; re-derive the honest (leakage-free) val accuracy.
4. **Run the never-run experiments** (now allowed, and they're the top reviewer asks): EXP-P1 queue (A), EXP-P6 multi-seed (E), EXP-P7 fine-grained (C/D/E), open-vocab (E/D). Verify the semantic-queue actually differs from hard_neg before trusting P1.
5. **Rewrite, on honest numbers:** §1 novelty (per `NOVELTY_REVISED.md` — the main axis); §6.2.3 two-mode severe-OOD; scoring description + exact-match column; Table 7 → fairness 2×2; Figure 2 with 7 datasets; reframe Stage-1 as auxiliary; fix the Part-D method-text constants.
6. **Prune** the safe stale branches + pod mirrors, and reclaim ~56 GB from `training_output/`.

> **Open framing decisions for the authors (not code):** (a) HARTH — adopt corrected 29.1% (recommended for a clean resubmission) vs keep the old framing; the MEMORY note's "<3% / never cite 29.1%" applied only to the gagged rebuttal, not this resubmission. (b) Whether to keep the model-scaling story at all given Table 8's unverified provenance. (c) How hard to lean on the (surprising) "rich metadata slightly hurts" fairness result.
