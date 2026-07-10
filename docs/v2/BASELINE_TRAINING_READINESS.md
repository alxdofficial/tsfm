# Baseline Training Readiness — steps to a fair full comparison

**Status:** NOT ready to run the full baseline table. 2 of 6 baselines exist; the
2 that exist have blocking data/staleness issues; the other 4 are unbuilt.

_Authored 2026-07-10 after the Opus full-sweep (`docs/v2/full_sweep_findings_opus.md`).
Every claim below was re-verified directly against the code/data, not taken from
the agent report — the sweep mis-framed several of these (noted inline)._

Final intended baseline set (settled earlier):
- **Built:** CrossHAR, LiMU-BERT (ConSE tier).
- **Planned:** UniMTS (cosine tier), ssl-wearables (ConSE tier), NormWear (bespoke
  L1/MSiTF adapter), DeepConvLSTM (from-scratch supervised floor, FS-1%/10% only).

Data-flow reminder (both fixes below regenerate down this chain):
```
data/<ds>/sessions   (converter, deterministic)
   -> export_raw.py           -> benchmark_data/raw/<ds>/subject_*.csv
        -> preprocess_tsfm_eval.py   -> processed/tsfm_eval/<ds>/   (HALO reads this, native rate)
        -> preprocess_limubert.py    -> processed/limubert/<ds>/    (ConSE baselines read this, 20 Hz)
```

---

## STEP 1 — Cross-cutting data fixes (block *every* model; do first)

These corrupt the eval for all models, so they must land before any baseline (or
HALO) is scored. **Raw data for both is present locally**, so both are fixable now.

### 1a. HARTH subject S006 recorded at 100 Hz, stored as 50 Hz (2× time-warp)
- **Verified reality (differs from sweep):** the converter is **already correct** —
  `datascripts/harth/convert.py` has `_infer_sample_rate` (line 122) and
  `_resample_to_target_rate` (line 133); I confirmed it infers **100.00 Hz for
  S006** and the resample guard triggers (`abs(native-50)=50 > 0.5`). The sweep's
  claim that it "hard-codes 50 Hz with no per-file detection" is FALSE.
- **The bug is stale on-disk data**, not code: the committed `data/harth/sessions`
  was converted before that logic existed. Proof: converted/raw sample-count ratio
  is identical for S006 (2.409) and the true-50 Hz subjects S008 (2.361) / S009
  (2.451). If S006 had been downsampled its ratio would be ~half. Its on-disk
  `timestamp_sec` reads 0.02 s (fabricated 50 Hz) while holding the full 100 Hz
  sample count → every 6 s window really spans 3 s of motion, and the 50→20 Hz
  eval resample under-samples it 2×.
- **Fix:** re-run `python datascripts/harth/convert.py` (deterministic, seed=42+subject).
  S006 will resample 100→50 Hz (row count ~halves); all other subjects unchanged.
- **Then regenerate downstream:** `export_raw.py --datasets harth` →
  `preprocess_tsfm_eval.py --datasets harth` (HALO copy) +
  `preprocess_limubert.py --datasets harth` (baseline copy). Both current
  `tsfm_eval/harth` (Jul 3) and `limubert/harth` inherit the corruption.
- **Affects:** HALO harth row + all baseline harth rows (harth is a headline test set).

### 1b. mobiact baseline (limubert) processed copy has a stale label vocabulary
- **Verified reality (differs from sweep):** the corruption is real but **NOT silent**.
  `processed/limubert/mobiact/{label_20_120.npy,mapping.json}` are Feb-14 and use the
  OLD vocab (`fall_backward_knees→2`), while `data/mobiact/labels.json`, the
  `eval_v2` label config, AND HALO's own `tsfm_eval/mobiact` copy (Jul 3) all use
  the CORRECTED vocab (`fall_forward_knees→4`, `fall_backward_sitting→2`, ...).
- **`base.load_gt` already guards this** (base.py:88-100): it compares
  `mapping.json.activity_to_idx` to the current `eval_v2 idx_to_label` and
  **raises `ValueError`** on mismatch. So the ConSE baselines currently **cannot be
  scored on mobiact at all** — they hard-fail, they do not silently produce wrong
  numbers. HALO's mobiact eval is already correct.
- **Fix (no reconvert needed — converter + eval already agree):**
  `export_raw.py --datasets mobiact` → `preprocess_limubert.py --datasets mobiact`
  to refresh the stale baseline copy so `mapping.json` matches the current vocab and
  the guard passes.
- **Sanity after:** the guard in `base.load_gt` passing for mobiact is itself the
  regression test. (The other 5 test sets already MATCH — verified.)

**Step-1 exit check:** `base.load_gt` succeeds for all 6 test sets; harth S006
converted rows ≈ half the prior count; `tsfm_eval/harth` metadata still 50 Hz but
now genuinely resampled.

---

## STEP 2 — Decide HALO's final training config (do before spending GPU on baselines)

**User decision required — not started.** The sweep found ~14 pp of HALO gains from
config changes verified against `hyperparameters.json`
(`use_memory_bank=true`, `use_mean_pooling=false`, `loss_type=infonce`,
`feature_extractor_type=spectral_temporal`). If HALO is going to be retrained with
any of these changed, baselines should be compared against the *final* HALO, so
settle this first. Candidate changes (verify each before adopting):
- `USE_MEMORY_BANK=0` (repo ablation claims +~8 pp; needs independent confirm of the
  ablation CSV + the soft-target-zeroing mechanism in semantic_loss.py).
- `use_mean_pooling=True` (repo ablation claims +5.8 pp; confirm on the 7/7 table).
- `loss_type=siglip` (already plumbed; SigLIP small-batch result).
- Tokenizer cutover to the filterbank (removes rate-inconsistent spectral path).
- Eval-only: de-underscore labels + prompt ensembling; center/whiten label
  prototypes + inverted-softmax.

These are **out of scope for "get ready to train baselines"** but gate the order of
operations. Tracked separately; do not retrain baselines until HALO config is frozen.

---

## STEP 3 — Build the 4 missing adapters + refresh the 2 stale ones

### 3a. Missing adapters (no file exists in `baselines/`)
| Baseline | Tier | Weights present? | Key build requirements (from sweep, to re-verify at build) |
|---|---|---|---|
| **DeepConvLSTM** | from-scratch FS-only | N/A (trained from scratch) | Ordóñez & Roggen 2016 recipe; min-max [0,1] norm (NOT z-score); 20 Hz/120-ts/6-ch limubert arrays; FS-1%/10% via `eval_v2.subject_disjoint_split(seed=3431)`; report full-shot. **`references/baselines/deepconvlstm/paper.pdf` is missing** — fetch it (open-access MDPI). Do this one FIRST (no weights, no network, simplest). |
| **ssl-wearables** | ConSE | ✅ **harnet cached** at `~/.cache/torch/hub/OxWearables_ssl-wearables_main` | 30 Hz, **g-units, gravity-present**, 3-ch (accel-only), (N,3,300) from RAW (not the 20 Hz m/s² limubert copy); freeze trunk, train head only; reconcile 10 s/300-ts window vs the 6 s ConSE grid (open question — see M-8). |
| **UniMTS** | cosine | ❌ HuggingFace `xiyuanz/UniMTS` not downloaded | accel-only 3-ch (released ckpt); per-dataset unit map; `--joint_list` SMPL-joint mapping per placement; first-10 s/200-ts wrap-pad window; identical de-underscored label strings as HALO. |
| **NormWear** | bespoke (NOT plain cosine) | ❌ backbone + MSiTF ckpt not downloaded | Needs `preprocess_normwear.py` (65 Hz resample + full preproc); MSiTF alignment ckpt + query-conditioned fusion; Clinical-TinyLlama label encoder; L1 scoring (driver hardwires cosine → needs a distance branch); reuse NormWear's shipped `activity` template (do NOT invent a prompt — M-9). |

### 3b. Stale built adapters — re-pretrain on the 11-dataset corpus
- **CrossHAR + LiMU-BERT** were pretrained Feb on the **10-dataset** corpus (no
  capture24). For the "same corpus" parity claim both must be re-pretrained on the
  current 11-dataset `train_datasets` (capture24 included, with its cap). Also
  delete the stale `test_output/baseline_evaluation/{crosshar,limubert}_zs_*.pt`
  caches so heads re-fit on refreshed embeddings.
- **Doc hygiene (HIGH regressions from sweep, cheap):** delete stale
  `test_output/baseline_evaluation/crosshar_evaluation.json`; fix
  `BASELINES_SETUP.md` + `README.md` to point at `run_baselines_v2.py` (not the
  gated legacy `evaluate_crosshar.py`); finish the NormWear/UniMTS doc edits
  (`baseline_flexibility.md` prose, `CROSSCHECK.md:161`).

### 3c. Ordering
1. **DeepConvLSTM first** — no weights, no network, exercises the from-scratch FS path.
2. **ssl-wearables** — weights already cached locally.
3. **UniMTS** — needs a HuggingFace download.
4. **NormWear** — most work (bespoke adapter + new preprocessing + LLM label encoder).
5. Re-pretrain CrossHAR + LiMU-BERT (GPU) once HALO config (Step 2) is frozen.

**Global exit check:** `run_baselines_v2.py --baselines <all 6>` runs end-to-end on
all 6 test sets without the `load_gt` guard firing, and `assemble_v2_table.py`
produces a full table with no unannotated partial averages.

---

## Progress log
- [x] 1a HARTH S006 reconvert + downstream regen — **DONE 2026-07-10.** Reconverted
  (S006 2548→1290 sessions, genuinely 50 Hz, ratio 1.14 vs ~2.4 for true-50Hz
  subjects); added `shutil.rmtree(sessions/)` to the converter (cleared 1366 orphans;
  on-disk==labels.json==37,941); regenerated `raw/harth`, `tsfm_eval/harth`
  (50 Hz, 49,713 windows), `limubert/harth`. 29 eval tests pass.
- [x] 1b mobiact limubert copy refresh — **DONE 2026-07-10.** Re-ran export_raw +
  preprocess_limubert; refreshed `limubert/mobiact/mapping.json` now has the
  corrected fall vocab (`fall_forward_knees→4`). **All 6 test sets now pass the
  `base.load_gt` guard** (mobiact was previously crashing).
- [ ] 2  HALO config decision (user)
- [ ] 3a-DeepConvLSTM adapter + paper.pdf  ← in progress
- [ ] 3a-ssl-wearables adapter
- [ ] 3a-UniMTS adapter
- [ ] 3a-NormWear adapter + preprocess_normwear.py
- [ ] 3b CrossHAR + LiMU-BERT re-pretrain + doc hygiene
