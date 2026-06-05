# HALO Eval + Training Performance Analysis

# HALO Eval + Training Performance Analysis — Ranked Synthesis

Hardware budget to exploit: **~18 GB free VRAM** and **24 CPU threads**. The single dominant cost is the MOMENT SVM-RBF inference (CPU, single-threaded libsvm, GPU idle). Most other wins are removing redundant double-passes, vectorizing per-sample Python loops, and raising conservative batch sizes.

Verified against source: the two `predict_svm_global` calls (evaluate_moment.py:754 open-set + :763 closed-set) recompute the full RBF kernel twice, and `decision_function` (line 324) already subsumes the open-set `predict` (line 320). The per-sample vote loop (evaluate_tsfm.py:666-669) and the double embedding extraction over the same `raw_data` (evaluate_tsfm.py:974 + :984) are confirmed.

---

## 1. EVAL SPEEDUPS (biggest first)

### E1. MOMENT SVM-RBF kernel computed TWICE per test set, single-threaded, GPU idle — **THE dominant eval cost**
- `val_scripts/human_activity_recognition/evaluate_moment.py:306-341, 754, 763`
- **Slow:** Each test dataset scores via `predict` (open-set) then `decision_function` (closed-set); each recomputes the full dense `N_test x N_SV` RBF Gram matrix against 9,162 support vectors of dim 6,144. HARTH = 47,330 windows -> ~5.3 TFLOP per pass, done twice (~10.6 TFLOP), all in single-threaded libsvm C. The 4090 is idle. This dominates the ~50 min HARTH run.
- **Root cause:** sklearn SVC inference is single-threaded libsvm (`n_jobs=-1` at line 287 only parallelizes GridSearchCV fit, not predict). `decision_function` already contains everything `predict` needs, so the open-set pass is redundant. RBF with 9,162 SVs x 6,144-d is a huge dense GEMM + elementwise `exp` with no BLAS multithreading or GPU.
- **Fix (two tiers):**
  - *Interim, low effort, zero math change:* call `decision_function` ONCE, cache the `(N, n_ovo)` scores, derive open-set (OvO vote argmax) AND closed-set (masked argmax) from that one matrix. Replicate libsvm's OvO vote-counting to keep numbers identical. (See E2 — this is the same fix.)
  - *Full, high effort:* recompute the kernel ONCE on GPU — move `support_vectors_`, `dual_coef_`, `intercept_`, `gamma` to CUDA; compute `K = exp(-gamma * cdist(test_emb, SV)^2)` in fp16/bf16, tiled over test rows (47330x9162 fp16 ~0.8 GB, well within 18 GB); reconstruct OvO decision via `dual_coef_ @ K-blocks + intercept`. Collapses 2 CPU passes into 1 GPU pass.
- **Speedup:** ~2x from the interim dedup alone; **10-50x** for the GPU reimplementation (HARTH SVM scoring: tens of minutes -> seconds-to-1-min). The ~50 min HARTH run becomes a few minutes.
- **Effort:** low (interim) / high (GPU). **Risk:** medium. **Binding:** CPU-bound today -> GPU-bound after fix.

### E2. Redundant open-set `predict()` pass — pure waste (subset of E1, but the cheap half)
- `val_scripts/human_activity_recognition/evaluate_moment.py:752-767`
- **Slow:** Lines 754 and 763 each run a full single-threaded RBF kernel evaluation over the same `test_emb` against the same 9,162 SVs. Open-set prediction is exactly recoverable from the closed-set `decision_function` scores.
- **Fix:** Call `decision_function` once, store scores, derive open-set by replicating libsvm OvO voting argmax, closed-set by masking. Small refactor of `predict_svm_global` + the two call sites; no GPU work needed.
- **Speedup:** ~2x on the SVM scoring stage (the part that dominates HARTH), **zero accuracy change**.
- **Effort:** low. **Risk:** medium (must replicate OvO voting exactly). **Binding:** CPU-bound. **— This is the single highest impact/effort ratio item in the whole report.**

### E3. TSFM per-patch majority-vote: Python loop over N windows with per-sample GPU sync + inner vote-count loop
- `val_scripts/human_activity_recognition/evaluate_tsfm.py:648-669`
- **Slow:** `for i in range(N)` over 47,330 windows; per window: CPU slice -> `.to(device)` (H2D per sample), tiny matmul, argmax, then `for v in votes: vote_counts[v] += 1` (scalar GPU index-increment per patch), then `.item()` (D2H sync). Called 2x/dataset (open + closed). For N=47k x ~30 patches that is >1.4M tiny ops + 47k syncs, doubled. Latency-bound, GPU mostly waiting.
- **Fix:** Move padded `(N,P,D)` patch tensor to GPU once; `sims = einsum('npd,ld->npl', ...)` in N-chunks (e.g. 4096 windows); per-patch argmax -> `(N,P)` votes, mask padding, aggregate with `scatter_add_`/`bincount` over L (one-hot scatter then argmax); `.cpu()` final `(N,)` once. Replace inner loop entirely.
- **Speedup:** **10-100x** for this function (latency-bound -> compute-bound).
- **Effort:** medium. **Risk:** low. **Binding:** latency/launch-bound -> fix makes it GPU-bound (bigger chunks help).

### E4. TSFM embedding extraction: conservative `batch_size=32` + encoder run TWICE over the full dataset
- `val_scripts/human_activity_recognition/evaluate_tsfm.py:77, 186-242, 245-320, 973-990`
- **Slow:** `TSFM_BATCH_SIZE=32` for both extractors. HARTH = ~1,480 batches; both `extract_tsfm_embeddings` (session pool) and `extract_tsfm_per_patch_embeddings` run back-to-back over the SAME `raw_data` (verified lines 974 + 984), so the encoder runs over all data twice. With d=384 and 18 GB free, BS=32 leaves the GPU heavily underutilized.
- **Fix:** (1) Derive the session-level mean-pooled embedding by masked-mean-pooling the per-patch embeddings -> encoder runs ONCE. (2) Raise extraction batch to 256-512 (no_grad + autocast, modest memory). (3) `pin_memory`/`non_blocking=True`.
- **Speedup:** ~2x from de-duplicating + 3-8x from larger batches = **~5-10x** on extraction.
- **Effort:** medium. **Risk:** medium. **Binding:** GPU-bound (bigger batches help) + redundant-work removal.

### E5. TSFM supervised fine-tune: deepcopy full model + 20 epochs from scratch, run TWICE per dataset (1% & 10%) — largest single eval component
- `val_scripts/human_activity_recognition/evaluate_tsfm.py:695-875, 753, 776, 816, 1031-1049`
- **Slow:** Called 2x/dataset. Each: `copy.deepcopy(model)`, AdamW over ALL params, up to 20 epochs full backprop through the encoder + full val pass/epoch + `copy.deepcopy(state_dict)` on every val improvement. `_forward_batch` rebuilds metadata lists every batch (lines 443-450). DataLoaders are `num_workers=0`, no `pin_memory`.
- **Fix (independent):** (1) raise `FINETUNE_BATCH_SIZE` (GradScaler already present); (2) `num_workers>0` + `pin_memory=True` + `non_blocking`; (3) hoist constant `channel_descs`/`sampling_rates`/`patch_sizes` out of `_forward_batch`; (4) store only trainable subset instead of full `state_dict` deepcopy; (5) **freeze backbone, fine-tune only semantic head / last block** (linear-probe-style) to slash backprop — matches the few-shot intent.
- **Speedup:** 2-5x from batch/dataloader/hoisting; **up to 5-10x more** if backbone frozen. This is the dominant eval cost, so absolute savings are large.
- **Effort:** medium. **Risk:** medium (freezing may shift few-shot numbers — validate). **Binding:** GPU-bound (batches) + parallelizable (dataloader) + redundant-work removal.

### E6. CrossHAR + LiMU-BERT zero-shot: classifier/GRU run TWICE (open-set then closed-set) over identical inputs
- `val_scripts/human_activity_recognition/evaluate_crosshar.py:938-947` (predict at 546-577); `evaluate_limubert.py:814, 824` (predict_gru_global at 407-438)
- **Slow:** Open-set calls `predict_*` then closed-set calls it AGAIN with the same input + a post-hoc logit mask. Full forward over all N test windows (CrossHAR) / ~6N sub-windows (LiMU-BERT GRU, sequential, ~280K rows for HARTH) recomputed for nothing.
- **Fix:** Single predict returns raw logits `(N, 87)` once; open-set = `logits.argmax(1)`, closed-set = `logits.masked_fill(~mask, -inf).argmax(1)` on cached logits. No second forward.
- **Speedup:** **~2x** per model on the zero-shot inference portion (GRU's sequential recurrence makes the LiMU-BERT win especially valuable).
- **Effort:** low. **Risk:** low. **Binding:** GPU-bound work, removed entirely.

### E7. LiMU-BERT `majority_vote_subwindows`: quadratic per-window mask scan
- `val_scripts/human_activity_recognition/evaluate_limubert.py:250-262`
- **Slow:** `for w in range(n_windows): mask = parent_ids == w` scans the full `parent_ids` array per window -> O(N*M) = O(6N^2). On HARTH (47K windows, ~280K sub-windows) that is billions of comparisons.
- **Fix:** 2D vote histogram: `H = np.zeros((n_windows, n_classes)); np.add.at(H, (parent_ids, sub_preds), 1); window_preds = H.argmax(1)`. O(M+N*C). Handle empty windows via zero-total-row detection.
- **Speedup:** **50-500x** on HARTH (quadratic -> linear).
- **Effort:** low. **Risk:** low. **Binding:** CPU-bound (vectorization fixes it; VRAM irrelevant).

### E8. CrossHAR: CPU InstanceNorm over whole dataset + numpy device round-trips
- `val_scripts/human_activity_recognition/evaluate_crosshar.py:926, 332-366, 323-329`
- **Slow:** `apply_instance_norm` runs `InstanceNorm1d` on CPU (single-threaded) over the full `(N,120,72)` array with numpy transposes; embeddings materialized to CPU numpy then re-uploaded by the classifier — a full-dataset D2H/H2D that is immediately undone.
- **Fix:** Do InstanceNorm on GPU per batch; keep embeddings on GPU into the classifier; or fuse extraction + classification so embeddings never leave the GPU.
- **Speedup:** **1.3-2x** on CrossHAR eval.
- **Effort:** medium. **Risk:** low. **Binding:** CPU-bound norm -> GPU; removes redundant transfers.

### E9. TSFM per-patch embeddings round-trip device->host->device (N tiny H2D copies)
- `val_scripts/human_activity_recognition/evaluate_tsfm.py:304-320, 657-658`
- **Slow:** Extractor moves every batch to CPU and returns a big `(N, max_P, D)` host tensor; the vote loop then re-uploads slices one sample at a time. Full round-trip, second leg done N times in tiny pieces.
- **Fix:** Keep per-patch embeddings on GPU (or stream N-chunks), run vectorized voting on-device. Folds into E3.
- **Speedup:** part of E3's 10-100x; independently removes N tiny H2D copies.
- **Effort:** medium. **Risk:** low. **Binding:** latency-bound transfers.

### E10. Per-sample scoring/label helpers across all eval scripts (vectorize the Python loops)
Grouped because each is individually small (seconds, not minutes) but trivial to fix and collectively remove hundreds of thousands of Python iterations:
- `grouped_zero_shot.py:248-293` `score_with_groups` — per-sample dict lookups + string lists, called 2x/dataset (~95K iters for HARTH). Fix: precompute int `test_local->group` and `global->group` lookup arrays, vectorize with fancy indexing, pass int group ids to sklearn. **10-100x** on the helper.
- `grouped_zero_shot.py:48-73, 199-245, 264-293` — `get_label_to_group_mapping()` rebuilt per call; memoize with `lru_cache`; `map_local_to_global_labels` -> array indexing. **5-20x.**
- `evaluate_moment.py:192-200`, `evaluate_tsfm.py:347-357`, `evaluate_limubert.py:225-234` (dup in `evaluate_crosshar.py:380-387`) — `get_window_labels` per-row `np.bincount().argmax()` over 47K windows. Fix: one-hot/`scipy.stats.mode(axis=1)` vectorized mode. **10-100x** on the helper.
- `evaluate_limubert.py:209-222` `reshape_and_merge` per-row `np.unique`. Fix: `same = (labels == labels[:,:1]).all(axis=1)`. **20-50x.**
- `evaluation_metrics.py:109-126` `compute_semantic_recall` triple-nested Python loop. Fix: integer group ids + `corpus_group_ids[top_k_indices]` compare. **10-30x.**
- `evaluation_metrics.py:176-219` `compute_embedding_quality_metrics` — `np.array(class_labels)` rebuilt in loop + O(n^2) intra-class + O(C^2) centroid loop. Fix: integer-code once, grouped reduction for centroids, closed-form intra-class tightness `(||sum||^2 - n)/(n(n-1))`. **3-20x.**
- **Binding for all of E10:** CPU-bound — vectorization fixes them, VRAM does not help. **Effort:** low (medium for embedding-quality). **Risk:** low.

### E11. MOMENT embedding extraction: fp32 (no autocast/bf16), conservative batch
- `val_scripts/human_activity_recognition/evaluate_moment.py:120-177, 61`
- **Slow:** MOMENT-1-large (~340M) forwarded in fp32; each 128-window batch expands to 768 univariate `(1,512)` series; `MOMENT_BATCH_SIZE=128` conservative; per-batch `.cpu().numpy()` syncs.
- **Fix:** `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` around the forward; raise batch to 256 under bf16 (monitor VRAM); accumulate on-GPU, transfer once.
- **Speedup:** ~1.5-2x on embedding extraction; compounds across 10 test + 10 train sets. (Secondary to the SVM, but bf16 precision is ample feeding an SVM.)
- **Effort:** low. **Risk:** low. **Binding:** GPU-bound (bigger batches help).

### E12. MOMENT SVM fit: GridSearchCV 9 C x 5 folds on 10k x 6144 RBF — cached, so one-time
- `val_scripts/human_activity_recognition/evaluate_moment.py:264-303, 713-726`
- **Slow:** 45 single-threaded libsvm fits when `moment_zs_svm.pkl` absent; high C (up to 10000) yields the 9,162-SV explosion that makes E1/E2 slow downstream.
- **Fix:** Mostly mitigated by joblib cache (one-time). To shrink the SV set and cut ALL downstream predict cost: lower C ceiling or coarse pre-search; **PCA/random-project 6144-d -> 256-512 dims** before the SVM (~12-24x on the d factor, compounds with E1); or GPU SVM (thundersvm/cuML).
- **Speedup:** ~0 on cached repeat; PCA-to-512 alone ~12x on the per-predict d factor (compounds with E1).
- **Effort:** medium. **Risk:** medium. **Binding:** one-time fit; PCA is the lever for inference.

---

## 2. TRAINING SPEEDUPS (for the ablation campaign)

### T1. One `pd.read_parquet()` per `__getitem__` — parquet open/metadata overhead dominates epoch time
- `datasets/imu_pretraining_dataset/multi_dataset_loader.py:368`
- **Slow:** Every sample reads its own `data.parquet`. A HARTH session is 56 rows x 6 channels yet `pd.read_parquet` takes ~30 ms cold — almost all footer/schema/Arrow->pandas overhead, not data. With 36,711 sessions, one pass is ~6-18 min of pure I/O/parse per epoch.
- **Fix:** Pre-pack sessions into a memory-mappable store read once (single `.npy`/`.npz`/Feather/LMDB keyed by session_id), so `__getitem__` is a slice not a file open. Simplest interim: in-memory LRU/dict cache of decoded session arrays per worker (whole HARTH set is a few hundred MB; `persistent_workers` survives epochs). Read drops ~30 ms -> <0.1 ms.
- **Speedup:** **5-30x** on data loading; **3-10x end-to-end** for a data-bound epoch.
- **Effort:** medium. **Risk:** medium. **Binding:** CPU/IO-bound + parallelizable (workers). **— Top training priority.**

### T2. Per-`__getitem__` recompute of channel grouping / regex / IMU-pattern scan (dataset-constant)
- `datasets/imu_pretraining_dataset/multi_dataset_loader.py:373-407, 436-450`
- **Slow:** Every sample recomputes IMU-pattern membership, `re.compile` (recompiled per call at line 53), `select_channel_groups`, channel filter, and base descriptions — all identical for a fixed dataset (channel aug disabled), repeated 36k/epoch.
- **Fix:** Precompute once per dataset in `__init__` (or lazy dict cache): filtered channel list, groups, deterministic `selected_channels` order, sampling rate, base descriptions. Hoist axis regex to module-level `re.compile`.
- **Speedup:** **1.2-2x** on CPU `__getitem__` (larger once T1 lands and this becomes the bottleneck).
- **Effort:** low. **Risk:** low. **Binding:** CPU-bound + parallelizable.

### T3. Frozen-text cache thrashes — label-aug produces unbounded distinct strings -> real transformer forward most steps
- `model/token_text_encoder.py:62-118`; `datasets/imu_pretraining_dataset/multi_dataset_loader.py:466-476` (augment_label rate=0.8)
- **Slow:** `TokenTextEncoder.encode()` runs the frozen MiniLM/MPNet on every uncached string. With `use_text_augmentation=True` at 0.8 rate (synonyms + templates), the exact-string cache misses constantly -> ~25+ fresh transformer forwards/step the cache was meant to eliminate. Worse, labels encoded TWICE/step (learnable `encode` + frozen `encode_frozen`).
- **Fix:** (1) Compute `encode_frozen()` on the canonical/base label (stable + cache-hot, better soft targets). (2) Batch-encode `encode` + `encode_frozen` in one tokenizer+transformer pass (shared backbone, same texts). (3) Pre-warm cache by encoding the finite synonym/template vocabulary at startup (it is enumerable). (4) Cap text-aug cardinality or precompute per-(dataset,label,template) embeddings off the hot path.
- **Speedup:** **1.2-1.5x** epoch (more for MPNet contrastive model).
- **Effort:** medium. **Risk:** medium. **Binding:** GPU-bound work, but the win is eliminating redundant forwards (work removal).

### T4. preprocess_imu_data: unfold + `F.interpolate` + z-score per sample on CPU in workers
- `model/preprocessing.py:58-63, 105-118, 159-168` (called at `multi_dataset_loader.py:511`); forced `.cpu()` at `preprocessing.py:42, 98`
- **Slow:** Per session, create_patches/interpolate/normalize run individually-vectorized but at sample granularity on CPU; per-call dispatch overhead paid 36k times while VRAM sits idle. The forced `.cpu()` actively blocks GPU batching.
- **Fix:** Move interpolate + z-score out of `__getitem__` to once-per-batch on GPU after collate (collate already pads to `(B, max_patches, T, C)`); run under autocast. Remove the hard `.cpu()`. `create_patches` math is already efficient — only its location is wrong.
- **Speedup:** **1.3-2x** on preprocessing once T1 is fixed; offloads CPU to idle GPU.
- **Effort:** medium. **Risk:** medium. **Binding:** CPU-bound -> GPU-bound (uses idle VRAM).

### T5. Label texts re-encoded 2-3x per optimizer window
- `training_scripts/human_activity_recognition/semantic_alignment_train.py:875-879` (standard 2x/step); `:1151-1152` + `:1240` (GradCache 2x Phase-1 + 1x Phase-3)
- **Slow:** `encode` + `encode_frozen` both tokenize+forward the same backbone on the same strings; GradCache Phase 3 re-encodes text already produced in Phase 1.
- **Fix:** Run frozen backbone ONCE per unique string set per window; derive learnable-pooled and frozen mean-pooled from the shared token tensor. In GradCache, cache Phase-1 text embeddings (autograd leaves) and reuse in Phase 3.
- **Speedup:** **1.1-1.4x** text-bound portions (more for GradCache with many micro-batches).
- **Effort:** medium. **Risk:** medium. **Binding:** GPU work removal.

### T6. Per-step metrics (`return_metrics=True`) and `get_label_to_group_mapping()` — many `.item()` syncs + dict rebuild every step
- `semantic_loss.py:287-342, 396-404, 472-500` (metrics); `semantic_loss.py:18-29, 300` + `label_groups.py:238-257` (mapping rebuild)
- **Slow:** Training always calls `criterion(..., return_metrics=True)`: ~15 `.item()` D2H syncs/step + 2 extra `(N x N)` matmuls (`raw_sim`, `text_similarity`). `_build_same_label_mask` rebuilds the full label->group dict every step (no memoization).
- **Fix:** Compute full metrics only every `PLOT_EVERY_N_BATCHES` steps (pass `return_metrics=False` otherwise); batch remaining `.item()` into one `torch.stack(...).tolist()`. Memoize `get_label_to_group_mapping` with `lru_cache` / build once at import.
- **Speedup:** **1.05-1.15x** in overhead-bound regime; removes a guaranteed per-step sync/alloc.
- **Effort:** low (mapping) / medium (metrics). **Risk:** low. **Binding:** sync/latency-bound (CPU↔GPU).

### T7. Memory-bank update: NaN + zero-norm `.any()` checks force GPU->CPU sync every step
- `training_scripts/human_activity_recognition/memory_bank.py:49-117` (esp. 64-80)
- **Slow:** `update()` runs `isnan().any()` + `norm()<1e-6 .any()` on imu and text every step, each forcing a D2H sync to evaluate the Python `if`. small_deep_v2 (current best) uses the memory bank, so this is live.
- **Fix:** Gate checks behind a debug flag (off by default) or run every N steps. Embeddings are already L2-normalized in forward.
- **Speedup:** removes ~4 syncs/step; **1.03-1.1x**.
- **Effort:** low. **Risk:** low. **Binding:** sync-bound.

### T8. flatten_per_patch_embeddings: `.item()` per valid patch (D2H sync per patch)
- `training_scripts/human_activity_recognition/semantic_alignment_train.py:749-770` (line 769)
- **Slow:** Per-patch mode `[label_texts[sid.item()] for sid in session_ids]` does one CUDA sync per valid patch (up to ~2048/call), every train/val/GradCache step.
- **Fix:** `session_ids.cpu().tolist()` once, index in pure Python; or build label expansion from CPU `patch_mask.sum()` counts via `repeat_interleave`.
- **Speedup:** ~2048 syncs/call -> 1; **1.05-1.2x** for per-patch runs.
- **Effort:** low. **Risk:** low. **Binding:** sync-bound (per-patch mode only).

### T9. float64 parquet -> torch, plus unconditional full-array NaN scan per sample
- `datasets/imu_pretraining_dataset/multi_dataset_loader.py:410-424`
- **Slow:** `.values` returns float64 (parquet stores float64), doubling bandwidth through NaN check + ffill/bfill + copy before `.float()`. `np.isnan().any()` scans every sample even when never-NaN.
- **Fix:** `.astype(np.float32, copy=False)` right after `.values` (or store float32 at pack time, ties to T1); gate ffill/bfill behind a per-dataset `has_nan` flag computed once.
- **Speedup:** **1.1-1.5x** on CPU preprocessing.
- **Effort:** low. **Risk:** low. **Binding:** CPU-bound + parallelizable.

### T10. Conditional / ablation-specific traps (cheap to fix, catastrophic if hit)
- **Multi-prototype soft targets: triple-nested Python loop** — `semantic_loss.py:242-244, 258-262`. O(B^2 * K) per-element GPU writes (e.g. 512*512*3 ~= 786K iters/step). Only bites `LABEL_BANK_NUM_PROTOTYPES > 1` ablations, but those runs would be drastically slow. Fix: vectorized block-diagonal scatter / reshape. **10-100x for prototype>1 runs.** Effort low, risk low.
- **Dead strong augmentations: scipy `interp1d` per-channel Python loops** — `augmentations.py:338-556, 164-215` (time_warp/magnitude_warp/resample/time_shift). Not on the active path (`aug_types=[]`) but one config flag from a 10-100x slowdown via CPU<->numpy round-trips. Fix: reimplement with torch `F.interpolate`/`grid_sample`, or clearly mark dead. (rotation_3d at 252-334 is already torch-vectorized and fine.)
- **GradCache CPU-stash via `.cpu().pin_memory()` per micro-batch** — `semantic_alignment_train.py:1174-1183, 1234-1238`. Dozens of D2H+H2D round-trips/window with fresh pinned allocs. GradCache is OFF for small_deep (default), affects only medium/large. Fix: skip stash when VRAM allows, or reuse a persistent pinned buffer. **1.1-1.3x for GradCache runs.**
- **GradCache 2x encoder forward** — `semantic_alignment_train.py:1148-1152, 1242-1246`. Inherent ~2x forward FLOPs tax. Not a bug — keep GradCache OFF where a useful batch fits (memory-bank gives many negatives at ~1x forward). Up to ~1.8-2x throughput for affected runs **if accuracy parity holds** (risk high — validate).

---

## 3. QUICK WINS (low effort, decent speedup)

| Item | File:line | Fix | Speedup | Binding |
|---|---|---|---|---|
| **E2** Dedup MOMENT predict pass | evaluate_moment.py:752-767 | `decision_function` once, derive both | ~2x SVM scoring | CPU |
| **E6** CrossHAR/LiMU dedup forward | crosshar:938-947, limubert:814,824 | Cache logits, mask post-hoc | ~2x ZS inference | GPU |
| **E7** majority_vote_subwindows | limubert.py:250-262 | `np.add.at` 2D histogram | 50-500x helper | CPU |
| **E10** window-label / scoring loops | moment:192, tsfm:347, limubert:209,225, grouped_zero_shot, evaluation_metrics | Vectorize mode, int group LUTs, `lru_cache` | 10-500x per helper | CPU |
| **E11** MOMENT bf16 + batch 256 | evaluate_moment.py:120-177, 61 | autocast bf16, bigger batch | 1.5-2x extract | GPU |
| Inference DataLoaders | limubert:344-345 etc, crosshar:475-476 etc | num_workers>0, pin_memory, bf16, batch 128-256 | 1.3-3x finetune | parallelizable/GPU |
| **T2** Hoist channel metadata | multi_dataset_loader.py:373-407 | Precompute per dataset | 1.2-2x getitem | CPU/parallel |
| **T6/T7/T8** Remove per-step syncs | semantic_loss.py, memory_bank.py:64-80, train:769 | lru_cache mapping, gate checks, `.cpu().tolist()` once | 1.05-1.3x train | sync-bound |
| **T9** float32 cast + gated NaN | multi_dataset_loader.py:410-424 | `astype(float32)`, has_nan flag | 1.1-1.5x preproc | CPU/parallel |
| Hoist eval forward metadata | evaluate_tsfm.py:229-231,281-283,443-450 | Build once, slice `[:bs]` | 1.02-1.1x | CPU host stalls |
| Label-bank memo + single test_emb upload | evaluate_tsfm.py:514-522,566-574,636-638,749-750 | Encode GLOBAL_LABELS once, upload test_emb once/dataset | 1.1-1.3x | redundant work |

---

## DO THESE FIRST (highest impact / lowest effort)

1. **E2 — Dedup the MOMENT SVM pass** (evaluate_moment.py:752-767). ~2x on the single dominant eval cost, low effort, zero accuracy change. *Best ratio in the report.* **CPU-bound.**
2. **E6 — Cache logits in CrossHAR + LiMU-BERT zero-shot** (crosshar:938-947, limubert:814/824). ~2x per model, trivial. **GPU-bound work removed.**
3. **E7 + E10 — Vectorize the per-sample Python loops** (majority_vote_subwindows 50-500x; window-label/scoring/recall helpers 10-100x). Low effort, removes hundreds of thousands of interpreter iterations. **CPU-bound — VRAM does not help.**
4. **E3 — Vectorize TSFM per-patch majority vote** (evaluate_tsfm.py:648-669). 10-100x, medium effort but high confidence; eliminates ~47k syncs x 2. **Latency-bound -> GPU-bound.**
5. **T1 — Pack sessions / in-memory cache to kill per-`__getitem__` parquet reads** (multi_dataset_loader.py:368). 3-10x end-to-end per epoch — the top training win for the ablation campaign. Start with the in-memory dict cache + `persistent_workers` (lowest effort). **CPU/IO-bound + parallelizable.**

Then, for larger payoff at medium effort: **E1** (GPU RBF kernel, 10-50x on HARTH SVM), **E5** (freeze TSFM backbone for few-shot, up to 5-10x on the largest eval component), **E4** (de-dup TSFM extraction + bigger batches, ~5-10x).

### Binding summary
- **CPU-bound (VRAM won't help; vectorize / multithread / pack data):** E2, E7, E8, E10, T1, T2, T9, and the per-step-sync items T6/T7/T8.
- **GPU-bound (bigger batches / autocast help — exploit 18 GB):** E1 (GPU kernel), E3, E4, E5, E6, E11, T4.
- **Parallelizable (concurrency / workers help — exploit 24 threads):** T1 (DataLoader workers + persistent cache), eval finetune DataLoaders, T2, T9; and per-dataset MOMENT SVM scoring could run in parallel processes if kept on CPU.


## DO FIRST
1. E2: In evaluate_moment.py (lines 752-767), call decision_function ONCE and derive both open-set (replicate libsvm OvO vote argmax) and closed-set (masked argmax) from the cached scores. Eliminates the redundant predict() RBF kernel pass for ~2x on the dominant eval cost, zero accuracy change. Low effort, CPU-bound.
2. E6: In evaluate_crosshar.py (938-947) and evaluate_limubert.py (814,824), make the predict path return raw logits once; derive open-set via logits.argmax(1) and closed-set via logits.masked_fill(~mask,-inf).argmax(1) from the cached tensor. ~2x per model on zero-shot inference (big for the sequential GRU). Low effort.
3. E7+E10: Vectorize the per-sample Python loops. Replace evaluate_limubert.py:250-262 majority_vote_subwindows with np.add.at 2D histogram (50-500x, quadratic->linear). Vectorize get_window_labels (moment:192, tsfm:347, limubert:225) via scipy.stats.mode(axis=1), reshape_and_merge (limubert:209), score_with_groups/grouped_zero_shot int LUTs + lru_cache, and compute_semantic_recall (evaluation_metrics:109). 10-500x per helper. Low effort, CPU-bound.
4. E3: Vectorize TSFM per-patch majority vote (evaluate_tsfm.py:648-669): move padded (N,P,D) to GPU once, einsum('npd,ld->npl') in N-chunks, per-patch argmax, scatter_add/bincount over L, single .cpu() at end. Replaces the per-sample H2D + inner vote loop + per-window .item() sync. 10-100x, medium effort, latency-bound->GPU-bound.
5. T1: Kill the per-__getitem__ parquet read (multi_dataset_loader.py:368). Quickest interim: in-memory dict/LRU cache of decoded session arrays per worker with persistent_workers (whole HARTH set is a few hundred MB). Full fix: pre-pack sessions into one mmap-able .npy/Feather/LMDB keyed by session_id so __getitem__ slices instead of opening a file. Read drops ~30ms->~0.1ms, 3-10x end-to-end per epoch. Top training win, CPU/IO-bound + parallelizable.