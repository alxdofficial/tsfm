# HALO "latest principled version" — build plan (no training until Phase F)

Goal (user, 2026-07-04): bring every decided/designed change into code, verify each
with unit + overfit-one-batch sanity (NOT a full training run), **review**, then spend
one retrain. Scope chosen: **Full principled model incl. M3**, with **M3 = streamable
encoder only** (dense per-frame objective deferred — see Phase D).

Status legend: [x] done · [~] partial · [ ] todo · [defer] out of scope for this push.

## Phase A — Tokenizer cutover (filterbank as default)
- [x] preprocessing: `zero_pad_patches` + `pad_to_size` mode (M1).
- [x] encoder: filterbank selectable, rate/N threaded through forward/encode_from_raw (M1).
- [x] dataloader: `dft_size` filterbank mode — zero-pad to S, carry true `N` in metadata.
- [ ] training-loop wiring: `SemanticAlignmentModel.forward` (semantic_alignment_train.py:456)
      + the standard/GradCache steps thread per-sample `sampling_rate_hz` + `patch_len_samples`
      (both already in `batch['metadata']`) into `encoder.forward`. `create_dataloaders`
      passes `dft_size` when the config is filterbank.
- [ ] calibration hook: `fit_norm_stats`/accumulate over the first (augmented) epoch, then freeze.
- [ ] save/load chain-of-custody: persist the 10 tokenizer hyperparams in the saved `encoder`
      config block; `model_loading.load_model` reads + passes them (finding-5 from the M1 sweep).
- Gate: overfit-one-batch on `small_deep_fb` reaches ~0 loss; loud failure if rate/N missing.

## Phase B — Conditioning text-split
- [x] dataloader drops the "sampled at {r}Hz, {D}s window" suffix in filterbank mode
      (kept for legacy). Decision: research_conditioning.md §7.

## Phase C — Component pruning (mostly config)
- [ ] Retrain config: label-bank OFF (`ABLATION_LEARNABLE_LABEL_BANK=0` / `use_mean_pooling=True`)
      — audit shows learned pooling is −5.83pp vs mean-pool; it's a training flag, not a code change.
- [defer→F] Delete the already-gated-off dead session-head classes (`TemporalAttention`,
      `MultiQueryPooling` in model/semantic_alignment.py:179/237 + the `if not per_patch_prediction`
      branch) once per-patch is committed. No forward/accuracy effect (already not built for headline).
- [defer→F] Checkpoint slimming (strip optimizer/scheduler + dead keys): re-save, ~355MB→~111MB.
- KEEP: `ChannelTextFusion` (core contribution); contrastive text_proj is Identity/0 params in small.

## Phase D — Dense per-frame objective + boundary/smoothness  [DEFERRED]
Blocked on data: every dataset is single-activity sessions (`labels.json` = session_id→[one label];
parquet has no per-timestep label). Per-frame labels + boundaries must be *synthesized* (concat
sessions) or sourced from re-converted free-living data (CAPTURE-24/ExtraSensory). Decision
(2026-07-04): defer. The per-patch head + per-patch InfoNCE already exist; on single-activity
windows the dense objective reduces to today's loss (no boundaries to learn). Revisit as a
data sub-project after review.

## Phase E — Streamable encoder (M3, in scope) — REVISED per docs/v2/research_streaming_design.md
Verdict of the literature validation: architecture/paradigm/backbone/RoPE are RIGHT (the deployed
ASR recipe — U2/U2++, dual-mode ASR); SSM/Mamba swap and two-model split correctly REJECTED. The
original recipe ("random-sample one mask per step, loss unchanged, strict zero-lookahead") is the
weakest defensible variant. Build the corrected recipe:

- [ ] RoPE over physical time: derive `patch_start_seconds (B,P)` from per-sample `patch_size_sec`;
      `TemporalPositionalEncoding` → no-op when RoPE on; apply rotary to Q,K in `TemporalSelfAttention`
      (transformer.py:82-102), positions (B,P)→(B*C,P). Channel axis untouched. Fixes the integer-index
      vs augmented-Δt bug. **MUST-TUNE the frequency band to HAR scales** (fast osc resolves ~0.5-1s
      patch spacing; slow osc period > max session span) and **re-anchor timestamps per session/window**
      (FP32 stability; RoTHP translation-invariance keeps attention identical if all cached keys share
      the anchor). Keep time strictly in the rotation (no Time2Vec concat). Config the base/n_freqs.
- [ ] Mask machinery {full, causal-∞, causal-window-W(sec), causal+K-lookahead}: per-sample (B,P,P)
      from physical times (|t_i−t_j|≤W); generalize the mask-combine (transformer.py:91-99). Add an
      **attention sink** (persistent anchor patch in every causal window — StreamingLLM). Expose **K
      (lookahead) as an inference latency dial**; K=0 stays first-class for fall/gesture latency.
- [ ] **RECIPE (the headline change): joint dual-forward + in-place offline→online distillation**
      (Dual-mode ASR, ICLR'21) — replace random single-mask sampling. Each step: forward the same
      patches under a FULL mask (teacher, stop-grad) and a CAUSAL-window mask (student); add a
      distillation loss = KL of the student's softmax-over-label-text-cosine toward the teacher's (or
      MSE on the projected embedding), SEPARATE from the InfoNCE term. Nearly free for HALO (only the
      mask differs — no causal-conv/pool/BN to fix). ~1.3-2× train compute, ZERO inference cost.
      Down-weight near-onset patches (teacher sees future the student can't). Sampling for the windows:
      ~50/50 full-vs-window (U2); warm-start causal modes from the full-context checkpoint.
- [ ] **Fix the train/serve mismatch (surfaced bug):** the offline session label uses an UNTRAINED
      mean-pool. Add a trained pooled-session objective under the same masks so the offline
      session-label path is represented at training time.
- [defer→deployment] Causal temporal label smoothing (TLS-RWKV) for the online per-patch stream — a
      deployment-time band-aid; the real fix is coupled to the deferred dense per-frame objective.
- Gate: streaming-equivalence (offline-causal vs incremental KV-cached per-patch outputs match to
  1e-5); overfit-one-batch stable under dual-forward.

**Note (coupled decision):** the Phase-D deferral and streaming interact — causal predictions near
activity ONSETS have least context and benefit most from BOTH lookahead and the offline teacher, and
those are exactly where the deferred dense/boundary supervision would help. Steady-state streaming
("current activity") is well-served now; onset/boundary precision is capped until dense supervision.

## Ablations to run at retrain (from research_streaming_design.md §5, subject-disjoint, rate-mixed)
random-mask-no-distill vs joint+distill vs +lookahead (measure offline→online drop for IMU/contrastive
— the ASR ~20% streaming gain is UNVERIFIED for us); K∈{0,1,2,3} latency knee; RoPE band on LONG
multi-activity sessions (offline full-mask + RoPE both extrapolate past the single-activity training
window); trained session-pool vs mean-pool; attention-sink long-run stability.

## Phase F — Assemble + review (before any training)
- [ ] Full-model overfit-one-batch + multi-rate/-channel forward.
- [ ] Adversarial debug sweep over the assembled model (esp. Phase E).
- [ ] Dead-class deletion + checkpoint-slim (from Phase C defer list).
- [ ] Present for user review. NO training until this passes.

## Then: M4 retrain (separate, user-gated)
`small_deep_fb` + label-bank-off + calibrate-then-freeze norm stats; rate-invariance + macro-F1
ablations; the "exact deletions" (remove CNN/spectral classes + interpolation) at cutover.
