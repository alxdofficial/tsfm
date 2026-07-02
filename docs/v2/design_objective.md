# HALO V2 — Objective Design Spec

*Component design for [REDESIGN_PLAN.md](REDESIGN_PLAN.md). Where this and the integrated plan disagree, the plan's §1 interface reconciliation wins.*

---

I have everything I need. I read the four required files plus the encoder, text-fusion module, tokenizer, positional encoding, data loader, and inference/eval path. Below is the concrete spec.

---

# HALO Unified Objective: Dense Per-Patch Text-Aligned Sequence Labeling

## 0. Key finding: the per-patch scaffold already exists — it just isn't *dense*

The code is closer to this design than the brief implies, which de-risks it:

- `SemanticAlignmentHead(per_patch_prediction=True)` (`model/semantic_alignment.py:454-458`) already returns `(B, P, output_dim)` and skips the temporal-attention + pooling head (the ~9.3M dead params).
- `flatten_per_patch_embeddings` (`semantic_alignment_train.py:799-820`) already flattens `(B,P,D) → (N_valid, D)` and expands the label per valid patch.
- The training branch (`semantic_alignment_train.py:947-960`) already runs symmetric InfoNCE + frozen-SBERT soft targets + MoCo queue over the flattened *patch* set.
- `ChannelTextFusion` (`token_text_encoder.py:431-438`) pools each channel's text **once** and broadcasts `(B,1,C,D)` across patches — it is already independent of `P`, so it works unchanged for `P=1` (streaming) or `P=full` (offline).
- The temporal attention already accepts a `(P,P)` `temporal_mask` that is threaded end-to-end (`encoder.py:201/301`, `transformer.py:394-432/91-107`) and used as an SDPA `attn_mask`; `transformer.py` test 9 (lines 811-819) literally builds a `torch.tril` causal mask and runs it.

So today's per-patch mode is **secretly the "segment-only supervision" special case**: every patch of a session is aligned to the *same* session-label text (`frozen_text.repeat_interleave(patch_counts)`, `semantic_alignment_train.py:951-953`). The redesign is three surgical changes on top: (1) give each patch its **own frame label** + boundary supervision; (2) drive temporal attention with a **causal/windowed mask** + KV-cache; (3) add the **two inference modes**. Everything else (soft targets, queue, channel-text fusion, open-vocab scoring) is preserved.

---

## 1. Loss: dense per-patch contrastive + boundary term

### 1.1 Objective
For a batch, flatten to `N` valid patches. Let `e_i ∈ R^D` be the L2-normalized IMU patch embedding (from the projection head, unchanged) and `t(y)` the text embedding of a frame label `y`. Keep the **symmetric InfoNCE with frozen-SBERT soft targets and the MoCo queue** — do not remove them. Justification: adjacent patches inside one activity are near-duplicate positives; under hard InfoNCE they would be pushed apart as negatives of each other. The existing frozen soft targets (`semantic_loss.py:150-176`) already solve exactly this at the label level, so the *only* change is that the target text now varies **per patch** instead of per session.

Concretely, replace the per-session frozen-text expansion at `semantic_alignment_train.py:951-953` with a per-frame-label lookup:

```
# was: flat_frozen = frozen_text.repeat_interleave(patch_counts)   # session label per patch
flat_frozen  = label_bank.encode_frozen(flat_frame_labels)          # frame label per patch
flat_text    = label_bank.encode(flat_frame_labels)                 # learnable, frame label per patch
loss_nce, m  = criterion(flat_imu, flat_text, flat_frame_labels,
                         frozen_text_embeddings=flat_frozen,
                         imu_queue=imu_queue, text_queue=text_queue)
```

No change is needed inside `InfoNCELoss._forward_single_prototype` — it already builds the `(N,N)` frozen-similarity soft-target matrix (`semantic_loss.py:158-176`); with distinct frame labels the matrix simply becomes informative at frame granularity. **Queue:** switch `TSFM_QUEUE_MODE=semantic` (the EXP-P1 path already stubbed at `semantic_alignment_train.py:263-265`) so a queued "walking" patch is not treated as a hard negative for an in-batch "strolling" patch.

### 1.2 Multi-class window / straddling patches (the Duan problem)
A patch that spans a boundary gets a **soft mixture target** instead of a single positive: `m_i ∈ Δ^{L}` where `m_i[k] = fraction of timesteps in patch i carrying label k`. This is fed as the target row directly (it fits the existing soft-target machinery — the "hard" one-hot diagonal at `semantic_loss.py:181-187` is replaced by `m_i` mass spread over the in-batch columns whose label ∈ support(`m_i`)). A pure-single-activity patch has a one-hot `m_i`, recovering current behavior. This is the direct fix to the fixed-window-contains-multiple-activities problem raised by [Duan 2023](https://consensus.app/papers/details/bbc337dc251c51cdb021b1799f3a6cb1/).

### 1.3 Boundary / transition term (temporal coherence for streaming)
Add two lightweight terms so streaming output is temporally smooth but sharp at true transitions, following [P2LHAP](https://consensus.app/papers/details/ca93f18a1445527fb380739fba835277/) (neighbor-patch smoothing) and [Duan 2023](https://consensus.app/papers/details/bbc337dc251c51cdb021b1799f3a6cb1/) (boundary-offset):

- **Smoothness (P2LHAP):** with per-patch posterior `p_i = softmax(scale · e_i·T_batchᵀ)`,
  `L_sm = mean_t  w_t · D(p_t, p_{t+1})`, `D = symmetric-KL` (or `1−cos(e_t,e_{t+1})`), and gate `w_t = 0` iff frame label changes between `t` and `t+1` (a true boundary), else `1`. Penalizes flicker *within* an activity, allows free jumps *at* boundaries.
- **Boundary head (Duan):** a tiny MLP `g(fused_t) → (b_t, o_t)` added beside `projection_head` (`semantic_alignment.py:414`). `b_t ∈ [0,1]` = "this patch contains a transition" (BCE vs frame-derived boundary flags); optional `o_t ∈ [0,1]` = normalized distance to nearest boundary (smooth-L1). At inference `b_t` gates the streaming smoother (§3b).

**Total:** `L = L_NCE(dense) + λ_sm·L_sm + λ_bd·L_bd`, defaults `λ_sm=0.1`, `λ_bd=0.3` (ablated). Only `L_NCE` touches the contrastive/text-aligned path; the boundary head is a 2-output regression that does not disturb CLIP-style alignment.

### 1.4 Obtaining per-patch (frame-level) labels
Data are session-labeled *today* (`labels.json` maps session→label list, `multi_dataset_loader.py:307/458`), **but the parquet already stores `timestamp_sec` per row** (`docs/DATA_FORMAT.md`), so frame labels are a converter/loader change, not a schema change:

1. **Native per-timestamp datasets** — PAMAP2, Opportunity, DSADS, RealDisp, MHEALTH, HARTH, Daphnet carry per-sample activity annotation upstream that the current converters collapse. Change `datascripts/{ds}/convert.py` to emit an extra `activity_label` column aligned to `timestamp_sec`. In `__getitem__`, for a patch covering `[t0,t1)`: `frame_label = mode(labels[t0:t1])`, `m_i = histogram(labels[t0:t1])`, `boundary_flag = (labels change within [t0,t1))`. Seq2Dense per-timestep supervision motivates this granularity ([Meena 2023](https://consensus.app/papers/details/63a4c84819735a278a71ed9d42ae0e96/)).
2. **Segment-only datasets** — UCI-HAR, HHAR, WISDM, KU-HAR, UniMIB, RecGym trials are single-activity. **Fallback:** all patches get the segment label (identical to today) and the intra-session boundary term is inactive. To still train the boundary head/smoothness on these, add **stitched-session augmentation**: at train time concatenate two different single-activity sessions (matched channel set / rate) into one recording with a known boundary. This synthesizes labeled transitions for free and is the main new augmentation.

The loader returns per sample: `frame_labels: List[str]` (len `P`), `label_mix: (P, ≤k)` sparse, `boundary_target: (P,)`, alongside the existing `patches`, `patch_mask`, `channel_descriptions`.

---

## 2. Causal / streaming encoder

### 2.1 The change is a mask, not a rewrite
`DualBranchTransformer` is bidirectional only because `temporal_mask=None`. Cross-channel attention is already *within-patch* (no temporal leakage). So streaming = drive `TemporalSelfAttention` with a mask (`transformer.py:91-99` already ANDs `mask` into the SDPA `attn_mask`):

- **Offline / segment:** `temporal_mask=None` (full bidirectional) or a large window.
- **Streaming / causal:** banded causal mask — patch `t` attends to `[t−W+1, t]` (`W` = bounded left context, e.g. 16–32 patches). Pure causal is `W=∞`; the band bounds memory and cost.

### 2.2 Train-both / eval-both (recommended over a single mode)
Sample the mask **per micro-batch** from `{full-context, causal-full, causal-windowed(W)}` with e.g. `{0.34, 0.33, 0.33}`. This makes **one weight set** valid under both offline (full) and online (causal-W) attention, and is a masking curriculum rather than two models. Cheaper streaming-first alternative (ablated): always train causal-windowed; at eval, offline uses a wider window. The mixed schedule avoids the train/test attention-pattern mismatch that a fixed causal-only model incurs when evaluated with full context.

### 2.3 Ring-buffer inference contract (O(1) per patch, fixed latency)
Per channel keep a KV ring buffer of the last `W` patch tokens; keep the current patch's `C` channel tokens for cross-channel attention.

- New patch arrives → tokenizer emits its token(s) → feature/token adapter → channel-text fusion (broadcast, `P=1`) → for each layer: temporal `Q_t` attends to cached `[t−W+1, t]` K/V (append `K_t,V_t`, evict oldest); cross-channel attention over the current `C` tokens.
- Cost per new patch: `O(W·d)` temporal + `O(C²·d)` cross-channel = **independent of stream length**. Latency = `patch_size_sec` (one patch) + smoother lag (§3b).
- **Positional encoding must change for unbounded streams.** Current temporal PE is *absolute sinusoidal* over a `max_patches=5000` buffer (`positional_encoding.py:43-64`); a long stream would run off the table and mismatch training. Replace with **RoPE** (relative by construction, exact under KV-cache) or windowed-relative sinusoid keyed to offset `t−j ∈ [0,W)`. `ChannelSemanticEncoding` is untouched.

---

## 3. Two inference modes (both score per-patch embeddings vs the text label bank — open-vocab preserved)

Let `T = label_bank.encode(candidate_labels)` (closed-set = dataset labels; open-set = all labels; **any** string set — open vocabulary preserved exactly as today).

**(a) Segment classification over a finished recording**
```
E, pmask = model.forward_from_raw(rec, ..., return_per_patch=True)   # (P,D)  already exists, evaluate_tsfm.py:288
S = E @ T.T                                                          # (P, L) cosine
# soft logit-pool (better than hard vote under imbalance):
seg_pred = argmax_k  Σ_{valid t} softmax(S_t / τ)[k]
```
Run the encoder full-context. This subsumes the current per-patch majority vote (`evaluate_zero_shot_majority_vote`, `evaluate_tsfm.py:602-653`) — hard vote stays available as the `argmax_t S_t` special case, but soft logit-pooling is the recommended default for the imbalance the brief flags (macro-F1 ≪ accuracy).

**(b) Streaming emission**
```
init ring buffer + KV-cache
for each arriving patch t:
    e_t = encode_causal(patch_t)          # O(1), §2.3
    p_t = softmax(scale · e_t @ T.T)       # (L,)
    b_t = boundary_head(fused_t)
    if b_t > θ_b: reset smoothing window    # don't blur across a true transition
    q_t = EMA/median over {p_{t-k..t}}      # smoothing, latency k patches
    emit  ŷ_t = argmax_k q_t
```
Latency = `(k + 0.5)·patch_size_sec`; sweeping `k` gives the latency/accuracy curve (ablation). This is the online-emission regime of classic streaming HAR ([Krishnan 2012](https://consensus.app/papers/details/2069a3a53c2a5cccb3e1c7ae323b37cc/); real-time 1 s windows [Ignatov 2017](https://consensus.app/papers/details/9276643ed6665234bd55f313b22d7574/)), now text-aligned and open-vocab.

Both modes are *identical scoring* (per-patch cosine vs `T`) — segment mode just pools before argmax, streaming mode smooths online. That single scoring rule is what lets you **consolidate evaluation** (brief goal 5): one metric family, macro-F1 primary per [Ghosh 2026](https://consensus.app/papers/details/5e6c92a6f65a5f029618ecf827a0f54c/), reported for both a segment protocol and a streaming (frame-F1 + boundary-tolerance) protocol.

---

## 4. Interface

- **New tokenizer input.** Encoder contract stays `patches: (B, P, L, C) → (B, P, C, d_model)` (`encoder.py:234-241`). If the redesigned tokenizer emits fixed-length analysis frames, feed them as `L`; if it emits tokens directly, add a thin `token_adapter: Linear(d_token→d_model)` that bypasses `feature_extractor`. Either way the encoder output shape and everything downstream are unchanged. The tokenizer must be **anti-aliased under causal downsampling** — streaming forbids the current no-filter `F.interpolate` resample (`preprocessing.py:112-115`); use a band-limited / DWT front end ([AaSP](https://consensus.app/papers/details/519c2b54e3ee5ed99ebb9d68b231a4c7/), [BlurPool](https://consensus.app/papers/details/74cb2454204e5c6191b34d32f37f05b4/), [Michau PNAS 2021](https://consensus.app/papers/details/5e035cb6e5425a97b056e1c803d74770/)) and normalization-first framing ([Ghaffari 2024](https://consensus.app/papers/details/59964a2584355a5ba01c23cb2b451dbc/)), with rate handled by an order-preserving reparametrization ([Koneripalli 2020](https://consensus.app/papers/details/92264aa39b645e5999bff8f27a41169f/)).
- **Positional encoding:** RoPE/windowed-relative temporal PE (§2.3); `ChannelSemanticEncoding` unchanged.
- **ChannelTextFusion stays as the language channel-encoding (brief goal 3):** it is already `P`-independent (`token_text_encoder.py:431-438`), so it is bit-identical for `P=1` streaming and `P=full` offline — no change. This preserves the per-channel placement+rate text conditioning validated by [GOAT](https://consensus.app/papers/details/ed52d465eb1d5e1f8a35fb611d3ba628/). Channel-independence keeps capacity/robustness ([Han TKDE'23](https://consensus.app/papers/details/6739ef37d4ab5af28979718e661fe3dd/), [PatchTST](https://consensus.app/papers/details/7425f108b08556ce919a01fa9d1376ac/)).

---

## 5. Training/data changes and ablations

**Training changes**
- Loss = dense InfoNCE (frame labels, kept soft targets + `QUEUE_MODE=semantic`) + `λ_sm·L_sm` + `λ_bd·L_bd`.
- Per-micro-batch attention-mask schedule (§2.2). Class-balanced sampling moves from session-level to **frame-level** counts.
- Boundary head added beside `projection_head`; per-patch mode already the default (`per_patch_prediction=True` in config for tiny/small_deep/medium/large).

**Data changes**
- Converters emit per-`timestamp_sec` `activity_label`; loader derives `frame_labels`, `label_mix`, `boundary_target` in `__getitem__`.
- Stitched-session augmentation for segment-only datasets to synthesize labeled boundaries.

**Ablations**
1. **Dense vs segment-only supervision** (frame labels vs all-patches=session-label): segment accuracy/macro-F1, frame-F1, boundary-F1, streaming detect-latency. Expect dense ≥ segment on frame/boundary, ≈ on segment accuracy.
2. **Causal vs bidirectional:** accuracy delta + cost — causal-windowed is `O(P·W)` vs bidirectional `O(P²)`; report the accuracy given up for streamability, and confirm train-both closes most of the gap.
3. **Streaming latency/accuracy curve:** sweep smoother `k` (and window `W`) → macro-F1 vs latency (ms).
4. **Boundary term on/off** (`λ_bd, λ_sm`): boundary-F1 and streaming flicker rate.
5. **Queue mode** `hard_neg` vs `semantic` at frame granularity (synonym patches as negatives?).

---

## Data digest for integration (exact hooks)

- `datascripts/{ds}/convert.py` → add `activity_label` column aligned to `timestamp_sec` (native per-timestamp datasets only).
- `datasets/imu_pretraining_dataset/multi_dataset_loader.py:__getitem__` (~458, 530-580) → return `frame_labels (List[str], len P)`, `label_mix (P,≤k)`, `boundary_target (P,)`; add stitched-session augmentation.
- `training_scripts/human_activity_recognition/semantic_alignment_train.py:947-960` → replace session-label expansion (951-953) with per-frame-label `encode`/`encode_frozen`; add `L_sm`, `L_bd`; add per-batch `temporal_mask` sampler; set `QUEUE_MODE=semantic`.
- `training_scripts/human_activity_recognition/semantic_loss.py` → **no core change** (soft-target `(N,N)` path at 150-187 already supports per-patch frame labels; add `label_mix` as the target row for straddling patches).
- `model/semantic_alignment.py:414/454-458` → add `boundary_head` (2-output MLP) beside `projection_head` in the per-patch branch.
- `model/transformer.py:91-99, 394-432` → pass banded-causal `temporal_mask`; add KV-cache path for streaming (`W` left context).
- `model/encoder.py:201-304` → `temporal_mask` already threaded; add `token_adapter` bypass for direct-token tokenizer.
- `model/positional_encoding.py:43-64` → replace absolute sinusoidal temporal PE with RoPE/windowed-relative.
- `model/token_text_encoder.py:431-438` (`ChannelTextFusion`) → **unchanged** (already `P`-independent; works for `P=1` streaming).
- `val_scripts/human_activity_recognition/evaluate_tsfm.py:288-653` → segment mode = soft logit-pool over `forward_from_raw(return_per_patch=True)`; add streaming loop (ring buffer + smoother + boundary gate); report macro-F1 (segment) and frame-F1 + boundary-tolerance (streaming).

**Preserved intact:** CLIP-style symmetric InfoNCE, frozen-SBERT soft targets, MoCo queue, LearnableLabelBank / open-vocab cosine scoring, and per-channel language conditioning (`ChannelTextFusion`).