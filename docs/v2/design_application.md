# HALO V2 — Application Design Spec

*Component design for [REDESIGN_PLAN.md](REDESIGN_PLAN.md). Where this and the integrated plan disagree, the plan's §1 interface reconciliation wins.*

---

I have everything I need. I've verified: no export/ONNX/CoreML/TorchScript/`predict`/`from_pretrained`/streaming code exists (grep only hit a `causal` mask helper in a test path and "predict" substrings in baselines); rotation aug is implemented but hard-disabled (`USE_ROTATION_AUGMENTATION = False # Disabled — destroys gravity orientation signal`); the tokenizer is `create_patches`→`F.interpolate(linear, no anti-alias)`→per-patch z-score; the feature extractor collapses within-patch shape via `AdaptiveAvgPool1d(1)` and its FFT bins are indexed by sample count (rate-dependent); `ChannelTextFusion` is the gated language-as-channel path; `per_patch_prediction=True` skips the temporal-attn+pooling head; and `forward_from_raw` does per-sample CPU Python preprocessing (not traceable).

---

# HALO → Deployable, Playable, Robust HAR — Design

## Consolidated ranking by adoption impact

| # | Deliverable | Section | Why it moves adoption |
|---|---|---|---|
| 1 | `HALO.from_pretrained().predict()` + **one** canonical slim checkpoint + model card | §2 | Nothing gets picked up without a 5-line API and a single blessed weight. |
| 2 | **Freeze/precompute the text bank** so SBERT drops out at inference | §3 | Makes "on-device" and "<10 ms predict" literally true; unblocks 1 & 7. |
| 3 | **Gravity-frame canonicalization + yaw-only aug** | §1a | Biggest real-world correctness gap today; the known deficit vs [UniMTS]. |
| 4 | **Physical-Hz basis-projection tokenizer** (one branch, rate-invariant) | §1b | Pillar-2 ask; kills aliasing/resampling; also becomes a single matmul → export-friendly. |
| 5 | **Open-vocab live label-swap** API + demo | §2/§4.1 | HALO's one capability no closed-set baseline has — the headline. |
| 6 | **Accel-only** via modality dropout | §1d | Doubles device coverage (cheap phones, low-power modes) for ~0 cost. |
| 7 | **Causal unified encoder + `HALOStreamSession`** + stateful export | §3/§4.3 | Turns it into a live, watch-runnable demo; ties segment+stream objective. |
| 8 | Core ML / ONNX / ExecuTorch artifacts + numeric-parity tests | §3 | The actual "runs on a phone" proof. |
| 9 | Corruption-robustness aug + gap masking | §1c | Survives real BLE streams; incremental but necessary. |
| 10 | Phone-in-pocket vs watch scenario | §4.2 | Concrete "works on my device" story; head-to-head with [UniMTS]. |

---

## 1. Robustness to real phone/watch conditions

### (a) Orientation / gravity — the crux (rank 3)

**Diagnosis.** `rotation_3d` (full SO(3), per body location) exists in `datasets/imu_pretraining_dataset/augmentations.py:252` but is disabled at `semantic_alignment_train.py:377` with the exact reason: it *destroys the gravity orientation signal*. That is the whole tension: mounting orientation is a **nuisance** you must be invariant to, but the gravity-relative posture (sit/stand/lie) is **signal** that full-SO(3) rotation scrambles. You cannot fix this with augmentation alone.

**Design — canonicalize, don't randomize.** Adopt **gravity-frame canonicalization** ([HAR-DoReMi]) as the primary fix, with a residual-yaw treatment borrowed, but *constrained*, from [UniMTS]:

1. Estimate gravity per timestep with a **Mahony/complementary filter** (fuse accel+gyro, O(1), the same math phones already run for `TYPE_GRAVITY` / iOS `CMDeviceMotion.gravity`). Rotate accel & gyro into a frame where gravity ≡ +Z. This removes the 2 tilt DOF of mounting nuisance while **preserving** the vertical/horizontal decomposition that encodes posture — precisely the signal HALO was protecting by keeping the device frame.
2. The one remaining DOF is **heading (yaw about vertical)**, which is genuinely unobservable from IMU without a magnetometer. Handle it with **yaw-only augmentation** (random rotation about +Z), *not* full SO(3). This keeps gravity intact yet forces heading-invariance. This is the correct intersection of [HAR-DoReMi] (frame) and [UniMTS] (rotation-invariance), and it re-enables the augmentation the code had to switch off.
3. **On-device is free**: consume the OS-provided gravity/linear-accel virtual sensors when present; fall back to Mahony (accel+gyro) or an accel-only low-pass estimate. Expose the gravity vector as an **explicit channel with its own language description** ("gravity direction, device frame") — the ChannelTextFusion path (`model/token_text_encoder.py:323`) then hands posture to the model for free, reinforcing pillar 3 (language-as-channel).

Why not learned canonicalization (STN / PCA frame): non-deterministic, harder to export, and Mahony is already the phone's native computation — zero marginal cost and reproducible.

**Ablation A1.** Train 3 variants on identical data; eval macro-F1 on multi-placement sets (RealWorld, Shoaib) and under a **test-time random-rotation sweep** (robustness curve):
(i) device frame, no aug (current); (ii) full-SO(3) aug, device frame ([UniMTS]-style); (iii) gravity-frame + yaw-only (proposed). Expect (iii) ≫ (i) on multi-placement and flat under test rotation while (i) collapses. Sub-ablation A1b: OS-gravity vs Mahony-estimated gravity (show the on-device path is lossless); A1c: with/without the explicit gravity-channel-as-language (isolates posture recovery).

### (b) Variable/unknown rate + unknown placement — the tokenizer (rank 4, pillar 2)

**Diagnosis.** `model/preprocessing.py`: seconds-based patches → `F.interpolate(mode='linear', align_corners=False)` to 64 (no anti-alias → aliasing, [BlurPool]/[Zhang 2019]) → per-patch z-score. `SpectralTemporalExtractor`: temporal branch `AdaptiveAvgPool1d(1)` **collapses within-patch shape to a mean**, and the spectral branch's `rfft(n=64)` bins map to Hz **via the sample count**, so identical bins mean different physical frequencies at different rates. Two LayerNorms exist only to patch a 22–30× branch-energy imbalance. Many moving parts, physically unanchored.

**Design — physically-anchored basis projection (one branch, rate-invariant, no resampling).** Replace the whole `create_patches→interpolate→dual-extractor` stack with a single principled projection:

- Patch = fixed **physical duration** T (seconds). At native rate f_s (or irregular timestamps t_n), the patch has N = round(f_s·T) samples.
- Fix K **physical** frequencies f_1…f_K (Hz) in the human-motion band [0, ~15 Hz], plus DC (= tilt after canonicalization) and a linear trend. For each band, cos/sin bases evaluated **at the actual sample times**:
  a_k = (2/N)·Σ_n x(t_n)·cos(2π f_k t_n), b_k = (2/N)·Σ_n x(t_n)·sin(2π f_k t_n), power p_k = a_k²+b_k².
- Token = [log(1+p_k)]_k ⊕ mean/tilt ⊕ trend, then per-channel norm.

This is a **Monte-Carlo estimate of a continuous-time inner product over [0,T]** ([Koneripalli] rate = order-preserving time reparametrization; [Li 2020] continuous-time; [Michau] fixed/learnable band-limited basis with perfect-reconstruction): more samples lower the estimator's variance but do **not** change its expectation → **rate-invariant by construction**, and it works on **irregular/gappy** timestamps because you never resample — you integrate. Bands above native Nyquist f_s/2 are zeroed **and masked**, with the language channel ("sampled at {rate} Hz") telling the model the band is absent. There is no decimation, so there is nothing to alias — a stronger guarantee than adding a BlurPool filter. It is literally **one matmul** Φ(t)·x (Φ fixed given the rate), which is why it also collapses the dual-branch + double-LayerNorm contraption into a single, export-traceable op ([Ghaffari 2024]: gains come from fixed filterbank + compression + norm, not learned filter shapes — so a fixed physical-Hz bank is the right default; keep [AaSP]-style learnable band-limited kernels as an optional variant, not the baseline).

**Placement** stays as ChannelTextFusion channel-text — the user literally passes `placement="wrist"`, which substitutes the channel-description string. Because features are now physically anchored, the rate text becomes a soft prior rather than a crutch.

**Ablation B1 (rate).** One model, evaluate zero-shot at native and resampled rates (×0.5, ×2, and a non-integer 37 Hz): (i) current interpolate-to-64+dual extractor; (ii) +anti-alias LP before resample (isolates the [BlurPool] bug-fix); (iii) proposed basis projection. Report macro-F1-vs-rate flatness + param count + latency (the "fewer moving parts" claim). **Ablation B2 (placement text):** correct vs wrong vs generic ("body-worn IMU") placement string on RealWorld/Shoaib (validates [GOAT]; quantifies sensitivity to the user's one word).

### (c) Noise / dropout / gaps (rank 9)

The basis tokenizer already tolerates gaps (integrate over available samples). Add: **per-patch coverage ratio** → route low-coverage patches through the existing `patch_mask` and down-weight in pooling; **train-time corruption aug** matching real BLE/watch streams — Bernoulli sample dropout, burst gaps (BLE stalls), datasheet-spec sensor noise, ±range clipping, clock jitter ([Krishnan 2012]); **predict() hygiene** — impute short gaps with the basis fit itself (it is a smoother), abstain when coverage < threshold. **Ablation C1:** macro-F1 vs corruption severity, trained with/without corruption aug × (basis vs interpolation tokenizer); report area-under-robustness-curve.

### (d) Unknown gyro availability — accel-only (rank 6)

Support is latent already (`channel_mask[:,3:]=False`, `has_gyro`). Make it first-class with **modality dropout** at train time (randomly drop gyro, and individual accel axes). Since sensors, rate, and placement are all expressed as channel-text + a channel mask, "accel-only wrist @ 50 Hz" is just a different description set — no architecture change (channel-independent design, [PatchTST]/[Han CI]). Provide an **accel-only gravity fallback** (low-pass) so posture classes survive without gyro. **Ablation D1:** macro-F1 for {accel+gyro, accel-only, accel-only+gravity-LP} × trained-with/without modality dropout; expect dropout training to close most of the accel-only gap and the gravity-LP channel to recover posture classes.

---

## 2. Developer artifact / playability (rank 1, 5)

**predict() contract** (new `halo/api.py`):
```python
model = HALO.from_pretrained("halo-small-deep-v2")   # slim ckpt + model card + frozen label/channel banks
pred = model.predict(
    window,                    # np.ndarray [T,C]; SI units (m/s^2, rad/s); acc-only [T,3] ok
    rate_hz=50,                # scalar or per-channel; approximate ok
    labels=["walking","running","cycling","sitting"],  # open-vocab strings
    placement="wrist",         # → channel-text; "pocket"/"chest"/...
    timestamps=None,           # optional [T] sec for irregular/gappy streams
    channels=("acc","gyro"),   # present sensors; enables accel-only
) -> Prediction(label, confidence, per_label: dict[str,float], embedding, abstained: bool)
```
Contract: auto-detect g vs m/s² by magnitude (warn); gravity-canon applied internally; `confidence = softmax(cosine/τ)` with an **abstain threshold** (max-cosine < τ_reject → `"unknown"`) so open-set is clean and reviewer-proof; deterministic, training-free; `predict_batch`; returns the embedding for reuse.

**One canonical checkpoint + model card.** Bless `training_output/semantic_alignment/small_deep_v2_4b3fdd6/best.pt` (per MEMORY, reproduces the headline; ≡ `halo_best.pt`) as the single `halo-small-deep-v2`. **Strip the dead temporal-attn+pooling head** that `per_patch_prediction=True` skips (`model/semantic_alignment.py:454`), fold LayerNorms, ship fp16. Model card: 11 training datasets + subject counts, placements/rates seen, known failure modes (heading ambiguity, accel-only posture), **macro-F1 as the primary metric** ([Ghosh 2026]), exact preprocessing (gravity frame + tokenizer), license.

**Runtime open-vocab label-swap.** Because classification = cosine vs text-embedded labels, swapping = re-encoding strings: `model.set_labels([...])` uses (a) an optionally-bundled MiniLM tower, (b) a precomputed bank, or (c) caller-supplied embeddings. Document abstain + free-text synonyms ("brushing teeth" just works). This is the differentiator — add a class from a sentence, zero retraining.

**Playable demo (pick the phone-web one).** A browser page using `DeviceMotion`/Generic Sensor API — open a URL on your phone (no install), stream accel(+gyro) over WebSocket to a tiny server running HALO (or an ONNX-web model fully client-side), show **live streaming labels + confidence** and a text box to **add a new activity live**. Fallback: a Colab that loads a recorded CSV and prints a scrolling label timeline. Plus a `halo record` / `halo predict file.csv --labels ...` CLI for reproducibility.

---

## 3. On-device export (rank 2, 7, 8)

**The move that makes on-device real: freeze the text side.** `model/config.py:11` already *claims* the text encoder is never deployed — but no code does it. Precompute:
- **Label matrix** `[num_labels × D]` from `LearnableLabelBank.encode(labels)` for the target label set.
- **Channel-text embeddings** for the enumerated placement × sensor × rate combos → constants fed into `ChannelTextFusion` (replacing its runtime SBERT pooling).

Then the on-device graph is: gravity-canon → basis tokenizer (matmul) → sensor encoder → channel-text fusion (channel embeddings = constants) → semantic head → embedding → matmul with label matrix → softmax. **No SBERT, no tokenizer vocab.**

**Make it traceable.** `forward_from_raw` (`semantic_alignment_train.py:611`) does per-sample CPU Python preprocessing over lists/dicts — not exportable. Add a pure-tensor `forward_export(x_or_patches, channel_emb_const, label_matrix_const)` with static shapes / dynamic axes for #patches and #channels. The basis tokenizer being a fixed matmul (vs `F.interpolate`-per-sample) is a concrete reason the §1b redesign *enables* export.

**Targets:** Core ML `.mlpackage` (iOS/watchOS; trace TorchScript of the sensor path — MHA traces fine); ONNX + ORT-Mobile/NNAPI/QNN (Android) and **ONNX-web** (browser demo); ExecuTorch `.pte` via `torch.export` (native Android). Precision fp16 default, int8 optional (per-tensor conv/linear; softmax/LayerNorm fp16). Report size (small ≈ 20–40 MB fp16, tiny ≈ 5–10 MB) + per-1 s-window latency on a mid phone.

**Streaming ties to the causal encoder (§4).** For true live labels the encoder must be **causal** (left-to-right) so a **KV-cache / ring buffer** emits one label per new patch without recomputing the window — O(1)/patch, which pairs perfectly with the per-patch head that already exists. Export the causal encoder as a **stateful** model (Core ML stateful / ONNX with KV I/O / ExecuTorch buffer mutation). Ship `HALOStreamSession` holding gravity-filter state + KV cache + a boundary-smoothing buffer ([P2LHAP]). **Validation:** PyTorch↔exported numeric parity (|Δcos| < 1e-3), latency/size report, demo running the exported artifact.

---

## 4. Unified objective + scenarios to showcase — pick 3

The unified objective is **[P2LHAP] patch-to-label**: the causal encoder emits a per-patch embedding (already the case), scored by cosine vs the label bank → per-patch labels give **both** segment classification (aggregate patches over a recording) **and** streaming (emit per patch), with neighbor-patch **boundary smoothing** for the multi-activity-window problem ([Duan 2023], [Meena 2023]). This single head/loss removes the "segment vs stream" fork and simplifies eval.

**Scenario 1 — Add a NEW custom activity live (headline).** HALO's unique capability vs closed-set baselines (LiMU-BERT, CrossHAR *cannot* do this). Evidence: on an activity absent from training (VTT-ConIoT industrial actions or MobiAct vehicle-entry), show typing the name (0 examples) yields non-trivial macro-F1, and 1–5 example prototypes jump it; contrast with closed-set baselines that structurally cannot. Deliver the live demo clip. Reviewer-proof because it's a *capability*, not a leaderboard delta.

**Scenario 2 — Phone-in-pocket vs watch (placement/orientation generalization).** Showcases §1a/§1b. Evidence: leave-one-placement-out on RealWorld/Shoaib and phone↔watch transfer, macro-F1 with placement text correct vs generic; head-to-head with [UniMTS] (where it competes). This is the "works on my device" story that orientation-fragile baselines fail.

**Scenario 3 — Fitness rep counting / gym (streaming + segmentation).** Showcases the unified causal objective and gives a crowd-pleasing watch demo (count push-ups/squats). Evidence: RecGym (train) → a held-out gym set; per-patch labels + peak-count on the vertical-accel band → rep counts; report frame-level macro-F1 + **rep-count MAE** + streaming latency.

**Drop from the headline:** *fall detection* — safety-critical rare-event eval (precision/recall at low false-alarms-per-day over long idle streams) invites scrutiny about imbalance and is hard to claim responsibly; mention as a stretch with the honest bar. *Gait/health* needs clinical-ish validation (cadence/symmetry) — future work.

---

## Data digest for integration

- **No prior art to reuse for export/predict/stream** — grep for `coreml|onnx|torchscript|torch.jit|from_pretrained|def predict|streaming` finds nothing usable (only a `causal` mask helper `model/transformer.py:812` and `predict` substrings in baseline evaluators). All of §2/§3 is greenfield: put the API in a new `halo/` package; put exporters in `scripts/export/`.
- **Tokenizer to replace:** `/home/alex/code/HALO/code/model/preprocessing.py` (`create_patches`, `interpolate_patches` = aliasing culprit, `normalize_patches`) and `SpectralTemporalExtractor` in `/home/alex/code/HALO/code/model/feature_extractor.py` (`AdaptiveAvgPool1d(1)` mean-collapse; `rfft(n=fft_size)` rate-dependent bins; dual LayerNorm energy patch). New basis-projection tokenizer is a fixed matmul → drops into `feature_extractor.py` as a new `feature_extractor_type` and is export-traceable.
- **Keep as-is (pillar 3):** `ChannelTextFusion` (`/home/alex/code/HALO/code/model/token_text_encoder.py:323`) — gated `fused = sensor_tokens + gate*channel_embs`; the language-as-channel path. For export, feed precomputed channel embeddings instead of running SBERT.
- **Orientation:** `rotation_3d` SO(3) aug already implemented (`/home/alex/code/HALO/code/datasets/imu_pretraining_dataset/augmentations.py:252`, triads grouped by body location via `group_channels_by_sensor`) but disabled at `/home/alex/code/HALO/code/training_scripts/human_activity_recognition/semantic_alignment_train.py:377`. Add a gravity-canon preprocessing stage + a **yaw-only** aug variant; re-enable under canonicalization.
- **Encoder/head:** `per_patch_prediction=True` (config `small_deep`) already yields per-patch embeddings and **skips** the temporal-attn+pooling head (`/home/alex/code/HALO/code/model/semantic_alignment.py:454`) — strip those dead params for the slim ckpt; make the encoder **causal** for streaming. Inference is per-patch cosine vs label bank + majority vote / mean-pool (`forward_from_raw`, `semantic_alignment_train.py:648`).
- **Canonical checkpoint:** `training_output/semantic_alignment/small_deep_v2_4b3fdd6/best.pt` (+ required sibling `hyperparameters.json`, loader at `/home/alex/code/HALO/code/val_scripts/human_activity_recognition/model_loading.py`). Ship as `halo-small-deep-v2`. d=384, 8 layers, `semantic_dim=384`, MiniLM text tower.
- **Data/config:** `benchmark_data/dataset_config.json` — 18 datasets, per-dataset `sampling_rate_hz`, `channels`, `core_channels` map (standard `acc_x…gyro_z` → original names), `placement`, `num_subjects`, `activities`; 11 train / 7 zero-shot; 87 labels → 34 groups (`datasets/imu_pretraining_dataset/label_groups.py`). Channel-desc format (keep for text bank): `"{dataset_desc} {channel_desc} (sampled at {rate}Hz, {patch_size}s window)"`. Rates in-corpus span 9–100 Hz (PAMAP2 9 Hz ↔ KU-HAR 100 Hz) — the concrete rate-invariance test bed for Ablation B1.
- **Eval consolidation (ties to metrics ask):** the per-patch unified head lets you report **one** frame-level macro-F1 (primary, per [Ghosh 2026]) + abstain-based open-set, retiring the 4-metric × synonym-ontology scoring in `/home/alex/code/HALO/code/val_scripts/human_activity_recognition/evaluate_tsfm.py` that invites nitpicking; fix subject-disjoint splits at the same time (leakage, [Gholamiangonabadi 2020], [Rehman 2024]).

## Citations
- [AaSP](https://consensus.app/papers/details/519c2b54e3ee5ed99ebb9d68b231a4c7/) · [BlurPool/Zhang 2019](https://consensus.app/papers/details/74cb2454204e5c6191b34d32f37f05b4/) · [Michau wavelet](https://consensus.app/papers/details/5e035cb6e5425a97b056e1c803d74770/) · [Wave-U-Net DWT](https://consensus.app/papers/details/a6ecb83c936a5363a48db16cec92f805/) · [Ghaffari 2024](https://consensus.app/papers/details/59964a2584355a5ba01c23cb2b451dbc/) · [Koneripalli rate-invariant](https://consensus.app/papers/details/92264aa39b645e5999bff8f27a41169f/) · [Li 2020 continuous-conv](https://consensus.app/papers/details/de98bb0beeff5be18dd239ce2ab9f9b9/) · [Han CI](https://consensus.app/papers/details/6739ef37d4ab5af28979718e661fe3dd/) · [PatchTST](https://consensus.app/papers/details/7425f108b08556ce919a01fa9d1376ac/) · [P2LHAP](https://consensus.app/papers/details/ca93f18a1445527fb380739fba835277/) · [Meena Seq2Dense](https://consensus.app/papers/details/63a4c84819735a278a71ed9d42ae0e96/) · [Duan 2023](https://consensus.app/papers/details/bbc337dc251c51cdb021b1799f3a6cb1/) · [Krishnan 2012](https://consensus.app/papers/details/2069a3a53c2a5cccb3e1c7ae323b37cc/) · [Ignatov 2017](https://consensus.app/papers/details/9276643ed6665234bd55f313b22d7574/) · [UniMTS](https://consensus.app/papers/details/f924af83748257a8b57407578dfb8e84/) · [HAR-DoReMi](https://consensus.app/papers/details/30228e1951c751a18e53494118d60ef6/) · [GOAT](https://consensus.app/papers/details/ed52d465eb1d5e1f8a35fb611d3ba628/) · [Ghosh 2026](https://consensus.app/papers/details/5e6c92a6f65a5f029618ecf827a0f54c/) · [Gholamiangonabadi 2020](https://consensus.app/papers/details/f677839713b55ef087ff10478c05aec1/) · [Rehman 2024](https://consensus.app/papers/details/fcdfa1dfc9985f95ba51f666f8800f90/)