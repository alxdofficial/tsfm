# Baseline heterogeneity-flexibility matrix

How much each model can flex on each axis of data heterogeneity — i.e. whether the axis is
handled **in the model** or must be **fixed in preprocessing**. This justifies the per-baseline
input policy (each model gets the input format its architecture requires — see
`EVALUATION_PROTOCOL_V2.md`) and is the substance of HALO's contribution: HALO is the only model
flexible on every axis *by construction*, so it is also the only one that needs no per-model
resampling/coercion.

**Flexibility levels**

| Level | Meaning |
|---|---|
| **native** | handled in-model, no information loss (flexible by construction) |
| **resample** | handled, but by implicit resampling/interpolation — **lossy** (aliases high-frequency content) |
| **pad+mask** | variable count handled by zero-padding + masking (no loss, wastes capacity) |
| **fixed** | one value required; heterogeneity pushed entirely into preprocessing |
| **none** | not represented / unsupported |

## Matrix

| Model | Sampling rate | Channel count | Window / session length | Sensor modality | Placement-aware | Streaming | Open-vocab labels |
|---|---|---|---|---|---|---|---|
| **LiMU-BERT** | fixed 20 Hz | fixed 6 (pad+mask) | fixed 6 s (120 ts) | fixed acc+gyro | none | offline | no (ConSE bridge) |
| **CrossHAR** | fixed 20 Hz | fixed 6 (pad+mask) | fixed 6 s (120 ts) | fixed acc+gyro | none | offline | no (ConSE bridge) |
| **ssl-wearables** | fixed 30 Hz | fixed 3 — **accel only** | fixed 10 s (300 ts) | **accel only** (no gyro/mag) | none | offline | no (ConSE bridge) |
| **DeepConvLSTM** | fixed (train-time) | fixed | fixed window | fixed | none | offline\* | no (closed softmax) |
| **UniMTS** | **resample** (must be told rate) | native (skeleton graph + mask) | **fixed 10 s** (@20 Hz: first-10 s truncate + wrap-pad) | **accel only (3-ch)**◇ | **native** (SMPL joint) | offline | yes (text-aligned) |
| **NormWear** | **fixed 65 Hz** (resample, lossy) | native (per-ch + cross-ch attn) | fixed (65 Hz-tuned CWT) | native (multivariate) | none | offline | yes (text-aligned)§ |
| **HALO (ours)** | **native (invariant)** | **native (ch-indep + mask)** | **native (variable patch)** | **native** | **native (language)** | **offline + online** | **yes (text-aligned)** |

\* DeepConvLSTM's LSTM is causal (technically streamable) but we run it offline as the few-shot floor.
◇ UniMTS is **accelerometer-only (3-ch)** — its released checkpoint is pretrained *and* evaluated with no gyroscope (verified from run_pretrain.sh / run_evaluation.sh). Earlier docs wrongly said "6-axis acc+gyro." It also needs per-dataset unit scaling (m/s² accel) — a single global scale mis-scales g-unit test sets ~9.8×.
§ NormWear zero-shot is **not a plain cosine**: it requires a separate MSiTF alignment checkpoint + query-conditioned fusion (the sensor embedding depends on a task-query sentence, so it is not a label-independent per-window embedding). It also resamples every signal to a **fixed 65 Hz** (CWT scales hardcoded in sample units; `get_embedding` only auto-resamples above 256 Hz, so a naive HAR-rate call runs the 65 Hz-tuned CWT on native data). Corrected from the earlier "flexible CWT" reading after verifying its released code. Pretraining is almost entirely non-IMU physiological signals (PPG/ECG/EEG/GSR) — a domain asymmetry to disclose.

## How each handles (or side-steps) heterogeneity

- **LiMU-BERT / CrossHAR** — handled entirely in *preprocessing*, not the model. A learned
  positional embedding `Embedding(120, 72)` welds them to 20 Hz / 6 s windows (position = index,
  no physical time); every dataset is bin-and-mean resampled to 20 Hz and coerced to a fixed
  6-channel acc+gyro layout, zero-padding missing sensors. The model is rate- and channel-blind by
  design. Window is a fixed 120 timesteps (CrossHAR uses the full sequence; LiMU-BERT reshapes it
  into 20-step sub-windows).
- **ssl-wearables (harnet)** — fixed 30 Hz / 10 s / **3-channel accelerometer-only** 1D-ResNet. It
  **cannot ingest gyroscope or magnetometer at all**; resample to 30 Hz upstream. Most handicapped
  on gyro-bearing test sets (disclose).
- **DeepConvLSTM** — from-scratch supervised floor; fixed rate/channels/window per experiment, no
  heterogeneity mechanism.
- **UniMTS** — handles rate by **resampling internally** (pass `--original_sampling_rate`;
  interpolates to a fixed rate → aliases HF content, **not invariant**) and placement by mapping
  each sensor to the nearest joint of a canonical **22-joint SMPL skeleton graph** + masking empty
  joints (genuinely placement-aware). Expects 6-axis (acc+gyro) per placed sensor.
- **NormWear** — **CWT scalogram** tokenization is length/rate-adaptive, and per-channel scalograms
  + cross-channel attention accept variable multivariate inputs — the closest baseline to HALO's
  tokenizer, but it is a per-signal CWT (not a physical-Hz filterbank) and does not claim
  rate-invariance by construction.
- **HALO (ours)** — flexible on *every* axis by construction: rate-invariant physical-Hz filterbank
  (**no resampling**) + Nyquist observability masks; channel-independent tokenizer + masks (any
  count / modality); language channel-conditioning (placement + sensor semantics); variable-duration
  patches with RoPE over physical time; one weight set for offline + streaming.

## Why this matters for fairness

Because the fix-upstream group (LiMU-BERT, CrossHAR, ssl-wearables, DeepConvLSTM) is
heterogeneity-blind, the *data* must be adapted to each — that is exactly why capture24 and
inclusivehar need a **20 Hz copy** for LiMU-BERT/CrossHAR and a **30 Hz/3-ch copy** for
ssl-wearables. Giving each baseline its required input format is not a handicap — it is its design.
HALO **not needing any of this** is the rate-invariance / heterogeneity claim itself, demonstrated
by the multi-rate ablation (baselines require resampling / degrade; HALO stays flat) and separated
from its input advantages by the parity row (HALO forced to 20 Hz + neutral channel text).
