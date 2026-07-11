# Baseline Implementation Notes

**Status:** canonical notes for the six active V2 baselines, updated 2026-07-11.
Historical MOMENT, LanHAR, and LLaSA notes were removed because those models are
not part of the resubmission comparison. Evaluation rules shared by every model
live in [`EVALUATION_PROTOCOL_V2.md`](EVALUATION_PROTOCOL_V2.md); operational
go/no-go status lives in
[`../v2/BASELINE_TRAINING_READINESS.md`](../v2/BASELINE_TRAINING_READINESS.md).

This document distinguishes three kinds of statements:

- **Published contract:** behavior described by the cited paper or official code.
- **HALO adaptation:** a deliberate change needed to place the model in the common
  benchmark. It is not presented as a reproduction of the paper's headline task.
- **Known issue:** a condition that must be fixed or disclosed before a result is
  suitable for the paper.

## Comparison Roles

The active models do not all receive the same kind of training. They must not be
described collectively as "baselines trained on the HALO corpus."

| Model | Comparison role | Work performed by our harness |
|---|---|---|
| CrossHAR | Corpus-matched self-supervised baseline | Pretrain a backbone on the source corpus, freeze it, then fit a 94-way source-label head |
| LiMU-BERT | Corpus-matched self-supervised baseline | Pretrain a backbone on the source corpus, freeze it, then fit a 94-way source-label GRU |
| SSL-Wearables | Externally pretrained foundation model | Load released UK-Biobank weights and fit a source-label head; optionally report a separately named full-fine-tune variant |
| UniMTS | Externally pretrained text-aligned model | Load released weights and evaluate directly; no HAR training in our harness |
| NormWear | Externally pretrained text-aligned model | Load released backbone, MSiTF, and text model and evaluate directly; no HAR training in our harness |
| DeepConvLSTM | Supervised floor | Train from scratch on each target dataset for FS-1%, FS-10%, and full-shot only |

The corpus-matched and externally pretrained groups answer different questions.
CrossHAR and LiMU-BERT help isolate architecture under a shared source corpus.
SSL-Wearables, UniMTS, and NormWear test whether a released model with much more,
or qualitatively different, pretraining transfers to this benchmark. DeepConvLSTM
is not a zero-shot baseline.

## Scale And Input Summary

Parameter counts below were measured from the instantiated models in this
repository. "Sensor-side" excludes a text tower when all query and label
embeddings can be precomputed. "Adapted" is the subset optimized by the current
harness.

| Model | Input used here | Temporal support | Total parameters | Sensor-side / adapted parameters |
|---|---|---:|---:|---:|
| CrossHAR | 20 Hz, 120 x 6 | 6 s | 531,988 | 469,342 adapted head; 62,646 frozen encoder |
| LiMU-BERT | 20 Hz, 120 x 6 | 6 s, classified as six 1 s sequences | 72,800 | 10,154 adapted GRU; 62,646 frozen encoder |
| SSL-Wearables harnet5 | 30 Hz, 150 x 3 acceleration | central 5 s | 4,538,782 | 310,878 adapted head; 4,227,904 frozen trunk |
| UniMTS | 20 Hz, 200 x 22-joint acceleration tensor | 6 s repeated to 10 s | 68,607,997 | 5,179,900 sensor encoder; text tower can be offline |
| NormWear | intended: 65 Hz, 390 x real channels | 6 s | 1,293,856,459 | 193,808,074 sensor + MSiTF; 1.10B TinyLlama can be offline |
| DeepConvLSTM | 20 Hz, 120 x 6 | 6 s | about 457,280 + 129 x classes | all parameters trained per target |

For context, the current HALO Small-Deep model is about 25.8M active sensor-side
parameters and uses an offline frozen MiniLM text encoder. Results tables must
report all three quantities where applicable: adapted/trainable, sensor-side
deployed, and total including text towers. Reporting CrossHAR as only 62.6K, for
example, omits its larger classifier.

## 1. CrossHAR

**Sources:** Hong et al., IMWUT 2024 [1]; official implementation [2].
**Adapter:** `val_scripts/human_activity_recognition/baselines/crosshar.py`.
**Tier:** closed vocabulary, bridged to target labels with ConSE.

### Published Contract

CrossHAR learns IMU representations with hierarchical self-supervision: masked
reconstruction is followed by a joint reconstruction and temporal-contrastive
stage. The official workflow uses 20 Hz, six-second, six-channel IMU windows,
then trains a Transformer classifier on source labels and evaluates transfer to
an unseen dataset [1,2]. The saved `model_masked_*` state is the representation
encoder after the contrastive stage has also back-propagated through it; the
separate contrastive projection is not required for embedding extraction.

### HALO Adaptation

- The local architecture matches the released encoder and loads the checkpoint
  strictly. Per-window `InstanceNorm1d` mirrors the official `IMUDataset` path.
- The frozen `(120,72)` sequence representation is passed to the released-style
  `Transformer_ft` architecture.
- The head predicts the common 94-label source vocabulary. Its softmax is bridged
  to each target vocabulary with top-10 ConSE rather than pretending CrossHAR is
  natively open-vocabulary.
- The current head schedule is 100 epochs, Adam at `1e-3`, batch 512, unweighted
  cross-entropy, and best source-validation loss.

### Caveats And Required Disclosure

- **Stale backbone:** the available checkpoint predates Capture24 and therefore
  represents the former ten-dataset corpus. The cloud recipe refits only the head.
  It is not valid for a claim that CrossHAR and HALO saw the same 11 sources.
- **Schedule ambiguity:** official `config/pretrain.json` specifies 1,600 epochs,
  with 800 joint contrastive epochs. The repository's custom combined-data config
  specifies 200/100. A paper-faithful run and a compute-matched run are both
  defensible, but they must be separately named and their optimizer-step counts
  reported; the shorter schedule cannot be called the published schedule.
- **Head imbalance:** source classes range from 16 to 13,961 windows. The current
  unweighted random-window fit differs from HALO's group-balanced source sampling
  and uses same-subject windows for model selection.
- **Calibration:** ConSE weights depend on classifier confidence. A temperature
  must be fitted on held-out source subjects only, never on a target dataset [7].
- **Memory:** materializing all `(N,120,72)` source embeddings needs about 4.2 GiB
  before train/validation copies. Use at least 32 GiB host RAM.

**Current gate:** architecture smoke test passes; final scientific result is
blocked on current-corpus pretraining, source-subject validation, calibration,
and provenance capture.

## 2. LiMU-BERT

**Sources:** Xu et al., SenSys 2021 [3]; official implementation [4].
**Adapter:** `val_scripts/human_activity_recognition/baselines/limubert.py`.
**Tier:** closed vocabulary, bridged to target labels with ConSE.

### Published Contract

LiMU-BERT applies BERT-style masked reconstruction to normalized six-axis IMU
sequences. The paper fixes the input at 20 Hz and 120 timesteps and uses a
lightweight GRU over 20-timestep sequences for downstream tasks [3]. Its reported
training protocol pretrains for 3,200 epochs and trains downstream classifiers
for 700 epochs [3,4].

### HALO Adaptation

- Each six-second encoder output is divided into six 20-step representations.
- A 94-way GRU is trained on source labels. At evaluation, six sub-window
  softmaxes are averaged into one window distribution before ConSE.
- The current cached backbone and head were trained for 100 epochs each. This is
  a deliberately shortened configuration, not the paper's training budget.
- Acceleration is divided by 9.8 in the model input path, matching the official
  normalization only when the processed array is first expressed in m/s^2.

### Caveats And Required Disclosure

- **Remote source missing:** the clean pod receives a checkpoint but not the
  ignored `LIMU-BERT-Public` Python package, so model setup currently fails.
- **Stale and undertrained checkpoint:** its training log states ten datasets,
  111,589 windows, and 100 pretraining epochs. It predates Capture24 and is much
  shorter than the 3,200/700 paper schedule.
- **Unit errors:** UCI-HAR, HAPT, and UniMiB acceleration is still stored near
  g-scale while the preprocessor treats it as m/s^2; dividing it again by 9.8
  makes these inputs about 9.8 times too small. RecGym is min-max-normalized and
  has no recoverable physical scale. Unit provenance must be fixed per dataset.
- **Transition labels:** the processed label tensor repeats each six-second
  majority label at every timestep. Consequently, the official homogeneous
  one-second-subwindow filter is vacuous and cannot remove transition segments.
- **Unequal optimization:** six sub-windows per parent make 100 LiMU-BERT head
  epochs roughly six times as many head updates as 100 CrossHAR epochs. Report
  optimizer steps, not just epochs.
- **Calibration and validation:** the current head uses unweighted CE and a
  random-window validation split. Use held-out source subjects and source-only
  temperature scaling [7].

**Current gate:** blocked operationally and scientifically until the package,
units, labels, current-corpus checkpoint, and training schedule are resolved.

## 3. SSL-Wearables

**Sources:** Yuan et al., npj Digital Medicine 2024 [5]; official code [6].
**Adapter:** `val_scripts/human_activity_recognition/baselines/ssl_wearables.py`.
**Tier:** externally pretrained closed vocabulary, bridged with ConSE.

### Published Contract

The released harnet family is pretrained with multi-task self-supervision on
roughly 700,000 person-days from about 100,000 UK-Biobank participants. The paper
uses wrist-worn, gravity-present tri-axial acceleration, linearly resampled to
30 Hz in ten-second windows. Its downstream evaluation compares frozen-trunk
training and full fine-tuning; full fine-tuning performs better [5].

### HALO Adaptation

- The current row uses `harnet5`, whose native input is 150 samples at 30 Hz.
  Each common six-second window is center-cropped to five seconds.
- Inputs come from dedicated 30 Hz, g-unit, gravity-present arrays. They do not
  reuse the 20 Hz m/s^2 LiMU-BERT tensors.
- The trunk is frozen and a 94-way released-style `EvaClassifier` head is fitted
  for 100 epochs with Adam at `1e-3`, batch 512, unweighted CE, and best
  source-validation accuracy.
- Only eight of eleven source datasets are usable. KUHAR and RecGym lack the
  required physical gravity-present acceleration, and the aligned UniMiB export
  cannot currently be reconstructed.

### Caveats And Required Disclosure

- The correct row name is **SSL-Wearables harnet5 frozen-head ConSE**, not simply
  "SSL-Wearables." It uses a shorter context and a weaker downstream protocol
  than the paper's headline ten-second, full-fine-tune result [5].
- A strong full-source-fine-tuned variant may be reported separately, using only
  source data and the paper's subject-wise validation and axis/rotation handling.
  It must not replace or be conflated with the frozen probe.
- The 700,000-person-day external corpus is orders of magnitude larger than
  HALO's source corpus. This is an advantage of the baseline and must be visible
  in the table rather than folded into a generic "pretrained" label.
- Twenty-nine of the 94 output classes have no positive SSL-head examples. They
  remain output logits but are not learned as positive classes.
- `torch.hub` currently follows an unpinned `main` branch. Pin the upstream commit
  and released weight checksum before a final run.
- The head still needs class/dataset balancing, source-subject validation, and
  source-only calibration for a fair ConSE comparison [7].

**Current gate:** operationally conditional on a pinned network fetch; publishable
only with the precise frozen-probe name and the above corpus/context disclosures.

## 4. UniMTS

**Sources:** Zhang et al., NeurIPS 2024 [8]; official code and weights [9].
**Adapter:** `val_scripts/human_activity_recognition/baselines/unimts.py`.
**Tier:** native text-aligned cosine zero-shot.

### Published Contract

UniMTS aligns synthetic motion time series with enriched text using an ST-GCN
over a 22-joint skeleton and a CLIP text tower. Synthetic HumanML3D motion gives
all-joint coverage; random joint masking and rotation augmentation target device
placement and orientation robustness. Real datasets are converted to m/s^2,
resampled to 20 Hz, and padded or truncated to ten seconds [8,9].

### HALO Adaptation

- The released checkpoint used here is accelerometer-only. Gyroscope and STFT
  branches are disabled based on the checkpoint structure.
- The common benchmark provides one core sensor placement. Its acceleration is
  written into one dataset-specific SMPL joint; the other 21 joints are zero.
- A six-second, 120-sample window is wrap-padded to 200 samples. Signal and text
  embeddings are L2-normalized and compared by cosine similarity.
- No target examples and no source-label head are used.

### Caveats And Required Disclosure

- **Remote assets missing:** neither the ignored source tree nor checkpoint is in
  the cloud bundle. The recipe also names `xiyuanzh/UniMTS`, while the released
  Hugging Face account is `xiyuanz/UniMTS` [9].
- **Context adaptation:** repeating four seconds of a six-second window is not the
  paper's native ten-second observation. This row measures a six-second benchmark
  adaptation and must say so.
- **Placement adaptation:** official loaders can place MotionSense at two joints
  and Shoaib at five locations. The primary HALO comparison intentionally uses
  the same one core stream available to the other models. A multi-placement
  UniMTS result is a separate native-capability row, not a replacement.
- **Resampling:** the current shared arrays use temporal bin means; official
  UniMTS code uses SciPy Fourier resampling. The difference is a preprocessing
  deviation and should be tested or retained as an explicit parity choice.
- **Units and gravity:** each dataset must follow its official conversion rather
  than a global scale assumption. InclusiveHAR has no official UniMTS recipe and
  therefore needs an explicitly registered placement/gravity decision.
- Target labels are currently tokenized verbatim. Any underscore replacement or
  prompt change must be frozen before evaluation and applied to every rerun.

**Current gate:** blocked until source, checkpoint, revision, and per-dataset
input contracts are packaged and validated on a clean pod.

## 5. NormWear

**Sources:** Luo et al., arXiv 2024/revised 2025 [10]; official code [11].
**Adapter:** `val_scripts/human_activity_recognition/baselines/normwear.py`.
**Tier:** native text-aligned L1 retrieval, not cosine.

### Published Contract

NormWear builds channel-independent CWT scalograms and fuses channel tokens with
a query-conditioned MSiTF module. The paper standardizes preprocessing to 65 Hz
and six seconds, detrends each channel, and applies Gaussian smoothing with
standard deviation 1.3. Its zero-shot activity path aligns the sensor output
with TinyLlama text embeddings and retrieves labels by Manhattan distance [10,11].

### HALO Adaptation

- The model uses all real benchmark IMU channels and the native activity query,
  `What is the current activity?`.
- The adapter repairs an upstream GPU inconsistency by keeping CWT input as a
  NumPy array while moving generated tensors and model execution to CUDA.
- It enables the upstream pure-Torch Ricker CWT because SciPy removed the older
  `signal.cwt` API.
- Scores are negative L1 distance so the shared driver can use argmax without
  changing NormWear's native ranking.

### Caveats And Required Disclosure

- **Current preprocessing is invalid:** the adapter feeds 20 Hz x 120 tensors.
  Upstream `get_embedding` only auto-resamples when the supplied rate is greater
  than 256 Hz, so 20 Hz silently reaches fixed sample-scale CWT filters. The
  physical frequency and expected 390-to-387-sample geometry are wrong.
- The final path must explicitly resample 20 to 65 Hz, apply the published
  detrend and Gaussian smoothing, amplitude-normalize as in official code, and
  then pass `sampling_rate=65` [10,11].
- Accelerometer-only datasets currently arrive with three zero-padded gyroscope
  channels. NormWear is channel-flexible and should receive only real channels;
  fake channels can change MSiTF aggregation.
- Snake-case labels materially change TinyLlama representations. Use frozen,
  naturalized target strings and the released activity answer template.
- The upstream MSiTF loader catches checkpoint errors and continues. The harness
  must verify a strict state-dict load and record both checkpoint hashes.
- The released model contains about 1.294B parameters, but the TinyLlama query and
  label embeddings can be precomputed; report both total and 193.8M sensor-side
  parameters.
- The external pretraining corpus spans heterogeneous physiological and IMU data.
  It is not corpus-matched to HALO. Recheck the exact released checkpoint's data
  manifest for held-out-test overlap before publication.

**Current gate:** blocked on native preprocessing, real-channel handling, strict
weights, natural label text, and clean-pod packaging.

## 6. DeepConvLSTM

**Sources:** Ordonez and Roggen, Sensors 2016 [12]; released notebook [13].
**Adapter:** `val_scripts/human_activity_recognition/baselines/deepconvlstm.py`.
**Tier:** supervised few-shot/full-shot only.

### Published Contract

DeepConvLSTM combines four temporal convolution layers with two LSTM layers so
the CNN extracts local sensor patterns and the recurrent stack models their
temporal evolution. The paper reports 64 convolution filters with kernel length
5, two 128-unit LSTMs, dropout 0.5, RMSProp, and short overlapping windows on the
Opportunity and Skoda tasks [12,13].

### HALO Adaptation

- The local PyTorch reimplementation uses four `Conv2d(64,(5,1))` layers, two
  128-unit LSTM layers, dropout 0.5, and a dataset-specific softmax head.
- It consumes the shared 20 Hz, 120 x 6 benchmark window. Acceleration-only data
  has zero-padded gyroscope columns so the architecture remains fixed.
- Per-channel min-max normalization is fitted on the selected training windows
  only. This replaces the paper's Opportunity-specific hard-coded ranges.
- Training uses unweighted cross-entropy, RMSProp at `1e-3` with `alpha=0.9`,
  batch 64, at most 100 epochs, and patience 15 on validation macro-F1.
- FS-1% and FS-10% source windows are class-balanced. Full-shot retains the
  natural training distribution.

### Caveats And Required Disclosure

- This is a reimplementation, not a bit-for-bit port. PyTorch LSTM behavior,
  initialization, the six-second non-overlapping window, channel count, and batch
  size differ from the paper's Theano/Lasagne experiments [12,13].
- It must only be compared with HALO's supervised FS/full-shot rows. Placing it in
  the ZS-XD table would assign zero-shot capability it does not possess.
- A single split and seed are inadequate for low-data claims. Run at least five
  registered seeds or subject-group folds and report mean, standard deviation,
  and every fold. Shoaib's current single test subject cannot support a
  non-degenerate subject bootstrap interval.
- HALO few-shot currently selects by validation accuracy while DeepConvLSTM
  selects by macro-F1. The final shared protocol must use the same selection
  metric, preferably the primary macro-F1.
- Record the best epoch, subject IDs, normalization extrema, class counts, and
  failure status for every dataset/rate. Partial JSON must never count as a run.

**Current gate:** numerical and clean-pod smoke tests pass; final reporting is
blocked on multi-seed/fold evaluation, common selection criteria, and strict
failure/provenance handling.

## Shared Implementation Requirements

### Source Validation And Calibration

Source-label heads must use subject-disjoint validation where subject IDs are
available. No target labels, target validation windows, or target-derived
temperature may influence zero-shot model selection. ConSE models require one
source-validation temperature per head because their semantic mixture directly
uses softmax magnitudes. Temperature scaling is a one-parameter post-hoc
calibration method fitted after training [7].

### Class And Dataset Balance

The current 94-label source corpus is highly imbalanced across both activities
and datasets. A faithful unweighted-paper run may be retained, but the fair
corpus-matched comparison should also use one pre-registered source sampler or
loss policy across HALO, CrossHAR, LiMU-BERT, and the SSL source head. Report both
when the fairness policy departs from a baseline paper.

### Data Quantity

"Same datasets" does not imply the same amount of usable signal. Current fixed
six-second arrays expose about 218.1 hours to CrossHAR/LiMU-BERT. The SSL head
uses about 116.5 hours after five-second center crops. HALO's current native
session budget is configured for roughly 399 hours. Every result must report
effective windows, seconds, source datasets, exclusions, and sampling policy.

### Model-Native Preprocessing

Fairness means preserving each model's documented input contract, not forcing
byte-identical tensors through incompatible models. The benchmark must hold the
underlying target windows and available sensor information fixed, then apply a
frozen model-specific transform. Rate, units, gravity convention, channel
selection, temporal crop/padding, and normalization belong in result provenance.

### Reproducibility Record

Every final result artifact must contain:

- repository Git SHA and dirty/clean state;
- data-bundle SHA-256 and canonical label-config hash;
- upstream repository commit and model/checkpoint SHA-256;
- model role, input contract, included/excluded source datasets, and effective
  training hours;
- adapted, sensor-side, and total parameter counts;
- loss, sampler, optimizer, learning rate, batch size, optimizer steps, maximum
  and selected epoch, early-stopping metric, and calibration temperature;
- split/fold subject IDs and all random seeds;
- per-dataset completion status, wall time, and software/GPU versions.

## References

1. Hong et al. ["CrossHAR: Generalizing Cross-dataset Human Activity Recognition via Hierarchical Self-Supervised Pretraining."](https://doi.org/10.1145/3659597) IMWUT 8(2), 2024. DOI: 10.1145/3659597.
2. Hong et al. [Official CrossHAR implementation.](https://github.com/kingdomrush2/CrossHAR)
3. Xu et al. ["LIMU-BERT: Unleashing the Potential of Unlabeled Data for IMU Sensing Applications."](https://tanrui.github.io/pub/LIMU_BERT.pdf) SenSys, 2021. DOI: 10.1145/3485730.3485937.
4. Xu et al. [Official LIMU-BERT implementation.](https://github.com/dapowan/LIMU-BERT-Public)
5. Yuan et al. ["Self-supervised learning for human activity recognition using 700,000 person-days of wearable data."](https://www.nature.com/articles/s41746-024-01062-3) npj Digital Medicine 7:91, 2024. DOI: 10.1038/s41746-024-01062-3.
6. OxWearables. [Official SSL-Wearables implementation.](https://github.com/OxWearables/ssl-wearables)
7. Guo et al. ["On Calibration of Modern Neural Networks."](https://proceedings.mlr.press/v70/guo17a.html) ICML, 2017.
8. Zhang et al. ["UniMTS: Unified Pre-training for Motion Time Series."](https://arxiv.org/abs/2410.19818) NeurIPS, 2024.
9. Zhang et al. [Official UniMTS implementation](https://github.com/xiyuanzh/UniMTS) and [released weights](https://huggingface.co/xiyuanz/UniMTS).
10. Luo et al. ["Toward Foundation Model for Multivariate Wearable Sensing of Physiological Signals."](https://arxiv.org/abs/2412.09758) arXiv:2412.09758, revised 2025. DOI: 10.1145/3803808.
11. Luo et al. [Official NormWear implementation.](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear)
12. Ordonez and Roggen. ["Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition."](https://www.mdpi.com/1424-8220/16/1/115) Sensors 16(1):115, 2016. DOI: 10.3390/s16010115.
13. Sussex WearLab. [Released DeepConvLSTM notebook.](https://github.com/sussexwearlab/DeepConvLSTM)
