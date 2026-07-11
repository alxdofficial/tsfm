"""NormWear (multivariate wearable-signal foundation model) internals for the v2 harness.

NormWear is a channel-independent ViT over ricker-CWT scalograms of each sensor channel, with a
query-conditioned MSiTF aggregator that fuses the per-channel patch tokens into a single 2048-d
vector, aligned to a TinyLlama text encoder. Zero-shot HAR compares the signal embedding to each
label's TinyLlama embedding by MANHATTAN (L1) distance, argmin => our bespoke "l1" adapter tier
(NOT cosine: the spaces are asymmetric and the native metric is L1, so a dot product has the wrong
sign/geometry).

Verified against auxiliary_repos/NormWear/zero_shot/msitf_fusion.py + main_model.py + sentence_template.py:
- get_embedding(x=(bn,nvar,L), sampling_rate) -> (bn,nvar,P,768); MSiTF -> (bn,2048) query-conditioned.
- Channel-independent, NO joint layout: feed all 6 IMU channels (acc+gyro) as nvar=6.
- get_embedding applies NO normalization, and NormWear pretraining amplitude-normalized, so we
  per-window per-channel z-score first (this also removes the static gravity DC, fine: the CWT +
  1st/2nd-difference sub-bands emphasize AC).
- sampling_rate only matters when >256 Hz; at 20 Hz it is a no-op (fixed ricker scales), passed honestly.
- The default backbone builds with optimized_cwt=False -> scipy.signal.cwt (removed in modern scipy);
  we flip sensor_model.optimized_cwt=True to use the pure-torch ricker CWT.
- Weights (GitHub release v1.0.0-alpha): normwear_pretrain_ckpt.pth (backbone),
  normwear_msitf_zeroshot_last_checkpoint-5.pth (aggregator). Text: TinyLlama (~2.2GB from HF).
"""

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
NORMWEAR_PARENT = PROJECT_ROOT / "auxiliary_repos"          # package parent (NormWear.* relative imports)
NORMWEAR_REPO = NORMWEAR_PARENT / "NormWear"
BACKBONE_CKPT = NORMWEAR_REPO / "checkpoints" / "normwear_pretrain_ckpt.pth"
MSITF_CKPT = NORMWEAR_REPO / "checkpoints" / "normwear_msitf_zeroshot_last_checkpoint-5.pth"
LIMU_DIR = PROJECT_ROOT / "benchmark_data" / "processed" / "limubert"

EMB_DIM = 2048
SAMPLING_RATE = 20        # limubert grid; no numeric effect (NormWear only resamples >256 Hz)
QUERY = "What is the current activity?"           # native 'activity' question_template[0]
ANSWER_TEMPLATE = "This subject is presently {}."  # native 'activity' answer_template[0]


def load_normwear_model(device):
    """NormWearZeroShot(backbone + MSiTF aggregator + frozen TinyLlama text encoder), everything
    frozen, with the pure-torch ricker CWT enabled. TinyLlama is fetched from HF on first run."""
    if str(NORMWEAR_PARENT) not in sys.path:
        sys.path.insert(0, str(NORMWEAR_PARENT))
    from NormWear.zero_shot.msitf_fusion import NormWearZeroShot  # noqa: E402

    model = NormWearZeroShot(weight_path=str(BACKBONE_CKPT), msitf_ckpt=str(MSITF_CKPT),
                             use_query=True, rel_only=False).to(device).eval()
    model.sensor_model.optimized_cwt = True   # avoid removed scipy.signal.cwt
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def compute_query(model) -> torch.Tensor:
    """One task-query embedding (1, 2048) reused for every window (native HAR protocol)."""
    return model.txt_encode([QUERY])


@torch.no_grad()
def _signal_encode_np(model, x_np, query, device):
    """GPU-correct reimplementation of NormWearZeroShot.signal_encode.

    The released signal_encode does `device = x.device` then get_embedding does `x.numpy()` --
    self-contradictory (a CUDA tensor can't .numpy(); a numpy array has no .device), so the released
    GPU path is broken. get_embedding expects a numpy/CPU input and uses its `device` arg to move the
    spectrogram + backbone onto the GPU. We pass numpy x + device=cuda and replicate the exact
    query-broadcast + aggregator call from signal_encode.
    """
    sensor_out = model.sensor_model.get_embedding(x_np, sampling_rate=SAMPLING_RATE, device=device)  # (bn,nvar,P,768)
    q = query.expand(sensor_out.shape[0], query.shape[1]) if query.shape[0] == 1 else query
    bn, nvar, P, E = sensor_out.shape
    q = q.unsqueeze(1).expand(bn, nvar * P, q.shape[1])
    return model.aggregator(sensor_out, q, device=device, rel_only=model.rel_only, use_query=model.use_query)  # (bn,2048)


@torch.no_grad()
def window_embeddings(ds: str, model, device, query_emb=None, batch=128) -> np.ndarray:
    """(N,2048) query-conditioned signal embeddings from the 6-ch 20 Hz limubert windows.

    limubert (N,120,6) -> (N,6,120) -> per-window per-channel z-score -> signal_encode (GPU-fixed).
    """
    if query_emb is None:
        query_emb = compute_query(model)
    X = np.load(str(LIMU_DIR / ds / "data_20_120.npy")).astype(np.float32)   # (N,120,6)
    X = np.transpose(X, (0, 2, 1))                                            # (N,6,120)
    # De-fabricate channels: acc-only datasets are zero-padded to 6 channels upstream. NormWear is
    # channel-INDEPENDENT and pools across channels, so a constant-zero "gyro" enters that pool as a
    # real observation and distorts the embedding. Keep only REAL channels — a padded channel is
    # exactly 0 everywhere (max|x|==0), so any channel with a nonzero sample is real. NormWear
    # accepts variable nvar. (The 65 Hz native-rate path — vs this 20 Hz grid — is a separate gate.)
    real = np.abs(X).max(axis=(0, 2)) > 1e-8                                  # (6,) bool
    X = X[:, real, :]                                                         # (N, nvar_real, 120)
    # NormWear's native per-channel normalization (modules/signal_preprocess.basic_preproc:58-65):
    # detrend (remove linear trend incl. the static gravity DC) then divide by mean|x| (amplitude
    # normalize into NormWear's ~unit regime). We skip its bandpass (lc/hc tuned for >=65 Hz
    # physiological signal, inappropriate at 20 Hz IMU).
    from scipy import signal as _sig
    X = _sig.detrend(X, axis=2, type="linear")
    X = X / (np.mean(np.abs(X), axis=2, keepdims=True) + 1e-6)
    outs = []
    for i in range(0, len(X), batch):
        # C-contiguous: NormWear's calc_cwt uses x.view() (main_model.py:109), which errors on the
        # non-contiguous array left by our transpose+slice (astype order='K' preserves layout).
        xb = np.ascontiguousarray(X[i:i + batch], dtype=np.float32)           # numpy (b,6,120)
        emb = _signal_encode_np(model, xb, query_emb, device)                # (b,2048)
        outs.append(emb.float().cpu().numpy())
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def encode_labels(label_strings, model, device) -> np.ndarray:
    """(n_labels,2048) TinyLlama embeddings of the label strings in NormWear's activity answer template."""
    sents = [ANSWER_TEMPLATE.format(l.strip()) for l in label_strings]
    return model.txt_encode(sents).float().cpu().numpy()


def l1_scores(win: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """(N,C) higher-is-better scores = -Manhattan distance (native NormWear metric is L1 argmin).
    Kept as -dist so a standard argmax/predict_from_similarity picks the nearest label."""
    dist = np.abs(win[:, None, :] - lab[None, :, :]).sum(-1)                  # (N,C)
    return -dist
