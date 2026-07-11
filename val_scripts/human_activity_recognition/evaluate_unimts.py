"""UniMTS (Unified Pre-training for Motion Time Series, NeurIPS'24) internals for the v2 harness.

UniMTS is a text-aligned accelerometer model: an ST-GCN over a 22-node SMPL skeleton graph,
contrastively aligned to a (fine-tuned) CLIP ViT-B/32 text tower. Zero-shot HAR = cosine
similarity between a window's IMU embedding and each label string's text embedding in the shared
512-d CLIP space => our "cosine" adapter tier.

Released weights (HF xiyuanz/UniMTS, checkpoint/UniMTS.pth) are ACCELEROMETER-ONLY (in_channels=3;
gyro/stft branches OFF -- verified from the checkpoint) and expect 20 Hz, m/s^2 WITH gravity, and
NO per-window normalization (an internal BatchNorm handles scaling). Input tensor is
(N, C=3, T=200, V=22, M=1): a single physical IMU is written into ONE SMPL joint's 3 accel channels
and the other 21 joints are zero-filled (UniMTS trains with random-joint masking, so zero joints are
valid at inference). We source from the 20 Hz limubert windows (already correct rate+units), drop
gyro, place accel at the dataset's placement joint, and wrap-pad 120->200 samples.

Faithful-recipe details verified by reading auxiliary_repos/UniMTS/{contrastive,model,data}.py.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UNIMTS_REPO = PROJECT_ROOT / "auxiliary_repos" / "UniMTS"
UNIMTS_CKPT = UNIMTS_REPO / "checkpoint" / "UniMTS.pth"
LIMU_DIR = PROJECT_ROOT / "benchmark_data" / "processed" / "limubert"

# --- input config (verified against the released code) ---
PAD_LEN = 200          # UniMTS pads/truncates every dataset to 200 samples (10 s @ 20 Hz)
N_JOINTS = 22          # SMPL skeleton graph nodes
EMB_DIM = 512          # shared CLIP ViT-B/32 space

# Placement -> SMPL joint index (from the graph child->parent tree + UniMTS data.py dataset
# assignments). A single IMU stream is placed at one joint; others zero-filled. Default = pelvis(0).
# 0=pelvis/root, 5=R-hip(pocket/thigh), 9=spine1(waist/lower-back), 21=R-wrist.
JOINT_BY_DS = {
    "motionsense": 5,    # front trouser pocket -> R-hip
    "mobiact":     5,    # trouser pocket -> R-hip
    "realworld":   9,    # waist -> spine1
    "inclusivehar": 9,   # waist -> spine1
    "harth":       9,    # lower back / thigh -> spine1
    "shoaib":      5,    # multi-position stream -> R-hip default
}
DEFAULT_JOINT = 0        # pelvis/root fallback for unlisted datasets


def load_unimts_model(device):
    """ContrastiveModule(acc-only ST-GCN + fine-tuned CLIP text tower) with UniMTS.pth loaded.

    The pretrained state_dict is ContrastiveModule.model.state_dict() (327 keys: CLIP text tower
    minus visual, logit_scale, text_projection, + acc.* ST-GCN); it loads into model.model with
    strict=True (0 missing / 0 unexpected). model.visual is deleted in the constructor.
    """
    if str(UNIMTS_REPO) not in sys.path:
        sys.path.insert(0, str(UNIMTS_REPO))
    from contrastive import ContrastiveModule  # noqa: E402  (repo-local import)

    args = SimpleNamespace(gyro=0, stft=0, stage="evaluation")  # acc-only, no finetune head
    model = ContrastiveModule(args).to(device)
    sd = torch.load(str(UNIMTS_CKPT), map_location=device, weights_only=True)
    missing, unexpected = model.model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, f"UniMTS load mismatch: missing={missing} unexpected={unexpected}"
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _joint_for(ds: str) -> int:
    return JOINT_BY_DS.get(ds, DEFAULT_JOINT)


@torch.no_grad()
def window_embeddings(ds: str, model, device, batch=256) -> np.ndarray:
    """(N,512) L2-normalized IMU embeddings from the 20 Hz limubert windows.

    limubert (N,120,6) -> drop gyro -> place accel at the dataset's SMPL joint (others zero) ->
    wrap-pad 120->200 -> (N,3,200,22,1) -> ST-GCN -> (N,512).
    """
    X = np.load(str(LIMU_DIR / ds / "data_20_120.npy")).astype(np.float32)  # (N,120,6)
    acc = X[:, :, 0:3]                                                       # accel only, 20Hz m/s^2 +gravity
    N, T, _ = acc.shape
    joint = _joint_for(ds)
    embs = []
    for s in range(0, N, batch):
        a = acc[s:s + batch]                                                # (b,T,3)
        allx = np.zeros((a.shape[0], T, N_JOINTS, 3), np.float32)
        allx[:, :, joint, :] = a                                            # single-joint placement
        if T < PAD_LEN:
            allx = np.pad(allx, ((0, 0), (0, PAD_LEN - T), (0, 0), (0, 0)), mode="wrap")
        else:
            allx = allx[:, :PAD_LEN]
        x = torch.from_numpy(allx).to(device).permute(0, 3, 1, 2).unsqueeze(-1)  # (b,3,200,22,1)
        e = model.encode_image(x)                                           # (b,512)
        e = e / e.norm(dim=-1, keepdim=True)
        embs.append(e.float().cpu().numpy())
    return np.concatenate(embs, axis=0)


@torch.no_grad()
def encode_labels(label_strings, model, device) -> np.ndarray:
    """(n_labels,512) L2-normalized text embeddings via the fine-tuned CLIP text tower in the
    checkpoint (model.encode_text). UniMTS uses the RAW label string (no 'a photo of' template)."""
    import clip
    tok = clip.tokenize([s.strip() for s in label_strings]).to(device)      # (n,77)
    t = model.encode_text(tok)                                              # (n,512)
    t = t / t.norm(dim=-1, keepdim=True)
    return t.float().cpu().numpy()
