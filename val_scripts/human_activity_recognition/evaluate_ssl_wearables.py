"""ssl-wearables (OxWearables harnet) internals + zero-shot ConSE head-fit.

Mirrors evaluate_crosshar.py / evaluate_limubert.py: this module holds the model
loader, the frozen-trunk feature extractor, the head architecture, and the DEFERRED
head-fit ``main()`` that trains the 87-way head on combined_train and caches it to
test_output/baseline_evaluation/ssl_wearables_zs_head.pt. The thin adapter
(baselines/ssl_wearables.py) imports the loader + window loader from here.

harnet input contract (verified against ~/.cache/torch/hub/OxWearables_ssl-wearables_main/
hubconf.py + sslearning/models/accNet.py + data_parsing/*.py):
  30 Hz, 3-ch accel-only, g-units WITH gravity, NO per-window standardization.

Window reconciliation: the eval grid is 6 s (== limubert 120-sample @ 20 Hz windows).
harnet5's native receptive field is 5 s / 150 samples @ 30 Hz -- the only harnet variant
that fits inside a 6 s window without padding or rate distortion. We therefore run harnet5
and center-crop each 6 s / 180-sample 30 Hz window to its central 5 s / 150 samples.
(harnet10 = 10 s / 300 samples and harnet30 = 30 s / 900 samples cannot be filled from a
6 s window without >=4 s of padding or up-sampling that breaks the 30 Hz kernel timing, so
they are NOT used on this grid.)
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from val_scripts.human_activity_recognition.grouped_zero_shot import (
    load_global_labels, map_local_to_global_labels,
)

BENCHMARK_DIR = PROJECT_ROOT / "benchmark_data"
LIMU_DIR = BENCHMARK_DIR / "processed" / "limubert"
SSL_DIR = BENCHMARK_DIR / "processed" / "ssl_wearables"
OUTPUT_DIR = PROJECT_ROOT / "test_output" / "baseline_evaluation"
DATASET_CONFIG_PATH = BENCHMARK_DIR / "dataset_config.json"

with open(DATASET_CONFIG_PATH) as f:
    DATASET_CONFIG = json.load(f)
TRAIN_DATASETS = DATASET_CONFIG["train_datasets"]

# --- model / window config ---
HARNET_NAME = "harnet5"     # 5 s @ 30 Hz -> 150 samples (fits the 6 s eval grid)
FULL_TS = 180               # 6 s @ 30 Hz (as produced by preprocess_ssl_wearables.py)
TS_LEN = 150               # harnet5 native input length (center crop of FULL_TS)
EMB_DIM = 512              # harnet5 trunk output (harnet10/30 -> 1024)
HEAD_HIDDEN = 512          # EvaClassifier hidden width (ssl-wearables downstream head)

# head-fit hyperparameters (parity with crosshar/limubert ConSE head-fit)
CLASSIFIER_EPOCHS = 100
CLASSIFIER_BATCH_SIZE = 512
CLASSIFIER_LR = 1e-3
CLASSIFIER_SEED = 3431
EMBED_BATCH = 512


def _hub_dir() -> Path:
    return Path(torch.hub.get_dir()) / "OxWearables_ssl-wearables_main"


def load_ssl_model(name: str, num_classes: int, device):
    """Load a pretrained OxWearables harnet with a fresh ``class_num``-way EvaClassifier head.

    The pretrained .mdl loads ONLY into ``feature_extractor`` (hubconf.load_weights filters
    out ``classifier.*``); the EvaClassifier head is randomly initialized. We freeze the trunk.
    """
    hubdir = _hub_dir()
    if hubdir.exists():
        model = torch.hub.load(str(hubdir), name, class_num=num_classes,
                               pretrained=True, source="local")
    else:  # offline cache miss -> fetch once from GitHub
        model = torch.hub.load("OxWearables/ssl-wearables", name, class_num=num_classes,
                               pretrained=True, source="github", trust_repo=True)
    model.to(device)
    for p in model.feature_extractor.parameters():
        p.requires_grad_(False)
    model.train(False)
    return model


def load_ssl_windows(ds: str) -> np.ndarray:
    """Load (N, TS_LEN, 3) 30 Hz g-with-gravity windows, center-cropped from FULL_TS.

    Aligned 1:1 with benchmark_data/processed/limubert/<ds>/label_20_120.npy.
    """
    path = SSL_DIR / ds / "data_30_180.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run: python benchmark_data/scripts/preprocess_ssl_wearables.py "
            f"--datasets {ds}")
    x = np.load(str(path)).astype(np.float32)          # (N, 180, 3)
    off = (x.shape[1] - TS_LEN) // 2
    return x[:, off:off + TS_LEN, :]                    # (N, 150, 3)


@torch.no_grad()
def extract_trunk_features(model, x_ntc: np.ndarray, device, batch=EMBED_BATCH) -> np.ndarray:
    """(N, TS_LEN, 3) g-with-gravity -> (N, EMB_DIM) frozen trunk features."""
    x = np.transpose(x_ntc, (0, 2, 1))                  # (N, 3, TS_LEN)
    feats = []
    for s in range(0, len(x), batch):
        b = torch.from_numpy(x[s:s + batch]).float().to(device)
        f = model.feature_extractor(b)                  # (B, C, 1)
        feats.append(f.flatten(1).cpu().numpy())        # (B, EMB_DIM)
    return np.concatenate(feats, axis=0).astype(np.float32)


def get_window_labels(labels_raw: np.ndarray, label_index: int = 0) -> np.ndarray:
    """Per-window majority activity label (min-subtracted); matches the crosshar/limubert
    head-fit label pipeline. (Test-time GT is handled offset-free by base.load_gt, not here.)"""
    act = labels_raw[:, :, label_index]
    act = act - int(np.min(act))
    return np.array([np.bincount(r.astype(int)).argmax() for r in act], dtype=np.int64)


def build_head(num_classes: int, device):
    """Fresh EvaClassifier(EMB_DIM -> HEAD_HIDDEN -> num_classes) -- ssl-wearables' own
    downstream head (accNet.EvaClassifier). Obtained from a throwaway harnet so the module
    definition stays identical to the released code."""
    return load_ssl_model(HARNET_NAME, num_classes, device).classifier.to(device)


# =============================================================================
# DEFERRED head-fit (do NOT run during smoke test; needs ssl-preprocessed train sets)
# =============================================================================

def main():
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    globals_labels = load_global_labels()

    model = load_ssl_model(HARNET_NAME, num_classes=len(globals_labels), device=device)

    # 1) frozen-trunk features + global labels over the ssl-preprocessed train sets.
    # Datasets whose accel cannot be expressed as physical g-with-gravity (harnet's input
    # contract) have no data_30_180.npy and are skipped LOUDLY here -- never silently. As of
    # this writing: kuhar (gravity-removed linear accel), recgym (min-max normalized [0,1],
    # non-physical), unimib_shar (source subject-map acc_labels.npy lost -> cannot re-export
    # ssl windows aligned to the limubert grid). See preprocess_ssl_wearables.INCOMPATIBLE_ACCEL.
    feats, labs, used, skipped = [], [], [], []
    for ds in TRAIN_DATASETS:
        ssl_path = SSL_DIR / ds / "data_30_180.npy"
        if not ssl_path.exists():
            skipped.append(ds); continue
        used.append(ds)
    print(f"ssl-wearables head-fit corpus: {len(used)}/{len(TRAIN_DATASETS)} train sets = {used}")
    if skipped:
        print(f"  EXCLUDED (no ssl-preprocessed data; incompatible/missing): {skipped}")
    for ds in tqdm(used, desc="ssl-wearables | train features"):
        x = load_ssl_windows(ds)                                    # (N,150,3)
        lab_raw = np.load(str(LIMU_DIR / ds / "label_20_120.npy"))   # (N,120,2)
        local = get_window_labels(lab_raw)
        gl = map_local_to_global_labels(local, ds, DATASET_CONFIG, globals_labels)
        feats.append(extract_trunk_features(model, x, device))
        labs.append(gl)
    X = np.concatenate(feats, 0); Y = np.concatenate(labs, 0)

    # 2) 90/10 split, train the EvaClassifier head (frozen trunk)
    rng = np.random.RandomState(CLASSIFIER_SEED)
    idx = np.arange(len(X)); rng.shuffle(idx)
    val_n = int(len(X) * 0.1); vi, ti = idx[:val_n], idx[val_n:]
    head = build_head(len(globals_labels), device)
    opt = torch.optim.Adam(head.parameters(), lr=CLASSIFIER_LR)
    crit = nn.CrossEntropyLoss()
    tl = DataLoader(TensorDataset(torch.from_numpy(X[ti]), torch.from_numpy(Y[ti])),
                    batch_size=CLASSIFIER_BATCH_SIZE, shuffle=True)
    best_acc, best_sd = -1.0, None
    for _ in tqdm(range(CLASSIFIER_EPOCHS), desc="ssl-wearables | ZS head"):
        head.train()
        for xb, yb in tl:
            opt.zero_grad()
            loss = crit(head(xb.to(device)), yb.to(device)); loss.backward(); opt.step()
        head.train(False)
        with torch.no_grad():
            va = (head(torch.from_numpy(X[vi]).to(device)).argmax(1).cpu().numpy() == Y[vi]).mean()
        if va > best_acc:
            best_acc, best_sd = va, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    if best_sd is not None:
        head.load_state_dict(best_sd)
    torch.save(head.state_dict(), str(OUTPUT_DIR / "ssl_wearables_zs_head.pt"))
    print(f"saved head (val_acc={best_acc:.3f}) -> {OUTPUT_DIR/'ssl_wearables_zs_head.pt'}")


if __name__ == "__main__":
    import os
    if os.environ.get("TSFM_ALLOW_SSL_HEADFIT") != "1":
        print("This trains the deferred ssl-wearables ConSE head. "
              "Set TSFM_ALLOW_SSL_HEADFIT=1 to run; then evaluate with:\n"
              "  python val_scripts/human_activity_recognition/run_baselines_v2.py --baselines ssl_wearables")
        raise SystemExit(1)
    main()
