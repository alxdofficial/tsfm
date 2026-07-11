#!/usr/bin/env python3
"""Re-fit the ConSE-tier baseline zero-shot heads on the CURRENT global label vocabulary.

The ConSE baselines (crosshar, limubert, ssl_wearables) each train a single classifier whose
output vocabulary is benchmark_data/processed/limubert/global_label_mapping.json. Whenever that
vocabulary changes -- e.g. adding capture24 as an 11th train set, or correcting a dataset label
(dsads lying_side -> lying_right_side) -- every cached head becomes stale (wrong class count and
index alignment) and MUST be re-fit. run_baselines_v2 guards this with a shape check.

This driver re-fits the crosshar + limubert heads using the SAME building blocks and fit logic as
their legacy evaluate_*.main() (same seeds, same 90/10 split, same train_*_classifier), but WITHOUT
running the legacy v1 scoring (whose numbers are known not to reconcile with the v2 protocol). The
ssl_wearables head has its own dedicated fitter (evaluate_ssl_wearables.main, run with
TSFM_ALLOW_SSL_HEADFIT=1); re-run that separately.

Usage:
    python val_scripts/human_activity_recognition/refit_conse_heads.py [--baselines crosshar limubert]
Then evaluate with:
    python val_scripts/human_activity_recognition/run_baselines_v2.py --baselines crosshar limubert
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from val_scripts.human_activity_recognition.grouped_zero_shot import load_global_labels


def _split(n, seed):
    rng = np.random.RandomState(seed)
    idx = np.arange(n); rng.shuffle(idx)
    val_n = int(n * 0.1)
    return idx[val_n:], idx[:val_n]   # train_idx, val_idx


def _limubert_train_embeddings_live(L, bert, global_labels, device):
    """Compute LiMU-BERT sub-window training embeddings LIVE (the precomputed embed_*.npy cache
    under auxiliary_repos/ is not present in this checkout). Mirrors the adapter's forward path:
    raw data_20_120 -> normalize_for_limubert -> bert -> reshape_and_merge to (M,20,72) sub-windows,
    with per-sub-window local labels mapped to global indices."""
    from tqdm import tqdm
    all_emb, all_lab = [], []
    for ds in tqdm(L.TRAIN_DATASETS, desc="LiMU-BERT | train embeddings (live)"):
        data, lab_raw = L.load_raw_data(ds)                 # (N,120,6), (N,120,2)
        normed = L.normalize_for_limubert(data)
        embs = []
        with torch.no_grad():
            for s in range(0, len(normed), 512):
                b = torch.from_numpy(normed[s:s + 512]).float().to(device)
                embs.append(bert(b).cpu().numpy())
        emb = np.concatenate(embs, axis=0).astype(np.float32)   # (N,120,72)
        sub, local_labels, _ = L.reshape_and_merge(emb, lab_raw)  # (M,20,72), (M,)
        gl = L.map_local_to_global_labels(local_labels, ds, L.DATASET_CONFIG, global_labels)
        all_emb.append(sub); all_lab.append(gl)
    return np.concatenate(all_emb, 0), np.concatenate(all_lab, 0)


def refit_limubert(device, global_labels):
    import val_scripts.human_activity_recognition.evaluate_limubert as L
    print(f"\n[limubert] loading encoder + extracting training embeddings "
          f"({len(L.TRAIN_DATASETS)} train sets)...")
    bert = L.load_limubert_model(device); bert.eval()
    train_emb, train_lab = _limubert_train_embeddings_live(L, bert, global_labels, device)
    ti, vi = _split(len(train_emb), L.CLASSIFIER_SEED)
    print(f"[limubert] fitting GRU head: {len(ti)} train / {len(vi)} val sub-windows, "
          f"{len(global_labels)} classes")
    clf = L.train_gru_classifier(
        train_emb[ti], train_lab[ti], train_emb[vi], train_lab[vi],
        num_classes=len(global_labels), device=device, desc="LiMU-BERT | ZS GRU (refit)")
    out = L.OUTPUT_DIR / "limubert_zs_gru.pt"
    torch.save(clf.state_dict(), str(out))
    print(f"[limubert] saved -> {out}")


def refit_crosshar(device, global_labels):
    import val_scripts.human_activity_recognition.evaluate_crosshar as C
    if not Path(C.CROSSHAR_CHECKPOINT).exists():
        print(f"[crosshar] SKIP: backbone checkpoint missing at {C.CROSSHAR_CHECKPOINT}")
        return
    print(f"\n[crosshar] loading encoder + extracting training embeddings "
          f"({len(C.TRAIN_DATASETS)} train sets)...")
    model = C.load_crosshar_model(str(C.CROSSHAR_CHECKPOINT), device)
    train_emb, train_lab = C.load_crosshar_training_embeddings(model, global_labels, device)
    ti, vi = _split(len(train_emb), C.CLASSIFIER_SEED)
    print(f"[crosshar] fitting Transformer_ft head: {len(ti)} train / {len(vi)} val, "
          f"{len(global_labels)} classes")
    clf = C.train_transformer_classifier(
        train_emb[ti], train_lab[ti], train_emb[vi], train_lab[vi],
        num_classes=len(global_labels), device=device, desc="CrossHAR | ZS Transformer_ft (refit)")
    out = C.OUTPUT_DIR / "crosshar_zs_transformer.pt"
    torch.save(clf.state_dict(), str(out))
    print(f"[crosshar] saved -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baselines", nargs="+", default=["crosshar", "limubert"],
                    choices=["crosshar", "limubert"])
    args = ap.parse_args()

    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_labels = load_global_labels()
    print(f"Re-fitting ConSE heads on {len(global_labels)}-way vocab | device={device} "
          f"| baselines={args.baselines}")

    if "limubert" in args.baselines:
        refit_limubert(device, global_labels)
    if "crosshar" in args.baselines:
        refit_crosshar(device, global_labels)


if __name__ == "__main__":
    main()
