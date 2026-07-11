"""Few-shot driver for from-scratch supervised baselines (protocol v2).

run_baselines_v2.py has NO few-shot path (it only handles conse/cosine ZS
tiers). This module is the missing FS harness: for each ``tier='fewshot'``
adapter in base.REGISTRY and each test dataset it

  1. loads the 20 Hz limubert windows + subject-disjoint ground truth
     (base.load_gt -> keep_idx alignment),
  2. builds subject-disjoint train/val/test splits (ev2.subject_disjoint_split,
     seed=3431) and a balanced FS subsample of TRAIN
     (ev2.balanced_subsample_indices) at rate in {0.01, 0.10, 1.0},
  3. fits the adapter's normalizer on the TRAIN subsample only,
  4. trains the adapter's model FROM SCRATCH with early stopping on the
     subject-disjoint val split (val macro-F1),
  5. scores the held-out test split with base.score (offset-free v2 metrics).

Mirrors evaluate_tsfm_v2.py:evaluate_fewshot_subject_disjoint (same split +
subsample helpers, same seed 3431) so HALO and this floor are directly
comparable.

Usage:
    python val_scripts/human_activity_recognition/run_fewshot_v2.py
    python ... run_fewshot_v2.py --baselines deepconvlstm --datasets motionsense harth
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from val_scripts.human_activity_recognition import eval_v2 as ev2
from val_scripts.human_activity_recognition import baselines as B  # noqa: F401 (registers adapters)
from val_scripts.human_activity_recognition.baselines import base

OUTPUT_DIR = PROJECT_ROOT / "test_output" / "eval_v2"
FALLBACK_DATASETS = ["motionsense", "realworld", "mobiact", "shoaib", "harth", "inclusivehar"]

# ---- Pre-registered FS training schedule (from-scratch; do not tune post-hoc)
FS_SEED = ev2.BOOTSTRAP_SEED           # 3431 -- same as HALO FS split/subsample seed
FS_MAX_EPOCHS = 100                    # from-scratch needs more than a fine-tune; early-stopped
FS_PATIENCE = 15                       # epochs w/o val-macroF1 improvement before stop
FS_BATCH_SIZE = 64                     # paper=100; capped for small FS subsamples
FS_RATES = {"fs_1pct": 0.01, "fs_10pct": 0.10, "fs_full": 1.0}


def _seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _macro_f1(model, loader, device) -> float:
    from sklearn.metrics import f1_score
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            preds.extend(logits.argmax(1).cpu().numpy().tolist())
            gts.extend(yb.numpy().tolist())
    if not gts:
        return 0.0
    return float(f1_score(gts, preds, average="macro", zero_division=0))


def train_and_eval_fewshot(
    adapter,
    data: np.ndarray,          # (N, T, C) 20 Hz windows, aligned 1:1 with gt_names
    gt_names: List[str],       # N label strings (dataset vocab L_D)
    subjects: np.ndarray,      # (N,) subject id per window
    labels_ld: List[str],      # ordered class vocabulary
    rate: float,
    device: torch.device,
    *,
    seed: int = FS_SEED,
    max_epochs: int = FS_MAX_EPOCHS,
    patience: int = FS_PATIENCE,
    batch_size: int = FS_BATCH_SIZE,
) -> Dict:
    """One (dataset, rate) run -> base.score metric bundle on the test split."""
    _seed_everything(seed)
    name_to_idx = {n: i for i, n in enumerate(labels_ld)}
    y = np.array([name_to_idx[n] for n in gt_names], dtype=np.int64)

    tr_idx, va_idx, te_idx = ev2.subject_disjoint_split(subjects, seed=seed)
    tr_idx, counts = ev2.balanced_subsample_indices(
        tr_idx, gt_names, rate=rate, seed=seed, return_counts=True)

    # Min-max [0,1] normalization fit on the TRAIN subsample only.
    stats = adapter.fit_normalizer(data[tr_idx])
    Xtr = adapter.apply_normalizer(data[tr_idx], stats)
    Xva = adapter.apply_normalizer(data[va_idx], stats)
    Xte = adapter.apply_normalizer(data[te_idx], stats)

    def loader(X, idx, shuffle):
        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y[idx]).long())
        bs = min(batch_size, max(1, len(idx)))
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)

    train_loader = loader(Xtr, tr_idx, True)
    val_loader = loader(Xva, va_idx, False)
    test_loader = loader(Xte, te_idx, False)

    model = adapter.build_model(n_classes=len(labels_ld)).to(device)
    optimizer = adapter.make_optimizer(model)
    criterion = nn.CrossEntropyLoss()

    best_f1, best_state, wait = -1.0, None, 0
    for _ in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()
        vf1 = _macro_f1(model, val_loader, device)
        if vf1 > best_f1:
            best_f1, best_state, wait = vf1, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            preds.extend(model(xb.to(device)).argmax(1).cpu().numpy().tolist())

    gt_te = [labels_ld[i] for i in y[te_idx]]
    pred_te = [labels_ld[i] for i in preds]
    metrics = base.score(gt_te, pred_te, subjects[te_idx], extra={
        "n_train_windows": int(len(tr_idx)),
        "train_class_counts": counts,
        "n_test_subjects": int(np.unique(subjects[te_idx]).size),
        "best_val_f1_macro": float(best_f1),
        "split": "subject_disjoint",
    })
    if device.type == "cuda":
        del model, best_state
        torch.cuda.empty_cache()
    return metrics


def run_dataset(adapter, ds: str, device, rates=FS_RATES) -> Dict:
    L_D, gt_names, subjects, keep_idx = base.load_gt(ds)
    data = np.load(str(base.BENCH_LIMU / ds / "data_20_120.npy")).astype(np.float32)[keep_idx]
    out = {"n_windows": int(len(gt_names)), "n_classes": len(L_D)}
    for tag, rate in rates.items():
        m = train_and_eval_fewshot(adapter, data, gt_names, subjects, L_D, rate, device)
        out[tag] = m
        ci = (f"[{m['f1_macro_ci_lo']:.1f},{m['f1_macro_ci_hi']:.1f}]"
              if not m.get("ci_degenerate") else "[degenerate]")
        print(f"  {ds:12} {tag:8} F1={m['f1_macro']:5.1f} {ci} "
              f"bAcc={m['balanced_accuracy']:5.1f} Acc={m['accuracy']:5.1f} "
              f"(train={m['n_train_windows']})")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    fs_names = sorted(n for n, a in base.REGISTRY.items() if a.tier == "fewshot")
    ap.add_argument("--baselines", nargs="*", default=fs_names)
    ap.add_argument("--datasets", nargs="*", default=FALLBACK_DATASETS)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Protocol v2 FS | device={device} | fewshot registry={fs_names} | run={args.baselines}")

    for name in args.baselines:
        adapter = base.REGISTRY.get(name)
        if adapter is None or adapter.tier != "fewshot":
            print(f"!! '{name}' is not a registered fewshot baseline (have {fs_names})")
            continue
        adapter.setup(device)
        results = {"_baseline": name, "_tier": "fewshot", "_seed": FS_SEED}
        out_path = OUTPUT_DIR / f"fewshot_v2_{name}.json"
        print(f"\n{'#'*60}\n# {name.upper()} (FS, from-scratch)\n{'#'*60}")
        for ds in args.datasets:
            try:
                results[ds] = run_dataset(adapter, ds, device)
            except Exception as e:
                import traceback
                print(f"!! {name}/{ds} FAILED: {e}")
                traceback.print_exc()
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2, default=float)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
