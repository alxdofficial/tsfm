"""
Generic baseline evaluation driver for protocol v2 (ZS-XD).

One loop over the adapter REGISTRY — no per-baseline dispatch. ConSE-tier
adapters produce per-window softmax over the 87 global labels (bridged to each
dataset's own vocabulary with ConSE); cosine-tier adapters produce embeddings +
text prototypes scored directly. Add a baseline by dropping a module in
`baselines/`; it appears here automatically.

Usage:
    python val_scripts/human_activity_recognition/run_baselines_v2.py
    python ... --baselines crosshar --datasets motionsense shoaib
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from val_scripts.human_activity_recognition import eval_v2
from val_scripts.human_activity_recognition import baselines as B

OUTPUT_DIR = PROJECT_ROOT / "test_output" / "eval_v2"
DATASETS = ["motionsense", "realworld", "mobiact", "shoaib", "opportunity", "harth"]


def run_one(name: str, datasets, device, sbert, out_path: Path) -> dict:
    adapter = B.REGISTRY[name]
    print(f"\n{'#'*60}\n# {name.upper()} (v2, tier={adapter.tier})\n{'#'*60}")
    state = adapter.setup(device)
    GLOBAL = B.global_labels()
    results = {"_baseline": name, "_tier": adapter.tier}

    for ds in datasets:
        L_D, gt_names, subjects, keep_idx = B.load_gt(ds)
        if adapter.tier == "conse":
            probs = adapter.window_probs(ds, state, device)[keep_idx]
            preds, info = eval_v2.conse_predict(probs, GLOBAL, L_D, encode=sbert)
            extra = {"reachability_lb": info["reachability_lb"],
                     "n_predicted_classes": info["n_predicted_classes"]}
        else:  # cosine
            emb = adapter.window_embeddings(ds, state, device)[keep_idx]
            text = adapter.encode_labels(L_D, state, device)
            preds = eval_v2.predict_from_similarity(emb @ text.T, L_D)
            extra = None
        results[ds] = B.score(gt_names, preds, subjects, extra)
        r = results[ds]
        ci = (f"[{r['f1_macro_ci_lo']:.1f},{r['f1_macro_ci_hi']:.1f}]"
              if not r.get("ci_degenerate") else "[degenerate]")
        print(f"  {ds:12} ZS-XD F1={r['f1_macro']:5.1f} {ci}  bAcc={r['balanced_accuracy']:5.1f}  "
              f"Acc={r['accuracy']:5.1f}" + (f"  reach_lb={extra['reachability_lb']:.2f}" if extra else ""))
        with open(out_path, "w") as f:   # incremental save
            json.dump(results, f, indent=2, default=float)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baselines", nargs="*", default=sorted(B.REGISTRY.keys()))
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sbert = eval_v2.get_sbert_encoder()
    print(f"Protocol v2 | device={device} | registry={sorted(B.REGISTRY)} | run={args.baselines}")

    for name in args.baselines:
        if name not in B.REGISTRY:
            print(f"!! unknown baseline '{name}' (known: {sorted(B.REGISTRY)})")
            continue
        out_path = OUTPUT_DIR / f"baseline_v2_{name}.json"
        try:
            run_one(name, args.datasets, device, sbert, out_path)
        except Exception as e:
            import traceback
            print(f"!! {name} FAILED: {e}")
            traceback.print_exc()
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
