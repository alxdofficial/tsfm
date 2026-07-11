"""
Generic baseline evaluation driver for protocol v2 (ZS-XD).

One loop over the adapter REGISTRY — no per-baseline dispatch. ConSE-tier
adapters produce per-window softmax over the global baseline labels (bridged to each
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
DATASET_CONFIG_PATH = PROJECT_ROOT / "benchmark_data" / "dataset_config.json"
FALLBACK_DATASETS = ["motionsense", "realworld", "mobiact", "shoaib", "harth", "inclusivehar"]


def load_default_datasets() -> list[str]:
    if not DATASET_CONFIG_PATH.exists():
        return FALLBACK_DATASETS
    with open(DATASET_CONFIG_PATH) as f:
        cfg = json.load(f)
    return cfg.get("zero_shot_datasets", FALLBACK_DATASETS)


def run_one(name: str, datasets, device, sbert, out_path: Path) -> dict:
    adapter = B.REGISTRY[name]
    print(f"\n{'#'*60}\n# {name.upper()} (v2, tier={adapter.tier})\n{'#'*60}")
    state = adapter.setup(device)
    GLOBAL = B.global_labels()
    results = {"_baseline": name, "_tier": adapter.tier,
               "_requested_datasets": list(datasets), "_status": "incomplete"}
    failed: dict = {}
    # Never write the FINAL path incrementally: a mid-run crash must not leave a
    # partial file that reads as a complete result. Stream to a .partial sidecar and
    # only atomically promote it to out_path once EVERY requested dataset succeeded.
    partial_path = out_path.with_suffix(".partial.json")

    for ds in datasets:
        try:
            L_D, gt_names, subjects, keep_idx = B.load_gt(ds)
            if adapter.tier == "conse":
                probs = adapter.window_probs(ds, state, device)[keep_idx]
                if probs.shape[1] != len(GLOBAL):
                    raise ValueError(
                        f"{name}/{ds}: classifier emits {probs.shape[1]} classes, "
                        f"but global_label_mapping.json has {len(GLOBAL)} labels. "
                        "Rebuild the cached classifier and global label mapping together."
                    )
                preds, info = eval_v2.conse_predict(probs, GLOBAL, L_D, encode=sbert)
                extra = {"reachability_lb": info["reachability_lb"],
                         "n_predicted_classes": info["n_predicted_classes"]}
            elif adapter.tier == "l1":
                # Bespoke tier (NormWear): window + label embeddings in an asymmetric learned space,
                # scored by NEGATIVE Manhattan distance (native metric is L1 argmin). Not a dot product.
                emb = adapter.window_embeddings(ds, state, device)[keep_idx]
                text = adapter.encode_labels(L_D, state, device)
                scores = -np.abs(emb[:, None, :] - text[None, :, :]).sum(-1)   # (N,C), higher=better
                preds = eval_v2.predict_from_similarity(scores, L_D)
                extra = None
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
        except Exception as e:
            import traceback
            failed[ds] = repr(e)
            print(f"  !! {name}/{ds} FAILED: {e}")
            traceback.print_exc()
        with open(partial_path, "w") as f:   # incremental save to the SIDECAR only
            json.dump(results, f, indent=2, default=float)

    results["_failed_datasets"] = failed
    if failed:
        results["_status"] = "failed"
        with open(partial_path, "w") as f:
            json.dump(results, f, indent=2, default=float)
        raise RuntimeError(
            f"{name}: {len(failed)}/{len(datasets)} dataset(s) failed "
            f"({sorted(failed)}); partial kept at {partial_path.name}, "
            f"final {out_path.name} NOT produced")
    results["_status"] = "complete"
    with open(partial_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    partial_path.replace(out_path)   # atomic promote: final exists only when complete
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baselines", nargs="*", default=sorted(B.REGISTRY.keys()))
    ap.add_argument("--datasets", nargs="*", default=load_default_datasets())
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sbert = eval_v2.get_sbert_encoder()
    print(f"Protocol v2 | device={device} | registry={sorted(B.REGISTRY)} | run={args.baselines}")

    failures, ran = [], []
    for name in args.baselines:
        if name not in B.REGISTRY:
            print(f"!! unknown baseline '{name}' (known: {sorted(B.REGISTRY)})")
            failures.append(name)
            continue
        if B.REGISTRY[name].tier not in ("conse", "cosine", "l1"):
            # Non-ZS tiers (e.g. 'fewshot' DeepConvLSTM) are handled by their own
            # driver (run_fewshot_v2.py); skip them here rather than error.
            print(f".. skipping '{name}' (tier={B.REGISTRY[name].tier}); run its own driver")
            continue
        out_path = OUTPUT_DIR / f"baseline_v2_{name}.json"
        # Drop any stale final output BEFORE running so a crash can't leave an old
        # complete-looking file in place; the final is (re)created only on success.
        if out_path.exists():
            out_path.unlink()
        ran.append(name)
        try:
            run_one(name, args.datasets, device, sbert, out_path)
        except Exception as e:
            print(f"!! {name} FAILED: {e}")
            failures.append(name)
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if failures:
        print(f"\n!! {len(failures)} baseline(s) FAILED: {sorted(failures)}")
        sys.exit(1)
    print(f"\n✓ all {len(ran)} baseline(s) complete, every requested dataset present.")


if __name__ == "__main__":
    main()
