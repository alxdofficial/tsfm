"""Assemble the v2 ZS-XD comparison table (HALO + baselines) from the per-model
result JSONs in test_output/eval_v2/. Prints a markdown table + averages."""
import json
from pathlib import Path

ED = Path(__file__).resolve().parent.parent.parent / "test_output" / "eval_v2"
DATASETS = ["motionsense", "realworld", "mobiact", "shoaib", "harth", "inclusivehar"]

# (label, json file, key path to the zs metrics dict, tier)
# Baseline set (post V2 cleanup): CrossHAR + LiMU-BERT kept; MOMENT/LanHAR/LLaSA
# dropped (slow / weak / undeployable). UniMTS + ssl-wearables to be added as
# adapters (released weights) — their rows appear here once their JSONs exist.
MODELS = [
    ("HALO (Small-Deep)", "tsfm_v2_native_native.json", "zs_xd", "text-aligned"),
    ("HALO (parity)", "tsfm_v2_neutral_20.json", "zs_xd", "text-aligned"),
    ("CrossHAR †", "baseline_v2_crosshar.json", None, "conse"),
    ("LiMU-BERT †", "baseline_v2_limubert.json", None, "conse"),
    ("UniMTS", "baseline_v2_unimts.json", None, "cosine"),          # planned
    ("ssl-wearables †", "baseline_v2_ssl_wearables.json", None, "conse"),  # planned
]

# Trainable/active parameter count per model — disclosed so the ~400x capacity gap is visible
# (the single largest HALO advantage). "released" = frozen released-weights foundation model.
PARAMS = {
    "HALO (Small-Deep)": "~25.8M",
    "HALO (parity)": "~25.8M",
    "CrossHAR †": "~62.6K",
    "LiMU-BERT †": "~62.6K",
    "UniMTS": "released",
    "ssl-wearables †": "released",
}


def get_zs(payload, ds, subkey):
    if ds not in payload:
        return None
    node = payload[ds]
    if subkey:
        node = node.get(subkey, {})
    return node


def main():
    N = len(DATASETS)
    rows = []
    for label, fn, subkey, tier in MODELS:
        p = ED / fn
        if not p.exists():
            continue
        data = json.load(open(p))
        f1s = {}
        for ds in DATASETS:
            m = get_zs(data, ds, subkey)
            f1s[ds] = m.get("f1_macro") if m else None
        avail = [v for v in f1s.values() if v is not None]
        avg = sum(avail) / len(avail) if avail else None
        rows.append((label, tier, PARAMS.get(label, "—"), f1s, avg, len(avail)))

    # markdown
    hdr = "| Model | tier | params | " + " | ".join(DATASETS) + f" | **avg (n/{N})** |"
    sep = "|" + "---|" * (N + 4)
    print(hdr)
    print(sep)
    for label, tier, params, f1s, avg, n in rows:
        cells = " | ".join(f"{f1s[d]:.1f}" if f1s[d] is not None else "—" for d in DATASETS)
        if avg is None:
            avgcell = "—"
        else:
            # never place an average over a partial support next to a full-support one
            # as if comparable: annotate the count and flag incomplete rows.
            avgcell = f"**{avg:.1f}** ({n}/{N})" + (" ⚠" if n < N else "")
        print(f"| {label} | {tier} | {params} | {cells} | {avgcell} |")

    print("\n⚠ = averaged over fewer than the full held-out set (an N/A cell) — NOT directly "
          "comparable to a full-support average; read per-dataset cells instead.")
    print("params = trainable/active parameters (HALO Small-Deep ~25.8M active; ConSE baselines "
          "~62.6K encoder + a small head; 'released' = frozen released-weights foundation model). "
          "The ~400x capacity gap is HALO's largest disclosed advantage.")
    print("\n† = ConSE bridge (closed-vocab classifier -> softmax over 87 train "
          "labels -> convex combo of SBERT label embeddings -> argmax over L_D). "
          "macro-F1, exact match, subject-disjoint not needed (zero-shot).")


if __name__ == "__main__":
    main()
