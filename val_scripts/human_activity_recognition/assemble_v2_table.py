"""Assemble the v2 ZS-XD comparison table (HALO + baselines) from the per-model
result JSONs in test_output/eval_v2/. Prints a markdown table + averages."""
import json
from pathlib import Path

ED = Path(__file__).resolve().parent.parent.parent / "test_output" / "eval_v2"
DATASETS = ["motionsense", "realworld", "mobiact", "shoaib", "opportunity", "harth"]

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


def get_zs(payload, ds, subkey):
    if ds not in payload:
        return None
    node = payload[ds]
    if subkey:
        node = node.get(subkey, {})
    return node


def main():
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
        rows.append((label, tier, f1s, avg))

    # markdown
    hdr = "| Model | tier | " + " | ".join(DATASETS) + " | **avg** |"
    sep = "|" + "---|" * (len(DATASETS) + 3)
    print(hdr)
    print(sep)
    for label, tier, f1s, avg in rows:
        cells = " | ".join(f"{f1s[d]:.1f}" if f1s[d] is not None else "—" for d in DATASETS)
        print(f"| {label} | {tier} | {cells} | **{avg:.1f}** |" if avg is not None
              else f"| {label} | {tier} | {cells} | — |")

    print("\n† = ConSE bridge (closed-vocab classifier -> softmax over 87 train "
          "labels -> convex combo of SBERT label embeddings -> argmax over L_D). "
          "macro-F1, exact match, subject-disjoint not needed (zero-shot).")


if __name__ == "__main__":
    main()
