"""
Generate pre-registered label configs for evaluation protocol v2.

For each zero-shot test dataset, emits benchmark_data/eval_v2/labels/{dataset}.json:
  - labels:            L_D — the dataset's own label strings, verbatim from the
                        preprocessing metadata (which the datascripts converters
                        produced from the dataset's documented label names).
                        These are FROZEN once reviewed; the v2 protocol forbids
                        rephrasing at evaluation time.
  - idx_to_label:      the authoritative code->name mapping copied from
                        tsfm_eval metadata.json (label_native.npy codes are
                        activity_to_idx values — no offset arithmetic, ever).
  - common_classes:    exact-string 1:1 matches against the 87-label training
                        vocabulary (objective, computed).
  - proposed_semantic_pairs: SBERT nearest-neighbour candidates for the
                        common-classes table that are NOT exact matches.
                        Marked PENDING_REVIEW — a human must promote them to
                        common_classes (or delete them) before the
                        common-classes table is reported.

Usage:
    python benchmark_data/scripts/generate_eval_v2_labels.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_DIR = PROJECT_ROOT / "benchmark_data"
TSFM_EVAL_DIR = BENCHMARK_DIR / "processed" / "tsfm_eval"
GLOBAL_LABEL_PATH = BENCHMARK_DIR / "processed" / "limubert" / "global_label_mapping.json"
OUT_DIR = BENCHMARK_DIR / "eval_v2" / "labels"

# Frozen ConSE/bridge hyperparameter (pre-registered)
SBERT_MODEL = "all-MiniLM-L6-v2"
SEMANTIC_PAIR_MIN_COS = 0.60  # proposal threshold only; pairs still need human review

# Pre-registered v2 test set (decided 2026-07-02): ONE flat tier, no severe-OOD
# category. HARTH is a regular test dataset. VTT-ConIoT is dropped from the
# benchmark (its ~50% no-training-equivalent construction labels made every
# model's zero-shot number a coverage artifact rather than a capability signal).
# opportunity demoted to appendix 2026-07 (4 subjects -> degenerate CIs; object/ambient
# sensors, not phone/watch). Converter + data kept for an optional appendix row.
EVALUATED_DATASETS = [
    "motionsense",
    "realworld",
    "mobiact",
    "shoaib",
    "harth",
    "inclusivehar",
]


def main():
    test_datasets = EVALUATED_DATASETS

    with open(GLOBAL_LABEL_PATH) as f:
        train_vocab = json.load(f)["labels"]

    # SBERT for semantic-pair *proposals* (review-gated; not used in scoring here)
    from sentence_transformers import SentenceTransformer
    import numpy as np
    sbert = SentenceTransformer(SBERT_MODEL)
    train_emb = sbert.encode([l.replace("_", " ") for l in train_vocab],
                             normalize_embeddings=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for ds in test_datasets:
        meta_path = TSFM_EVAL_DIR / ds / "metadata.json"
        if not meta_path.exists():
            print(f"SKIP {ds}: no tsfm_eval metadata (run preprocess_tsfm_eval.py first)")
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        activity_to_idx = meta["activity_to_idx"]
        idx_to_label = {int(v): k for k, v in activity_to_idx.items()}
        labels = [idx_to_label[i] for i in sorted(idx_to_label)]

        # Exact-string common classes vs the training vocabulary (objective)
        common = sorted(set(labels) & set(train_vocab))

        # SBERT-proposed semantic 1:1 pairs for non-exact labels (PENDING_REVIEW)
        proposals = {}
        test_only = [l for l in labels if l not in common]
        if test_only:
            test_emb = sbert.encode([l.replace("_", " ") for l in test_only],
                                    normalize_embeddings=True)
            sims = test_emb @ train_emb.T  # (n_test_only, 87)
            for i, lbl in enumerate(test_only):
                j = int(np.argmax(sims[i]))
                cos = float(sims[i, j])
                if cos >= SEMANTIC_PAIR_MIN_COS:
                    proposals[lbl] = {
                        "train_label": train_vocab[j],
                        "cosine": round(cos, 4),
                        "status": "PENDING_REVIEW",
                    }

        config = {
            "dataset": ds,
            "protocol_version": "v2",
            "source": (
                "Label strings verbatim from benchmark_data/processed/tsfm_eval/"
                f"{ds}/metadata.json activity_to_idx (produced by datascripts/{ds}/"
                "convert.py from the dataset's documented label names). "
                "Frozen for eval v2 — no rephrasing at evaluation time."
            ),
            "labels": labels,
            "idx_to_label": {str(k): v for k, v in sorted(idx_to_label.items())},
            "n_classes": len(labels),
            "sampling_rate_hz": meta["sampling_rate_hz"],
            "n_subjects": len(meta.get("subject_to_idx", {})),
            "common_classes_exact": common,
            "proposed_semantic_pairs": proposals,
            "conse": {"text_encoder": SBERT_MODEL, "pooling": "mean", "top_T": 10},
        }

        out_path = OUT_DIR / f"{ds}.json"
        with open(out_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"{ds:14} {len(labels):3d} classes | {len(common):2d} exact-common | "
              f"{len(proposals):2d} proposed pairs -> {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
