"""
Baseline evaluation under protocol v2 (ZS-XD).

Routes each paper baseline through the v2 scoring in eval_v2.py, reusing the
cached zero-shot classifiers/models (no retraining):

  Closed-vocabulary classifiers -> ConSE bridge (Norouzi et al. 2014):
    * LiMU-BERT   : GRU softmax over 20-step sub-windows, mean-pooled to a
                    per-window distribution over the 87 global training labels.
    * MOMENT      : SVM-RBF softmax over one-vs-rest decision scores (T=1).
    * CrossHAR    : Transformer_ft softmax over the 87 global labels.
  Text-aligned -> Tier-1 cosine (no bridge):
    * LanHAR      : SciBERT text prototypes of L_D vs gravity-aligned sensor
                    embeddings.
  Generative -> map generated text to nearest L_D string:
    * LLaSA       : 7B, zero-shot only, gated behind --include-llasa.

All baselines run on the 20 Hz LiMU-BERT-format windows
(benchmark_data/processed/limubert/{ds}/{data_20_120.npy,label_20_120.npy}).
Ground truth is derived ONLY via eval_v2.window_ground_truth (offset-free);
the baselines' own get_window_labels (v1 min-subtraction, the HARTH bug) is
never used. See docs/baselines/EVALUATION_PROTOCOL_V2.md.

Usage:
    python val_scripts/human_activity_recognition/evaluate_baselines_v2.py
    python ... --baselines limubert crosshar --datasets motionsense shoaib
    python ... --include-llasa            # adds the 7B generative baseline
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from val_scripts.human_activity_recognition import eval_v2
from val_scripts.human_activity_recognition.grouped_zero_shot import load_global_labels

BENCH_LIMU = PROJECT_ROOT / "benchmark_data" / "processed" / "limubert"
OUTPUT_DIR = PROJECT_ROOT / "test_output" / "eval_v2"
CACHED_DIR = PROJECT_ROOT / "test_output" / "baseline_evaluation"

DATASETS = ["motionsense", "realworld", "mobiact", "shoaib", "opportunity", "harth"]
CONSE_BASELINES = ["limubert", "moment", "crosshar"]
COSINE_BASELINES = ["lanhar"]
GENERATIVE_BASELINES = ["llasa"]


# =============================================================================
# Shared ground truth (offset-free, v2)
# =============================================================================

def load_gt(ds: str):
    """Return (labels_raw, L_D, gt_names, subjects, keep_idx) for a dataset.

    labels_raw is the 20 Hz (N, W, 2) array; keep_idx indexes the original N
    windows so any per-window model output aligns via output[keep_idx].
    """
    cfg = eval_v2.load_label_config(ds)
    L_D = cfg["labels"]
    idx_to_label = {int(k): v for k, v in cfg["idx_to_label"].items()}  # int-cast!
    labels_raw = np.load(str(BENCH_LIMU / ds / "label_20_120.npy"))
    gt_names, subjects, keep_idx = eval_v2.window_ground_truth(labels_raw, idx_to_label)
    return labels_raw, L_D, gt_names, subjects, keep_idx


def score(gt_names, pred_names, subjects, extra: dict = None) -> dict:
    """v2 metric bundle: macro-F1 (primary) + balanced-acc + CIs + per-class."""
    m = eval_v2.classification_metrics(gt_names, pred_names)
    m.update(eval_v2.subject_bootstrap_ci(gt_names, pred_names, subjects, metric="f1_macro"))
    m.update({k + "_ba": v for k, v in
              eval_v2.subject_bootstrap_ci(gt_names, pred_names, subjects,
                                           metric="balanced_accuracy").items()
              if k.startswith("balanced_accuracy_ci")})
    m["per_class_f1"] = eval_v2.per_class_f1(gt_names, pred_names)
    if extra:
        m.update(extra)
    return m


# =============================================================================
# LiMU-BERT (ConSE)
# =============================================================================

def limubert_probs(ds: str, state: dict, device) -> np.ndarray:
    """Per-window softmax over 87 global labels via GRU on sub-windows."""
    import val_scripts.human_activity_recognition.evaluate_limubert as L
    bert, clf = state["bert"], state["clf"]
    raw = np.load(str(BENCH_LIMU / ds / "data_20_120.npy")).astype(np.float32)
    labels_raw = np.load(str(BENCH_LIMU / ds / "label_20_120.npy"))
    normed = L.normalize_for_limubert(raw)

    # LiMU-BERT embeddings (N, 120, 72) — forward-only inference.
    embs = []
    with torch.no_grad():
        for s in range(0, len(normed), 512):
            b = torch.from_numpy(normed[s:s + 512]).float().to(device)
            embs.append(bert(b).cpu().numpy())
    emb = np.concatenate(embs, axis=0).astype(np.float32)

    # Split into 20-step sub-windows; keep parent id per sub-window.
    sub, _sub_lab, parent = L.reshape_and_merge(emb, labels_raw)

    # Per-sub-window softmax over 87 labels.
    sub_probs = []
    with torch.no_grad():
        for s in range(0, len(sub), 512):
            b = torch.from_numpy(sub[s:s + 512]).float().to(device)
            sub_probs.append(F.softmax(clf(b), dim=1).cpu().numpy())
    sub_probs = np.concatenate(sub_probs, axis=0)

    # Aggregate sub-window -> window by MEAN softmax over children.
    N = len(emb)
    win = np.zeros((N, sub_probs.shape[1]), dtype=np.float64)
    cnt = np.zeros(N, dtype=np.int64)
    np.add.at(win, parent, sub_probs)
    np.add.at(cnt, parent, 1)
    nz = cnt > 0
    win[nz] /= cnt[nz][:, None]
    win[~nz] = 1.0 / win.shape[1]  # robustness: uncovered window -> uniform
    return win


def setup_limubert(device) -> dict:
    import val_scripts.human_activity_recognition.evaluate_limubert as L
    GLOBAL = load_global_labels()
    bert = L.load_limubert_model(device)
    bert.eval()
    clf = L.GRUClassifier(input_dim=L.EMB_DIM, num_classes=len(GLOBAL)).to(device)
    # First-party cached classifier (pure state_dict) — weights_only load.
    clf.load_state_dict(torch.load(str(CACHED_DIR / "limubert_zs_gru.pt"),
                                   map_location=device, weights_only=True))
    clf.train(False)
    return {"bert": bert, "clf": clf}


# =============================================================================
# MOMENT (ConSE)
# =============================================================================

def moment_probs(ds: str, state: dict, device) -> np.ndarray:
    import val_scripts.human_activity_recognition.evaluate_moment as M
    from scipy.special import softmax
    raw = np.load(str(BENCH_LIMU / ds / "data_20_120.npy")).astype(np.float32)
    test_emb = M.extract_moment_embeddings(state["model"], raw, device,
                                           batch_size=M.MOMENT_BATCH_SIZE)  # (N,6144)
    svm = state["svm"]
    scores = svm.decision_function(test_emb)  # (N, n_classes) ovr
    # Scatter into the full 87-wide column space (robust if SVM ever omits a class).
    N = scores.shape[0]
    full = np.full((N, 87), -1e9, dtype=np.float64)
    full[:, svm.classes_.astype(int)] = scores
    return softmax(full, axis=1)  # T=1 (pre-registered default; see protocol doc)


def setup_moment(device) -> dict:
    import joblib
    import val_scripts.human_activity_recognition.evaluate_moment as M
    model = M.load_moment_model(device)
    # First-party artifact produced by our own evaluate_moment.py (a trusted
    # sklearn SVC); joblib.load is required to restore the fitted estimator.
    svm = joblib.load(str(CACHED_DIR / "moment_zs_svm.pkl"))
    return {"model": model, "svm": svm}


# =============================================================================
# CrossHAR (ConSE)
# =============================================================================

def crosshar_probs(ds: str, state: dict, device) -> np.ndarray:
    import val_scripts.human_activity_recognition.evaluate_crosshar as C
    raw = np.load(str(BENCH_LIMU / ds / "data_20_120.npy")).astype(np.float32)
    emb = C.extract_crosshar_embeddings(state["enc"], raw, device, batch_size=512)  # (N,120,72)
    clf = state["clf"]
    probs = []
    with torch.no_grad():
        for s in range(0, len(emb), 512):
            b = torch.from_numpy(emb[s:s + 512]).float().to(device)
            probs.append(F.softmax(clf(b), dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def setup_crosshar(device) -> dict:
    import val_scripts.human_activity_recognition.evaluate_crosshar as C
    GLOBAL = load_global_labels()
    enc = C.load_crosshar_model(str(C.CROSSHAR_CHECKPOINT), device)
    clf = C.TransformerClassifier(input_dim=C.EMB_DIM, num_classes=len(GLOBAL)).to(device)
    # First-party cached classifier (pure state_dict) — weights_only load.
    clf.load_state_dict(torch.load(str(CACHED_DIR / "crosshar_zs_transformer.pt"),
                                   map_location=device, weights_only=True))
    clf.train(False)
    return {"enc": enc, "clf": clf}


# =============================================================================
# LanHAR (Tier-1 cosine)
# =============================================================================

def lanhar_predict(ds: str, state: dict, device, keep_idx: np.ndarray, L_D: List[str]) -> List[str]:
    import val_scripts.human_activity_recognition.evaluate_lanhar as LH
    model, tokenizer = state["model"], state["tokenizer"]
    raw, _ = LH.load_raw_data(ds)
    raw = LH.gravity_align_dataset(raw)  # mandatory (fs=20Hz), matches training
    test_emb = LH.extract_lanhar_embeddings(model, raw, device, batch_size=256)  # (N,768) L2-normed

    protos_dict = LH.build_text_protos(L_D)
    text_protos = LH.build_zero_shot_prototypes(model, tokenizer, protos_dict, L_D, device)
    text_np = text_protos.detach().cpu().numpy()

    emb_kept = test_emb[keep_idx]
    sims = emb_kept @ text_np.T  # (N', L)
    return eval_v2.predict_from_similarity(sims, L_D)


def setup_lanhar(device) -> dict:
    import val_scripts.human_activity_recognition.evaluate_lanhar as LH
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(LH.BERT_MODEL_NAME)
    model = LH.LanHARModel(LH.BERT_MODEL_NAME).float().to(device)
    model.load_state_dict(torch.load(str(CACHED_DIR / "lanhar_model.pt"),
                                     map_location=device, weights_only=True))
    model.eval()
    return {"model": model, "tokenizer": tokenizer}


# =============================================================================
# LLaSA (generative)
# =============================================================================

def llasa_predict(ds: str, state: dict, device, keep_idx: np.ndarray, L_D: List[str]) -> List[str]:
    import re as _re
    import val_scripts.human_activity_recognition.evaluate_llasa as LL
    model, tokenizer = state["model"], state["tokenizer"]
    sbert = state["sbert"]
    data = np.load(str(BENCH_LIMU / ds / "data_20_120.npy")).astype(np.float32)
    Le = sbert(L_D)

    preds = []
    for i in tqdm(keep_idx, desc=f"LLaSA | {ds}", leave=False):
        p = LL.classify_sample(model, tokenizer, data[i], L_D, device)
        if p not in L_D:  # 'unclear' or free text -> SBERT nearest L_D
            re_emb = sbert([_re.sub(r"[^\w\s]", "", str(p)).strip().lower() or "unknown"])
            p = L_D[int((re_emb @ Le.T).argmax())]
        preds.append(p)
    return preds


def setup_llasa(device) -> dict:
    import val_scripts.human_activity_recognition.evaluate_llasa as LL
    tokenizer, model, _ = LL.load_llasa_model(device="cuda")
    return {"model": model, "tokenizer": tokenizer, "sbert": eval_v2.get_sbert_encoder()}


# =============================================================================
# Driver
# =============================================================================

CONSE_PROBS = {"limubert": limubert_probs, "moment": moment_probs, "crosshar": crosshar_probs}
SETUPS = {"limubert": setup_limubert, "moment": setup_moment, "crosshar": setup_crosshar,
          "lanhar": setup_lanhar, "llasa": setup_llasa}


def run_baseline(name: str, datasets: List[str], device, out_path: Path) -> dict:
    print(f"\n{'#'*64}\n# {name.upper()} (v2)\n{'#'*64}")
    state = SETUPS[name](device)
    GLOBAL = load_global_labels()
    sbert = eval_v2.get_sbert_encoder()
    results = {"_baseline": name, "_tier": ("conse" if name in CONSE_BASELINES
                                            else "cosine" if name in COSINE_BASELINES
                                            else "generative")}
    for ds in datasets:
        _, L_D, gt_names, subjects, keep_idx = load_gt(ds)
        if name in CONSE_BASELINES:
            probs = CONSE_PROBS[name](ds, state, device)[keep_idx]
            preds, info = eval_v2.conse_predict(probs, GLOBAL, L_D, encode=sbert)
            extra = {"reachability_lb": info["reachability_lb"],
                     "n_predicted_classes": info["n_predicted_classes"]}
        elif name in COSINE_BASELINES:
            preds = lanhar_predict(ds, state, device, keep_idx, L_D)
            extra = None
        else:  # generative
            preds = llasa_predict(ds, state, device, keep_idx, L_D)
            extra = None
        results[ds] = score(gt_names, preds, subjects, extra)
        r = results[ds]
        ci = f"[{r['f1_macro_ci_lo']:.1f},{r['f1_macro_ci_hi']:.1f}]" if not r.get("ci_degenerate") else "[degenerate]"
        print(f"  {ds:12} ZS-XD F1={r['f1_macro']:5.1f} {ci}  bAcc={r['balanced_accuracy']:5.1f}  "
              f"Acc={r['accuracy']:5.1f}" + (f"  reach_lb={extra['reachability_lb']:.2f}" if extra else ""))
        with open(out_path, "w") as f:  # incremental save
            json.dump(results, f, indent=2, default=float)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baselines", nargs="*", default=CONSE_BASELINES + COSINE_BASELINES)
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--include-llasa", action="store_true", help="add the 7B generative baseline (ZS-only, slow)")
    args = ap.parse_args()

    baselines = list(args.baselines)
    if args.include_llasa and "llasa" not in baselines:
        baselines.append("llasa")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Protocol v2 | device={device} | baselines={baselines} | datasets={args.datasets}")

    for name in baselines:
        out_path = OUTPUT_DIR / f"baseline_v2_{name}.json"
        try:
            run_baseline(name, args.datasets, device, out_path)
        except Exception as e:  # one baseline failing must not sink the rest
            import traceback
            print(f"!! {name} FAILED: {e}")
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
