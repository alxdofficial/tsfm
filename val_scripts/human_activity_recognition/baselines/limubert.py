"""LiMU-BERT adapter (ConSE tier): frozen LiMU-BERT encoder + cached GRU
classifier. GRU runs on 20-step sub-windows; per-sub-window softmaxes are
mean-pooled per parent window into a per-window distribution over 87 labels."""

import numpy as np
import torch
import torch.nn.functional as F

from .base import CACHED_DIR, BENCH_LIMU, ConSEAdapter, global_labels, register


@register
class LiMUBERTAdapter(ConSEAdapter):
    name = "limubert"

    def setup(self, device):
        import val_scripts.human_activity_recognition.evaluate_limubert as L
        bert = L.load_limubert_model(device)
        bert.eval()
        clf = L.GRUClassifier(input_dim=L.EMB_DIM, num_classes=len(global_labels())).to(device)
        # First-party cached classifier (pure state_dict) — weights_only load.
        clf.load_state_dict(torch.load(str(CACHED_DIR / "limubert_zs_gru.pt"),
                                       map_location=device, weights_only=True))
        clf.train(False)
        return {"bert": bert, "clf": clf}

    def window_probs(self, ds, state, device):
        import val_scripts.human_activity_recognition.evaluate_limubert as L
        bert, clf = state["bert"], state["clf"]
        raw = np.load(str(BENCH_LIMU / ds / "data_20_120.npy")).astype(np.float32)
        labels_raw = np.load(str(BENCH_LIMU / ds / "label_20_120.npy"))
        normed = L.normalize_for_limubert(raw)

        embs = []
        with torch.no_grad():
            for s in range(0, len(normed), 512):
                b = torch.from_numpy(normed[s:s + 512]).float().to(device)
                embs.append(bert(b).cpu().numpy())
        emb = np.concatenate(embs, axis=0).astype(np.float32)   # (N,120,72)

        sub, _sub_lab, parent = L.reshape_and_merge(emb, labels_raw)  # 20-step sub-windows
        sub_probs = []
        with torch.no_grad():
            for s in range(0, len(sub), 512):
                b = torch.from_numpy(sub[s:s + 512]).float().to(device)
                sub_probs.append(F.softmax(clf(b), dim=1).cpu().numpy())
        sub_probs = np.concatenate(sub_probs, axis=0)

        # Mean-pool sub-window softmaxes back to per-window distributions.
        N = len(emb)
        win = np.zeros((N, sub_probs.shape[1]), dtype=np.float64)
        cnt = np.zeros(N, dtype=np.int64)
        np.add.at(win, parent, sub_probs)
        np.add.at(cnt, parent, 1)
        nz = cnt > 0
        win[nz] /= cnt[nz][:, None]
        win[~nz] = 1.0 / win.shape[1]   # uncovered window -> uniform (robustness)
        return win
