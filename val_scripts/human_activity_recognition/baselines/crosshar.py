"""CrossHAR adapter (ConSE tier): masked-Transformer encoder + cached
Transformer_ft classifier -> per-window softmax over the baseline classifier labels."""

import numpy as np
import torch
import torch.nn.functional as F

from .base import CACHED_DIR, BENCH_LIMU, ConSEAdapter, global_labels, register


@register
class CrossHARAdapter(ConSEAdapter):
    name = "crosshar"

    def setup(self, device):
        import val_scripts.human_activity_recognition.evaluate_crosshar as C
        enc = C.load_crosshar_model(str(C.CROSSHAR_CHECKPOINT), device)
        clf = C.TransformerClassifier(input_dim=C.EMB_DIM, num_classes=len(global_labels())).to(device)
        # First-party cached classifier (pure state_dict) — weights_only load.
        clf.load_state_dict(torch.load(str(CACHED_DIR / "crosshar_zs_transformer.pt"),
                                       map_location=device, weights_only=True))
        clf.train(False)
        return {"enc": enc, "clf": clf}

    def window_probs(self, ds, state, device):
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
