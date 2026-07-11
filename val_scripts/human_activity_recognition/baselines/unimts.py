"""UniMTS adapter (cosine tier): text-aligned ST-GCN accelerometer encoder + fine-tuned CLIP
text tower -> per-window 512-d embeddings scored by cosine similarity to label-text embeddings.

Released weights are accelerometer-only; input comes from the 20 Hz limubert windows with a
single-joint SMPL placement per dataset. Recipe + input contract live in evaluate_unimts.py.
"""

import numpy as np

from .base import CosineAdapter, register


@register
class UniMTSAdapter(CosineAdapter):
    name = "unimts"

    def setup(self, device):
        import val_scripts.human_activity_recognition.evaluate_unimts as U
        model = U.load_unimts_model(device)
        return {"model": model}

    def window_embeddings(self, ds, state, device) -> np.ndarray:
        import val_scripts.human_activity_recognition.evaluate_unimts as U
        return U.window_embeddings(ds, state["model"], device)

    def encode_labels(self, L_D, state, device) -> np.ndarray:
        import val_scripts.human_activity_recognition.evaluate_unimts as U
        return U.encode_labels(L_D, state["model"], device)
