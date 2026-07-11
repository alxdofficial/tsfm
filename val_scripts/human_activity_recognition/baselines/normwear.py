"""NormWear adapter (l1 tier): channel-independent ViT + query-conditioned MSiTF aggregator +
TinyLlama text encoder -> per-window 2048-d embedding scored by MANHATTAN (L1) distance (argmin)
against label-text embeddings. NOT cosine -- the signal/text spaces are asymmetric and NormWear's
native metric is L1. The driver's "l1" branch computes -|win-lab|_1 (higher=better). Recipe +
input contract live in evaluate_normwear.py.
"""

from .base import CosineAdapter, register


@register
class NormWearAdapter(CosineAdapter):
    name = "normwear"
    tier = "l1"   # bespoke: L1-distance argmin, handled by run_baselines_v2's l1 branch

    def setup(self, device):
        import val_scripts.human_activity_recognition.evaluate_normwear as NW
        model = NW.load_normwear_model(device)
        query_emb = NW.compute_query(model)   # (1,2048), reused for every window
        return {"model": model, "query_emb": query_emb}

    def window_embeddings(self, ds, state, device):
        import val_scripts.human_activity_recognition.evaluate_normwear as NW
        return NW.window_embeddings(ds, state["model"], device, query_emb=state["query_emb"])

    def encode_labels(self, L_D, state, device):
        import val_scripts.human_activity_recognition.evaluate_normwear as NW
        return NW.encode_labels(L_D, state["model"], device)
