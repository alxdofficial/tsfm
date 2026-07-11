"""ssl-wearables adapter (ConSE tier): frozen OxWearables harnet Resnet trunk
(30 Hz, 3-ch accelerometer, g-units WITH gravity) + cached EvaClassifier head ->
per-window softmax over the 87 global baseline labels.

Reads benchmark_data/processed/ssl_wearables/<ds>/data_30_180.npy (produced by
benchmark_data/scripts/preprocess_ssl_wearables.py); NOT the 20 Hz m/s^2 limubert copy.
The cached head (ssl_wearables_zs_head.pt) is fit by evaluate_ssl_wearables.main()
(deferred training; run with TSFM_ALLOW_SSL_HEADFIT=1). Faithful-recipe details and the
input contract live in evaluate_ssl_wearables.py.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .base import CACHED_DIR, ConSEAdapter, global_labels, register


@register
class SSLWearablesAdapter(ConSEAdapter):
    name = "ssl_wearables"

    def setup(self, device):
        import val_scripts.human_activity_recognition.evaluate_ssl_wearables as S
        model = S.load_ssl_model(S.HARNET_NAME, num_classes=len(global_labels()), device=device)
        # First-party cached head (pure state_dict) — weights_only load.
        head_sd = torch.load(str(CACHED_DIR / "ssl_wearables_zs_head.pt"),
                             map_location=device, weights_only=True)
        model.classifier.load_state_dict(head_sd)
        model.train(False)
        return {"model": model}

    def window_probs(self, ds, state, device):
        import val_scripts.human_activity_recognition.evaluate_ssl_wearables as S
        model = state["model"]
        x = S.load_ssl_windows(ds)                 # (N, 150, 3) 30Hz g-with-gravity
        x = np.transpose(x, (0, 2, 1))             # (N, 3, 150)
        probs = []
        with torch.no_grad():
            for s in range(0, len(x), 512):
                b = torch.from_numpy(x[s:s + 512]).float().to(device)
                probs.append(F.softmax(model(b), dim=1).cpu().numpy())
        return np.concatenate(probs, axis=0)       # (N, 87)
