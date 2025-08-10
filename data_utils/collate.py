import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def pad_collate(batch):
    """
    Pads variable-length context windows to max P in the batch.
    Returns:
      - patches: (B, P, T, D) float32
      - patch_mean_std_min_max: (B, 4, P, D) float32
      - rel_ms: (B, P) float32   [ms since first patch of each item]
      - pad_mask: (B, P) bool    [True=valid, False=pad]
      - timestamps: original (optionally keep), metadata, target
    """
    patches = [item['patches'] for item in batch]         # (P,T,D)
    stats   = [item['patch_mean_std_min_max'] for item in batch]  # (4,P,D)
    stamps  = [item['timestamps'] for item in batch]
    meta    = [item['metadata'] for item in batch]

    # lengths
    lengths = [p.size(0) for p in patches]
    B, maxP = len(batch), max(lengths)

    # pad patches -> (B,P,T,D)
    padded_patches = pad_sequence(patches, batch_first=True, padding_value=0.0)

    # stats: (4,P,D) -> (P,4,D) pad -> (B,P,4,D) -> (B,4,P,D)
    stats_P4D = [s.permute(1,0,2) for s in stats]
    padded_stats = pad_sequence(stats_P4D, batch_first=True, padding_value=0.0).permute(0,2,1,3)

    # rel_ms per item, then pad to (B,P)
    rel_ms_list = []
    for ts in stamps:
        arr = np.asarray(ts)
        if np.issubdtype(arr.dtype, np.datetime64):
            base = arr[0]
            rel = (arr - base).astype('timedelta64[ms]').astype(np.int64).astype(np.float32)
        else:
            rel = arr.astype(np.float32)
        rel_ms_list.append(torch.from_numpy(rel))
    padded_rel_ms = pad_sequence(rel_ms_list, batch_first=True, padding_value=0.0)  # (B,P)

    # pad mask (True=valid)
    pad_mask = torch.zeros(B, maxP, dtype=torch.bool)
    for i, L in enumerate(lengths):
        pad_mask[i, :L] = True

    # (optional) keep timestamps padded if you still want them around
    max_len = max(len(t) for t in stamps)
    padded_timestamps = [ts + [ts[-1]] * (max_len - len(ts)) for ts in stamps]

    return {
        "patches": padded_patches,                     # (B,P,T,D)
        "patch_mean_std_min_max": padded_stats,        # (B,4,P,D)
        "rel_ms": padded_rel_ms,                       # (B,P)
        "pad_mask": pad_mask,                          # (B,P) True=valid
        "timestamps": padded_timestamps,               # list (optional)
        "metadata": meta,
        "target": None
    }
