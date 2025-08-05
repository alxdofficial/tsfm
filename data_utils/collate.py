import torch
from torch.nn.utils.rnn import pad_sequence

print(torch.cuda.is_available())            # True = GPU is available
print(torch.cuda.device_count())            # Number of GPUs
print(torch.cuda.current_device())          # Active device index
print(torch.cuda.get_device_name(0))        # GPU name (if available)

def pad_collate(batch):
    """
    Pads variable-length context windows to match the max length in batch.
    Returns batched dict with shape: (B, P, T, D), etc.
    """
    patches = [item['patches'] for item in batch]  # each (P, T, D)
    stats = [item['patch_mean_std_min_max'] for item in batch]  # each (4, P, D)
    timestamps = [item['timestamps'] for item in batch]
    metadata = [item['metadata'] for item in batch]

    # Pad patches → (B, P, T, D)
    padded_patches = pad_sequence(patches, batch_first=True)

    # Fix stats: (4, P, D) → (P, 4, D), then pad → (B, P, 4, D)
    stats = [s.permute(1, 0, 2) for s in stats]  # (P, 4, D)
    padded_stats = pad_sequence(stats, batch_first=True)  # (B, P, 4, D)
    padded_stats = padded_stats.permute(0, 2, 1, 3)  # (B, 4, P, D)

    # Pad timestamps to max P
    max_len = max(len(t) for t in timestamps)
    padded_timestamps = [ts + [ts[-1]] * (max_len - len(ts)) for ts in timestamps]

    return {
        "patches": padded_patches,              # (B, P, T, D)
        "patch_mean_std_min_max": padded_stats, # (B, 4, P, D)
        "timestamps": padded_timestamps,        # list of B lists of len P
        "metadata": metadata,
        "target": None
    }
