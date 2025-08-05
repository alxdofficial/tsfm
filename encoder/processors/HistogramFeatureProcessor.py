import torch
import torch.nn as nn
import numpy as np
class HistogramFeatureProcessor:
    def __init__(self, num_bins=10):
        self.num_bins = num_bins
        self.feature_dim = num_bins + 1
        self.norm = nn.LayerNorm(self.feature_dim)

    def process(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch: (B, T, D)
        Returns:
            (B, D, num_bins + 1)
        """
        B, T, D = patch.shape
        out = torch.zeros((B, D, self.feature_dim), dtype=torch.float32, device=patch.device)

        for b in range(B):
            for d in range(D):
                x = patch[b, :, d]
                min_, max_ = x.min(), x.max()
                hist = torch.histc(x, bins=self.num_bins, min=min_.item(), max=max_.item())
                proportions = hist / T
                probs = proportions[proportions > 0]
                entropy = -torch.sum(probs * torch.log2(probs)) if len(probs) > 0 else torch.tensor(0.0, device=x.device)
                out[b, d] = torch.cat([proportions, entropy.unsqueeze(0)])

        return self.norm(out)
