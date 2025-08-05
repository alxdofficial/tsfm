import torch
import torch.nn as nn

class HistogramFeatureProcessor:
    def __init__(self, num_bins=10):
        self.num_bins = num_bins
        self.feature_dim = num_bins + 1  # +1 for entropy
        self.norm = nn.LayerNorm(self.feature_dim)

    def process(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch: (B, T, D)
        Returns:
            (B, D, num_bins + 1)
        """
        B, T, D = patch.shape
        device = patch.device
        self.norm = self.norm.to(device)

        # (B, T, D) → (B, D, T)
        x = patch.transpose(1, 2)  # (B, D, T)

        # Compute min and max per (B, D)
        x_min = x.min(dim=-1, keepdim=True).values  # (B, D, 1)
        x_max = x.max(dim=-1, keepdim=True).values  # (B, D, 1)

        # Avoid division by zero
        x_range = x_max - x_min
        x_range[x_range < 1e-6] = 1.0

        # Normalize to [0, 1]
        x_norm = (x - x_min) / x_range  # (B, D, T)

        # Discretize into bins [0, num_bins-1]
        bin_indices = torch.clamp((x_norm * self.num_bins).long(), max=self.num_bins - 1)  # (B, D, T)

        # Compute histogram: flatten first
        flat_bin = bin_indices.reshape(B * D, T)
        hist = torch.zeros((B * D, self.num_bins), device=device)  # (B*D, bins)
        hist.scatter_add_(1, flat_bin, torch.ones_like(flat_bin, dtype=torch.float32))  # count bins

        proportions = hist / T  # (B*D, bins)

        # Compute entropy
        probs = proportions.clone()
        probs[probs == 0] = 1  # log(1) = 0 avoids NaNs
        entropy = -torch.sum(proportions * torch.log2(probs), dim=1, keepdim=True)  # (B*D, 1)

        # Concatenate and reshape
        features = torch.cat([proportions, entropy], dim=1).view(B, D, -1)  # (B, D, num_bins + 1)
        return self.norm(features)
