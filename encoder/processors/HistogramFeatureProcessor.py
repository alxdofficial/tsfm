import torch
import torch.nn as nn
from encoder.processors.debug import _ensure_dir, _save_csv, _to_np, visualize_histogram_features
import os

class HistogramFeatureProcessor:
    def __init__(self, num_bins=10, eps: float = 1e-8):
        self.num_bins = num_bins
        self.feature_dim = num_bins + 1  # +1 for entropy
        self.eps = eps
        # NOTE: removed LayerNorm. We keep semantic normalizations below.

    def process(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch: (B, T, D)
        Returns:
            (B, D, num_bins + 1) where the first num_bins are proportions (sum=1),
            and the last dim is entropy normalized to [0,1] by dividing by log2(num_bins).
        """
        B, T, D = patch.shape
        device = patch.device

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

        proportions = hist / (T + self.eps)  # (B*D, bins), sum≈1
        # Re-normalize for numerical safety (exactly sum to 1)
        proportions = proportions / (proportions.sum(dim=1, keepdim=True) + self.eps)

        # Compute entropy and normalize to [0,1] by dividing by log2(num_bins)
        probs = proportions.clamp_min(self.eps)
        entropy = -torch.sum(proportions * torch.log2(probs), dim=1, keepdim=True)  # (B*D, 1)
        entropy = entropy / (torch.log2(torch.tensor(float(self.num_bins), device=device)) + self.eps)

        # Concatenate and reshape
        features = torch.cat([proportions, entropy], dim=1).view(B, D, -1)  # (B, D, num_bins + 1)

        # visualize_histogram_features(
        #     patch,
        #     features,
        #     out_dir=os.path.join("debug_out", "hist"),
        #     num_bins=self.num_bins
        # )
        return features
