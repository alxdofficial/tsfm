import torch
from encoder.processors.debug import _ensure_dir, _save_csv, _to_np, visualize_correlation_summary
import os

class CorrelationSummaryProcessor:
    def __init__(self):
        self.feature_dim = 3  # [argmax_idx/D, argmin_idx/D, mean_corr]

    def process(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch: Tensor of shape (B, T, D)
        Returns:
            Tensor of shape (B, D, 3)
        """
        B, T, D = patch.shape
        device = patch.device
        out = torch.zeros((B, D, self.feature_dim), dtype=torch.float32, device=device)

        if D == 1:
            out[:] = torch.tensor([0.0, 0.0, 1.0], device=device)
            return out

        # --- Step 1: Compute correlation matrices per batch ---
        x = patch  # (B, T, D)
        x_centered = x - x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-6  # (B, 1, D)
        x_norm = x_centered / std  # (B, T, D)
        corr = torch.matmul(x_norm.transpose(1, 2), x_norm) / (T - 1)  # (B, D, D)
        corr = torch.nan_to_num(corr, nan=0.0)  # Clean up nans

        # --- Step 2: Mask diagonal to compute argmax/argmin ---
        eye = torch.eye(D, device=device).unsqueeze(0)  # (1, D, D)
        mask = ~eye.bool()  # (1, D, D), True where off-diagonal
        corr_masked = corr.masked_fill(~mask, float('-inf'))  # For argmax
        argmax_idx = torch.argmax(corr_masked, dim=-1).float()  # (B, D)

        corr_masked = corr.masked_fill(~mask, float('inf'))  # For argmin
        argmin_idx = torch.argmin(corr_masked, dim=-1).float()  # (B, D)

        # --- Step 3: Mean absolute correlation excluding diagonal ---
        abs_corr = torch.abs(corr)
        sum_corr = abs_corr.masked_fill(~mask, 0.0).sum(dim=-1)  # (B, D)
        mean_corr = sum_corr / (D - 1)

        # --- Step 4: Stack features ---
        out = torch.stack([
            argmax_idx / D,
            argmin_idx / D,
            mean_corr
        ], dim=-1)  # (B, D, 3)

        # visualize_correlation_summary(
        #     patch,
        #     out,
        #     out_dir=os.path.join("debug_out", "corr")
        # )
        return out
