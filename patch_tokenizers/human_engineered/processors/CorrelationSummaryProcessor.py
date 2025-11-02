import torch
from patch_tokenizers.human_engineered.processors.debug import _ensure_dir, _save_csv, _to_np, visualize_correlation_summary
import os

class CorrelationSummaryProcessor:
    def __init__(self, debug: bool = False, debug_dir: str = "debug_out/corr"):
        self.feature_dim = 3  # idx_max, idx_min, mean_abs_corr
        self.debug_dir = debug_dir


    def process(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch: (B, T, D)
        Returns:
            (B, D, 3) with every feature in [0, 1]
        """
        B, T, D = patch.shape
        device = patch.device
        out = torch.zeros((B, D, self.feature_dim), dtype=torch.float32, device=device)

        # Degenerate single-channel case
        if D == 1:
            # idx terms collapse to 0, mean_corr=1 by definition (perfect self-corr)
            out[:] = torch.tensor([0.0, 0.0, 1.0], device=device)
            return out

        # --- Correlation ---
        x = patch
        x_centered = x - x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(1e-6)       # (B,1,D)
        x_norm = x_centered / std                               # (B,T,D)
        corr = (x_norm.transpose(1, 2) @ x_norm) / (T - 1)      # (B,D,D)
        corr = torch.nan_to_num(corr, nan=0.0)
        corr = corr.clamp(-1.0, 1.0)                            # <- numeric safety

        # --- Off-diagonal mask ---
        eye = torch.eye(D, dtype=torch.bool, device=device).unsqueeze(0)  # (1,D,D)
        offdiag = ~eye

        # --- Argmax/argmin of *signed* corr over off-diagonal ---
        # use ±inf masking so arg* ignores diagonal
        corr_for_max = corr.masked_fill(~offdiag, float('-inf'))
        corr_for_min = corr.masked_fill(~offdiag, float('inf'))
        argmax_idx = torch.argmax(corr_for_max, dim=-1).float()  # (B,D)
        argmin_idx = torch.argmin(corr_for_min, dim=-1).float()  # (B,D)

        # Normalize indices to [0,1]
        denom = float(max(D - 1, 1))
        idx_max_norm = (argmax_idx / denom).clamp(0.0, 1.0)
        idx_min_norm = (argmin_idx / denom).clamp(0.0, 1.0)

        # --- Mean absolute correlation over off-diagonal ---
        mean_abs_corr = (
            corr.abs().masked_fill(~offdiag, 0.0).sum(dim=-1) / (D - 1)
        ).clamp(0.0, 1.0)  # (B,D)

        # --- Stack ---
        out = torch.stack([idx_max_norm, idx_min_norm, mean_abs_corr], dim=-1)


        # _ensure_dir(self.debug_dir)
        # visualize_correlation_summary(
        #     patch, out, out_dir=self.debug_dir, title_prefix="corr"
        # )

        return out
